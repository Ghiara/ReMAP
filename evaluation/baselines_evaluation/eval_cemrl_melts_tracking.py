#!/usr/bin/env python3
"""
Evaluate CEMRL and MELTS on cheetah-multi-task goal/velocity tracking.

Produces the same tracking_data.json format as PEARL/RL2 results in
  final_results/pearl_rl2_tracking/tracking_data.json
plus 4 PNG plots (cemrl_goal_tracking.png, cemrl_velocity_tracking.png,
melts_goal_tracking.png, melts_velocity_tracking.png).

Usage:
    python eval_cemrl_melts_tracking.py \
        --cemrl-dir output/cemrl_baseline/cheetah-multi-task/2026_05_11_16_51_40_cemrl_cheetah_true_gmm \
        --melts-dir output/melts_baseline/cheetah-multi-task/2026_05_12_06_36_21_melts_cheetah_dpmm \
        --gpu 0 --num-trajs 3
"""

import os
import sys
import copy
import glob
import json
import argparse
import tempfile

import numpy as np
from pathlib import Path

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

import third_party.rlkit.torch.pytorch_util as ptu
from third_party.rlkit.envs import ENVS
from third_party.rlkit.envs.wrappers import NormalizedBoxEnv
from third_party.rlkit.torch.sac.policies import TanhGaussianPolicy
from third_party.rlkit.torch.networks import FlattenMlp
from configs.default import default_config
from train.run_task_inference_high_level_policy_training import deep_update_dict, npify_dict
from third_party.tigr.task_inference.prediction_networks import DecoderMDP, ExtendedDecoderMDP
from third_party.tigr.agent_module import Agent

# ------------------------------------------------------------------ #
#  Tracking targets — must match pearl_rl2_tracking/tracking_data.json
# ------------------------------------------------------------------ #
GOAL_TARGETS     = [-9.02, -5.10, -3.14,  3.14,  5.10,  9.02]
VELOCITY_TARGETS = [-2.35, -1.75, -1.45,  1.45,  1.75,  2.35]

OBS_X_VEL = 8   # qvel[0]  – forward x-velocity
OBS_X_POS = 17  # body_com("torso")[0] – x position

COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']


# ------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------ #
def find_weights_dir(exp_dir):
    d = os.path.join(exp_dir, 'weights')
    if os.path.isdir(d):
        return d
    candidates = glob.glob(os.path.join(exp_dir, '**', 'weights'), recursive=True)
    return sorted(candidates)[-1] if candidates else None


def list_available_itrs(weights_dir):
    itrs = []
    for f in glob.glob(os.path.join(weights_dir, 'encoder_itr_*.pth')):
        try:
            itrs.append(int(os.path.basename(f).replace('encoder_itr_', '').replace('.pth', '')))
        except ValueError:
            pass
    return sorted(itrs)


def set_goal_task(env, goal):
    """Inject a goal task directly into the cheetah env and reset."""
    raw = env._wrapped_env if hasattr(env, '_wrapped_env') else env
    base_task = 2 if goal >= 0 else 3
    raw._task = {
        'base_task': base_task,
        'specification': goal,
        'color': np.array([1, 1, 0]) if goal >= 0 else np.array([0, 1, 1]),
    }
    raw.base_task = base_task
    raw.task_specification = goal
    raw._goal = goal
    try:
        raw.recolor()
    except Exception:
        pass
    obs = env.reset()
    return obs[0] if isinstance(obs, (list, tuple)) else obs


def set_velocity_task(env, vel):
    """Inject a velocity task directly into the cheetah env and reset."""
    raw = env._wrapped_env if hasattr(env, '_wrapped_env') else env
    base_task = 0 if vel >= 0 else 1
    raw._task = {
        'base_task': base_task,
        'specification': vel,
        'color': np.array([1, 0, 0]) if vel >= 0 else np.array([0, 0, 1]),
    }
    raw.base_task = base_task
    raw.task_specification = vel
    raw._goal = vel
    try:
        raw.recolor()
    except Exception:
        pass
    obs = env.reset()
    return obs[0] if isinstance(obs, (list, tuple)) else obs


# ------------------------------------------------------------------ #
#  Model loading
# ------------------------------------------------------------------ #
def load_tigr_agent(exp_dir, config_path, inference_option, gpu=0):
    """
    Build encoder + policy networks exactly as in run_toy_training.experiment(),
    load the latest checkpoint, return (env, agent, info_dict).
    """
    variant = copy.deepcopy(default_config)
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            exp_params = json.load(f)
        variant = deep_update_dict(exp_params, variant)
    else:
        print(f'  Warning: config {config_path} not found, using defaults.')

    variant['inference_option'] = inference_option

    # --- Build env ---
    env_params = copy.deepcopy(variant['env_params'])
    use_normalized = env_params.pop('use_normalized_env', True)
    env = ENVS[variant['env_name']](**env_params)
    if use_normalized:
        env = NormalizedBoxEnv(env)

    obs_dim    = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    reward_dim = 1

    latent_dim    = variant['algo_params']['latent_size']
    time_steps    = variant['algo_params']['time_steps']
    num_classes   = variant['reconstruction_params']['num_classes']
    net_complex   = variant['reconstruction_params']['net_complex_enc_dec']
    enc_type      = variant['algo_params']['encoder_type']
    ts_combo      = variant['algo_params']['timestep_combination']

    # Mirror the override in run_toy_training.experiment()
    encoding_mode = variant['algo_params']['encoding_mode']
    if enc_type in ['gru'] and encoding_mode != 'trajectory':
        print(f'  Info: forcing encoding_mode to "trajectory" for encoder_type="{enc_type}"')
        encoding_mode = 'trajectory'
    elif enc_type in ['transformer', 'conv'] and encoding_mode != 'transitionSharedY':
        print(f'  Info: forcing encoding_mode to "transitionSharedY" for encoder_type="{enc_type}"')
        encoding_mode = 'transitionSharedY'
    M             = variant['algo_params']['sac_layer_size']
    context_type  = variant['algo_params']['sac_context_type']
    use_dn        = variant['algo_params']['use_data_normalization']
    permute       = variant['algo_params']['permute_samples']

    # Encoder input dimension
    if encoding_mode == 'transitionSharedY':
        encoder_input_dim = obs_dim + reward_dim + obs_dim
        shared_dim = int(encoder_input_dim * net_complex)
    elif encoding_mode == 'trajectory':
        encoder_input_dim = time_steps * (obs_dim + reward_dim + obs_dim)
        shared_dim = int(encoder_input_dim / time_steps * net_complex)
    else:
        raise NotImplementedError(f'Unknown encoding_mode: {encoding_mode}')

    # --- Select inference module ---
    if inference_option == 'true_gmm':
        from third_party.tigr.task_inference.true_gmm_inference import DecoupledEncoder
        bnp_model = None

    elif inference_option == 'dpmm':
        from third_party.tigr.task_inference.dpmm_inference import DecoupledEncoder
        from third_party.tigr.task_inference.dpmm_bnp import BNPModel
        dp = variant.get('dpmm_params', {})
        # BNP cluster assignments are not used by the policy at eval time
        # (the assignments are discarded in Agent.get_action). We create a
        # minimal BNPModel that never auto-fits during evaluation.
        bnp_model = BNPModel(
            save_dir=tempfile.mkdtemp(),
            start_epoch=int(1e12),
            gamma0=dp.get('gamma0', 5.0),
            num_lap=dp.get('num_lap', 200),
            fit_interval=dp.get('fit_interval', 'epoch'),
            kl_method=dp.get('kl_method', 'hard'),
            birth_kwargs=dp.get('birth_kwargs', {}),
            merge_kwargs=dp.get('merge_kwargs', {}),
        )
    else:
        raise ValueError(f'Unsupported inference_option: {inference_option}')

    encoder = DecoupledEncoder(
        shared_dim,
        encoder_input_dim,
        latent_dim,
        num_classes,
        time_steps,
        encoding_mode=encoding_mode,
        timestep_combination=ts_combo,
        encoder_type=enc_type,
        bnp_model=bnp_model,
    )

    policy_latent_dim = latent_dim if context_type == 'sample' else latent_dim * 2
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim + policy_latent_dim,
        action_dim=action_dim,
        latent_dim=policy_latent_dim,
        hidden_sizes=[M, M, M],
    )

    # --- Load weights ---
    weights_dir = find_weights_dir(exp_dir)
    if weights_dir is None:
        raise FileNotFoundError(f'No weights/ directory under {exp_dir}')
    available_itrs = list_available_itrs(weights_dir)
    if not available_itrs:
        raise FileNotFoundError(f'No encoder_itr_*.pth in {weights_dir}')
    itr = max(available_itrs)
    print(f'  Found checkpoints: {available_itrs}')
    print(f'  Loading itr={itr} from {weights_dir}')

    for name, net in [('encoder', encoder), ('policy', policy)]:
        pth = os.path.join(weights_dir, f'{name}_itr_{itr}.pth')
        net.load_state_dict(torch.load(pth, map_location='cpu'))
        print(f'    Loaded {pth}')

    # --- Load data-normalisation stats (if used) ---
    stats_dict = None
    if use_dn:
        stats_path = os.path.join(weights_dir, 'stats_dict.json')
        if os.path.exists(stats_path):
            with open(stats_path) as f:
                stats_dict = npify_dict(json.load(f))
            print(f'    Loaded stats_dict from {stats_path}')
        else:
            print('    Warning: use_data_normalization=True but stats_dict.json not found.')

    agent = Agent(
        encoder, policy,
        use_sample=(context_type == 'sample'),
        simple_env=variant.get('simple_env', False),
    )

    info = {
        'obs_dim':               obs_dim,
        'time_steps':            time_steps,
        'encoding_mode':         encoding_mode,
        'permute_samples':       permute,
        'use_data_normalization': use_dn,
        'stats_dict':            stats_dict,
        'max_path_length':       variant['algo_params']['max_path_length'],
    }
    return env, agent, info


# ------------------------------------------------------------------ #
#  Custom rollout for a single target
# ------------------------------------------------------------------ #
def custom_rollout(env, agent, info, set_task_fn, target, obs_idx, num_trajs=3):
    """
    Run num_trajs episodes for the given target, accumulating context across
    all trajectories (same adaptation budget as PEARL / RL2).

    Returns a list of per-timestep signal arrays, one per trajectory.
    The last element is the 'adapted' trajectory used for comparison.
    """
    max_path_length = info['max_path_length']
    time_steps      = info['time_steps']
    obs_dim         = info['obs_dim']
    encoding_mode   = info['encoding_mode']
    use_dn          = info['use_data_normalization']
    stats           = info['stats_dict']
    enc_input_dim   = obs_dim + 1 + obs_dim   # [obs, reward, next_obs] per step

    # Context tensor: (1, time_steps, enc_input_dim) — zeros initially
    context = torch.zeros(1, time_steps, enc_input_dim)

    # Set task once; env.reset() is called at the start of each trajectory
    set_task_fn(env, target)

    trajs = []
    for _traj in range(num_trajs):
        obs = env.reset()
        if isinstance(obs, (list, tuple)):
            obs = obs[0]

        signals = []
        for _step in range(max_path_length):
            # Build encoder input from accumulated context
            # For 'trajectory' mode the encoder gets a flat vector (1, T*input_dim);
            # for 'transitionSharedY' it receives (1, T, input_dim).
            enc_input = context.detach().clone()
            if encoding_mode == 'trajectory':
                enc_input = enc_input.view(1, -1)   # (1, time_steps * enc_input_dim)
            # else: keep (1, time_steps, enc_input_dim) for transitionSharedY
            enc_input = enc_input.to(ptu.device)

            with torch.no_grad():
                (action, _), _ = agent.get_action(enc_input, obs, deterministic=True)

            next_obs, reward, done, *_ = env.step(action)
            if isinstance(next_obs, (list, tuple)):
                next_obs = next_obs[0]

            signals.append(float(next_obs[obs_idx]))

            # Build normalised context step [o, r, next_o]
            if use_dn and stats is not None:
                o_ctx  = (obs       - stats['observations']['mean'])      / (stats['observations']['std']      + 1e-9)
                r_ctx  = (np.array([reward]) - stats['rewards']['mean'])  / (stats['rewards']['std']           + 1e-9)
                no_ctx = (next_obs  - stats['next_observations']['mean']) / (stats['next_observations']['std'] + 1e-9)
            else:
                o_ctx  = obs
                r_ctx  = np.array([reward])
                no_ctx = next_obs

            step_ctx = torch.tensor(
                np.concatenate([o_ctx, r_ctx.flatten(), no_ctx]).astype(np.float32),
                dtype=torch.float32,
            ).unsqueeze(0).unsqueeze(0)                          # (1, 1, enc_input_dim)
            context = torch.cat([context, step_ctx], dim=1)[:, -time_steps:]

            obs = next_obs
            if done:
                break

        trajs.append(signals)

    return trajs


# ------------------------------------------------------------------ #
#  Plotting (same style as plot_tracking_results.py)
# ------------------------------------------------------------------ #
def plot_tracking(results, targets, ylabel, title, save_path, target_label_prefix='Target'):
    """One line per target showing the adapted (last) trajectory."""
    T = max(len(results[t][-1]) for t in targets)
    timesteps = np.arange(T)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, target in enumerate(targets):
        traj = results[target][-1]
        color = COLORS[i % len(COLORS)]
        ax.plot(timesteps[:len(traj)], traj, color=color, linewidth=1.8,
                label=f'{target_label_prefix} = {target:.2f}')
        ax.axhline(y=target, color=color, linestyle='--', linewidth=1.0, alpha=0.55)

    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='best', ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {save_path}')


# ------------------------------------------------------------------ #
#  Main
# ------------------------------------------------------------------ #
def main():
    parser = argparse.ArgumentParser(
        description='Evaluate CEMRL and MELTS on cheetah goal/velocity tracking'
    )
    parser.add_argument(
        '--cemrl-dir', type=str,
        default='output/cemrl_baseline/cheetah-multi-task/'
                '2026_05_11_16_51_40_cemrl_cheetah_true_gmm',
    )
    parser.add_argument(
        '--melts-dir', type=str,
        default='output/melts_baseline/cheetah-multi-task/'
                '2026_05_12_06_36_21_melts_cheetah_dpmm',
    )
    parser.add_argument('--cemrl-config', type=str,
                        default='configs/cemrl_cheetah_tigr_config.json')
    parser.add_argument('--melts-config', type=str,
                        default='configs/melts_cheetah_config.json')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num-trajs', type=int, default=3,
                        help='Episodes per target (context accumulates across them)')
    parser.add_argument('--out-dir', type=str,
                        default='final_results/cemrl_melts_tracking')
    args = parser.parse_args()

    ptu.set_gpu_mode(args.gpu >= 0, args.gpu)
    os.makedirs(args.out_dir, exist_ok=True)
    os.environ.setdefault('DEBUG', '0')
    os.environ.setdefault('PLOT',  '0')

    def to_list(seq):
        return [float(v) for v in seq]

    json_data = {
        'goal_targets':     GOAL_TARGETS,
        'velocity_targets': VELOCITY_TARGETS,
    }

    eval_plan = [
        ('cemrl', args.cemrl_dir, args.cemrl_config, 'true_gmm'),
        ('melts', args.melts_dir, args.melts_config, 'dpmm'),
    ]

    for method, exp_dir, config, inference_opt in eval_plan:
        print(f'\n{"="*65}')
        print(f'  Loading {method.upper()}')
        print(f'  exp_dir : {exp_dir}')
        print(f'  config  : {config}')
        print(f'{"="*65}')

        env, agent, info = load_tigr_agent(exp_dir, config, inference_opt, args.gpu)
        if ptu.gpu_enabled():
            agent.to(ptu.device)

        # ---------- Goal Tracking ----------
        print(f'\n[{method.upper()}] Goal Tracking  targets = {GOAL_TARGETS}')
        goal_results = {}
        for t in GOAL_TARGETS:
            trajs = custom_rollout(env, agent, info,
                                   set_goal_task, t, OBS_X_POS, args.num_trajs)
            goal_results[t] = trajs
            print(f'  goal={t:>7.2f}  adapted_mean={np.mean(trajs[-1]):>8.3f}')

        # ---------- Velocity Tracking ----------
        print(f'\n[{method.upper()}] Velocity Tracking  targets = {VELOCITY_TARGETS}')
        vel_results = {}
        for t in VELOCITY_TARGETS:
            trajs = custom_rollout(env, agent, info,
                                   set_velocity_task, t, OBS_X_VEL, args.num_trajs)
            vel_results[t] = trajs
            print(f'  vel={t:>7.2f}  adapted_mean={np.mean(trajs[-1]):>8.3f}')

        # ---------- Store in JSON structure ----------
        json_data[method] = {
            'goal':     {str(t): [to_list(tr) for tr in trajs]
                         for t, trajs in goal_results.items()},
            'velocity': {str(t): [to_list(tr) for tr in trajs]
                         for t, trajs in vel_results.items()},
        }

        # ---------- Plots ----------
        plot_tracking(
            goal_results, GOAL_TARGETS,
            ylabel='X Position',
            title=f'{method.upper()} – Goal Position Tracking (cheetah-multi-task)',
            save_path=os.path.join(args.out_dir, f'{method}_goal_tracking.png'),
            target_label_prefix='Goal',
        )
        plot_tracking(
            vel_results, VELOCITY_TARGETS,
            ylabel='X Velocity',
            title=f'{method.upper()} – Velocity Tracking (cheetah-multi-task)',
            save_path=os.path.join(args.out_dir, f'{method}_velocity_tracking.png'),
            target_label_prefix='Target Vel',
        )

        env.close()

    # ---------- Save tracking_data.json ----------
    json_path = os.path.join(args.out_dir, 'tracking_data.json')
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f'\n  tracking_data.json → {json_path}')
    print(f'\nDone. All outputs written to: {args.out_dir}')


if __name__ == '__main__':
    main()
