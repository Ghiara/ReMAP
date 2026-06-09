#!/usr/bin/env python3
"""
Run PEARL and RL2 on the specified goal/velocity targets and produce 4 plots:
  1. PEARL  – Goal Tracking      (x-pos over 200 timesteps)
  2. PEARL  – Velocity Tracking  (x-vel over 200 timesteps)
  3. RL2    – Goal Tracking
  4. RL2    – Velocity Tracking

Raw trajectories are saved to:
  final_results/pearl_rl2_tracking/tracking_data.npz   (numpy)
  final_results/pearl_rl2_tracking/tracking_data.json  (human-readable)

Usage (from project root):
  python plot_tracking_results.py [--gpu 0] [--num-trajs 3] [--out-dir <dir>]
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

from third_party.rlkit.envs import ENVS
from third_party.rlkit.envs.wrappers import NormalizedBoxEnv
from third_party.rlkit.samplers.util import rollout
import third_party.rlkit.torch.pytorch_util as ptu

# ------------------------------------------------------------------ #
#  Targets specified by the user
# ------------------------------------------------------------------ #
GOAL_TARGETS     = [-9.02, -5.10, -3.14, 3.14, 5.10, 9.02]
VELOCITY_TARGETS = [-2.35, -1.75, -1.45, 1.45, 1.75, 2.35]

# Observation indices in cheetah-multi-task
OBS_X_VEL = 8   # qvel[0]  – forward velocity
OBS_X_POS = 17  # body_com("torso")[0] – x position

# Experiment directories (edit if timestamps differ)
PEARL_EXP_DIR = "output/pearl_baseline_full/cheetah-multi-task/2026_04_13_23_30_02"
RL2_EXP_DIR   = "output/cheetah-multi-task/2026_04_19_06_26_05"


# ------------------------------------------------------------------ #
#  Utilities
# ------------------------------------------------------------------ #
def deep_update_dict(fr, to):
    for k, v in fr.items():
        if isinstance(v, dict):
            deep_update_dict(v, to[k])
        else:
            to[k] = v
    return to


def set_goal_task(env, goal):
    base_task = 2 if goal >= 0 else 3
    env._task = {
        'base_task': base_task,
        'specification': goal,
        'color': np.array([1, 1, 0]) if goal >= 0 else np.array([0, 1, 1]),
    }
    env.base_task = base_task
    env.task_specification = goal
    env._goal = goal
    env.reset()


def set_velocity_task(env, vel):
    base_task = 0 if vel >= 0 else 1
    env._task = {
        'base_task': base_task,
        'specification': vel,
        'color': np.array([1, 0, 0]) if vel >= 0 else np.array([0, 0, 1]),
    }
    env.base_task = base_task
    env.task_specification = vel
    env._goal = vel
    env.reset()


# ------------------------------------------------------------------ #
#  PEARL loading and evaluation
# ------------------------------------------------------------------ #
def load_pearl_agent(exp_dir):
    from configs.pearl_default import default_config
    from third_party.rlkit.torch.sac.policies import TanhGaussianPolicy
    from third_party.rlkit.torch.networks import FlattenMlp, MlpEncoder, RecurrentEncoder
    from third_party.rlkit.torch.sac.agent import PEARLAgent

    variant_path = os.path.join(exp_dir, 'variant.json')
    with open(variant_path) as f:
        exp_params = json.load(f)
    variant = deep_update_dict(exp_params, default_config.copy())

    env = NormalizedBoxEnv(ENVS[variant['env_name']](**variant['env_params']))
    obs_dim    = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    reward_dim = 1
    latent_dim = variant['latent_size']
    net_size   = variant['net_size']
    recurrent  = variant['algo_params']['recurrent']

    use_next_obs = variant['algo_params']['use_next_obs_in_context']
    use_ib       = variant['algo_params']['use_information_bottleneck']

    enc_in  = (2 * obs_dim + action_dim + reward_dim) if use_next_obs else (obs_dim + action_dim + reward_dim)
    enc_out = latent_dim * 2 if use_ib else latent_dim

    EncoderModel = RecurrentEncoder if recurrent else MlpEncoder
    context_encoder = EncoderModel(
        hidden_sizes=[200, 200, 200],
        input_size=enc_in,
        output_size=enc_out,
    )
    policy = TanhGaussianPolicy(
        hidden_sizes=[net_size, net_size, net_size],
        obs_dim=obs_dim + latent_dim,
        latent_dim=latent_dim,
        action_dim=action_dim,
    )
    agent = PEARLAgent(latent_dim, context_encoder, policy, **variant['algo_params'])

    weights_dir = os.path.join(exp_dir, 'weights')
    context_encoder.load_state_dict(
        torch.load(os.path.join(weights_dir, 'context_encoder.pth'), map_location='cpu'))
    policy.load_state_dict(
        torch.load(os.path.join(weights_dir, 'policy.pth'), map_location='cpu'))

    print(f"[PEARL] Loaded from {weights_dir}")
    return env, agent, variant


def collect_trajectories_pearl(env, agent, variant, set_task_fn, targets,
                                obs_idx, num_trajs=3):
    """Returns dict: target -> list of per-timestep signal arrays (one per traj)."""
    max_path_length = variant['algo_params']['max_path_length']
    results = {}
    for target in targets:
        set_task_fn(env, target)
        agent.clear_z()
        trajs = []
        for _ in range(num_trajs):
            path = rollout(env, agent, max_path_length=max_path_length,
                           accum_context=True)
            signal = path['observations'][:, obs_idx]
            trajs.append(signal)
            agent.infer_posterior(agent.context)
        results[target] = trajs
        print(f"  PEARL target={target:>6.2f}  adapted_mean={float(np.mean(trajs[-1])):>7.3f}")
    return results


# ------------------------------------------------------------------ #
#  RL2 loading and evaluation
# ------------------------------------------------------------------ #
def load_rl2_agent(exp_dir):
    from configs.rl2_default import default_config
    from third_party.rlkit.torch.rl2.rl2_agent import RL2Agent

    variant_path = os.path.join(exp_dir, 'variant.json')
    with open(variant_path) as f:
        exp_params = json.load(f)
    variant = deep_update_dict(exp_params, default_config.copy())

    env = NormalizedBoxEnv(ENVS[variant['env_name']](**variant['env_params']))
    obs_dim    = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    net_size   = variant.get('net_size', 300)

    agent = RL2Agent(obs_dim=obs_dim, action_dim=action_dim,
                     hidden_size=net_size, num_layers=1)

    weights_dir = os.path.join(exp_dir, 'weights')
    state_dict = torch.load(os.path.join(weights_dir, 'policy.pth'), map_location='cpu')
    if isinstance(state_dict, dict) and 'policy' in state_dict and not any(
            k.startswith('lstm') or k.startswith('mean') or k.startswith('log_std')
            for k in state_dict):
        state_dict = state_dict['policy']
    agent.policy.load_state_dict(state_dict)

    print(f"[RL2]  Loaded from {weights_dir}")
    return env, agent, variant


def collect_trajectories_rl2(env, agent, variant, set_task_fn, targets,
                              obs_idx, num_trajs=3):
    max_path_length = variant['algo_params']['max_path_length']
    results = {}
    for target in targets:
        set_task_fn(env, target)
        agent.clear_z()   # reset LSTM hidden state
        trajs = []
        for _ in range(num_trajs):
            path = rollout(env, agent, max_path_length=max_path_length,
                           accum_context=True)
            signal = path['observations'][:, obs_idx]
            trajs.append(signal)
            agent.infer_posterior(agent.context)
        results[target] = trajs
        print(f"  RL2   target={target:>6.2f}  adapted_mean={float(np.mean(trajs[-1])):>7.3f}")
    return results


# ------------------------------------------------------------------ #
#  Plotting
# ------------------------------------------------------------------ #
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']


def plot_tracking(results, targets, ylabel, title, save_path, target_label_prefix='Target'):
    """
    One solid line per target showing the adapted (last) trajectory.
    A dashed horizontal line at the target value for reference.
    x-axis: timestep (0..199), y-axis: observed signal.
    """
    T = max(len(results[t][-1]) for t in targets)
    timesteps = np.arange(T)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, target in enumerate(targets):
        traj = results[target][-1]   # last = adapted trajectory
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
    print(f"  Saved: {save_path}")


# ------------------------------------------------------------------ #
#  Main
# ------------------------------------------------------------------ #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num-trajs', type=int, default=3,
                        help='Trajectories per task (exploration + adapted)')
    parser.add_argument('--out-dir', type=str,
                        default='final_results/pearl_rl2_tracking')
    args = parser.parse_args()

    ptu.set_gpu_mode(args.gpu >= 0, args.gpu)
    os.makedirs(args.out_dir, exist_ok=True)

    # ==================== PEARL ====================
    print("\n" + "="*60)
    print(" Loading PEARL agent")
    print("="*60)
    pearl_env, pearl_agent, pearl_variant = load_pearl_agent(PEARL_EXP_DIR)
    if ptu.gpu_enabled():
        pearl_agent.to(ptu.device)

    print("\n[PEARL] Goal Tracking")
    pearl_goal = collect_trajectories_pearl(
        pearl_env, pearl_agent, pearl_variant,
        set_goal_task, GOAL_TARGETS, OBS_X_POS, num_trajs=args.num_trajs)

    print("\n[PEARL] Velocity Tracking")
    pearl_vel = collect_trajectories_pearl(
        pearl_env, pearl_agent, pearl_variant,
        set_velocity_task, VELOCITY_TARGETS, OBS_X_VEL, num_trajs=args.num_trajs)

    # ==================== RL2 ====================
    print("\n" + "="*60)
    print(" Loading RL2 agent")
    print("="*60)
    rl2_env, rl2_agent, rl2_variant = load_rl2_agent(RL2_EXP_DIR)
    if ptu.gpu_enabled():
        rl2_agent.to(ptu.device)

    print("\n[RL2]  Goal Tracking")
    rl2_goal = collect_trajectories_rl2(
        rl2_env, rl2_agent, rl2_variant,
        set_goal_task, GOAL_TARGETS, OBS_X_POS, num_trajs=args.num_trajs)

    print("\n[RL2]  Velocity Tracking")
    rl2_vel = collect_trajectories_rl2(
        rl2_env, rl2_agent, rl2_variant,
        set_velocity_task, VELOCITY_TARGETS, OBS_X_VEL, num_trajs=args.num_trajs)

    # ==================== 4 Plots ====================
    print("\n" + "="*60)
    print(" Generating 4 plots")
    print("="*60)

    plot_tracking(
        pearl_goal, GOAL_TARGETS,
        ylabel='X Position',
        title='PEARL – Goal Position Tracking (200 steps)',
        save_path=os.path.join(args.out_dir, 'pearl_goal_tracking.png'),
        target_label_prefix='Goal',
    )
    plot_tracking(
        pearl_vel, VELOCITY_TARGETS,
        ylabel='X Velocity',
        title='PEARL – Velocity Tracking (200 steps)',
        save_path=os.path.join(args.out_dir, 'pearl_velocity_tracking.png'),
        target_label_prefix='Target Vel',
    )
    plot_tracking(
        rl2_goal, GOAL_TARGETS,
        ylabel='X Position',
        title='RL2 – Goal Position Tracking (200 steps)',
        save_path=os.path.join(args.out_dir, 'rl2_goal_tracking.png'),
        target_label_prefix='Goal',
    )
    plot_tracking(
        rl2_vel, VELOCITY_TARGETS,
        ylabel='X Velocity',
        title='RL2 – Velocity Tracking (200 steps)',
        save_path=os.path.join(args.out_dir, 'rl2_velocity_tracking.png'),
        target_label_prefix='Target Vel',
    )

    # ==================== Save raw data ====================
    print("\n" + "="*60)
    print(" Saving raw data")
    print("="*60)

    # NPZ: each key is (num_trajs, T)
    npz_data = {}
    for t in GOAL_TARGETS:
        k = f"{t:.2f}".replace('-', 'm').replace('.', 'p')
        npz_data[f"pearl_goal_{k}"] = np.stack(pearl_goal[t])
        npz_data[f"rl2_goal_{k}"]   = np.stack(rl2_goal[t])
    for t in VELOCITY_TARGETS:
        k = f"{t:.2f}".replace('-', 'm').replace('.', 'p')
        npz_data[f"pearl_vel_{k}"] = np.stack(pearl_vel[t])
        npz_data[f"rl2_vel_{k}"]   = np.stack(rl2_vel[t])
    npz_data['goal_targets']     = np.array(GOAL_TARGETS)
    npz_data['velocity_targets'] = np.array(VELOCITY_TARGETS)

    npz_path = os.path.join(args.out_dir, 'tracking_data.npz')
    np.savez(npz_path, **npz_data)
    print(f"  Saved: {npz_path}")

    # JSON
    def to_list(arr):
        return arr.tolist() if isinstance(arr, np.ndarray) else list(arr)

    json_data = {
        'goal_targets': GOAL_TARGETS,
        'velocity_targets': VELOCITY_TARGETS,
        'pearl': {
            'goal':     {str(t): [to_list(tr) for tr in trajs] for t, trajs in pearl_goal.items()},
            'velocity': {str(t): [to_list(tr) for tr in trajs] for t, trajs in pearl_vel.items()},
        },
        'rl2': {
            'goal':     {str(t): [to_list(tr) for tr in trajs] for t, trajs in rl2_goal.items()},
            'velocity': {str(t): [to_list(tr) for tr in trajs] for t, trajs in rl2_vel.items()},
        },
    }
    json_path = os.path.join(args.out_dir, 'tracking_data.json')
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"  Saved: {json_path}")

    print(f"\nDone. All outputs written to: {args.out_dir}")


if __name__ == '__main__':
    main()
