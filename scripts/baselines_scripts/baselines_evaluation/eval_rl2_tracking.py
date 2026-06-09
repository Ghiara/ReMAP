#!/usr/bin/env python3
"""
Evaluate RL2 agent's velocity tracking and goal reaching ability.

Mirrors eval_pearl_tracking.py so that the two baselines can be compared
directly on the same metrics and plots.

Usage:
    python eval_rl2_tracking.py \
        --exp-dir output/cheetah-multi-task/<TIMESTAMP> \
        --config configs/rl2_cheetah_config.json \
        --gpu 0 --num-trajs 3
"""
import os, sys, json, argparse
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
from third_party.rlkit.torch.rl2.rl2_agent import RL2Agent
from third_party.rlkit.samplers.util import rollout
from configs.rl2_default import default_config
import third_party.rlkit.torch.pytorch_util as ptu

TASK_NAMES = {
    0: 'velocity_forward', 1: 'velocity_backward',
    2: 'goal_forward', 3: 'goal_backward',
    4: 'flip_forward', 5: 'stand_front', 6: 'stand_back',
    7: 'jump', 8: 'direction_forward', 9: 'direction_backward', 10: 'velocity',
}

# Observation indices (cheetah-multi-task)
OBS_X_VEL = 8    # qvel[0]
OBS_X_POS = 17   # body_com("torso")[0]


def deep_update_dict(fr, to):
    for k, v in fr.items():
        if isinstance(v, dict):
            deep_update_dict(v, to[k])
        else:
            to[k] = v
    return to


# ------------------------------------------------------------------ #
#  Agent loading
# ------------------------------------------------------------------ #

def load_agent(variant, weights_dir):
    env = NormalizedBoxEnv(ENVS[variant['env_name']](**variant['env_params']))
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    net_size = variant.get('net_size', 300)

    agent = RL2Agent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_size=net_size,
        num_layers=1,
    )

    policy_path = os.path.join(weights_dir, 'policy.pth')
    if not os.path.exists(policy_path):
        raise FileNotFoundError(
            f"Policy weights not found at {policy_path}. "
            "Make sure --exp-dir points to the experiment root (containing weights/)."
        )
    print(f"Loading policy weights from {policy_path}")
    state_dict = torch.load(policy_path, map_location='cpu')
    # The logger saves state_dicts directly; handle both raw state_dict
    # and wrapped dict (e.g. {'policy': state_dict})
    if isinstance(state_dict, dict) and 'policy' in state_dict and not any(
            k.startswith('lstm') or k.startswith('mean') or k.startswith('log_std')
            for k in state_dict):
        state_dict = state_dict['policy']
    agent.policy.load_state_dict(state_dict)
    return env, agent


# ------------------------------------------------------------------ #
#  Task helpers
# ------------------------------------------------------------------ #

def set_velocity_task(env, vel):
    """Set env to a velocity task with given target velocity.
    Positive vel -> base_task 0 (velocity_forward).
    Negative vel -> base_task 1 (velocity_backward).
    """
    base_task = 0 if vel >= 0 else 1
    color = np.array([1, 0, 0]) if vel >= 0 else np.array([0, 0, 1])
    env._task = {'base_task': base_task, 'specification': vel, 'color': color}
    env.base_task = base_task
    env.task_specification = vel
    env._goal = vel
    env.reset()


def set_goal_task(env, goal):
    """Set env to a goal_forward/backward task with given target position."""
    base_task = 2 if goal >= 0 else 3
    env._task = {
        'base_task': base_task, 'specification': goal,
        'color': np.array([1, 1, 0]) if goal >= 0 else np.array([0, 1, 1]),
    }
    env.base_task = base_task
    env.task_specification = goal
    env._goal = goal
    env.reset()


# ------------------------------------------------------------------ #
#  Evaluation functions
# ------------------------------------------------------------------ #

def evaluate_velocity_tracking(env, agent, variant, velocities, num_trajs=3):
    """
    For each target velocity, run num_trajs episodes with the LSTM state
    preserved across episodes (RL2 in-context adaptation).
    Record per-timestep achieved velocity.
    """
    max_path_length = variant['algo_params']['max_path_length']
    results = {}

    for vel in velocities:
        set_velocity_task(env, vel)
        agent.clear_z()   # reset LSTM hidden state for this task

        all_vels, all_returns = [], []
        for traj_idx in range(num_trajs):
            # LSTM state carries over between trajectories (RL2 adaptation)
            path = rollout(env, agent, max_path_length=max_path_length,
                           accum_context=True)
            obs = path['observations']   # (T, obs_dim)
            achieved_vel = obs[:, OBS_X_VEL]
            all_vels.append(achieved_vel)
            all_returns.append(float(np.sum(path['rewards'])))
            # infer_posterior is a no-op for RL2 but kept for interface parity
            agent.infer_posterior(agent.context)

        last_vel = all_vels[-1]
        results[vel] = {
            'target': vel,
            'achieved_velocities': [v.tolist() for v in all_vels],
            'mean_achieved_vel': float(np.mean(last_vel)),
            'std_achieved_vel': float(np.std(last_vel)),
            'tracking_error': float(np.mean(np.abs(last_vel - vel))),
            'returns': all_returns,
            'adapted_return': all_returns[-1],
        }
        print(f"  vel_target={vel:>5.1f}  achieved={np.mean(last_vel):>6.2f}"
              f"\xb1{np.std(last_vel):>5.2f}  error={np.mean(np.abs(last_vel - vel)):>5.2f}"
              f"  return={all_returns[-1]:>8.2f}")

    return results


def evaluate_goal_tracking(env, agent, variant, goals, num_trajs=3):
    """
    For each target goal, run num_trajs episodes with LSTM state preserved.
    Record per-timestep x-position.
    """
    max_path_length = variant['algo_params']['max_path_length']
    results = {}

    for goal in goals:
        set_goal_task(env, goal)
        agent.clear_z()   # reset LSTM hidden state for this task

        all_pos, all_returns = [], []
        for traj_idx in range(num_trajs):
            path = rollout(env, agent, max_path_length=max_path_length,
                           accum_context=True)
            obs = path['observations']
            achieved_pos = obs[:, OBS_X_POS]
            all_pos.append(achieved_pos)
            all_returns.append(float(np.sum(path['rewards'])))
            agent.infer_posterior(agent.context)

        last_pos = all_pos[-1]
        final_pos = float(last_pos[-1])
        results[goal] = {
            'target': goal,
            'achieved_positions': [p.tolist() for p in all_pos],
            'final_position': final_pos,
            'distance_to_goal': float(np.abs(final_pos - goal)),
            'mean_position': float(np.mean(last_pos)),
            'returns': all_returns,
            'adapted_return': all_returns[-1],
        }
        print(f"  goal_target={goal:>6.1f}  final_pos={final_pos:>7.2f}  "
              f"dist_to_goal={np.abs(final_pos - goal):>6.2f}  return={all_returns[-1]:>8.2f}")

    return results


# ------------------------------------------------------------------ #
#  Plotting (same layout as eval_pearl_tracking.py)
# ------------------------------------------------------------------ #

def plot_velocity_tracking_curves(vel_results, save_dir):
    velocities = sorted(vel_results.keys())
    n = len(velocities)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if n == 1:
        axes = [axes]
    else:
        axes = np.atleast_1d(axes).flatten()
    for i, vel in enumerate(velocities):
        ax = axes[i]
        trajs = vel_results[vel]['achieved_velocities']
        for j, tv in enumerate(trajs):
            alpha = 0.3 if j < len(trajs) - 1 else 1.0
            lw = 1 if j < len(trajs) - 1 else 2
            label = f'Traj {j+1}' + (' (adapted)' if j == len(trajs)-1 else ' (explore)')
            ax.plot(tv, alpha=alpha, linewidth=lw, label=label)
        ax.axhline(y=vel, color='r', linestyle='--', linewidth=2, label=f'Target={vel:.1f}')
        direction = 'Backward' if vel < 0 else 'Forward'
        ax.set_title(f'Target Vel = {vel:.1f} ({direction})', fontsize=10, fontweight='bold')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('X-Velocity')
        ax.legend(fontsize=7, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(vel - 3, vel + 3)
    for i in range(n, len(axes)):
        fig.delaxes(axes[i])
    fig.suptitle('RL2: Velocity Tracking per Timestep', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'velocity_tracking_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_dir}/velocity_tracking_curves.png")


def plot_velocity_summary(vel_results, save_dir):
    velocities = sorted(vel_results.keys())
    targets = list(velocities)
    achieved = [vel_results[v]['mean_achieved_vel'] for v in velocities]
    errors = [vel_results[v]['tracking_error'] for v in velocities]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(velocities))
    width = 0.35
    ax1.bar(x - width/2, targets, width, label='Target Velocity', color='#2196F3', alpha=0.8)
    ax1.bar(x + width/2, achieved, width, label='Achieved Velocity (RL2)', color='#FF9800', alpha=0.8)
    ax1.set_xlabel('Velocity Task')
    ax1.set_ylabel('Velocity')
    ax1.set_title('Target vs Achieved Velocity', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{v:.1f}' for v in velocities])
    ax1.legend()
    ax1.grid(True, axis='y', alpha=0.3)
    ax2.bar(x, errors, color='#f44336', alpha=0.8)
    ax2.set_xlabel('Target Velocity')
    ax2.set_ylabel('Mean |Achieved - Target|')
    ax2.set_title('Velocity Tracking Error', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{v:.1f}' for v in velocities])
    ax2.grid(True, axis='y', alpha=0.3)
    fig.suptitle('RL2 Baseline: Velocity Tracking Performance', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'velocity_tracking_summary.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_dir}/velocity_tracking_summary.png")


def plot_goal_tracking_curves(goal_results, save_dir):
    goals = sorted(goal_results.keys())
    n = len(goals)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    for i, goal in enumerate(goals):
        if i >= len(axes):
            break
        ax = axes[i]
        trajs = goal_results[goal]['achieved_positions']
        for j, tp in enumerate(trajs):
            alpha = 0.3 if j < len(trajs) - 1 else 1.0
            lw = 1 if j < len(trajs) - 1 else 2
            label = f'Traj {j+1}' + (' (adapted)' if j == len(trajs)-1 else ' (explore)')
            ax.plot(tp, alpha=alpha, linewidth=lw, label=label)
        ax.axhline(y=goal, color='r', linestyle='--', linewidth=2, label=f'Target={goal:.0f}')
        ax.set_title(f'Target Goal = {goal:.0f}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('X-Position')
        ax.legend(fontsize=7, loc='best')
        ax.grid(True, alpha=0.3)
    for i in range(len(goals), len(axes)):
        fig.delaxes(axes[i])
    fig.suptitle('RL2: Goal Position Tracking per Timestep', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'goal_tracking_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_dir}/goal_tracking_curves.png")


def plot_goal_summary(goal_results, save_dir):
    goals = sorted(goal_results.keys())
    targets = list(goals)
    final_pos = [goal_results[g]['final_position'] for g in goals]
    distances = [goal_results[g]['distance_to_goal'] for g in goals]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(goals)); width = 0.35
    ax1.bar(x - width/2, targets, width, label='Target', color='#4CAF50', alpha=0.8)
    ax1.bar(x + width/2, final_pos, width, label='Final Position (RL2)', color='#FF9800', alpha=0.8)
    ax1.set_xlabel('Goal Task'); ax1.set_ylabel('X-Position')
    ax1.set_title('Target vs Final Position', fontweight='bold')
    ax1.set_xticks(x); ax1.set_xticklabels([f'{g:.0f}' for g in goals])
    ax1.legend(); ax1.grid(True, axis='y', alpha=0.3)
    ax2.bar(x, distances, color='#f44336', alpha=0.8)
    ax2.set_xlabel('Target Goal Position'); ax2.set_ylabel('|Final Position - Target|')
    ax2.set_title('Goal Reaching Error', fontweight='bold')
    ax2.set_xticks(x); ax2.set_xticklabels([f'{g:.0f}' for g in goals])
    ax2.grid(True, axis='y', alpha=0.3)
    fig.suptitle('RL2 Baseline: Goal Reaching Performance', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'goal_tracking_summary.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_dir}/goal_tracking_summary.png")


def plot_combined_scatter(vel_results, goal_results, save_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    vels = sorted(vel_results.keys())
    vt = list(vels); va = [vel_results[v]['mean_achieved_vel'] for v in vels]
    ax1.scatter(vt, va, s=100, c='#FF9800', edgecolors='black', zorder=3, label='RL2')
    lim = [min(min(vt), min(va))-0.5, max(max(vt), max(va))+0.5]
    ax1.plot(lim, lim, 'k--', alpha=0.5, label='Ideal')
    ax1.set_xlabel('Target Velocity'); ax1.set_ylabel('Achieved Velocity')
    ax1.set_title('Velocity Tracking', fontweight='bold')
    ax1.legend(); ax1.grid(True, alpha=0.3); ax1.set_aspect('equal', adjustable='box')

    goals = sorted(goal_results.keys())
    gt = list(goals); ga = [goal_results[g]['final_position'] for g in goals]
    ax2.scatter(gt, ga, s=100, c='#FF9800', edgecolors='black', zorder=3, label='RL2')
    lim2 = [min(min(gt), min(ga))-2, max(max(gt), max(ga))+2]
    ax2.plot(lim2, lim2, 'k--', alpha=0.5, label='Ideal')
    ax2.set_xlabel('Target Goal Position'); ax2.set_ylabel('Final Position')
    ax2.set_title('Goal Reaching', fontweight='bold')
    ax2.legend(); ax2.grid(True, alpha=0.3); ax2.set_aspect('equal', adjustable='box')

    fig.suptitle('RL2 Baseline: Target vs Achieved', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'tracking_scatter.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_dir}/tracking_scatter.png")


# ------------------------------------------------------------------ #
#  Main
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(description='RL2 Velocity/Goal Tracking Evaluation')
    parser.add_argument('--exp-dir', type=str, required=True,
                        help='Experiment root directory (e.g. output/cheetah-multi-task/<TIMESTAMP>)')
    parser.add_argument('--config', type=str, default='configs/rl2_cheetah_config.json')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num-trajs', type=int, default=3,
                        help='Trajectories per task (LSTM state preserved across them)')
    parser.add_argument('--velocities', type=str, nargs='+',
                        default=['-3.0,-2.0,-1.0,1.0,2.0,3.0,4.0,5.0'],
                        help='Target velocities, comma-separated (spaces around commas are fine)')
    parser.add_argument('--goals', type=str, nargs='+',
                        default=['-15.0,-10.0,-5.0,5.0,10.0,15.0,20.0,25.0'],
                        help='Target goal positions, comma-separated (spaces around commas are fine)')
    args = parser.parse_args()

    # Load config
    variant = default_config.copy()
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            exp_params = json.load(f)
        variant = deep_update_dict(exp_params, variant)

    ptu.set_gpu_mode(True, args.gpu)

    # Weights are saved by the rlkit logger under <exp_dir>/weights/
    weights_dir = os.path.join(args.exp_dir, 'weights')
    print(f"Loading model from: {weights_dir}")
    env, agent = load_agent(variant, weights_dir)
    if ptu.gpu_enabled():
        agent.to('cuda')

    velocities = [float(v) for v in ','.join(args.velocities).replace(' ', '').split(',') if v]
    goals = [float(g) for g in ','.join(args.goals).replace(' ', '').split(',') if g]

    plot_dir = os.path.join(args.exp_dir, 'tracking_plots')
    os.makedirs(plot_dir, exist_ok=True)

    print(f"\nEnvironment: {variant['env_name']}")
    print(f"Max path length: {variant['algo_params']['max_path_length']}")
    print(f"Trajectories per task: {args.num_trajs}")

    # ==================== Velocity Tracking ====================
    print(f"\n{'='*70}")
    print(f" Velocity Tracking Evaluation: {velocities}")
    print(f"{'='*70}")
    vel_results = evaluate_velocity_tracking(env, agent, variant, velocities,
                                             num_trajs=args.num_trajs)

    # ==================== Goal Tracking ====================
    print(f"\n{'='*70}")
    print(f" Goal Reaching Evaluation: {goals}")
    print(f"{'='*70}")
    goal_results = evaluate_goal_tracking(env, agent, variant, goals,
                                          num_trajs=args.num_trajs)

    # ==================== Generate Plots ====================
    print(f"\n{'='*70}")
    print(" Generating plots...")
    print(f"{'='*70}")
    plot_velocity_tracking_curves(vel_results, plot_dir)
    plot_velocity_summary(vel_results, plot_dir)
    plot_goal_tracking_curves(goal_results, plot_dir)
    plot_goal_summary(goal_results, plot_dir)
    plot_combined_scatter(vel_results, goal_results, plot_dir)

    # ==================== Save raw data ====================
    save_data = {
        'velocity_tracking': {str(k): v for k, v in vel_results.items()},
        'goal_tracking': {str(k): v for k, v in goal_results.items()},
    }
    data_path = os.path.join(plot_dir, 'tracking_data.json')
    with open(data_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Raw data saved to: {data_path}")

    # ==================== Print Summary Table ====================
    print(f"\n{'='*70}")
    print(" VELOCITY TRACKING SUMMARY")
    print(f"{'='*70}")
    print(f"{'Target':>8s} {'Achieved':>10s} {'Std':>8s} {'Error':>8s} {'Return':>10s}")
    print("-" * 50)
    for vel in sorted(vel_results.keys()):
        r = vel_results[vel]
        print(f"{vel:>8.1f} {r['mean_achieved_vel']:>10.2f} {r['std_achieved_vel']:>8.2f} "
              f"{r['tracking_error']:>8.2f} {r['adapted_return']:>10.2f}")

    print(f"\n{'='*70}")
    print(" GOAL REACHING SUMMARY")
    print(f"{'='*70}")
    print(f"{'Target':>8s} {'Final Pos':>10s} {'Distance':>10s} {'Return':>10s}")
    print("-" * 44)
    for goal in sorted(goal_results.keys()):
        r = goal_results[goal]
        print(f"{goal:>8.1f} {r['final_position']:>10.2f} {r['distance_to_goal']:>10.2f} "
              f"{r['adapted_return']:>10.2f}")

    print(f"\nAll plots saved to: {plot_dir}/")
    env.close()
    print("Done!")


if __name__ == '__main__':
    main()
