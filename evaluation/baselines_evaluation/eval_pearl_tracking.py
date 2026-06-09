#!/usr/bin/env python3
"""
Evaluate PEARL agent's actual velocity tracking and goal reaching ability.

Measures per-timestep achieved velocity / position vs target,
then generates comparison plots.

Usage:
    python eval_pearl_tracking.py \
        --exp-dir output/pearl_baseline_full/cheetah-multi-task/2026_04_13_23_30_02 \
        --config configs/pearl_cheetah_multi_config.json \
        --gpu 0 --num-trajs 3
"""
import os, sys, json, argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

from third_party.rlkit.envs import ENVS
from third_party.rlkit.envs.wrappers import NormalizedBoxEnv
from third_party.rlkit.torch.sac.policies import TanhGaussianPolicy
from third_party.rlkit.torch.networks import FlattenMlp, MlpEncoder, RecurrentEncoder
from third_party.rlkit.torch.sac.agent import PEARLAgent
from third_party.rlkit.samplers.util import rollout
from configs.pearl_default import default_config
import third_party.rlkit.torch.pytorch_util as ptu

TASK_NAMES = {
    0: 'velocity_forward', 1: 'velocity_backward',
    2: 'goal_forward', 3: 'goal_backward',
    4: 'flip_forward', 5: 'stand_front', 6: 'stand_back',
    7: 'jump', 8: 'direction_forward', 9: 'direction_backward', 10: 'velocity',
}

# Observation indices
OBS_X_VEL = 8    # qvel[0]
OBS_X_POS = 17   # body_com("torso")[0]

COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']


def deep_update_dict(fr, to):
    for k, v in fr.items():
        if type(v) is dict:
            deep_update_dict(v, to[k])
        else:
            to[k] = v
    return to


def load_agent(variant, weights_dir):
    env = NormalizedBoxEnv(ENVS[variant['env_name']](**variant['env_params']))
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    reward_dim = 1

    latent_dim = variant['latent_size']
    context_encoder_input_dim = (
        2 * obs_dim + action_dim + reward_dim
        if variant['algo_params']['use_next_obs_in_context']
        else obs_dim + action_dim + reward_dim
    )
    context_encoder_output_dim = (
        latent_dim * 2
        if variant['algo_params']['use_information_bottleneck']
        else latent_dim
    )
    net_size = variant['net_size']
    recurrent = variant['algo_params']['recurrent']
    encoder_model = RecurrentEncoder if recurrent else MlpEncoder

    context_encoder = encoder_model(
        hidden_sizes=[200, 200, 200],
        input_size=context_encoder_input_dim,
        output_size=context_encoder_output_dim,
    )
    policy = TanhGaussianPolicy(
        hidden_sizes=[net_size, net_size, net_size],
        obs_dim=obs_dim + latent_dim,
        latent_dim=latent_dim,
        action_dim=action_dim,
    )
    agent = PEARLAgent(
        latent_dim, context_encoder, policy, **variant['algo_params']
    )
    context_encoder.load_state_dict(
        torch.load(os.path.join(weights_dir, 'context_encoder.pth'), map_location='cpu')
    )
    policy.load_state_dict(
        torch.load(os.path.join(weights_dir, 'policy.pth'), map_location='cpu')
    )
    return env, agent


def set_velocity_task(env, vel):
    """Set env to a velocity task (forward for +, backward for -)."""
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


def evaluate_velocity_tracking(env, agent, variant, velocities, num_trajs=3):
    """
    For each target velocity, run num_trajs episodes.
    Record per-timestep achieved velocity.
    Return: dict[vel] -> {
        'target': vel,
        'achieved_velocities': list of arrays (per traj),
        'mean_achieved': mean velocity across last traj,
        'tracking_error': mean |achieved - target| across last traj,
        'returns': list of returns,
    }
    """
    max_path_length = variant['algo_params']['max_path_length']
    results = {}

    for vel in velocities:
        set_velocity_task(env, vel)
        agent.clear_z()

        all_vels = []
        all_returns = []

        for traj_idx in range(num_trajs):
            path = rollout(env, agent, max_path_length=max_path_length, accum_context=True)
            # Extract per-timestep x-velocity from observations
            obs = path['observations']  # (T, obs_dim)
            achieved_vel = obs[:, OBS_X_VEL]  # (T,)
            all_vels.append(achieved_vel)
            all_returns.append(float(np.sum(path['rewards'])))

            if traj_idx >= 0:
                agent.infer_posterior(agent.context)

        # Use the last trajectory (after adaptation) as primary metric
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
        print(f"  vel_target={vel:>5.1f}  achieved={np.mean(last_vel):>6.2f}±{np.std(last_vel):>5.2f}  "
              f"error={np.mean(np.abs(last_vel - vel)):>5.2f}  return={all_returns[-1]:>8.2f}")

    return results


def evaluate_goal_tracking(env, agent, variant, goals, num_trajs=3):
    """
    For each target goal, run num_trajs episodes.
    Record per-timestep x-position.
    """
    max_path_length = variant['algo_params']['max_path_length']
    results = {}

    for goal in goals:
        set_goal_task(env, goal)
        agent.clear_z()

        all_pos = []
        all_returns = []

        for traj_idx in range(num_trajs):
            path = rollout(env, agent, max_path_length=max_path_length, accum_context=True)
            obs = path['observations']
            achieved_pos = obs[:, OBS_X_POS]
            all_pos.append(achieved_pos)
            all_returns.append(float(np.sum(path['rewards'])))

            if traj_idx >= 0:
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


# ============================= Plotting =============================

def plot_velocity_tracking_curves(vel_results, save_dir):
    """Plot one adapted trajectory per target in a single tracking figure."""
    velocities = sorted(vel_results.keys())
    max_t = max(len(vel_results[v]['achieved_velocities'][-1]) for v in velocities)
    timesteps = np.arange(max_t)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, vel in enumerate(velocities):
        traj = vel_results[vel]['achieved_velocities'][-1]
        color = COLORS[i % len(COLORS)]
        ax.plot(timesteps[:len(traj)], traj, color=color, linewidth=1.8,
                label=f'Target Vel = {vel:.2f}')
        ax.axhline(y=vel, color=color, linestyle='--', linewidth=1.0, alpha=0.55)

    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('X Velocity', fontsize=12)
    ax.set_title('PEARL - Velocity Tracking (cheetah-multi-task)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='best', ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.savefig(os.path.join(save_dir, 'velocity_tracking_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_dir}/velocity_tracking_curves.png")


def plot_velocity_summary(vel_results, save_dir):
    """Bar chart: target vs achieved velocity + tracking error."""
    velocities = sorted(vel_results.keys())
    targets = [v for v in velocities]
    achieved = [vel_results[v]['mean_achieved_vel'] for v in velocities]
    errors = [vel_results[v]['tracking_error'] for v in velocities]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: target vs achieved
    x = np.arange(len(velocities))
    width = 0.35
    bars1 = ax1.bar(x - width/2, targets, width, label='Target Velocity', color='#2196F3', alpha=0.8)
    bars2 = ax1.bar(x + width/2, achieved, width, label='Achieved Velocity (PEARL)', color='#FF9800', alpha=0.8)
    ax1.set_xlabel('Velocity Task')
    ax1.set_ylabel('Velocity')
    ax1.set_title('Target vs Achieved Velocity', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{v:.1f}' for v in velocities])
    ax1.legend()
    ax1.grid(True, axis='y', alpha=0.3)

    # Add ideal line
    lims = [min(min(targets), min(achieved)) - 0.5, max(max(targets), max(achieved)) + 0.5]

    # Right: tracking error
    ax2.bar(x, errors, color='#f44336', alpha=0.8)
    ax2.set_xlabel('Target Velocity')
    ax2.set_ylabel('Mean |Achieved - Target|')
    ax2.set_title('Velocity Tracking Error', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{v:.1f}' for v in velocities])
    ax2.grid(True, axis='y', alpha=0.3)

    fig.suptitle('PEARL Baseline: Velocity Tracking Performance', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'velocity_tracking_summary.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_dir}/velocity_tracking_summary.png")


def plot_goal_tracking_curves(goal_results, save_dir):
    """Plot one adapted trajectory per target in a single tracking figure."""
    goals = sorted(goal_results.keys())
    max_t = max(len(goal_results[g]['achieved_positions'][-1]) for g in goals)
    timesteps = np.arange(max_t)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, goal in enumerate(goals):
        traj = goal_results[goal]['achieved_positions'][-1]
        color = COLORS[i % len(COLORS)]
        ax.plot(timesteps[:len(traj)], traj, color=color, linewidth=1.8,
                label=f'Goal = {goal:.2f}')
        ax.axhline(y=goal, color=color, linestyle='--', linewidth=1.0, alpha=0.55)

    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('X Position', fontsize=12)
    ax.set_title('PEARL - Goal Position Tracking (cheetah-multi-task)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='best', ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.savefig(os.path.join(save_dir, 'goal_tracking_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_dir}/goal_tracking_curves.png")


def plot_goal_summary(goal_results, save_dir):
    """Bar chart: target vs final position + distance to goal."""
    goals = sorted(goal_results.keys())
    targets = [g for g in goals]
    final_pos = [goal_results[g]['final_position'] for g in goals]
    distances = [goal_results[g]['distance_to_goal'] for g in goals]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(goals))
    width = 0.35
    ax1.bar(x - width/2, targets, width, label='Target Position', color='#4CAF50', alpha=0.8)
    ax1.bar(x + width/2, final_pos, width, label='Final Position (PEARL)', color='#FF9800', alpha=0.8)
    ax1.set_xlabel('Goal Task')
    ax1.set_ylabel('X-Position')
    ax1.set_title('Target vs Final Position', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{g:.0f}' for g in goals])
    ax1.legend()
    ax1.grid(True, axis='y', alpha=0.3)

    ax2.bar(x, distances, color='#f44336', alpha=0.8)
    ax2.set_xlabel('Target Goal Position')
    ax2.set_ylabel('|Final Position - Target|')
    ax2.set_title('Goal Reaching Error', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{g:.0f}' for g in goals])
    ax2.grid(True, axis='y', alpha=0.3)

    fig.suptitle('PEARL Baseline: Goal Reaching Performance', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'goal_tracking_summary.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_dir}/goal_tracking_summary.png")


def plot_combined_scatter(vel_results, goal_results, save_dir):
    """Scatter plot: target vs achieved for both velocity and goal tasks."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Velocity scatter
    vels = sorted(vel_results.keys())
    v_target = [v for v in vels]
    v_achieved = [vel_results[v]['mean_achieved_vel'] for v in vels]
    ax1.scatter(v_target, v_achieved, s=100, c='#FF9800', edgecolors='black', zorder=3, label='PEARL')
    lim = [min(min(v_target), min(v_achieved)) - 0.5, max(max(v_target), max(v_achieved)) + 0.5]
    ax1.plot(lim, lim, 'k--', alpha=0.5, label='Ideal (y=x)')
    ax1.set_xlabel('Target Velocity', fontsize=12)
    ax1.set_ylabel('Achieved Velocity', fontsize=12)
    ax1.set_title('Velocity Tracking', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal', adjustable='box')

    # Goal scatter
    goals = sorted(goal_results.keys())
    g_target = [g for g in goals]
    g_achieved = [goal_results[g]['final_position'] for g in goals]
    ax2.scatter(g_target, g_achieved, s=100, c='#FF9800', edgecolors='black', zorder=3, label='PEARL')
    lim2 = [min(min(g_target), min(g_achieved)) - 2, max(max(g_target), max(g_achieved)) + 2]
    ax2.plot(lim2, lim2, 'k--', alpha=0.5, label='Ideal (y=x)')
    ax2.set_xlabel('Target Goal Position', fontsize=12)
    ax2.set_ylabel('Final Position', fontsize=12)
    ax2.set_title('Goal Reaching', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal', adjustable='box')

    fig.suptitle('PEARL Baseline: Target vs Achieved', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'tracking_scatter.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_dir}/tracking_scatter.png")


def main():
    parser = argparse.ArgumentParser(description='PEARL Velocity/Goal Tracking Evaluation')
    parser.add_argument('--exp-dir', type=str, required=True)
    parser.add_argument('--config', type=str, default='configs/pearl_cheetah_multi_config.json')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num-trajs', type=int, default=3)
    parser.add_argument('--velocities', type=str, default='-2.35,-1.75,-1.45,1.45,1.75,2.35')
    parser.add_argument('--goals', type=str, default='-9.02,-5.10,-3.14,3.14,5.10,9.02')
    args = parser.parse_args()

    # Load config
    variant = default_config.copy()
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            exp_params = json.load(f)
        variant = deep_update_dict(exp_params, variant)

    ptu.set_gpu_mode(True, args.gpu)

    weights_dir = os.path.join(args.exp_dir, 'weights')
    print(f"Loading model from: {weights_dir}")
    env, agent = load_agent(variant, weights_dir)
    if ptu.gpu_enabled():
        agent.to('cuda')

    # Parse velocity and goal lists
    velocities = [float(v.strip()) for v in args.velocities.split(',')]
    goals = [float(g.strip()) for g in args.goals.split(',')]

    # Create output directory for plots
    plot_dir = os.path.join(args.exp_dir, 'tracking_plots')
    os.makedirs(plot_dir, exist_ok=True)

    print(f"\nMax path length: {variant['algo_params']['max_path_length']}")
    print(f"Trajectories per task: {args.num_trajs}")

    # ==================== Velocity Tracking ====================
    print(f"\n{'='*70}")
    print(f" Velocity Tracking Evaluation: {velocities}")
    print(f"{'='*70}")
    vel_results = evaluate_velocity_tracking(env, agent, variant, velocities, num_trajs=args.num_trajs)

    # ==================== Goal Tracking ====================
    print(f"\n{'='*70}")
    print(f" Goal Reaching Evaluation: {goals}")
    print(f"{'='*70}")
    goal_results = evaluate_goal_tracking(env, agent, variant, goals, num_trajs=args.num_trajs)

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
    print("Done!")


if __name__ == '__main__':
    main()
