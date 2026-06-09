#!/usr/bin/env python3
"""
Evaluate a trained PEARL agent on cheetah-multi-task environment.

Tests the agent's ability to:
  - Track different target velocities (forward/backward)
  - Reach different goal positions (forward/backward)
  - Other tasks: flip, stand, jump, direction

Outputs per-task-type performance breakdown for comparison with hierarchical methods.

Usage:
    # Evaluate the latest run
    python eval_pearl_cheetah_baseline.py \
        --exp-dir output/pearl_baseline_full/cheetah-multi-task/<TIMESTAMP> \
        --config configs/pearl_cheetah_multi_config.json \
        --gpu 0

    # Evaluate with custom tasks (e.g., specific velocities)
    python eval_pearl_cheetah_baseline.py \
        --exp-dir output/pearl_baseline_full/cheetah-multi-task/<TIMESTAMP> \
        --config configs/pearl_cheetah_multi_config.json \
        --num-trajs 3 \
        --save-results
"""
import os
import sys
import json
import argparse
import numpy as np
from collections import defaultdict
from pathlib import Path

import torch

# Add project root to path
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


# Task type names for readable output
TASK_NAMES = {
    0: 'velocity_forward',
    1: 'velocity_backward',
    2: 'goal_forward',
    3: 'goal_backward',
    4: 'flip_forward',
    5: 'stand_front',
    6: 'stand_back',
    7: 'jump',
    8: 'direction_forward',
    9: 'direction_backward',
    10: 'velocity',
}


def deep_update_dict(fr, to):
    for k, v in fr.items():
        if type(v) is dict:
            deep_update_dict(v, to[k])
        else:
            to[k] = v
    return to


def load_agent(variant, weights_dir):
    """Load a trained PEARL agent from saved weights."""
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
        latent_dim,
        context_encoder,
        policy,
        **variant['algo_params']
    )

    # Load weights
    context_encoder.load_state_dict(
        torch.load(os.path.join(weights_dir, 'context_encoder.pth'), map_location='cpu')
    )
    policy.load_state_dict(
        torch.load(os.path.join(weights_dir, 'policy.pth'), map_location='cpu')
    )

    return env, agent


def evaluate_on_tasks(env, agent, task_indices, variant, num_trajs=3, num_exploration_trajs=1):
    """
    Evaluate agent on specific task indices.
    Returns per-task results with task type, specification, and episode returns.
    """
    max_path_length = variant['algo_params']['max_path_length']
    results = []

    for i, idx in enumerate(task_indices):
        env.reset_task(idx)
        task_info = env._task
        base_task = task_info['base_task']
        specification = task_info['specification']
        task_name = TASK_NAMES.get(base_task, f'unknown_{base_task}')

        agent.clear_z()

        episode_returns = []
        episode_rewards_per_step = []

        for traj_idx in range(num_trajs):
            path = rollout(
                env, agent,
                max_path_length=max_path_length,
                accum_context=True,
            )
            ep_return = float(np.sum(path['rewards']))
            episode_returns.append(ep_return)
            episode_rewards_per_step.append(path['rewards'].flatten().tolist())

            # After exploration trajectories, do posterior inference
            if traj_idx + 1 >= num_exploration_trajs:
                agent.infer_posterior(agent.context)

        results.append({
            'task_idx': int(idx),
            'task_name': task_name,
            'base_task': int(base_task),
            'specification': float(specification) if isinstance(specification, (int, float, np.floating)) else specification,
            'episode_returns': episode_returns,
            'mean_return': float(np.mean(episode_returns)),
            'std_return': float(np.std(episode_returns)),
            # The last trajectory (after adaptation) is most relevant
            'adapted_return': episode_returns[-1] if len(episode_returns) > 1 else episode_returns[0],
        })

        print(f"  [{i+1}/{len(task_indices)}] {task_name:25s} spec={specification:>8.3f}  "
              f"mean_ret={np.mean(episode_returns):>8.2f}  adapted_ret={results[-1]['adapted_return']:>8.2f}")

    return results


def print_summary(results, title="Evaluation Summary"):
    """Print per-task-type summary statistics."""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)

    # Group by task type
    task_groups = defaultdict(list)
    for r in results:
        task_groups[r['task_name']].append(r)

    print(f"\n{'Task Type':<25s} {'#Tasks':>6s} {'Mean Return':>12s} {'Std':>10s} {'Adapted Ret':>12s} {'Spec Range':>20s}")
    print("-" * 90)

    all_returns = []
    all_adapted = []

    for task_name in sorted(task_groups.keys()):
        group = task_groups[task_name]
        mean_returns = [r['mean_return'] for r in group]
        adapted_returns = [r['adapted_return'] for r in group]
        specs = [r['specification'] for r in group]

        all_returns.extend(mean_returns)
        all_adapted.extend(adapted_returns)

        spec_range = f"[{min(specs):.2f}, {max(specs):.2f}]"
        print(f"{task_name:<25s} {len(group):>6d} {np.mean(mean_returns):>12.2f} "
              f"{np.std(mean_returns):>10.2f} {np.mean(adapted_returns):>12.2f} {spec_range:>20s}")

    print("-" * 90)
    print(f"{'OVERALL':<25s} {len(results):>6d} {np.mean(all_returns):>12.2f} "
          f"{np.std(all_returns):>10.2f} {np.mean(all_adapted):>12.2f}")
    print("=" * 80)


def evaluate_custom_velocities(env, agent, variant, velocities, num_trajs=3):
    """
    Evaluate the agent on a custom set of target velocities.
    Useful for comparing with your hierarchical method on the same velocity set.
    """
    print("\n--- Custom Velocity Evaluation ---")
    max_path_length = variant['algo_params']['max_path_length']
    results = []

    for vel in velocities:
        # Manually set the task
        env._task = {
            'base_task': 0,  # velocity_forward
            'specification': vel,
            'color': np.array([1, 0, 0]),
        }
        env.base_task = 0
        env.task_specification = vel
        env._goal = vel
        env.reset()

        agent.clear_z()
        episode_returns = []

        for traj_idx in range(num_trajs):
            path = rollout(
                env, agent,
                max_path_length=max_path_length,
                accum_context=True,
            )
            ep_return = float(np.sum(path['rewards']))
            episode_returns.append(ep_return)

            if traj_idx >= 0:  # infer posterior after every trajectory
                agent.infer_posterior(agent.context)

        mean_ret = np.mean(episode_returns)
        adapted_ret = episode_returns[-1]
        results.append({
            'velocity': vel,
            'mean_return': mean_ret,
            'adapted_return': adapted_ret,
            'all_returns': episode_returns,
        })
        print(f"  velocity={vel:>6.2f}  mean_return={mean_ret:>8.2f}  adapted_return={adapted_ret:>8.2f}")

    return results


def evaluate_custom_goals(env, agent, variant, goal_positions, num_trajs=3):
    """
    Evaluate the agent on a custom set of goal positions.
    """
    print("\n--- Custom Goal Position Evaluation ---")
    max_path_length = variant['algo_params']['max_path_length']
    results = []

    for goal in goal_positions:
        base_task = 2 if goal >= 0 else 3  # goal_forward or goal_backward
        env._task = {
            'base_task': base_task,
            'specification': goal,
            'color': np.array([1, 1, 0]) if goal >= 0 else np.array([0, 1, 1]),
        }
        env.base_task = base_task
        env.task_specification = goal
        env._goal = goal
        env.reset()

        agent.clear_z()
        episode_returns = []

        for traj_idx in range(num_trajs):
            path = rollout(
                env, agent,
                max_path_length=max_path_length,
                accum_context=True,
            )
            ep_return = float(np.sum(path['rewards']))
            episode_returns.append(ep_return)

            if traj_idx >= 0:
                agent.infer_posterior(agent.context)

        mean_ret = np.mean(episode_returns)
        adapted_ret = episode_returns[-1]
        results.append({
            'goal_position': goal,
            'mean_return': mean_ret,
            'adapted_return': adapted_ret,
            'all_returns': episode_returns,
        })
        print(f"  goal={goal:>8.2f}  mean_return={mean_ret:>8.2f}  adapted_return={adapted_ret:>8.2f}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate PEARL on Cheetah Multitask')
    parser.add_argument('--exp-dir', type=str, required=True,
                        help='Path to experiment directory (contains weights/ folder)')
    parser.add_argument('--config', type=str, default='configs/pearl_cheetah_multi_config.json',
                        help='Path to config JSON file')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('--num-trajs', type=int, default=3,
                        help='Number of trajectories per task (first is exploration, rest are adaptation)')
    parser.add_argument('--eval-train-tasks', action='store_true',
                        help='Also evaluate on training tasks')
    parser.add_argument('--save-results', action='store_true',
                        help='Save detailed results to JSON')
    parser.add_argument('--custom-velocities', type=str, default=None,
                        help='Comma-separated list of custom velocities to test, e.g. "1.0,2.0,3.0,4.0,5.0"')
    parser.add_argument('--custom-goals', type=str, default=None,
                        help='Comma-separated list of custom goal positions, e.g. "5.0,10.0,15.0,20.0,-5.0,-10.0"')
    args = parser.parse_args()

    # Load config
    variant = default_config.copy()
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            exp_params = json.load(f)
        variant = deep_update_dict(exp_params, variant)

    # Setup GPU
    ptu.set_gpu_mode(True, args.gpu)

    # Load weights
    weights_dir = os.path.join(args.exp_dir, 'weights')
    if not os.path.exists(weights_dir):
        print(f"Error: weights directory not found at {weights_dir}")
        print(f"Available contents: {os.listdir(args.exp_dir)}")
        sys.exit(1)

    print(f"Loading model from: {weights_dir}")
    env, agent = load_agent(variant, weights_dir)

    if ptu.gpu_enabled():
        agent.to('cuda')

    tasks = env.get_all_task_idx()
    n_train = variant['n_train_tasks']
    n_eval = variant['n_eval_tasks']
    train_tasks = list(tasks[:n_train])
    eval_tasks = list(tasks[-n_eval:])

    print(f"\nEnvironment: {variant['env_name']}")
    print(f"Train tasks: {n_train}, Eval tasks: {n_eval}")
    print(f"Trajectories per task: {args.num_trajs}")
    print(f"Max path length: {variant['algo_params']['max_path_length']}")

    all_results = {}

    # ==================== 1. Evaluate on test tasks ====================
    print(f"\n{'='*80}")
    print(f" Evaluating on {len(eval_tasks)} TEST tasks...")
    print(f"{'='*80}")
    test_results = evaluate_on_tasks(
        env, agent, eval_tasks, variant,
        num_trajs=args.num_trajs,
        num_exploration_trajs=variant['algo_params']['num_exp_traj_eval'],
    )
    print_summary(test_results, "TEST Tasks Summary")
    all_results['test_tasks'] = test_results

    # ==================== 2. Evaluate on train tasks (optional) ====================
    if args.eval_train_tasks:
        print(f"\n{'='*80}")
        print(f" Evaluating on {len(train_tasks)} TRAIN tasks...")
        print(f"{'='*80}")
        train_results = evaluate_on_tasks(
            env, agent, train_tasks, variant,
            num_trajs=args.num_trajs,
            num_exploration_trajs=variant['algo_params']['num_exp_traj_eval'],
        )
        print_summary(train_results, "TRAIN Tasks Summary")
        all_results['train_tasks'] = train_results

    # ==================== 3. Custom velocity evaluation ====================
    if args.custom_velocities:
        velocities = [float(v.strip()) for v in args.custom_velocities.split(',')]
        custom_vel_results = evaluate_custom_velocities(
            env, agent, variant, velocities, num_trajs=args.num_trajs
        )
        all_results['custom_velocities'] = custom_vel_results

    # ==================== 4. Custom goal position evaluation ====================
    if args.custom_goals:
        goals = [float(g.strip()) for g in args.custom_goals.split(',')]
        custom_goal_results = evaluate_custom_goals(
            env, agent, variant, goals, num_trajs=args.num_trajs
        )
        all_results['custom_goals'] = custom_goal_results

    # ==================== 5. Default: sweep velocities and goals ====================
    if not args.custom_velocities and not args.custom_goals:
        # Sweep forward velocities [1.0, 5.0]
        sweep_vels = np.linspace(1.0, 5.0, 9).tolist()
        print(f"\n--- Velocity Sweep: {sweep_vels} ---")
        vel_sweep_results = evaluate_custom_velocities(
            env, agent, variant, sweep_vels, num_trajs=args.num_trajs
        )
        all_results['velocity_sweep'] = vel_sweep_results

        # Sweep goal positions [5.0, 25.0]
        sweep_goals = np.linspace(5.0, 25.0, 5).tolist()
        print(f"\n--- Goal Position Sweep: {sweep_goals} ---")
        goal_sweep_results = evaluate_custom_goals(
            env, agent, variant, sweep_goals, num_trajs=args.num_trajs
        )
        all_results['goal_sweep'] = goal_sweep_results

    # ==================== Save results ====================
    if args.save_results:
        results_path = os.path.join(args.exp_dir, 'eval_results.json')
        # Convert numpy types for JSON serialization
        def convert(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=convert)
        print(f"\nResults saved to: {results_path}")

    env.close()
    print("\nEvaluation complete!")


if __name__ == '__main__':
    main()
