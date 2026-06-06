#!/usr/bin/env python3
"""
Evaluate a trained RL2 agent on cheetah-multi-task environment.

Tests the agent's ability to:
  - Track different target velocities (forward/backward)
  - Reach different goal positions (forward/backward)
  - Other tasks: flip, stand, jump, direction

Outputs per-task-type performance breakdown for comparison with PEARL and other baselines.

Usage:
    # Evaluate the latest run
    python eval_rl2_cheetah_baseline.py \
        --exp-dir output/rl2_baseline/cheetah-multi-task/<TIMESTAMP> \
        --config configs/rl2_cheetah_multi_config.json \
        --gpu 0

    # Evaluate with custom tasks and save results
    python eval_rl2_cheetah_baseline.py \
        --exp-dir output/rl2_baseline/cheetah-multi-task/<TIMESTAMP> \
        --config configs/rl2_cheetah_multi_config.json \
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

from rlkit.envs import ENVS
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.torch.rl2.rl2_policies import LSTMPolicy, LSTMQNetwork
from rlkit.torch.rl2.rl2_agent import RL2Agent
from rlkit.torch.networks import FlattenMlp
from rlkit.samplers.util import rollout
from configs.pearl_default import default_config
import rlkit.torch.pytorch_util as ptu


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
    """Load a trained RL2 agent from saved weights."""
    env = NormalizedBoxEnv(ENVS[variant['env_name']](**variant['env_params']))
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    
    net_size = variant.get('net_size', 300)
    
    # Instantiate networks
    policy = LSTMPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_size=net_size,
        num_layers=1,
    )
    
    qf1 = LSTMQNetwork(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_size=net_size,
    )
    
    qf2 = LSTMQNetwork(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_size=net_size,
    )
    
    vf = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim,
        output_size=1,
    )
    
    # Create agent
    agent = RL2Agent(
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        vf=vf,
        obs_dim=obs_dim,
        action_dim=action_dim,
    )
    
    # Load weights
    print(f"Loading weights from {weights_dir}")
    policy.load_state_dict(torch.load(os.path.join(weights_dir, 'policy.pth')))
    qf1.load_state_dict(torch.load(os.path.join(weights_dir, 'qf1.pth')))
    qf2.load_state_dict(torch.load(os.path.join(weights_dir, 'qf2.pth')))
    vf.load_state_dict(torch.load(os.path.join(weights_dir, 'vf.pth')))
    
    return env, agent


def evaluate_on_tasks(env, agent, task_indices, variant, num_trajs=1, num_exploration_trajs=0):
    """
    Evaluate agent on specific tasks
    
    Args:
        env: environment
        agent: RL2 agent
        task_indices: list of task indices to evaluate on
        num_trajs: number of trajectories per task
        num_exploration_trajs: number of exploration trajectories before evaluation
    
    Returns:
        results: dict with per-task and aggregate statistics
    """
    max_path_length = variant['algo_params']['max_path_length']
    
    results = {}
    task_returns = defaultdict(list)
    task_succ_rates = defaultdict(list)
    
    for task_idx in task_indices:
        env.reset_task(task_idx)
        task_name = TASK_NAMES.get(env.task_id, f'task_{task_idx}')
        
        print(f"  Evaluating task {task_idx} ({task_name})...")
        
        # Reset agent for this task
        agent.clear_z()
        agent.reset_hidden_states()
        
        # Collect exploration trajectories (for context)
        for _ in range(num_exploration_trajs):
            path = rollout(
                env,
                agent,
                max_path_length=max_path_length,
                accum_context=True,
                animated=False,
                save_obs_dict=False,
            )
        
        # Collect evaluation trajectories
        task_returns_list = []
        task_succ_list = []
        
        for traj_num in range(num_trajs):
            path = rollout(
                env,
                agent,
                max_path_length=max_path_length,
                accum_context=(traj_num == 0),  # Only first traj contributes to context
                animated=False,
                save_obs_dict=False,
            )
            
            task_return = np.sum(path['rewards'])
            task_returns_list.append(task_return)
            
            # Check task success (if env provides it)
            if 'success' in path:
                task_succ = np.mean(path['success'])
            else:
                task_succ = 0.0
            task_succ_list.append(task_succ)
        
        task_returns[task_name].extend(task_returns_list)
        task_succ_rates[task_name].extend(task_succ_list)
    
    # Aggregate results
    results['task_returns'] = dict(task_returns)
    results['task_succ_rates'] = dict(task_succ_rates)
    
    # Compute statistics
    all_returns = []
    for returns in task_returns.values():
        all_returns.extend(returns)
    
    results['avg_return'] = np.mean(all_returns)
    results['std_return'] = np.std(all_returns)
    results['min_return'] = np.min(all_returns)
    results['max_return'] = np.max(all_returns)
    
    return results


def evaluate_custom_velocities(env, agent, variant, velocities, num_trajs=1):
    """Evaluate on specific velocity targets"""
    results = {}
    velocity_returns = {}
    
    for vel in velocities:
        print(f"  Evaluating velocity target: {vel}")
        env.set_task({'velocity': vel})
        
        agent.clear_z()
        agent.reset_hidden_states()
        
        traj_returns = []
        for _ in range(num_trajs):
            path = rollout(
                env,
                agent,
                max_path_length=variant['algo_params']['max_path_length'],
                animated=False,
                save_obs_dict=False,
            )
            traj_returns.append(np.sum(path['rewards']))
        
        velocity_returns[f'vel_{vel}'] = {
            'mean': np.mean(traj_returns),
            'std': np.std(traj_returns),
            'returns': traj_returns
        }
    
    results['velocity_returns'] = velocity_returns
    return results


def evaluate_custom_goals(env, agent, variant, goals, num_trajs=1):
    """Evaluate on specific goal positions"""
    results = {}
    goal_returns = {}
    
    for goal in goals:
        print(f"  Evaluating goal position: {goal}")
        env.set_task({'goal_pos': goal})
        
        agent.clear_z()
        agent.reset_hidden_states()
        
        traj_returns = []
        for _ in range(num_trajs):
            path = rollout(
                env,
                agent,
                max_path_length=variant['algo_params']['max_path_length'],
                animated=False,
                save_obs_dict=False,
            )
            traj_returns.append(np.sum(path['rewards']))
        
        goal_returns[f'goal_{goal}'] = {
            'mean': np.mean(traj_returns),
            'std': np.std(traj_returns),
            'returns': traj_returns
        }
    
    results['goal_returns'] = goal_returns
    return results


def print_summary(results, title):
    """Print evaluation results summary"""
    print(f"\n{'='*80}")
    print(f" {title}")
    print(f"{'='*80}")
    
    print(f"\nAggregate Statistics:")
    print(f"  Avg Return:    {results['avg_return']:.2f}")
    print(f"  Std Return:    {results['std_return']:.2f}")
    print(f"  Min Return:    {results['min_return']:.2f}")
    print(f"  Max Return:    {results['max_return']:.2f}")
    
    print(f"\nPer-Task Performance:")
    if 'task_returns' in results:
        for task_name, returns_list in results['task_returns'].items():
            mean_return = np.mean(returns_list)
            std_return = np.std(returns_list)
            print(f"  {task_name:20s}: {mean_return:8.2f} ± {std_return:6.2f}")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate trained RL2 agent on Cheetah multitask environment'
    )
    parser.add_argument(
        '--exp-dir',
        type=str,
        required=True,
        help='Path to experiment directory containing saved weights'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/rl2_cheetah_multi_config.json',
        help='Path to config JSON file'
    )
    parser.add_argument(
        '--num-trajs',
        type=int,
        default=1,
        help='Number of evaluation trajectories per task'
    )
    parser.add_argument(
        '--gpu',
        type=int,
        default=0,
        help='GPU device ID'
    )
    parser.add_argument(
        '--eval-train-tasks',
        action='store_true',
        help='Also evaluate on training tasks'
    )
    parser.add_argument(
        '--custom-velocities',
        type=str,
        default=None,
        help='Comma-separated list of velocities to evaluate (e.g., "0.5,1.0,2.0")'
    )
    parser.add_argument(
        '--custom-goals',
        type=str,
        default=None,
        help='Comma-separated list of goal positions to evaluate (e.g., "5.0,10.0,15.0")'
    )
    parser.add_argument(
        '--save-results',
        action='store_true',
        help='Save evaluation results to JSON file'
    )
    
    args = parser.parse_args()
    
    # Load config
    variant = default_config.copy()
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            exp_params = json.load(f)
        variant = deep_update_dict(exp_params, variant)
    
    # Set GPU
    ptu.set_gpu_mode(True, args.gpu)
    
    # Load agent and environment
    env, agent = load_agent(variant, args.exp_dir)
    
    if ptu.gpu_enabled():
        agent.policy.to('cuda')
        agent.qf1.to('cuda')
        agent.qf2.to('cuda')
        agent.vf.to('cuda')
    
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
        num_exploration_trajs=variant['algo_params'].get('num_exp_traj_eval', 1),
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
            num_exploration_trajs=variant['algo_params'].get('num_exp_traj_eval', 1),
        )
        print_summary(train_results, "TRAIN Tasks Summary")
        all_results['train_tasks'] = train_results
    
    # ==================== 3. Custom velocity evaluation ====================
    if args.custom_velocities:
        velocities = [float(v.strip()) for v in args.custom_velocities.split(',')]
        print(f"\n{'='*80}")
        print(f" Evaluating CUSTOM velocities: {velocities}")
        print(f"{'='*80}")
        custom_vel_results = evaluate_custom_velocities(
            env, agent, variant, velocities, num_trajs=args.num_trajs
        )
        all_results['custom_velocities'] = custom_vel_results
    
    # ==================== 4. Custom goal position evaluation ====================
    if args.custom_goals:
        goals = [float(g.strip()) for g in args.custom_goals.split(',')]
        print(f"\n{'='*80}")
        print(f" Evaluating CUSTOM goal positions: {goals}")
        print(f"{'='*80}")
        custom_goal_results = evaluate_custom_goals(
            env, agent, variant, goals, num_trajs=args.num_trajs
        )
        all_results['custom_goals'] = custom_goal_results
    
    # ==================== 5. Default: sweep velocities and goals ====================
    if not args.custom_velocities and not args.custom_goals:
        # Sweep forward velocities [1.0, 5.0]
        sweep_vels = np.linspace(1.0, 5.0, 9).tolist()
        print(f"\n{'='*80}")
        print(f" Velocity Sweep: {sweep_vels}")
        print(f"{'='*80}")
        vel_sweep_results = evaluate_custom_velocities(
            env, agent, variant, sweep_vels, num_trajs=args.num_trajs
        )
        all_results['velocity_sweep'] = vel_sweep_results
        
        # Sweep goal positions [5.0, 25.0]
        sweep_goals = np.linspace(5.0, 25.0, 5).tolist()
        print(f"\n{'='*80}")
        print(f" Goal Position Sweep: {sweep_goals}")
        print(f"{'='*80}")
        goal_sweep_results = evaluate_custom_goals(
            env, agent, variant, sweep_goals, num_trajs=args.num_trajs
        )
        all_results['goal_sweep'] = goal_sweep_results
    
    # ==================== Save results ====================
    if args.save_results:
        results_path = os.path.join(args.exp_dir, 'rl2_eval_results.json')
        
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
