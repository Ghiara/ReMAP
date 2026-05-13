#!/usr/bin/env python3
"""
Evaluate a trained CEMRL agent on cheetah-multi-task environment.

Loads a trained CEMRL checkpoint and runs showcase_all evaluation,
then prints per-task-type performance for comparison with our method.

Usage:
    # Find the experiment directory first:
    ls output/cemrl_baseline/cheetah-multi-task/

    # Then evaluate:
    python eval_cemrl_cheetah_baseline.py \
        --exp-dir output/cemrl_baseline/cheetah-multi-task/<TIMESTAMP> \
        --itr 500 \
        --gpu 0

    # Save results to JSON:
    python eval_cemrl_cheetah_baseline.py \
        --exp-dir output/cemrl_baseline/cheetah-multi-task/<TIMESTAMP> \
        --itr 500 --save-results
"""
import argparse
import copy
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_toy_training import experiment, deep_update_dict
from configs.default import default_config


TASK_NAMES = {
    0: 'velocity_forward',
    1: 'velocity_backward',
    2: 'goal_forward',
    3: 'goal_backward',
    4: 'flip_forward',
    5: 'stand_front',
    6: 'stand_back',
    7: 'jump',
}


def find_weights_dir(exp_dir):
    """Return path to weights sub-directory inside an experiment folder."""
    weights_dir = os.path.join(exp_dir, 'weights')
    if os.path.isdir(weights_dir):
        return weights_dir
    # Fallback: search recursively
    candidates = glob.glob(os.path.join(exp_dir, '**', 'weights'), recursive=True)
    if candidates:
        return sorted(candidates)[-1]
    return None


def list_available_itrs(weights_dir):
    """Return sorted list of saved iteration numbers."""
    pth_files = glob.glob(os.path.join(weights_dir, 'encoder_itr_*.pth'))
    itrs = []
    for f in pth_files:
        try:
            itr = int(os.path.basename(f).replace('encoder_itr_', '').replace('.pth', ''))
            itrs.append(itr)
        except ValueError:
            pass
    return sorted(itrs)


def parse_showcase_results(results_json):
    """
    Parse showcase_all_results.json produced by the TIGR training algorithm.

    Returns dict: task_type_name -> list of episode returns.
    """
    if not os.path.exists(results_json):
        return None

    with open(results_json, 'r') as f:
        data = json.load(f)

    # data is a list of paths; each path is a list of transition dicts
    task_returns = defaultdict(list)
    for path in data:
        if not path:
            continue
        episode_return = sum(t['reward'] for t in path)
        base_task = path[0].get('base_task', -1)
        task_name = TASK_NAMES.get(base_task, f'task_{base_task}')
        task_returns[task_name].append(episode_return)

    return dict(task_returns)


def print_results_table(task_returns, method_name='CEMRL'):
    """Pretty-print per-task-type evaluation results."""
    print()
    print("=" * 60)
    print(f"  {method_name} Evaluation Results  (cheetah-multi-task)")
    print("=" * 60)
    print(f"  {'Task Type':<25} {'Mean Return':>12} {'Std':>8} {'N':>5}")
    print("  " + "-" * 55)

    all_returns = []
    for task_name in sorted(task_returns.keys()):
        returns = task_returns[task_name]
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        print(f"  {task_name:<25} {mean_ret:>12.2f} {std_ret:>8.2f} {len(returns):>5}")
        all_returns.extend(returns)

    print("  " + "-" * 55)
    if all_returns:
        print(f"  {'OVERALL':<25} {np.mean(all_returns):>12.2f} {np.std(all_returns):>8.2f} {len(all_returns):>5}")
    print("=" * 60)
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate a trained CEMRL agent on cheetah-multi-task'
    )
    parser.add_argument(
        '--exp-dir', type=str, required=True,
        help='Path to the training experiment directory (contains weights/ sub-dir)',
    )
    parser.add_argument(
        '--itr', type=int, default=None,
        help='Checkpoint iteration to load (default: use latest available)',
    )
    parser.add_argument(
        '--config', type=str,
        default='configs/cemrl_cheetah_tigr_config.json',
        help='Config JSON file used during training',
    )
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument(
        '--save-results', action='store_true',
        help='Save evaluation results JSON to --exp-dir',
    )
    args = parser.parse_args()

    # Locate weights directory
    weights_dir = find_weights_dir(args.exp_dir)
    if weights_dir is None:
        print(f"ERROR: No \'weights\' directory found under {args.exp_dir}")
        print("Make sure training completed and weights were saved.")
        sys.exit(1)
    print(f"Weights directory: {weights_dir}")

    # Determine iteration to load
    available_itrs = list_available_itrs(weights_dir)
    if not available_itrs:
        print(f"ERROR: No checkpoint .pth files found in {weights_dir}")
        sys.exit(1)
    itr = args.itr if args.itr is not None else max(available_itrs)
    print(f"Available checkpoints: {available_itrs}")
    print(f"Loading iteration    : {itr}")

    # Build variant for showcase_all
    variant = copy.deepcopy(default_config)
    variant['inference_option'] = 'true_gmm'

    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            exp_params = json.load(f)
        variant = deep_update_dict(exp_params, variant)
    else:
        print(f"Warning: Config \'{args.config}\' not found; using defaults.")

    variant['train_or_showcase'] = 'showcase_all'
    variant['path_to_weights'] = weights_dir
    variant['showcase_itr'] = itr
    variant['util_params']['gpu_id'] = args.gpu
    variant['util_params']['use_gpu'] = True
    variant['util_params']['base_log_dir'] = args.exp_dir
    variant['util_params']['exp_name'] = 'cemrl_eval'
    variant['inference_option'] = 'true_gmm'

    print()
    print("=" * 70)
    print("CEMRL Evaluation  (showcase_all)")
    print(f"  Exp dir    : {args.exp_dir}")
    print(f"  Weights    : {weights_dir}")
    print(f"  Iteration  : {itr}")
    print(f"  GPU        : {args.gpu}")
    print("=" * 70)
    print()

    experiment(variant)

    # Parse and display results
    results_json = os.path.join(
        args.exp_dir, variant['util_params']['exp_name'],
        'showcase_all_results.json'
    )
    # Also try directly in exp_dir
    if not os.path.exists(results_json):
        results_json = os.path.join(args.exp_dir, 'showcase_all_results.json')
    if not os.path.exists(results_json):
        candidates = glob.glob(
            os.path.join(args.exp_dir, '**', 'showcase_all_results.json'),
            recursive=True,
        )
        if candidates:
            results_json = sorted(candidates)[-1]

    if os.path.exists(results_json):
        task_returns = parse_showcase_results(results_json)
        if task_returns:
            print_results_table(task_returns, method_name='CEMRL')
            if args.save_results:
                summary_path = os.path.join(args.exp_dir, 'eval_summary.json')
                summary = {
                    k: {'mean': float(np.mean(v)), 'std': float(np.std(v)), 'n': len(v)}
                    for k, v in task_returns.items()
                }
                with open(summary_path, 'w') as f:
                    json.dump(summary, f, indent=2)
                print(f"Summary saved to: {summary_path}")
    else:
        print(f"Note: Results JSON not found at {results_json}")
        print("Evaluation output may be in a sub-directory of the experiment folder.")


if __name__ == '__main__':
    main()
