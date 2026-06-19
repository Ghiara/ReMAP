#!/usr/bin/env python3
"""
Train CEMRL baseline on Cheetah multi-task environment.

Uses the TIGR framework with true_gmm (Gaussian Mixture Model) task inference,
which corresponds to the CEMRL algorithm.
Reference: https://github.com/zhenshan-bing/cemrl

Usage:
    python run_cemrl_cheetah_baseline.py \
        --config ReMAP/configs/cemrl_cheetah_tigr_config.json \
        --gpu 0

    python run_cemrl_cheetah_baseline.py \
        --gpu 0 --output-dir output/cemrl_baseline
"""
import argparse
import copy
import json
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))


def _resolve_project_path(path_value: str) -> str:
    path = Path(path_value)
    if path.is_absolute():
        cwd = Path.cwd().resolve()
        try:
            relative_to_cwd = path.resolve().relative_to(cwd)
        except ValueError:
            return str(path)
        return str(PROJECT_ROOT / relative_to_cwd)
    return str(PROJECT_ROOT / path)


from ReMAP.train.train_multi_task_inference_high_level_policy import experiment, deep_update_dict
from ReMAP.configs.default import default_config


def main():
    parser = argparse.ArgumentParser(
        description='Train CEMRL (true_gmm) on Cheetah multi-task environment'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='ReMAP/configs/cemrl_cheetah_tigr_config.json',
        help='Path to config JSON file',
    )
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output/cemrl_baseline',
        help='Base output directory for logs',
    )
    parser.add_argument(
        '--num-workers', type=int, default=None,
        help='Number of CPU workers for rollout collection',
    )
    args = parser.parse_args()
    args.config = _resolve_project_path(args.config)
    args.output_dir = _resolve_project_path(args.output_dir)

    # Start from the project default config
    variant = copy.deepcopy(default_config)
    # Force CEMRL inference option
    variant['inference_option'] = 'true_gmm'
    variant['util_params']['base_log_dir'] = args.output_dir
    variant['util_params']['exp_name'] = 'cemrl_cheetah_true_gmm'
    variant['train_or_showcase'] = 'train'

    # Overlay with JSON config if provided
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            exp_params = json.load(f)
        variant = deep_update_dict(exp_params, variant)
    else:
        print(f"Warning: Config file \'{args.config}\' not found. Using defaults.")

    # CLI overrides
    variant['util_params']['gpu_id'] = args.gpu
    variant['util_params']['use_gpu'] = True
    variant['util_params']['debug'] = args.debug
    variant['util_params']['base_log_dir'] = args.output_dir
    if args.num_workers is not None:
        variant['util_params']['num_workers'] = args.num_workers
    # Always keep CEMRL inference option
    variant['inference_option'] = 'true_gmm'
    variant['train_or_showcase'] = 'train'

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(_resolve_project_path('logs'), exist_ok=True)
    os.makedirs(_resolve_project_path('melts/temp_cemrl'), exist_ok=True)

    print("=" * 70)
    print("CEMRL Training  (true_gmm inference, cheetah-multi-task)")
    print(f"  Output dir : {args.output_dir}")
    print(f"  GPU        : {args.gpu}")
    print(f"  Epochs     : {variant['algo_params']['num_train_epochs']}")
    print("=" * 70)

    experiment(variant)


if __name__ == '__main__':
    main()
