#!/usr/bin/env python3
"""Run PEARL training on Cheetah multitask environment."""
import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from pearl_util.pearl_launch_experiment import experiment
from configs.pearl_default import default_config


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


def deep_update_dict(fr, to):
    """Update dict of dicts with new values."""
    for k, v in fr.items():
        if type(v) is dict:
            deep_update_dict(v, to[k])
        else:
            to[k] = v
    return to


def main():
    parser = argparse.ArgumentParser(
        description='Train PEARL on Cheetah Multitask environment'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/pearl_cheetah_multi_config.json',
        help='Path to config JSON file'
    )
    parser.add_argument(
        '--gpu',
        type=int,
        default=0,
        help='GPU device ID to use'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode (verbose output and logs to debug directory)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output/pearl_baseline',
        help='Base output directory for logs'
    )

    args = parser.parse_args()
    args.config = _resolve_project_path(args.config)
    args.output_dir = _resolve_project_path(args.output_dir)

    variant = default_config.copy()
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            exp_params = json.load(f)
        variant = deep_update_dict(exp_params, variant)
    else:
        print(f"Warning: Config file {args.config} not found. Using default config.")

    variant['util_params']['gpu_id'] = args.gpu
    variant['util_params']['debug'] = args.debug
    variant['util_params']['base_log_dir'] = args.output_dir

    print('=' * 80)
    print('PEARL Training Configuration')
    print('=' * 80)
    print(json.dumps(variant, indent=2, default=str))
    print('=' * 80)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(_resolve_project_path('logs'), exist_ok=True)

    print("\nStarting PEARL training...")
    experiment(variant)


if __name__ == '__main__':
    main()
