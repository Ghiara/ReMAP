#!/usr/bin/env python3
"""
Train MELTS baseline on Cheetah multi-task environment.

Uses the TIGR framework with DPMM (Dirichlet Process Mixture Model) task
inference, which corresponds to the MELTS algorithm.
Reference: https://github.com/Ghiara/MELTS

Usage:
    python run_melts_cheetah_baseline.py \
        --config ReMAP/configs/melts_cheetah_config.json \
        --gpu 0

    python run_melts_cheetah_baseline.py \
        --gpu 0 --output-dir output/melts_baseline
"""



#这个脚本是为了训练MELTS算法在Cheetah多任务环境中的基线模型。它使用TIGR框架，并且强制使用DPMM（Dirichlet Process Mixture Model）进行任务推断，
# 这就是MELTS算法的核心。用户可以通过命令行参数指定配置文件、GPU设备ID、输出目录等选项。脚本会加载默认配置，并根据提供的JSON配置文件进行覆盖，最后调用experiment函数开始训练过程。


import argparse
import copy
import json
import os
import sys
from datetime import datetime
from pathlib import Path

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
        description='Train MELTS (dpmm inference) on Cheetah multi-task environment'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='ReMAP/configs/melts_cheetah_config.json',
        help='Path to config JSON file',
    )
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output/melts_baseline',
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
    # Force MELTS inference option
    variant['inference_option'] = 'dpmm'
    variant['util_params']['base_log_dir'] = args.output_dir
    variant['util_params']['exp_name'] = 'melts_cheetah_dpmm'
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
    # Always keep MELTS inference option
    variant['inference_option'] = 'dpmm'
    variant['train_or_showcase'] = 'train'

    if 'dpmm_params' in variant:
        temp_root = variant['dpmm_params'].get('save_dir', 'melts/temp_bnp')
        run_name = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}_{variant['util_params']['exp_name']}"
        unique_temp_dir = os.path.join(temp_root, run_name)
        variant['dpmm_params']['save_dir'] = unique_temp_dir
        if 'reconstruction_params' in variant:
            variant['reconstruction_params']['temp_folder'] = unique_temp_dir

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(_resolve_project_path('logs'), exist_ok=True)
    if 'dpmm_params' in variant:
        os.makedirs(variant['dpmm_params']['save_dir'], exist_ok=True)

    print("=" * 70)
    print("MELTS Training  (dpmm inference, cheetah-multi-task)")
    print(f"  Output dir : {args.output_dir}")
    print(f"  GPU        : {args.gpu}")
    print(f"  Epochs     : {variant['algo_params']['num_train_epochs']}")
    if 'dpmm_params' in variant:
        print(f"  DPMM temp  : {variant['dpmm_params']['save_dir']}")
    print("=" * 70)

    experiment(variant)


if __name__ == '__main__':
    main()
