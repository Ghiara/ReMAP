#!/usr/bin/env python3
"""
Run RL2 training on Cheetah multitask environment

Usage:
    python run_rl2_cheetah_baseline.py \
        --config configs/rl2_cheetah_config.json \
        --gpu 0
"""
#这个脚本是为了在Cheetah多任务环境上运行RL2训练的。它使用了一个默认的配置，并允许用户通过命令行参数覆盖一些配置选项，
# 如GPU设备ID、调试模式和输出目录。脚本会加载配置文件，设置实验参数，并调用experiment函数来开始训练过程。

import argparse
import json
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

from rl2_util.rl2_launch_experiment import experiment, deep_update_dict
from configs.rl2_default import default_config


# RL2 default config
rl2_default_config = {
    'env_name': 'cheetah-multi-task',
    'n_train_tasks': 80,
    'n_eval_tasks': 40,
    'net_size': 300,
    'path_to_weights': None,
    'env_params': {
        'n_tasks': 120,
        'randomize_tasks': True,
    },
    'algo_params': {
        'meta_batch': 16,
        'num_iterations': 500,
        'num_initial_steps': 2000,
        'num_tasks_sample': 5,
        'num_train_steps_per_itr': 2000,
        'num_evals': 2,
        'num_steps_per_eval': 600,
        'full_eval_interval': 50,
        'batch_size': 256,
        'embedding_batch_size': 64,
        'embedding_mini_batch_size': 64,
        'max_path_length': 200,
        'discount': 0.99,
        'soft_target_tau': 0.005,
        'policy_lr': 3E-4,
        'qf_lr': 3E-4,
        'vf_lr': 3E-4,
        'reward_scale': 5.,
        'num_exp_traj_eval': 1,
        'dump_eval_paths': False,
        'num_inner_steps': 5,  # RL2 specific
        'inner_lr': 1e-4,  # RL2 specific
    },
    'util_params': {
        'base_log_dir': 'output',
        'use_gpu': True,
        'gpu_id': 0,
        'debug': False,
        'docker': False,
    }
}


def main():
    parser = argparse.ArgumentParser(
        description='Train RL2 on Cheetah Multitask environment'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/rl2_cheetah_multi_config.json',
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
        default='output',
        help='Base output directory for logs'
    )
    
    args = parser.parse_args()
    
    # Load config
    variant = rl2_default_config.copy()
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            exp_params = json.load(f)
        variant = deep_update_dict(exp_params, variant)
    else:
        print(f"Warning: Config file {args.config} not found. Using default config.")
    
    # Override with command line arguments
    variant['util_params']['gpu_id'] = args.gpu
    variant['util_params']['debug'] = args.debug
    variant['util_params']['base_log_dir'] = args.output_dir
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Run experiment
    print("\nStarting RL2 training...")
    experiment(variant)


if __name__ == '__main__':
    main()
