#!/usr/bin/env python3
"""
Run PEARL training on Cheetah multitask environment
This script serves as the baseline for comparing with hierarchical training
"""


#这个脚本是为了在Cheetah多任务环境上运行PEARL训练的基线版本。它加载配置文件，设置GPU设备，
# 创建输出目录，并启动PEARL训练实验。用户可以通过命令行参数指定配置文件路径、GPU设备ID、是否启用调试模式以及输出目录。
import argparse
import json
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

from pearl_util.pearl_launch_experiment import experiment
from configs.pearl_default import default_config


def deep_update_dict(fr, to):
    """Update dict of dicts with new values"""
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
        default='output',
        help='Base output directory for logs'
    )
    
    args = parser.parse_args()
    
    # Load config
    variant = default_config.copy()
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
    
    # Print configuration
    print("=" * 80)
    print("PEARL Training Configuration")
    print("=" * 80)
    print(json.dumps(variant, indent=2, default=str))
    print("=" * 80)
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Run experiment
    print("\nStarting PEARL training...")
    experiment(variant)


if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""
Run PEARL training on Cheetah multitask environment
This script serves as the baseline for comparing with hierarchical training
"""
import argparse
import json
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

from pearl_util.pearl_launch_experiment import experiment
from configs.pearl_default import default_config


def deep_update_dict(fr, to):
    """Update dict of dicts with new values"""
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
        default='output',
        help='Base output directory for logs'
    )
    
    args = parser.parse_args()
    
    # Load config
    variant = default_config.copy()
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
    
    # Print configuration
    print("=" * 80)
    print("PEARL Training Configuration")
    print("=" * 80)
    print(json.dumps(variant, indent=2, default=str))
    print("=" * 80)
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Run experiment
    print("\nStarting PEARL training...")
    experiment(variant)


if __name__ == '__main__':
    main()
