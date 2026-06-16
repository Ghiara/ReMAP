"""
This file serves as an example how to instantiate an algorithm instance
from a configuration file and run it afterwards.

Author(s):
    Julius Durmann
Contact:
    julius.durmann@tum.de
Date:
    2023-04-06
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
META_RL_ROOT = PROJECT_ROOT / 'third_party' / 'Meta_RL'
BUNDLED_SUBMODULES = [
    META_RL_ROOT,
    META_RL_ROOT / 'submodules' / 'rlkit',
    META_RL_ROOT / 'submodules' / 'meta-environments-main',
    META_RL_ROOT / 'submodules' / 'MRL-analysis-tools-main',
    PROJECT_ROOT,
]
for module_path in reversed(BUNDLED_SUBMODULES):
    module_path = str(module_path)
    if module_path in sys.path:
        sys.path.remove(module_path)
    sys.path.insert(0, module_path)

import os
import torch
from datetime import datetime
from zoneinfo import ZoneInfo

from smrl.experiment.experiment_setup import setup_experiment
from configs.base_configuration import config
# from configs.base_configuration_bigger_envs import config
from configs.environment_factory import toy1d_rand, toy1d


# Environment
config['environment_factory'] = toy1d


# GPU available?
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"    # Makes sure the listing is in the same order as with nvidia-smi command
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"    # Set to "" if you do not want to use GPUs
print(f"GPU available: {'Yes' if torch.cuda.is_available() else 'No'}")

# Multithreading
os.environ["MULTITHREADING"] = "False"   # Set to "False" to not use multithreading
if os.environ["MULTITHREADING"] == "True":
    import ray
    ray.init(num_cpus=12)


# Setup experiment, modify logging parameters
# experiment_name = f"{config['environment_factory'].__name__}_{config['description']['name']}"
experiment_name = "toy1d_MaxAction_1"
config['algorithm_kwargs']['num_epochs'] = 5_000
current_time = datetime.now(ZoneInfo('Europe/Berlin'))
timestamped_experiment_name = f"{experiment_name}_{current_time.strftime('%Y-%m-%d_%H-%M-%S')}"
logger_kwargs = {
    'log_dir': str(PROJECT_ROOT / "output" / "toy1d-single-task" / timestamped_experiment_name),
    'snapshot_mode': 'gap_and_last',
    'snapshot_gap': 10,
}
algorithm, description = setup_experiment(
    experiment_name, 
    config,
    # path_to_weights='/home/ubuntu/juan/Meta-RL/data/experiments_thesis/step1_biggerNN_-10_10/_2024-05-15_09-16-38',
    logger_kwargs=logger_kwargs
)


# RUN EXPERIMENT
algorithm.train()
