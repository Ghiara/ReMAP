# ReMAP: Inference Reutilization for Meta-Reinforcement Learning

This repository contains the current implementation of our ReMAP pipeline for meta-reinforcement learning with inference reutilization. The codebase is organized around three main pieces:

1. low-level policy training on the target agent,
2. high-level policy and task inference module training,
3. evaluation against both ReMAP and several meta-RL baselines.

The repository also includes all major third-party dependencies used by the project under `third_party/`, so the README below focuses on the code structure that is actually present in this repo today.

## Installation

### 1. Clone the repository

Clone the repository first and enter the project root:

```bash
git clone https://github.com/Ghiara/ReMAP.git
cd ReMAP
```

If you are working from a fork or a renamed local directory, just make sure you end up in the repository root, where `README.md`, `environment.yml`, `train/`, and `third_party/` are located.

### 2. Create the Conda environment

We recommend using Conda, and the current environment definition for this project is [`environment.yml`](environment.yml).

Create the environment with:

```bash
conda env create -f environment.yml
conda activate ReMAP
```

The environment name defined in the file is `ReMAP`.



### 3. Install local third-party packages

After activating `ReMAP`, install the local packages used by this repository:

```bash
pip install -e ./third_party/CARE
pip install -e ./third_party/Meta_RL
pip install -e ./third_party/Meta_RL/meta-environments-main
pip install -e ./third_party/Meta_RL/submodules/MRL-analysis-tools-main
pip install -e ./third_party/Meta_RL/submodules/rlkit
pip install -e ./third_party/Meta_RL/submodules/symmetrizer
pip install -e ./third_party/SAC
pip install -e ./third_party/meta_rand_envs
pip install -e ./third_party/rand_param_envs
```

### 4. CUDA and PyTorch

If your machine does not already have CUDA configured, you may need:

```bash
sudo apt install nvidia-cuda-toolkit
nvcc --version
```

If PyTorch needs to be installed or repaired manually, one working setup used in this project is:

```bash
conda install pytorch torchvision torchaudio cudatoolkit=11.8 -c pytorch
pip install torch==2.1.0 torchaudio==2.1.0 torchvision==0.16.0
```


## MuJoCo Setup

Several training and evaluation scripts in this repository depend on MuJoCo-based environments.

### 1. Download MuJoCo

Download the MuJoCo 2.1.0 Linux package from:

`https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz`

### 2. Extract MuJoCo to the expected location

This project expects the MuJoCo files under:

```bash
/root/.mujoco/mujoco210
```

A typical setup looks like this:

```bash
mkdir -p /root/.mujoco
tar -xzf mujoco210-linux-x86_64.tar.gz -C /root/.mujoco
```

After extraction, the binaries should be located at:

```bash
/root/.mujoco/mujoco210/bin
```

### 3. Export MuJoCo library paths

Before running training or evaluation, make sure these paths are available:

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
```

If you do not want to set them every time, add the same lines to `~/.bashrc`.



## Project Structure

The most important tracked folders in this repository are:

```text
.
|-- configs/
|-- evaluation/
|-- pearl_util/
|-- rl2_util/
|-- scripts/
|-- third_party/
|-- train/
|-- environment.yml
```

### `configs/`

Configuration files for training and evaluation.

- `toy_config.py`, `toy2d_config.py`: configuration for the simple environment used in task inference learning.
- `transfer_config.py`: transfer-related configuration.
- `half_cheetah_multi.py`, `hopper_multi.py`, `walker_multi.py`, `ant_multi.py`: low-level policy environment configs.
- `*_config.json`: baseline configs for CEMRL, MELTS, PEARL, and RL2.
- `default.py`, `pearl_default.py`, `rl2_default.py`: default config templates used by the different pipelines.

### `train/`

Main training entrypoints for this project.

- `train/train_low_level_policy.py`: trains the low-level policy for the target agent.
- `train/run_task_inference_high_level_policy_training.py`: trains the high-level policy together with the task inference module.
- `train/baselines_training/`: baseline training scripts for CEMRL, MELTS, PEARL, and RL2.

In other words, `train/` contains:

- low-level policy training,
- high-level policy and task inference module training,
- baseline training.

### `evaluation/`

Evaluation and deployment scripts.

- `evaluation/task_inference_high_level_cross_agent_deployment.py`: ReMAP deployment / cross-agent evaluation script.
- `evaluation/baselines_evaluation/`: evaluation scripts for the baselines and their tracking performance.

This folder therefore contains:

- baseline evaluation scripts,
- ReMAP deployment and evaluation scripts.

### `third_party/`

Vendored external dependencies used by this repository.

- `third_party/SAC`: low-level SAC implementation and task-conditioned MuJoCo environments used for low-level control.
- `third_party/tigr`: task inference and high-level training components used by the ReMAP pipeline.
- `third_party/rlkit`: RL infrastructure used by the meta-RL and baseline pipelines.
- `third_party/CARE`: BNP / clustering-related dependency used by inference components.
- `third_party/Meta_RL`: upstream meta-RL codebase and submodules used for related components and utilities.
- `third_party/meta_rand_envs`, `third_party/rand_param_envs`: environment packages for meta-RL experiments.

### `scripts/`

Utility scripts for analysis, plotting, logging, and batch execution.

- plotting and diagnostic helpers such as `plot_from_csv.py`, `plot_latent.py`, and `diagnostic_decoder_prediction_confusion_matrix.py`,
- logging helpers such as `tb_logging.py` and `vis_logging.py`,
- shell helpers in `scripts/bash_training_scripts/` for launching repeated experiments and multi-seed runs.

### `pearl_util/` and `rl2_util/`

Utilities used by the PEARL and RL2 baselines respectively.

## Training

### 1. Train the low-level policy

The low-level controller is trained through:

```bash
python train/train_low_level_policy.py --env cheetah
```

Available environment choices in the current script are:

```bash
python train/train_low_level_policy.py --env hopper
python train/train_low_level_policy.py --env walker2d
python train/train_low_level_policy.py --env ant
```

The corresponding environment-specific configs are loaded from `configs/half_cheetah_multi.py`, `configs/hopper_multi.py`, `configs/walker_multi.py`, and `configs/ant_multi.py`.

Outputs are written under `output/low_level_policy/`.

### 2. Train the task inference module and high-level policy

The main ReMAP training entrypoint is:

```bash
python train/run_task_inference_high_level_policy_training.py
```

By default, this script uses the internal toy/high-level configuration defined in `configs/toy_config.py`. You can modify that file directly for your experiment.

The script also exposes useful CLI options:

```bash
python train/run_task_inference_high_level_policy_training.py --name exp1 --ti_option dpmm
```

Common `ti_option` values in the current implementation include:

- `dpmm`
- `single_gaussian`
- `true_gmm`
- `stick_break`

Outputs are written to the log directory defined in the corresponding config.

### 3. Train baselines

Baseline training scripts are under `train/baselines_training/`.

Examples:

```bash
python train/baselines_training/run_cemrl_cheetah_baseline.py
python train/baselines_training/run_melts_cheetah_baseline.py
python train/baselines_training/run_pearl_cheetah_baseline.py
python train/baselines_training/run_rl2_cheetah_baseline.py
```

These scripts use the JSON configs in `configs/`.

## Evaluation

### ReMAP deployment / evaluation

For ReMAP deployment and cross-agent evaluation:

```bash
python evaluation/task_inference_high_level_cross_agent_deployment.py
```

### Baseline evaluation

Baseline evaluation scripts are under `evaluation/baselines_evaluation/`.

Examples:

```bash
python evaluation/baselines_evaluation/eval_pearl_tracking.py --exp-dir <path_to_experiment_dir>
python evaluation/baselines_evaluation/eval_rl2_tracking.py --exp-dir <path_to_experiment_dir>
python evaluation/baselines_evaluation/eval_cemrl_melts_tracking.py --cemrl-dir <path> --melts-dir <path>
python evaluation/baselines_evaluation/baseline_tarcking_evalaution.py
```

These scripts are mainly used to compare target tracking behavior and adaptation quality between ReMAP and the baselines.

## Outputs

Experiment artifacts are typically written to:

- `output/`: checkpoints, progress logs, videos, and run directories,
- `final_results/`: collected result summaries and tracking evaluation artifacts,
- `logs/`: auxiliary logs.

## Notes

- Some old script names from earlier versions of the project are no longer the main entrypoints. This README only documents the scripts that are currently present in the tracked repository.
- If a run fails at import time, first verify that the editable installs from `third_party/` were completed in the active `ReMAP` environment.
- For MuJoCo or rendering issues, check your CUDA, OpenGL, and `LD_LIBRARY_PATH` setup first.
