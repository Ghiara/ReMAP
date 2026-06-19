# ReMAP




<div align="center">

Official implementation of the paper: **Knowledge Reutilization in Meta-Reinforcement Learning**

Yuan Meng<sup>1,2,*</sup>, Bo Wang<sup>1,2,*</sup>, Juan de los Rios Ruiz<sup>1</sup>, Xiangtong Yao<sup>1</sup>, 
Zhenshan Bing<sup>2,&dagger;</sup>, Fuchun Sun<sup>3</sup>, and Alois Knoll<sup>1</sup> 

<sup>1</sup>School of Computation, Information and Technology, Technical University of Munich, Munich, Germany  
<sup>2</sup>State Key Laboratory for Novel Software Technology, Nanjing University, Suzhou, China  
<sup>3</sup>Department of Computer Science and Technology, Tsinghua University, Beijing, China

<sup>*</sup>Equal contribution. The work was done during the research visit at Nanjing University.  
<sup>&dagger;</sup>Corresponding author: bing@nju.edu.cn

</div>

## Abstract

Meta-reinforcement learning enables fast adaptation by extracting shared structure from related tasks, but existing end-to-end methods often couple task inference with embodiment-specific control.
This coupling can obscure non-parametric task semantics, reduce sample efficiency, and limit cross-agent reuse.
We propose a meta-knowledge reutilization framework that learns task-level knowledge on a dynamics-simplified agent and transfers it to heterogeneous agents.
The framework uses a Bayesian non-parametric prior to organize latent task modes and a high-level policy to generate task-level magnitude guidance.
To bridge reusable task knowledge with different embodiments, we introduce a semantic-magnitude interface and a lightweight temporal adaptor, which convert frozen meta-knowledge into temporally aligned subgoals for embodiment-specific low-level controllers.
Experiments on multiple locomotion agents show that our framework reduces final-step tracking error by \(94.75\%\)--\(99.79\%\) compared with recent state-of-the-art baselines and achieves comparable deployment performance with about \(23.8\%\) of their interaction data.

## Introduction to Repository

### Installation

Run all installation commands from the repository root:

```bash
git clone https://github.com/Ghiara/ReMAP.git
cd ReMAP
```

#### 1. Create the environment

Assume we use conda to manage the dependencies. The environment is [`environment.yml`](environment.yml):

```bash
conda env create -f environment.yml
conda activate remap
```

#### 2. Install PyTorch if needed

PyTorch / CUDA is machine-dependent, so if your environment does not already provide it, install a matching build before the editable install. One setup used with this project is:

```bash
conda install pytorch torchvision torchaudio cudatoolkit=11.8 -c pytorch
```

#### 3. Install the project once from the root

This repository now exposes a root editable install, so you no longer need a long list of `pip install -e` commands for individual subfolders:

```bash
python -m pip install -e .
```

This root install wires up the packages imported by the project, including:

- `ReMAP`
- `smrl`
- `specific`
- `meta_envs`
- `mrl_analysis`
- `rlkit`
- `bnpy`


### MuJoCo Setup

Several training and evaluation scripts depend on MuJoCo 2.1.0. A typical setup is:

```bash
mkdir -p ~/.mujoco
tar -xzf mujoco210-linux-x86_64.tar.gz -C ~/.mujoco
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin"
```

If your NVIDIA libraries are not already on the loader path, you may also need:

```bash
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/lib/nvidia"
```


### Project Structure

Key folders:

```text
.
|-- ReMAP/
|   |-- configs/
|   |-- evaluation/
|   |-- scripts/
|   |-- train/
|   `-- utils/
|-- third_party/
|-- environment.yml
|-- setup.py
```

- `ReMAP/configs/`: experiment and baseline configuration files
- `ReMAP/train/`: training entrypoints and launchers
- `ReMAP/evaluation/`: evaluation and deployment entrypoints
- `ReMAP/scripts/`: plotting, logging, and analysis helpers
- `ReMAP/utils/`: PEARL / RL2 related utilities
- `third_party/`: vendored dependencies used by the project

### Training

Training instructions are documented in [`ReMAP/train/TRAINING_README.md`](ReMAP/train/TRAINING_README.md).

### Evaluation

Evaluation instructions are documented in [`ReMAP/evaluation/EVALUATION_README.md`](ReMAP/evaluation/EVALUATION_README.md).

### Data

The source data for the paper experiments are provided in [`data/`](data/).
They include CSV files for ReMAP results, baseline comparisons, ablation
studies, reward statistics, and single-task case studies used in the main paper
and supplementary material.
