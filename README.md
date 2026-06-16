# ReMAP

ReMAP is a meta-reinforcement learning project for inference reutilization. This repository vendors the project code together with the local research dependencies it needs under `third_party/`.

## Installation

Run all installation commands from the repository root:

```bash
git clone <your-repo-url>
cd ReMAP
```

### 1. Create the environment

The recommended base environment is still [`environment.yml`](environment.yml):

```bash
conda env create -f environment.yml
conda activate ReMAP
```

### 2. Install PyTorch if needed

PyTorch / CUDA is machine-dependent, so if your environment does not already provide it, install a matching build before the editable install. One setup used with this project is:

```bash
conda install pytorch torchvision torchaudio cudatoolkit=11.8 -c pytorch
```

### 3. Install the project once from the root

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

It also keeps the vendored code under `third_party/` usable without depending on your local clone path or parent directory name.

## MuJoCo Setup

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

## Dependencies

The Python dependencies used by this project are now centralized in the root editable install (`setup.py`), instead of being spread across several manual sub-package install commands.

The repo still includes:

- [`environment.yml`](environment.yml) for the full Conda environment
- [`setup.py`](setup.py) for the root `pip install -e .`
- vendored research dependencies under `third_party/`

## Project Structure

Key folders:

```text
.
|-- configs/
|-- evaluation/
|-- scripts/
|-- third_party/
|-- train/
|-- utils/
|-- environment.yml
|-- setup.py
```

- `configs/`: experiment and baseline configuration files
- `train/`: training entrypoints and launchers
- `evaluation/`: evaluation and deployment entrypoints
- `scripts/`: plotting, logging, and analysis helpers
- `utils/`: PEARL / RL2 related utilities
- `third_party/`: vendored dependencies used by the project

## Training

Training instructions are documented in [`train/TRAINING_README.md`](train/TRAINING_README.md).

## Evaluation

Evaluation instructions are documented in [`evaluation/EVALUATION_README.md`](evaluation/EVALUATION_README.md).
