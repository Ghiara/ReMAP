# Training README

This folder now includes a one-click launcher:

```bash
bash ReMAP/train/run_training.sh <target> [args...]
```

The launcher should be run from the repository root, the directory that contains the top-level `README.md`.

The launcher injects the repository root and the bundled Meta_RL submodules into `PYTHONPATH`, so imports such as `ReMAP.*`, `third_party.*`, `meta_envs.*`, `rlkit.*`, and `mrl_analysis.*` resolve consistently across all training entrypoints.

## Supported targets

- `low-level`
  Runs [train_low_level_policy.py](train_low_level_policy.py).
  Example:
  ```bash
  bash ReMAP/train/run_training.sh low-level --env cheetah
  ```

- `high-single`
  Runs [train_single_task_inference_high_level_policy.py](train_single_task_inference_high_level_policy.py).
  This script currently uses its built-in defaults and saves to `output/goal_tracking_toy/<timestamped_run_name>`.
  Example:
  ```bash
  bash ReMAP/train/run_training.sh high-single
  ```

- `high-multi`
  Runs [train_multi_task_inference_high_level_policy.py](train_multi_task_inference_high_level_policy.py).
  Common example:
  ```bash
  bash ReMAP/train/run_training.sh high-multi --gpu 0 --ti_option dpmm --name exp1
  ```
  Optional CLI args supported by the underlying script:
  - `--config <json_path>`
  - `--name <run_suffix>`
  - `--ti_option <dpmm|single_gaussian|true_gmm|stick_break>`
  - `--gpu <id>`
  - `--num_workers <n>`
  - `--use_mp`

- `baseline-cemrl`
  Runs [baselines_training/run_cemrl_cheetah_baseline.py](baselines_training/run_cemrl_cheetah_baseline.py).
  Example:
  ```bash
  bash ReMAP/train/run_training.sh baseline-cemrl --gpu 0
  ```

- `baseline-melts`
  Runs [baselines_training/run_melts_cheetah_baseline.py](baselines_training/run_melts_cheetah_baseline.py).
  Example:
  ```bash
  bash ReMAP/train/run_training.sh baseline-melts --gpu 0
  ```

- `baseline-pearl`
  Runs [baselines_training/run_pearl_cheetah_baseline.py](baselines_training/run_pearl_cheetah_baseline.py).
  Example:
  ```bash
  bash ReMAP/train/run_training.sh baseline-pearl --gpu 0
  ```

- `baseline-rl2`
  Runs [baselines_training/run_rl2_cheetah_baseline.py](baselines_training/run_rl2_cheetah_baseline.py).
  Example:
  ```bash
  bash ReMAP/train/run_training.sh baseline-rl2 --gpu 0
  ```

## Composite targets

- `baseline-all`
  Sequentially runs CEMRL, MELTS, PEARL, and RL2 with each script's default settings.

- `remap-all`
  Sequentially runs:
  1. low-level policy training on `cheetah`
  2. low-level policy training on `hopper`
  3. low-level policy training on `walker2d`
  4. low-level policy training on `ant`
  5. single-task high-level training
  6. multi-task high-level training

- `all`
  Sequentially runs the full ReMAP pipeline above, then `baseline-all`.

For composite targets, the launcher intentionally uses default arguments only. If you need custom GPUs, config files, or output directories, run each target individually.

## Typical outputs

- Low-level policy:
  `output/low_level_policy/`

- Single-task high-level policy:
  `output/goal_tracking_toy/<timestamped_run_name>/`

- Multi-task high-level policy:
  Usually under the `base_log_dir` configured in the selected JSON/default config.

- Baselines:
  Usually under:
  - `output/cemrl_baseline/`
  - `output/melts_baseline/`
  - `output/pearl_baseline/`
  - `output/rl2_baseline/`

## Compatibility note

Some baseline scripts still depend on the compatibility wrapper `run_task_inference_high_level_policy_training.py`, which forwards to the active multi-task implementation in `train_multi_task_inference_high_level_policy.py`.
