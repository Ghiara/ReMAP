# Evaluation README

This folder includes a one-click evaluation launcher:

```bash
bash evaluation/run_evaluation.sh <target> [args...]
```

Run the launcher from the `ReMAP` root directory:

```bash
cd /root/bayes-tmp/bowang/ReMAP
```

The launcher injects `ReMAP` and the bundled Meta_RL submodules into `PYTHONPATH`, so imports such as `third_party.*`, `configs.*`, `train.*`, `meta_envs.*`, `rlkit.*`, and `mrl_analysis.*` resolve consistently across the evaluation entrypoints.

## Supported targets

- `baseline-tracking`
  Runs [baseline_tracking_evalaution.py](/root/bayes-tmp/bowang/ReMAP/evaluation/baseline_tracking_evalaution.py).
  This is the unified tracking evaluation for CEMRL, MELTS, PEARL, and RL2.
  Example:
  ```bash
  bash evaluation/run_evaluation.sh baseline-tracking
  bash evaluation/run_evaluation.sh baseline-tracking --gpu 0 --num-trajs 5
  ```
  Common CLI args supported by the underlying script:
  - `--cemrl-dir <path>`
  - `--melts-dir <path>`
  - `--pearl-dir <path>`
  - `--rl2-dir <path>`
  - `--cemrl-config <json_path>`
  - `--melts-config <json_path>`
  - `--pearl-config <json_path>`
  - `--rl2-config <json_path>`
  - `--gpu <id>`
  - `--num-trajs <n>`
  - `--out-dir <path>`

- `single-deploy`
  Runs [single_task_inference_high_level_cross_agent_deployment.py](/root/bayes-tmp/bowang/ReMAP/evaluation/single_task_inference_high_level_cross_agent_deployment.py).
  This script currently uses its built-in defaults and config references.
  Example:
  ```bash
  bash evaluation/run_evaluation.sh single-deploy
  ```

- `multi-deploy`
  Runs [multi_task_inference_high_level_cross_agent_deployment.py](/root/bayes-tmp/bowang/ReMAP/evaluation/multi_task_inference_high_level_cross_agent_deployment.py).
  This script currently uses its built-in defaults and config references.
  Example:
  ```bash
  bash evaluation/run_evaluation.sh multi-deploy
  ```

## Composite targets

- `remap-all`
  Sequentially runs:
  1. single-task ReMAP deployment evaluation
  2. multi-task ReMAP deployment evaluation

- `all`
  Sequentially runs:
  1. single-task ReMAP deployment evaluation
  2. multi-task ReMAP deployment evaluation
  3. unified baseline tracking evaluation

For composite targets, the launcher intentionally uses each stage's default arguments. If you need custom paths or flags for the baseline evaluation, run `baseline-tracking` individually.

## Typical outputs

- Single-task deployment evaluation:
  Usually under `ReMAP/output/single_task_inference_results/` or the save path configured inside the script.

- Multi-task deployment evaluation:
  Usually under the output paths configured by `configs/transfer_config.py` and the deployment script.

- Baseline tracking evaluation:
  Usually under `ReMAP/final_results/baselines_tracking_evaluation/` unless overridden by `--out-dir`.

## Notes

- `baseline_tracking_evalaution.py` keeps its existing filename, including the current spelling, so the launcher uses that exact file.
- The single-task and multi-task deployment evaluation scripts are currently research-style entrypoints with built-in defaults rather than fully parameterized CLIs. The launcher preserves that behavior and simply makes them one-command runnable.
