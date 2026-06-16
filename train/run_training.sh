#!/usr/bin/env bash
set -euo pipefail

TRAIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REMAP_ROOT="$(cd "$TRAIN_DIR/.." && pwd)"
cd "$REMAP_ROOT"

print_usage() {
  cat <<'USAGE'
Usage:
  bash train/run_training.sh <target> [args...]

Targets:
  low-level         Run low-level policy training.
  high-single       Run single-task high-level training.
  high-multi        Run multi-task high-level training.
  baseline-cemrl    Run CEMRL baseline training.
  baseline-melts    Run MELTS baseline training.
  baseline-pearl    Run PEARL baseline training.
  baseline-rl2      Run RL2 baseline training.
  baseline-all      Run all four baselines sequentially with their defaults.
  remap-all         Run low-level + single-task high-level + multi-task high-level.
  all               Run remap-all and baseline-all.

Examples:
  bash train/run_training.sh low-level --env cheetah
  bash train/run_training.sh high-multi --gpu 0 --ti_option dpmm --name exp1
  bash train/run_training.sh baseline-cemrl --gpu 0
  bash train/run_training.sh baseline-all
USAGE
}

run_python() {
  local script_path="$1"
  shift
  local extra_pythonpath="$REMAP_ROOT:$REMAP_ROOT/third_party/Meta_RL:$REMAP_ROOT/third_party/Meta_RL/submodules/rlkit:$REMAP_ROOT/third_party/Meta_RL/submodules/meta-environments-main:$REMAP_ROOT/third_party/Meta_RL/submodules/MRL-analysis-tools-main"
  PYTHONPATH="$extra_pythonpath${PYTHONPATH:+:$PYTHONPATH}" python "$script_path" "$@"
}

if [[ $# -lt 1 ]]; then
  print_usage
  exit 1
fi

target="$1"
shift || true

case "$target" in
  low-level)
    run_python "$REMAP_ROOT/train/train_low_level_policy.py" "$@"
    ;;
  high-single)
    if [[ $# -gt 0 ]]; then
      echo "high-single does not expose CLI args in the current code; ignoring extra args: $*"
    fi
    run_python "$REMAP_ROOT/train/train_single_task_inference_high_level_policy.py"
    ;;
  high-multi)
    run_python "$REMAP_ROOT/train/train_multi_task_inference_high_level_policy.py" "$@"
    ;;
  baseline-cemrl)
    run_python "$REMAP_ROOT/train/baselines_training/run_cemrl_cheetah_baseline.py" "$@"
    ;;
  baseline-melts)
    run_python "$REMAP_ROOT/train/baselines_training/run_melts_cheetah_baseline.py" "$@"
    ;;
  baseline-pearl)
    run_python "$REMAP_ROOT/train/baselines_training/run_pearl_cheetah_baseline.py" "$@"
    ;;
  baseline-rl2)
    run_python "$REMAP_ROOT/train/baselines_training/run_rl2_cheetah_baseline.py" "$@"
    ;;
  baseline-all)
    if [[ $# -gt 0 ]]; then
      echo "baseline-all uses each baseline's default arguments. For custom flags, run each baseline target individually."
    fi
    run_python "$REMAP_ROOT/train/baselines_training/run_cemrl_cheetah_baseline.py"
    run_python "$REMAP_ROOT/train/baselines_training/run_melts_cheetah_baseline.py"
    run_python "$REMAP_ROOT/train/baselines_training/run_pearl_cheetah_baseline.py"
    run_python "$REMAP_ROOT/train/baselines_training/run_rl2_cheetah_baseline.py"
    ;;
  remap-all)
    if [[ $# -gt 0 ]]; then
      echo "remap-all uses the default arguments of each stage. For custom flags, run low-level/high-single/high-multi individually."
    fi
    run_python "$REMAP_ROOT/train/train_low_level_policy.py" --env cheetah
    run_python "$REMAP_ROOT/train/train_low_level_policy.py" --env hopper
    run_python "$REMAP_ROOT/train/train_low_level_policy.py" --env walker2d
    run_python "$REMAP_ROOT/train/train_low_level_policy.py" --env ant
    run_python "$REMAP_ROOT/train/train_single_task_inference_high_level_policy.py"
    run_python "$REMAP_ROOT/train/train_multi_task_inference_high_level_policy.py"
    ;;
  all)
    if [[ $# -gt 0 ]]; then
      echo "all uses default arguments for the composite pipeline. For custom flags, run each target individually."
    fi
    run_python "$REMAP_ROOT/train/train_low_level_policy.py" --env cheetah
    run_python "$REMAP_ROOT/train/train_low_level_policy.py" --env hopper
    run_python "$REMAP_ROOT/train/train_low_level_policy.py" --env walker2d
    run_python "$REMAP_ROOT/train/train_low_level_policy.py" --env ant
    run_python "$REMAP_ROOT/train/train_single_task_inference_high_level_policy.py"
    run_python "$REMAP_ROOT/train/train_multi_task_inference_high_level_policy.py"
    run_python "$REMAP_ROOT/train/baselines_training/run_cemrl_cheetah_baseline.py"
    run_python "$REMAP_ROOT/train/baselines_training/run_melts_cheetah_baseline.py"
    run_python "$REMAP_ROOT/train/baselines_training/run_pearl_cheetah_baseline.py"
    run_python "$REMAP_ROOT/train/baselines_training/run_rl2_cheetah_baseline.py"
    ;;
  -h|--help|help)
    print_usage
    ;;
  *)
    echo "Unknown target: $target"
    print_usage
    exit 1
    ;;
esac
