#!/usr/bin/env bash
set -euo pipefail

EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REMAP_ROOT="$(cd "$EVAL_DIR/.." && pwd)"
cd "$REMAP_ROOT"

print_usage() {
  cat <<'USAGE'
Usage:
  bash evaluation/run_evaluation.sh <target> [args...]

Targets:
  baseline-tracking  Run unified baseline tracking evaluation.
  single-deploy      Run single-task ReMAP deployment evaluation.
  multi-deploy       Run multi-task ReMAP deployment evaluation.
  remap-all          Run single-deploy and multi-deploy sequentially.
  all                Run remap-all and then baseline-tracking.

Examples:
  bash evaluation/run_evaluation.sh baseline-tracking
  bash evaluation/run_evaluation.sh baseline-tracking --gpu 0 --num-trajs 5
  bash evaluation/run_evaluation.sh single-deploy
  bash evaluation/run_evaluation.sh multi-deploy
USAGE
}

run_python() {
  local script_path="$1"
  shift
  local extra_pythonpath="$REMAP_ROOT:$REMAP_ROOT/third_party/Meta_RL:$REMAP_ROOT/third_party/Meta_RL/submodules/rlkit:$REMAP_ROOT/third_party/Meta_RL/submodules/meta-environments-main:$REMAP_ROOT/third_party/Meta_RL/submodules/MRL-analysis-tools-main"
  PYTHONPATH="$extra_pythonpath${PYTHONPATH:+:$PYTHONPATH}" python "$script_path" "$@"
}

print_stage() {
  local title="$1"
  printf '\n%s\n' "================================================================"
  printf '%s\n' "$title"
  printf '%s\n' "================================================================"
}

if [[ $# -lt 1 ]]; then
  print_usage
  exit 1
fi

target="$1"
shift || true

case "$target" in
  baseline-tracking)
    print_stage "Running unified baseline tracking evaluation"
    run_python "$REMAP_ROOT/evaluation/baseline_tracking_evalaution.py" "$@"
    ;;
  single-deploy)
    if [[ $# -gt 0 ]]; then
      echo "single-deploy currently uses the script's built-in defaults; ignoring extra args: $*"
    fi
    print_stage "Running single-task ReMAP deployment evaluation"
    run_python "$REMAP_ROOT/evaluation/single_task_inference_high_level_cross_agent_deployment.py"
    ;;
  multi-deploy)
    if [[ $# -gt 0 ]]; then
      echo "multi-deploy currently uses the script's built-in defaults; ignoring extra args: $*"
    fi
    print_stage "Running multi-task ReMAP deployment evaluation"
    run_python "$REMAP_ROOT/evaluation/multi_task_inference_high_level_cross_agent_deployment.py"
    ;;
  remap-all)
    if [[ $# -gt 0 ]]; then
      echo "remap-all uses the default arguments of each stage. For custom baseline flags, run baseline-tracking individually."
    fi
    print_stage "Running single-task ReMAP deployment evaluation"
    run_python "$REMAP_ROOT/evaluation/single_task_inference_high_level_cross_agent_deployment.py"
    print_stage "Running multi-task ReMAP deployment evaluation"
    run_python "$REMAP_ROOT/evaluation/multi_task_inference_high_level_cross_agent_deployment.py"
    ;;
  all)
    if [[ $# -gt 0 ]]; then
      echo "all uses default arguments for the composite pipeline. For custom baseline flags, run baseline-tracking individually."
    fi
    print_stage "Running single-task ReMAP deployment evaluation"
    run_python "$REMAP_ROOT/evaluation/single_task_inference_high_level_cross_agent_deployment.py"
    print_stage "Running multi-task ReMAP deployment evaluation"
    run_python "$REMAP_ROOT/evaluation/multi_task_inference_high_level_cross_agent_deployment.py"
    print_stage "Running unified baseline tracking evaluation"
    run_python "$REMAP_ROOT/evaluation/baseline_tracking_evalaution.py"
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
