#!/usr/bin/env python3
import argparse
import copy
import csv
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
THIS_DIR = Path(__file__).resolve().parent
BUNDLED_PATHS = [
    REPO_ROOT,
    REPO_ROOT / "third_party" / "Meta_RL",
    REPO_ROOT / "third_party" / "Meta_RL" / "submodules" / "rlkit",
    REPO_ROOT / "third_party" / "Meta_RL" / "submodules" / "meta-environments-main",
    REPO_ROOT / "third_party" / "Meta_RL" / "submodules" / "MRL-analysis-tools-main",
    THIS_DIR,
]
for bundled_path in reversed(BUNDLED_PATHS):
    bundled_path = str(bundled_path)
    if bundled_path in sys.path:
        sys.path.remove(bundled_path)
    sys.path.insert(0, bundled_path)

import third_party.rlkit.torch.pytorch_util as ptu
from ReMAP.configs.pearl_default import default_config as pearl_default_config
from ReMAP.configs.rl2_default import default_config as rl2_default_config


COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
GOAL_TARGETS = [-9.02, -5.10, -3.14, 3.14, 5.10, 9.02]
VELOCITY_TARGETS = [-2.35, -1.75, -1.45, 1.45, 1.75, 2.35]

TASK_GROUPS = {
    "velocity_backward": VELOCITY_TARGETS[:3],
    "velocity_forward": VELOCITY_TARGETS[3:],
    "goal_backward": GOAL_TARGETS[:3],
    "goal_forward": GOAL_TARGETS[3:],
}


def load_tracking_helpers():
    """Load optional per-baseline tracking helpers when this evaluator runs."""
    try:
        from eval_cemrl_melts_tracking import (
            OBS_X_POS,
            OBS_X_VEL,
            custom_rollout,
            load_tigr_agent,
            plot_tracking as plot_tigr_tracking,
            set_goal_task as set_tigr_goal_task,
            set_velocity_task as set_tigr_velocity_task,
        )
        from eval_pearl_tracking import (
            deep_update_dict as pearl_deep_update_dict,
            evaluate_goal_tracking as evaluate_pearl_goal_tracking,
            evaluate_velocity_tracking as evaluate_pearl_velocity_tracking,
            load_agent as load_pearl_agent,
        )
        from eval_rl2_tracking import (
            deep_update_dict as rl2_deep_update_dict,
            evaluate_goal_tracking as evaluate_rl2_goal_tracking,
            evaluate_velocity_tracking as evaluate_rl2_velocity_tracking,
            load_agent as load_rl2_agent,
        )
    except ModuleNotFoundError as exc:
        missing = exc.name
        raise RuntimeError(
            "baseline-tracking requires the per-baseline evaluator modules "
            "`eval_cemrl_melts_tracking.py`, `eval_pearl_tracking.py`, and "
            "`eval_rl2_tracking.py` next to `baseline_tracking_evalaution.py`. "
            f"Missing module: {missing!r}."
        ) from exc

    globals().update(locals())


def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified tracking evaluation for CEMRL, MELTS, PEARL, and RL2."
    )
    parser.add_argument(
        "--cemrl-dir",
        type=str,
        default="output/cemrl_baseline_run1/cemrl_baseline_run1 copy/cheetah-multi-task/2026_05_12_11_29_09_cemrl_cheetah_true_gmm",
    )
    parser.add_argument(
        "--melts-dir",
        type=str,
        default="output/melts_baseline/cheetah-multi-task/2026_05_17_00_59_52_melts_cheetah_dpmm",
    )
    parser.add_argument(
        "--pearl-dir",
        type=str,
        default="output/pearl_baseline_run1/cheetah-multi-task/2026_05_12_16_29_35",
    )
    parser.add_argument(
        "--rl2-dir",
        type=str,
        default="output/RL2_baseline_run1",
    )
    parser.add_argument("--cemrl-config", type=str, default="ReMAP/configs/cemrl_cheetah_tigr_config.json")
    parser.add_argument("--melts-config", type=str, default="ReMAP/configs/melts_cheetah_config.json")
    parser.add_argument("--pearl-config", type=str, default="ReMAP/configs/pearl_cheetah_multi_config.json")
    parser.add_argument("--rl2-config", type=str, default="ReMAP/configs/rl2_cheetah_config.json")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num-trajs", type=int, default=3)
    parser.add_argument(
        "--out-dir",
        type=str,
        default="final_results/baselines_tracking_evaluation",
    )
    return parser.parse_args()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_json_config(default_config, config_path, updater):
    variant = copy.deepcopy(default_config)
    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            exp_params = json.load(f)
        variant = updater(exp_params, variant)
    return variant


def summarize_tigr_results(targets, all_trajs, signal_type):
    results = {}
    for target in targets:
        trajs = all_trajs[target]
        last = np.asarray(trajs[-1], dtype=np.float32)
        returns = [float("nan")] * len(trajs)
        item = {
            "target": float(target),
            "returns": returns,
            "adapted_return": float("nan"),
        }
        if signal_type == "velocity":
            item.update({
                "achieved_velocities": [[float(v) for v in traj] for traj in trajs],
                "mean_achieved_vel": float(np.mean(last)),
                "std_achieved_vel": float(np.std(last)),
                "tracking_error": float(np.mean(np.abs(last - target))),
            })
        else:
            final_pos = float(last[-1]) if len(last) else float("nan")
            item.update({
                "achieved_positions": [[float(v) for v in traj] for traj in trajs],
                "final_position": final_pos,
                "distance_to_goal": float(np.abs(final_pos - target)),
                "mean_position": float(np.mean(last)) if len(last) else float("nan"),
            })
        results[target] = item
    return results


def run_tigr_method(method_name, exp_dir, config_path, inference_option, out_dir, num_trajs):
    print(f"\n{'=' * 72}")
    print(f"[{method_name.upper()}] loading from {exp_dir}")
    print(f"{'=' * 72}")
    env, agent, info = load_tigr_agent(exp_dir, config_path, inference_option)
    if ptu.gpu_enabled():
        agent.to(ptu.device)

    goal_trajs = {}
    for target in GOAL_TARGETS:
        trajs = custom_rollout(
            env, agent, info, set_tigr_goal_task, target, OBS_X_POS, num_trajs
        )
        goal_trajs[target] = trajs
        print(f"  goal={target:>7.2f} adapted_mean={np.mean(trajs[-1]):>8.3f}")

    vel_trajs = {}
    for target in VELOCITY_TARGETS:
        trajs = custom_rollout(
            env, agent, info, set_tigr_velocity_task, target, OBS_X_VEL, num_trajs
        )
        vel_trajs[target] = trajs
        print(f"  vel={target:>7.2f} adapted_mean={np.mean(trajs[-1]):>8.3f}")

    method_out_dir = os.path.join(out_dir, method_name)
    ensure_dir(method_out_dir)
    plot_tigr_tracking(
        goal_trajs,
        GOAL_TARGETS,
        ylabel="X Position",
        title=f"{method_name.upper()} - Goal Position Tracking",
        save_path=os.path.join(method_out_dir, "goal_tracking.png"),
        target_label_prefix="Goal",
    )
    plot_tigr_tracking(
        vel_trajs,
        VELOCITY_TARGETS,
        ylabel="X Velocity",
        title=f"{method_name.upper()} - Velocity Tracking",
        save_path=os.path.join(method_out_dir, "velocity_tracking.png"),
        target_label_prefix="Target Vel",
    )

    env.close()
    return {
        "goal_tracking": summarize_tigr_results(GOAL_TARGETS, goal_trajs, "goal"),
        "velocity_tracking": summarize_tigr_results(VELOCITY_TARGETS, vel_trajs, "velocity"),
        "raw_goal_trajs": {str(k): v for k, v in goal_trajs.items()},
        "raw_velocity_trajs": {str(k): v for k, v in vel_trajs.items()},
    }


def run_pearl_method(exp_dir, config_path, out_dir, num_trajs):
    print(f"\n{'=' * 72}")
    print(f"[PEARL] loading from {exp_dir}")
    print(f"{'=' * 72}")
    variant = load_json_config(pearl_default_config, config_path, pearl_deep_update_dict)
    weights_dir = os.path.join(exp_dir, "weights")
    env, agent = load_pearl_agent(variant, weights_dir)
    if ptu.gpu_enabled():
        agent.to("cuda")

    vel_results = evaluate_pearl_velocity_tracking(env, agent, variant, VELOCITY_TARGETS, num_trajs=num_trajs)
    goal_results = evaluate_pearl_goal_tracking(env, agent, variant, GOAL_TARGETS, num_trajs=num_trajs)

    method_out_dir = os.path.join(out_dir, "pearl")
    ensure_dir(method_out_dir)
    plot_tracking_from_metric_results(
        goal_results,
        GOAL_TARGETS,
        value_key="achieved_positions",
        ylabel="X Position",
        title="PEARL - Goal Position Tracking (cheetah-multi-task)",
        save_path=os.path.join(method_out_dir, "goal_tracking.png"),
        target_label_prefix="Goal",
    )
    plot_tracking_from_metric_results(
        vel_results,
        VELOCITY_TARGETS,
        value_key="achieved_velocities",
        ylabel="X Velocity",
        title="PEARL - Velocity Tracking (cheetah-multi-task)",
        save_path=os.path.join(method_out_dir, "velocity_tracking.png"),
        target_label_prefix="Target Vel",
    )

    env.close()
    return {
        "goal_tracking": goal_results,
        "velocity_tracking": vel_results,
    }


def run_rl2_method(exp_dir, config_path, out_dir, num_trajs):
    print(f"\n{'=' * 72}")
    print(f"[RL2] loading from {exp_dir}")
    print(f"{'=' * 72}")
    variant = load_json_config(rl2_default_config, config_path, rl2_deep_update_dict)
    weights_dir = os.path.join(exp_dir, "weights")
    env, agent = load_rl2_agent(variant, weights_dir)
    if ptu.gpu_enabled():
        agent.to("cuda")

    vel_results = evaluate_rl2_velocity_tracking(env, agent, variant, VELOCITY_TARGETS, num_trajs=num_trajs)
    goal_results = evaluate_rl2_goal_tracking(env, agent, variant, GOAL_TARGETS, num_trajs=num_trajs)

    method_out_dir = os.path.join(out_dir, "rl2")
    ensure_dir(method_out_dir)
    plot_tracking_from_metric_results(
        goal_results,
        GOAL_TARGETS,
        value_key="achieved_positions",
        ylabel="X Position",
        title="RL2 - Goal Position Tracking (cheetah-multi-task)",
        save_path=os.path.join(method_out_dir, "goal_tracking.png"),
        target_label_prefix="Goal",
    )
    plot_tracking_from_metric_results(
        vel_results,
        VELOCITY_TARGETS,
        value_key="achieved_velocities",
        ylabel="X Velocity",
        title="RL2 - Velocity Tracking (cheetah-multi-task)",
        save_path=os.path.join(method_out_dir, "velocity_tracking.png"),
        target_label_prefix="Target Vel",
    )

    env.close()
    return {
        "goal_tracking": goal_results,
        "velocity_tracking": vel_results,
    }


def safe_nanmean(values):
    valid = [v for v in values if not np.isnan(v)]
    return float(np.mean(valid)) if valid else float("nan")


def build_group_summary(method_name, method_results):
    rows = []
    vel_results = method_results["velocity_tracking"]
    goal_results = method_results["goal_tracking"]

    for group_name, targets in TASK_GROUPS.items():
        if group_name.startswith("velocity"):
            errors = [vel_results[target]["tracking_error"] for target in targets]
            returns = [vel_results[target]["adapted_return"] for target in targets]
        else:
            errors = [goal_results[target]["distance_to_goal"] for target in targets]
            returns = [goal_results[target]["adapted_return"] for target in targets]

        rows.append({
            "method": method_name,
            "task_group": group_name,
            "targets": [float(t) for t in targets],
            "mean_error": safe_nanmean(errors),
            "mean_adapted_return": safe_nanmean(returns),
        })
    return rows


def save_summary_csv(summary_rows, csv_path):
    fieldnames = ["method", "task_group", "targets", "mean_error", "mean_adapted_return"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            row = dict(row)
            row["targets"] = ",".join(str(v) for v in row["targets"])
            writer.writerow(row)


def plot_group_comparison(summary_rows, save_path):
    methods = ["cemrl", "melts", "pearl", "rl2"]
    groups = list(TASK_GROUPS.keys())
    method_colors = {
        "cemrl": "#1f77b4",
        "melts": "#ff7f0e",
        "pearl": "#2ca02c",
        "rl2": "#d62728",
    }
    fig, axes = plt.subplots(2, 1, figsize=(12, 9))
    width = 0.18
    x = np.arange(len(groups))

    for method_idx, method in enumerate(methods):
        method_rows = {row["task_group"]: row for row in summary_rows if row["method"] == method}
        errors = [method_rows[group]["mean_error"] for group in groups]
        returns = [method_rows[group]["mean_adapted_return"] for group in groups]
        offset = (method_idx - 1.5) * width
        axes[0].bar(x + offset, errors, width=width, label=method.upper(), color=method_colors[method], alpha=0.85)
        axes[1].bar(x + offset, returns, width=width, label=method.upper(), color=method_colors[method], alpha=0.85)

    axes[0].set_title("Mean Tracking Error by Task Group", fontweight="bold")
    axes[0].set_ylabel("Error")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(groups, rotation=15)
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].legend()

    axes[1].set_title("Mean Adapted Return by Task Group", fontweight="bold")
    axes[1].set_ylabel("Return")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(groups, rotation=15)
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_tracking_from_metric_results(results, targets, value_key, ylabel, title, save_path, target_label_prefix):
    max_t = max(len(results[target][value_key][-1]) for target in targets)
    timesteps = np.arange(max_t)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, target in enumerate(targets):
        traj = results[target][value_key][-1]
        color = COLORS[i % len(COLORS)]
        ax.plot(
            timesteps[:len(traj)],
            traj,
            color=color,
            linewidth=1.8,
            label=f"{target_label_prefix} = {target:.2f}",
        )
        ax.axhline(y=target, color=color, linestyle="--", linewidth=1.0, alpha=0.55)

    ax.set_xlabel("Time Step", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="best", ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_json_ready(obj):
    if isinstance(obj, dict):
        return {str(k): make_json_ready(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_ready(v) for v in obj]
    if isinstance(obj, tuple):
        return [make_json_ready(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def main():
    args = parse_args()
    load_tracking_helpers()
    ensure_dir(args.out_dir)
    os.environ.setdefault("DEBUG", "0")
    os.environ.setdefault("PLOT", "0")
    ptu.set_gpu_mode(args.gpu >= 0, args.gpu)

    all_results = {
        "task_groups": TASK_GROUPS,
        "goal_targets": GOAL_TARGETS,
        "velocity_targets": VELOCITY_TARGETS,
        "num_trajs": args.num_trajs,
        "methods": {},
    }

    all_results["methods"]["cemrl"] = run_tigr_method(
        "cemrl", args.cemrl_dir, args.cemrl_config, "true_gmm", args.out_dir, args.num_trajs
    )
    all_results["methods"]["melts"] = run_tigr_method(
        "melts", args.melts_dir, args.melts_config, "dpmm", args.out_dir, args.num_trajs
    )
    all_results["methods"]["pearl"] = run_pearl_method(
        args.pearl_dir, args.pearl_config, args.out_dir, args.num_trajs
    )
    all_results["methods"]["rl2"] = run_rl2_method(
        args.rl2_dir, args.rl2_config, args.out_dir, args.num_trajs
    )

    summary_rows = []
    for method_name, method_results in all_results["methods"].items():
        summary_rows.extend(build_group_summary(method_name, method_results))
    all_results["group_summary"] = summary_rows

    json_path = os.path.join(args.out_dir, "baseline_tracking_results.json")
    with open(json_path, "w") as f:
        json.dump(make_json_ready(all_results), f, indent=2)

    csv_path = os.path.join(args.out_dir, "baseline_tracking_summary.csv")
    save_summary_csv(summary_rows, csv_path)

    plot_path = os.path.join(args.out_dir, "baseline_tracking_comparison.png")
    plot_group_comparison(summary_rows, plot_path)

    print(f"\nSaved unified JSON to: {json_path}")
    print(f"Saved summary CSV to:  {csv_path}")
    print(f"Saved comparison plot: {plot_path}")


if __name__ == "__main__":
    main()
