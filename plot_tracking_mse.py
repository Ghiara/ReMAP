"""
Plot tracking MSE per time step: Ours vs PEARL vs RL2.

Task mapping (confirmed from CSV data):
  task_idx 0/1 -> goal-reaching  (x_after converges to true_goal)
  task_idx 2/3 -> velocity       (vx_after tracks true_goal directly)
"""

import csv
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = (
    "/home/ubuntu/yuanmeng/bo/MRL-Inference-Reutilization/output/"
    "toy1d-multi-task/2026_01_06_20_48_56_default_dpmm_seed0_regular_loss_"
    "true_time_steps48_cheetah/DECODER_EVAL/logs/subgoals_ep11.csv"
)
BASELINE_JSON = (
    "/home/ubuntu/yuanmeng/bo/MRL-Inference-Reutilization/"
    "final_results/pearl_rl2_tracking/tracking_data.json"
)
OUT_DIR  = os.path.dirname(BASE)
OUT_GOAL = os.path.join(OUT_DIR, "goal_tracking_mse_comparison.pdf")
OUT_VEL  = os.path.join(OUT_DIR, "velocity_tracking_mse_comparison.pdf")

STEPS = 100
GOAL_TASK_IDS     = {0, 1}   # x_after -> true_goal
VELOCITY_TASK_IDS = {2, 3}   # vx_after -> true_goal

C_OURS  = "#2ca02c"
C_PEARL = "#d62728"
C_RL2   = "#1f77b4"


def load_ours(path):
    rows = list(csv.DictReader(open(path)))
    ep_rows = {}
    for r in rows:
        ep_rows.setdefault(int(r["episode"]), []).append(r)

    goal_bufs, vel_bufs = [], []
    for ep in sorted(ep_rows.keys()):
        episode  = ep_rows[ep]
        task_idx = int(episode[0]["true_task_idx"])
        true_goal = float(episode[0]["true_goal"])

        buf = np.full(STEPS, np.nan)
        for r in episode:
            t_w = int(r["t"]) - ep * STEPS
            if 0 <= t_w < STEPS:
                if task_idx in GOAL_TASK_IDS:
                    buf[t_w] = (float(r["x_after"]) - true_goal) ** 2
                elif task_idx in VELOCITY_TASK_IDS:
                    buf[t_w] = (float(r["vx_after"]) - true_goal) ** 2

        if task_idx in GOAL_TASK_IDS:
            goal_bufs.append(buf)
        elif task_idx in VELOCITY_TASK_IDS:
            vel_bufs.append(buf)

    return np.vstack(goal_bufs), np.vstack(vel_bufs)


def load_baseline(path, method):
    d = json.load(open(path))
    data = d[method]
    goal_bufs, vel_bufs = [], []
    for target_str, trajs in data["goal"].items():
        target = float(target_str)
        for traj in trajs:
            goal_bufs.append([(x - target) ** 2 for x in traj])
    for target_str, trajs in data["velocity"].items():
        target_v = float(target_str)
        for traj in trajs:
            vel_bufs.append([(v - target_v) ** 2 for v in traj])
    return np.array(goal_bufs), np.array(vel_bufs)


def _add_curve(ax, mse_mat, label, color):
    T     = mse_mat.shape[1]
    steps = np.arange(T)
    mean  = np.nanmean(mse_mat, axis=0)
    std   = np.nanstd(mse_mat, axis=0)
    ax.plot(steps, mean, color=color, linewidth=2, label=label)
    ax.fill_between(steps, np.maximum(mean - std, 0), mean + std,
                    color=color, alpha=0.15)


def make_figure(ours_mse, pearl_mse, rl2_mse, ylabel, title, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    _add_curve(ax, ours_mse,  "Ours",  C_OURS)
    _add_curve(ax, pearl_mse, "PEARL", C_PEARL)
    _add_curve(ax, rl2_mse,   "RL2",   C_RL2)

    ax.set_xlabel("Time Step", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlim(0, 100)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.30)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved -> {out_path}")
    plt.close()


def main():
    ours_goal, ours_vel   = load_ours(BASE)
    pearl_goal, pearl_vel = load_baseline(BASELINE_JSON, "pearl")
    rl2_goal,   rl2_vel   = load_baseline(BASELINE_JSON, "rl2")

    print(f"Ours  : goal {ours_goal.shape}  vel {ours_vel.shape}")
    print(f"PEARL : goal {pearl_goal.shape}  vel {pearl_vel.shape}")
    print(f"RL2   : goal {rl2_goal.shape}  vel {rl2_vel.shape}")

    make_figure(
        ours_goal, pearl_goal, rl2_goal,
        ylabel="Goal Tracking MSE  (x_t - g)^2",
        title="Goal Tracking MSE per Time Step\n(mean +/- std, all goal episodes)",
        out_path=OUT_GOAL,
    )
    make_figure(
        ours_vel, pearl_vel, rl2_vel,
        ylabel="Velocity Tracking MSE  (vx_t - v*)^2",
        title="Velocity Tracking MSE per Time Step\n(mean +/- std, all velocity episodes)",
        out_path=OUT_VEL,
    )


if __name__ == "__main__":
    main()
