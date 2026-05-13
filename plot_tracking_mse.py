"""
Plot tracking MSE over within-episode timesteps.

For each within-episode step t_within = 0..99, compute per-episode:
  - Goal inference MSE  : (subgoal_value - true_goal)^2  — all 12 episodes
  - Velocity tracking MSE: (vx_after - target_v)^2       — velocity episodes only (ep0-5)

Then average across the relevant episodes.
No cumulative/rolling averaging — purely instantaneous MSE at each step.

Task types (from task_dict.json):
  0: velocity_forward   1: velocity_backward
  2: goal_forward       3: goal_backward
"""

import csv
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ── config ──────────────────────────────────────────────────────────────
DATA_FILE = (
    "/home/ubuntu/yuanmeng/bo/MRL-Inference-Reutilization/output/"
    "toy1d-multi-task/2026_01_06_20_48_56_default_dpmm_seed0_regular_loss_"
    "true_time_steps48_cheetah/DECODER_EVAL/logs/subgoals_ep11.csv"
)
OUT_DIR  = os.path.dirname(DATA_FILE)
OUT_INF  = os.path.join(OUT_DIR, "tracking_mse_plot.pdf")
OUT_VEL  = os.path.join(OUT_DIR, "velocity_mse_plot.pdf")

STEPS_PER_EPISODE  = 100
CONVERGENCE_T      = 25
VELOCITY_TASK_IDS  = {0, 1}   # velocity_forward, velocity_backward
# ────────────────────────────────────────────────────────────────────────


def load_data(path):
    """
    Returns:
      inf_mse[ep_idx, t]  — all episodes, goal inference MSE
      vel_mse[vel_ep_idx, t] — velocity episodes only, velocity tracking MSE
    """
    rows = list(csv.DictReader(open(path)))
    ep_rows = {}
    for r in rows:
        ep = int(r["episode"])
        ep_rows.setdefault(ep, []).append(r)

    n_ep = len(ep_rows)
    inf_mse = np.full((n_ep, STEPS_PER_EPISODE), np.nan)
    vel_rows = []   # collect velocity-task episode data

    for i, ep in enumerate(sorted(ep_rows.keys())):
        episode_data = ep_rows[ep]
        task_idx = int(episode_data[0]["true_task_idx"])

        # signed target velocity for velocity tasks
        if task_idx in VELOCITY_TASK_IDS:
            true_goal = float(episode_data[0]["true_goal"])
            spec      = float(episode_data[0]["spec_of_episode"])
            target_v  = (1 if true_goal > 0 else -1) * spec
            vel_buf   = np.full(STEPS_PER_EPISODE, np.nan)
            vel_rows.append(vel_buf)

        for r in episode_data:
            t_global = int(r["t"])
            t_within = t_global - ep * STEPS_PER_EPISODE
            if 0 <= t_within < STEPS_PER_EPISODE:
                goal = float(r["true_goal"])
                sg   = float(r["subgoal_value"])
                inf_mse[i, t_within] = (sg - goal) ** 2

                if task_idx in VELOCITY_TASK_IDS:
                    vx = float(r["vx_after"])
                    vel_buf[t_within] = (vx - target_v) ** 2

    vel_mse = np.vstack(vel_rows)  # shape (n_vel_episodes, STEPS_PER_EPISODE)
    return inf_mse, vel_mse


def print_table(label, mean, std, steps=(0, 5, 10, 15, 20, 25, 30, 40, 50, 75, 99)):
    print(f"\n{label}")
    print(f"  {'t':>3}  {'mean MSE':>10}  {'std':>8}")
    print("  " + "-" * 26)
    for t in steps:
        print(f"  {t:3d}  {mean[t]:10.5f}  {std[t]:8.5f}")
    pre  = np.mean(mean[:CONVERGENCE_T])
    post = np.mean(mean[CONVERGENCE_T:])
    print(f"\n  Avg t=0..{CONVERGENCE_T-1}  : {pre:.5f}")
    print(f"  Avg t={CONVERGENCE_T}..99 : {post:.5f}")
    if pre > 0:
        print(f"  Reduction          : {(1 - post / pre) * 100:.1f}%")


def _single_mse_figure(mean, std, ylabel, title, n_ep, col, out_path):
    """Save a single-panel MSE figure."""
    steps = np.arange(STEPS_PER_EPISODE)

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(
        f"{title}\n(mean ± std over {n_ep} evaluation episodes)",
        fontsize=13, fontweight="bold"
    )

    ax.plot(steps, mean, color=col, linewidth=2, label="Mean MSE")
    ax.fill_between(
        steps,
        np.maximum(mean - std, 0),
        mean + std,
        color=col, alpha=0.20, label="±1 std"
    )

    # convergence marker
    ax.axvline(x=CONVERGENCE_T, color="grey", linestyle="--",
               linewidth=1.5, label=f"t = {CONVERGENCE_T}")

    # annotate value at t=25
    v = mean[CONVERGENCE_T]
    ax.annotate(
        f"MSE = {v:.4f} at t = {CONVERGENCE_T}",
        xy=(CONVERGENCE_T, v),
        xytext=(CONVERGENCE_T + 20, v + mean[0] * 0.12),
        arrowprops=dict(arrowstyle="->", color="dimgrey", lw=1.2),
        fontsize=9, color="dimgrey"
    )

    # summary stats box
    post_mean = np.mean(mean[CONVERGENCE_T:])
    ax.text(
        0.97, 0.96,
        f"Avg MSE  t ≥ {CONVERGENCE_T}: {post_mean:.5f}",
        transform=ax.transAxes, fontsize=9, va="top", ha="right",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="lightyellow", alpha=0.85)
    )

    ax.set_xlabel("Time Step", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlim(0, 200)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.30)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.close()


def save_mse_csv(mean, std, out_path):
    """Save per-timestep MSE mean and std to a CSV file."""
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestep", "mean_mse", "std_mse"])
        for t in range(len(mean)):
            writer.writerow([t, mean[t], std[t]])
    print(f"Saved → {out_path}")


def main():
    inf_mse, vel_mse = load_data(DATA_FILE)

    n_ep_inf = inf_mse.shape[0]
    n_ep_vel = vel_mse.shape[0]
    print(f"Loaded {n_ep_inf} episodes total, {n_ep_vel} velocity-task episodes")

    inf_mean = np.nanmean(inf_mse, axis=0)
    inf_std  = np.nanstd (inf_mse, axis=0)
    vel_mean = np.nanmean(vel_mse, axis=0)
    vel_std  = np.nanstd (vel_mse, axis=0)

    print_table("Goal Inference MSE  (subgoal_value − true_goal)²", inf_mean, inf_std)
    print_table("Velocity Tracking MSE  (vx_after − target_v)²",    vel_mean, vel_std)

    # ── Save MSE source data as CSV ──────────────────────────────────────
    save_mse_csv(inf_mean, inf_std,
                 os.path.join(OUT_DIR, "tracking_mse_inference.csv"))
    save_mse_csv(vel_mean, vel_std,
                 os.path.join(OUT_DIR, "tracking_mse_velocity.csv"))

    # ── Figure 1: Goal Inference MSE ────────────────────────────────────
    _single_mse_figure(
        inf_mean, inf_std,
        ylabel=r"Goal Inference MSE  $(\hat{g}_t - g)^2$",
        title="Goal Inference MSE per Time Step",
        n_ep=n_ep_inf,
        col="#d62728",
        out_path=OUT_INF,
    )

    # ── Figure 2: Cheetah Velocity Tracking MSE ─────────────────────────
    _single_mse_figure(
        vel_mean, vel_std,
        ylabel=r"Velocity Tracking MSE  $(v_t - v^*)^2$",
        title="Cheetah Velocity Tracking MSE per Time Step",
        n_ep=n_ep_vel,
        col="#1f77b4",
        out_path=OUT_VEL,
    )


if __name__ == "__main__":
    main()
