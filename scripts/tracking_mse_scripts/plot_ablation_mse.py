"""
Plot goal tracking MSE and velocity tracking MSE over within-episode timesteps
for the ablation study: Ours (DPMM) vs GMM vs Stick-Breaking vs Single Gaussian.

For each within-episode step t = 0..STEPS_PER_EPISODE-1:
  - Goal tracking MSE    : (x_before  - true_goal)^2  — goal episodes
  - Velocity tracking MSE: (vx_before - true_goal)^2  — velocity episodes

Corrected task ID mapping (true_task_idx in CSV):
  0 = goal_forward,     1 = goal_backward
  2 = velocity_forward, 3 = velocity_backward
"""

import csv
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ── config ───────────────────────────────────────────────────────────────
ABLATION_DIR = (
    "/home/ubuntu/yuanmeng/bo/MRL-Inference-Reutilization/output/"
    "toy1d-multi-task/2026_01_06_20_48_56_default_dpmm_seed0_regular_loss_"
    "true_time_steps48_cheetah/DECODER_EVAL/mse_ablation_study_comparison"
)

# {display_name: csv_filename}
ABLATION_FILES = {
    "Ours (DPMM)":     "dpmm_subgoals_ep11.csv",
    "GMM":             "gmm_subgoals_ep11.csv",
    "Stick-Breaking":  "stick_breaking_subgoals_ep11.csv",
    "Single Gaussian": "single_gaussian_subgoals_ep11.csv",
}

OUT_DIR = ABLATION_DIR

STEPS_PER_EPISODE = 100
CONVERGENCE_T     = 25
GOAL_TASK_IDS     = {0, 1}   # goal_forward, goal_backward
VELOCITY_TASK_IDS = {2, 3}   # velocity_forward, velocity_backward

METHOD_COLORS = {
    "Ours (DPMM)":     "#d62728",
    "GMM":             "#1f77b4",
    "Stick-Breaking":  "#ff7f0e",
    "Single Gaussian": "#2ca02c",
}
# ────────────────────────────────────────────────────────────────────────


def load_csv(path):
    """
    Parse a subgoals CSV file.

    Task ID mapping:
      0 = goal_forward,     1 = goal_backward    → use x_before  vs true_goal
      2 = velocity_forward, 3 = velocity_backward → use vx_before vs true_goal

    Returns
    -------
    goal_mse : np.ndarray, shape (n_goal_eps, STEPS_PER_EPISODE)
        (x_before - true_goal)^2 for goal-task episodes.
    vel_mse  : np.ndarray, shape (n_vel_eps, STEPS_PER_EPISODE)
        (vx_before - true_goal)^2 for velocity-task episodes.
    """
    rows = list(csv.DictReader(open(path)))
    ep_rows = {}
    for r in rows:
        ep = int(r["episode"])
        ep_rows.setdefault(ep, []).append(r)

    goal_bufs = []
    vel_bufs  = []

    for ep in sorted(ep_rows.keys()):
        episode_data = ep_rows[ep]
        task_idx  = int(episode_data[0]["true_task_idx"])
        true_goal = float(episode_data[0]["true_goal"])

        if task_idx in GOAL_TASK_IDS:
            buf = np.full(STEPS_PER_EPISODE, np.nan)
            goal_bufs.append(buf)
        elif task_idx in VELOCITY_TASK_IDS:
            buf = np.full(STEPS_PER_EPISODE, np.nan)
            vel_bufs.append(buf)
        else:
            buf = None

        for r in episode_data:
            t_global = int(r["t"])
            t_within = t_global - ep * STEPS_PER_EPISODE
            if buf is None or not (0 <= t_within < STEPS_PER_EPISODE):
                continue
            if task_idx in GOAL_TASK_IDS:
                buf[t_within] = (float(r["x_before"])  - true_goal) ** 2
            elif task_idx in VELOCITY_TASK_IDS:
                buf[t_within] = (float(r["vx_before"]) - true_goal) ** 2

    goal_mse = np.vstack(goal_bufs) if goal_bufs else np.full((1, STEPS_PER_EPISODE), np.nan)
    vel_mse  = np.vstack(vel_bufs)  if vel_bufs  else np.full((1, STEPS_PER_EPISODE), np.nan)
    return goal_mse, vel_mse


def compute_stats(mse_array):
    """Return (mean, std) per timestep, ignoring NaN."""
    return np.nanmean(mse_array, axis=0), np.nanstd(mse_array, axis=0)


def print_table(label, mean, std,
                steps=(0, 5, 10, 15, 20, 25, 30, 40, 50, 75, 99)):
    print(f"\n{label}")
    print(f"  {'t':>3}  {'mean MSE':>10}  {'std':>8}")
    print("  " + "-" * 26)
    for t in steps:
        if t < len(mean):
            print(f"  {t:3d}  {mean[t]:10.5f}  {std[t]:8.5f}")
    pre  = np.mean(mean[:CONVERGENCE_T])
    post = np.mean(mean[CONVERGENCE_T:])
    print(f"\n  Avg t=0..{CONVERGENCE_T-1}  : {pre:.5f}")
    print(f"  Avg t={CONVERGENCE_T}..{len(mean)-1} : {post:.5f}")
    if pre > 0:
        print(f"  Reduction          : {(1 - post / pre) * 100:.1f}%")


def save_combined_csv(methods_stats, out_path):
    """
    Save per-timestep MSE mean and std for all methods into one CSV.

    Columns: timestep, <method>_mean_mse, <method>_std_mse, ...
    """
    T = next(iter(methods_stats.values()))[0].shape[0]
    headers = ["timestep"]
    for name in methods_stats:
        headers += [f"{name}_mean_mse", f"{name}_std_mse"]

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for t in range(T):
            row = [t]
            for name, (mean, std) in methods_stats.items():
                row += [mean[t], std[t]]
            writer.writerow(row)
    print(f"Saved → {out_path}")


def plot_comparison(methods_stats, ylabel, title, out_path):
    """Multi-method MSE comparison figure with shaded ±1 std bands."""
    T     = next(iter(methods_stats.values()))[0].shape[0]
    steps = np.arange(T)

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    for name, (mean, std) in methods_stats.items():
        col = METHOD_COLORS.get(name, "black")
        ax.plot(steps, mean, color=col, linewidth=2, label=name)
        ax.fill_between(
            steps,
            np.maximum(mean - std, 0),
            mean + std,
            color=col, alpha=0.15,
        )

    ax.axvline(x=CONVERGENCE_T, color="grey", linestyle="--",
               linewidth=1.5, label=f"t = {CONVERGENCE_T}")

    ax.set_xlabel("Time Step", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlim(0, T - 1)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.30)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.close()


def main():
    goal_stats = {}
    vel_stats  = {}

    for name, fname in ABLATION_FILES.items():
        path = os.path.join(ABLATION_DIR, fname)
        g_mse, v_mse = load_csv(path)
        goal_stats[name] = compute_stats(g_mse)
        vel_stats[name]  = compute_stats(v_mse)
        print(f"{name:<20}: {g_mse.shape[0]} goal eps, {v_mse.shape[0]} vel eps")

    # ── Print summary tables ──────────────────────────────────────────────
    for name, (mean, std) in goal_stats.items():
        print_table(f"Goal Tracking MSE — {name}  (x_before − true_goal)²",
                    mean, std)
    for name, (mean, std) in vel_stats.items():
        print_table(f"Velocity Tracking MSE — {name}  (vx_before − true_goal)²",
                    mean, std)

    # ── Save combined CSV source data ─────────────────────────────────────
    save_combined_csv(
        goal_stats,
        os.path.join(OUT_DIR, "ablation_mse_goal.csv"),
    )
    save_combined_csv(
        vel_stats,
        os.path.join(OUT_DIR, "ablation_mse_velocity.csv"),
    )

    # ── Figure 1: Goal Tracking MSE ───────────────────────────────────────
    plot_comparison(
        goal_stats,
        ylabel=r"Goal Tracking MSE  $(x_t - g)^2$",
        title="Goal Tracking MSE — Ablation Study",
        out_path=os.path.join(OUT_DIR, "ablation_goal_mse.pdf"),
    )

    # ── Figure 2: Velocity Tracking MSE ──────────────────────────────────
    plot_comparison(
        vel_stats,
        ylabel=r"Velocity Tracking MSE  $(v_t - v^*)^2$",
        title="Velocity Tracking MSE — Ablation Study",
        out_path=os.path.join(OUT_DIR, "ablation_velocity_mse.pdf"),
    )


if __name__ == "__main__":
    main()
