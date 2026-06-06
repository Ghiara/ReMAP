"""
Plot goal tracking MSE and velocity tracking MSE over within-episode timesteps.

Compares Ours vs PEARL vs RL2 vs CEMRL vs MELTS.

For each within-episode step t = 0..STEPS_PER_EPISODE-1:
  - Goal tracking MSE    : (x_after - true_goal)^2     — goal episodes (task_idx 2,3)
  - Velocity tracking MSE: (vx_after - target_v)^2     — velocity episodes (task_idx 0,1)

Baseline data (PEARL, RL2, CEMRL, MELTS) stores raw trajectories as arrays of
shape (n_seeds, T) keyed by target value string.  MSE is computed relative to
each method's own target values.

Task types (from task_dict.json):
  0: velocity_forward   1: velocity_backward
  2: goal_forward       3: goal_backward
"""

import csv
import json
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
PEARL_RL2_JSON = (
    "/home/ubuntu/yuanmeng/bo/MRL-Inference-Reutilization/"
    "final_results/pearl_rl2_tracking/tracking_data.json"
)
CEMRL_MELTS_JSON = (
    "/home/ubuntu/yuanmeng/bo/MRL-Inference-Reutilization/"
    "final_results/cemrl_melts_tracking/tracking_data.json"
)

OUT_DIR = os.path.dirname(DATA_FILE)

STEPS_PER_EPISODE = 100   # ours: 100 steps; baselines truncated to this length
CONVERGENCE_T     = 25
GOAL_TASK_IDS     = {0, 1}   # goal_forward, goal_backward
VELOCITY_TASK_IDS = {2, 3}   # velocity_forward, velocity_backward

# colour palette — consistent across both figures
METHOD_COLORS = {
    "Ours":  "#d62728",
    "PEARL": "#1f77b4",
    "RL2":   "#ff7f0e",
    "CEMRL": "#2ca02c",
    "MELTS": "#9467bd",
}
# ────────────────────────────────────────────────────────────────────────


# ── data loaders ─────────────────────────────────────────────────────────

def load_ours(path):
    """
    Parse the subgoals CSV and return per-episode MSE matrices.

    Corrected task ID mapping (task_idx in CSV):
      0 = goal_forward,     1 = goal_backward
      2 = velocity_forward, 3 = velocity_backward

    Goal tracking MSE    : (x_before - true_goal)^2  — goal episodes (task_idx 0,1)
    Velocity tracking MSE: (vx_before - true_goal)^2 — velocity episodes (task_idx 2,3)
      For velocity tasks, true_goal IS the signed velocity target directly.

    Returns
    -------
    goal_mse : np.ndarray, shape (n_goal_eps, STEPS_PER_EPISODE)
    vel_mse  : np.ndarray, shape (n_vel_eps, STEPS_PER_EPISODE)
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
                x_pos = float(r["x_before"])
                buf[t_within] = (x_pos - true_goal) ** 2
            elif task_idx in VELOCITY_TASK_IDS:
                vx = float(r["vx_before"])
                buf[t_within] = (vx - true_goal) ** 2

    goal_mse = np.vstack(goal_bufs) if goal_bufs else np.full((1, STEPS_PER_EPISODE), np.nan)
    vel_mse  = np.vstack(vel_bufs)  if vel_bufs  else np.full((1, STEPS_PER_EPISODE), np.nan)
    return goal_mse, vel_mse


def load_baseline(json_path, method_key):
    """
    Load trajectory data for one baseline method from a JSON file.

    JSON structure expected:
      {
        "goal_targets":     [list of float],
        "velocity_targets": [list of float],
        "<method_key>": {
          "goal":     { "<target_str>": [[seed0_t0, seed0_t1, ...], [seed1_...], ...] },
          "velocity": { "<target_str>": [[...], ...] }
        }
      }

    Each trajectory array has shape (n_seeds, T_max); we take the first
    STEPS_PER_EPISODE columns.

    Returns
    -------
    goal_mse : np.ndarray, shape (n_targets * n_seeds, STEPS_PER_EPISODE)
    vel_mse  : np.ndarray, shape (n_targets * n_seeds, STEPS_PER_EPISODE)
    """
    with open(json_path) as f:
        d = json.load(f)

    data         = d[method_key]
    goal_targets = d["goal_targets"]
    vel_targets  = d["velocity_targets"]
    T = STEPS_PER_EPISODE

    goal_rows = []
    for target in goal_targets:
        traj = np.array(data["goal"][str(target)])   # (n_seeds, T_max)
        traj = traj[:, :T]
        for seed_traj in traj:
            goal_rows.append((seed_traj - target) ** 2)

    vel_rows = []
    for target in vel_targets:
        traj = np.array(data["velocity"][str(target)])  # (n_seeds, T_max)
        traj = traj[:, :T]
        for seed_traj in traj:
            vel_rows.append((seed_traj - target) ** 2)

    goal_mse = np.vstack(goal_rows)
    vel_mse  = np.vstack(vel_rows)
    return goal_mse, vel_mse


# ── stats helpers ────────────────────────────────────────────────────────

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


# ── output helpers ───────────────────────────────────────────────────────

def save_combined_csv(methods_stats, out_path):
    """
    Save per-timestep MSE mean and std for ALL methods into one CSV.

    methods_stats : OrderedDict {method_name: (mean_array, std_array)}
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
    """
    Multi-method MSE comparison figure with shaded ±1 std bands.

    methods_stats : dict {method_name: (mean_array, std_array)}
    """
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


# ── main ─────────────────────────────────────────────────────────────────

def main():
    # ── Load ours ────────────────────────────────────────────────────────
    ours_goal_mse, ours_vel_mse = load_ours(DATA_FILE)
    ours_goal_mean, ours_goal_std = compute_stats(ours_goal_mse)
    ours_vel_mean,  ours_vel_std  = compute_stats(ours_vel_mse)
    print(f"Ours  : {ours_goal_mse.shape[0]} goal eps, "
          f"{ours_vel_mse.shape[0]} velocity eps")

    # ── Load baselines ───────────────────────────────────────────────────
    baseline_cfg = [
        ("PEARL", PEARL_RL2_JSON,   "pearl"),
        ("RL2",   PEARL_RL2_JSON,   "rl2"),
        ("CEMRL", CEMRL_MELTS_JSON, "cemrl"),
        ("MELTS", CEMRL_MELTS_JSON, "melts"),
    ]
    baselines = {}
    for name, json_path, key in baseline_cfg:
        g_mse, v_mse = load_baseline(json_path, key)
        baselines[name] = {
            "goal": compute_stats(g_mse),
            "vel":  compute_stats(v_mse),
        }
        print(f"{name:<5}: {g_mse.shape[0]} goal traj, {v_mse.shape[0]} vel traj")

    # ── Assemble ordered stats dicts (Ours first) ─────────────────────────
    goal_stats = {"Ours": (ours_goal_mean, ours_goal_std)}
    vel_stats  = {"Ours": (ours_vel_mean,  ours_vel_std)}
    for name in ["PEARL", "RL2", "CEMRL", "MELTS"]:
        goal_stats[name] = baselines[name]["goal"]
        vel_stats[name]  = baselines[name]["vel"]

    # ── Print summary tables ──────────────────────────────────────────────
    for name, (mean, std) in goal_stats.items():
        print_table(f"Goal Tracking MSE  — {name}  (x_after − true_goal)²",
                    mean, std)
    for name, (mean, std) in vel_stats.items():
        print_table(f"Velocity Tracking MSE  — {name}  (vx_after − target_v)²",
                    mean, std)

    # ── Save combined CSV source data ─────────────────────────────────────
    save_combined_csv(
        goal_stats,
        os.path.join(OUT_DIR, "tracking_mse_goal.csv"),
    )
    save_combined_csv(
        vel_stats,
        os.path.join(OUT_DIR, "tracking_mse_velocity.csv"),
    )

    # ── Figure 1: Goal Tracking MSE comparison ───────────────────────────
    plot_comparison(
        goal_stats,
        ylabel=r"Goal Tracking MSE  $(x_t - g)^2$",
        title="Goal Tracking MSE per Time Step — Ours vs Baselines",
        out_path=os.path.join(OUT_DIR, "comparison_goal_mse.pdf"),
    )

    # ── Figure 2: Velocity Tracking MSE comparison ───────────────────────
    plot_comparison(
        vel_stats,
        ylabel=r"Velocity Tracking MSE  $(v_t - v^*)^2$",
        title="Velocity Tracking MSE per Time Step — Ours vs Baselines",
        out_path=os.path.join(OUT_DIR, "comparison_velocity_mse.pdf"),
    )


if __name__ == "__main__":
    main()

