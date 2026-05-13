"""
Visualize Walker2d 4-Task benchmark.
Saves 100 frames per task to agent_task_frames/walker2d/{task_name}/.
Also saves a composite preview figure.
Run with: conda run -n bo-infer python visualiza_walker_tasks.py
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mplimg

os.chdir(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, 'submodules/SAC')
from sac_envs.walker_multi import WalkerMulti

WALKER_CFG_PATH = 'output/low_level_policy/walker_full_06_07_multi_task/config.json'
OUTPUT_BASE     = 'agent_task_frames/walker2d'

# dt = 0.04s per step (frame_skip=4, timestep=0.01)
# 150 steps ≈ 6 seconds; save 100 frames evenly spaced
N_STEPS  = 150
N_FRAMES = 100
IMG_SIZE = 512

# ─── Task definitions ───────────────────────────────────────────────────────
TASK_VARIANTS = ['forward_vel', 'backward_vel', 'goal_front', 'goal_back']

TASK_LABELS = {
    'forward_vel':  '(a) Run forward',
    'backward_vel': '(b) Run back',
    'goal_front':   '(c) Goal in front',
    'goal_back':    '(d) Goal in back',
}

TASK_COLORS = {
    'forward_vel':  [1.0, 0.0, 0.0],
    'backward_vel': [0.0, 1.0, 0.0],
    'goal_front':   [1.0, 1.0, 0.0],
    'goal_back':    [0.0, 1.0, 1.0],
}

TASK_SPECS = {
    'forward_vel':  2.0,
    'backward_vel': -2.0,
    'goal_front':   5.0,
    'goal_back':    -5.0,
}


# ─── Environment setup ───────────────────────────────────────────────────────
def build_env():
    cfg = json.load(open(WALKER_CFG_PATH))
    env = WalkerMulti(cfg)
    return env, cfg


def set_task(env, cfg, task_name):
    task_idx = cfg['tasks'][task_name]
    spec     = TASK_SPECS[task_name]
    env.base_task = task_idx
    env.task = np.zeros(max(cfg['tasks'].values()) + 1)
    env.task[task_idx] = spec
    env.norm = max(abs(spec), 1.0)
    env.reset()
    # Recolor agent (geom 0 = floor, skip it)
    geom_rgba = env.model.geom_rgba.copy()
    geom_rgba[1:, :3] = np.array(TASK_COLORS[task_name])
    env.model.geom_rgba[:] = geom_rgba


def get_action(task_name, i, n_steps, action_dim):
    t     = i / n_steps
    phase = 2 * np.pi * t * 3   # 3 gait cycles
    if task_name in ('forward_vel', 'goal_front'):
        a = np.array([
             0.5 * np.sin(phase),
             0.5 * np.sin(phase + np.pi),
             0.3,
             0.5 * np.sin(phase + np.pi),
             0.5 * np.sin(phase),
             0.3,
        ])
    else:
        a = np.array([
            -0.5 * np.sin(phase),
            -0.5 * np.sin(phase + np.pi),
            -0.3,
            -0.5 * np.sin(phase + np.pi),
            -0.5 * np.sin(phase),
            -0.3,
        ])
    return a[:action_dim]


# ─── Render & save frames ───────────────────────────────────────────────────
def render_and_save(env, cfg, task_name, n_steps=N_STEPS, n_frames=N_FRAMES, img_size=IMG_SIZE):
    """Run task, capture n_frames evenly, save as PNG, return list of frames."""
    set_task(env, cfg, task_name)
    action_dim = env.action_space.shape[0]

    # Indices at which to capture a frame
    capture_idx = set(np.linspace(0, n_steps - 1, n_frames, dtype=int))

    out_dir = os.path.join(OUTPUT_BASE, task_name)
    os.makedirs(out_dir, exist_ok=True)

    saved_frames = []
    frame_num    = 0

    for i in range(n_steps):
        action = get_action(task_name, i, n_steps, action_dim)
        try:
            env.step(action)
        except Exception:
            break

        if i in capture_idx:
            img = env.get_image(width=img_size, height=img_size, camera_name=None)
            img = img[::-1]   # mujoco returns upside-down
            path = os.path.join(out_dir, f'frame_{frame_num:04d}.png')
            mplimg.imsave(path, img)
            saved_frames.append(img)
            frame_num += 1

    print(f"  [{task_name}] saved {frame_num} frames → {out_dir}/")
    return saved_frames


# ─── Composite preview figure ────────────────────────────────────────────────
def make_composite(all_frames, output_path, img_size=IMG_SIZE):
    n = len(TASK_VARIANTS)
    fig, axes = plt.subplots(1, n, figsize=(n * 2.5, 3.2))
    fig.patch.set_facecolor('#c8c8c8')

    for ax, task_name, frames in zip(axes, TASK_VARIANTS, all_frames):
        frame = frames[len(frames) // 2]   # middle frame
        ax.imshow(frame)
        ax.set_xlabel(TASK_LABELS[task_name], fontsize=8, labelpad=4)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle(
        r'Four base tasks in the benchmark environment $\it{Walker2d~4}$-$\it{Task}$.',
        fontsize=9, y=0.02, ha='left', x=0.01, fontstyle='italic')

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    png_path = output_path.replace('.pdf', '.png')
    fig.savefig(png_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    fig.savefig(output_path, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"Composite saved:\n  {png_path}\n  {output_path}")
    plt.close(fig)


# ─── Main ────────────────────────────────────────────────────────────────────
def main(output_path='walker_4task_visualization.pdf'):
    print("Building environment...")
    env, cfg = build_env()

    all_frames = []
    for task_name in TASK_VARIANTS:
        print(f"Rendering: {task_name}  ({N_STEPS} steps → {N_FRAMES} frames)...")
        try:
            frames = render_and_save(env, cfg, task_name)
        except Exception as e:
            print(f"  WARNING: {e} — blank frames")
            frames = [np.ones((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8) * 200] * N_FRAMES
        all_frames.append(frames)

    make_composite(all_frames, output_path)


if __name__ == '__main__':
    out = sys.argv[1] if len(sys.argv) > 1 else 'walker_4task_visualization.pdf'
    main(out)
