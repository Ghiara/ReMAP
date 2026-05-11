"""
Visualize Half-Cheetah 8-Task benchmark for paper figures.
Each task is shown with its unique color in a characteristic pose.
Run with: conda run -n bo-infer python visualize_cheetah_tasks.py
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from meta_rand_envs.half_cheetah_multi_env import HalfCheetahMixtureEnv

# ─── Task definitions matching sample_tasks() colors ───────────────────────
TASK_VARIANTS = [
    'velocity_forward', 'velocity_backward',
    'goal_forward', 'goal_backward',
    'flip_forward',
    'stand_front', 'stand_back',
    'jump',
]

TASK_LABELS = [
    '(a) Run forward',
    '(b) Run back',
    '(c) Goal in front',
    '(d) Goal in back',
    '(h) Front flip',
    '(e) Front stand',
    '(f) Back stand',
    '(g) Jump',
]

# Colors from sample_tasks() in half_cheetah_multi_env.py
TASK_COLORS = {
    'velocity_forward':  [1.0,  0.0,  0.0],
    'velocity_backward': [0.0,  1.0,  0.0],
    'goal_forward':      [1.0,  1.0,  0.0],
    'goal_backward':     [0.0,  1.0,  1.0],
    'flip_forward':      [0.5,  0.5,  0.0],
    'stand_front':       [1.0,  0.0,  0.5],
    'stand_back':        [0.5,  0.0,  1.0],
    'jump':              [0.5,  0.5,  0.5],
}

# Arrow directions: (dx, dy) in image space, None = no arrow, 'circle' = circle
TASK_ARROWS = {
    'velocity_forward':  ('right', None),
    'velocity_backward': ('left',  None),
    'goal_forward':      ('down',  None),   # goal in front = move forward, arrow down-ish
    'goal_backward':     ('down',  None),
    'flip_forward':      ('circle', None),
    'stand_front':       ('arc_front', None),
    'stand_back':        ('arc_back', None),
    'jump':              ('up',    None),
}

# How many steps to run for each task to get a representative pose
WARMUP_STEPS = {
    'velocity_forward':  60,
    'velocity_backward': 60,
    'goal_forward':      50,
    'goal_backward':     50,
    'flip_forward':      80,
    'stand_front':       120,
    'stand_back':        120,
    'jump':              60,
}


def build_env():
    env = HalfCheetahMixtureEnv(
        task_variants=TASK_VARIANTS,
        n_train_tasks=8,
        n_eval_tasks=8,
        change_mode='',
        log_scale_limit=0,
    )
    return env


def set_task(env, task_name):
    """Manually set the task and recolor."""
    idx = TASK_VARIANTS.index(task_name)
    color = TASK_COLORS[task_name]
    env.base_task = idx
    env.task_specification = _default_spec(task_name)
    env.color = color
    env.recolor()


def _default_spec(task_name):
    specs = {
        'velocity_forward':  3.0,
        'velocity_backward': -3.0,
        'goal_forward':      15.0,
        'goal_backward':     -15.0,
        'flip_forward':      3.0 * np.pi,
        'stand_front':       np.pi / 3,
        'stand_back':        -np.pi / 3,
        'jump':              2.5,
    }
    return specs[task_name]


def get_task_frame(env, task_name, img_size=300):
    """Reset env, run warmup steps with task-specific actions, return frame."""
    set_task(env, task_name)
    env.reset()

    steps = WARMUP_STEPS[task_name]
    action_dim = env.action_space.shape[0]

    for i in range(steps):
        if task_name in ('velocity_forward', 'goal_forward'):
            action = np.array([0.5, -0.5,  0.5, -0.5,  0.5,  0.5] + [0.0] * (action_dim - 6))
        elif task_name in ('velocity_backward', 'goal_backward'):
            action = np.array([-0.5, 0.5, -0.5, 0.5, -0.5, -0.5] + [0.0] * (action_dim - 6))
        elif task_name == 'flip_forward':
            # Encourage forward rotation
            action = np.array([1.0, -1.0, 1.0, -0.5, 0.5, -0.5] + [0.0] * (action_dim - 6))
        elif task_name == 'stand_front':
            # Rear legs push up
            action = np.array([0.0, 0.5, -1.0, 0.5, 0.0, 0.0] + [0.0] * (action_dim - 6))
        elif task_name == 'stand_back':
            action = np.array([0.0, -0.5, 1.0, -0.5, 0.0, 0.0] + [0.0] * (action_dim - 6))
        elif task_name == 'jump':
            t = i / steps
            if t < 0.5:
                action = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0] + [0.0] * (action_dim - 6))
            else:
                action = np.array([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0] + [0.0] * (action_dim - 6))
        else:
            action = env.action_space.sample() * 0.3

        action = action[:action_dim]
        try:
            env.step(action)
        except Exception:
            break

    img = env.get_image(width=img_size, height=img_size, camera_name=None)
    # mujoco returns upside-down images
    img = img[::-1]
    return img


def add_arrow_overlay(ax, task_name, img_size=300):
    """Add task-characteristic yellow arrows onto the axes."""
    cx, cy = img_size / 2, img_size / 2
    arrow_kw = dict(color='yellow', linewidth=2.5, zorder=10)
    arr_kw = dict(arrowstyle='->', color='yellow', lw=2.5, mutation_scale=20, zorder=10)

    if task_name in ('velocity_forward', 'goal_forward'):
        ax.annotate('', xy=(cx + 70, cy - 20), xytext=(cx - 20, cy - 20),
                    arrowprops=dict(arrowstyle='->', color='yellow', lw=2.5, mutation_scale=22))
    elif task_name in ('velocity_backward', 'goal_backward'):
        ax.annotate('', xy=(cx - 70, cy - 20), xytext=(cx + 20, cy - 20),
                    arrowprops=dict(arrowstyle='->', color='yellow', lw=2.5, mutation_scale=22))
    elif task_name == 'jump':
        ax.annotate('', xy=(cx, cy - 80), xytext=(cx, cy + 10),
                    arrowprops=dict(arrowstyle='->', color='yellow', lw=2.5, mutation_scale=22))
        ax.annotate('', xy=(cx, cy + 80), xytext=(cx, cy - 10),
                    arrowprops=dict(arrowstyle='->', color='yellow', lw=2.5, mutation_scale=22))
    elif task_name == 'stand_front':
        # Arc arrow suggesting front-stand pivot
        theta = np.linspace(np.pi * 0.9, np.pi * 0.1, 30)
        r = 55
        xs = cx + r * np.cos(theta)
        ys = cy + r * np.sin(theta) - 20
        ax.plot(xs, ys, color='yellow', lw=2.5, zorder=10)
        ax.annotate('', xy=(xs[-1], ys[-1]), xytext=(xs[-2], ys[-2]),
                    arrowprops=dict(arrowstyle='->', color='yellow', lw=2.5, mutation_scale=18))
    elif task_name == 'stand_back':
        theta = np.linspace(np.pi * 0.1, np.pi * 0.9, 30)
        r = 55
        xs = cx + r * np.cos(theta)
        ys = cy + r * np.sin(theta) - 20
        ax.plot(xs, ys, color='yellow', lw=2.5, zorder=10)
        ax.annotate('', xy=(xs[-1], ys[-1]), xytext=(xs[-2], ys[-2]),
                    arrowprops=dict(arrowstyle='->', color='yellow', lw=2.5, mutation_scale=18))
    elif task_name == 'flip_forward':
        # Circle with arrow indicating rotation
        theta = np.linspace(0, 2 * np.pi * 0.85, 60)
        r = 55
        xs = cx + r * np.cos(theta)
        ys = cy - 20 + r * np.sin(theta)
        ax.plot(xs, ys, color='yellow', lw=2.5, zorder=10)
        ax.annotate('', xy=(xs[-1], ys[-1]), xytext=(xs[-2], ys[-2]),
                    arrowprops=dict(arrowstyle='->', color='yellow', lw=2.5, mutation_scale=18))


def render_all_tasks(output_path='cheetah_8task_visualization.pdf', img_size=300):
    print("Building environment...")
    env = build_env()

    n_tasks = len(TASK_VARIANTS)
    frames = []

    for task_name in TASK_VARIANTS:
        print(f"  Rendering: {task_name}")
        try:
            frame = get_task_frame(env, task_name, img_size=img_size)
        except Exception as e:
            print(f"    WARNING: failed ({e}), using blank frame")
            frame = np.ones((img_size, img_size, 3), dtype=np.uint8) * 200
        frames.append(frame)

    # ── Compose figure ────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, n_tasks, figsize=(n_tasks * 2.5, 3.2))
    fig.patch.set_facecolor('#c8c8c8')   # light grey background like original

    for ax, frame, task_name, label in zip(axes, frames, TASK_VARIANTS, TASK_LABELS):
        ax.imshow(frame)
        add_arrow_overlay(ax, task_name, img_size)
        ax.set_xlabel(label, fontsize=8, labelpad=4)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle('Eight base tasks in the benchmark environment $\\it{Half}$-$\\it{Cheetah~8}$-$\\it{Task}$.',
                 fontsize=9, y=0.02, ha='left', x=0.01,
                 fontstyle='italic')

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    # Save PNG and PDF
    png_path = output_path.replace('.pdf', '.png')
    fig.savefig(png_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    fig.savefig(output_path, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"\nSaved:\n  {png_path}\n  {output_path}")
    plt.close(fig)


if __name__ == '__main__':
    out = sys.argv[1] if len(sys.argv) > 1 else 'cheetah_8task_visualization.pdf'
    render_all_tasks(output_path=out, img_size=300)
