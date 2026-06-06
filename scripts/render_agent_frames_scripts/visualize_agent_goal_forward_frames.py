"""


How to use:
This scripts helps visualize the showcase frame for hopper, walker, cheetah, ant, only for the goal forward, you can use visualize_all_agents.py first
the render settings are the same, and the only thing you need to do is to change the path config path for each agent, 
and run the script, it will save the frames to the corresponding folder.



Generate goal_forward frames for walker2d, ant, hopper, and toy agents.
Saves PNG frames to:
  agent_task_frames/<agent>/goal_forward_frames/frame_XXXX.png

Run with:
    conda run -n bo-infer python visualize_agent_goal_forward_frames.py
"""

import os
import sys
import json
import numpy as np
import imageio

os.chdir(os.path.dirname(os.path.abspath(__file__)))

import mujoco_py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sac_envs.walker_multi import WalkerMulti
from sac_envs.hopper_multi import HopperMulti
from sac_envs.ant_multi_old import AntMulti
from rlkit.envs.toy1d_multi import Toy1dMultiTaskkWrappedEnv

# ─── Output directory ────────────────────────────────────────────────────────
OUTPUT_ROOT = 'toy_test'
N_FRAMES    = 100
IMG_W       = 400
IMG_H       = 400

# ─── Agent configs ────────────────────────────────────────────────────────────
# Paths to config.json for each multi-task env
WALKER_CFG_PATH = 'output/low_level_policy/walker_multi_new_config_v0_run0/config.json'
HOPPER_CFG_PATH = 'output/low_level_policy/hopper_multi_new_config_v3_run0/config.json'
ANT_CFG_PATH    = 'output/low_level_policy/ant_multi_new_config_v4_run1/config.json'


# ─── Camera setup (matches cheetah get_image / viewer_setup) ─────────────────
def initialize_camera(env):
    """
    Register an offscreen render context using the same camera parameters
    as the cheetah visualization (visualiza_cheetah_tasks.py):
      type=2 (fixed), fixedcamid=0 — uses the first camera defined in the XML
      (named "track", mode="trackcom") so the camera tracks the agent body.
    Must be called once after env creation, before any render_frame calls.
    """
    sim = env.sim
    viewer = mujoco_py.MjRenderContextOffscreen(sim)
    camera = viewer.cam
    camera.type = 2          # mjCAMERA_FIXED — use the XML-defined camera
    camera.fixedcamid = 0    # index 0 = camera named "track" in worldbody
    sim.add_render_context(viewer)


# ─── Helper: set all body geoms to red (index 1+ skips the floor) ────────────
def set_color_red(env):
    """Recolor all body geoms to red, matching cheetah velocity_forward color."""
    geom_rgba = env.model.geom_rgba.copy()
    geom_rgba[1:, :3] = np.array([1.0, 0.0, 0.0])
    env.model.geom_rgba[:] = geom_rgba


# ─── Helper: offscreen render ─────────────────────────────────────────────────
def render_frame(env, width=IMG_W, height=IMG_H):
    """Render one frame using the offscreen context set by initialize_camera."""
    img = env.sim.render(width=width, height=height, camera_name=None)
    return img[::-1].copy()  # mujoco returns upside-down


def add_label(frame: np.ndarray, label: str, color_rgb=(1.0, 1.0, 0.0)) -> np.ndarray:
    """Overlay a task label on the frame using matplotlib."""
    fig, ax = plt.subplots(figsize=(frame.shape[1] / 100, frame.shape[0] / 100), dpi=100)
    ax.imshow(frame)
    ax.text(
        0.5, 0.03, label,
        transform=ax.transAxes,
        ha='center', va='bottom',
        fontsize=10, fontweight='bold',
        color=color_rgb,
        bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.5)
    )
    ax.axis('off')
    fig.tight_layout(pad=0)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[..., :3]
    w, h = fig.canvas.get_width_height()
    labeled = buf.reshape(h, w, 3)
    plt.close(fig)
    return labeled


# ─── Walker2d ─────────────────────────────────────────────────────────────────
def record_walker_goal_forward():
    agent_name = 'walker2d'
    task_label = 'Goal Forward'
    label_color = (1.0, 1.0, 0.0)
    out_dir = os.path.join(OUTPUT_ROOT, agent_name, 'goal_forward_frames')
    os.makedirs(out_dir, exist_ok=True)

    print(f'[walker2d] Building env...')
    cfg = json.load(open(WALKER_CFG_PATH))
    env = WalkerMulti(cfg)
    initialize_camera(env)

    # goal_front task: idx 0, set a forward goal spec
    env.base_task = cfg['tasks']['goal_front']
    env.task = np.zeros(max(cfg['tasks'].values()) + 1)
    env.task[env.base_task] = 8.0   # goal position: 8 m forward
    env.norm = max(abs(env.task[env.base_task]), 1.0)
    env.reset()
    set_color_red(env)

    action_dim = env.action_space.shape[0]
    print(f'[walker2d] Recording {N_FRAMES} frames...')

    for step in range(N_FRAMES):
        # Walking gait: alternating leg actions
        t = step / N_FRAMES
        phase = 2 * np.pi * t * 3  # 3 gait cycles
        action = np.array([
            0.5 * np.sin(phase),
            0.5 * np.sin(phase + np.pi),
            0.5 * np.sin(phase),
            0.5 * np.sin(phase + np.pi),
            0.3,
            0.3,
        ])
        action = action[:action_dim]
        try:
            env.step(action)
        except Exception as e:
            print(f'  step error at {step}: {e}')
            break

        frame = render_frame(env)
        frame = add_label(frame, task_label, label_color)
        imageio.imwrite(os.path.join(out_dir, f'frame_{step:04d}.png'), frame)

    print(f'[walker2d] Saved {N_FRAMES} frames to {out_dir}/')


# ─── Hopper ───────────────────────────────────────────────────────────────────
def record_hopper_goal_forward():
    agent_name = 'hopper'
    task_label = 'Goal Forward'
    label_color = (1.0, 1.0, 0.0)
    out_dir = os.path.join(OUTPUT_ROOT, agent_name, 'goal_forward_frames')
    os.makedirs(out_dir, exist_ok=True)

    print(f'[hopper] Building env...')
    cfg = json.load(open(HOPPER_CFG_PATH))
    env = HopperMulti(cfg)
    initialize_camera(env)

    env.base_task = cfg['tasks']['goal_front']
    env.task = np.zeros(max(cfg['tasks'].values()) + 1)
    env.task[env.base_task] = 8.0
    env.norm = max(abs(env.task[env.base_task]), 1.0)
    env.reset()
    set_color_red(env)

    action_dim = env.action_space.shape[0]
    print(f'[hopper] Recording {N_FRAMES} frames...')

    for step in range(N_FRAMES):
        t = step / N_FRAMES
        phase = 2 * np.pi * t * 4
        action = np.array([
            0.6 * np.sin(phase),
            0.8 * np.sin(phase + np.pi / 3),
            0.6 * np.sin(phase + 2 * np.pi / 3),
        ])
        action = action[:action_dim]
        try:
            env.step(action)
        except Exception as e:
            print(f'  step error at {step}: {e}')
            break

        frame = render_frame(env)
        frame = add_label(frame, task_label, label_color)
        imageio.imwrite(os.path.join(out_dir, f'frame_{step:04d}.png'), frame)

    print(f'[hopper] Saved {N_FRAMES} frames to {out_dir}/')


# ─── Ant ──────────────────────────────────────────────────────────────────────
def record_ant_goal_forward():
    """
    Ant's old multitask env uses 'goal_front' for the +x direction,
    analogous to 'goal forward' for other agents.
    """
    agent_name = 'ant'
    task_label = 'Goal Forward'
    label_color = (1.0, 1.0, 0.0)
    out_dir = os.path.join(OUTPUT_ROOT, agent_name, 'goal_forward_frames')
    os.makedirs(out_dir, exist_ok=True)

    print(f'[ant] Building env...')
    cfg = json.load(open(ANT_CFG_PATH))
    env = AntMulti(cfg)
    initialize_camera(env)

    # Old AntMulti config uses goal_front for positive x movement.
    env.base_task = cfg['tasks']['goal_front']
    env.task = np.zeros(max(cfg['tasks'].values()) + 1)
    env.task[env.base_task] = 8.0
    env.norm = max(abs(env.task[env.base_task]), 1.0)
    env.reset()
    env.update_task(env.task)

    action_dim = env.action_space.shape[0]
    print(f'[ant] Recording {N_FRAMES} frames...')

    for step in range(N_FRAMES):
        t = step / N_FRAMES
        phase = 2 * np.pi * t * 3
        # Ant locomotion: 8-dim action (4 hip joints + 4 ankle joints)
        action = np.array([
             0.5 * np.sin(phase),
            -0.5 * np.sin(phase),
             0.5 * np.sin(phase + np.pi),
            -0.5 * np.sin(phase + np.pi),
             0.8,
             0.8,
             0.8,
             0.8,
        ])
        action = action[:action_dim]
        try:
            env.step(action)
        except Exception as e:
            print(f'  step error at {step}: {e}')
            break

        frame = render_frame(env)
        frame = add_label(frame, task_label, label_color)
        imageio.imwrite(os.path.join(out_dir, f'frame_{step:04d}.png'), frame)

    print(f'[ant] Saved {N_FRAMES} frames to {out_dir}/')


# ─── Toy 1D ───────────────────────────────────────────────────────────────────
def record_toy_goal_forward():
    agent_name = 'toy1d'
    task_label = 'Goal Forward'
    label_color = (1.0, 1.0, 0.0)
    out_dir = os.path.join(OUTPUT_ROOT, agent_name, 'goal_forward_frames')
    os.makedirs(out_dir, exist_ok=True)

    print(f'[toy1d] Building env...')
    env = Toy1dMultiTaskkWrappedEnv(
        n_train_tasks=20,
        n_eval_tasks=10,
        task_variants=['goal_forward', 'goal_backward', 'velocity_forward', 'velocity_backward'],
    )

    # Tracking camera follows the box body; floor checker scrolls to show motion
    sim = env.sim
    viewer = mujoco_py.MjRenderContextOffscreen(sim)
    camera = viewer.cam
    camera.type = 1             # mjCAMERA_TRACKING - follows the tracked body
    camera.trackbodyid = 1      # body 1 = "box"
    camera.distance = 5.0      # close enough to see the box clearly
    camera.elevation = -20     # same tilt as cheetah
    camera.azimuth = 90        # side view: x-axis goes left->right
    sim.add_render_context(viewer)

    # Find a task with goal_forward and positive goal
    goal_fwd_idx_in_bt = env.task_variants.index('goal_forward')
    task_idx = next(
        i for i, t in enumerate(env.tasks)
        if t['base_task'] == goal_fwd_idx_in_bt and t['specification'] > 0
    )
    env.reset_task(task_idx)
    env.color = np.array([1.0, 0.0, 0.0])
    env.recolor()
    print(f'  Using task {task_idx}: base_task={env.base_task}, spec={env.task_specification:.2f}')

    action_dim = env.action_space.shape[0]
    print(f'[toy1d] Recording {N_FRAMES} frames...')

    for step in range(N_FRAMES):
        # 1D env: push forward
        action = np.array([0.8] * action_dim)
        try:
            env.step(action)
        except Exception as e:
            print(f'  step error at {step}: {e}')
            break

        # toy1d uses set_joint_qpos directly (no do_simulation), so we need
        # sim.forward() to propagate qpos -> body xpos before rendering
        env.sim.forward()

        frame = render_frame(env, width=800, height=800)
        frame = add_label(frame, task_label, label_color)
        imageio.imwrite(os.path.join(out_dir, f'frame_{step:04d}.png'), frame)

    print(f'[toy1d] Saved {N_FRAMES} frames to {out_dir}/')


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    record_walker_goal_forward()
    record_hopper_goal_forward()
    record_ant_goal_forward()
    record_toy_goal_forward()

    print(f'\nAll done. Frames saved under {OUTPUT_ROOT}/')
