"""
Record 8 MP4 videos of Half-Cheetah performing each task with its task color.
Uses MjRenderContextOffscreen (no display needed).

Run:
    DISPLAY=:99 conda run -n bo-infer python record_cheetah_videos.py
"""

import os
import sys
import numpy as np

os.environ.setdefault("DISPLAY", ":99")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import mujoco_py
import imageio
import cv2

from meta_rand_envs.half_cheetah_multi_env import HalfCheetahMixtureEnv

# ──────────────────────────────────────────────────────────────────────────────
# Task configuration
# ──────────────────────────────────────────────────────────────────────────────
TASK_VARIANTS = [
    "velocity_forward",
    "velocity_backward",
    "goal_forward",
    "goal_backward",
    "flip_forward",
    "stand_front",
    "stand_back",
    "jump",
]

TASK_LABELS = {
    "velocity_forward":  "(a) Run forward",
    "velocity_backward": "(b) Run back",
    "goal_forward":      "(c) Goal in front",
    "goal_backward":     "(d) Goal in back",
    "flip_forward":      "(h) Front flip",
    "stand_front":       "(e) Front stand",
    "stand_back":        "(f) Back stand",
    "jump":              "(g) Jump",
}

# Colors from sample_tasks()
TASK_COLORS = {
    "velocity_forward":  np.array([1.0, 0.0, 0.0]),   # red
    "velocity_backward": np.array([0.0, 1.0, 0.0]),   # green
    "goal_forward":      np.array([1.0, 1.0, 0.0]),   # yellow
    "goal_backward":     np.array([0.0, 1.0, 1.0]),   # cyan
    "flip_forward":      np.array([0.5, 0.5, 0.0]),   # olive
    "stand_front":       np.array([1.0, 0.0, 0.5]),   # pink-red
    "stand_back":        np.array([0.5, 0.0, 1.0]),   # purple
    "jump":              np.array([0.5, 0.5, 0.5]),   # grey
}

TASK_SPECS = {
    "velocity_forward":  3.0,
    "velocity_backward": -3.0,
    "goal_forward":      15.0,
    "goal_backward":     -15.0,
    "flip_forward":      3.0 * np.pi,
    "stand_front":       np.pi / 3.0,
    "stand_back":        -np.pi / 3.0,
    "jump":              2.5,
}

# Number of frames to record per video (at ~20fps → 5s each)
N_FRAMES = 100
IMG_W, IMG_H = 1280, 720
FPS = 20
OUTPUT_DIR = "cheetah_task_videos"


# ──────────────────────────────────────────────────────────────────────────────
# Scripted action policies (heuristic, no trained policy needed)
# ──────────────────────────────────────────────────────────────────────────────
def get_action(task_name: str, step: int, action_dim: int) -> np.ndarray:
    t = step / N_FRAMES
    phase = 2 * np.pi * t * 3   # 3 cycles over the episode

    if task_name in ("velocity_forward", "goal_forward"):
        # Gallop forward: alternating legs
        a = np.array([
            np.sin(phase),         # bthigh
            -np.sin(phase + 0.5),  # bshin
            np.sin(phase + 1.0),   # bfoot
            -np.sin(phase),        # fthigh
            np.sin(phase + 0.5),   # fshin
            -np.sin(phase + 1.0),  # ffoot
        ]) * 0.9
    elif task_name in ("velocity_backward", "goal_backward"):
        # Gallop backward
        a = np.array([
            -np.sin(phase),
            np.sin(phase + 0.5),
            -np.sin(phase + 1.0),
            np.sin(phase),
            -np.sin(phase + 0.5),
            np.sin(phase + 1.0),
        ]) * 0.9
    elif task_name == "flip_forward":
        # Tuck and rotate: strong torso rotation + alternating limbs
        a = np.array([
            np.sin(phase * 2),
            -1.0,
            np.sin(phase),
            -np.sin(phase * 2),
            1.0,
            -np.sin(phase),
        ]) * 1.0
    elif task_name == "stand_front":
        # Push back legs to raise front
        a = np.array([
            -0.3,                    # bthigh - pull back
            0.8 + 0.2 * np.sin(phase),  # bshin - push up
            -0.5,                    # bfoot
            0.5,                     # fthigh
            -0.3 * np.sin(phase),    # fshin
            0.1,                     # ffoot
        ])
    elif task_name == "stand_back":
        # Push front legs to raise back
        a = np.array([
            0.5,                     # bthigh
            -0.3 * np.sin(phase),    # bshin
            0.1,                     # bfoot
            -0.3,                    # fthigh - pull forward
            0.8 + 0.2 * np.sin(phase),  # fshin - push up
            -0.5,                    # ffoot
        ])
    elif task_name == "jump":
        # Compress and extend legs rhythmically
        cycle = step % 30
        if cycle < 10:
            # compress
            a = np.array([-1.0, -1.0, -0.5, -1.0, -1.0, -0.5])
        elif cycle < 20:
            # extend / launch
            a = np.array([1.0, 1.0, 0.5, 1.0, 1.0, 0.5])
        else:
            # land / neutral
            a = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    else:
        a = np.zeros(6)

    full = np.zeros(action_dim)
    full[:len(a)] = a
    return np.clip(full, -1.0, 1.0)


# ──────────────────────────────────────────────────────────────────────────────
# Text overlay helper
# ──────────────────────────────────────────────────────────────────────────────
def add_label(frame: np.ndarray, text: str, color_rgb: np.ndarray) -> np.ndarray:
    img = frame.copy()
    h, w = img.shape[:2]

    # Semi-transparent black bar at bottom
    bar_h = 44
    overlay = img.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (30, 30, 30), -1)
    img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)

    # Colored dot
    dot_rgb = (int(color_rgb[2] * 255), int(color_rgb[1] * 255), int(color_rgb[0] * 255))  # BGR
    cv2.circle(img, (20, h - bar_h // 2), 10, dot_rgb, -1)
    cv2.circle(img, (20, h - bar_h // 2), 10, (255, 255, 255), 1)

    # Label text
    cv2.putText(img, text, (38, h - bar_h // 2 + 6),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    return img


# ──────────────────────────────────────────────────────────────────────────────
# Main recording loop
# ──────────────────────────────────────────────────────────────────────────────
def build_env() -> HalfCheetahMixtureEnv:
    print("Initializing environment...")
    env = HalfCheetahMixtureEnv(
        task_variants=TASK_VARIANTS,
        n_train_tasks=8,
        n_eval_tasks=8,
        change_mode="",
        log_scale_limit=0,
    )
    # Force offscreen renderer
    env.viewer = mujoco_py.MjRenderContextOffscreen(env.sim, device_id=-1)
    # Close-up side-view tracking camera
    env.viewer.cam.type = 1          # CAMERA_TRACKING (follows body)
    env.viewer.cam.trackbodyid = 1   # torso
    env.viewer.cam.distance = 3.5    # zoom in close
    env.viewer.cam.elevation = -15   # slight downward angle
    env.viewer.cam.azimuth = 90      # side view
    env._viewers["rgb_array"] = env.viewer
    return env


def set_task(env: HalfCheetahMixtureEnv, task_name: str):
    idx = TASK_VARIANTS.index(task_name)
    color = TASK_COLORS[task_name]
    spec = TASK_SPECS[task_name]
    # _task is read by reset_model(); must be set before any reset() call
    env._task = {'base_task': idx, 'specification': spec, 'color': color}
    env.base_task = idx
    env.task_specification = spec
    env.color = color
    env.recolor()


def render_frame(env: HalfCheetahMixtureEnv) -> np.ndarray:
    # Offscreen render – returns (H, W, 3) uint8, upside-down
    img = env.sim.render(width=IMG_W, height=IMG_H, camera_name=None)
    return img[::-1].copy()  # flip vertically


def record_task(env: HalfCheetahMixtureEnv, task_name: str, task_dir: str):
    print(f"  Recording: {task_name} → {task_dir}/")
    os.makedirs(task_dir, exist_ok=True)

    set_task(env, task_name)

    # reset_model() reads env._task and calls recolor() internally
    env.reset()

    action_dim = env.action_space.shape[0]
    out_path = os.path.join(task_dir, f"{task_name}.mp4")
    writer = imageio.get_writer(out_path, fps=FPS, quality=10, macro_block_size=None)

    label = TASK_LABELS[task_name]
    color = TASK_COLORS[task_name]

    for step in range(N_FRAMES):
        action = get_action(task_name, step, action_dim)
        try:
            env.step(action)
        except RuntimeError as e:
            print(f"    step error: {e}")
            break

        frame = render_frame(env)
        frame = add_label(frame, label, color)
        writer.append_data(frame)

        frame_path = os.path.join(task_dir, f"frame_{step:04d}.png")
        imageio.imwrite(frame_path, frame)

    writer.close()
    print(f"    Saved {N_FRAMES} frames → {task_dir}/{task_name}.mp4 + frame_0000~{N_FRAMES-1:04d}.png")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    env = build_env()

    for task_name in TASK_VARIANTS:
        task_dir = os.path.join(OUTPUT_DIR, task_name)
        try:
            record_task(env, task_name, task_dir)
        except Exception as e:
            print(f"  ERROR for {task_name}: {e}")
            import traceback; traceback.print_exc()

    print(f"\nDone! Output structure:")
    for task_name in TASK_VARIANTS:
        task_dir = os.path.join(OUTPUT_DIR, task_name)
        if os.path.isdir(task_dir):
            files = sorted(os.listdir(task_dir))
            mp4s = [f for f in files if f.endswith(".mp4")]
            pngs = [f for f in files if f.endswith(".png")]
            print(f"  {task_dir}/  ({len(mp4s)} mp4, {len(pngs)} png)")


if __name__ == "__main__":
    main()
