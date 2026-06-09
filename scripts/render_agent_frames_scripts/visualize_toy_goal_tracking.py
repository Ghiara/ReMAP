"""

How to use:
This scripts helps visualize the showcase frame for toy, 
the render settings are the same, and the only thing you need to do is to change the path config path for each agent, 
and run the script, it will save the frames to the corresponding folder.



Render 100 frames of the toy1d environment on a goal_forward task.

The toy1d XML (submodules/meta_rand_envs/meta_rand_envs/toy1d.xml) contains a
"track" camera with mode="trackcom" that follows the sphere body.  The checker-
board floor texture scrolls in the frame as the sphere moves, making the motion
clearly visible.

Output: toy_test/toy1d/goal_forward_frames/frame_XXXX.png

Run with:
    conda run -n bo-infer python visualize_toy_goal_forward.py
"""

import os
import numpy as np
import imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from third_party.rlkit.envs.toy1d_multi import Toy1dMultiTaskkWrappedEnv

# ─── Config ──────────────────────────────────────────────────────────────────
OUT_DIR  = os.path.join('agent_task_frames', 'toy1d', 'goal_forward_frames')
N_FRAMES = 100
IMG_W    = 512
IMG_H    = 512


# ─── Label overlay ────────────────────────────────────────────────────────────
def add_label(frame: np.ndarray, label: str,
              color_rgb=(1.0, 1.0, 0.0)) -> np.ndarray:
    fig, ax = plt.subplots(
        figsize=(frame.shape[1] / 100, frame.shape[0] / 100), dpi=100)
    ax.imshow(frame)
    ax.text(
        0.5, 0.03, label,
        transform=ax.transAxes,
        ha='center', va='bottom',
        fontsize=11, fontweight='bold',
        color=color_rgb,
        bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.5),
    )
    ax.axis('off')
    fig.tight_layout(pad=0)
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    labeled = buf.reshape(h, w, 3)
    plt.close(fig)
    return labeled


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print('[toy1d] Building env...')
    env = Toy1dMultiTaskkWrappedEnv(
        n_train_tasks=20,
        n_eval_tasks=10,
        task_variants=['goal_forward', 'goal_backward',
                       'velocity_forward', 'velocity_backward'],
    )

    # Pick a goal_forward task with a positive (forward) goal specification
    goal_fwd_bt = env.task_variants.index('goal_forward')
    task_idx = next(
        i for i, t in enumerate(env.tasks)
        if t['base_task'] == goal_fwd_bt and t['specification'] > 0
    )
    env.reset_task(task_idx)
    goal_spec = env.task_specification

    # Recolor the sphere red to match other agents
    env.color = np.array([1.0, 0.0, 0.0])
    env.recolor()

    print(f'  task_idx={task_idx}  goal={goal_spec:.2f} m')

    action_dim = env.action_space.shape[0]
    print(f'[toy1d] Recording {N_FRAMES} frames to {OUT_DIR}/ ...')

    for step in range(N_FRAMES):
        # Simple proportional controller: push toward the goal
        obs = env._get_obs()        # [x_position, x_velocity]
        x_pos = obs[0]
        error = goal_spec - x_pos
        action = np.clip(np.array([error * 0.5] * action_dim), -1.0, 1.0)

        try:
            env.step(action)
        except Exception as e:
            print(f'  step error at {step}: {e}')
            break

        # mujoco returns images upside-down; flip back
        # The "track" camera is defined in the XML with mode="trackcom" so it
        # automatically follows the sphere body — no manual setup needed.
        img = env.sim.render(width=IMG_W, height=IMG_H, camera_name='track')
        img = img[::-1].copy()

        # label = f'Goal Forward  |  goal={goal_spec:.1f} m  |  pos={x_pos:.2f} m'
        # img = add_label(img)

        path = os.path.join(OUT_DIR, f'frame_{step:04d}.png')
        imageio.imwrite(path, img)

    print(f'[toy1d] Done. {N_FRAMES} frames saved to {OUT_DIR}/')


if __name__ == '__main__':
    main()
