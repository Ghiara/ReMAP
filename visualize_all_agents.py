"""
Render 4 agents (Walker2d, Hopper, Ant, HalfCheetah) x their tasks.
Saves 100 frames per task to agent_task_frames/{agent}/{task_name}/.
Also saves a composite preview per agent.
Run with: conda run -n bo-infer python visualize_all_agents.py
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
from sac_envs.walker_multi  import WalkerMulti
from sac_envs.hopper_multi  import HopperMulti
from sac_envs.ant_multi     import AntMulti
sys.path.insert(0, 'submodules/meta_rand_envs')
from meta_rand_envs.half_cheetah_multi_env import HalfCheetahMixtureEnv

OUTPUT_BASE = 'agent_task_frames'

# dt ≈ 0.04s/step → 150 steps ≈ 6 s; capture 100 frames evenly
N_STEPS  = 150
N_FRAMES = 100
IMG_SIZE = 512

# ════════════════════════════════════════════════════════════════════════════
# Agent configs
# ════════════════════════════════════════════════════════════════════════════
AGENTS = {
    'walker2d': dict(
        cfg_path='output/low_level_policy/walker_full_06_07_multi_task/config.json',
        task_variants=['forward_vel', 'backward_vel', 'goal_front', 'goal_back'],
        task_labels={
            'forward_vel':  '(a) Run forward',
            'backward_vel': '(b) Run back',
            'goal_front':   '(c) Goal in front',
            'goal_back':    '(d) Goal in back',
        },
        task_colors={
            'forward_vel':  [1.0, 0.0, 0.0],
            'backward_vel': [0.0, 1.0, 0.0],
            'goal_front':   [1.0, 1.0, 0.0],
            'goal_back':    [0.0, 1.0, 1.0],
        },
        task_specs={
            'forward_vel':  2.0,
            'backward_vel': -2.0,
            'goal_front':   5.0,
            'goal_back':    -5.0,
        },
    ),
    'hopper': dict(
        cfg_path='output/low_level_policy/hopper_multi_multi_task/config.json',
        task_variants=['forward_vel', 'backward_vel', 'goal_front', 'goal_back'],
        task_labels={
            'forward_vel':  '(a) Run forward',
            'backward_vel': '(b) Run back',
            'goal_front':   '(c) Goal in front',
            'goal_back':    '(d) Goal in back',
        },
        task_colors={
            'forward_vel':  [1.0, 0.0, 0.0],
            'backward_vel': [0.0, 1.0, 0.0],
            'goal_front':   [1.0, 1.0, 0.0],
            'goal_back':    [0.0, 1.0, 1.0],
        },
        task_specs={
            'forward_vel':  2.0,
            'backward_vel': -2.0,
            'goal_front':   5.0,
            'goal_back':    -5.0,
        },
    ),
    'ant': dict(
        cfg_path='output/low_level_policy/ant_multitask_vel_goal/config.json',
        task_variants=['goal_left', 'goal_right', 'velocity_left', 'velocity_right'],
        task_labels={
            'goal_left':      '(a) Goal left',
            'goal_right':     '(b) Goal right',
            'velocity_left':  '(c) Vel left',
            'velocity_right': '(d) Vel right',
        },
        task_colors={
            'goal_left':      [1.0, 1.0, 0.0],
            'goal_right':     [0.0, 1.0, 1.0],
            'velocity_left':  [1.0, 0.0, 0.0],
            'velocity_right': [0.0, 1.0, 0.0],
        },
        task_specs={
            'goal_left':      -5.0,
            'goal_right':      5.0,
            'velocity_left':  -2.0,
            'velocity_right':  2.0,
        },
    ),
    'cheetah': dict(
        cfg_path=None,   # HalfCheetahMixtureEnv doesn't need a JSON config
        task_variants=['velocity_forward', 'velocity_backward', 'goal_forward', 'goal_backward'],
        task_labels={
            'velocity_forward':  '(a) Run forward',
            'velocity_backward': '(b) Run back',
            'goal_forward':      '(c) Goal in front',
            'goal_backward':     '(d) Goal in back',
        },
        task_colors={
            'velocity_forward':  [1.0, 0.0, 0.0],
            'velocity_backward': [0.0, 1.0, 0.0],
            'goal_forward':      [1.0, 1.0, 0.0],
            'goal_backward':     [0.0, 1.0, 1.0],
        },
        task_specs={
            'velocity_forward':  3.0,
            'velocity_backward': -3.0,
            'goal_forward':      15.0,
            'goal_backward':     -15.0,
        },
    ),
}

# ════════════════════════════════════════════════════════════════════════════
# Build environments
# ════════════════════════════════════════════════════════════════════════════
def build_env(agent_name, acfg):
    if agent_name == 'walker2d':
        cfg = json.load(open(acfg['cfg_path']))
        return WalkerMulti(cfg), cfg
    elif agent_name == 'hopper':
        cfg = json.load(open(acfg['cfg_path']))
        return HopperMulti(cfg), cfg
    elif agent_name == 'ant':
        cfg = json.load(open(acfg['cfg_path']))
        # task_variants = acfg['task_variants']
        # env = AntMulti(task_variants=task_variants)
        return AntMulti(cfg), cfg
    elif agent_name == 'cheetah':
        all_variants = ['velocity_forward', 'velocity_backward',
                        'goal_forward', 'goal_backward']
        env = HalfCheetahMixtureEnv(
            task_variants=all_variants,
            n_train_tasks=4, n_eval_tasks=4,
            change_mode='', log_scale_limit=0,
        )
        return env, None


# ════════════════════════════════════════════════════════════════════════════
# Set task + recolor
# ════════════════════════════════════════════════════════════════════════════
def set_task(agent_name, env, cfg, task_name, acfg):
    color = acfg['task_colors'][task_name]
    spec  = acfg['task_specs'][task_name]

    if agent_name in ('walker2d', 'hopper'):
        task_idx = cfg['tasks'][task_name]
        env.base_task = task_idx
        env.task = np.zeros(max(cfg['tasks'].values()) + 1)
        env.task[task_idx] = spec
        env.norm = max(abs(spec), 1.0)
        env.reset()
        geom_rgba = env.model.geom_rgba.copy()
        geom_rgba[1:, :3] = np.array(color)
        env.model.geom_rgba[:] = geom_rgba

    elif agent_name == 'ant':
        bt2t = env.bt2t
        env.change_task({
            'base_task':     bt2t[task_name],
            'specification': spec,
            'color':         color,
        })
        env.reset()

    elif agent_name == 'cheetah':
        idx = env.bt2t[task_name]
        # _task is read by reset_model(), must be set before env.reset()
        env._task = {'base_task': idx, 'specification': spec}
        env.change_task({'base_task': idx, 'specification': spec, 'color': color})
        env.reset()


# ════════════════════════════════════════════════════════════════════════════
# Action generation
# ════════════════════════════════════════════════════════════════════════════
def get_action(agent_name, task_name, i, n_steps, action_dim):
    t     = i / n_steps
    phase = 2 * np.pi * t * 3

    if agent_name in ('walker2d', 'hopper'):
        if task_name in ('forward_vel', 'goal_front'):
            a = np.array([ 0.5*np.sin(phase),  0.5*np.sin(phase+np.pi),  0.3,
                           0.5*np.sin(phase+np.pi),  0.5*np.sin(phase),  0.3])
        else:
            a = np.array([-0.5*np.sin(phase), -0.5*np.sin(phase+np.pi), -0.3,
                          -0.5*np.sin(phase+np.pi), -0.5*np.sin(phase), -0.3])

    elif agent_name == 'ant':
        if task_name in ('velocity_right', 'goal_right'):
            a = np.array([ 0.5*np.sin(phase),  0.5*np.sin(phase+np.pi),
                           0.5*np.sin(phase+np.pi), 0.5*np.sin(phase),
                           0.3, 0.3, 0.3, 0.3])
        else:
            a = np.array([-0.5*np.sin(phase), -0.5*np.sin(phase+np.pi),
                          -0.5*np.sin(phase+np.pi), -0.5*np.sin(phase),
                          -0.3, -0.3, -0.3, -0.3])

    elif agent_name == 'cheetah':
        if task_name in ('velocity_forward', 'goal_forward'):
            a = np.array([ 0.5, -0.5,  0.5, -0.5,  0.5,  0.5])
        else:
            a = np.array([-0.5,  0.5, -0.5,  0.5, -0.5, -0.5])

    return a[:action_dim]


# ════════════════════════════════════════════════════════════════════════════
# Render & save frames
# ════════════════════════════════════════════════════════════════════════════
def render_and_save(agent_name, env, cfg, task_name, acfg,
                    n_steps=N_STEPS, n_frames=N_FRAMES, img_size=IMG_SIZE):
    set_task(agent_name, env, cfg, task_name, acfg)
    action_dim  = env.action_space.shape[0]
    capture_idx = set(np.linspace(0, n_steps - 1, n_frames, dtype=int))

    out_dir = os.path.join(OUTPUT_BASE, agent_name, task_name)
    os.makedirs(out_dir, exist_ok=True)

    saved_frames = []
    frame_num    = 0

    for i in range(n_steps):
        action = get_action(agent_name, task_name, i, n_steps, action_dim)
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

    print(f"    [{task_name}] saved {frame_num} frames → {out_dir}/")
    return saved_frames


# ════════════════════════════════════════════════════════════════════════════
# Composite preview figure
# ════════════════════════════════════════════════════════════════════════════
def make_composite(agent_name, acfg, all_frames, img_size=IMG_SIZE):
    variants = acfg['task_variants']
    labels   = acfg['task_labels']
    n = len(variants)

    fig, axes = plt.subplots(1, n, figsize=(n * 2.5, 3.2))
    fig.patch.set_facecolor('#c8c8c8')

    for ax, task_name, frames in zip(axes, variants, all_frames):
        frame = frames[len(frames) // 2]
        ax.imshow(frame)
        ax.set_xlabel(labels[task_name], fontsize=8, labelpad=4)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    title_map = {
        'walker2d': r'Walker2d~4\text{-}Task',
        'hopper':   r'Hopper~4\text{-}Task',
        'ant':      r'Ant~4\text{-}Task',
        'cheetah':  r'HalfCheetah~4\text{-}Task',
    }
    fig.suptitle(
        rf'Four base tasks in the benchmark environment $\it{{{title_map[agent_name]}}}$.',
        fontsize=9, y=0.02, ha='left', x=0.01, fontstyle='italic')

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    out_png = f'{agent_name}_4task_visualization.png'
    out_pdf = f'{agent_name}_4task_visualization.pdf'
    fig.savefig(out_png, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    fig.savefig(out_pdf, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"  Composite: {out_png}  {out_pdf}")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════
def main(agents_to_run=None):
    if agents_to_run is None:
        agents_to_run = list(AGENTS.keys())

    for agent_name in agents_to_run:
        acfg = AGENTS[agent_name]
        print(f"\n{'='*60}")
        print(f"Agent: {agent_name}")
        print(f"{'='*60}")

        env, cfg = build_env(agent_name, acfg)
        all_frames = []

        for task_name in acfg['task_variants']:
            print(f"  Rendering: {task_name}  ({N_STEPS} steps → {N_FRAMES} frames)...")
            try:
                frames = render_and_save(agent_name, env, cfg, task_name, acfg)
            except Exception as e:
                print(f"    WARNING: {e} — using blank frames")
                frames = [np.ones((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8) * 200] * N_FRAMES
            all_frames.append(frames)

        make_composite(agent_name, acfg, all_frames)


if __name__ == '__main__':
    # Optionally pass agent names as args: python visualize_all_agents.py walker2d hopper
    requested = sys.argv[1:] if len(sys.argv) > 1 else None
    main(agents_to_run=requested)
