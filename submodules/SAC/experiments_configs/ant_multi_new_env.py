import numpy as np

pi = 3.141592

# Ant multi-task environment config — mirrors half_cheetah_multi_env.py exactly.
#
# Task vector layout (task_dim = max(tasks.values())+1 = 4):
#   task[0] = goal_front    — target x-position (positive, ant moves forward)
#   task[1] = goal_back     — target x-position (negative, ant moves backward)
#   task[2] = forward_vel   — target x-velocity (positive)
#   task[3] = backward_vel  — target x-velocity (negative)
#
# Observation layout (31-dim):
#   qpos[1:]    : 14 dims  (skip global x-pos; y, z, quat[4], joints[8])
#   qvel        : 14 dims  (vx, vy, vz, wx, wy, wz, joint-vels[8])
#   torso_com   :  3 dims  (centre-of-mass in world frame)

config = dict(
    epochs=3000,
    max_traj_len=500,
    memory_size=1e6,
    batch_size_memory=256,
    batch_size=20,
    policy_update_steps=2048,
    gamma=0.99,
    alpha=0.2,
    lr=3e-4,
    reward_scale=1,

    # Task specification ranges
    max_goal=[0.2, 10],      # goal-tracking range (metres)
    max_vel=[0.0, 2.5],      # velocity range (m/s)

    env='ant_multi_new',
    experiment_name='ant_multitask_cheetah_style',
    task_dim=4,              # = max(tasks.values()) + 1

    hidden_layers_actor=[300, 300, 300, 300],
    hidden_layers_critic=[300, 300, 300, 300],

    save_after_episodes=5,
    plot_every=10,

    # Task index assignment — identical structure to cheetah config
    tasks=dict(
        forward_vel=2,
        backward_vel=3,
        goal_front=0,
        goal_back=1,
    ),

    curriculum=dict(
        max_vel=200,
        change_tasks_after=[100, 200, 300],
        changes_per_trajectory=[2, 3, 6],
        max_steps_epochs=[1],
        max_steps=[1000],
        random_initialization=1,
    ),

    random_restart_after=1,
)
