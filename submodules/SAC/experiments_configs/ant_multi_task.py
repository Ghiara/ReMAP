import numpy as np

pi = 3.141592

# Ant multi-task low-level policy config
# Mirrors new_cheetah_training_server1_diff_taskid structure exactly.
#
# task vector layout (task_dim = max(tasks.values())+1 = 4):
#   task[0] = goal_left  (negative x position, e.g. -1 ~ -10 m)
#   task[1] = goal_right (positive x position, e.g. +1 ~ +10 m)
#   task[2] = velocity_left  (negative x velocity, e.g. -1 ~ -2.5 m/s)
#   task[3] = velocity_right (positive x velocity, e.g. +1 ~ +2.5 m/s)
#
# obs layout from ant_multi_old.py: qpos(15) + qvel(14) + torso_com(3) = 32-dim
#   obs[0]  = torso x-position (qpos[0])
#   obs[15] = torso x-velocity (qvel[0])
#   obs[29] = torso_com x  (get_body_com("torso")[0])  ← used for obs mapping

config = dict(
    epochs=30000,
    max_traj_len=500,
    memory_size=1e6,
    batch_size_memory=256,
    batch_size=50,
    policy_update_steps=2048,
    gamma=0.99,
    alpha=0.2,
    lr=3e-4,
    reward_scale=1,

    # Task specification ranges (aligned with toy1d env and cheetah)
    # toy velocity_x: [0.5, 3.0], toy pos_x: [1.0, 25.0]
    max_goal=[1.0, 10.0],       # goal tracking range (meters)
    max_jump=[1.5, 3.0],
    max_rot=[pi / 6., pi / 2.],
    max_vel=[1.0, 2.5],         # velocity range (m/s)
    max_rot_vel=[2. * pi, 4. * pi],

    env='ant_multi',
    experiment_name='ant_multitask_vel_goal',

    # task_dim = max(tasks.values()) + 1 = 4 (computed automatically in train script)
    task_dim=4,

    hidden_layers_actor=[300, 300, 300, 300],
    hidden_layers_critic=[300, 300, 300, 300],

    save_after_episodes=5,
    plot_every=10,

    tasks=dict(
        goal_left=0,
        goal_right=1,
        velocity_left=2,
        velocity_right=3,
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

    pretrained=None,
)
