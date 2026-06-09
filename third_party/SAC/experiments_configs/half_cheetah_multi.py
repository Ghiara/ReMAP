from third_party.SAC.sac_envs.walker import WalkerGoal
from third_party.SAC.sac_envs.hopper import HopperGoal
import numpy as np

pi = 3.141592

config = dict(
    epochs = 3000,
    max_traj_len = 500,
    memory_size = 1e+6,
    batch_size_memory = 256,
    batch_size = 20,
    policy_update_steps = 2048,
    gamma = 0.99,
    alpha = 0.2,
    lr = 3e-4,
    reward_scale = 1,

    max_goal = [0.5, 10.0],
    max_jump = [1.5, 3.0],
    max_rot = [pi / 6.0, pi / 2.0],
    max_vel = [0.5, 2.5],
    max_rot_vel = [2.0 * pi, 4.0 * pi],

    env = 'half_cheetah_multi',
    experiment_name = 'cheetah_multitask_new_config_v0_run2',
    task_dim = 5,

    hidden_layers_actor = [300, 300, 300, 300],
    hidden_layers_critic = [300, 300, 300, 300],

    save_after_episodes = 5,
    plot_every = 10,

    reward_params = dict(
        forward_balance_target=0.0,
        forward_balance_weight_goal=0.40,
        forward_balance_weight_vel=0.32,
        forward_balance_soft_limit=0.16,
        forward_balance_hard_limit=0.40,
        forward_balance_hard_penalty=0.80,
        forward_pitch_rate_weight=0.02,
        forward_balance_bonus=0.10,
        forward_balance_bonus_window=0.08,
    ),

    task_sampling_weights = dict(
        goal_front=1.5,
        goal_back=1.0,
        forward_vel=1.5,
        backward_vel=1.0,
    ),

    tasks = dict(
        forward_vel=2,
        backward_vel=3,
        goal_front=0,
        goal_back=1,
    ),

    curriculum = dict(
        max_vel=200,
        change_tasks_after=[100, 200, 300],
        changes_per_trajectory=[2, 3, 6],
        max_steps_epochs=[1],
        max_steps=[1000],
        random_initialization=1,
        difficulty_steps=[1, 200, 500, 900, 1400],
        difficulty_scales=[0.45, 0.60, 0.75, 0.90, 1.00],
    ),

    random_restart_after = 1,

    # pretrained = dict(path='/home/ubuntu/juan/Meta-RL/experiments_transfer_function/new_cheetah_training/half_cheetah_21_06',
    #                   file_name='epoch_1700')
)
