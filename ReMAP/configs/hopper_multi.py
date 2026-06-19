from third_party.SAC.sac_envs.walker import WalkerGoal
from third_party.SAC.sac_envs.hopper import HopperGoal
import numpy as np

pi = 3.141592

config = dict(
    # Total number of training epochs
    epochs = 3000,

    # Initial maximum number of steps per trajectory (dynamically overridden by curriculum.max_steps)
    max_traj_len = 500,

    # Replay buffer capacity (larger values reduce forgetting of old distributions, but use more memory)
    memory_size = 1e+6,

    # Number of trajectories sampled per epoch (environment rollouts)
    batch_size = 20,

    # Number of samples drawn from the replay buffer for each gradient update
    batch_size_memory = 256,

    # Number of policy/value network update steps per epoch
    policy_update_steps = 2048,

    # Discount factor; the closer to 1, the more long-term return is emphasized
    gamma = 0.99,

    # SAC temperature coefficient controlling the exploration entropy weight
    alpha = 0.2,

    # Learning rate (shared by actor and critic)
    lr = 3e-4,

    # Reward scaling factor (passed to SAC)
    reward_scale = 1,

    # Target-tracking task magnitude range [min, max], in meters (used by goal_front/goal_back)
    max_goal = [2.0, 15.0],

    # Jump task range (not mainly used in the current four-task training)
    max_jump = [1.5, 3.],

    # Rotation posture task range (not mainly used in the current four-task training)
    max_rot = [pi / 6., pi / 2.],

    # Velocity-tracking task magnitude range [min, max], in m/s (used by forward_vel/backward_vel)
    max_vel = [0.5, 4.0],

    # Rotational angular velocity task range (not mainly used in the current four-task training)
    max_rot_vel = [2. * pi, 4. * pi],

    # Environment name (used by the training script to instantiate HopperMulti)
    env = 'hopper_multi',

    # Experiment name for this run (output directory name)
    experiment_name = 'hopper_multi_new_config_v3_BASH_TEST',

    # Upper bound of the task vector dimension (usually equal to max(tasks.values()) + 1)
    task_dim = 5,

    # Actor hidden layer structure
    hidden_layers_actor = [300,300,300,300],

    # Critic hidden layer structure
    hidden_layers_critic = [300,300,300,300],

    # Save model and logs every N epochs
    save_after_episodes = 10,

    # Plotting/visualization interval (used by the logging module)
    plot_every = 5,

    reward_params = dict(
        velocity_ctrl_weight = 1.2e-3,
        goal_ang_limit = 0.4,
        forward_ang_limit_base = 0.44,
        forward_ang_limit_bonus = 0.12,
        backward_ang_limit_base = 0.50,
        backward_ang_limit_bonus = 0.10,
    ),

    task_sampling_weights = dict(
        forward_vel = 2.4,
        backward_vel = 2.4,
        goal_front = 1.0,
        goal_back = 1.0,
    ),

    # Mapping from tasks to task-vector indices
    # goal_front / goal_back: position targets
    # forward_vel / backward_vel: velocity targets
    tasks = dict(
                forward_vel=2, backward_vel=3,
                goal_front=0, goal_back=1,
                ),

    # Initial x-position sampling range after enabling random_reset
    random_reset_pos_range = [-8.0, 8.0],

    # Initial x-velocity sampling range after enabling random_reset
    random_reset_vel_range = [-0.5, 0.5],

    curriculum = dict(
        # Epoch threshold for unlocking full difficulty of velocity tasks
        # In the training code: when episode < max_vel, the velocity task is simplified (target velocity is halved)
        max_vel=250,

        # Stage boundaries (in epochs) for the within-trajectory task-switching curriculum
        # After exceeding a threshold, the corresponding changes_per_trajectory value is used
        change_tasks_after = [200, 350, 550],

        # Approximate number of task switches within each trajectory
        # 0 means a single task for the whole trajectory; larger values mean more frequent switching and harder training
        changes_per_trajectory = [0, 1, 3],

        # Stage boundaries (in epochs) for the maximum trajectory length curriculum
        max_steps_epochs = [150,350,700],

        # max_traj_len used in each corresponding stage
        # Current setting: start with 500, then increase to 1000 and keep training with long trajectories
        max_steps = [500,1000,1000],

        # Enable random_reset (random initial position/velocity) starting from this epoch
        random_initialization = 35000,
    ),


)
