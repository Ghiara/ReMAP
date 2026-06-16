from third_party.SAC.sac_envs.walker import WalkerGoal
from third_party.SAC.sac_envs.hopper import HopperGoal
import numpy as np

pi = 3.141592

config = dict(
    # Total number of training epochs
    epochs = 3000,

    # Initial maximum number of steps per trajectory (dynamically overridden by curriculum.max_steps)
    max_traj_len = 300,

    # Replay buffer capacity (larger values reduce forgetting of old distributions, but use more memory)
    memory_size = 1e+6,

    # Number of trajectories sampled per epoch (environment rollouts)
    batch_size = 20,

    # Number of samples drawn from the replay buffer for each update
    batch_size_memory = 128,

    # Number of actor/critic update steps per epoch
    policy_update_steps = 1024,

    # Discount factor; the closer to 1, the more long-term return is emphasized
    gamma = 0.99,

    # SAC temperature coefficient controlling the policy entropy term weight
    alpha = 0.2,

    # Learning rate (shared by actor and critic)
    lr = 3e-4,

    # Reward scaling factor (passed to SAC)
    reward_scale = 1,

    # Position target range [min, max] (in meters), used for goal_front / goal_back
    max_goal = [2.0, 15.0],

    # Velocity target range [min, max] (in m/s), used for forward_vel / backward_vel
    max_vel = [0.5, 4.0],

    # Angle range for standing/posture tasks (not mainly used in the current four-task training)
    max_rot = [pi / 6., pi / 2.],

    # Rotational angular velocity range (not mainly used in the current four-task training)
    max_rot_vel = [2. * pi, 4. * pi],

    # Environment name (used by the training script to instantiate WalkerMulti)
    env = 'walker_multi',

    # Experiment name (output directory and log prefix)
    experiment_name = 'walker_multi_new_config_v0_BASH_TEST',

    # Upper bound of the task vector dimension (generally equal to max(tasks.values()) + 1)
    task_dim = 5,

    # Mapping from task names to task-vector indices
    # goal_front / goal_back: position tracking tasks
    # forward_vel / backward_vel: velocity tracking tasks
    tasks = dict(
                forward_vel=2, backward_vel=3,
                goal_front=0, goal_back=1,
                 ),

    # Actor network hidden layer structure
    hidden_layers_actor = [300,300,300,300],

    # Critic network hidden layer structure
    hidden_layers_critic = [300,300,300,300],

    # Save model and training curves every N epochs
    save_after_episodes = 20,

    # Logging plot interval (used by the visualization module)
    plot_every = 100,

    # Initial x-position sampling range after enabling random_reset
    random_reset_pos_range = [-10.0, 10.0],

    # Initial x-velocity sampling range after enabling random_reset
    random_reset_vel_range = [-0.5, 0.5],

    curriculum = dict(
        # Epoch threshold for unlocking full difficulty of velocity tasks
        # In the training code: when episode < max_vel, the velocity task is simplified (target velocity is halved)
        max_vel=150,

        # Stage boundaries (in epochs) for the within-trajectory task-switching curriculum
        # After exceeding a threshold, the corresponding changes_per_trajectory value is used
        change_tasks_after = [150,250,400],

        # Number of task switches within each trajectory
        # 0 means one task for the entire trajectory; larger values mean more frequent switching and harder training
        changes_per_trajectory = [0,2,4],

        # Stage boundaries (in epochs) for the maximum trajectory length curriculum
        max_steps_epochs = [150,350,700],

        # max_traj_len used in each stage
        # Note: this affects whether the traj_len curve you observe becomes longer or shorter
        max_steps = [300,500,700],

        # Enable random_reset (random initial state) starting from this epoch
        random_initialization = 30000,
    ),

)
