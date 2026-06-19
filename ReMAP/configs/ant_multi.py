import numpy as np

pi = 3.141592

config = dict(
    # Total number of training epochs
    epochs=3000,

    # Initial maximum number of steps per trajectory (dynamically overridden by curriculum.max_steps)
    # Start with shorter trajectories so the ant learns basic stable control first, then gradually extend the horizon
    max_traj_len=200,

    # Replay buffer capacity (larger values reduce forgetting of old distributions, but use more memory)
    memory_size=1e6,

    # Number of samples drawn from the replay buffer for each update
    batch_size_memory=256,

    # Number of trajectories sampled per epoch (environment rollouts)
    batch_size=20,

    # Number of actor/critic update steps per epoch
    policy_update_steps=2048,

    # Discount factor; the closer to 1, the more long-term return is emphasized
    gamma=0.99,

    # SAC temperature coefficient controlling entropy regularization strength (degree of exploration)
    alpha=0.2,

    # Learning rate (shared by actor and critic)
    lr=3e-4,

    # Reward scaling factor (passed to SAC)
    reward_scale=1,

    # Position target range [min, max] (in meters), used for goal_front / goal_back
    max_goal=[1.0, 10.0],

    # Jump task range (not mainly used in the current four-task training)
    max_jump=[1.5, 3.0],

    # Posture angle range (not mainly used in the current four-task training)
    max_rot=[pi / 6., pi / 2.],

    # Velocity target range [min, max] (in m/s), used for forward_vel / backward_vel
    max_vel=[0.4, 2.5],

    # Angular velocity range (not mainly used in the current four-task training)
    max_rot_vel=[2. * pi, 4. * pi],

    # Environment name (used by the training script to instantiate AntMulti)
    env='ant_multi',

    # Experiment name (output directory name)
    experiment_name='ant_multi_new_config_v4_BASH_TEST',

    # Upper bound of the task vector dimension (usually equal to max(tasks.values()) + 1)
    task_dim=4,

    # Actor network hidden layer structure
    hidden_layers_actor=[300, 300, 300, 300],

    # Critic network hidden layer structure
    hidden_layers_critic=[300, 300, 300, 300],

    # Save model and training curves every N epochs
    save_after_episodes=10,

    # Logging plot interval (used by the visualization module)
    plot_every=10,

    # Mapping from task names to task-vector indices
    # goal_front / goal_back: position target tracking
    # forward_vel / backward_vel: velocity target tracking
    tasks=dict(
        goal_front=0,
        goal_back=1,
        forward_vel=2,
        backward_vel=3,
    ),

    # Stable low-level control for Ant: start resets from a normal standing height to reduce explosive initial contacts
    initial_torso_height=0.72,
    reset_xy_noise=0.03,
    reset_joint_noise=0.03,
    reset_height_noise=0.01,
    reset_velocity_noise=0.03,

    # Sample the four tasks in a balanced way, with slightly higher weight on backward tasks to avoid undertraining goal_back
    task_sample_weights=dict(
        goal_front=1.0,
        goal_back=1.25,
        forward_vel=1.0,
        backward_vel=1.25,
    ),

    # Initial x-position sampling range after enabling random_reset
    random_reset_pos_range=[-8.0, 8.0],

    # Initial x-velocity sampling range after enabling random_reset
    random_reset_vel_range=[-0.5, 0.5],

    curriculum=dict(
        # Epoch threshold for unlocking full difficulty of velocity tasks
        # In the training code: when episode < max_vel, the velocity task is simplified (target velocity is halved)
        max_vel=150,

        # Stage boundaries (in epochs) for the within-trajectory task-switching curriculum
        # After exceeding a threshold, the corresponding changes_per_trajectory value is used
        change_tasks_after=[200, 450, 800],

        # Number of task switches within each trajectory
        # 0 means one task for the entire trajectory; larger values make training harder but improve switching adaptation
        changes_per_trajectory=[0, 1, 2],

        # Stage boundaries (in epochs) for the trajectory length curriculum
        # In the training loop, switching happens only when episode > threshold, so:
        # 1-150: 200 steps
        # 151-350: 300 steps
        # 351-700: 500 steps
        # 701+: 700 steps
        max_steps_epochs=[150, 400, 800],

        # max_traj_len for each stage (only increases, never decreases)
        # First let the ant learn stable locomotion over short horizons, then gradually increase long-horizon credit assignment difficulty
        max_steps=[250, 350, 500],

        # Enable random_reset starting from this epoch
        # The current value is large, effectively keeping random_reset disabled during most of training
        random_initialization=1200,
    ),

    # Legacy retained setting (mostly unused in the current training loop)
    random_restart_after=1,

    # Pretrained model entry point (None means training from scratch)
    use_termination_after=80,
    termination_ramp_epochs=320,

    # More strongly encourage four-foot ground contact and torso stability; do not reward jumping high to game progress
    healthy_reward=0.55,
    termination_penalty=-8.0,
    target_torso_height=0.70,
    min_torso_height=0.34,
    max_torso_height=1.05,
    min_upright=0.55,
    max_lateral_offset=1.20,
    max_vertical_velocity=1.80,
    max_pitch_rate=7.0,
    max_state_abs=100.0,
    height_penalty_weight=1.0,
    upright_penalty_weight=1.0,
    vertical_velocity_penalty_weight=0.16,
    pitch_rate_penalty_weight=0.035,
    action_smooth_penalty_weight=0.006,
    goal_progress_weight=0.8,
    goal_ctrl_cost=8e-3,
    vel_ctrl_cost=8e-3,

    pretrained=None,
)
