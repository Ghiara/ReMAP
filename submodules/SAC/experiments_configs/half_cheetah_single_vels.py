from sac_envs.walker import WalkerGoal
from sac_envs.hopper import HopperGoal
import numpy as np

pi = 3.141592

config = dict(
    # epochs = 30000,
    # max_traj_len = 300,
    # memory_size = 1e+5,
    # batch_size = 256,
    # gamma = 0.99,
    # alpha = 0.2,
    # lr = 3e-5,
    # reward_scale = 1,

    #TEST NEW ARCHITECTURE
    epochs = 30000,
    max_traj_len = 400,
    memory_size = 1e+6,
    batch_size = 64,
    batch_size_memory = 256,
    policy_update_steps = 1024, 
    gamma = 0.99,
    alpha = 0.2,
    lr = 3e-5,
    reward_scale = 5,

    max_goal = [2, 10],
    max_vel = [0.5, 3.0],
 

    env = 'cheetah_velocity_single',
    # experiment_name = 'Walker_deeper_really_change_task',
    experiment_name = 'cheetah_velocity_single_low_level(forward_vel)',
    task_dim = 1,

    hidden_layers_actor = [300,300,300],
    hidden_layers_critic = [300,300,300],

    save_after_episodes = 10,


    #following lines are added by chongjin as a test
    plot_every = 10,

    tasks = dict(
                
                forward_vel=1,
                #goal_right=1, goal_up=0, goal_down=0
                 ), 
    curriculum = dict(
        max_vel=3.0,
        change_tasks_after = [300,400],
        changes_per_trajectory = [2,4],
        max_steps_epochs = [200,350],             #change trajectory length
        max_steps = [450,600],                    #change trajectory length
        random_initialization = 20000,
    ),

    use_termination_after = 40


)