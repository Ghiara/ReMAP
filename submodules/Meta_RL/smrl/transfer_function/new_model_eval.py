
#----------------------------------------------------------------------------------------


#  replicating the inference module reuse without hierarchical policy
"""
This file contain shte function ``model_evaluation()`` which visualizes
the training progress of trained agents as well of their final models - both
in terms of exemplary trajectories and latent space plots.

Author(s): 
    Julius Durmann
Contact: 
    julius.durmann@tum.de
Date: 
    2023-04-06
"""


# from transfer_configs.half_cheetah_config import config

# from experiments_configs.walker_multi import config as walker_config

import os
from typing import Tuple
from pathlib import Path
import json
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import gym
import numpy as np
import pandas as pd
import torch

from smrl.experiment.analysis import load_results
from smrl.utility.console_strings import bold, italic
from smrl.utility.ops import CustomJsonEncoder
# from smrl.policies.exploration import RandomMemoryPolicy, RandomPolicy

from mrl_analysis.utility.interfaces import MdpEncoder, MetaRLPolicy, MetaQFunction
from mrl_analysis.trajectory_rollout.encoding import encodings_from_encoder
from mrl_analysis.plots.latent_space import plot_latent_space, plot_latent_space_axis_over_task
from mrl_analysis.video.video_creator import VideoCreator
from mrl_analysis.utility.data_smoothing import smooth_plot, smooth_fill_between

from main_config import HARDCODED
from smrl.policies.meta_policy import PretrainedCheetah
from smrl.vae.encoder_decorators.io_modification import InputOutputDecorator
# from meta_envs.mujoco.cheetah import HalfCheetahEnvExternalTask
from meta_envs.toy_goal.toy_1d import Toy1D, Toy1dVelocity
from specific.encoder_transfer import map_cheetah_to_toy1d, map_ant_to_toy1d_velocity, map_cheetah_to_toy1d_velocity
from sac_envs.half_cheetah_multi import HalfCheetahMixtureEnv
from sac_envs.walker_multi import WalkerMulti
from sac_envs.hopper_multi import HopperMulti
from sac_envs.ant_multi_old import AntMulti
from smrl.transfer_function.new_PathCollector import TrajectoryGeneratorWithTransferFunction
from collections import OrderedDict
# from smrl.trainers.transfer_function import TransferFunction
from model import PolicyNetwork as TransferFunction

from mrl_analysis.plots.plot_settings import *

from mrl_analysis.video.video_creator_replay import VideoCreatorReplay

# HARDCODED = True

# Parameters
video_creator = VideoCreator()
video_creator_replay = VideoCreatorReplay(fps=30, width=800, height=600)
video_creator.fps = 30

DEVICE = 'cpu'
max_path_l = 200

def get_complex_agent(env, complex_agent_config):
    pretrained = complex_agent_config['experiments_repo']+complex_agent_config['experiment_name']+f"/models/policy_model/epoch_{complex_agent_config['epoch']}.pth"
    #extarct the number of states and actions from env observation_space and action_space
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    #Retrieves the action bounds (minimum and maximum values) from the environment's action space.
    action_bounds = [env.action_space.low[0], env.action_space.high[0]]
    # Creates a TransferFunction object, passing the extracted environment details and the path to the pretrained model.
    # This is a pytorch nn.Module == low-level policy
    transfer_function = TransferFunction(
        n_states=n_states,
        n_actions=n_actions,
        action_bounds=action_bounds,
        pretrained=pretrained
        )
    #Moves the TransferFunction object to the specified device (cpu in this case).
    transfer_function.to(DEVICE)
    return transfer_function

def model_evaluation(
    path_to_data: str, 
    save_dir: str, 
    config,
    itr: int = None, 
    env: gym.Env = None,
    policy: MetaRLPolicy = None,
    figure_size: Tuple[int,int] = (20, 10),
    trajectory_1d: bool = True,
    trajectory_2d: bool = False,
    multithreading = True,
    n_trajectories: int = 50,
    color_by: str = 'rewards',
    specification_keyword: str = 'goal_velocity',  # How tasks are named
    create_video: bool = True,
    show_plots: bool = False,
):
    """Creates evaluation figures from a directory of training data.

    Parameters
    ----------
    path_to_data : str
        Training directory. Must contain
            Weights: 'params.pkl' or 'itr_<nr.>.pkl'
            Progress file: 'progress.csv'
            Config file: 'variant.json'
    save_dir : str
        Directory into which plots are saved
    itr : int, optional
        Iteration of weights, by default None
    env : gym.Env, optional
        Environment (if the environment in the configuration file should 
        not be used), by default None
    policy : MetaRLPolicy, optional
        Policy for rollouts (if the trained policy should not be used),
        by default None
    figure_size : Tuple[int,int], optional
        Size of the saved figures, by default (20, 10)
    trajectory_1d : bool, optional
        Set to True to create 1d-trajectory plots (first dimension), by default True
    trajectory_2d : bool, optional
        Set to True to create 2d-trajectory plots (first two dimensions), by default False
    multithreading : bool, optional
        Set to True to use ray for rollouts, by default True
    n_trajectories : int, optional
        Number of trajectories for evaluation, by default 50
    color_by : str, optional
        Color coding for task-encoding plots, choose one of
        |'observations'|'rewards'|'actions'|'next_observations'|'terminals'|, 
        by default 'rewards'
    specification_keyword : str, optional
        Keyword in task dictionaries (dependent on environment) which belongs
        to the task feature, by default 'goal'
    show_plots : bool, optional
        Set to True to print plots to screen, by default False
    """
    
    def save_figure(fig: Figure, save_as: Path):
        # fig.set_size_inches(*figure_size)
        fig.tight_layout()
        fig.savefig(save_as.with_suffix('.svg'))
        fig.savefig(save_as.with_suffix('.png'))
        fig.savefig(save_as.with_suffix('.pdf'))

    # os.environ['MULTITHREADING'] = "True" if multithreading else "False"

    # Load data
    path_to_data = Path(path_to_data)
    #load the .pkl files and the progress.csv file
    result_dict = load_results(path_to_data, itr=itr)
    name = path_to_data.name
    progress: pd.DataFrame = result_dict['progress']
    max_path_length = result_dict['config']['algorithm_kwargs']['max_path_length']
    max_path_length = max_path_l

    # Initialize directory for storing images
    dir = Path(save_dir).joinpath(name)
    dir.mkdir(exist_ok=True, parents=True)

    print(italic("\nmax_path_length: " + str(max_path_length)))
    print(bold("\nEvaluation for " + italic(name) + f", Epoch {result_dict['epoch']}"))

    # Save config file to directory
    with open(dir.joinpath("config.json"), mode="w") as file:
        json.dump(result_dict['config'], file, indent=4, cls=CustomJsonEncoder)
    progress.to_csv(dir.joinpath("progress.csv"), sep=",")

    # Figures

    # Policy training
    try:
        print("Policy training plots ...")

        x = progress['trainer/Policy trainer/num train calls']

        fig, axs = plt.subplots(1, figsize=figure_size)
        smooth_fill_between(axs, x, progress['eval/Returns Min'], progress['eval/Returns Max'], color='lightgrey', alpha=0.3)
        smooth_plot(axs, x, progress['eval/Average Returns'], color=c_avg, label='Average returns')
        smooth_plot(axs, x, progress['eval/Returns Max'], color=c_max, label='Maximum returns', linestyle='dashed', plot_original=False)
        smooth_plot(axs, x, progress['eval/Returns Min'], color=c_min, label='Minimum returns', linestyle='dashed', plot_original=False)
        axs.set_xlabel('Train steps (policy)')
        axs.legend()
        save_figure(fig, dir.joinpath("returns"))

        fig, axs = plt.subplots(1, figsize=figure_size)
        smooth_plot(axs, x, progress['trainer/Policy trainer/Log Pis Mean'], label="Mean logprobs")
        smooth_plot(axs, x, progress['trainer/Policy trainer/Log Pis Min'], label="Minimum logprobs")
        smooth_plot(axs, x, progress['trainer/Policy trainer/Log Pis Max'], label="Maximum logprobs")
        axs.legend()
        axs.set_xlabel('Train steps (policy)')
        save_figure(fig, dir.joinpath("Action logprobs"))

        fig, axs = plt.subplots(1, figsize=figure_size)
        smooth_plot(axs, x, progress['trainer/Policy trainer/policy/normal/std Mean'], label="Standard deviation (mean)")
        smooth_plot(axs, x, progress['trainer/Policy trainer/policy/normal/std Min'], label="Standard deviation (min)")
        smooth_plot(axs, x, progress['trainer/Policy trainer/policy/normal/std Max'], label="Standard deviation (max)")
        axs.set_xlabel('Train steps (policy)')
        axs.legend()
        save_figure(fig, dir.joinpath("Action variation"))

        fig, axs = plt.subplots(1, figsize=figure_size)
        smooth_plot(axs, x, progress['trainer/Policy trainer/Q Targets Mean'], label="Mean Q")
        smooth_fill_between(axs, x, 
                        progress['trainer/Policy trainer/Q Targets Mean'] - progress['trainer/Policy trainer/Q Targets Std'],
                        progress['trainer/Policy trainer/Q Targets Mean'] + progress['trainer/Policy trainer/Q Targets Std'],
                        alpha=0.3,
                        color=c_avg,
                        )
        smooth_plot(axs, x, progress['trainer/Policy trainer/Q Targets Min'], label="Min Q", color=c_min)
        smooth_plot(axs, x, progress['trainer/Policy trainer/Q Targets Max'], label="Max Q", color=c_max)
        axs.set_xlabel('Train steps (policy)')
        axs.legend()
        save_figure(fig, dir.joinpath("Q-target values"))

        fig, axs = plt.subplots(1, figsize=figure_size)
        smooth_plot(axs, x, progress['trainer/Policy trainer/QF1 Loss'], label="QF1 loss")
        smooth_plot(axs, x, progress['trainer/Policy trainer/QF2 Loss'], label="QF2 loss")
        axs.set_xlabel('Train steps (policy)')
        axs.legend()
        axs.set_yscale('log')
        save_figure(fig, dir.joinpath("qfunction_loss"))

        fig, axs = plt.subplots(1, figsize=figure_size)
        smooth_plot(axs, x, progress['trainer/Policy trainer/Policy Loss'], label="Policy loss")
        axs.set_xlabel('Train steps (policy)')
        axs.legend()
        save_figure(fig, dir.joinpath("policy_loss"))
    except KeyError:
        print("(skipped)")


    # Inference training
    try:
        print("Inference training plots ...")
        x = progress['trainer/Inference trainer/num train calls']

        fig, axs = plt.subplots(1, figsize=figure_size)
        smooth_plot(axs, x, progress['trainer/Inference trainer/elbo'])
        axs.set_xlabel('Train steps (inference)')
        axs.legend(['ELBO'])
        save_figure(fig, dir.joinpath("elbo"))

        fig, axs = plt.subplots(1, figsize=figure_size)
        smooth_plot(axs, x, progress['trainer/Inference trainer/reward prediction error'], label="Reward prediction error")
        smooth_plot(axs, x, progress['trainer/Inference trainer/state prediction error'], label="State prediction error")
        axs.set_xlabel('Train steps (inference)')
        axs.set_yscale('log')
        axs.legend()
        save_figure(fig, dir.joinpath("prediction_errors"))
    except KeyError:
        print("(skipped)")


    # Load networks
    encoder: MdpEncoder = result_dict['encoder']
    decoder: MdpEncoder = result_dict['decoder']
    if env is None:
        env = result_dict['eval_env']
    
    if HARDCODED == True:
        # result_dict['eval_policy'] = RandomPolicy(action_dim=1, std=torch.tensor(0.5))
        result_dict['eval_policy'] = PretrainedCheetah()
    if policy is None:
        policy = result_dict['eval_policy']
        from smrl.policies.meta_policy import MakeDeterministic
        if isinstance(policy, MakeDeterministic):
            policy = policy.stochastic_policy

    value_function: MetaQFunction = result_dict['qf1']

    #set the netweorks to eval mode, dropout/batchnorm will be disabled
    encoder.train(False)
    decoder.train(False)
    policy.train(False)
    value_function.train(False)


    # Rollout trajectories     important for the inference module evaluation
    # simple_env = Toy1D(25,25, min_pos=.0, max_pos=10.0, max_action=0.5)
    simple_env = Toy1dVelocity(
        n_train_tasks=25,
        n_eval_tasks=25,
        task_generation_mode='random',
        max_action=0.5,
        max_vel=3.0
    )
    #with load_model func we can align the task space in encoder
    encoder, env, transfer_function = load_model(encoder, config)
    decoder = None
    value_function = None
    # create a instance of TrajectoryGeneratorWithTransferFunction
    # the encoder now i have used has been decorated with map_cheetah_to_toy1d, so it can process the  complex transitions from cheetah env
    tg = TrajectoryGeneratorWithTransferFunction(simple_env, env, policy, encoder, transfer_function, decoder, value_function)
    # tg = TrajectoryGenerator(env, policy, encoder, None, value_function)
    #generate n_trajectories trajectories with max_path_length length
    trajectories = tg.run(n_trajectories, max_path_length)

    #sub goal tracking analysis
    # === Subgoal Tracking Visualization ===
    print("Plotting subgoal tracking with real rollout...")

    subgoal_dir = dir.joinpath("subgoal_tracking")
    subgoal_dir.mkdir(exist_ok=True, parents=True)

    for i, traj in enumerate(trajectories[:10]):  # 可视化前 10 条轨迹
        if 'subgoals' not in traj:
            print(f"Trajectory {i+1} missing subgoal data, skipping...")
            continue

        subgoals = traj['subgoals']
        pre_states = traj['pre_states']
        post_states = traj['post_states']

        # 从真实 rollout 中提取连续的 velocity 轨迹
        # 你之前确定过 velocity 对应 qvel[0]，一般在 obs 的最后部分
        # 为安全起见，这里取 obs 的第一维或打印看具体索引
        real_velocity = traj['observations'][:, 15] if traj['observations'].shape[1] > 15 else traj['observations'][:,0]

        # === 绘图 ===
        plt.figure(figsize=(10, 5))
        #plt.plot(real_velocity, color='black', alpha=0.6, label='Real continuous velocity (rollout)')
        # plt.plot(np.arange(len(real_velocity)), real_velocity, color='black', alpha=0.9, linewidth=2, label='Real continuous velocity (rollout)', zorder=10)
        # 为了对齐：每次 subgoal 执行包含多个 micro-step（比如 5），
        # 我们假设 subgoal[i] 对应 rollout 中的 [i*5:(i+1)*5]
        stride = len(real_velocity) // len(subgoals)
        subgoal_steps = np.arange(0, len(subgoals) * stride, stride)

        np.savetxt(
            subgoal_dir.joinpath(f"subgoal_tracking_traj_{i+1}.csv"),
            np.vstack([subgoal_steps, subgoals, pre_states, post_states, real_velocity[subgoal_steps]]).T,
            delimiter=",",
            header="step,subgoal,pre_state,post_state,real_velocity",
            comments=""
        )


        plt.plot(subgoal_steps, subgoals, 'r-o', label='High-level subgoal (target)', zorder=20)
        plt.plot(subgoal_steps, pre_states, 'b--', label='Pre-state (before micro-steps)', zorder=15)
        plt.plot(subgoal_steps, post_states, 'g--', label='Post-state (after micro-steps)', zorder=15)

        # === 🟥 摔倒检测标记线 ===
        # if 'fall_step' in traj and traj['fall_step'] is not None:
        #     fall_x = traj['fall_step']
        #     plt.axvline(fall_x, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Fall detected')
        #     plt.text(fall_x + 2, np.max(real_velocity) * 0.8, 'FALL', color='red', fontsize=9, rotation=90, alpha=0.8)

        plt.xlabel("Environment step")
        plt.ylabel("Velocity (m/s)")
        plt.title(f"Subgoal Tracking (Trajectory {i+1})")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(subgoal_dir.joinpath(f"subgoal_tracking_traj_{i+1}.png"))
        plt.close()




    # Debugging information
    print("Debugging Information:")
    # Step 1. 检查 observation 和 action 的维度
    print("DEBUG >> Observation shape:", trajectories[0]['observations'].shape)
    print("DEBUG >> Action shape:", trajectories[0]['actions'].shape)

    # Step 2. 检查使用的 policy 类型
    print("DEBUG >> Policy class:", type(policy))

    # Step 3. 检查 transfer function 是否加载了 cheetah 模型
    num_params = sum(p.numel() for p in transfer_function.parameters())
    print("DEBUG >> Transfer function params:", num_params)


    encodings, goals, color_information = [], [], []
    for traj in trajectories:
        for i in range(encoder.context_size, len(traj['observations']), 50):
            encodings.append(traj['encodings'][i][None])
            goals.append(traj['tasks'][i][specification_keyword][None, ...])
            if color_by is not None:
                color_information.append(traj['contexts'][color_by][i, -1])
    encodings = np.concatenate(encodings)
    goals = np.concatenate(goals)
    if color_by is not None:
        color_information = np.concatenate(color_information)
    else:
        color_information = None

    # Save trajectories to csv files
    dir.joinpath("trajectories/").mkdir(parents=True, exist_ok=True)
    for i, traj in enumerate(trajectories):
        for k, v in traj['tasks'][0].items():
            if isinstance(v, np.ndarray):
                traj['task/'+k] = np.vstack([task[k] for task in traj['tasks']])
            else:
                traj['task/'+k] = np.array([task[k] for task in traj['tasks']])
            if traj['task/'+k].ndim < 2:
                traj['task/'+k] = traj['task/'+k][:,None]
        keys, values = [], []
        for key, value in traj.items():
            if isinstance(value, np.ndarray):
                if value.ndim < 2:
                    continue
                keys.extend([key + "." + str(i) for i in range(value.shape[1])])
                values.append(value)

        header = "".join([k + "," for k in keys])[:-1]
        values = np.hstack(values)
        np.savetxt(dir.joinpath("trajectories").joinpath(f"{i+1}.csv"), values, header=header, delimiter=",", comments="")


    # Latent space figures
    print("Latent space figures ...")
    fig, axs = plot_latent_space(encodings[:,:2], goals, size=figure_size[1])
    save_figure(fig, dir.joinpath("latent-space_time-ordered"))

    fig, axs = plot_latent_space_axis_over_task(encodings[:,:2], goals, color_information)
    fig.set_size_inches(*figure_size)
    save_figure(fig, dir.joinpath("tasks-and-encodings_time-ordered"))


    if create_video:
        print("Video ...")
        # video_creator_replay.create_video(
        #     result_dict, video_length=20.0, 
        #     save_as=str(dir.joinpath("video_ant_render.mp4")), 
        #     env_reset_interval=max_path_length, 
        #     width=env.screen_width, height=env.screen_height,
        # )

        # video_creator_replay.create_video_from_trajectories(
        #     env=result_dict['eval_env'],
        #     trajectories=trajectories,
        #     save_as='./evaluation/experiments_thesis/replay/ant_inference_demo.mp4',
        #     show_info=True,
        #     overlay_latent=True,
        #     overlay_goal=True,
        #     overlay_reward=True,
        # )

    # Movements
    if trajectory_1d:
        fig, axs = plt.subplots(2, figsize=(figure_size[0], 2*figure_size[1]))
        dim = 0
        axs[0].set_title("Velocity")
        axs[1].set_title("Latent distribution (mean and variance)")
        # axs[2].set_title("Q-function value")
        # axs[3].set_ylabel("Action distribution")
        # axs[3].set_title("Actions")
        for a in axs:
            a.set_xlabel("Time (in steps)")
        for i, traj in enumerate(trajectories[::10]):
            tasks = [task[specification_keyword][dim] for task in traj['tasks']]
            # change the index here if change the complex agent
            axs[0].plot(range(len(traj['observations'][:,15])), traj['observations'][:,15], c=f"C{i}")
            axs[0].plot(range(len(traj['observations'][:,15])), tasks, c=f"C{i}", linestyle='dashed', alpha=0.5)
            axs[1].fill_between(range(len(traj['latent_mean'])), (traj['latent_mean'] + traj['latent_std'])[:,0], (traj['latent_mean'] - traj['latent_std'])[:,0], color=f"C{i}", alpha=0.2)
            axs[1].plot(range(len(traj['latent_mean'])), traj['latent_mean'][:,0], c=f"C{i}")
            # axs[2].plot(range(len(traj['qvalues'])), traj['qvalues'], c=f"C{i}")
            # axs[3].fill_between(range(len(traj['actions'])), (traj['action_mean'] - traj['action_std'])[:,0], (traj['action_mean'] + traj['action_std'])[:,0], alpha=0.3, color=f"C{i}")
            # axs[3].plot(range(len(traj['actions'])), traj['action_mean'], c=f"C{i}")
            # axs[3].plot(range(len(traj['actions'])), traj['actions'], c=f"C{i}")
        save_figure(fig, dir.joinpath("trajectory-1d"))
    if trajectory_2d:
        fig, axs = plt.subplots(1, 2, figsize=(2*figure_size[1], figure_size[1]))
        dim = 0
        axs[0].set_title("Trajectory")
        axs[1].set_title("Encodings")
        for i, traj in enumerate(trajectories[::10]):
            tasks = [np.atleast_2d(task[specification_keyword]) for task in traj['tasks']]
            tasks = np.concatenate(tasks)
            axs[0].plot(traj['observations'][:,0], traj['observations'][:,1], c=f"C{i}", alpha=0.7)
            axs[0].scatter(tasks[:,0], tasks[:,1], c=f"C{i}")
            axs[1].plot(traj['encodings'][:,0], traj['encodings'][:,1], c=f"C{i}", alpha=0.7)
        save_figure(fig, dir.joinpath("trajectory-2d"))

    # Latent space, randomized
    print("Collecting encodings ...")
    #temporary commented out, because it is not working with the new transfer function
    # encodings, tasks, contexts, trajectories = encodings_from_encoder(
    #     encoder, transfer_function, env, n_trajectories, max_path_length, 
    #     encodings_per_trajectory=25, randomize_samples=True,
    # )
    # goals = np.array([task[specification_keyword] for task in tasks])
    # 如果 tasks 已经是 numpy 数组
    if isinstance(tasks, (list, tuple)):
        goals = np.array(tasks)
    else:
        goals = tasks  # 已经是 np.ndarray 了

    #addtional debug
    min_len = min(len(encodings), len(goals))
    encodings = encodings[:min_len]
    goals = goals[:min_len]




    # if color_by is not None:
    #     color_information = contexts[color_by][:,-1,0]
    # else:
    #     color_information = None

    # Latent space figures
    print("Latent space figures ...")
    fig, axs = plot_latent_space(encodings[:,:2], goals, size=figure_size[1])
    save_figure(fig, dir.joinpath("latent-space_randomized"))

    # fig, axs = plot_latent_space_axis_over_task(encodings[:,:2], goals, color_information)
    # fig.set_size_inches(*figure_size)
    # save_figure(fig, dir.joinpath("tasks-and-encodings_randomized"))


    if show_plots: plt.show()
    plt.close('all')    # Reduce memory requirements by closing open plots
  
    # import pickle

    # # Save replay data for later rendering
    # replay_data = {
    #     'trajectories': trajectories,
    #     'env_name': type(env).__name__,
    #     'config': result_dict['config'] if 'config' in result_dict else config,
    #     'ant_env_config': getattr(env, 'env_config', getattr(env, 'config', None))
    # }

    # save_path = Path(save_dir).joinpath(name).joinpath("replay_data.pkl")
    # with open(save_path, "wb") as f:
    #     pickle.dump(replay_data, f)

    # print(f"Saved replay data for later rendering at: {save_path}")


    # Video

    print(italic("Done!"))
    print("\n\n")


def load_model(encoder, config):

    '''
    Define the low-level policy and agent to test
    '''
    complex_agent_config = dict(
        experiments_repo = '/home/ubuntu/yuanmeng/bo/MRL-Inference-Reutilization/output/low_level_policy/',
        experiment_name = 'new_cheetah_training_server1_diff_taskid',
        epoch = 300,
    )
    with open(os.path.join(complex_agent_config['experiments_repo'],complex_agent_config['experiment_name'], 'config.json'), "r") as file:
        env_config = json.load(file)

    complex_agent_config['environment'] = HalfCheetahMixtureEnv(env_config)
    # env of the complex agent
    env = complex_agent_config['environment']

    # A wrapper that modifies the encoder from toy env to adapt it for the specific complex environment.
    # map_cheetah_to_toy1d function is defined as input_map in InputOutputDecorator, which preprocess the 5 tensors of the MDP encoder input
    encoder = InputOutputDecorator(encoder, map_cheetah_to_toy1d_velocity, observation_dim=env.observation_space, action_dim=env.action_space, latent_dim=1, encoding_dim=1, context_size=5)
    #get the low-level policy action from the complex agent
    transfer_function = get_complex_agent(env, complex_agent_config)
    return encoder, env, transfer_function

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    
    '''
    Deinfe the inference module
    '''
    paths = [

        '/home/ubuntu/yuanmeng/bo/MRL-Inference-Reutilization/submodules/Meta_RL/data/experiments_thesis/step1_biggerNN_velocity_-4_4_v2_right_directions/_2025-11-08_23-55-50'



        # "data/transfer_encoders/toy1d_attention-np",
        # "data/transfer_encoders/toy1d_log-rand",

    ]


    for path in paths:
        with open(os.path.join(path, 'variant.json'), "r") as file:
            config = json.load(file)
        model_evaluation(
            path, #path to the inference model
            save_dir='./evaluation/experiments_thesis/transfer_cheetah_velocity_stride5_v1_infer_velright_scale_down_lowlevel_vel_test/',
            config=config,
            # save_dir = './data/delete',
            create_video=True,
            figure_size=(8,6),
            trajectory_2d=False,
            color_by=None,
        )