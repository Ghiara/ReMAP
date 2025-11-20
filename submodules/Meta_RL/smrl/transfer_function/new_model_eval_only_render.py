
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
from meta_envs.toy_goal.toy_1d import Toy1D
from specific.encoder_transfer import map_cheetah_to_toy1d
from sac_envs.half_cheetah_multi import HalfCheetahMixtureEnv
from sac_envs.walker_multi import WalkerMulti
from sac_envs.hopper_multi import HopperMulti
from sac_envs.ant_multi_old import AntMulti
from smrl.transfer_function.new_PathCollector import TrajectoryGeneratorWithTransferFunction
from collections import OrderedDict
# from smrl.trainers.transfer_function import TransferFunction
from model import PolicyNetwork as TransferFunction

from mrl_analysis.plots.plot_settings import *

# HARDCODED = True

# Parameters
video_creator = VideoCreator()
video_creator.fps = 30
DEVICE = 'cpu'
max_path_l = 200

def get_complex_agent(env, complex_agent_config):
    print("=== Debug: env.action_space.low/high ===", env.action_space.low[0], env.action_space.high[0])

    pretrained = complex_agent_config['experiments_repo']+complex_agent_config['experiment_name']+f"/models/policy_model/epoch_{complex_agent_config['epoch']}.pth"
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    action_bounds = [env.action_space.low[0], env.action_space.high[0]]
    transfer_function = TransferFunction(
        n_states=n_states,
        n_actions=n_actions,
        action_bounds=action_bounds,
        pretrained=pretrained
        )
    transfer_function.to(DEVICE)
    return transfer_function

def model_evaluation(
    path_to_data: str, 
    save_dir: str, 
    config,
    itr: int = None, 
    n_trajectories: int = 5,
    create_video: bool = True,
):
    """Generate Mujoco-rendered video of Ant agent using transferred inference module.
       No plots or additional logs are generated.
    """

    # --------------------------------------------------------------------------
    # 1. Load results and environment
    # --------------------------------------------------------------------------
    print("\n=== Loading inference module ===")
    path_to_data = Path(path_to_data)
    result_dict = load_results(path_to_data, itr=itr)
    name = path_to_data.name

    encoder: MdpEncoder = result_dict['encoder']
    policy: MetaRLPolicy = result_dict['eval_policy']

    # --------------------------------------------------------------------------
    # 2. Load complex (Ant) environment and transfer function
    # --------------------------------------------------------------------------
    print("=== Loading Ant environment and pretrained policy ===")
    encoder, env, transfer_function = load_model(encoder, config)

    env.render_mode = 'rgb_array'
    env.screen_width = 1280
    env.screen_height = 720

    decoder = None
    value_function = None

    # --------------------------------------------------------------------------
    # 3. Generate trajectories using Ant + inference encoder
    # --------------------------------------------------------------------------
    print("=== Generating trajectories ===")
    tg = TrajectoryGeneratorWithTransferFunction(
        env, env, policy, encoder, transfer_function, decoder, value_function
    )
    trajectories = tg.run(n_trajectories, max_path_l)

    print("Collected", len(trajectories), "trajectories.")

    # --------------------------------------------------------------------------
    # 4. Add green goal marker (if available)
    # --------------------------------------------------------------------------
    if hasattr(env, "viewer") and hasattr(env, "goal_position"):
        goal = getattr(env, "goal_position")
        try:
            env.viewer.add_marker(
                pos=np.array([goal[0], goal[1], 0]),
                size=np.array([0.25, 0.25, 0.25]),
                rgba=np.array([0, 1, 0, 1]),
            )
            print("Added green goal marker at:", goal)
        except Exception as e:
            print("Warning: could not add goal marker:", e)

    # --------------------------------------------------------------------------
    # 5. Generate and save Mujoco-rendered video
    # --------------------------------------------------------------------------
    if create_video:
        print("=== Creating Mujoco rendered video ===")
        output_dir = Path(save_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        save_path = str(output_dir.joinpath(f"{name}_ant_render.mp4"))

        video_creator.fps = 60
        # ⚠️ 注意这里用 transfer_function 替换 policy
        result_dict_for_video = {
            'encoder': encoder,
            'decoder': decoder if decoder is not None else encoder,  # dummy
            'policy': transfer_function,   # ✅ 替换！
            'eval_env': env
        }
        video_creator.create_video(
            result_dict_for_video,
            video_length=15.0,
            save_as=save_path,
            env_reset_interval=max_path_l,
            width=1280,
            height=720,
            override_env=env,     # ✅ 强制使用 Ant 环境！
        )


        print("✅ Video saved to:", save_path)

    print(italic("Done! Only video generated."))

def load_model(encoder, config):

    '''
    Define the low-level policy and agent to test
    '''
    complex_agent_config = dict(
        experiments_repo = '/home/ubuntu/yuanmeng/bo/MRL-Inference-Reutilization/output/low_level_policy/',
        experiment_name = 'ant_multi_old_architecture_only_goal_left_with_termination_goal_tracking',
        epoch = 400,
    )
    with open(os.path.join(complex_agent_config['experiments_repo'],complex_agent_config['experiment_name'], 'config.json'), "r") as file:
        env_config = json.load(file)

    complex_agent_config['environment'] = AntMulti(env_config)
    env = complex_agent_config['environment']

    # Old 
    encoder = InputOutputDecorator(encoder, map_cheetah_to_toy1d, observation_dim=env.observation_space, action_dim=env.action_space, latent_dim=1, encoding_dim=1, context_size=5)
    transfer_function = get_complex_agent(env, complex_agent_config)
    
    # ✅ Override task sampler: only positive goals in [2, 10]
    def sample_task_positive_only():
        goal = np.random.uniform(2.0, 10.0)
        env.task = np.array([goal, 0.0])
        if hasattr(env, "goal_position"):
            env.goal_position = np.array([goal, 0.0])
        return env.task

    env.sample_task = sample_task_positive_only
    print("✅ Overrode env.sample_task(): positive goal range [2.0, 10.0] meters")
    
    return encoder, env, transfer_function

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    
    '''
    Deinfe the inference module
    '''
    paths = [

        '/home/ubuntu/yuanmeng/bo/MRL-Inference-Reutilization/submodules/Meta_RL/data/experiments/toy1d_MaxAction_1_2025-02-05_14-55-41'



        # "data/transfer_encoders/toy1d_attention-np",
        # "data/transfer_encoders/toy1d_log-rand",

    ]


    for path in paths:
        with open(os.path.join(path, 'variant.json'), "r") as file:
            config = json.load(file)
        model_evaluation(
            path,
            save_dir='./evaluation/experiments_thesis/transfer_ant_test_mujoco_render/',
            config=config,
            # save_dir = './data/delete',
            create_video=True,
            #figure_size=(8,6),
            #trajectory_2d=False,
            #color_by=None,
        )