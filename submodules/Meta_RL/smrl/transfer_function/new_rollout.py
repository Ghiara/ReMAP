"""
This module contains functions for policy rollouts.

Author(s): 
    Julius Durmann
Contact: 
    julius.durmann@tum.de
Date: 
    2022-12-12

Note:
    The functions in this module serve the same functionality as those of
    `rlkit/samplers/rollout_functions.py`. See https://github.com/rail-berkeley/rlkit.
"""

import numpy as np
import torch
import copy
import random
from smrl.utility.console_strings import print_to_terminal
from main_config import SAVE_REWARD_PLOTS_EVALUATION

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from pathlib import Path
from datetime import datetime
import pytz
import os


from typing import Callable, List, Tuple

from mrl_analysis.utility.interfaces import MdpEncoder, MetaRLPolicy, MetaEnv


def rollout_with_encoder(simple_env, encoder: MdpEncoder, transfer_function, context_size: int):
    """Wrapper function which ensures that an encoder can be specified
    without breaking the argument structure of rlkit MdpPathCollector.

    The wrapped function is `_rollout_with_encoder` (see below).

    Parameters
    ----------
    encoder : Encoder
        An MDP encoder.
    context_size : int
        Length of the context sequence used for the encoder.
    """
    def rollout_fn(
        env,
        agent,
        max_path_length=np.inf,
        render=False,
        render_kwargs=None,
        preprocess_obs_for_policy_fn=None,
        get_action_kwargs=None,
        return_dict_obs=False,
        full_o_postprocess_func=None,
        reset_callback=None,
    ):
        return _rollout_with_encoder(
            simple_env, env, agent, encoder, transfer_function, context_size,
            max_path_length, render,
            render_kwargs, preprocess_obs_for_policy_fn,
            get_action_kwargs, return_dict_obs, full_o_postprocess_func,
            reset_callback
        )
    return rollout_fn


def _rollout_with_encoder(
        simple_env: MetaEnv,
        env: MetaEnv,
        agent: MetaRLPolicy,
        encoder: MdpEncoder,
        transfer_function,
        context_size: int,
        max_path_length: int = np.inf,
        render: bool = False,
        render_kwargs: dict = None,
        preprocess_obs_for_policy_fn: Callable = None,
        get_action_kwargs: dict = None,
        return_dict_obs: bool = False,
        full_o_postprocess_func: Callable = None,
        reset_callback: Callable = None,
) -> dict:
    """Rollout function for Meta Reinforcement Learning. Returns one trajectory.
    An encoder is used for computing contextual information about the task.

    The path is sampled until either 
    1) The environment returns 'done' or
    2) The maximum path length (`max_path_length`) is reached.

    NOTE: This function extends the rollout function from `rlkit/samplers/rollout_functions.py`. 
    Large portions of the code are copied from rlkit.

    Parameters
    ----------
    env : gym.Env
        Environment for interaction.
    agent : MetaRLPolicy
        The agent/policy used for sampling actions.
    encoder : MdpEncoder
        The encoder used for extracting a task representation from
        contextual information.
    context_size : int
        Length of the context sequence
    max_path_length : int, optional
        Maximum length of a sampled path, by default np.inf
    render : bool, optional
        Activates (True) or deactivates (False) environment rendering,
        by default False
    render_kwargs : dict, optional
        Arguments passed to render(), will be ignored if render==False,
        by default None
    preprocess_obs_for_policy_fn : Callable, optional
        Function which preprocesses observations before
        they are passed on to the policy, by default None
    get_action_kwargs : dict, optional
        Additional arguments (besides observation) passed to
        the policy (policy.get_action), by default None
    return_dict_obs : bool, optional
        If True, the returned observations won't be mapped to a numpy array,
        by default False
    full_o_postprocess_func : Callable, optional
        Function which is called between action sampling from
        the policy and environment update (step), by default None
    reset_callback : Callable, optional
        Function which is called before sampling the path but
        after env.reset().
        Takes as inputs the environment, policy (agent), and
        current observation,
        by default None

    Returns
    -------
    dict
        Dictionary with path data:
        - "observations"
        - "encodings"
        - "actions"
        - "rewards"
        - "next_observations"
        - "terminals"
        - "dones"
        - "agent_infos"
        - "env_infos"
    """

    # Variable initialization
    if render_kwargs is None:
        render_kwargs = {}
    if get_action_kwargs is None:
        get_action_kwargs = {}
    if preprocess_obs_for_policy_fn is None:
        preprocess_obs_for_policy_fn = lambda x: x

    # TODO: If environment spaces are discrete, the shape argument might not be correct
    observations = np.zeros((max_path_length, *env.observation_space.shape))
    actions = np.zeros((max_path_length, *env.action_space.shape))
    rewards = np.zeros((max_path_length, 1))
    next_observations = np.zeros((max_path_length, *env.observation_space.shape))
    terminals = np.zeros((max_path_length, 1), dtype=np.bool8)
    dones = np.zeros((max_path_length, 1), dtype=np.bool8)
    encodings = np.zeros((max_path_length, encoder.encoding_dim))
    not_healthy_agent = False

    agent_infos = []
    env_infos = []
    tasks = []
    # 在 Sampling 前定义记录容器
    subgoals, pre_states, post_states = [], [], []


    path_length = 0
    fall_step = None
    #reset the high-level policy
    agent.reset()
    #randomly sample a task for the simple env
    simple_env.sample_task()
    o, env_info = env.reset()
    if reset_callback:
        reset_callback(env, agent, o)
    if render:
        env.render(**render_kwargs)

    # Sampling
    # start rolling out until done or max_path_length
    while path_length < max_path_length:
        # fobidden to use back propagation here, just forward pass(encoder infer the latent)
        with torch.no_grad():
            # Encode context (sample from posterior)
            #estimate the context window that sends to the encoder
            #path_length may be smaller than context_size at beginning, it means the current time steps in rollout 
            index_low, index_high = max(0, path_length - context_size), path_length
            #transfer the context(context = context_size X transitions) to the encoding(latent variable z)
            #This latent variable z will be sampled from the posterior distribution q(z|c) z:latent variable, c:context
            # waht will encoding look like? encoding size = encoder.encoding_dim:1 dim
            encoding = encoder.get_encoding(
                observations[index_low : index_high],
                actions[index_low : index_high],
                rewards[index_low : index_high],
                next_observations[index_low : index_high],
                terminals[index_low : index_high],
            )
            # encoding = env.task['goal'][None,0] # This line can be used for debugging purposes with Toy1D
            
            # Get action from policy
            #this line didn't work because the o_for_agent is overwritten later
            o_for_agent = preprocess_obs_for_policy_fn(o)
            #assign the o_for_agent to be the x-position of the complex agent in the complex env
            o_for_agent = env.sim.data.qvel[0]# need to change according to the velocity or position tracking task
            o_cheetah = preprocess_obs_for_policy_fn(o)
            # agent: high-level policy  
            #High level policy generates subgoals based on the current observation(obs of simple agent) and the inferred encoding
            #simple_action:relative subgoal for the transfer function to generate complex action
            #simple_action represents the relative change to the current poisition or velocity of the complex agent
            simple_action, agent_info = agent.get_action(torch.tensor([o_for_agent], dtype=torch.float32), encoding, **get_action_kwargs)
            #simple_obs means the absolute position or velocity(subgoal) of the complex agent after applying the subgoal(simple_action), the simple_obs will be send to the env
            simple_obs = simple_action + o_for_agent

            # 记录 subgoal 及执行前状态
            subgoals.append(float(simple_obs))
            pre_states.append(float(env.sim.data.qvel[0]))  # 或 qpos[0]，视你跟踪的目标（velocity/position）
            #TODO: change for new arc
            # task = np.zeros(..)
            # task[env.base_task] = task[0]
            task = np.zeros_like(env.task)
            # if simple_action<0:
            #     env.base_task = env.config['tasks']['goal_back']
            # else:
            #test only for goal left
            #only ant use velocity_right
            env.base_task = env.config['tasks']['forward_vel']
        #set the simple_obs[0] as the goal value for the selected base task
        task[env.base_task] = simple_obs[0]
            # task = np.array([task[0],0])

        if full_o_postprocess_func:
            full_o_postprocess_func(env, agent, o)
        #write the subgoal to the real env (env-conditioning)
        #env will change its reward function according to the new subgoal task
        env.update_task(task)
        # env.update_task(np.array([task]))
        #use 5 micro steps to reach the new updated task(subgoal)
        for i in range(5):
        # Environment step
            complex_action = transfer_function.get_action(torch.tensor(o_cheetah), torch.tensor(task), return_dist=False)
            #scale down the complex action to avoid the complex agent to move too fast
            complex_action = 0.5 * complex_action
            next_o, r, internal_done, trunc, env_info = env.step(complex_action.detach().cpu().numpy())
            # next_o = env.sim.data.qpos
            #detect if the complex agent falls during the rollout(debugging purpose)
            if hasattr(env, "sim") and env.sim.data.qpos[2] < 0.3:
                print(f"[DEBUG] Fall detected at global step {path_length}, "
                      f"velocity={env.sim.data.qvel[0]:.2f}, z={env.sim.data.qpos[2]:.2f}")
                # 你也可以在这里设置一个标志位，记录摔倒的时刻
                fall_step = path_length

            if 'fall_step' not in locals():
                fall_step = None
            if env.sim.data.qpos[2] < 0.3 and fall_step is None:
                fall_step = path_length


            o_cheetah = next_o
            print(o_cheetah)
            #use the following line to debug the goal tracking of the complex agent
            # print('X_pos:', env.sim.data.qpos[0])
            #use the following line to debug the velocity tracking of the complex agent
            print('X_vel:', env.sim.data.qvel[0])
            if 'reached_goal' in env_info:
                if internal_done:
                    print('Agent not healthy')
                    not_healthy_agent = True
                    break
                if env_info['reached_goal']:
                    print('reached goal after timestep:', i)
                    break
            else:
                if internal_done:
                    break

        # 记录执行后状态
        post_states.append(float(env.sim.data.qvel[0]))


        #change the reward if evaluate the goal tracking performance
        # r = - np.abs(env.sim.data.qpos[0] - simple_env.task['goal'])
        #change the reward if evaluate the goal tracking performance
        #TODO: NEED TO CHECK IF THERE IS VELOCITY IN THE VELOCITY TOY ENV
        r = - np.abs(env.sim.data.qvel[0] - simple_env.task['goal_velocity'])
        # -3 for goal tracking inference reuse, needs to be changed for velocity tracking(different agent have different index for velocity)
        #simple_action = next_o[-3] - o[-3]
        simple_action = next_o[8] - o[8]
        if render:
            env.render(**render_kwargs)

        terminal = False
        done=False
        # if done:
        #     # terminal=False if TimeLimit caused termination
        #     if not env_info.pop('TimeLimit.truncated', False):
        #         terminal = True
        ## For randomization purposes
        # o = o*multiplier
        # a = a*multiplier
        # r = r*multiplier
        # next_o = next_o*multiplier

        # Store new transition to trajectory
        observations[path_length, :] = o_cheetah
        actions[path_length, :] = simple_action
        rewards[path_length, :] = r
        next_observations[path_length, :] = next_o
        terminals[path_length, :] = terminal
        dones[path_length, :] = done
        encodings[path_length, :] = encoding

        agent_infos.append(agent_info)
        env_infos.append(env_info)
        tasks.append(simple_env.task)
        path_length += 1


        # if done:
        #     break

        o = next_o


    if SAVE_REWARD_PLOTS_EVALUATION:

        current_time = datetime.now().astimezone(pytz.timezone('Europe/Berlin'))
        def save_figure(fig: Figure, save_as: Path):
            fig.savefig(save_as.with_suffix('.png'))

        directory = 'data/reward_plots/evaluation'
        if not os.path.exists(directory):
            os.makedirs(directory)
        # Sanitize the file name and create the path
        goal_str = str(env.task['goal']).replace('[', '').replace(']', '').replace('.', ',')  # Replace/remove invalid chars
        current_time = datetime.now()
        path = f'data/reward_plots/{goal_str}_{current_time.strftime("%Y-%m-%d_%H-%M-%S")}'

        fig, axs = plt.subplots(1, figsize=(10,5))
        axs.plot(rewards, alpha=0.3)
        axs.set_xlabel('Reward')
        # axs.legend()
        save_figure(fig, Path(path))
    
    return dict(
        observations=observations[:path_length],
        encodings=encodings[:path_length],
        actions=actions[:path_length],
        rewards=rewards[:path_length],
        next_observations=next_observations[:path_length],
        terminals=terminals[:path_length],
        dones=dones[:path_length],
        agent_infos=agent_infos,
        env_infos=env_infos,
        tasks=tasks,
        not_healthy_agent = not_healthy_agent,
        subgoals=np.array(subgoals),
        pre_states=np.array(pre_states),
        post_states=np.array(post_states),
        fall_step = fall_step
    )
