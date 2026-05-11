#test for stick breaking inference model with cheetah transfer, then use here stick_break_inference.DecoupledEncoder
from tigr.task_inference.dpmm_inference import DecoupledEncoder
# from configs.toy_config import toy_config
import numpy as np
from rlkit.envs import ENVS
from tigr.task_inference.dpmm_bnp import BNPModel
import torch
import os
from rlkit.torch.sac.policies import TanhGaussianPolicy
from sac_envs.half_cheetah_multi import HalfCheetahMixtureEnv
from model import PolicyNetwork as TransferFunction
import rlkit.torch.pytorch_util as ptu
from collections import OrderedDict
import cv2
from typing import List, Any, Dict, Callable
import json
import imageio
import rlkit.torch.pytorch_util as ptu
from tigr.task_inference.prediction_networks import DecoderMDP, ExtendedDecoderMDP
import matplotlib.pyplot as plt
import random
from collections import namedtuple
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from sac_envs.walker import WalkerGoal
from sac_envs.hopper import HopperGoal
from sac_envs.half_cheetah_multi import HalfCheetahMixtureEnv
from sac_envs.hopper_multi import HopperMulti
from sac_envs.walker_multi import WalkerMulti
from sac_envs.ant_multi import AntMulti
from sac_envs.walker_multi import WalkerMulti

from agent import SAC
from model import ValueNetwork, QvalueNetwork, PolicyNetwork

from mrl_analysis.plots.plot_settings import *
from vis_utils.vis_logging import log_all, _frames_to_gif
import pandas as pd


# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE = 'cuda'
ptu.set_gpu_mode(True)

# TODO: einheitliches set to device
simple_env_dt = 0.05
sim_time_steps = 10
max_path_len = 100
num_trajectories = 50
plot_every = 5
loss_criterion = nn.CrossEntropyLoss()

class Memory():

    def __init__(self, memory_size):
        self.memory_size = memory_size
        self.memory = []
        self.Transition = namedtuple('Transition',
                        ('task', 'simple_obs', 'simple_action', 'mu'))
        self.batch_size = 256
        self.task_dim = 1
        self.latent_dim = 4
        self.simple_obs_dim = 2
        self.simple_action_dim = 1

    def add(self, *transition):
        self.memory.append(self.Transition(*transition))
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)
        assert len(self.memory) <= self.memory_size

    def sample(self, size):
        return random.sample(self.memory, size)

    def __len__(self):
        return len(self.memory)
    
    def unpack(self, batch):
        batch = self.Transition(*zip(*batch))
        
        tasks = torch.cat(batch.task).view(self.batch_size, self.task_dim).to(DEVICE)
        simple_obs = torch.cat(batch.simple_obs).view(self.batch_size, self.simple_obs_dim).to(DEVICE)
        simple_action = torch.cat(batch.simple_action).view(self.batch_size, self.simple_action_dim).to(DEVICE)
        mu = torch.cat(batch.mu).view(self.batch_size, self.latent_dim).to(DEVICE)

        return tasks, simple_obs, simple_action, mu

def get_encoder(path, shared_dim, encoder_input_dim):
    path = os.path.join(path, 'weights')
    for filename in os.listdir(path):
        if filename.startswith('encoder'):
            name = os.path.join(path, filename)
    
    # Important: Gru and Conv only work with trajectory encoding
    if variant['algo_params']['encoder_type'] in ['gru'] and variant['algo_params']['encoding_mode'] != 'trajectory':
        print(f'\nInformation: Setting encoding mode to trajectory since encoder type '
              f'"{variant["algo_params"]["encoder_type"]}" doesn\'t work with '
              f'"{variant["algo_params"]["encoding_mode"]}"!\n')
        variant['algo_params']['encoding_mode'] = 'trajectory'
    elif variant['algo_params']['encoder_type'] in ['transformer', 'conv'] and variant['algo_params']['encoding_mode'] != 'transitionSharedY':
        print(f'\nInformation: Setting encoding mode to trajectory since encoder type '
              f'"{variant["algo_params"]["encoder_type"]}" doesn\'t work with '
              f'"{variant["algo_params"]["encoding_mode"]}"!\n')
        variant['algo_params']['encoding_mode'] = 'transitionSharedY'

    encoder = DecoupledEncoder(
        shared_dim,
        encoder_input_dim,
        num_classes = variant['reconstruction_params']['num_classes'],
        latent_dim = variant['algo_params']['latent_size'],
        time_steps = variant['algo_params']['time_steps'],
        encoding_mode=variant['algo_params']['encoding_mode'],
        timestep_combination=variant['algo_params']['timestep_combination'],
        encoder_type=variant['algo_params']['encoder_type'],
        bnp_model=bnp_model
    )
    encoder.load_state_dict(torch.load(name, map_location='cpu'))
    encoder.to(DEVICE)
    return encoder

def get_simple_agent(path, obs_dim, policy_latent_dim, action_dim, m):
    path = os.path.join(path, 'weights')
    for filename in os.listdir(path):
        if filename.startswith('policy'):
            name = os.path.join(path, filename)
    
    policy = TanhGaussianPolicy(
        obs_dim=(obs_dim + policy_latent_dim),
        action_dim=action_dim,
        latent_dim=policy_latent_dim,
        hidden_sizes=[m,m,m],
    )
    policy.load_state_dict(torch.load(name, map_location='cpu'))
    policy.to(DEVICE)
    return policy

def get_complex_agent(env, complex_agent):
    pretrained = complex_agent['experiments_repo']+complex_agent['experiment_name']+f"/models/policy_model/epoch_{complex_agent['epoch']}.pth"
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

def cheetah_to_simple_env_map(
    # observations: torch.Tensor, 
    observations,
    next_observations: torch.Tensor):
    """
    Maps transitions from the cheetah environment
    to the discrete, one-dimensional goal environment.
    """

    ### little help: [0:3] gives elements in positions 0,1,2 
    simple_observations = np.zeros(obs_dim)
    simple_observations[...,0] = observations[...,-3]
    simple_observations[...,1] = observations[...,8]

    next_simple_observations = np.zeros(obs_dim)
    next_simple_observations[...,0] = next_observations[...,-3]
    next_simple_observations[...,1] = next_observations[...,8]

    return simple_observations, next_simple_observations

def cheetah_to_simple_env_obs(obs):
    simple_observations = np.zeros(obs_dim)
    simple_observations[...,0] = obs[...,-3]
    # simple_observations[...,1:3] = obs[...,1:3]
    # simple_observations[...,3:] = obs[...,7:10]
    simple_observations[...,1] = obs[...,8]
    return simple_observations

def general_obs_map(env):
    simple_observations = np.zeros(obs_dim)
    simple_observations[...,0] = env.sim.data.qpos[0]    
    simple_observations[...,1] = env.sim.data.qvel[0]    
    return simple_observations

def get_decoder(path, action_dim, obs_dim, reward_dim, latent_dim, output_action_dim, net_complex_enc_dec, variant):
    path = os.path.join(path, 'weights')
    for filename in os.listdir(path):
        if filename.startswith('decoder'):
            name = os.path.join(path, filename)
    output_action_dim = 8
    decoder = ExtendedDecoderMDP(
        action_dim,
        obs_dim,
        reward_dim,
        latent_dim,
        output_action_dim,
        net_complex_enc_dec,
        variant['env_params']['state_reconstruction_clip'],
    ) 

    decoder.load_state_dict(torch.load(name, map_location='cpu'))

    for param in decoder.parameters():
        param.requires_grad = False
    for param in decoder.task_decoder.last_fc.parameters():
        param.requires_grad = True

    # ===== 3. 解冻倒数第二层 fc2 =====
    for p in decoder.task_decoder.fc2.parameters():
        p.requires_grad = True


    # ===== 3. 解冻倒数第3层 fc1 =====
    for p in decoder.task_decoder.fc1.parameters():
        p.requires_grad = True


    decoder.to(DEVICE)
    return decoder

def create_tsne(latent_variables, task_labels, path):
    from sklearn.manifold import TSNE
    #import matplotlib.pyplot as plt
    save_as = os.path.join(path , 'tsne_test.png')
    save_as = os.path.join(path , 'tsne_test.pdf')
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(latent_variables)

    # Plot
    plt.figure(figsize=(10, 6))
    unique_labels = np.unique(task_labels)
    for label in unique_labels:
        idx = task_labels == label
        plt.scatter(tsne_results[idx, 0], tsne_results[idx, 1], label=label, alpha=0.7)
    plt.legend()
    plt.title('t-SNE Visualization of Latent Variables')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.savefig(save_as)
    plt.close()

def save_plot(loss_history, name:str, path=f'{os.getcwd()}/plots'):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        plt.figure()
        # Plotting the loss
        plt.plot(loss_history)
        plt.title('Loss over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(os.path.join(path,name+'.png'))
        plt.savefig(os.path.join(path,name+'.pdf'))

        plt.close()


# === 🧩 工具函数：保存一个 tensorboard step ===
def _write_tensorboard_step(save_root, step_idx, latents, true_ids, pred_ids, specs=None):
    """
    把一批样本写成 tensorboard 兼容格式：
    <root>/<00001>/default/{metadata.tsv, tensors.tsv}
    metadata 每行: "<true_id> [<spec>] -> <pred_id>"
    """
    import csv, os, numpy as np
    step_dir = os.path.join(save_root, f"{step_idx:05d}", "default")
    os.makedirs(step_dir, exist_ok=True)

    np.savetxt(os.path.join(step_dir, "tensors.tsv"),
               np.vstack(latents), delimiter="\t")

    if specs is None:
        specs = [0.0] * len(true_ids)

    with open(os.path.join(step_dir, "metadata.tsv"), "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        for t, s, p in zip(true_ids, specs, pred_ids):
            writer.writerow([f"{int(t)} [{float(s):.3f}] -> {int(p)}"])




# === 选一个“均衡的四类任务 + 随机 spec”的任务 ===
def _balanced_task_for_episode(env, range_dict, episode, rng=None):
    """
    返回 env.sample_task 可接收的 task 字典：
    {'base_task': <str>, 'specification': <float>}
    四类轮流：goal_front, goal_back, forward_vel, backward_vel
    spec 从各自区间随机采样（back 及 backward 取负号）
    """
    if rng is None:
        rng = np.random.default_rng()
    t = env.config['tasks']  # 例如 {'goal_front':0, 'goal_back':1, 'forward_vel':2, 'backward_vel':3}

    order = ['goal_front', 'goal_back', 'forward_vel', 'backward_vel']
    base_name = order[episode % 4]

    if base_name in ['goal_front', 'goal_back']:
        # 位置任务：用你脚本里 range_dict['pos_x'] 的范围
        lo, hi = range_dict['pos_x']                # e.g. [0.5, 25]
        spec = float(rng.uniform(lo, hi))
        if base_name == 'goal_back':
            spec = -spec
    else:
        # 速度任务：用你脚本里 range_dict['velocity_x'] 的范围
        lo, hi = range_dict['vel_x']           # e.g. [0.5, 3.0]
        spec = float(rng.uniform(lo, hi))
        if base_name == 'backward_vel':
            spec = -spec

    return {'base_task': base_name, 'specification': spec}


def rollout(env, encoder, decoder, optimizer, simple_agent, step_predictor, transfer_function, memory, 
            variant, obs_dim, actions_dim, max_path_len, 
            n_tasks, inner_loop_steps, save_video_path, experiment_name,
            current_inference_path_name, tasks=None, beta=0.1):
    #
    range_dict = OrderedDict(
        pos_x = [0.5, 25],       # 正方向（前方）位置
        vel_x = [0.3, 3.0],      # 正方向速度
    )
    
    save_after_episodes = 5
    value_loss_history, q_loss_history, policy_loss_history, rew_history = [], [], [], []
    path = save_video_path
    
 
    loss_history = []

    #new log varibale for subgoals
    episode_logs = []
    global_step = 0


    # === 🧩 新增：t-SNE latent 导出设置 ===
    EXPORT_ROOT = os.path.join(save_video_path, "tensorboard_transfer")
    EXPORT_SAMPLES_PER_STEP = 1000  # 每导出一次保存多少个样本点，可自行调大
    export_latents, export_true, export_pred = [], [], []
    export_step_idx = 1


    pos_reward, vel_reward = 0, 0
    tasks_pos, tasks_vel = [], []
    # prepare for plotting the x position and velocity in each episode
    x_pos_plot, x_vel_plot = [],[]
    if tasks:# if specific tasks are given
        num_trajectory = len(tasks)
        save_after_episodes = 1
    else: #else use default num_trajectories like 50 to rollout
        num_trajectory= num_trajectories
    #start rollout for each trajectory
    for episode in range(num_trajectory):
        sim_time_steps_list = []  # 用于记录本 episode 每次预测的步长
        video = False
        if episode % 1 == 0:#for each of the episodes
            frames = []
            image_info = dict(reward = [],
            obs = [],
            base_task = [],
            complex_action = [],
            simple_action = [],
            action = [])
            video = True
        
        #these save the x position and velocity at each timestep in the rollout, and will be used for plotting the pos/vel trajectory
        x_pos_curr, x_vel_curr = [],[]
            
        print(f"Inference Path: {current_inference_path_name}, Episode: {episode}")

        #initialize the high level counter
        path_length = 0
        # env.reset() returns (obs, info), now we only need the initial obs
        obs = env.reset()[0]
        #get the current pos and vel
        x_pos_curr.append(env.sim.data.qpos[0])
        x_vel_curr.append(env.sim.data.qvel[0])
        #reset the simple env, make the simple_env can start with clean state
        simple_env.reset_model()#???????? why we need the simple env here?
        #n_tasks: number of tasks to be sampled in the current rollout
        #time_steps: number of timesteps to be used for encoding(length of the context)
        #ONE CAUTION: context has the shape of (n_tasks, time_steps, obs_dim + 1 + obs_dim), where the +1 is for reward dim
        #transtion here is defined as (obs, reward, next_obs)
        contexts = torch.zeros((n_tasks, variant['algo_params']['time_steps'], obs_dim + 1 + obs_dim), device=DEVICE)
        l_vars = []
        labels = []

        done = 0
        episode_reward = 0
        loss = []
        value_loss, q_loss, policy_loss = [], [], []
        # old sampling method    
        # if tasks:
        #     task = env.sample_task(task=tasks[episode], test=True)
        # else:
        #     task = env.sample_task(test=True)
        #     count = 0
        #     while env.base_task not in [env.config.get('tasks',{}).get('goal_front'), 
        #                                     env.config.get('tasks',{}).get('goal_back'), 
        #                                     env.config.get('tasks',{}).get('forward_vel'), 
        #                                     env.config.get('tasks',{}).get('backward_vel')]:
        #         task = env.sample_task(test=True)
        #         count+=1
        #         if count>150:
        #             return 'Failed to sample task. Attempted 150 times.'

        #new balanced sampling method
        if tasks:
            # 你自己传了 tasks 列表就按它来（每个元素类似 {'base_task':..., 'specification':...}）
            task_cfg = tasks[episode % len(tasks)]
        else:
            # 四类轮流 + 每集随机 spec（亮度）
            #task_cfg(example):{'base_task': 'forward_vel', 'specification': 0.5}
            task_cfg = _balanced_task_for_episode(env, range_dict, episode)

        # 让complex环境按我们指定的 base_task + spec 设置任务
        task = env.sample_task(task=task_cfg, test=True)


        # ✅ 强制覆盖 env.base_task 以确保任务四类轮换,get the base_task from the task config
        base_name = task_cfg['base_task']
        
        env.base_task = env.config['tasks'][base_name]

        # 这行用来导出 metadata 里的 [spec]（一集内保持常数即可）
        # not change in episode, will be saved later in the "spec_of_episode" field of episode_logs
        spec_for_logging = float(task_cfg['specification'])

        # empty loss for the decoder training in the current episode, but now not used beacause the decoder is not trained
        _loss  = []
        #each of the loop means the high level step(subgoal generation)
        #max_path_len: maximal number of high level steps in the current rollout
        #If there is no early termination, the rollout will last for max_path_len high level steps
        #Here we rollout for max_path_len high level steps
        for path_length in range(max_path_len):

            # get encodings
            # Sychronize the position/velocity of the complex env with the simple env
            simple_env.sim.data.set_joint_qpos('boxslideX', env.sim.data.qpos[0])
            simple_env.sim.data.set_joint_qvel('boxslideX', env.sim.data.qvel[0])
            #detach: cut off the gradient flow
            encoder_input = contexts.detach().clone()
            #encoder_input shape: (n_tasks, time_steps*(obs_dim + 1 + obs_dim))
            #enocder wants the input shape to be (batch_size, input_dim)
            encoder_input = encoder_input.view(encoder_input.shape[0], -1).to(DEVICE)
            #input the context to the encoder to get the latent variable
            #mu: task latent variable(very important for task inference!!!)
            mu, log_var = encoder(encoder_input)

            # === 🧩 记录 latent (μ) 与任务标签 ===
            export_latents.append(mu.detach().cpu().numpy().squeeze())
            #get the true label from the env
            true_label = int(env.base_task)

            # base_name = task_cfg['base_task'] 

            # DISPLAY_LABEL = {
            #     'forward_vel': 0,
            #     'backward_vel': 1,
            #     'goal_front':  2,
            #     'goal_back':   3,
            # }

            # Decoder 预测标签
            #general_obs_map: get the simple observation(x,vx) from the complex env
            #decoder doesnt use the real action from the complex env, but use a zero action here
            #mu.squeeze(): make the mu from shape (1, latent_dim) to (latent_dim,)
            #decoder output:  reconstructed obs, reconstructed reward, logits, we only need the logits here
            _, _, logits = decoder(ptu.from_numpy(general_obs_map(env)),
                                   torch.zeros_like(torch.tensor([0.0])).to(DEVICE),
                                   0, mu.squeeze())
            #get the predicted label from the decoder. find the index of the max logit after softmax
            pred_label = int(torch.argmax(torch.nn.functional.softmax(logits), dim=0))

            export_true.append(true_label)
            # the decoder before is not correct, the simple action should not be zero all the time
            export_pred.append(pred_label)#??????这个decoder有问题

            # === 🧩 保存任务参数 spec（用于亮度编码） ===
            try:
                current_spec = float(spec_for_logging)  # 每个任务的目标值，如速度/位置
            except Exception:
                current_spec = 0.0  # 若取不到就用 0 填充

            # 同步保存
            if 'export_specs' not in locals():
                export_specs = []
            export_specs.append(current_spec)


            # 批量导出一次
            if len(export_latents) >= EXPORT_SAMPLES_PER_STEP:
                _write_tensorboard_step(EXPORT_ROOT, export_step_idx,
                                        export_latents, export_true, export_pred, export_specs)
                export_latents, export_true, export_pred, export_specs= [], [], [], []
                export_step_idx += 1


            #get the real observation from the complex env
            obs_before_sim = env._get_obs()
            # map the complex observation to simple observation, not in toy env
            #simple_obs_before: as input to the simple agent to get the simple action
            #simple_obs_before: as input to the decoder to get the task prediction
            simple_obs_before = general_obs_map(env)

            # Save latent vars
            #simple_obs_before: [pos_x, vel_x]  mu:[mu_1, mu_2, ..., mu_d]
            #policy_input: [pos_x, vel_x, mu_1, mu_2, ..., mu_d]
            policy_input = torch.cat([ptu.from_numpy(simple_obs_before), mu.squeeze()], dim=-1)
            #simple_agent(simple_obs + mu) to get the simple action
            simple_action = simple_agent.get_torch_actions(policy_input, deterministic=True)
            
            _,_, logits = decoder(ptu.from_numpy(simple_obs_before), simple_action, 0, mu.squeeze())
            task_prediction = torch.argmax(torch.nn.functional.softmax(logits), dim=0)
            # task_prediction = 0
            #?????should use decoder here to get the task prediction from the simple observation and simple action

            #use the simple_env to step with the simple action to get the simple observation after the simple step
            #_simple_obs: the next simple observation after stepping with simple action
            #_simple_obs: [pos_x, vel_x]  simple_obs after stepping the high level simple action in the toy env
            _simple_obs,_,_,_ = simple_env.step(simple_action.detach().cpu().numpy())
            #construct the simple_obs multi-task one-hot vector
            #simple_obs: [goal_front, goal_back, forward_vel, backward_vel]
            # EXAMPLE: simple_obs: tensor([0, 0, 2.5, 0]) means the subgoal is to run forward with velocity 2.5
            # this simple_obs will be used in the transfer function to get the complex action
            simple_obs = torch.zeros_like(torch.tensor(task)).to(DEVICE)
            
            #write the value of the _simple_obs to the correct position in the simple_obs one-hot vector according to the task_prediction
            #We use here task_prediction, instead of env.base_task, to test the performance of the decoder
            #If use env.base_task, it means we assume we know the true task in the transfer learning, which is not realistic, and may lead to data leakage
            if task_prediction in [env.config.get('tasks',{}).get('goal_front'), env.config.get('tasks',{}).get('goal_back')]:
               # use simple_action to decide the direction of the selected task, so the decoder decides the task, simple action decides the direction 
                if simple_action.item()>0:
                    simple_obs[env.config.get('tasks',{}).get('goal_front')] = _simple_obs[0].item()
                else:
                    simple_obs[env.config.get('tasks',{}).get('goal_back')] = _simple_obs[0].item()
            else:
                if simple_action.item()>0:
                    simple_obs[env.config.get('tasks',{}).get('forward_vel')] = np.clip(_simple_obs[1].item(), -3,3)
                else:
                    simple_obs[env.config.get('tasks',{}).get('backward_vel')] = np.clip(_simple_obs[1].item(), -3,3)


            #get the the base task prediction from the simple_obs one-hot vector
            #DEBUG: why not use task_prediction directly?
            base_task_pred = torch.argmax(torch.abs(simple_obs))
            #construct new action variable that will be used in the step predictor and for logging
            action = torch.zeros_like(simple_obs).to(DEVICE)
            action[base_task_pred] = simple_obs[base_task_pred]
            #desired_state: the desired subgoal that we want complex env to achieve
            desired_state = action
            #????? NOrmailize here only for position task?  DEBUG!!!!
            action_normalize = simple_obs_before[0].item() - simple_obs[0].item()
            #set the maximal steps for the step predictor
            max_steps = 20
            #calculate sim steps from step predictor, decide how many low level steps to simulate to reach the desired subgoal
            #input to the step predictor: current complex env observation and the desired subgoal
            sim_time_steps = int(torch.clamp(step_predictor.choose_action(obs, desired_state.detach().cpu().numpy(), use_torch=True).squeeze()*max_steps,1,max_steps))
            
            #debug for step predictor
            #sim_time_steps = 20

            sim_time_steps_list.append(sim_time_steps)


            # record the information about the current subgoal

            # 新增：把本步 subgoal 类型 + 数值、潜变量、状态等都记下来
            subgoal_idx = int(base_task_pred.item())  # 我们真正执行的目标维度
            subgoal_val = float(action[subgoal_idx].item())  # 这步的目标数值
            subgoal_kind_pred = int(task_prediction.item())  # decoder 的分类（用于对比）

            # 真实环境状态（before 和 after 都建议记）
            true_x_before = float(env.sim.data.qpos[0])
            true_vx_before = float(env.sim.data.qvel[0])


            episode_logs.append({
            "t": global_step,
            "episode": episode,
            "mu": mu.detach().float().cpu().numpy().tolist(),  # 便于后续关联
            "true_task_idx": int(env.base_task),               # 环境真值标签
            "pred_task_idx": subgoal_kind_pred,                # decoder 预测标签
            "exec_task_idx": subgoal_idx,                      # 实际执行的目标维度（可能与 pred 不同）
            "simple_action": float(simple_action.squeeze().detach().cpu().numpy()),
            "subgoal_value": subgoal_val,                      # 本步 subgoal 数值（x* 或 v*）
            "sim_time_steps": int(sim_time_steps),
            "x_before": true_x_before,
            "vx_before": true_vx_before,
            # 下面两个在推进 sim_time_steps 之后再回写
            "x_after": None,
            "vx_after": None,
            "low_level_r": None,   # 稍后写入 low_level_r
            "spec_of_episode": float(spec_for_logging),  # 这一集配置的目标规格（便于分析）
            })
            global_step += 1

            for i in range(sim_time_steps):
                #ptu.from_numpy(obs): trasnfer the numpy observation to torch tensor and send to DEVICE
                #transfer_function: input(complex_obs, simple_subgoal), output: complex_action
                complex_action = transfer_function.get_action(ptu.from_numpy(obs), simple_obs, return_dist=False)
                #based on the complex action to step the complex env and get the next observation
                next_obs, r, done, truncated, env_info = env.step(complex_action.detach().cpu().numpy(), healthy_scale = 0)
                obs = next_obs
                if video:
                    image = env.render()
                    frames.append(image)
                    image_info['reward'].append(r)
                    image_info['obs'].append(cheetah_to_simple_env_map(obs_before_sim, obs)[0])
                    image_info['base_task'].append(env.task)
                    image_info['complex_action'].append(complex_action)
                    image_info['simple_action'].append(simple_obs)
                    image_info['action'].append(simple_action.detach().cpu().numpy())



                episode_reward += r
            # After stepping the complex env for sim_time_steps, we get the next_obs and update the simple_env
            simple_env.sim.data.set_joint_qpos('boxslideX', env.sim.data.qpos[0])
            simple_env.sim.data.set_joint_qvel('boxslideX', env.sim.data.qvel[0])
            x_pos_curr.append(env.sim.data.qpos[0])
            x_vel_curr.append(env.sim.data.qvel[0])
            task_idx = env.base_task
            task_idx = torch.tensor([task_idx]).to("cpu")
            #DEBUG: not save the transition to the memory for decoder fine-tuning
            # memory.add(task_idx, ptu.from_numpy(simple_obs_before), simple_action, mu.squeeze())            
            #sim_time_steps_list.append(sim_time_steps)
            #low_level_r: the reward for the low level controller to reach the desired subgoal in minimal steps
            #DEBUG: the low level reward(velocity) design here might have problem, need to be checked

            if base_task_pred in [env.config.get('tasks',{}).get('goal_front')]:
                low_level_r = - np.abs(action[env.config['tasks']['goal_front']].detach().cpu().numpy()-env.sim.data.qpos[0])/np.abs(action_normalize)- beta * sim_time_steps
                low_level_r = np.clip(low_level_r, -2, 1)
            elif base_task_pred in [env.config.get('tasks',{}).get('goal_back')]:
                low_level_r = - np.abs(action[env.config['tasks']['goal_back']].detach().cpu().numpy()-env.sim.data.qpos[0] )/np.abs(action_normalize)- beta * sim_time_steps
                low_level_r = np.clip(low_level_r, -2, 1)
            elif base_task_pred in [env.config.get('tasks',{}).get('forward_vel')]:
                low_level_r = - np.abs(action[env.config['tasks']['forward_vel']].detach().cpu().numpy()-env.sim.data.qvel[0])/np.abs(action[env.config['tasks']['forward_vel']].item())- beta * sim_time_steps
                low_level_r = np.clip(low_level_r, -2, 1)
            elif base_task_pred in [env.config.get('tasks',{}).get('backward_vel')]:
                low_level_r = - np.abs(action[env.config['tasks']['backward_vel']].detach().cpu().numpy()-env.sim.data.qvel[0])/np.abs(action[env.config['tasks']['backward_vel']].item())- beta * sim_time_steps
                low_level_r = np.clip(low_level_r, -2, 1)
            elif base_task_pred in [env.config.get('tasks',{}).get('stand_front')]:
                low_level_r = - np.abs(action[env.config['tasks']['stand_front']].detach().cpu().numpy()-env.sim.data.qpos[2])/np.abs(action[env.config['tasks']['stand_front']].item()) - beta * sim_time_steps
                low_level_r = np.clip(low_level_r, -2, 1)
            elif base_task_pred in [env.config.get('tasks',{}).get('stand_back')]:
                low_level_r = - np.abs(action[env.config['tasks']['stand_back']].detach().cpu().numpy()-env.sim.data.qpos[2])/np.abs(action[env.config['tasks']['stand_back']].item()) - beta * sim_time_steps
                low_level_r = np.clip(low_level_r, -2, 1)
            elif base_task_pred in [env.config.get('tasks',{}).get('jump')]:
                low_level_r = - np.abs(action[env.config['tasks']['jump']].detach().cpu().numpy()-np.abs(env.sim.data.qvel[1]))/np.abs(action[env.config['tasks']['jump']].item()) - beta * sim_time_steps
                low_level_r = np.clip(low_level_r, -2, 1)


            # 循环推进环境之后：
            #append the after-simulation state to the last log entry
            last = episode_logs[-1]
            last["x_after"] = float(env.sim.data.qpos[0])
            last["vx_after"] = float(env.sim.data.qvel[0])
            last["low_level_r"] = float(low_level_r)  # 这个变量你在下面 reward 分支算好了

            
            # transfer the desired_state(one hot+ value) to float32 tensor
            desired_state = desired_state.to(torch.float32)
            #save a SAC style transition to the step predictor replay buffer
            # obs_before_sim: complex observation before the low level simulation
            # next_obs: complex observation after the low level simulation
            # low_level_r: the reward for the low level controller to reach the desired subgoal in minimal steps
            # sim_time_steps: the number of low level steps simulated
            # desired_state: the desired subgoal that we want complex env to achieve
            # step_predictor training
            step_predictor.store(obs_before_sim, low_level_r, done, np.array([sim_time_steps]), next_obs, desired_state)
            low_level_losses = step_predictor.train(episode, False)
            #map the complex observation after LL simulation to the simple env, which will be used in encoder
            simple_obs_after = general_obs_map(env)
            
            #create a vector shaped (1, obs_dim + 1 + obs_dim)
            data = torch.cat([ptu.from_numpy(simple_obs_before), torch.unsqueeze(torch.tensor(r, device=DEVICE), dim=0), ptu.from_numpy(simple_obs_after)], dim=0).unsqueeze(dim=0)
            #update the context and therfore mu can be also updated
            context = torch.cat([contexts.squeeze(), data], dim=0)
            contexts = context[-time_steps:, :]
            contexts = contexts.unsqueeze(0).to(torch.float32)


            #DEBUG: because the memory.add is commented, decoder there will not be fine-tuned
            if len(memory) < memory.batch_size:
                continue
            else: 
                batch = memory.sample(memory.batch_size)
                tasks_batch, simple_obs_batch, simple_action_batch, mu_batch = memory.unpack(batch)
                _,_, logits_batch = decoder(simple_obs_batch.detach(), simple_action_batch.detach(), 0, mu_batch.squeeze().detach())
                loss = loss_criterion(logits_batch.squeeze(), tasks_batch.squeeze().detach())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                _loss.append(loss)


        # Collect rewards for box plots
        if env.base_task in [env.config.get('tasks',{}).get('goal_front'), env.config.get('tasks',{}).get('goal_back')]:
            # Position task
            rewards_data[current_inference_path_name]['position'].append(episode_reward)
        elif env.base_task in [env.config.get('tasks',{}).get('forward_vel'), env.config.get('tasks',{}).get('backward_vel')]:
            # Velocity task
            rewards_data[current_inference_path_name]['velocity'].append(episode_reward)

        if len(_loss)>0:
            loss_history.append(torch.stack(_loss).mean().detach().cpu().numpy())


        if episode % save_after_episodes == 0 and episode!=0:
            file = os.path.join(save_video_path, 'weights/retrained_decoder.pth')
            torch.save(decoder.state_dict(), file)
        # video = False
        if video:
            # save_plot(np.array(loss_history), name='task_loss', path=f'{os.getcwd()}/delete_videos')
            size = frames[0].shape

            # Save to corresponding repo
            fps=20
            save_as = f'{save_video_path}/videos/transfer_{episode}.mp4'
            # video = cv2.VideoWriter(save_as, cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), True)
            # Write frames to video
            _frames_to_gif(frames, image_info, save_as)
            # video.release()

        if env.base_task in [env.config.get('tasks',{}).get('goal_front'), env.config.get('tasks',{}).get('goal_back')]:
            tasks_pos.append(task[env.base_task])
            x_pos_plot.append(x_pos_curr)
            pos_reward += np.clip(episode_reward, -2*path_length*sim_time_steps, 0)
        elif env.base_task in [env.config.get('tasks',{}).get('forward_vel'), env.config.get('tasks',{}).get('backward_vel')]:
            x_vel_plot.append(x_vel_curr)
            tasks_vel.append(task[env.base_task])
            vel_reward += episode_reward
        if tasks is None:
            if len(x_pos_plot) > 5:
                x_pos_plot.pop(0)
                tasks_pos.pop(0)
            if len(x_vel_plot) > 5:
                x_vel_plot.pop(0)
                tasks_vel.pop(0)

        # ✅ 在这里添加步长分布直方图绘制
        # import numpy as np, matplotlib.pyplot as plt, os

        if len(sim_time_steps_list) > 0:
            plt.figure(figsize=(6,4))
            plt.hist(sim_time_steps_list,
                     bins=np.arange(1, max_steps+2)-0.5,
                     rwidth=0.8,
                     color='steelblue',
                     edgecolor='black')
            plt.xticks(range(1, max_steps+1))
            plt.xlabel('Predicted sim_time_steps')
            plt.ylabel('Frequency')
            plt.title(f'Episode {episode}: Step Predictor Output Distribution')

            avg_steps = np.mean(sim_time_steps_list)
            print(f"[Episode {episode}] step_predictor mean={avg_steps:.2f}, "
                  f"min={np.min(sim_time_steps_list)}, max={np.max(sim_time_steps_list)}")

            save_dir = os.path.join(inference_path, "plots", "step_hist")
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f"step_dist_ep{episode}.png"), dpi=200)
            plt.close()
        # ✅ 到此结束，下面继续原来的轨迹绘图逻辑        


        if episode%plot_every == 0 or tasks:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
            colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'orange', 'purple', 'brown']
            window_size = 10

            def moving_average(data, window_size):
                # """Compute the moving average using a sliding window."""
                # window = np.ones(int(window_size))/float(window_size)
                # return np.convolve(data, window, 'same')
                from scipy.ndimage.filters import gaussian_filter1d
                return gaussian_filter1d(data, sigma=2)

            for i, x_pos in enumerate(x_pos_plot):
                color = f'C{i}'
                x_pos = moving_average(x_pos, window_size)
                # Plot position on the first (left) axis
                ax1.plot(np.arange(len(x_pos)), np.array(x_pos), label='Position', color=color)
                # if tasks[i][0]!=0:
                ax1.plot(np.arange(len(x_pos)), np.ones(len(x_pos))*tasks_pos[i], linestyle='--', color=color, alpha=0.5)
            
            ax1.set_xlabel('Time (s)', fontsize=32)
            ax1.set_ylabel('Position (m)', fontsize=32)
            ax1.tick_params(axis='y', labelsize=24)
            ax1.tick_params(axis='x', labelsize=24)
            # ax1.set_title(f'Avg Reward: {pos_reward/(episode+1)}')

            # Create a second axis sharing the same x-axis
            for i, x_vel in enumerate(x_vel_plot):
                color = f'C{i}'
                x_vel = moving_average(x_vel, window_size)
                ax2.plot(np.arange(len(x_vel)), np.array(x_vel), label='Velocity', color=color)
                # if tasks[i][3]!=0: 
                ax2.plot(np.arange(len(x_vel)), np.ones(len(x_vel))*tasks_vel[i], linestyle='--', color=color, alpha=0.5)
            ax2.set_xlabel('Time (s)', fontsize=32)
            ax2.tick_params(axis='y', labelsize=24)
            ax2.tick_params(axis='x', labelsize=24)
            ax2.set_ylabel('Velocity (m/s)', fontsize=32)
            # ax2.set_title(f'Avg Reward: {vel_reward/(episode+1)}')
            plt.tight_layout()
            plt.subplots_adjust(hspace=0.4)

            # Save the figure to a fileHalfCheetahMixtureEnv
            dir = Path(os.path.join('{os.getcwd()}/trajectories', experiment_name, current_inference_path_name, 'beta0.1_old'))
            filename = os.path.join(dir,f"epoch_{episode}.png")
            filename = os.path.join(dir,f"epoch_{episode}.pdf")

            if tasks:
                filename = os.path.join(dir,f"final.png")
                filename = os.path.join(dir,f"final_{max_steps}.pdf")
            dir.mkdir(exist_ok=True, parents=True)
            plt.savefig(filename, dpi=300)  # Save the figure with 300 dpi resolution
            plt.close()
            # === 🧩 写出最后未满批次的数据 ===
            if len(export_latents) > 0:
                _write_tensorboard_step(EXPORT_ROOT, export_step_idx,
                                        export_latents, export_true, export_pred, export_specs)
                print(f"[INFO] Saved final latent batch to step {export_step_idx}")
            print(f"[INFO] Latent embeddings exported under {EXPORT_ROOT}")
        # ✅ 检查采样均衡性（插在 for episode 的结尾）
        if (episode + 1) % 4 == 0 or episode == num_trajectory - 1:
            from collections import Counter
            if len(export_true) > 0 and len(export_specs) > 0:
                print("\n================= Balanced Sampling Check =================")
                print(f"[Episode {episode}] Task distribution: {Counter(export_true)}")
                print(f"[Episode {episode}] Spec range: {min(export_specs):.3f} to {max(export_specs):.3f}")
                print("============================================================\n")
        

        if len(episode_logs) > 0:
            logs_dir = os.path.join(inference_path, "logs")
            os.makedirs(logs_dir, exist_ok=True)
            df_ep = pd.DataFrame(episode_logs)
            csv_path = os.path.join(logs_dir, f"subgoals_ep{episode}.csv")
            df_ep.to_csv(csv_path, index=False)
            print(f"[LOG] Saved subgoal trace to {csv_path}")

        # === 🧩 Unified Subgoal Tracking Plot: real rollout trajectory + subgoals ===
        try:
            df = pd.DataFrame([d for d in episode_logs if d["episode"] == episode])

            # 根据任务类型选轨迹字段
            POS_TASKS = [
                env.config['tasks'].get('goal_front'),
                env.config['tasks'].get('goal_back')
            ]
            VEL_TASKS = [
                env.config['tasks'].get('forward_vel'),
                env.config['tasks'].get('backward_vel')
            ]

            is_pos = df["exec_task_idx"].isin(POS_TASKS).any()
            is_vel = df["exec_task_idx"].isin(VEL_TASKS).any()

            # 统一成一个绘图
            fig, ax = plt.subplots(figsize=(12,5))

            # ---- 1) 画真实 rollouts ----
            if is_pos:
                ax.plot(df.index, df["x_after"], label="Real x (after)", linewidth=2)
                ylabel = "x (m)"
            else:
                ax.plot(df.index, df["vx_after"], label="Real vx (after)", linewidth=2)
                ylabel = "vx (m/s)"

            # ---- 2) 画 subgoal 轨迹 ----
            ax.step(df.index, df["subgoal_value"], where="post",
                    linestyle="--", linewidth=2, label="Subgoal value (x* / vx*)")

            # ---- 3) 画 subgoal 是否 track 成功（颜色展示） ----
            # 计算误差
            if is_pos:
                err = abs(df["x_after"] - df["subgoal_value"])
            else:
                err = abs(df["vx_after"] - df["subgoal_value"])

            # 用散点颜色展示误差大小
            sc = ax.scatter(df.index, 
                            df["subgoal_value"], 
                            c=err, cmap="coolwarm", s=80,
                            label="tracking error")

            plt.colorbar(sc, ax=ax, label="|real - subgoal|")

            # ---- 细节设置 ----
            ax.set_title(f"Episode {episode} – Unified Subgoal Tracking", fontsize=18)
            ax.set_xlabel("Global time step (subgoal step)")
            ax.set_ylabel(ylabel)
            ax.legend()
            plt.tight_layout()

            # 保存图
            save_dir = os.path.join(inference_path, "plots", "unified_subgoal")
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f"unified_ep{episode}.png"), dpi=200)
            plt.close()

            print(f"[PLOT] Saved unified subgoal tracking figure for episode {episode}")

        except Exception as e:
            print("[ERROR] unified subgoal plotting failed:", e)






if __name__ == "__main__":
    from configs.transfer_config import transfer_config as config

    inference_path = config['inference_path']
    complex_agent = config['complex_agent']

    # Initialize rewards data
    rewards_data = {}

    # Loop over inference paths
    # for inference_path in inference_paths:

    print("inference_path:", inference_path['path'])
    current_inference_path_name = inference_path['name']
    inference_path = inference_path['path']
    rewards_data[current_inference_path_name] = {'velocity': [], 'position': []}

    with open(complex_agent['experiments_repo'] + complex_agent['experiment_name'] + '/config.json', 'r') as file:
        env_config = json.load(file)

    if env_config['env'] == 'hopper':
        env = HopperGoal()
    elif env_config['env'] == 'walker':
        env = WalkerGoal()
    elif env_config['env'] == 'half_cheetah_multi':
        env = HalfCheetahMixtureEnv(env_config)
    elif env_config['env'] == 'hopper_multi':
        env = HopperMulti(env_config)
    elif env_config['env'] == 'walker_multi':
        env = WalkerMulti(env_config)
    elif env_config['env'] == 'ant_multi':
        env = AntMulti()
    env.render_mode = 'rgb_array'

    with open(f'{inference_path}/variant.json', 'r') as file:
        variant = json.load(file)

    m = variant['algo_params']['sac_layer_size']
    simple_env = ENVS[variant['env_name']](**variant['env_params'])         # Just used for initilization purposes

    ### PARAMETERS ###
    obs_dim = int(np.prod(simple_env.observation_space.shape))
    action_dim = int(np.prod(simple_env.action_space.shape))
    net_complex_enc_dec = variant['reconstruction_params']['net_complex_enc_dec']
    latent_dim = variant['algo_params']['latent_size']
    time_steps = variant['algo_params']['time_steps']
    num_classes = variant['reconstruction_params']['num_classes']
    # max_path_len = variant['algo_params']['max_path_length']
    reward_dim = 1
    encoder_input_dim = time_steps * (obs_dim + reward_dim + obs_dim)
    shared_dim = int(encoder_input_dim / time_steps * net_complex_enc_dec)
    if variant['algo_params']['sac_context_type']  == 'sample':
        policy_latent_dim = latent_dim
    else:
        policy_latent_dim  = latent_dim * 2

    
    bnp_model = BNPModel(
        save_dir=variant['dpmm_params']['save_dir'],
        start_epoch=variant['dpmm_params']['start_epoch'],
        gamma0=variant['dpmm_params']['gamma0'],
        num_lap=variant['dpmm_params']['num_lap'],
        fit_interval=variant['dpmm_params']['fit_interval'],
        kl_method=variant['dpmm_params']['kl_method'],
        birth_kwargs=variant['dpmm_params']['birth_kwargs'],
        merge_kwargs=variant['dpmm_params']['merge_kwargs']
    )

    step_predictor = SAC(n_states=env.observation_space.shape[0],
            n_actions=1,
            task_dim = env.task.shape[0],   # desired state
            hidden_layers_actor = [64,64,64,64,64],
            hidden_layers_critic = [64,64,64,64,64],
            memory_size=1e+6,
            batch_size=512,
            gamma=0.9,
            alpha=0.2,
            lr=3e-4,
            action_bounds=[-50,50],
            reward_scale=1, 
            device=DEVICE
            # pretrained=dict(path='/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_08_25_12_38_14_default_true_gmm/cheetah_26_08', epoch='low_level')
            )

    memory = Memory(1e+6)
    encoder = get_encoder(inference_path, shared_dim, encoder_input_dim)
    simple_agent = get_simple_agent(inference_path, obs_dim, policy_latent_dim, action_dim, m)
    transfer_function = get_complex_agent(env, complex_agent)
    output_action_dim = env.task.shape[0]
    decoder = get_decoder(inference_path, action_dim, obs_dim, reward_dim, latent_dim, output_action_dim, net_complex_enc_dec, variant)
    optimizer = optim.Adam(decoder.parameters(), lr=3e-4)

    ### ROLLOUT ###
    rollout(env, encoder, decoder, optimizer, simple_agent,  step_predictor,
                                    transfer_function, memory, variant, obs_dim, action_dim, 
                                    max_path_len, n_tasks=1, inner_loop_steps=10, save_video_path=inference_path, experiment_name=complex_agent['experiment_name'],
                                    current_inference_path_name=current_inference_path_name)
    
    '''
    After the striding predictor is trained, plot the results with symmetric goals
    '''
    # taskid: order = ['goal_front:0', 'goal_back:1', 'forward_vel:2', 'backward_vel:3']
    tasks = [
        {'base_task':'goal_back', 'specification':0.9},
        {'base_task':'goal_back', 'specification':0.5},
        {'base_task':'goal_back', 'specification':0.3},
        {'base_task':'goal_front', 'specification':0.3},
        {'base_task':'goal_front', 'specification':0.5},
        {'base_task':'goal_front', 'specification':0.9},
        {'base_task':'backward_vel', 'specification':0.9},
        {'base_task':'backward_vel', 'specification':0.5},
        {'base_task':'backward_vel', 'specification':0.1},
        # {'base_task':'backward_vel', 'specification':1.0},
        # {'base_task':'forward_vel', 'specification':1.0},
        {'base_task':'forward_vel', 'specification':0.1},
        {'base_task':'forward_vel', 'specification':0.5},
        {'base_task':'forward_vel', 'specification':0.9},
                ]
    rollout(env, encoder, decoder, optimizer, simple_agent, step_predictor,
                                    transfer_function, memory, variant, obs_dim, action_dim, 
                                    max_path_len, n_tasks=1, inner_loop_steps=10, save_video_path=inference_path, experiment_name=complex_agent['experiment_name'],
                                    current_inference_path_name=current_inference_path_name, tasks=tasks)


    '''
    Create the box plot
    '''
    #import matplotlib.pyplot as plt

    # Assuming inference_paths and rewards_data are already defined

    boxplot_data = []
    x_labels = []
    positions = []
    pos = 1  # Starting position for the first box

    #for i, inference_path in enumerate(inference_path):
    inference_name = current_inference_path_name
    # Get rewards for position and velocity tasks
    position_rewards = rewards_data[inference_name]['position']
    velocity_rewards = rewards_data[inference_name]['velocity']
    
    # Append data
    boxplot_data.extend([position_rewards, velocity_rewards])
    
    # Append labels
    x_labels.extend([f"{inference_name}\nPosition", f"{inference_name}\nVelocity"])
    
    # Append positions
    positions.extend([pos, pos + 1])
    
    # Update position for next inference path
    pos += 3  # Adding space between groups

    # Create the box plot and retrieve the dictionary of artists
    plt.figure(figsize=(12, 6))
    box = plt.boxplot(boxplot_data, positions=positions, widths=0.6, showfliers=False, patch_artist=True)

    # Customize median colors
    medians = box['medians']
    for i, median in enumerate(medians):
        if i % 2 == 0:  # Even index: Position
            median.set_color('blue')    # Set color for position medians
            median.set_linewidth(2)     # Optional: set line width
        else:           # Odd index: Velocity
            median.set_color('green')     # Set color for velocity medians
            median.set_linewidth(2)     # Optional: set line width

    # Optional: Customize box colors for better visualization
    boxes = box['boxes']
    for i, box_patch in enumerate(boxes):
        if i % 2 == 0:  # Even index: Position
            box_patch.set_facecolor('#ADD8E6')  # Light blue
        else:           # Odd index: Velocity
            box_patch.set_facecolor('#90EE90')  # Light pink

    # Set x-axis labels
    plt.xticks(positions, x_labels, rotation=45, ha='right')

    # Set y-axis label
    plt.ylabel('Rewards')

    # Add grid lines
    plt.grid(True, linestyle='--', alpha=0.5)

    # Optional: Add a legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#ADD8E6', edgecolor='blue', label='Position'),
                    Patch(facecolor='#90EE90', edgecolor='green', label='Velocity')]
    plt.legend(handles=legend_elements, loc='lower right')

    # Adjust layout
    plt.tight_layout()

    # Save the plot
    plt.savefig(f'{os.getcwd()}/rewards_boxplot.png', dpi=300)
    plt.savefig(f'{os.getcwd()}/rewards_boxplot.pdf', dpi=300)

    # Show the plot
    plt.show()