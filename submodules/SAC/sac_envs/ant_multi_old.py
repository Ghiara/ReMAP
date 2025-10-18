#This file is a modified version of the original ant_multi.py file,
#which uses the "old" architecture, to be adapted with "train_low_level_policy.py"
#the "old" architecture is the one also used in Cheetah, hopper and walker
#written by Chongjin


import gym
from sac_envs.base_envs.ant import AntEnv
from gym.utils.ezpickle import EzPickle
from gym import utils
from gym.spaces import Box
from typing import List, Tuple
import numpy as np
import mujoco_py
from meta_envs.pygame_rendering import PygameRenderer



class AntMulti(AntEnv, utils.EzPickle):
    
    def __init__(self, config, healthy_scale = 1, render_mode: str = 'rgb_array', *args, **kwargs):
        super().__init__(render_mode=render_mode, *args, **kwargs)
        self.observation_space = Box(
                low=-np.inf, high=np.inf, shape=(32,), dtype=np.float64
            )
        self.healthy_scale = healthy_scale
        self.screen_height = 400
        self.screen_width = 400
        self.config = config
        self.task = self.sample_task()

        # epoch counter
        self.current_epoch = 0

    # def render(self, mode: str = None, width: int = 800, height: int = 400):
    #     if mode is None: mode = self.render_mode

    #     self.sim.model.cam_pos[0][0] = self.get_body_com('torso')[0]
    #     self.sim.model.cam_pos0[0][0] = self.get_body_com('torso')[0]
    #     img = self.sim.render(
    #         width=width,
    #         height=height,
    #         camera_name=None,
    #     )
    #     img = np.rot90(img, 2)

    #     if mode != 'human': 
    #         return img
    #     if self.renderer is None:
    #         self.renderer = PygameRenderer()
    #     self.renderer.render_image(img, title="Half-Cheetah")
    #     self.renderer.clock.tick(self.metadata['render_fps'])
        
    def _initialize_camera(self):
        # set camera parameters for viewing
        sim = self.sim
        self.viewer = mujoco_py.MjRenderContextOffscreen(sim)
        camera = self.viewer.cam
        camera.type = 1
        camera.trackbodyid = 0
        camera.elevation = -20
        sim.add_render_context(self.viewer)

    
    def set_epoch(self, epoch):

        """训练脚本每个 epoch 开始时调用"""
        self.current_epoch = epoch

     #----------------------------------------------------------   
    def step(self, action, healthy_scale=None):
        if healthy_scale is not None:
            self.healthy_scale = healthy_scale

        xpos_before = self.get_body_com('torso')[0]
        state, reward, terminated, _, info = super().step(action)
        xpos_after = self.get_body_com('torso')[0]
        ob = self._get_obs()

        # --- 判定 terminated ---
        s = self.state_vector()
        finite_ok = np.isfinite(s).all()
        z = float(self.sim.data.qpos[2])
        height_ok = z > 0.18
        terminated = not (finite_ok and height_ok)

        healthy_reward = 1.0        # 存活奖励
        if self.current_epoch < self.config.get("use_termination_after", 0):
            terminated = False

        healthy_penalty = 0
        if terminated:
            healthy_penalty = -10   # 摔倒惩罚
            healthy_reward = 0.0

        # --- 按任务类型计算奖励 ---
        if self.base_task in [self.config.get('tasks', {}).get('goal_left'),
                            self.config.get('tasks', {}).get('goal_right')]:
            # Goal tracking
            reward_run = - np.abs(xpos_after - self.task[self.base_task]) / (np.abs(self.task[self.base_task]) + 1e-6)
            reward_ctrl = -0.1 * np.sum(np.square(action))
            reward = reward_run + reward_ctrl + healthy_reward * self.healthy_scale + healthy_penalty

        elif self.base_task in [self.config.get('tasks', {}).get('velocity_left'),
                                self.config.get('tasks', {}).get('velocity_right')]:
            # Velocity tracking
            forward_vel = (xpos_after - xpos_before) / self.dt
            reward_run = - np.abs(forward_vel - self.task[self.base_task]) / (np.abs(self.task[self.base_task]) + 1e-6)
            reward_ctrl = -0.1 * np.sum(np.square(action))
            reward = reward_run + reward_ctrl + healthy_reward * self.healthy_scale + healthy_penalty



        # if self.base_task in [self.config.get('tasks', {}).get('goal_left'),
        #                     self.config.get('tasks', {}).get('goal_right')]:
        #     # Goal (position) tracking
        #     target = self.task[self.base_task]
        #     dist_before = abs(xpos_before - target)
        #     dist_after = abs(xpos_after - target)
        #     reward_progress = dist_before - dist_after   # 朝目标靠近就有正奖励
        #     reward_ctrl = -0.1 * np.sum(np.square(action))
        #     reward = 3.0 * reward_progress + 0.2 * healthy_reward * self.healthy_scale + reward_ctrl + healthy_penalty

        # elif self.base_task in [self.config.get('tasks', {}).get('velocity_left'),
        #                         self.config.get('tasks', {}).get('velocity_right')]:
        #     # Velocity tracking
        #     forward_vel = (xpos_after - xpos_before) / self.dt
        #     target = self.task[self.base_task]
        #     reward_run = - abs(forward_vel - target) / (abs(target) + 1e-6)
        #     reward_ctrl = -0.1 * np.sum(np.square(action))
        #     reward = 3.0 * reward_run + 0.2 * healthy_reward * self.healthy_scale + reward_ctrl + healthy_penalty

        else:
            raise RuntimeError("base task not recognized")

        # --- 返 ---
        # return ob, reward, terminated, False, dict(
        #         xpos=xpos_after,
        #         reward_task=reward_progress if 'reward_progress' in locals() else reward_run,
        #         reward_ctrl=reward_ctrl,
        #         true_task=self.task
        # )
        return ob, reward, terminated, False, dict(
            reward_run=reward_run,
            reward_ctrl=reward_ctrl,
            true_task=self.task
        )


    
    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
            self.get_body_com("torso").flat, 
        ]).astype(np.float32).flatten()



    def update_task(self, task):
        self.task = task
        # 如果任务是 goal_x (位置任务)，就把红球移过去
        if self.base_task in [self.config.get('tasks', {}).get('goal_left'),
                            self.config.get('tasks', {}).get('goal_right')]:
            goal_x = self.task[self.base_task]
            geom_id = self.sim.model.geom_name2id("goal")
            self.sim.model.geom_pos[geom_id] = np.array([goal_x, 0.0, 0.5])  # 放在x轴对应位置


    def update_base_task(self, base_task):
        self.base_task = base_task

    def reset(self):
        obs = super().reset()
        # new_obs = np.append(self.get_body_com("torso")[0], obs[0])
        new_obs = self._get_obs()
        return new_obs, {}
    
    
    
    def random_reset(self, x_pos_range=[-10,10], x_vel_range=[-0.1,0.1]):
        obs = super().reset()
        # new_obs = np.append(self.get_body_com("torso")[0], obs[0])

        qpos = self.init_qpos + self.np_random.uniform(
            low=-0.1, high=0.1, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.standard_normal(self.model.nv) * 0.1
        qpos[0] = np.random.random() * (x_pos_range[1] - x_pos_range[0]) + x_pos_range[0]
        qvel[0] = np.random.random() * (x_vel_range[1] - x_vel_range[0]) + x_vel_range[0]
        self.set_state(qpos, qvel)

        new_obs = self._get_obs()
        return new_obs, {}
    
    
    
    def sample_task(self, task=None):
    
        self.task = np.zeros(max(self.config['tasks'].values()) + 1)

        base_task = np.random.choice(list(self.config['tasks'].keys()))
        self.base_task = self.config.get('tasks', {}).get(base_task)
        mult = np.random.random()

        if task:
            base_task = task['base_task']
            self.base_task = self.config.get('tasks', {}).get(base_task)
            mult = task['specification']

        if base_task in ['goal_left']:
            self.task[self.base_task] = - (mult * (self.config['max_goal'][1] - self.config['max_goal'][0])+ self.config['max_goal'][0])


        elif base_task in ['goal_right']:
            self.task[self.base_task] = mult * (self.config['max_goal'][1] - self.config['max_goal'][0]) + self.config['max_goal'][0]       
            
        elif base_task in ['velocity_left']:
            self.task[self.base_task] = - (mult * (self.config['max_vel'][1] - self.config['max_vel'][0]) + self.config['max_vel'][0])
        elif base_task in ['velocity_right']:
            self.task[self.base_task] = mult * (self.config['max_vel'][1] - self.config['max_vel'][0]) + self.config['max_vel'][0]
        
        elif base_task == 'jump':
            self.task[self.base_task] = mult * (self.config['max_jump'][1] - self.config['max_jump'][0]) + self.config['max_jump'][0]

        else:
            raise ValueError("Task not found")
        return self.task
    