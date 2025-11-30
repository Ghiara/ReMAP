# #This file is a modified version of the original ant_multi.py file,
# #which uses the "old" architecture, to be adapted with "train_low_level_policy.py"
# #the "old" architecture is the one also used in Cheetah, hopper and walker
# #written by bo


# import gym
# from sac_envs.base_envs.ant import HalfCheetahEnv
# from gym.utils.ezpickle import EzPickle
# from gym import utils
# from gym.spaces import Box
# from typing import List, Tuple
# import numpy as np
# import mujoco_py
# from meta_envs.pygame_rendering import PygameRenderer



# class HalfCheetahMultiVel(HalfCheetahEnv, utils.EzPickle):
    
#     def __init__(self, config, healthy_scale = 1, render_mode: str = 'rgb_array', *args, **kwargs):
#         super().__init__(render_mode=render_mode, *args, **kwargs)
#         self.observation_space = Box(
#                 low=-np.inf, high=np.inf, shape=(32,), dtype=np.float64
#             )
#         self.healthy_scale = healthy_scale
#         self.screen_height = 400
#         self.screen_width = 400
#         self.config = config
#         self.task = self.sample_task()

#         # epoch counter
#         self.current_epoch = 0

#     def _initialize_camera(self):
#         # set camera parameters for viewing
#         sim = self.sim
#         self.viewer = mujoco_py.MjRenderContextOffscreen(sim)
#         camera = self.viewer.cam
#         camera.type = 1
#         camera.trackbodyid = 0
#         camera.elevation = -20
#         sim.add_render_context(self.viewer)

#     def set_epoch(self, epoch):

#         """训练脚本每个 epoch 开始时调用"""
#         self.current_epoch = epoch

#     def step(self, action, healthy_scale=None):

#         if healthy_scale is not None:
#             self.healthy_scale = healthy_scale

#         xpos_before = self.get_body_com('torso')[0]
#         state, reward, terminated, _, info = super().step(action)
#         xpos_after = self.get_body_com('torso')[0]
#         ob = self._get_obs()

#         # --- 判定 terminated ---
#         s = self.state_vector()
#         finite_ok = np.isfinite(s).all()
#         z = float(self.sim.data.qpos[2])
#         height_ok = z > 0.18
#         terminated = not (finite_ok and height_ok)

#         healthy_reward = 1.0        # 存活奖励
#         if self.current_epoch < self.config.get("use_termination_after", 0):
#             terminated = False

#         healthy_penalty = 0
#         if terminated:
#             healthy_penalty = -10   # 摔倒惩罚
#             healthy_reward = 0

#         # --- 按任务类型计算奖励 ---
#         if self.base_task in [self.config.get('tasks', {}).get('goal_left'),
#                             self.config.get('tasks', {}).get('goal_right')]:
#             # Goal tracking
#             reward_run = - np.abs(xpos_after - self.task[self.base_task]) / (np.abs(self.task[self.base_task]) + 1e-6)
#             reward_ctrl = -0.1 * np.sum(np.square(action))
#             reward = reward_run + reward_ctrl + healthy_reward * self.healthy_scale + healthy_penalty

#         elif self.base_task in [self.config.get('tasks', {}).get('backward_vel'),
#                                 self.config.get('tasks', {}).get('forward_vel')]:
#             # Velocity tracking
#             forward_vel = (xpos_after - xpos_before) / self.dt
#             reward_run = - np.abs(forward_vel - self.task[self.base_task]) / (np.abs(self.task[self.base_task]) + 1e-6)
#             reward_ctrl = -0.1 * np.sum(np.square(action))
#             reward = reward_run + reward_ctrl + healthy_reward * self.healthy_scale + healthy_penalty

#         else:
#             raise RuntimeError("base task not recognized")

#         # --- 返回 ---
#         return ob, reward, terminated, False, dict(
#             reward_run=reward_run,
#             reward_ctrl=reward_ctrl,
#             true_task=self.task
#         )

    
#     def _get_obs(self):
#         return np.concatenate([
#             self.sim.data.qpos.flat,
#             self.sim.data.qvel.flat,
#             self.get_body_com("torso").flat, 
#         ]).astype(np.float32).flatten()



#     def update_task(self, task):
#         self.task = task

#     def update_base_task(self, base_task):
#         self.base_task = base_task

#     def reset(self):
#         obs = super().reset()
#         # new_obs = np.append(self.get_body_com("torso")[0], obs[0])
#         new_obs = self._get_obs()
#         return new_obs, {}
    
    
    
#     def random_reset(self, x_pos_range=[-10,10], x_vel_range=[-0.1,0.1]):
#         obs = super().reset()
#         # new_obs = np.append(self.get_body_com("torso")[0], obs[0])

#         qpos = self.init_qpos + self.np_random.uniform(
#             low=-0.1, high=0.1, size=self.model.nq
#         )
#         qvel = self.init_qvel + self.np_random.standard_normal(self.model.nv) * 0.1
#         qpos[0] = np.random.random() * (x_pos_range[1] - x_pos_range[0]) + x_pos_range[0]
#         qvel[0] = np.random.random() * (x_vel_range[1] - x_vel_range[0]) + x_vel_range[0]
#         self.set_state(qpos, qvel)

#         new_obs = self._get_obs()
#         return new_obs, {}
    
    
    
#     def sample_task(self, task=None):
    
#         self.task = np.zeros(max(self.config['tasks'].values()) + 1)

#         base_task = np.random.choice(list(self.config['tasks'].keys()))
#         self.base_task = self.config.get('tasks', {}).get(base_task)
#         mult = np.random.random()

#         if task:
#             base_task = task['base_task']
#             self.base_task = self.config.get('tasks', {}).get(base_task)
#             mult = task['specification']

#         if base_task in ['goal_left']:
#             self.task[self.base_task] = - (mult * (self.config['max_goal'][1] - self.config['max_goal'][0])+ self.config['max_goal'][0])


#         elif base_task in ['goal_right']:
#             self.task[self.base_task] = mult * (self.config['max_goal'][1] - self.config['max_goal'][0]) + self.config['max_goal'][0]       
            
#         elif base_task in ['backward_vel']:
#             self.task[self.base_task] = - (mult * (self.config['max_vel'][1] - self.config['max_vel'][0]) + self.config['max_vel'][0])
#         elif base_task in ['forward_vel']:
#             self.task[self.base_task] = mult * (self.config['max_vel'][1] - self.config['max_vel'][0]) + self.config['max_vel'][0]
        
#         elif base_task == 'jump':
#             self.task[self.base_task] = mult * (self.config['max_jump'][1] - self.config['max_jump'][0]) + self.config['max_jump'][0]

#         else:
#             raise ValueError("Task not found")
#         return self.task



#--------------------------------------------------------------------

# cheetah_velocity_single.py
# Only-velocity HalfCheetah env, compatible with train_low_level_policy.py
# written by bo (remixed from AntMulti)

import gym
import numpy as np
from gym import utils
from gym.spaces import Box

from sac_envs.base_envs.half_cheetah import HalfCheetahEnv


class CheetahVelocityEnv(HalfCheetahEnv, utils.EzPickle):

    def __init__(self, config, healthy_scale: float = 1.0,
                 render_mode: str = "rgb_array", *args, **kwargs):
        super().__init__(render_mode=render_mode, *args, **kwargs)

        # 直接根据 _get_obs() 的输出设置 obs 维度，避免手动写数字
        obs_example = self._get_obs()
        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=obs_example.shape,
            dtype=np.float32
        )

        self.config = config
        self.healthy_scale = healthy_scale

        # epoch counter，用于控制何时启用摔倒终止（如果需要）
        self.current_epoch = 0

        # 只抽 velocity 相关任务
        self.task = self.sample_task()

    # =============== 公共接口（给 trainer 用） ===============

    def set_epoch(self, epoch: int):
        """训练脚本每个 epoch 开始时调用，用来切换终止策略等。"""
        self.current_epoch = epoch

    def update_task(self, task):
        """外部可以显式指定任务向量（比如固定 spec 进行 eval）"""
        self.task = task

    def update_base_task(self, base_task):
        """外部可以直接把 base_task 索引设置进去"""
        self.base_task = self.config.get("tasks", {}).get(base_task)

    # =============== 关键：step 逻辑 ===============

    def step(self, action, healthy_scale=None):
        if healthy_scale is not None:
            self.healthy_scale = healthy_scale

        # x 方向位置，用于计算前进速度
        xpos_before = self.get_body_com("torso")[0]

        # HalfCheetahEnv 的 step：应用动作 + 物理推进 + 基础 reward
        state, base_reward, terminated, truncated, info = super().step(action)

        xpos_after = self.get_body_com("torso")[0]
        ob = self._get_obs()

        # Half-Cheetah 理论上不会像 Ant 那样“摔倒”，
        # 一般只用时间上限截断即可，所以这里终止条件可以简单很多。
        # 如果你希望“姿态太离谱就算摔倒”，可以自己加一个判定：
        s = self.state_vector()
        finite_ok = np.isfinite(s).all()
        # HalfCheetah 的 qpos[1] 通常是竖直方向高度，这里给个很宽松的范围
        z = float(self.sim.data.qpos[1])
        height_ok = (z > 0.4) and (z < 1.5)

        if self.current_epoch < self.config.get("use_termination_after", 0):
            # 前若干个 epoch 不用摔倒终止，方便早期探索
            terminated = False
        else:
            terminated = not (finite_ok and height_ok)

        # 存活奖励/惩罚（可选，比 Ant 小一点）
        healthy_reward = 1.0
        healthy_penalty = 0.0
        if terminated:
            healthy_reward = 0.0
            healthy_penalty = -5.0

        # ============= 只保留 Velocity 任务 =============
        # 只允许 backward_vel / forward_vel 两类 base_task
        if self.base_task in [
            self.config.get("tasks", {}).get("backward_vel"),
            self.config.get("tasks", {}).get("forward_vel"),
        ]:
            forward_vel = (xpos_after - xpos_before) / self.dt

            # target vel 存在 self.task[self.base_task]
            target_vel = self.task[self.base_task]

            # 这里 reward 形式可以和 Ant 保持一致：
            #   -abs(v - v*) / (|v*| + eps)
            # 如果你想让惩罚更强，可以乘一个权重 w_vel
            w_vel = self.config.get("w_vel", 1.0)
            reward_run = - np.abs(forward_vel - target_vel) / (
                np.abs(target_vel) + 1e-6
            ) * w_vel

            # 动作 L2 惩罚
            w_ctrl = self.config.get("w_ctrl", 0.1)
            reward_ctrl = - w_ctrl * np.sum(np.square(action))

            reward = (
                reward_run
                + reward_ctrl
                + healthy_reward * self.healthy_scale
                + healthy_penalty
            )
        else:
            # 理论上不会走到这里，因为我们只配置 velocity 任务
            raise RuntimeError("base task not recognized (expect velocity only).")

        info = info or {}
        info.update(
            dict(
                reward_run=reward_run,
                reward_ctrl=reward_ctrl,
                true_task=self.task,
            )
        )

        return ob, reward, terminated, truncated, info

    # =============== 观测定义 ===============

    def _get_obs(self):
        # 根据你项目里 HalfCheetah 的定义来。
        # 如果你想和 AntMulti 类似：qpos + qvel + torso COM
        return np.concatenate(
            [
                self.sim.data.qpos.flat,
                self.sim.data.qvel.flat,
                self.get_body_com("torso").flat,
            ]
        ).astype(np.float32)

    # =============== reset / random_reset ===============

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        new_obs = self._get_obs()
        return new_obs, info

    def random_reset(self, x_vel_range=(-0.5, 0.5)):
        """如果你也想像 Ant 那样随机初始化速度，可以简单做一个版本"""
        obs, info = super().reset()

        qpos = self.init_qpos + self.np_random.uniform(
            low=-0.05, high=0.05, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.standard_normal(self.model.nv) * 0.1

        # 只随机 x 方向速度
        qvel[0] = np.random.random() * (x_vel_range[1] - x_vel_range[0]) + x_vel_range[0]

        self.set_state(qpos, qvel)
        new_obs = self._get_obs()
        return new_obs, {}

    # =============== 任务采样（只包含 velocity） ===============

    def sample_task(self, task=None):
        """
        任务向量 task: one-hot + value 的形式，
        self.config['tasks'] 比如 {'backward_vel': 0, 'forward_vel': 1}
        对应 max_vel = [min, max]
        """
        self.task = np.zeros(max(self.config["tasks"].values()) + 1)

        if task is not None:
            # 外部指定 base_task 和 specification（0~1）
            base_task_name = task["base_task"]
            self.base_task = self.config["tasks"][base_task_name]
            mult = task["specification"]
        else:
            # 随机从 left/right 选择一种
            base_task_name = np.random.choice(list(self.config["tasks"].keys()))
            self.base_task = self.config["tasks"][base_task_name]
            mult = np.random.random()

        max_vel_low, max_vel_high = self.config["max_vel"]

        if base_task_name == "backward_vel":
            # 负向速度
            self.task[self.base_task] = -(
                mult * (max_vel_high - max_vel_low) + max_vel_low
            )
        elif base_task_name == "forward_vel":
            # 正向速度
            self.task[self.base_task] = (
                mult * (max_vel_high - max_vel_low) + max_vel_low
            )
        else:
            raise ValueError("Only backward_vel / forward_vel are supported.")

        return self.task

