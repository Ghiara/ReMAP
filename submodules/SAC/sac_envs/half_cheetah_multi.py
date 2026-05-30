# import numpy as np
# from sac_envs.base_envs.half_cheetah import HalfCheetahEnv
# from gym.spaces.box import Box
# from gym import utils
# import mujoco_py
# import os


# class HalfCheetahMixtureEnv(HalfCheetahEnv, utils.EzPickle):
#     def __init__(self, config, healthy_scale = 1, render_mode: str = 'rgb_array', *args, **kwargs):
#         self.healthy_scale = healthy_scale
#         self.screen_height = 400
#         self.screen_width = 400
#         self.termination_possible = False
#         super().__init__(*args, **kwargs)
#         self.observation_space = Box(
#                 low=-np.inf, high=np.inf, shape=(20,), dtype=np.float64
#             )
#         self.reached_goal = 0
#         self.config = config
#         self.reward_params = config.get('reward_params', {})
#         self.task = self.sample_task()
#         self.r_w_track      = self.reward_params.get('w_track', 1.0)
#         self.r_w_energy     = self.reward_params.get('w_energy', 1e-3)
#         self.r_w_smooth_vel = self.reward_params.get('w_smooth_vel', 0.1)
#         self.r_w_smooth_act = self.reward_params.get('w_smooth_act', 1e-3)
#         self.r_w_pitch      = self.reward_params.get('w_pitch', 0.5)
#         self.r_alpha        = self.reward_params.get('vx_filter_alpha', 0.8)
#         self.r_scale        = self.reward_params.get('reward_scale', 3.0)
        
#         # === 新增：用于平滑 reward 的临时量 ===
#         self._vx_filt = None          # 低通滤波后的 vx
#         self._prev_vx_filt = None     # 上一步滤波 vx
#         self._prev_action = None      # 上一步动作 (用于Δa惩罚)
# # class HalfCheetahMixtureEnv(HalfCheetahEnv, utils.EzPickle):
# #     def __init__(self, healthy_scale = 1, render_mode: str = 'rgb_array', *args, **kwargs):
# #         super().__init__(render_mode=render_mode,*args, **kwargs)
# #         self.observation_space = Box(
# #                 low=-np.inf, high=np.inf, shape=(20,), dtype=np.float64
# #             )
# #         self.healthy_scale = healthy_scale
# #         self.screen_height = 400
# #         self.screen_width = 400
# #         self.task = 0
# #         self.termination_possible = False

#     def step(self, action, healthy_scale=0, norm=True):
#         # Task is 5 dimensional -> it can either be jump, go forward/backward, rotation, velocity and flip velocity
#         # this can be just q pos and qvel[0]
#         # change task after some steps

#         xposbefore = np.copy(self.sim.data.qpos)
#         try:
#             result = super().step(action)
#         except:
#             raise RuntimeError("Simulation error, common error is action = nan")

#         xposafter = np.copy(self.sim.data.qpos)
#         xvelafter = np.copy(self.sim.data.qvel)

#         ob = self._get_obs()
#         # xposafter = ob[-3:]
#         # xvelafter = ob[8:11]

#         # if task[3]!=0:  # 'velocity'
#         #     reward_run = - np.abs(xvelafter[0] - self.task_specification)
#         #     reward_ctrl = -0.5 * 1e-1 * np.sum(np.square(action))
#         #     reward = reward_ctrl * 1.0 + reward_run / np.abs(self.task_specification)
        
#         if self.base_task in [self.config.get('tasks',{}).get('goal_front'), self.config.get('tasks',{}).get('goal_back')]:  # 'goal'
#             if not norm:
#                 norm = 1
#             else:
#                 norm = np.abs(self.norm)

#             reward_run = - np.abs(xposafter[0] - self.task[self.base_task]) / norm
#             # reward_ctrl = -0.5 * 1e-1 * np.sum(np.square(action))
#             reward_ctrl = 0
#             reward = reward_ctrl * 1.0 + reward_run 
#             # if np.abs(xposafter[0] - self.task[0]) < 0.1:
#             #     self.reached_goal += 1
#             #     if self.reached_goal == 20:
#             #         reward+=1
#         elif self.base_task in [self.config.get('tasks',{}).get('rotation_front'), self.config.get('tasks',{}).get('rotation_back')] :  # 'flipping'   distance tp -2*pi
#             # reward_run = - np.abs(xvelafter[2] - self.task[4])
#             # reward_run = -np.abs(xposafter[2] - self.task[4])
#             # reward_ctrl = -0.5 * 1e-1 * np.sum(np.square(action))
#             # reward = reward_ctrl * 1.0 + reward_run / np.abs(self.task[4])
#             # if np.abs(xposafter[2] - self.task[4]) < 5/360*2*np.pi:
#             #     self.reached_goal += 1
#             #     if self.reached_goal == 20:
#             #         reward+=1
#             if not norm:
#                 norm = 1
#             else:
#                 norm = np.abs(self.norm)
#             reward_run = - np.abs(xposafter[2] - self.task[0])
#             # reward_ctrl = -0.5 * 1e-1 * np.sum(np.square(action))
#             reward_ctrl = 0
#             reward = reward_ctrl * 1.0 + reward_run / norm

#         elif self.base_task in [self.config.get('tasks',{}).get('stand_front'), self.config.get('tasks',{}).get('stand_back')]:  # 'stand_up'
#             if not norm:
#                 norm = 1
#             else:
#                 norm = np.abs(self.norm)
#             reward_run = - np.abs(xposafter[2] - self.task[self.base_task]) / norm
#             # reward_ctrl = -0.5 * 1e-1 * np.sum(np.square(action))
#             reward_ctrl = 0
#             reward = reward_ctrl * 1.0 + reward_run 
#             # if np.abs(xposafter[2] - self.task[2]) < 5/360*2*np.pi:
#             #     self.reached_goal += 1
#             #     if self.reached_goal == 20:
#             #         reward+=1

#         elif self.base_task == self.config.get('tasks',{}).get('jump'):  # 'jump'
#             if not norm:
#                 norm = 1
#             else:
#                 norm = np.abs(self.task[self.base_task])
#             reward_run = - np.abs(np.abs(xvelafter[1]) - self.task[self.base_task]) / norm
#             # reward_ctrl = -0.5 * 1e-1 * np.sum(np.square(action))
#             reward_ctrl = 0
#             reward = reward_ctrl * 1.0 + reward_run 
#             # if np.abs(xvelafter[1]) - self.task[1] < 0.1:
#             #     self.reached_goal += 1
#             #     if self.reached_goal == 20:
#             #         reward+=1

#         # elif self.base_task == 4 or self.base_task == 5:  # 'direction'
#         #     reward_run = (xposafter[0] - xposbefore[0]) / self.dt * np.sign(self.task[0])
#         #     reward_ctrl = -0.5 * 1e-1 * np.sum(np.square(action))
#         #     reward = reward_ctrl * 1.0 + reward_run

#         #old velocity task reward function construction
#         # elif self.base_task in [self.config.get('tasks',{}).get('forward_vel'), self.config.get('tasks',{}).get('backward_vel')]: # velocity
#         #     forward_vel = xvelafter[0]
#         #     if not norm:
#         #         norm = 1
#         #     else:
#         #         norm = np.abs(self.task[self.base_task])
#         #     reward_run = -1.0 * np.abs(forward_vel - self.task[self.base_task]) / norm
#         #     # reward_ctrl = -0.5 * 1e-1 * np.sum(np.square(action))
#         #     reward_ctrl = 0
#         #     reward = reward_ctrl * 1.0 + reward_run

#         elif self.base_task in [self.config['tasks']['forward_vel'],
#                                 self.config['tasks']['backward_vel']]:

#             v_target = float(self.task[self.base_task])

#             # --- 低通滤波 vx ---
#             vx_raw = float(xvelafter[0])
#             if self._vx_filt is None:
#                 self._vx_filt = vx_raw
#             else:
#                 alpha = self.r_alpha
#                 self._vx_filt = alpha * self._vx_filt + (1 - alpha) * vx_raw
#             vx_filt = float(self._vx_filt)

#             # --- norm ---
#             v_norm = max(abs(v_target), 1e-3)

#             # ① tracking
#             track_term = - abs(vx_filt - v_target) / v_norm

#             # ② energy
#             energy_term = - float(np.sum(np.square(action)))

#             # ③ smooth velocity
#             if self._prev_vx_filt is None:
#                 dv = 0.0
#             else:
#                 dv = abs(vx_filt - float(self._prev_vx_filt))
#             smooth_vel_term = - dv

#             # ④ smooth action
#             if self._prev_action is None:
#                 da2 = 0.0
#             else:
#                 da2 = float(np.sum(np.square(action - self._prev_action)))
#             smooth_act_term = - da2

#             # ⑤ pitch
#             pitch = float(xposafter[2])
#             pitch_term = - abs(pitch)

#             # --- 加权和 + scale ---
#             reward = (
#                 self.r_w_track      * track_term +
#                 self.r_w_energy     * energy_term +
#                 self.r_w_smooth_vel * smooth_vel_term +
#                 self.r_w_smooth_act * smooth_act_term +
#                 self.r_w_pitch      * pitch_term
#             )

#             reward = self.r_scale * reward
#             reward = float(np.clip(reward, -10.0, 0.0))

#             reward_run = track_term
#             reward_ctrl = reward - track_term  # 仅用于 info

#             self._prev_vx_filt = vx_filt
#             self._prev_action = np.array(action, copy=True)


#         else:
#             raise RuntimeError("base task not recognized")


#         # if np.abs(reward_run)<0.2:
#         #     done = True

#         # print(str(self.base_task) + ": " + str(reward))
#         # compared to gym original, we have the possibility to terminate, if the cheetah lies on the back
#         if self.termination_possible:
#             state = self.state_vector()
#             notdone = np.isfinite(state).all() and state[2] >= -2.5 and state[2] <= 2.5
#             done = not notdone
#         else:
#             done = False
#         return ob, reward, done, False, dict(reward_run=reward_run, reward_ctrl=reward_ctrl,
#                                       true_task=self.task)

#     # from pearl rlkit
#     def _get_obs(self):
#         return np.concatenate([
#             self.sim.data.qpos.flat[1:],
#             self.sim.data.qvel.flat,
#             self.get_body_com("torso").flat,
#         ]).astype(np.float32).flatten()

#     # def reset_model(self):
#     #     # reset changepoint
#     #     self.positive_change_point = self.positive_change_point_basis + np.random.random() * self.change_point_interval
#     #     self.negative_change_point = self.negative_change_point_basis - np.random.random() * self.change_point_interval

#     #     # reset tasks
#     #     self.base_task = self._task['base_task']
#     #     self.task_specification = self._task['specification']
#     #     self.recolor()

#     #     # standard
#     #     qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
#     #     qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
#     #     self.set_state(qpos, qvel)
#     #     return self._get_obs()

#     # def get_image(self, width=256, height=256, camera_name=None):
#     #     if self.viewer is None or type(self.viewer) != mujoco_py.MjRenderContextOffscreen:
#     #         self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim)
#     #         self.viewer_setup()
#     #         self._viewers['rgb_array'] = self.viewer

#     #     # use sim.render to avoid MJViewer which doesn't seem to work without display
#     #     return self.sim.render(
#     #         width=width,
#     #         height=height,
#     #         camera_name=camera_name,
#     #     )

#     # def viewer_setup(self):
#     #     self.viewer.cam.type = 2
#     #     self.viewer.cam.fixedcamid = 0

#     # def change_task(self, spec):
#     #     self.base_task = spec['base_task']
#     #     self.task_specification = spec['specification']
#     #     self._goal = spec['specification']
#     #     self.color = spec['color']
#     #     self.recolor()

#     # def recolor(self):
#     #     geom_rgba = self._init_geom_rgba.copy()
#     #     rgb_value = self.color
#     #     geom_rgba[1:, :3] = np.asarray(rgb_value)
#     #     self.model.geom_rgba[:] = geom_rgba

#     def update_task(self, task):
#         self.task = task

#     def reset(self):
#         obs = super().reset()
#         # new_obs = np.append(self.get_body_com("torso")[0], obs[0])
#         new_obs = self._get_obs()
#         # === 新增：重置平滑状态 ===
#         self._vx_filt = float(self.sim.data.qvel[0])
#         self._prev_vx_filt = None
#         self._prev_action = None
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

#         # === 新增：重置平滑状态 ===
#         self._vx_filt = float(self.sim.data.qvel[0])
#         self._prev_vx_filt = None
#         self._prev_action = None
        
#         return new_obs, {}
    
#     def sample_task(self, test=False, task=None):
#         self.task = np.zeros(max(self.config['tasks'].values())+1)
#         # {'velocity_forward': 0, 'velocity_backward': 1, 'goal_forward': 4, 'goal_backward': 5, 
#         # 'flip_forward': 6, 'stand_front': 3, 'stand_back': 2, 'jump': 7, flip_backward = 8,
#         # 'direction_forward': -1, 'direction_backward': -1, 'velocity': -1}
#         base_task = np.random.choice(list(self.config['tasks'].keys()))
#         self.base_task = self.config.get('tasks',{}).get(base_task)
#         mult = np.random.random()
#         if task:
#             base_task = task['base_task']
#             self.base_task = self.config.get('tasks',{}).get(base_task)
#             mult = task['specification']
#         if base_task == 'goal_front':
#             if test:
#                 self.task[self.base_task] = mult * (self.config['max_goal'][1] - self.config['max_goal'][0]) + self.config['max_goal'][0]
#                 self.norm = self.task[self.base_task]
#             else:
#                 self.norm = mult + 0.5
#                 self.task[self.base_task] = self.sim.data.qpos[0] + self.norm
#         elif base_task == 'goal_back':
#             if test:
#                 self.task[self.base_task] = - (mult * (self.config['max_goal'][1] - self.config['max_goal'][0]) + self.config['max_goal'][0])
#                 self.norm = self.task[self.base_task]
#             else:
#                 self.norm = mult + 0.5
#                 self.task[self.base_task] = self.sim.data.qpos[0] - self.norm
#         elif base_task == 'forward_vel':
#             self.task[self.base_task] = mult * (self.config['max_vel'][1] - self.config['max_vel'][0]) + self.config['max_vel'][0]
#             self.norm = self.task[self.base_task]
#         elif base_task == 'backward_vel':
#             self.task[self.base_task] = - (mult * (self.config['max_vel'][1] - self.config['max_vel'][0]) + self.config['max_vel'][0])
#             self.norm = self.task[self.base_task]
#         elif base_task == 'stand_front':
#             self.task[self.base_task] = mult * (self.config['max_rot'][1] - self.config['max_rot'][0]) + self.config['max_rot'][0]
#             self.norm = self.task[self.base_task]
#         elif base_task == 'stand_back':
#             self.task[self.base_task] = - (mult * (self.config['max_rot'][1] - self.config['max_rot'][0]) + self.config['max_rot'][0])
#             self.norm = self.task[self.base_task]
#         elif base_task == 'rotation_front': # instead of rotation velocity, sample how many flips
#             # sign = np.random.choice(np.array([1,2]))
#             # self.task[0] = (-1)**sign*(mult * (self.config['max_rot_vel'][1] - self.config['max_rot_vel'][0]) + self.config['max_rot_vel'][0])

#             # self.task[self.base_task] = -(mult * (self.config['max_rot_vel'][1] - self.config['max_rot_vel'][0]) + self.config['max_rot_vel'][0])
#             self.task[self.base_task] = self.sim.data.qpos[2]//(2*np.pi) + 2*np.pi
#             self.norm = self.task[self.base_task]
#             # flips = np.random.choice(np.array([1,2,3]))
#             # self.task[4] = -2*np.pi*flips

#         elif base_task == 'rotation_back': # instead of rotation velocity, sample how many flips
#             # sign = np.random.choice(np.array([1,2]))
#             # self.task[0] = (-1)**sign*(mult * (self.config['max_rot_vel'][1] - self.config['max_rot_vel'][0]) + self.config['max_rot_vel'][0])

#             self.task[self.base_task] = self.sim.data.qpos[2]//(2*np.pi) - 2*np.pi
#             self.norm = self.task[self.base_task]
#         # elif base_task == 8:
#         #     self.task[0] = -(mult * (self.config['max_rot_vel'][1] - self.config['max_rot_vel'][0]) + self.config['max_rot_vel'][0])
#         elif base_task == 'jump':
#             self.task[self.base_task] = mult * (self.config['max_jump'][1] - self.config['max_jump'][0]) + self.config['max_jump'][0]
#             self.norm = self.task[self.base_task]
#         else: 
#             print('Task not found')
#         return self.task



# below is the original cheetah training setup

import numpy as np
from sac_envs.base_envs.half_cheetah import HalfCheetahEnv
from gym.spaces.box import Box
from gym import utils
import mujoco_py


class HalfCheetahMixtureEnv(HalfCheetahEnv, utils.EzPickle):
    def __init__(self, config, healthy_scale=1, render_mode: str = 'rgb_array', *args, **kwargs):
        self.healthy_scale = healthy_scale
        self.screen_height = 400
        self.screen_width = 400
        self.termination_possible = False
        super().__init__(render_mode=render_mode, *args, **kwargs)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float64)
        self.reached_goal = 0
        self.config = config
        self.training_episode = 0
        self.reward_params = config.get('reward_params', {})
        self.task_sampling_weights = config.get('task_sampling_weights', {})
        self.forward_balance_target = float(self.reward_params.get('forward_balance_target', 0.0))
        self.forward_balance_weight_goal = float(self.reward_params.get('forward_balance_weight_goal', 0.35))
        self.forward_balance_weight_vel = float(self.reward_params.get('forward_balance_weight_vel', 0.30))
        self.forward_balance_soft_limit = float(self.reward_params.get('forward_balance_soft_limit', 0.18))
        self.forward_balance_hard_limit = float(self.reward_params.get('forward_balance_hard_limit', 0.45))
        self.forward_balance_hard_penalty = float(self.reward_params.get('forward_balance_hard_penalty', 0.75))
        self.forward_pitch_rate_weight = float(self.reward_params.get('forward_pitch_rate_weight', 0.02))
        self.forward_balance_bonus = float(self.reward_params.get('forward_balance_bonus', 0.10))
        self.forward_balance_bonus_window = float(self.reward_params.get('forward_balance_bonus_window', 0.10))
        self.task = self.sample_task()

    def set_training_episode(self, episode: int):
        self.training_episode = int(episode)

    def _curriculum_scale(self) -> float:
        curriculum = self.config.get('curriculum', {})
        milestones = curriculum.get('difficulty_steps', [1, 200, 500, 900, 1400])
        scales = curriculum.get('difficulty_scales', [0.45, 0.60, 0.75, 0.90, 1.00])
        idx = 0
        for i, milestone in enumerate(milestones):
            if self.training_episode >= milestone:
                idx = i
        return scales[idx]

    def _forward_balance_reward(self, torso_pitch: float, torso_pitch_rate: float, weight: float):
        pitch_error = abs(torso_pitch - self.forward_balance_target)
        soft_excess = max(0.0, pitch_error - self.forward_balance_soft_limit)
        soft_norm = max(self.forward_balance_soft_limit, 1e-6)
        pitch_penalty = -soft_excess / soft_norm
        rate_penalty = -abs(torso_pitch_rate)
        bonus = self.forward_balance_bonus if pitch_error < self.forward_balance_bonus_window else 0.0
        hard_penalty = -self.forward_balance_hard_penalty if pitch_error > self.forward_balance_hard_limit else 0.0
        balance_reward = (
            weight * pitch_penalty
            + self.forward_pitch_rate_weight * rate_penalty
            + bonus
            + hard_penalty
        )
        return float(balance_reward), float(pitch_error)

    def _initialize_camera(self):
        sim = self.sim
        self.viewer = mujoco_py.MjRenderContextOffscreen(sim)
        camera = self.viewer.cam
        camera.type = 1
        camera.trackbodyid = 0
        camera.elevation = -20
        sim.add_render_context(self.viewer)

    def viewer_setup(self):
        self.viewer.cam.type = 2
        self.viewer.cam.fixedcamid = 0

    def get_image(self, width=256, height=256, camera_name=None):
        if self.viewer is None or type(self.viewer) != mujoco_py.MjRenderContextOffscreen:
            self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim)
            self.viewer_setup()
            self._viewers['rgb_array'] = self.viewer
        return self.sim.render(width, height, camera_name=camera_name)

    def step(self, action, healthy_scale=0, norm=True):
        xposbefore = np.copy(self.sim.data.qpos)
        try:
            super().step(action)
        except Exception as exc:
            raise RuntimeError('Simulation error, common error is action = nan') from exc

        xposafter = np.copy(self.sim.data.qpos)
        xvelafter = np.copy(self.sim.data.qvel)
        torso_pitch = float(xposafter[2])
        torso_pitch_rate = float(xvelafter[2])
        balance_reward = 0.0
        pitch_error = abs(torso_pitch - self.forward_balance_target)

        ob = self._get_obs()

        if self.base_task in [self.config.get('tasks', {}).get('goal_front'), self.config.get('tasks', {}).get('goal_back')]:
            norm = 1 if not norm else np.abs(self.norm)
            reward_run = -np.abs(xposafter[0] - self.task[self.base_task]) / norm
            reward_ctrl = 0.0
            reward = reward_run
            if self.base_task == self.config.get('tasks', {}).get('goal_front'):
                balance_reward, pitch_error = self._forward_balance_reward(
                    torso_pitch=torso_pitch,
                    torso_pitch_rate=torso_pitch_rate,
                    weight=self.forward_balance_weight_goal,
                )
                reward += balance_reward

        elif self.base_task in [self.config.get('tasks', {}).get('rotation_front'), self.config.get('tasks', {}).get('rotation_back')]:
            norm = 1 if not norm else np.abs(self.norm)
            reward_run = -np.abs(xposafter[2] - self.task[0])
            reward_ctrl = 0.0
            reward = reward_run / norm

        elif self.base_task in [self.config.get('tasks', {}).get('stand_front'), self.config.get('tasks', {}).get('stand_back')]:
            norm = 1 if not norm else np.abs(self.norm)
            reward_run = -np.abs(xposafter[2] - self.task[self.base_task]) / norm
            reward_ctrl = 0.0
            reward = reward_run

        elif self.base_task == self.config.get('tasks', {}).get('jump'):
            norm = 1 if not norm else np.abs(self.task[self.base_task])
            reward_run = -np.abs(np.abs(xvelafter[1]) - self.task[self.base_task]) / norm
            reward_ctrl = 0.0
            reward = reward_run

        elif self.base_task in [self.config.get('tasks', {}).get('forward_vel'), self.config.get('tasks', {}).get('backward_vel')]:
            forward_vel = xvelafter[0]
            norm = 1 if not norm else np.abs(self.task[self.base_task])
            reward_run = -1.0 * np.abs(forward_vel - self.task[self.base_task]) / norm
            reward_ctrl = 0.0
            reward = reward_run
            if self.base_task == self.config.get('tasks', {}).get('forward_vel'):
                balance_reward, pitch_error = self._forward_balance_reward(
                    torso_pitch=torso_pitch,
                    torso_pitch_rate=torso_pitch_rate,
                    weight=self.forward_balance_weight_vel,
                )
                reward += balance_reward
        else:
            raise RuntimeError('base task not recognized')

        if self.termination_possible:
            state = self.state_vector()
            notdone = np.isfinite(state).all() and state[2] >= -2.5 and state[2] <= 2.5
            done = not notdone
        else:
            done = False
        return ob, reward, done, False, dict(
            reward_run=reward_run,
            reward_ctrl=reward_ctrl,
            reward_balance=balance_reward,
            torso_pitch=torso_pitch,
            torso_pitch_rate=torso_pitch_rate,
            torso_pitch_error=pitch_error,
            true_task=self.task,
        )

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat,
            self.get_body_com('torso').flat,
        ]).astype(np.float32).flatten()

    def update_task(self, task):
        self.task = task

    def update_base_task(self, base_task):
        self.base_task = base_task

    def reset(self):
        super().reset()
        new_obs = self._get_obs()
        return new_obs, {}

    def random_reset(self, x_pos_range=[-10, 10], x_vel_range=[-0.1, 0.1]):
        super().reset()
        qpos = self.init_qpos + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.standard_normal(self.model.nv) * 0.1
        qpos[0] = np.random.random() * (x_pos_range[1] - x_pos_range[0]) + x_pos_range[0]
        qvel[0] = np.random.random() * (x_vel_range[1] - x_vel_range[0]) + x_vel_range[0]
        self.set_state(qpos, qvel)
        new_obs = self._get_obs()
        return new_obs, {}

    def sample_task(self, test=False, task=None):
        self.task = np.zeros(max(self.config['tasks'].values()) + 1)
        task_keys = list(self.config['tasks'].keys())
        task_weights = np.array(
            [float(self.task_sampling_weights.get(task_key, 1.0)) for task_key in task_keys],
            dtype=np.float64,
        )
        if np.sum(task_weights) <= 0:
            task_weights = np.ones_like(task_weights)
        task_weights = task_weights / task_weights.sum()
        base_task = np.random.choice(task_keys, p=task_weights)
        self.base_task = self.config.get('tasks', {}).get(base_task)
        mult = np.random.random()
        if task:
            base_task = task['base_task']
            self.base_task = self.config.get('tasks', {}).get(base_task)
            mult = task['specification']

        scale = self._curriculum_scale()
        goal_min, goal_max = self.config['max_goal']
        vel_min, vel_max = self.config['max_vel']
        goal_upper = goal_min + (goal_max - goal_min) * scale
        vel_upper = vel_min + (vel_max - vel_min) * scale

        if base_task == 'goal_front':
            if test:
                self.task[self.base_task] = mult * (goal_max - goal_min) + goal_min
                self.norm = self.task[self.base_task]
            else:
                self.norm = goal_min + (goal_upper - goal_min) * (mult ** 0.75)
                self.task[self.base_task] = self.sim.data.qpos[0] + self.norm
        elif base_task == 'goal_back':
            if test:
                self.task[self.base_task] = -(mult * (goal_max - goal_min) + goal_min)
                self.norm = self.task[self.base_task]
            else:
                self.norm = goal_min + (goal_upper - goal_min) * (mult ** 0.75)
                self.task[self.base_task] = self.sim.data.qpos[0] - self.norm
        elif base_task == 'forward_vel':
            self.task[self.base_task] = vel_min + (vel_upper - vel_min) * (mult ** 0.75)
            self.norm = self.task[self.base_task]
        elif base_task == 'backward_vel':
            self.task[self.base_task] = -(vel_min + (vel_upper - vel_min) * (mult ** 0.75))
            self.norm = self.task[self.base_task]
        elif base_task == 'stand_front':
            self.task[self.base_task] = mult * (self.config['max_rot'][1] - self.config['max_rot'][0]) + self.config['max_rot'][0]
            self.norm = self.task[self.base_task]
        elif base_task == 'stand_back':
            self.task[self.base_task] = -(mult * (self.config['max_rot'][1] - self.config['max_rot'][0]) + self.config['max_rot'][0])
            self.norm = self.task[self.base_task]
        elif base_task == 'rotation_front':
            self.task[self.base_task] = self.sim.data.qpos[2] // (2 * np.pi) + 2 * np.pi
            self.norm = self.task[self.base_task]
        elif base_task == 'rotation_back':
            self.task[self.base_task] = self.sim.data.qpos[2] // (2 * np.pi) - 2 * np.pi
            self.norm = self.task[self.base_task]
        elif base_task == 'jump':
            self.task[self.base_task] = mult * (self.config['max_jump'][1] - self.config['max_jump'][0]) + self.config['max_jump'][0]
            self.norm = self.task[self.base_task]
        else:
            print('Task not found')
        return self.task
