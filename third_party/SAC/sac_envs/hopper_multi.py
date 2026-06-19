import gym
from third_party.SAC.sac_envs.base_envs.hopper import HopperEnv
from gym.utils.ezpickle import EzPickle
from typing import List, Tuple
import numpy as np
import mujoco_py
from meta_envs.pygame_rendering import PygameRenderer
from gym.spaces import Box


class HopperMulti(HopperEnv):
    # rewrite this to give as input the dictionary
    def __init__(self, config, healthy_scale = 1, render_mode: str = 'rgb_array', *args, **kwargs):
        super().__init__(render_mode=render_mode, *args, **kwargs)
        self.observation_space = Box(
                low=-np.inf, high=np.inf, shape=(14,), dtype=np.float64
            )
        self.healthy_scale = healthy_scale
        self.screen_height = 400
        self.screen_width = 400
        self.config = config
        self.training_episode = 0
        self.reward_params = config.get('reward_params', {})
        self.task_sampling_weights = config.get('task_sampling_weights', {})
        self.task = self.sample_task()

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

    def viewer_setup(self):
        self.viewer.cam.type = 2
        self.viewer.cam.fixedcamid = 0

    def get_image(self, width=256, height=256, camera_name=None):
        if self.viewer is None or type(self.viewer) != mujoco_py.MjRenderContextOffscreen:
            self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim)
            self.viewer_setup()
            self._viewers['rgb_array'] = self.viewer
        return self.sim.render(width, height, camera_name=camera_name)


    def set_training_episode(self, episode: int):
        self.training_episode = int(episode)

    def _curriculum_scale(self) -> float:
        # Expand command difficulty progressively to stabilize learning.
        milestones = [1, 200, 500, 900, 1400]
        scales = [0.45, 0.60, 0.75, 0.90, 1.00]
        idx = 0
        for i, m in enumerate(milestones):
            if self.training_episode >= m:
                idx = i
        return scales[idx]

    def _velocity_reward_terms(self, forward_vel: float, target_vel: float, action) -> tuple[float, float]:
        vel_norm = max(float(self.config['max_vel'][1]), 1.0)
        vel_error = abs(forward_vel - target_vel)
        base_track = -vel_error / vel_norm

        near_bonus = 0.0
        if vel_error < 0.40:
            near_bonus += 0.35
        if vel_error < 0.22:
            near_bonus += 0.45

        target_speed = abs(target_vel)
        speed_ratio = min(target_speed / max(float(self.config['max_vel'][1]), 1e-6), 1.0)
        direction_match = np.sign(forward_vel) == np.sign(target_vel) or abs(forward_vel) < 1e-6
        progress_term = 0.0
        if direction_match:
            progress_term = min(abs(forward_vel), target_speed) / max(target_speed, 1e-6)
        progress_term = 0.30 * speed_ratio * progress_term

        action_penalty = -float(self.reward_params.get('velocity_ctrl_weight', 1.5e-3)) * np.sum(np.square(action))
        reward_run = base_track + near_bonus + progress_term
        return reward_run, action_penalty

    def step(self, action, healthy_scale=None):
        if healthy_scale is not None:
            self.healthy_scale = healthy_scale
        # Task is 5 dimensional -> it can either be jump, go forward/backward, rotation, velocity and flip velocity
        # this can be just q pos and qvel[0]
        # change task after some steps


        xposbefore = self.get_body_com('torso')[0]
        distance_before = np.abs(self.task[0] - self.get_body_com('torso')[0])
        state, reward, terminated, _, info = super().step(action)
        state = np.append(self.get_body_com("torso")[0], state)
        distance_after = np.abs(self.task[0] - self.get_body_com('torso')[0])
        xposafter = self.get_body_com('torso')[0]

        ob = self._get_obs()


        posafter, height, ang = self.sim.data.qpos[0:3]
        s = self.state_vector()
        forward_vel_task = self.base_task == self.config.get('tasks', {}).get('forward_vel')
        backward_vel_task = self.base_task == self.config.get('tasks', {}).get('backward_vel')
        target_speed = abs(float(self.task[self.base_task])) if self.base_task in self.config.get('tasks', {}).values() else 0.0
        if backward_vel_task:
            speed_ratio = min(target_speed / max(float(self.config['max_vel'][1]), 1e-6), 1.0)
            base_limit = float(self.reward_params.get('backward_ang_limit_base', 0.50))
            high_speed_bonus = float(self.reward_params.get('backward_ang_limit_bonus', 0.10))
            ang_limit = base_limit + high_speed_bonus * speed_ratio
        elif forward_vel_task:
            speed_ratio = min(target_speed / max(float(self.config['max_vel'][1]), 1e-6), 1.0)
            base_limit = float(self.reward_params.get('forward_ang_limit_base', 0.42))
            high_speed_bonus = float(self.reward_params.get('forward_ang_limit_bonus', 0.10))
            ang_limit = base_limit + high_speed_bonus * speed_ratio
        else:
            ang_limit = float(self.reward_params.get('goal_ang_limit', 0.4))
        terminated = not (
            np.isfinite(s).all()
            and (np.abs(s[2:]) < 100).all()
            and (height > 0.7)
            and (abs(ang) < ang_limit)
        )
        healthy_penalty = 0
        if terminated:
            healthy_penalty = -10

        healthy_reward = 0.7

        # if task[3]!=0:  # 'velocity'
        #     reward_run = - np.abs(xvelafter[0] - self.task_specification)
        #     reward_ctrl = -0.5 * 1e-1 * np.sum(np.square(action))
        #     reward = reward_ctrl * 1.0 + reward_run / np.abs(self.task_specification)
        
        if self.base_task in [self.config.get('tasks',{}).get('goal_front'), self.config.get('tasks',{}).get('goal_back')]:  # goal tracking
            dist_before = np.abs(xposbefore - self.task[self.base_task])
            dist_after = np.abs(xposafter - self.task[self.base_task])
            progress = dist_before - dist_after
            goal_norm = max(float(self.config['max_goal'][1]), 1.0)

            # Dense shaping: distance minimization + progress bonus + near-goal bonus.
            reward_run = -dist_after / goal_norm + 0.8 * progress / goal_norm
            if dist_after < 1.0:
                reward_run += 0.5
            if dist_after < 0.4:
                reward_run += 0.7

            reward_ctrl = -1e-3 * np.sum(np.square(action))
            healthy_reward = 1.0
            reward = reward_run + reward_ctrl + healthy_reward * self.healthy_scale

        elif self.base_task in [self.config.get('tasks',{}).get('forward_vel'), self.config.get('tasks',{}).get('backward_vel')]:  # velocity tracking
            healthy_reward = 1.0
            forward_vel = (xposafter - xposbefore) / self.dt
            reward_run, reward_ctrl = self._velocity_reward_terms(
                forward_vel=forward_vel,
                target_vel=float(self.task[self.base_task]),
                action=action,
            )
            reward = reward_run + reward_ctrl + healthy_reward * self.healthy_scale
        else:
            raise RuntimeError("base task not recognized")
        
        return ob, reward, terminated, False, dict(reward_run=reward_run, reward_ctrl=reward_ctrl,
                                      true_task=self.task)

    # from pearl rlkit
    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat,
            self.get_body_com("torso").flat,
        ]).astype(np.float32).flatten()


    def update_task(self, task):
        self.task = task

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
            low=-0.02, high=0.02, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.standard_normal(self.model.nv) * 0.1
        qpos[0] = np.random.random() * (x_pos_range[1] - x_pos_range[0]) + x_pos_range[0]
        qvel[0] = np.random.random() * (x_vel_range[1] - x_vel_range[0]) + x_vel_range[0]
        self.set_state(qpos, qvel)

        new_obs = self._get_obs()
        return new_obs, {}
    
    
    def sample_task(self, test=False, task=None):
        self.task = np.zeros(max(self.config['tasks'].values())+1)
        # {'velocity_forward': 0, 'velocity_backward': 1, 'goal_forward': 4, 'goal_backward': 5, 
        # 'flip_forward': 6, 'stand_front': 3, 'stand_back': 2, 'jump': 7, flip_backward = 8,
        # 'direction_forward': -1, 'direction_backward': -1, 'velocity': -1}
        task_keys = list(self.config['tasks'].keys())
        task_weights = np.array([
            float(self.task_sampling_weights.get(k, 1.0)) for k in task_keys
        ], dtype=np.float64)
        if np.sum(task_weights) <= 0:
            task_weights = np.ones_like(task_weights)
        task_weights = task_weights / task_weights.sum()
        base_task = np.random.choice(task_keys, p=task_weights)
        self.base_task = self.config.get('tasks',{}).get(base_task)
        mult = np.random.random()
        if task:
            base_task = task['base_task']
            self.base_task = self.config.get('tasks',{}).get(base_task)
            mult = task['specification']

        scale = self._curriculum_scale()
        goal_min, goal_max = self.config['max_goal']
        vel_min, vel_max = self.config['max_vel']
        goal_upper = goal_min + (goal_max - goal_min) * scale
        vel_upper = vel_min + (vel_max - vel_min) * scale

        if base_task == 'goal_front':
            # Bias toward higher command magnitudes to improve long-range tracking.
            mag = goal_min + (goal_upper - goal_min) * (mult ** 0.7)
            self.task[self.base_task] = mag
            self.norm = max(goal_max, 1.0)
        elif base_task == 'goal_back':
            mag = goal_min + (goal_upper - goal_min) * (mult ** 0.7)
            self.task[self.base_task] = -mag
            self.norm = max(goal_max, 1.0)
        elif base_task == 'forward_vel':
            # Bias forward velocity samples slightly toward the high-speed end.
            mag = vel_min + (vel_upper - vel_min) * (mult ** 0.60)
            self.task[self.base_task] = mag
            self.norm = max(vel_max, 1.0)
        elif base_task == 'backward_vel':
            # Match forward velocity curriculum so backward high-speed commands are also trained often.
            mag = vel_min + (vel_upper - vel_min) * (mult ** 0.60)
            self.task[self.base_task] = -mag
            self.norm = max(vel_max, 1.0)
        elif base_task == 'stand_front':
            self.task[self.base_task] = mult * (self.config['max_rot'][1] - self.config['max_rot'][0]) + self.config['max_rot'][0]
            self.norm = self.task[self.base_task]
        elif base_task == 'stand_back':
            self.task[self.base_task] = - (mult * (self.config['max_rot'][1] - self.config['max_rot'][0]) + self.config['max_rot'][0])
            self.norm = self.task[self.base_task]
        # elif base_task == 6: # instead of rotation velocity, sample how many flips
        #     sign = np.random.choice(np.array([1,2]))
        #     self.task[0] = (-1)**sign*(mult * (self.config['max_rot_vel'][1] - self.config['max_rot_vel'][0]) + self.config['max_rot_vel'][0])
        #     # flips = np.random.choice(np.array([1,2,3]))
        #     # self.task[4] = -2*np.pi*flips
        # elif base_task == 8:
        #     self.task[0] = -(mult * (self.config['max_rot_vel'][1] - self.config['max_rot_vel'][0]) + self.config['max_rot_vel'][0])
        elif base_task == 'jump':
            self.task[self.base_task] = mult * (self.config['max_jump'][1] - self.config['max_jump'][0]) + self.config['max_jump'][0]
            self.norm = self.task[self.base_task]
        else: 
            print('Task not found')
        return self.task