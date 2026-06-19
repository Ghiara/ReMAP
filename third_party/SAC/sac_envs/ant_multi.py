#This file is a modified version of the original ant_multi.py file,
#which uses the "old" architecture, to be adapted with "train_low_level_policy.py"
#the "old" architecture is the one also used in Cheetah, hopper and walker



import gym
from third_party.SAC.sac_envs.base_envs.ant import AntEnv
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

        # epoch / curriculum counters must be initialized before sampling task
        self.current_epoch = 0
        self.training_episode = 0
        self.prev_action = None
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

    def set_epoch(self, epoch):

        """训练脚本每个 epoch 开始时调用"""
        self.current_epoch = epoch

    def set_training_episode(self, episode: int):
        self.training_episode = int(episode)

    def _curriculum_scale(self) -> float:
        milestones = [1, 200, 500, 900, 1400]
        scales = [0.45, 0.60, 0.75, 0.90, 1.00]
        idx = 0
        for i, m in enumerate(milestones):
            if self.training_episode >= m:
                idx = i
        return scales[idx]

    def _stability_metrics(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        torso_rot = np.array(self.sim.data.get_body_xmat('torso')).reshape(3, 3)
        free_joint_ang_vel = qvel[3:6] if qvel.shape[0] >= 6 else qvel[:3]
        return dict(
            height=float(qpos[2]),
            vertical_vel=float(qvel[2]) if qvel.shape[0] > 2 else 0.0,
            pitch_rate=float(np.linalg.norm(free_joint_ang_vel)),
            lateral_offset=float(self.get_body_com('torso')[1]),
            upright=float(torso_rot[2, 2]),
        )

    def _sample_initial_state(self, x_pos=None, x_vel=None):
        qpos = np.array(self.init_qpos, copy=True)
        qvel = np.array(self.init_qvel, copy=True)

        xy_noise = float(self.config.get('reset_xy_noise', 0.04))
        joint_noise = float(self.config.get('reset_joint_noise', 0.04))
        height_noise = float(self.config.get('reset_height_noise', 0.015))
        velocity_noise = float(self.config.get('reset_velocity_noise', 0.04))
        initial_height = float(self.config.get('initial_torso_height', 0.72))

        qpos[:2] += self.np_random.uniform(low=-xy_noise, high=xy_noise, size=2)
        qpos[2] = initial_height + self.np_random.uniform(low=-height_noise, high=height_noise)
        if qpos.shape[0] >= 7:
            qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
        if qpos.shape[0] > 7:
            qpos[7:] += self.np_random.uniform(low=-joint_noise, high=joint_noise, size=qpos.shape[0] - 7)

        qvel += self.np_random.standard_normal(self.model.nv) * velocity_noise
        if x_pos is not None:
            qpos[0] = x_pos
        if x_vel is not None:
            qvel[0] = x_vel
        return qpos, qvel

    def _termination_thresholds(self):
        start = int(self.config.get('use_termination_after', 0))
        ramp = max(int(self.config.get('termination_ramp_epochs', 600)), 1)
        if self.current_epoch <= start:
            mix = 0.0
        else:
            mix = min((self.current_epoch - start) / ramp, 1.0)

        relaxed = dict(
            min_torso_height=0.16,
            max_torso_height=1.25,
            min_upright=0.10,
            max_lateral_offset=1.80,
            max_vertical_velocity=5.50,
            max_pitch_rate=16.0,
        )
        target = dict(
            min_torso_height=float(self.config.get('min_torso_height', 0.24)),
            max_torso_height=float(self.config.get('max_torso_height', 0.95)),
            min_upright=float(self.config.get('min_upright', 0.45)),
            max_lateral_offset=float(self.config.get('max_lateral_offset', 0.90)),
            max_vertical_velocity=float(self.config.get('max_vertical_velocity', 2.40)),
            max_pitch_rate=float(self.config.get('max_pitch_rate', 10.0)),
        )

        thresholds = {}
        for key, relaxed_value in relaxed.items():
            target_value = target[key]
            thresholds[key] = relaxed_value + mix * (target_value - relaxed_value)
        return thresholds

     #----------------------------------------------------------   
    def step(self, action, healthy_scale=None):
        if healthy_scale is not None:
            self.healthy_scale = healthy_scale

        xposbefore = self.get_body_com('torso')[0]
        state, reward, terminated, _, info = super().step(action)
        state = np.append(self.get_body_com("torso")[0], state)
        xposafter = self.get_body_com('torso')[0]
        ob = self._get_obs()

        s = self.state_vector()
        metrics = self._stability_metrics()
        thresholds = self._termination_thresholds()
        finite_ok = np.isfinite(s).all()
        state_ok = (np.abs(s[2:]) < self.config.get('max_state_abs', 100.0)).all()
        height_ok = thresholds['min_torso_height'] <= metrics['height'] <= thresholds['max_torso_height']
        upright_ok = metrics['upright'] >= thresholds['min_upright']
        lateral_ok = abs(metrics['lateral_offset']) <= thresholds['max_lateral_offset']
        vertical_vel_ok = abs(metrics['vertical_vel']) <= thresholds['max_vertical_velocity']
        pitch_rate_ok = abs(metrics['pitch_rate']) <= thresholds['max_pitch_rate']

        terminated = not (
            finite_ok
            and state_ok
            and height_ok
            and upright_ok
            and lateral_ok
            and vertical_vel_ok
            and pitch_rate_ok
        )

        if self.current_epoch < self.config.get("use_termination_after", 0):
            terminated = False

        healthy_reward = self.config.get('healthy_reward', 0.35)
        termination_penalty = self.config.get('termination_penalty', -15.0) if terminated else 0.0

        height_target = self.config.get('target_torso_height', 0.48)
        height_penalty = -self.config.get('height_penalty_weight', 2.0) * abs(metrics['height'] - height_target)
        upright_penalty = -self.config.get('upright_penalty_weight', 1.5) * max(0.0, self.config.get('min_upright', 0.75) - metrics['upright'])
        vertical_penalty = -self.config.get('vertical_velocity_penalty_weight', 0.25) * abs(metrics['vertical_vel'])
        pitch_rate_penalty = -self.config.get('pitch_rate_penalty_weight', 0.03) * abs(metrics['pitch_rate'])
        smooth_penalty = 0.0
        if self.prev_action is not None:
            smooth_penalty = -self.config.get('action_smooth_penalty_weight', 0.0) * np.sum(np.square(action - self.prev_action))
        self.prev_action = np.array(action, copy=True)
        posture_reward = height_penalty + upright_penalty + vertical_penalty + pitch_rate_penalty + smooth_penalty

        cfg_tasks = self.config.get('tasks', {})
        if self.base_task in [cfg_tasks.get('goal_front'), cfg_tasks.get('goal_back')]:
            dist_before = np.abs(xposbefore - self.task[self.base_task])
            dist_after = np.abs(xposafter - self.task[self.base_task])
            progress = dist_before - dist_after
            goal_norm = max(float(self.config['max_goal'][1]), 1.0)

            reward_run = -dist_after / goal_norm + self.config.get('goal_progress_weight', 0.8) * progress / goal_norm
            if dist_after < 1.0:
                reward_run += 0.5
            if dist_after < 0.4:
                reward_run += 0.7

            reward_ctrl = -self.config.get('goal_ctrl_cost', 2e-3) * np.sum(np.square(action))
            reward = reward_run + reward_ctrl + posture_reward + healthy_reward * self.healthy_scale + termination_penalty

        elif self.base_task in [cfg_tasks.get('forward_vel'), cfg_tasks.get('backward_vel')]:
            forward_vel = (xposafter - xposbefore) / self.dt
            vel_norm = max(float(self.config['max_vel'][1]), 1.0)
            vel_error = np.abs(forward_vel - self.task[self.base_task])

            reward_run = -vel_error / vel_norm
            if vel_error < 0.35:
                reward_run += 0.4
            if vel_error < 0.20:
                reward_run += 0.5

            reward_ctrl = -self.config.get('vel_ctrl_cost', 3e-3) * np.sum(np.square(action))
            reward = reward_run + reward_ctrl + posture_reward + healthy_reward * self.healthy_scale + termination_penalty
        else:
            raise RuntimeError("base task not recognized")

        return ob, reward, terminated, False, dict(
            reward_run=reward_run,
            reward_ctrl=reward_ctrl,
            posture_reward=posture_reward,
            height=metrics['height'],
            upright=metrics['upright'],
            vertical_vel=metrics['vertical_vel'],
            pitch_rate=metrics['pitch_rate'],
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
        if self.base_task in [self.config.get('tasks', {}).get('goal_front'),
                            self.config.get('tasks', {}).get('goal_back')]:
            goal_x = self.task[self.base_task]
            geom_id = self.sim.model.geom_name2id("goal")
            self.sim.model.geom_pos[geom_id] = np.array([goal_x, 0.0, 0.5])  # 放在x轴对应位置


    def update_base_task(self, base_task):
        self.base_task = base_task

    def reset(self):
        obs = super().reset()
        self.prev_action = None
        # new_obs = np.append(self.get_body_com("torso")[0], obs[0])
        new_obs = self._get_obs()
        return new_obs, {}

    def reset_model(self):
        qpos, qvel = self._sample_initial_state()
        self.set_state(qpos, qvel)
        return self._get_obs()
    
    
    
    def random_reset(self, x_pos_range=[-10,10], x_vel_range=[-0.1,0.1]):
        obs = super().reset()
        # new_obs = np.append(self.get_body_com("torso")[0], obs[0])

        x_pos = np.random.random() * (x_pos_range[1] - x_pos_range[0]) + x_pos_range[0]
        x_vel = np.random.random() * (x_vel_range[1] - x_vel_range[0]) + x_vel_range[0]
        qpos, qvel = self._sample_initial_state(x_pos=x_pos, x_vel=x_vel)
        self.set_state(qpos, qvel)
        self.prev_action = None

        new_obs = self._get_obs()
        return new_obs, {}
    
    
    
    def sample_task(self, test=False, task=None):

        self.task = np.zeros(max(self.config['tasks'].values()) + 1)

        task_keys = list(self.config['tasks'].keys())
        configured_weights = self.config.get('task_sample_weights')
        if configured_weights:
            task_weights = np.array([configured_weights.get(k, 1.0) for k in task_keys], dtype=np.float64)
        else:
            task_weights = np.ones(len(task_keys), dtype=np.float64)
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
            mag = goal_min + (goal_upper - goal_min) * (mult ** 0.7)
            self.task[self.base_task] = mag
            self.norm = max(goal_max, 1.0)

        elif base_task == 'goal_back':
            mag = goal_min + (goal_upper - goal_min) * (mult ** 0.7)
            self.task[self.base_task] = -mag
            self.norm = max(goal_max, 1.0)

        elif base_task == 'forward_vel':
            mag = vel_min + (vel_upper - vel_min) * (mult ** 0.75)
            self.task[self.base_task] = mag
            self.norm = max(vel_max, 1.0)

        elif base_task == 'backward_vel':
            mag = vel_min + (vel_upper - vel_min) * (mult ** 0.75)
            self.task[self.base_task] = -mag
            self.norm = max(vel_max, 1.0)

        else:
            raise ValueError("Task not found")
        return self.task
    