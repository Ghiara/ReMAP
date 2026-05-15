import gym
from sac_envs.base_envs.ant import AntEnv
from gym.utils.ezpickle import EzPickle
from gym import utils
from gym.spaces import Box
import numpy as np
import mujoco_py


class AntMulti(AntEnv, utils.EzPickle):

    def __init__(self, config=None, healthy_scale=1, render_mode: str = 'rgb_array', **kwargs):
        super().__init__(render_mode=render_mode, **kwargs)
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(32,), dtype=np.float64
        )
        self.healthy_scale = healthy_scale
        self.config = config or {}
        self.current_epoch = 0
        self.color = [1.0, 0.0, 0.0]
        self.base_task = 0
        self.task_specification = 1.0
        self.norm = 1.0

        # bt2t: task name -> task index, built from config['tasks']
        tasks = self.config.get('tasks', {})
        self.bt2t = dict(tasks)

        n_tasks = max(tasks.values()) + 1 if tasks else 1
        self.task = np.zeros(n_tasks)

        self._init_geom_rgba = self.model.geom_rgba.copy()
        self._init_geom_rgba[0, 3] = 0  # hide goal sphere (index 0) by default

        if tasks:
            self.sample_task()

    # ------------------------------------------------------------------ camera
    def _initialize_camera(self):
        sim = self.sim
        self.viewer = mujoco_py.MjRenderContextOffscreen(sim)
        camera = self.viewer.cam
        camera.type = 2
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

    # ------------------------------------------------------------------ task
    def change_task(self, spec):
        self.base_task = spec['base_task']
        self.task_specification = spec['specification']
        self._goal = spec['specification']
        self.color = spec['color']
        # update task vector
        if self.base_task < len(self.task):
            self.task[self.base_task] = self.task_specification
        self.norm = max(abs(self.task_specification), 1e-6)
        self.recolor()

    def recolor(self):
        geom_rgba = self._init_geom_rgba.copy()
        geom_rgba[2:, :3] = np.asarray(self.color)  # 0=goal sphere, 1=floor, 2+=ant body
        self.model.geom_rgba[:] = geom_rgba

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    # ------------------------------------------------------------------ step
    def step(self, action, healthy_scale=None):
        if healthy_scale is not None:
            self.healthy_scale = healthy_scale

        xpos_before = self.get_body_com('torso')[0]
        state, reward, terminated, _, info = super().step(action)
        xpos_after = self.get_body_com('torso')[0]
        ob = self._get_obs()

        # termination check
        s = self.state_vector()
        z = float(self.sim.data.qpos[2])
        terminated = not (np.isfinite(s).all() and z > 0.18)
        if self.current_epoch < self.config.get('use_termination_after', 0):
            terminated = False

        healthy_reward = 1.0
        healthy_penalty = -10.0 if terminated else 0.0
        if terminated:
            healthy_reward = 0.0

        cfg_tasks = self.config.get('tasks', {})
        if self.base_task in [cfg_tasks.get('goal_left'), cfg_tasks.get('goal_right')]:
            reward_run = -np.abs(xpos_after - self.task[self.base_task]) / (
                np.abs(self.task[self.base_task]) + 1e-6)
            reward_ctrl = -0.1 * np.sum(np.square(action))
            reward = reward_run + reward_ctrl + healthy_reward * self.healthy_scale + healthy_penalty

        elif self.base_task in [cfg_tasks.get('velocity_left'), cfg_tasks.get('velocity_right')]:
            forward_vel = (xpos_after - xpos_before) / self.dt
            reward_run = -np.abs(forward_vel - self.task[self.base_task]) / (
                np.abs(self.task[self.base_task]) + 1e-6)
            reward_ctrl = -0.1 * np.sum(np.square(action))
            reward = reward_run + reward_ctrl + healthy_reward * self.healthy_scale + healthy_penalty

        else:
            raise RuntimeError(f"base_task {self.base_task} not recognized")

        return ob, reward, terminated, False, dict(
            reward_run=reward_run,
            reward_ctrl=reward_ctrl,
            true_task=self.task,
        )

    # ------------------------------------------------------------------ obs
    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
            self.get_body_com('torso').flat,
        ]).astype(np.float32).flatten()

    # ------------------------------------------------------------------ reset
    def reset(self):
        super().reset()
        return self._get_obs(), {}

    def random_reset(self, x_pos_range=(-10, 10), x_vel_range=(-0.1, 0.1)):
        super().reset()
        qpos = self.init_qpos + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.standard_normal(self.model.nv) * 0.1
        qpos[0] = np.random.uniform(*x_pos_range)
        qvel[0] = np.random.uniform(*x_vel_range)
        self.set_state(qpos, qvel)
        return self._get_obs(), {}

    # ------------------------------------------------------------------ task sampling
    def sample_task(self, task=None):
        tasks = self.config.get('tasks', {})
        if not tasks:
            return self.task

        self.task = np.zeros(max(tasks.values()) + 1)
        base_task = np.random.choice(list(tasks.keys()))
        self.base_task = tasks[base_task]
        mult = np.random.random()

        if task:
            base_task = task.get('base_task', base_task)
            if isinstance(base_task, str):
                self.base_task = tasks.get(base_task, self.base_task)
            else:
                self.base_task = base_task
            mult = task.get('specification', mult)

        if base_task == 'goal_left':
            self.task[self.base_task] = -(mult * (self.config['max_goal'][1] - self.config['max_goal'][0]) + self.config['max_goal'][0])
        elif base_task == 'goal_right':
            self.task[self.base_task] = mult * (self.config['max_goal'][1] - self.config['max_goal'][0]) + self.config['max_goal'][0]
        elif base_task == 'velocity_left':
            self.task[self.base_task] = -(mult * (self.config['max_vel'][1] - self.config['max_vel'][0]) + self.config['max_vel'][0])
        elif base_task == 'velocity_right':
            self.task[self.base_task] = mult * (self.config['max_vel'][1] - self.config['max_vel'][0]) + self.config['max_vel'][0]
        elif base_task == 'jump':
            self.task[self.base_task] = mult * (self.config['max_jump'][1] - self.config['max_jump'][0]) + self.config['max_jump'][0]
        else:
            print(f'Task not found: {base_task}')

        self.norm = max(abs(float(self.task[self.base_task])), 1e-6)
        self.task_specification = float(self.task[self.base_task])
        return self.task

    def sample_tasks(self, num_tasks):
        return [{'base_task': self.base_task,
                 'specification': self.task_specification,
                 'color': self.color}
                for _ in range(num_tasks)]

    def update_task(self, task):
        self.task = task

    def update_base_task(self, base_task):
        self.base_task = base_task
