import numpy as np
from sac_envs.base_envs.ant import AntEnv
from gym.spaces.box import Box
from gym import utils
import os


class AntMixtureEnv(AntEnv, utils.EzPickle):
    """
    Ant multi-task environment that mirrors the HalfCheetahMixtureEnv pattern.

    Observation layout (31-dim):
      qpos[1:]     : 14 dims  (skip global x-pos; keep y, z, quat[4], joints[8])
      qvel         : 14 dims  (vx, vy, vz, wx, wy, wz, joint-vels[8])
      torso_com    :  3 dims  (centre-of-mass of torso in world frame)

    Task vector (task_dim = max(tasks.values())+1):
      task[goal_front]   = target x-position (forward)
      task[goal_back]    = target x-position (backward)
      task[forward_vel]  = target x-velocity (positive)
      task[backward_vel] = target x-velocity (negative)

    Tasks are configured via config['tasks'] dict, e.g.
      tasks = dict(forward_vel=2, backward_vel=3, goal_front=0, goal_back=1)
    """

    def __init__(self, config, render_mode: str = 'rgb_array', *args, **kwargs):
        self.screen_height = 400
        self.screen_width = 400
        self.termination_possible = False
        super().__init__(*args, **kwargs)
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(31,), dtype=np.float64
        )
        self.reached_goal = 0
        self.config = config
        self.task = self.sample_task()

    # ------------------------------------------------------------------
    # Core step
    # ------------------------------------------------------------------

    def step(self, action, norm=True):
        xposbefore = np.copy(self.sim.data.qpos)
        try:
            self.do_simulation(action, self.frame_skip)
        except Exception:
            raise RuntimeError("Simulation error, common error is action = nan")

        xposafter = np.copy(self.sim.data.qpos)
        xvelafter = np.copy(self.sim.data.qvel)

        ob = self._get_obs()

        tasks = self.config.get('tasks', {})

        if self.base_task in [tasks.get('goal_front'), tasks.get('goal_back')]:
            # Goal-position tracking along x-axis
            if not norm:
                n = 1
            else:
                n = max(np.abs(self.norm), 1e-3)
            reward_run = -np.abs(xposafter[0] - self.task[self.base_task]) / n
            reward_ctrl = 0.0
            reward = reward_run

        elif self.base_task in [tasks.get('forward_vel'), tasks.get('backward_vel')]:
            # Velocity tracking along x-axis
            forward_vel = xvelafter[0]
            if not norm:
                n = 1
            else:
                n = max(np.abs(self.task[self.base_task]), 1e-3)
            reward_run = -1.0 * np.abs(forward_vel - self.task[self.base_task]) / n
            reward_ctrl = 0.0
            reward = reward_run

        else:
            raise RuntimeError("base task not recognized: {}".format(self.base_task))

        if self.termination_possible:
            state = self.state_vector()
            notdone = (
                np.isfinite(state).all()
                and state[2] >= 0.2
                and state[2] <= 1.0
            )
            done = not notdone
        else:
            done = False

        return ob, reward, done, False, dict(
            reward_run=reward_run,
            reward_ctrl=reward_ctrl,
            true_task=self.task,
        )

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_obs(self):
        # 14 + 14 + 3 = 31 dims
        return np.concatenate([
            self.sim.data.qpos.flat[1:],       # skip global x-pos
            self.sim.data.qvel.flat,
            self.get_body_com("torso").flat,
        ]).astype(np.float32).flatten()

    # ------------------------------------------------------------------
    # Resets
    # ------------------------------------------------------------------

    def reset(self):
        super().reset()
        new_obs = self._get_obs()
        return new_obs, {}

    def random_reset(self, x_pos_range=(-10, 10), x_vel_range=(-0.1, 0.1)):
        super().reset()
        qpos = self.init_qpos + self.np_random.uniform(
            low=-0.1, high=0.1, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.standard_normal(self.model.nv) * 0.1
        qpos[0] = (
            np.random.random() * (x_pos_range[1] - x_pos_range[0]) + x_pos_range[0]
        )
        qvel[0] = (
            np.random.random() * (x_vel_range[1] - x_vel_range[0]) + x_vel_range[0]
        )
        self.set_state(qpos, qvel)
        new_obs = self._get_obs()
        return new_obs, {}

    # ------------------------------------------------------------------
    # Task sampling
    # ------------------------------------------------------------------

    def sample_task(self, test=False, task=None):
        self.task = np.zeros(max(self.config['tasks'].values()) + 1)
        base_task = np.random.choice(list(self.config['tasks'].keys()))
        self.base_task = self.config.get('tasks', {}).get(base_task)
        mult = np.random.random()

        if task:
            base_task = task['base_task']
            self.base_task = self.config.get('tasks', {}).get(base_task)
            mult = task['specification']

        if base_task == 'goal_front':
            if test:
                self.task[self.base_task] = (
                    mult * (self.config['max_goal'][1] - self.config['max_goal'][0])
                    + self.config['max_goal'][0]
                )
                self.norm = self.task[self.base_task]
            else:
                self.norm = mult + 0.5
                self.task[self.base_task] = self.sim.data.qpos[0] + self.norm

        elif base_task == 'goal_back':
            if test:
                self.task[self.base_task] = -(
                    mult * (self.config['max_goal'][1] - self.config['max_goal'][0])
                    + self.config['max_goal'][0]
                )
                self.norm = self.task[self.base_task]
            else:
                self.norm = mult + 0.5
                self.task[self.base_task] = self.sim.data.qpos[0] - self.norm

        elif base_task == 'forward_vel':
            self.task[self.base_task] = (
                mult * (self.config['max_vel'][1] - self.config['max_vel'][0])
                + self.config['max_vel'][0]
            )
            self.norm = self.task[self.base_task]

        elif base_task == 'backward_vel':
            self.task[self.base_task] = -(
                mult * (self.config['max_vel'][1] - self.config['max_vel'][0])
                + self.config['max_vel'][0]
            )
            self.norm = self.task[self.base_task]

        else:
            print('Task not found: {}'.format(base_task))

        return self.task

    def update_task(self, task):
        self.task = task
