"""
RL2 with Soft Actor-Critic (SAC) for meta-reinforcement learning.

The LSTM policy adapts online through its hidden state (no explicit encoder).
Uses the same MetaRLAlgorithm base class as PEARL for data collection,
replay buffers, evaluation, and logging.
"""
import os
import copy
import third_party.rlkit.torch as torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from collections import OrderedDict

from third_party.rlkit.core import logger
from third_party.rlkit.core.rl_algorithm import MetaRLAlgorithm
from third_party.rlkit.core.eval_util import create_stats_ordered_dict
from third_party.rlkit.torch.rl2.networks import LSTMPolicy, LSTMQFunction
import third_party.rlkit.torch.pytorch_util as ptu


class RL2SoftActorCritic(MetaRLAlgorithm):
    """
    SAC-based meta-RL using the RL2 approach.

    The LSTM policy carries hidden state across an episode (or trial)
    so that it implicitly encodes the task.  Q- and V-networks can be
    either LSTM-based or plain MLPs; both are supported.
    """

    def __init__(
        self,
        env,
        train_tasks,
        eval_tasks,
        nets,

        policy_lr=3e-4,
        qf_lr=3e-4,
        vf_lr=3e-4,
        soft_target_tau=0.005,
        discount=0.99,
        reward_scale=5.0,
        policy_mean_reg_weight=1e-3,
        policy_std_reg_weight=1e-3,

        # RL2-specific
        inner_lr=1e-4,
        num_inner_steps=5,

        render_eval_paths=False,
        plotter=None,
        **kwargs
    ):
        # nets = [agent, qf1, qf2, vf]
        agent = nets[0]

        super().__init__(
            env=env,
            agent=agent,
            train_tasks=train_tasks,
            eval_tasks=eval_tasks,
            **kwargs
        )

        self.agent = agent
        self.sparse_rewards = False  # RL2 does not use sparse rewards
        self.qf1 = nets[1]
        self.qf2 = nets[2]
        self.vf = nets[3]
        self.target_vf = copy.deepcopy(self.vf)

        self.soft_target_tau = soft_target_tau
        self.policy_mean_reg_weight = policy_mean_reg_weight
        self.policy_std_reg_weight = policy_std_reg_weight

        self.inner_lr = inner_lr
        self.num_inner_steps = num_inner_steps
        self.render_eval_paths = render_eval_paths
        self.plotter = plotter

        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()

        self.policy_optimizer = Adam(self.agent.policy.parameters(), lr=policy_lr)
        self.qf1_optimizer = Adam(self.qf1.parameters(), lr=qf_lr)
        self.qf2_optimizer = Adam(self.qf2.parameters(), lr=qf_lr)
        self.vf_optimizer = Adam(self.vf.parameters(), lr=vf_lr)

    # ------------------------------------------------------------------ #
    #  Network management
    # ------------------------------------------------------------------ #

    @property
    def networks(self):
        return [self.agent, self.qf1, self.qf2, self.vf, self.target_vf]

    def training_mode(self, mode):
        for net in self.networks:
            net.train(mode)

    def to(self, device=None):
        if device is None:
            device = ptu.device
        for net in self.networks:
            net.to(device)

    # ------------------------------------------------------------------ #
    #  Data handling
    # ------------------------------------------------------------------ #

    def unpack_batch(self, batch):
        o = batch['observations'][None, ...]
        a = batch['actions'][None, ...]
        r = batch['rewards'][None, ...]
        no = batch['next_observations'][None, ...]
        t = batch['terminals'][None, ...]
        return [o, a, r, no, t]

    def sample_sac(self, indices):
        batches = [
            ptu.np_to_pytorch_batch(
                self.replay_buffer.random_batch(idx, batch_size=self.batch_size)
            )
            for idx in indices
        ]
        unpacked = [self.unpack_batch(b) for b in batches]
        unpacked = [[x[i] for x in unpacked] for i in range(len(unpacked[0]))]
        unpacked = [torch.cat(x, dim=0) for x in unpacked]
        return unpacked

    def sample_context(self, indices):
        if not hasattr(indices, '__iter__'):
            indices = [indices]
        batches = [
            ptu.np_to_pytorch_batch(
                self.enc_replay_buffer.random_batch(
                    idx, batch_size=self.embedding_batch_size
                )
            )
            for idx in indices
        ]
        context = [self.unpack_batch(b) for b in batches]
        context = [[x[i] for x in context] for i in range(len(context[0]))]
        context = [torch.cat(x, dim=0) for x in context]
        context = torch.cat(context, dim=2)
        return context

    # ------------------------------------------------------------------ #
    #  Training
    # ------------------------------------------------------------------ #

    def _do_training(self, indices):
        mb_size = self.embedding_mini_batch_size
        num_updates = self.embedding_batch_size // mb_size

        self.agent.clear_z(num_tasks=len(indices))

        for _ in range(num_updates):
            self._take_step(indices)
            self.agent.detach_z()

        self._update_target_network()
        self.agent.clear_z(num_tasks=1)  # reset to single-task mode for rollout

    def _take_step(self, indices):
        num_tasks = len(indices)

        # --- sample SAC data ---
        obs, actions, rewards, next_obs, terms = self.sample_sac(indices)
        t, b, _ = obs.size()
        obs_flat = obs.view(t * b, -1)
        act_flat = actions.view(t * b, -1)
        rew_flat = rewards.view(t * b, -1)
        nobs_flat = next_obs.view(t * b, -1)
        term_flat = terms.view(t * b, -1)

        # ===== Q-function targets =====
        with torch.no_grad():
            target_v = self.target_vf(nobs_flat)
            q_target = self.reward_scale * rew_flat + (1.0 - term_flat) * self.discount * target_v

        # --- Q updates ---
        q1_pred = self._q_forward(self.qf1, obs_flat, act_flat)
        q2_pred = self._q_forward(self.qf2, obs_flat, act_flat)
        qf1_loss = self.qf_criterion(q1_pred, q_target.detach())
        qf2_loss = self.qf_criterion(q2_pred, q_target.detach())

        self.qf1_optimizer.zero_grad()
        qf1_loss.backward()
        self.qf1_optimizer.step()

        self.qf2_optimizer.zero_grad()
        qf2_loss.backward()
        self.qf2_optimizer.step()

        # ===== Policy =====
        mean, log_std, _ = self.agent.policy(obs_flat)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        pre_tanh = dist.rsample()
        action_new = torch.tanh(pre_tanh)

        log_prob = dist.log_prob(pre_tanh).sum(dim=-1, keepdim=True)
        log_prob -= (2 * (np.log(2) - pre_tanh - F.softplus(-2 * pre_tanh))).sum(dim=-1, keepdim=True)

        q1_new = self._q_forward(self.qf1, obs_flat, action_new)
        q2_new = self._q_forward(self.qf2, obs_flat, action_new)
        min_q_new = torch.min(q1_new, q2_new)

        policy_loss = (log_prob - min_q_new).mean()

        mean_reg = self.policy_mean_reg_weight * (mean ** 2).mean()
        std_reg = self.policy_std_reg_weight * (log_std ** 2).mean()
        policy_loss += mean_reg + std_reg

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # ===== V-function =====
        with torch.no_grad():
            mean2, log_std2, _ = self.agent.policy(obs_flat)
            std2 = torch.exp(log_std2)
            dist2 = torch.distributions.Normal(mean2, std2)
            pre2 = dist2.rsample()
            act2 = torch.tanh(pre2)
            lp2 = dist2.log_prob(pre2).sum(-1, keepdim=True)
            lp2 -= (2 * (np.log(2) - pre2 - F.softplus(-2 * pre2))).sum(-1, keepdim=True)
            q1v = self._q_forward(self.qf1, obs_flat, act2)
            q2v = self._q_forward(self.qf2, obs_flat, act2)
            min_qv = torch.min(q1v, q2v)
            v_target = min_qv - lp2

        v_pred = self.vf(obs_flat)
        vf_loss = self.vf_criterion(v_pred, v_target.detach())

        self.vf_optimizer.zero_grad()
        vf_loss.backward()
        self.vf_optimizer.step()

        # ===== Logging =====
        if self.eval_statistics is None:
            self.eval_statistics = OrderedDict()
        self.eval_statistics['QF1 Loss'] = np.mean(ptu.get_numpy(qf1_loss))
        self.eval_statistics['QF2 Loss'] = np.mean(ptu.get_numpy(qf2_loss))
        self.eval_statistics['VF Loss'] = np.mean(ptu.get_numpy(vf_loss))
        self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(policy_loss))

    def _q_forward(self, qf, obs, action):
        """Forward pass through Q-network (handles both LSTM and MLP)."""
        if isinstance(qf, LSTMQFunction):
            q_val, _ = qf(obs, action)
            return q_val
        else:
            return qf(obs, action)

    def _update_target_network(self):
        ptu.soft_update_from_to(self.vf, self.target_vf, self.soft_target_tau)

    # ------------------------------------------------------------------ #
    #  Snapshot / evaluation helpers
    # ------------------------------------------------------------------ #

    def get_epoch_snapshot(self, epoch):
        # Override without calling super() (base class references
        # self.exploration_policy which doesn't exist for RL2)
        snapshot = OrderedDict(
            epoch=epoch,
            qf1=self.qf1.state_dict(),
            qf2=self.qf2.state_dict(),
            vf=self.vf.state_dict(),
            policy=self.agent.policy.state_dict(),
            target_vf=self.target_vf.state_dict(),
        )
        return snapshot

    def _can_evaluate(self):
        return True

    def _can_train(self):
        return True

    def pretrain(self):
        pass
