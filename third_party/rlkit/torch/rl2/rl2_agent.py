"""
RL2 Agent: Uses LSTM-based policy for fast adaptation.
Compatible with the PEARL agent interface so it works with
MetaRLAlgorithm's train() loop and InPlacePathSampler/rollout.
"""
import third_party.rlkit.torch as torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np

import third_party.rlkit.torch.pytorch_util as ptu
from third_party.rlkit.torch.core import np_ify
from third_party.rlkit.torch.rl2.networks import LSTMPolicy


class RL2Agent(nn.Module):
    """
    RL2 agent with LSTM policy for online adaptation.

    Implements the same interface as PEARLAgent so that
    MetaRLAlgorithm.train(), InPlacePathSampler, and rollout()
    can use it without modification.

    The LSTM hidden state acts as the "latent z" for task identification.
    """

    def __init__(
        self,
        obs_dim,
        action_dim,
        hidden_size=256,
        num_layers=1,
        policy_lr=3e-4,
        latent_dim=None,
        **kwargs
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # For compatibility: latent_dim defaults to hidden_size
        self.latent_dim = latent_dim if latent_dim is not None else hidden_size

        # LSTM policy
        self.policy = LSTMPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )

        self.policy_optimizer = Adam(self.policy.parameters(), lr=policy_lr)
        self.policy_lr = policy_lr

        # Hidden state (acts as latent z)
        self._hidden_state = None
        # Expose z as a tensor for compatibility with rollout / sampler
        self.z = ptu.zeros(1, self.latent_dim)
        self.z_means = ptu.zeros(1, self.latent_dim)
        self.z_vars = ptu.ones(1, self.latent_dim)
        # Context buffer
        self.context = None
        # Previous action/reward for RL2 LSTM input
        self._prev_action = torch.zeros(1, action_dim)
        self._prev_reward = torch.zeros(1, 1)

    # ---------- PEARL-compatible interface ----------

    def clear_z(self, num_tasks=1):
        """Reset LSTM hidden state and context (analogous to resetting z to prior)."""
        self._hidden_state = self.policy.init_hidden(num_tasks)
        self.z = ptu.zeros(num_tasks, self.latent_dim)
        self.z_means = ptu.zeros(num_tasks, self.latent_dim)
        self.z_vars = ptu.ones(num_tasks, self.latent_dim)
        self.context = None
        self._prev_action = ptu.zeros(num_tasks, self.action_dim)
        self._prev_reward = ptu.zeros(num_tasks, 1)

    def detach_z(self):
        """Detach hidden state from computation graph."""
        if self._hidden_state is not None:
            h, c = self._hidden_state
            self._hidden_state = (h.detach(), c.detach())
        self.z = self.z.detach()

    def sample_z(self):
        """No-op for RL2 (hidden state is deterministic given the trajectory)."""
        pass

    def update_context(self, inputs):
        """
        Append a single transition to the context buffer and update
        prev_action/prev_reward for the next LSTM step.
        """
        o, a, r, no, d, info = inputs
        o = ptu.from_numpy(o[None, None, ...])
        a_tensor = ptu.from_numpy(a[None, None, ...])
        r_tensor = ptu.from_numpy(np.array([r])[None, None, ...])
        no = ptu.from_numpy(no[None, None, ...])
        data = torch.cat([o, a_tensor, r_tensor], dim=2)
        if self.context is None:
            self.context = data
        else:
            self.context = torch.cat([self.context, data], dim=1)
        # Update previous action/reward for next LSTM step
        self._prev_action = ptu.from_numpy(a[None])                            # (1, action_dim)
        self._prev_reward = ptu.from_numpy(np.array([[r]], dtype=np.float32))  # (1, 1)

    def infer_posterior(self, context):
        """
        For RL2, 'posterior inference' means running context through the LSTM
        to update the hidden state. Since the hidden state is already updated
        step-by-step during get_action(), this is a no-op.
        """
        pass

    def compute_kl_div(self):
        """No KL divergence for RL2 (no variational inference)."""
        return ptu.zeros(1)

    def get_action(self, obs, deterministic=False):
        """
        Sample action from the LSTM policy.

        Returns in the PEARL-compatible format:
            ((action_np, agent_info), z_np)
        """
        if isinstance(obs, np.ndarray):
            obs = ptu.from_numpy(obs[None])  # (1, obs_dim)

        with torch.no_grad():
            mean, log_std, new_hidden = self.policy(
                obs,
                prev_action=self._prev_action,
                prev_reward=self._prev_reward,
                hidden_state=self._hidden_state,
            )
            self._hidden_state = new_hidden

            if deterministic:
                action = torch.tanh(mean)
            else:
                std = torch.exp(log_std)
                dist = torch.distributions.Normal(mean, std)
                pre_tanh = dist.rsample()
                action = torch.tanh(pre_tanh)

            action_np = ptu.get_numpy(action)[0]

            # Update z to reflect current hidden state (for logging / sampler)
            h = self._hidden_state[0]  # (num_layers, batch, hidden_size)
            self.z = h[-1]  # last layer: (batch, hidden_size)
            if self.z.shape[-1] >= self.latent_dim:
                z_out = self.z[:, :self.latent_dim]
            else:
                z_out = F.pad(self.z, (0, self.latent_dim - self.z.shape[-1]))

        agent_info = {}
        return (action_np, agent_info), np_ify(z_out.clone().detach())[0, :]

    def get_last_n_context_elements(self, n):
        if self.context is None:
            return None
        return self.context[:, -n:, :].detach().clone()

    # ---------- Training helpers ----------

    def forward_policy(self, obs, prev_action=None, prev_reward=None, hidden_state=None):
        """Forward pass through policy for training."""
        return self.policy(obs, prev_action=prev_action, prev_reward=prev_reward, hidden_state=hidden_state)

    def update_policy(self, loss):
        """Update policy with given loss."""
        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()

    def init_hidden(self, batch_size):
        """Initialize hidden state."""
        return self.policy.init_hidden(batch_size)

    def reset(self, num_tasks=1):
        """Alias for clear_z."""
        self.clear_z(num_tasks)

    def set_num_steps_total(self, n):
        """Compatibility with PEARL."""
        pass

    def log_diagnostics(self, eval_statistics):
        """Add RL2 specific diagnostics."""
        if self._hidden_state is not None:
            h = self._hidden_state[0]
            eval_statistics['RL2 hidden mean'] = float(h.mean())
            eval_statistics['RL2 hidden std'] = float(h.std())
