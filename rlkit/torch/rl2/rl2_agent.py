"""
RL2 Agent: Uses LSTM-based policy for fast adaptation.
Compatible with the PEARL agent interface so it works with
MetaRLAlgorithm's train() loop and InPlacePathSampler/rollout.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np

import rlkit.torch.pytorch_util as ptu
from rlkit.torch.core import np_ify
from rlkit.torch.rl2.networks import LSTMPolicy


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

    # ---------- PEARL-compatible interface ----------

    def clear_z(self, num_tasks=1):
        """Reset LSTM hidden state and context (analogous to resetting z to prior)."""
        self._hidden_state = self.policy.init_hidden(num_tasks)
        self.z = ptu.zeros(num_tasks, self.latent_dim)
        self.z_means = ptu.zeros(num_tasks, self.latent_dim)
        self.z_vars = ptu.ones(num_tasks, self.latent_dim)
        self.context = None

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
        Append a single transition to the context buffer.
        For RL2 the LSTM hidden state already incorporates the trajectory
        during get_action(), so this just stores data for compatibility.
        """
        o, a, r, no, d, info = inputs
        o = ptu.from_numpy(o[None, None, ...])
        a = ptu.from_numpy(a[None, None, ...])
        r = ptu.from_numpy(np.array([r])[None, None, ...])
        no = ptu.from_numpy(no[None, None, ...])
        data = torch.cat([o, a, r], dim=2)
        if self.context is None:
            self.context = data
        else:
            self.context = torch.cat([self.context, data], dim=1)

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

        # Ensure hidden state batch size matches input (e.g., after training with meta_batch > 1)
        if self._hidden_state is not None:
            h, c = self._hidden_state
            if h.size(1) != obs.size(0):
                self._hidden_state = self.policy.init_hidden(obs.size(0))

        with torch.no_grad():
            mean, log_std, new_hidden = self.policy(obs, self._hidden_state)
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

    def forward_policy(self, obs, hidden_state=None):
        """Forward pass through policy for training."""
        return self.policy(obs, hidden_state)

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
