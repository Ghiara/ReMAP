"""
Torch argmax policy
"""
import numpy as np
from third_party.Meta_RL.submodules.rlkit.rlkit.torch import nn

import third_party.rlkit.torch.pytorch_util as ptu
from third_party.rlkit.policies.base import Policy


class ArgmaxDiscretePolicy(nn.Module, Policy):
    def __init__(self, qf):
        super().__init__()
        self.qf = qf

    def get_action(self, obs):
        obs = np.expand_dims(obs, axis=0)
        obs = ptu.from_numpy(obs).float()
        q_values = self.qf(obs).squeeze(0)
        q_values_np = ptu.get_numpy(q_values)
        return q_values_np.argmax(), {}
