import abc
import logging

import numpy as np
import third_party.Meta_RL.submodules.rlkit.rlkit.torch as torch
import torch.nn.functional as F
from third_party.Meta_RL.submodules.rlkit.rlkit.torch import nn as nn

import third_party.rlkit.torch.pytorch_util as ptu
from third_party.rlkit.policies.base import ExplorationPolicy
from third_party.rlkit.torch.core import torch_ify, elem_or_tuple_to_numpy
from third_party.rlkit.torch.distributions import (
    Delta, TanhNormal, MultivariateDiagonalNormal, GaussianMixture, GaussianMixtureFull,
)
from third_party.rlkit.torch.networks import Mlp, CNN
from third_party.Meta_RL.submodules.rlkit.rlkit.torch.networks.basic import MultiInputSequential
from third_party.Meta_RL.submodules.rlkit.rlkit.torch.networks.stochastic.distribution_generator import (
    DistributionGenerator
)
from third_party.Meta_RL.submodules.rlkit.rlkit.torch.sac.policies.base import (
    TorchStochasticPolicy,
    PolicyFromDistributionGenerator,
    MakeDeterministic,
)


class LatentVariableModel(nn.Module):
    def __init__(
            self,
            encoder,
            decoder,
            **kwargs
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
