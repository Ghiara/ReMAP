from collections import OrderedDict

import numpy as np
import third_party.Meta_RL.submodules.rlkit.rlkit.torch as torch
import torch.optim as optim
from third_party.Meta_RL.submodules.rlkit.rlkit.torch import nn as nn
import torch.nn.functional as F

import third_party.rlkit.torch.pytorch_util as ptu
from third_party.rlkit.core.eval_util import create_stats_ordered_dict
from third_party.Meta_RL.submodules.rlkit.rlkit.torch.torch_rl_algorithm import TorchTrainer

from third_party.Meta_RL.submodules.rlkit.rlkit.util.io import load_local_or_remote_file

import random
from third_party.rlkit.torch.core import np_to_pytorch_batch
from third_party.rlkit.data_management.path_builder import PathBuilder

# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt

from third_party.rlkit.core import logger

import glob

class PathLoader:
    """
    Loads demonstrations and/or off-policy data into a Trainer
    """

    def load_demos(self, ):
        pass
