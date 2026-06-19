"""RL2 implementation for meta-reinforcement learning"""

from third_party.rlkit.torch.rl2.rl2_agent import RL2Agent
from third_party.rlkit.torch.rl2.rl2_sac import RL2SoftActorCritic
from third_party.rlkit.torch.rl2.networks import LSTMPolicy, LSTMQFunction

__all__ = [
    'RL2Agent',
    'RL2SoftActorCritic',
    'LSTMPolicy',
    'LSTMQFunction',
]
