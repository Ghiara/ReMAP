import torch as _torch
from torch import *  # noqa: F401,F403
from torch import nn

# Re-export torch symbols so existing rlkit code can treat
# `third_party.rlkit.torch` like the upstream torch module.
__all__ = [name for name in dir(_torch) if not name.startswith('_')] + ['nn']
