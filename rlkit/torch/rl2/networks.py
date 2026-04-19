"""
LSTM-based neural networks for RL2
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import numpy as np

from rlkit.torch.networks import Mlp
import rlkit.torch.pytorch_util as ptu


class LSTMPolicy(nn.Module):
    """LSTM-based policy network for RL2"""
    
    def __init__(
        self,
        obs_dim,
        action_dim,
        hidden_size=256,
        num_layers=1,
        use_tanh=True,
        init_w=3e-3,
        **kwargs
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_tanh = use_tanh
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=obs_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Mean output layer
        self.mean_fc = nn.Linear(hidden_size, action_dim)
        self.mean_fc.weight.data.uniform_(-init_w, init_w)
        self.mean_fc.bias.data.uniform_(-init_w, init_w)
        
        # Log std output layer
        self.log_std_fc = nn.Linear(hidden_size, action_dim)
        self.log_std_fc.weight.data.uniform_(-init_w, init_w)
        self.log_std_fc.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, obs, hidden_state=None):
        """
        Args:
            obs: (batch, seq_len, obs_dim) or (batch, obs_dim)
            hidden_state: tuple of (h, c) for LSTM state
        
        Returns:
            mean: (batch, seq_len, action_dim) or (batch, action_dim)
            log_std: (batch, seq_len, action_dim) or (batch, action_dim)
            hidden_state: new (h, c)
        """
        # Handle 2D input
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)  # (batch, 1, obs_dim)
            squeeze_output = True
        else:
            squeeze_output = False
        
        lstm_out, hidden_state = self.lstm(obs, hidden_state)
        
        mean = self.mean_fc(lstm_out)
        log_std = self.log_std_fc(lstm_out)
        log_std = torch.clamp(log_std, -20, 2)
        
        if squeeze_output:
            mean = mean.squeeze(1)
            log_std = log_std.squeeze(1)
        
        return mean, log_std, hidden_state
    
    def init_hidden(self, batch_size):
        """Initialize hidden state"""
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        if ptu.gpu_enabled():
            h = h.cuda()
            c = c.cuda()
        return (h, c)


class LSTMQFunction(nn.Module):
    """LSTM-based Q-function for RL2"""
    
    def __init__(
        self,
        obs_dim,
        action_dim,
        hidden_size=256,
        num_layers=1,
        init_w=3e-3,
        **kwargs
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM takes obs+action concatenated
        self.lstm = nn.LSTM(
            input_size=obs_dim + action_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Output Q-value
        self.fc = nn.Linear(hidden_size, 1)
        self.fc.weight.data.uniform_(-init_w, init_w)
        self.fc.bias.data.uniform_(-init_w, init_w)
    
    def forward(self, obs, action, hidden_state=None):
        """
        Args:
            obs: (batch, seq_len, obs_dim) or (batch, obs_dim)
            action: (batch, seq_len, action_dim) or (batch, action_dim)
            hidden_state: tuple of (h, c)
        
        Returns:
            q_value: (batch, seq_len, 1) or (batch, 1)
            hidden_state: new (h, c)
        """
        # Handle 2D inputs
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)
            action = action.unsqueeze(1)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Concatenate obs and action
        x = torch.cat([obs, action], dim=-1)
        
        lstm_out, hidden_state = self.lstm(x, hidden_state)
        q_value = self.fc(lstm_out)
        
        if squeeze_output:
            q_value = q_value.squeeze(1)
        
        return q_value, hidden_state
    
    def init_hidden(self, batch_size):
        """Initialize hidden state"""
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        if ptu.gpu_enabled():
            h = h.cuda()
            c = c.cuda()
        return (h, c)
