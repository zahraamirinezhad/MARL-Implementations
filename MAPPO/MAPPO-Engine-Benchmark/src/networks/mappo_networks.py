import torch as T
import torch.nn as nn
import numpy as np
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

class DecentralizedActor(nn.Module):
    """
    Decentralized Actor: Receives local observations (obs_dim) and generates action distribution.
    """
    def __init__(self, local_obs_dim, action_dim, is_continuous=False):
        super(DecentralizedActor, self).__init__()
        self.is_continuous = is_continuous

        self.network = nn.Sequential(
            layer_init(nn.Linear(local_obs_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh()
        )

        if self.is_continuous:
            self.mu_head = layer_init(nn.Linear(256, action_dim), std=0.01)
            self.log_std = nn.Parameter(T.zeros(1, action_dim))
        else:
            self.action_head = layer_init(nn.Linear(256, action_dim), std=0.01)

    def forward(self, local_obs):
        features = self.network(local_obs)
        if self.is_continuous:
            mu = self.mu_head(features)
            action_std = self.log_std.squeeze(0).exp().expand_as(mu)
            return Normal(mu, action_std)
        else:
            logits = self.action_head(features)
            return Categorical(logits=logits)

class CentralizedCritic(nn.Module):
    """
    Centralized Critic (CTDE): Receives Global State / Concatenated Observations (state_dim)
    and estimates the centralized state-value V(S).
    """
    def __init__(self, global_state_dim):
        super(CentralizedCritic, self).__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(global_state_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1), std=1.0)
        )

    def forward(self, global_state):
        return self.critic(global_state)
