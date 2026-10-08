import torch as T
import torch.nn as nn
import numpy as np
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, is_continuous=False, is_visual=False):
        super(ActorNetwork, self).__init__()
        self.is_continuous = is_continuous
        self.is_visual = is_visual

        if self.is_visual:
            self.network = nn.Sequential(
                layer_init(nn.Conv2d(input_dims[0], 32, 8, stride=4)),
                nn.ReLU(),
                layer_init(nn.Conv2d(32, 64, 4, stride=2)),
                nn.ReLU(),
                layer_init(nn.Conv2d(64, 64, 3, stride=1)),
                nn.ReLU(),
                nn.Flatten(),
                layer_init(nn.Linear(64 * 7 * 7, 512)),
                nn.ReLU()
            )
            fc_out_dim = 512
        else:
            self.network = nn.Sequential(
                layer_init(nn.Linear(*input_dims, 256)),
                nn.Tanh(),
                layer_init(nn.Linear(256, 256)),
                nn.Tanh()
            )
            fc_out_dim = 256

        if self.is_continuous:
            self.mu_head = layer_init(nn.Linear(fc_out_dim, n_actions), std=0.01)
            self.log_std = nn.Parameter(T.zeros(1, n_actions))
        else:
            self.action_head = layer_init(nn.Linear(fc_out_dim, n_actions), std=0.01)

    def forward(self, state):
        features = self.network(state)
        if self.is_continuous:
            mu = self.mu_head(features)
            # استفاده از squeeze برای پشتیبانی از ورودی‌های ۱ بعدی و ۲ بعدی
            action_std = self.log_std.squeeze(0).exp().expand_as(mu)
            return Normal(mu, action_std)
        else:
            logits = self.action_head(features)
            return Categorical(logits=logits)

class CriticNetwork(nn.Module):
    def __init__(self, input_dims, is_visual=False):
        super(CriticNetwork, self).__init__()
        if is_visual:
            self.critic = nn.Sequential(
                layer_init(nn.Conv2d(input_dims[0], 32, 8, stride=4)),
                nn.ReLU(),
                layer_init(nn.Conv2d(32, 64, 4, stride=2)),
                nn.ReLU(),
                layer_init(nn.Conv2d(64, 64, 3, stride=1)),
                nn.ReLU(),
                nn.Flatten(),
                layer_init(nn.Linear(64 * 7 * 7, 512)),
                nn.ReLU(),
                layer_init(nn.Linear(512, 1), std=1.0)
            )
        else:
            self.critic = nn.Sequential(
                layer_init(nn.Linear(*input_dims, 256)),
                nn.Tanh(),
                layer_init(nn.Linear(256, 256)),
                nn.Tanh(),
                layer_init(nn.Linear(256, 1), std=1.0)
            )

    def forward(self, state):
        return self.critic(state)
