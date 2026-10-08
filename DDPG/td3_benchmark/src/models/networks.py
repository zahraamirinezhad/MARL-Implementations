import torch as T
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class CNNEncoder(nn.Module):
    def __init__(self, input_shape: Tuple[int, ...]):
        super(CNNEncoder, self).__init__()
        # ورودی معمولاً (C, H, W) است
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        with T.no_grad():
            dummy = T.zeros(1, *input_shape)
            self.out_dim = self.conv(dummy).shape[1]

    def forward(self, x: T.Tensor) -> T.Tensor:
        return self.conv(x / 255.0)

class ActorNetwork(nn.Module):
    def __init__(self, lr: float, input_dims: Tuple[int, ...], fc1_dims: int, fc2_dims: int, n_actions: int, max_action: float, is_pixel: bool = False):
        super(ActorNetwork, self).__init__()
        self.is_pixel = is_pixel
        self.max_action = max_action

        if is_pixel:
            self.encoder = CNNEncoder(input_dims)
            in_dim = self.encoder.out_dim
        else:
            in_dim = input_dims[0]

        self.fc1 = nn.Linear(in_dim, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.mu = nn.Linear(fc2_dims, n_actions)
        self.optimizer = T.optim.Adam(self.parameters(), lr=lr)

    def forward(self, state: T.Tensor) -> T.Tensor:
        x = self.encoder(state) if self.is_pixel else state
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return T.tanh(self.mu(x)) * self.max_action

class CriticNetwork(nn.Module):
    def __init__(self, lr: float, input_dims: Tuple[int, ...], fc1_dims: int, fc2_dims: int, n_actions: int, is_pixel: bool = False):
        super(CriticNetwork, self).__init__()
        self.is_pixel = is_pixel

        if is_pixel:
            self.encoder = CNNEncoder(input_dims)
            in_dim = self.encoder.out_dim
        else:
            in_dim = input_dims[0]

        self.q1_fc1 = nn.Linear(in_dim + n_actions, fc1_dims)
        self.q1_fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.q1_out = nn.Linear(fc2_dims, 1)

        self.q2_fc1 = nn.Linear(in_dim + n_actions, fc1_dims)
        self.q2_fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.q2_out = nn.Linear(fc2_dims, 1)

        self.optimizer = T.optim.Adam(self.parameters(), lr=lr)

    def forward(self, state: T.Tensor, action: T.Tensor) -> Tuple[T.Tensor, T.Tensor]:
        x = self.encoder(state) if self.is_pixel else state
        sa = T.cat([x, action], dim=1)

        q1 = F.relu(self.q1_fc1(sa))
        q1 = F.relu(self.q1_fc2(q1))
        q1 = self.q1_out(q1)

        q2 = F.relu(self.q2_fc1(sa))
        q2 = F.relu(self.q2_fc2(q2))
        q2 = self.q2_out(q2)

        return q1, q2

    def Q1(self, state: T.Tensor, action: T.Tensor) -> T.Tensor:
        x = self.encoder(state) if self.is_pixel else state
        sa = T.cat([x, action], dim=1)
        q1 = F.relu(self.q1_fc1(sa))
        q1 = F.relu(self.q1_fc2(q1))
        return self.q1_out(q1)
