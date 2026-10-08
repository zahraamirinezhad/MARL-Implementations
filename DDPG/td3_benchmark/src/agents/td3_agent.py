import numpy as np
import torch as T
import torch.nn.functional as F
from typing import Tuple, Dict, Any
from src.models.networks import ActorNetwork, CriticNetwork
from src.buffers.replay_buffer import PrioritizedReplayBuffer
from src.utils.noise import GaussianNoise, OUNoise

class TD3Agent:
    def __init__(self, cfg: Dict[str, Any], input_dims: Tuple[int, ...], n_actions: int, max_action: float, min_action: float, device: T.device, is_pixel: bool = False):
        self.gamma = cfg['gamma']
        self.tau = cfg['tau']
        self.max_action = max_action
        self.min_action = min_action
        self.n_actions = n_actions
        self.device = device
        self.warmup = cfg['warmup']
        self.batch_size = cfg['batch_size']
        self.update_actor_iter = cfg['actor_update_interval']
        self.time_step = 0
        self.learn_step_cntr = 0

        self.memory = PrioritizedReplayBuffer(1000000, input_dims, n_actions, alpha=cfg['per_alpha'])
        self.beta = cfg['per_beta']
        self.beta_increment = cfg['per_beta_increment']

        if cfg['exploration_noise_type'] == 'ou':
            self.noise = OUNoise(n_actions, sigma=cfg['noise_std'])
        else:
            self.noise = GaussianNoise(n_actions, sigma=cfg['noise_std'])

        self.actor = ActorNetwork(cfg['alpha'], input_dims, 400, 300, n_actions, max_action, is_pixel).to(device)
        self.critic = CriticNetwork(cfg['beta'], input_dims, 400, 300, n_actions, is_pixel).to(device)

        self.target_actor = ActorNetwork(cfg['alpha'], input_dims, 400, 300, n_actions, max_action, is_pixel).to(device)
        self.target_critic = CriticNetwork(cfg['beta'], input_dims, 400, 300, n_actions, is_pixel).to(device)

        self.update_network_parameters(tau=1.0)

    def choose_action(self, observation: np.ndarray, evaluate: bool = False) -> np.ndarray:
        if self.time_step < self.warmup and not evaluate:
            action = np.random.uniform(self.min_action, self.max_action, size=self.n_actions)
        else:
            state = T.tensor(np.array([observation]), dtype=T.float32).to(self.device)
            self.actor.eval()
            with T.no_grad():
                action = self.actor(state).cpu().numpy()[0]
            self.actor.train()

            if not evaluate:
                action += self.noise()

        action = np.clip(action, self.min_action, self.max_action)
        if not evaluate:
            self.time_step += 1
        return action

    def remember(self, state: np.ndarray, action: np.ndarray, reward: float, new_state: np.ndarray, done: bool) -> None:
        self.memory.store_transition(state, action, reward, new_state, done)

    def learn(self) -> Tuple[float, float]:
        if self.memory.mem_cntr < self.batch_size:
            return 0.0, 0.0

        state, action, reward, new_state, done, indices, weights = self.memory.sample_buffer(self.batch_size, beta=self.beta)
        self.beta = min(1.0, self.beta + self.beta_increment)

        states = T.tensor(state, dtype=T.float32).to(self.device)
        actions = T.tensor(action, dtype=T.float32).to(self.device)
        rewards = T.tensor(reward, dtype=T.float32).unsqueeze(1).to(self.device)
        states_ = T.tensor(new_state, dtype=T.float32).to(self.device)
        dones = T.tensor(done, dtype=T.bool).unsqueeze(1).to(self.device)
        is_weights = T.tensor(weights, dtype=T.float32).unsqueeze(1).to(self.device)

        with T.no_grad():
            target_actions = self.target_actor(states_)
            target_noise = T.clamp(T.randn_like(target_actions) * 0.2, -0.5, 0.5)
            target_actions = T.clamp(target_actions + target_noise, self.min_action, self.max_action)

            q1_target, q2_target = self.target_critic(states_, target_actions)
            q_target = T.min(q1_target, q2_target)
            q_target[dones] = 0.0
            y = rewards + self.gamma * q_target

        q1, q2 = self.critic(states, actions)
        
        td_errors = T.abs(q1 - y).detach().cpu().numpy().squeeze()
        self.memory.update_priorities(indices, td_errors)

        critic_loss = (is_weights * F.mse_loss(q1, y, reduction='none') + is_weights * F.mse_loss(q2, y, reduction='none')).mean()

        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        self.learn_step_cntr += 1
        actor_loss = 0.0

        if self.learn_step_cntr % self.update_actor_iter == 0:
            actor_loss_val = -self.critic.Q1(states, self.actor(states)).mean()
            actor_loss = actor_loss_val.item()

            self.actor.optimizer.zero_grad()
            actor_loss_val.backward()
            self.actor.optimizer.step()

            self.update_network_parameters()

        return actor_loss, critic_loss.item()

    def update_network_parameters(self, tau: float = None) -> None:
        if tau is None:
            tau = self.tau

        for actor_param, target_actor_param in zip(self.actor.parameters(), self.target_actor.parameters()):
            target_actor_param.data.copy_(tau * actor_param.data + (1.0 - tau) * target_actor_param.data)

        for critic_param, target_critic_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_critic_param.data.copy_(tau * critic_param.data + (1.0 - tau) * target_critic_param.data)
