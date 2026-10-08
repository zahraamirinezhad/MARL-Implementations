import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from src.networks.mappo_networks import DecentralizedActor, CentralizedCritic

class MAPPOAgent:
    """
    Multi-Agent PPO (MAPPO) with Centralized Training & Decentralized Execution (CTDE).
    Supports Parameter Sharing across homogeneous agents.
    """
    def __init__(
        self,
        num_agents,
        local_obs_dim,
        global_state_dim,
        action_dim,
        gamma=0.99,
        alpha=3e-4,
        gae_lambda=0.95,
        policy_clip=0.2,
        dual_clip=3.0,
        batch_size=128,
        n_epochs=10,
        entropy_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        is_continuous=False,
        chkpt_dir='tmp/mappo'
    ):
        self.num_agents = num_agents
        self.local_obs_dim = local_obs_dim
        self.global_state_dim = global_state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.alpha = alpha
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.dual_clip = dual_clip
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.entropy_coef = entropy_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.is_continuous = is_continuous
        self.chkpt_dir = chkpt_dir

        os.makedirs(self.chkpt_dir, exist_ok=True)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        # Parameter-Shared Actor and Centralized Critic
        self.actor = DecentralizedActor(local_obs_dim, action_dim, is_continuous).to(self.device)
        self.critic = CentralizedCritic(global_state_dim).to(self.device)

        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=alpha,
            eps=1e-5
        )

    def choose_actions(self, local_obs_dict, global_state):
        """
        Executes Decentralized Action Selection while querying Centralized Value.
        """
        actions = {}
        log_probs = {}

        # Convert local observations to tensors
        obs_tensor = T.tensor(np.array(list(local_obs_dict.values())), dtype=T.float32).to(self.device)
        state_tensor = T.tensor(global_state, dtype=T.float32).to(self.device)

        with T.no_grad():
            dist = self.actor(obs_tensor)
            value = self.critic(state_tensor)
            action_samples = dist.sample()
            log_prob_samples = dist.log_prob(action_samples)

            if self.is_continuous:
                log_prob_samples = log_prob_samples.sum(axis=-1)

        agent_keys = list(local_obs_dict.keys())
        for idx, key in enumerate(agent_keys):
            actions[key] = action_samples[idx].cpu().numpy()
            log_probs[key] = log_prob_samples[idx].cpu().numpy()

        return actions, log_probs, value.squeeze(-1).cpu().numpy()

    def get_value(self, global_state):
        state_tensor = T.tensor(global_state, dtype=T.float32).to(self.device)
        with T.no_grad():
            value = self.critic(state_tensor).squeeze(-1)
        return value.cpu().numpy()

    def learn(self, memory_data, next_global_state):
        obs_b, state_b, actions_b, log_probs_b, rewards_b, dones_b, values_b = memory_data
        n_steps = len(rewards_b)

        # Centralized Advantage Estimation (GAE)
        advantages = np.zeros_like(rewards_b, dtype=np.float32)
        next_value = self.get_value(next_global_state)
        last_gae_lam = 0.0

        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                next_non_terminal = 1.0 - float(dones_b[t])
                next_values = next_value
            else:
                next_non_terminal = 1.0 - float(dones_b[t])
                next_values = values_b[t + 1]

            delta = rewards_b[t] + self.gamma * next_values * next_non_terminal - values_b[t]
            advantages[t] = last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam

        returns_b = advantages + values_b

        # Flatten Step & Agent dimensions for Training
        b_obs = T.tensor(obs_b.reshape(-1, self.local_obs_dim), dtype=T.float32).to(self.device)
        b_states = T.tensor(np.repeat(state_b, self.num_agents, axis=0), dtype=T.float32).to(self.device)
        b_actions = T.tensor(actions_b.reshape(-1), dtype=T.int64 if not self.is_continuous else T.float32).to(self.device)
        b_log_probs = T.tensor(log_probs_b.reshape(-1), dtype=T.float32).to(self.device)
        
        # Repeat advantages/returns per agent for parameter-shared update
        b_advantages = T.tensor(np.repeat(advantages, self.num_agents), dtype=T.float32).to(self.device)
        b_returns = T.tensor(np.repeat(returns_b, self.num_agents), dtype=T.float32).to(self.device)
        b_values = T.tensor(np.repeat(values_b, self.num_agents), dtype=T.float32).to(self.device)

        batch_size = b_obs.shape[0]
        indices = np.arange(batch_size)

        for epoch in range(self.n_epochs):
            np.random.shuffle(indices)
            for start in range(0, batch_size, self.batch_size):
                end = start + self.batch_size
                mb_idx = indices[start:end]

                mb_obs = b_obs[mb_idx]
                mb_states = b_states[mb_idx]
                mb_actions = b_actions[mb_idx]
                mb_old_log_probs = b_log_probs[mb_idx]
                mb_advantages = b_advantages[mb_idx]
                mb_returns = b_returns[mb_idx]
                mb_values = b_values[mb_idx]

                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                dist = self.actor(mb_obs)
                new_value = self.critic(mb_states).squeeze(-1)

                new_log_probs = dist.log_prob(mb_actions)
                if self.is_continuous:
                    new_log_probs = new_log_probs.sum(axis=-1)

                log_ratio = new_log_probs - mb_old_log_probs
                ratio = log_ratio.exp()

                # PPO Clipped Loss with Dual-Clip
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * T.clamp(ratio, 1 - self.policy_clip, 1 + self.policy_clip)

                if self.dual_clip is not None:
                    pg_loss_dual = T.max(pg_loss1, -mb_advantages * self.dual_clip)
                    pg_loss = T.where(mb_advantages < 0, pg_loss_dual, T.max(pg_loss1, pg_loss2)).mean()
                else:
                    pg_loss = T.max(pg_loss1, pg_loss2).mean()

                # Critic Loss with Value Clipping
                v_loss_unclipped = (new_value - mb_returns) ** 2
                v_clipped = mb_values + T.clamp(new_value - mb_values, -self.policy_clip, self.policy_clip)
                v_loss_clipped = (v_clipped - mb_returns) ** 2
                critic_loss = 0.5 * T.max(v_loss_unclipped, v_loss_clipped).mean()

                entropy_loss = dist.entropy().mean()
                total_loss = pg_loss + self.vf_coef * critic_loss - self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()), self.max_grad_norm)
                self.optimizer.step()
