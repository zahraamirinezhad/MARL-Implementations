import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from src.networks.actor_critic import ActorNetwork, CriticNetwork

class AdvancedPPOAgent:
    def __init__(
        self,
        n_actions,
        input_dims,
        gamma=0.99,
        alpha=3e-4,
        gae_lambda=0.95,
        policy_clip=0.2,
        dual_clip=3.0,
        target_kl=0.015,
        kl_penalty_coef=0.2,
        use_kl_penalty=False,
        batch_size=128,
        n_epochs=10,
        entropy_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        is_continuous=False,
        is_visual=False,
        chkpt_dir='tmp/ppo'
    ):
        self.gamma = gamma
        self.alpha = alpha
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.dual_clip = dual_clip
        self.target_kl = target_kl
        self.kl_penalty_coef = kl_penalty_coef
        self.use_kl_penalty = use_kl_penalty
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.entropy_coef = entropy_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.is_continuous = is_continuous
        self.chkpt_dir = chkpt_dir

        os.makedirs(self.chkpt_dir, exist_ok=True)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.actor = ActorNetwork(n_actions, input_dims, is_continuous, is_visual).to(self.device)
        self.critic = CriticNetwork(input_dims, is_visual).to(self.device)

        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=alpha,
            eps=1e-5
        )

    def choose_action(self, observation):
        state = T.tensor(observation, dtype=T.float32).to(self.device)
        with T.no_grad():
            dist = self.actor(state)
            value = self.critic(state)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            if self.is_continuous:
                log_prob = log_prob.sum(axis=-1)
        return action.cpu().numpy(), log_prob.cpu().numpy(), value.squeeze(-1).cpu().numpy()

    def get_value(self, observation):
        state = T.tensor(observation, dtype=T.float32).to(self.device)
        with T.no_grad():
            value = self.critic(state).squeeze(-1)
        return value.cpu().numpy()

    def learn(self, memory_data, next_obs, writer=None, global_step=0):
        obs_b, actions_b, log_probs_b, rewards_b, terms_b, truncs_b, values_b = memory_data
        n_steps, n_envs = rewards_b.shape

        advantages = np.zeros_like(rewards_b, dtype=np.float32)
        next_value = self.get_value(next_obs)
        last_gae_lam = 0.0

        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                next_non_terminal = 1.0 - terms_b[t]
                next_values = next_value
            else:
                next_non_terminal = 1.0 - terms_b[t]
                next_values = values_b[t + 1]

            delta = rewards_b[t] + self.gamma * next_values * next_non_terminal - values_b[t]
            advantages[t] = last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam

        returns_b = advantages + values_b

        b_obs = T.tensor(obs_b.reshape((-1,) + obs_b.shape[2:]), dtype=T.float32).to(self.device)
        b_actions = T.tensor(actions_b.reshape((-1,) + actions_b.shape[2:])).to(self.device)
        b_log_probs = T.tensor(log_probs_b.reshape(-1), dtype=T.float32).to(self.device)
        b_advantages = T.tensor(advantages.reshape(-1), dtype=T.float32).to(self.device)
        b_returns = T.tensor(returns_b.reshape(-1), dtype=T.float32).to(self.device)
        b_values = T.tensor(values_b.reshape(-1), dtype=T.float32).to(self.device)

        batch_size = b_obs.shape[0]
        indices = np.arange(batch_size)

        a_losses, c_losses, e_losses, kl_divs = [], [], [], []

        for epoch in range(self.n_epochs):
            np.random.shuffle(indices)
            for start in range(0, batch_size, self.batch_size):
                end = start + self.batch_size
                mb_idx = indices[start:end]

                mb_obs = b_obs[mb_idx]
                mb_actions = b_actions[mb_idx]
                mb_old_log_probs = b_log_probs[mb_idx]
                mb_advantages = b_advantages[mb_idx]
                mb_returns = b_returns[mb_idx]
                mb_values = b_values[mb_idx]

                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                dist = self.actor(mb_obs)
                new_value = self.critic(mb_obs).squeeze(-1)

                new_log_probs = dist.log_prob(mb_actions)
                if self.is_continuous:
                    new_log_probs = new_log_probs.sum(axis=-1)

                log_ratio = new_log_probs - mb_old_log_probs
                ratio = log_ratio.exp()

                with T.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean().item()
                    kl_divs.append(approx_kl)

                # 1. PPO Standard Clip & Dual-Clip Implementation
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * T.clamp(ratio, 1 - self.policy_clip, 1 + self.policy_clip)

                # Dual-clip extension for negative advantages
                if self.dual_clip is not None:
                    pg_loss_dual = T.max(pg_loss1, -mb_advantages * self.dual_clip)
                    pg_loss = T.where(mb_advantages < 0, pg_loss_dual, T.max(pg_loss1, pg_loss2)).mean()
                else:
                    pg_loss = T.max(pg_loss1, pg_loss2).mean()

                # 2. Adaptive KL Penalty Option
                if self.use_kl_penalty:
                    pg_loss += self.kl_penalty_coef * approx_kl

                # Critic Value Loss Clip
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

                a_losses.append(pg_loss.item())
                c_losses.append(critic_loss.item())
                e_losses.append(entropy_loss.item())

            # Early Stopping via Target KL
            if self.target_kl is not None and np.mean(kl_divs) > self.target_kl:
                break

        if writer is not None:
            writer.add_scalar('Loss/Actor', np.mean(a_losses), global_step)
            writer.add_scalar('Loss/Critic', np.mean(c_losses), global_step)
            writer.add_scalar('Diagnostics/KL_Divergence', np.mean(kl_divs), global_step)
