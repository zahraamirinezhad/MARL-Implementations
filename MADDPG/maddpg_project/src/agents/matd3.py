import torch as T
import torch.nn.functional as F
import numpy as np
from maddpg_project.src.networks.networks import ActorNetwork, TwinCriticNetwork

class Agent:
    def __init__(self, actor_dims, critic_dims, n_actions, n_agents, agent_idx, chkpt_dir, alpha=0.001, beta=0.001, fc1=64, fc2=64, gamma=0.95, tau=0.01):
        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions
        self.agent_name = f'agent_{agent_idx}'

        self.actor = ActorNetwork(alpha, actor_dims, fc1, fc2, n_actions, name=self.agent_name+'_actor', chkpt_dir=chkpt_dir)
        self.critic = TwinCriticNetwork(beta, critic_dims, fc1, fc2, n_agents, n_actions, name=self.agent_name+'_critic', chkpt_dir=chkpt_dir)
        self.target_actor = ActorNetwork(alpha, actor_dims, fc1, fc2, n_actions, name=self.agent_name+'_target_actor', chkpt_dir=chkpt_dir)
        self.target_critic = TwinCriticNetwork(beta, critic_dims, fc1, fc2, n_agents, n_actions, name=self.agent_name+'_target_critic', chkpt_dir=chkpt_dir)

        self.update_network_parameters(tau=1)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def choose_action(self, observation, evaluate=False):
        state = T.tensor(np.array([observation]), dtype=T.float32).to(self.actor.device)
        actions = self.actor(state)
        if not evaluate:
            noise = T.randn(self.n_actions).to(self.actor.device) * 0.1
            action = actions + noise
        else:
            action = actions
        action = T.clamp(action, 0.0, 1.0)
        return action.detach().cpu().numpy()[0]


class MATD3:
    def __init__(self, actor_dims, critic_dims, n_agents, n_actions, scenario='simple_adversary_v3', alpha=0.001, beta=0.001, fc1=64, fc2=64, gamma=0.95, tau=0.01, policy_update_freq=2, chkpt_dir='maddpg_project/saved_models'):
        self.agents = []
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.policy_update_freq = policy_update_freq
        self.learn_step_cntr = 0

        for agent_idx in range(self.n_agents):
            self.agents.append(Agent(actor_dims[agent_idx], critic_dims, n_actions, n_agents, agent_idx, alpha=alpha, beta=beta, fc1=fc1, fc2=fc2, gamma=gamma, tau=tau, chkpt_dir=chkpt_dir))

    def choose_action(self, raw_obs, evaluate=False):
        actions = []
        for agent_idx, agent in enumerate(self.agents):
            action = agent.choose_action(raw_obs[agent_idx], evaluate=evaluate)
            actions.append(action)
        return actions

    def learn(self, memory):
        if not memory.ready():
            return

        actor_states, states, actions, rewards, actor_new_states, states_, dones = memory.sample_buffer()

        all_agents_new_actions = []
        old_agents_actions = []

        for agent_idx, agent in enumerate(self.agents):
            new_pi = agent.target_actor(actor_new_states[agent_idx])
            # Target Policy Smoothing Noise (MATD3 feature)
            noise = (T.randn_like(new_pi) * 0.1).clamp(-0.2, 0.2)
            smoothed_new_pi = (new_pi + noise).clamp(0.0, 1.0)
            all_agents_new_actions.append(smoothed_new_pi)

            old_agents_actions.append(actions[agent_idx])

        new_actions = T.cat(all_agents_new_actions, dim=1)
        old_actions = T.cat(old_agents_actions, dim=1)

        # 1. Update Twin Critics
        for agent_idx, agent in enumerate(self.agents):
            with T.no_grad():
                q1_target, q2_target = agent.target_critic(states_, new_actions)
                q_target = T.min(q1_target, q2_target).flatten()
                q_target[dones[:, agent_idx]] = 0.0
                target = rewards[:, agent_idx] + agent.gamma * q_target

            q1, q2 = agent.critic(states, old_actions)
            critic_loss = F.mse_loss(target, q1.flatten()) + F.mse_loss(target, q2.flatten())

            agent.critic.optimizer.zero_grad()
            critic_loss.backward()
            agent.critic.optimizer.step()

        self.learn_step_cntr += 1

        # 2. Delayed Policy Updates (MATD3 feature)
        if self.learn_step_cntr % self.policy_update_freq == 0:
            for agent_idx, agent in enumerate(self.agents):
                mu_actions = []
                for idx, a in enumerate(self.agents):
                    if idx == agent_idx:
                        pi = a.actor(actor_states[idx])
                    else:
                        with T.no_grad():
                            pi = a.actor(actor_states[idx])
                    mu_actions.append(pi)

                mu_actions = T.cat(mu_actions, dim=1)
                actor_loss = -agent.critic.q1_forward(states, mu_actions).mean()

                agent.actor.optimizer.zero_grad()
                actor_loss.backward()
                agent.actor.optimizer.step()

                agent.update_network_parameters()

    def save_models(self):
        for agent in self.agents:
            agent.actor.save_checkpoint()
            agent.critic.save_checkpoint()

    def load_models(self):
        for agent in self.agents:
            agent.actor.load_checkpoint()
            agent.critic.load_checkpoint()
