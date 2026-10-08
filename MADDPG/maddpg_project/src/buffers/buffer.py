import numpy as np
import torch as T

class MultiAgentReplayBuffer:
    def __init__(self, max_size, critic_dims, actor_dims, n_actions, n_agents, batch_size, device='cuda:0'):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.n_agents = n_agents
        self.actor_dims = actor_dims
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.device = device

        self.state_memory = np.zeros((self.mem_size, critic_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, critic_dims), dtype=np.float32)
        self.reward_memory = np.zeros((self.mem_size, n_agents), dtype=np.float32)
        self.terminal_memory = np.zeros((self.mem_size, n_agents), dtype=bool)

        self.actor_state_memory = [np.zeros((self.mem_size, actor_dims[i]), dtype=np.float32) for i in range(n_agents)]
        self.actor_new_state_memory = [np.zeros((self.mem_size, actor_dims[i]), dtype=np.float32) for i in range(n_agents)]
        self.actor_action_memory = [np.zeros((self.mem_size, n_actions), dtype=np.float32) for i in range(n_agents)]

    def store_transition(self, raw_obs, state, action, reward, raw_obs_, state_, done):
        index = self.mem_cntr % self.mem_size

        for agent_idx in range(self.n_agents):
            self.actor_state_memory[agent_idx][index] = raw_obs[agent_idx]
            self.actor_new_state_memory[agent_idx][index] = raw_obs_[agent_idx]
            self.actor_action_memory[agent_idx][index] = action[agent_idx]

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        states = T.tensor(self.state_memory[batch], dtype=T.float32).to(self.device)
        states_ = T.tensor(self.new_state_memory[batch], dtype=T.float32).to(self.device)
        rewards = T.tensor(self.reward_memory[batch], dtype=T.float32).to(self.device)
        terminal = T.tensor(self.terminal_memory[batch], dtype=T.bool).to(self.device)

        actor_states = [T.tensor(self.actor_state_memory[i][batch], dtype=T.float32).to(self.device) for i in range(self.n_agents)]
        actor_new_states = [T.tensor(self.actor_new_state_memory[i][batch], dtype=T.float32).to(self.device) for i in range(self.n_agents)]
        actions = [T.tensor(self.actor_action_memory[i][batch], dtype=T.float32).to(self.device) for i in range(self.n_agents)]

        return actor_states, states, actions, rewards, actor_new_states, states_, terminal

    def ready(self):
        return self.mem_cntr >= self.batch_size
