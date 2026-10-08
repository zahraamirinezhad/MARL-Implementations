import numpy as np
import torch
from typing import Tuple, Dict, Any

class PrioritizedReplayBuffer:
    def __init__(self, max_size: int, input_shape: Tuple[int, ...], n_actions: int, alpha: float = 0.6):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.alpha = alpha

        self.state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=np.float32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool_)
        self.priorities = np.zeros((self.mem_size,), dtype=np.float32)

    def store_transition(self, state: np.ndarray, action: np.ndarray, reward: float, state_: np.ndarray, done: bool) -> None:
        index = self.mem_cntr % self.mem_size
        max_priority = self.priorities.max() if self.mem_cntr > 0 else 1.0

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.priorities[index] = max_priority

        self.mem_cntr += 1

    def sample_buffer(self, batch_size: int, beta: float = 0.4) -> Tuple[np.ndarray, ...]:
        max_mem = min(self.mem_cntr, self.mem_size)
        
        priorities = self.priorities[:max_mem]
        probs = priorities ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(max_mem, batch_size, p=probs, replace=False)

        total = max_mem
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        states = self.state_memory[indices]
        actions = self.action_memory[indices]
        rewards = self.reward_memory[indices]
        states_ = self.new_state_memory[indices]
        dones = self.terminal_memory[indices]

        return states, actions, rewards, states_, dones, indices, weights

    def update_priorities(self, batch_indices: np.ndarray, batch_priorities: np.ndarray) -> None:
        for idx, priority in zip(batch_indices, batch_priorities):
            self.priorities[idx] = priority + 1e-5
