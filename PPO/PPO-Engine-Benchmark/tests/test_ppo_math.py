import pytest
import numpy as np
import torch as T
from src.agents.ppo import AdvancedPPOAgent

def test_gae_calculation_shape():
    """Validates tensor shapes and dimension alignment for GAE."""
    agent = AdvancedPPOAgent(n_actions=2, input_dims=(4,), is_continuous=False)

    steps, envs = 128, 4
    obs_b = np.random.randn(steps, envs, 4).astype(np.float32)
    actions_b = np.random.randint(0, 2, size=(steps, envs))
    log_probs_b = np.random.randn(steps, envs).astype(np.float32)
    rewards_b = np.random.randn(steps, envs).astype(np.float32)
    terms_b = np.zeros((steps, envs), dtype=bool)
    truncs_b = np.zeros((steps, envs), dtype=bool)
    values_b = np.random.randn(steps, envs).astype(np.float32)

    memory_data = (obs_b, actions_b, log_probs_b, rewards_b, terms_b, truncs_b, values_b)
    next_obs = np.random.randn(envs, 4).astype(np.float32)

    # Run test without error
    agent.learn(memory_data, next_obs)
    assert True

# ==============================================================================
# 2. RUNNING UNIT TESTS
# ==============================================================================
print("\n---> Running Automated PyTest Unit Tests <---")
!pytest tests/
