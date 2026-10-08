import pytest
import numpy as np
from src.agents.mappo import MAPPOAgent

def test_mappo_ctde_forward_pass():
    """Validates CTDE Tensor dimensions and multi-agent forward updates."""
    num_agents = 3
    local_obs_dim = 8
    global_state_dim = num_agents * local_obs_dim
    action_dim = 5

    agent = MAPPOAgent(num_agents, local_obs_dim, global_state_dim, action_dim)

    # Dummy Multi-Agent Data
    obs_dict = {f"agent_{i}": np.random.randn(local_obs_dim).astype(np.float32) for i in range(num_agents)}
    global_state = np.random.randn(global_state_dim).astype(np.float32)

    actions, log_probs, value = agent.choose_actions(obs_dict, global_state)

    assert len(actions) == num_agents
    assert len(log_probs) == num_agents
    assert isinstance(value, (float, np.float32, np.ndarray))

# ==============================================================================
# 2. RUNNING AUTOMATED UNIT TESTS
# ==============================================================================
print("\n---> Running PyTest Unit Tests for CTDE MAPPO Engine <---")
!pytest tests/
