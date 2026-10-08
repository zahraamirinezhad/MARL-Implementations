import numpy as np
from src.buffers.replay_buffer import PrioritizedReplayBuffer

def test_buffer_sampling():
    buffer = PrioritizedReplayBuffer(max_size=100, input_shape=(8,), n_actions=2)
    for _ in range(10):
        s = np.random.randn(8)
        a = np.random.randn(2)
        r = 1.0
        s_ = np.random.randn(8)
        d = False
        buffer.store_transition(s, a, r, s_, d)

    states, actions, rewards, states_, dones, indices, weights = buffer.sample_buffer(batch_size=4)
    assert states.shape[0] == 4
    assert len(indices) == 4
    print("Unit test for PrioritizedReplayBuffer passed successfully!")

if __name__ == "__main__":
    test_buffer_sampling()
