import numpy as np
import gymnasium as gym

class RunningMeanStd:
    """Tracks running mean and variance for online normalization."""
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

class NormalizedEnvWrapper(gym.Wrapper):
    """Normalizes observations and scales rewards for numerical stability in PPO."""
    def __init__(self, env, gamma=0.99, epsilon=1e-8):
        super().__init__(env)
        self.gamma = gamma
        self.epsilon = epsilon
        self.obs_rms = RunningMeanStd(shape=self.observation_space.shape)
        self.ret_rms = RunningMeanStd(shape=())
        self.returns = np.zeros(self.unwrapped.num_envs if hasattr(self.unwrapped, 'num_envs') else 1)

    def step(self, action):
        obs, rews, dones, truncs, infos = self.env.step(action)
        self.obs_rms.update(obs)
        normalized_obs = np.clip((obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon), -10.0, 10.0)

        self.returns = self.returns * self.gamma + rews
        self.ret_rms.update(self.returns)
        scaled_rews = rews / np.sqrt(self.ret_rms.var + self.epsilon)

        self.returns[dones | truncs] = 0.0
        return normalized_obs, scaled_rews, dones, truncs, infos

    def reset(self, **kwargs):
        obs, infos = self.env.reset(**kwargs)
        self.obs_rms.update(obs)
        normalized_obs = np.clip((obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon), -10.0, 10.0)
        return normalized_obs, infos
