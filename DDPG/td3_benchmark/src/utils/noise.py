import numpy as np

class GaussianNoise:
    def __init__(self, action_dim: int, sigma: float = 0.1):
        self.action_dim = action_dim
        self.sigma = sigma

    def __call__(self) -> np.ndarray:
        return np.random.normal(0, self.sigma, size=self.action_dim)

class OUNoise:
    def __init__(self, action_dim: int, mu: float = 0.0, theta: float = 0.15, sigma: float = 0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self) -> None:
        self.state = np.ones(self.action_dim) * self.mu

    def __call__(self) -> np.ndarray:
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state
