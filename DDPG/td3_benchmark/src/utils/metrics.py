import numpy as np
from typing import Tuple

def compute_iqm(scores: np.ndarray) -> float:
    """Interquartile Mean (IQM) محاسبه"""
    q25, q75 = np.percentile(scores, [25, 75])
    filtered = scores[(scores >= q25) & (scores <= q75)]
    return float(np.mean(filtered)) if len(filtered) > 0 else float(np.mean(scores))

def stratified_bootstrap_ci(scores: np.ndarray, n_bootstraps: int = 1000, ci: float = 0.95) -> Tuple[float, float]:
    """Stratified Bootstrap Confidence Intervals محاسبه بازه‌های اطمینان"""
    bootstrapped_means = []
    n = len(scores)
    for _ in range(n_bootstraps):
        sample = np.random.choice(scores, size=n, replace=True)
        bootstrapped_means.append(np.mean(sample))
    
    lower = np.percentile(bootstrapped_means, (1 - ci) / 2 * 100)
    upper = np.percentile(bootstrapped_means, (1 + ci) / 2 * 100)
    return float(lower), float(upper)
