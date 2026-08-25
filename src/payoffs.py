from __future__ import annotations
import numpy as np


def call_payoff(s: np.ndarray, strike: float) -> np.ndarray:
    """European call payoff (S-K)^+."""
    s = np.asarray(s)
    return np.maximum(s - strike, 0.0)


def basket_call_payoff(s: np.ndarray, strike: float) -> np.ndarray:
    """Arithmetic basket call payoff (mean_i S_i-K)^+."""
    s = np.asarray(s)
    return np.maximum(np.mean(s, axis=-1) - strike, 0.0)


def max_call_payoff(s: np.ndarray, strike: float) -> np.ndarray:
    """Call on the maximum asset, (max_i S_i-K)^+."""
    s = np.asarray(s)
    return np.maximum(np.max(s, axis=-1) - strike, 0.0)
