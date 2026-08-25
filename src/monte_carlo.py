from __future__ import annotations
import time
import numpy as np
import pandas as pd
from scipy.stats import norm
from .sde import gbm_exact_terminal


def black_scholes_call(s0, strike: float, r: float, sigma: float, T: float):
    s0 = np.asarray(s0, dtype=float)
    if T <= 0:
        return np.maximum(s0 - strike, 0.0)
    vol = sigma * np.sqrt(T)
    d1 = (np.log(s0 / strike) + (r + 0.5 * sigma**2) * T) / vol
    d2 = d1 - vol
    return s0 * norm.cdf(d1) - strike * np.exp(-r*T) * norm.cdf(d2)


def mc_call_price(s0: float, strike: float, r: float, sigma: float, T: float,
                  n_paths: int, seed: int = 1234, antithetic: bool = False):
    rng = np.random.default_rng(seed)
    if antithetic:
        # Treat each (+Z,-Z) pair as one independent variance-reduced observation.
        half = max(2, n_paths // 2)
        z = rng.standard_normal(half)
        st_plus = gbm_exact_terminal(s0, r, sigma, T, z)
        st_minus = gbm_exact_terminal(s0, r, sigma, T, -z)
        p_plus = np.exp(-r*T) * np.maximum(st_plus - strike, 0.0)
        p_minus = np.exp(-r*T) * np.maximum(st_minus - strike, 0.0)
        pair_average = 0.5 * (p_plus + p_minus)
        est = float(np.mean(pair_average))
        se = float(np.std(pair_average, ddof=1) / np.sqrt(half))
        return est, se
    z = rng.standard_normal(n_paths)
    st = gbm_exact_terminal(s0, r, sigma, T, z)
    discounted = np.exp(-r*T) * np.maximum(st - strike, 0.0)
    est = float(np.mean(discounted))
    se = float(np.std(discounted, ddof=1) / np.sqrt(n_paths))
    return est, se


def mc_convergence_experiment(s0: float = 100.0, strike: float = 100.0,
                              r: float = 0.05, sigma: float = 0.2, T: float = 1.0,
                              n_values=(100, 300, 1_000, 3_000, 10_000, 30_000, 100_000),
                              reps: int = 80, seed: int = 2026) -> pd.DataFrame:
    truth = float(black_scholes_call(s0, strike, r, sigma, T))
    rng = np.random.default_rng(seed)
    rows = []
    for n in n_values:
        estimates = np.empty(reps)
        t0 = time.perf_counter()
        for j in range(reps):
            z = rng.standard_normal(n)
            st = gbm_exact_terminal(s0, r, sigma, T, z)
            estimates[j] = np.exp(-r*T) * np.maximum(st - strike, 0.0).mean()
        elapsed = time.perf_counter() - t0
        rmse = float(np.sqrt(np.mean((estimates - truth)**2)))
        rows.append({
            "n_paths": n, "rmse": rmse, "bias": float(np.mean(estimates)-truth),
            "empirical_sd": float(np.std(estimates, ddof=1)),
            "mean_estimate": float(np.mean(estimates)), "truth": truth,
            "elapsed_seconds": elapsed
        })
    out = pd.DataFrame(rows)
    slope = np.polyfit(np.log(out["n_paths"]), np.log(out["rmse"]), 1)[0]
    out["fitted_slope"] = slope
    return out
