from __future__ import annotations
import numpy as np
import pandas as pd


def gbm_exact_terminal(s0, r: float, sigma: float, T: float, z):
    """Exact GBM transition over [0,T] for standard-normal z."""
    return np.asarray(s0) * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * np.asarray(z))


def gbm_em_terminal(s0: float, r: float, sigma: float, T: float, n_steps: int,
                    rng: np.random.Generator, n_paths: int) -> np.ndarray:
    dt = T / n_steps
    s = np.full(n_paths, float(s0))
    for _ in range(n_steps):
        z = rng.standard_normal(n_paths)
        s *= 1.0 + r * dt + sigma * np.sqrt(dt) * z
    return s


def coupled_strong_rmse(s0: float, r: float, sigma: float, T: float, n_steps: int,
                        n_paths: int = 100_000, seed: int = 1234,
                        chunk_size: int = 10_000) -> float:
    """RMS strong error at T using the same Brownian path for exact GBM and EM."""
    rng = np.random.default_rng(seed + n_steps)
    dt = T / n_steps
    sum_sq = 0.0
    done = 0
    while done < n_paths:
        m = min(chunk_size, n_paths - done)
        s_em = np.full(m, float(s0))
        wT = np.zeros(m)
        for _ in range(n_steps):
            z = rng.standard_normal(m)
            dW = np.sqrt(dt) * z
            wT += dW
            s_em *= 1.0 + r * dt + sigma * dW
        s_ex = s0 * np.exp((r - 0.5 * sigma**2) * T + sigma * wT)
        sum_sq += np.sum((s_em - s_ex) ** 2)
        done += m
    return float(np.sqrt(sum_sq / n_paths))


def em_moment_exact(s0: float, r: float, sigma: float, T: float,
                    n_steps: int, power: int) -> float:
    """Exact first/second moment of the Euler-Maruyama GBM discretization."""
    dt = T / n_steps
    if power == 1:
        one_step = 1.0 + r * dt
    elif power == 2:
        one_step = (1.0 + r * dt) ** 2 + sigma**2 * dt
    else:
        raise ValueError("Only power=1 or power=2 is implemented.")
    return float((s0**power) * (one_step**n_steps))


def gbm_moment_exact(s0: float, r: float, sigma: float, T: float, power: int) -> float:
    """Exact p-th moment of GBM for p=1 or p=2 (formula is valid more generally)."""
    p = float(power)
    return float((s0**p) * np.exp(p * r * T + 0.5 * p * (p - 1.0) * sigma**2 * T))


def convergence_tables(s0: float = 100.0, r: float = 0.05, sigma: float = 0.2,
                       T: float = 1.0, steps=(4, 8, 16, 32, 64, 128, 256),
                       n_paths: int = 80_000) -> tuple[pd.DataFrame, pd.DataFrame]:
    strong_rows = []
    for n in steps:
        err = coupled_strong_rmse(s0, r, sigma, T, n, n_paths=n_paths)
        strong_rows.append({"n_steps": n, "dt": T/n, "strong_rmse": err})
    strong = pd.DataFrame(strong_rows)
    strong_slope = np.polyfit(np.log(strong["dt"]), np.log(strong["strong_rmse"]), 1)[0]
    strong["fitted_slope"] = strong_slope

    weak_rows = []
    for p in (1, 2):
        truth = gbm_moment_exact(s0, r, sigma, T, p)
        for n in steps:
            approx = em_moment_exact(s0, r, sigma, T, n, p)
            weak_rows.append({
                "moment_power": p, "n_steps": n, "dt": T/n,
                "em_expectation": approx, "exact_expectation": truth,
                "weak_abs_error": abs(approx - truth)
            })
    weak = pd.DataFrame(weak_rows)
    weak["fitted_slope"] = np.nan
    for p in (1, 2):
        idx = weak["moment_power"] == p
        slope = np.polyfit(np.log(weak.loc[idx, "dt"]), np.log(weak.loc[idx, "weak_abs_error"]), 1)[0]
        weak.loc[idx, "fitted_slope"] = slope
    return strong, weak
