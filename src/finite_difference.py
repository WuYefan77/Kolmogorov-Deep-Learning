from __future__ import annotations
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import factorized
from .monte_carlo import black_scholes_call


def _operator_coefficients(S: np.ndarray, r: float, sigma: float, dS: float):
    a = 0.5 * sigma**2 * S**2 / dS**2 - r * S / (2*dS)
    b = -sigma**2 * S**2 / dS**2 - r
    c = 0.5 * sigma**2 * S**2 / dS**2 + r * S / (2*dS)
    return a, b, c


def crank_nicolson_call(strike: float = 100.0, r: float = 0.05, sigma: float = 0.2,
                        T: float = 1.0, s_max: float = 400.0,
                        n_space: int = 400, n_time: int = 800):
    """Black-Scholes call at t=0 via Crank-Nicolson in tau=T-t."""
    S = np.linspace(0.0, s_max, n_space + 1)
    dS = S[1] - S[0]
    dt = T / n_time
    interior = S[1:-1]
    a, b, c = _operator_coefficients(interior, r, sigma, dS)
    m = len(interior)

    A = diags([
        -0.5*dt*a[1:],
        1.0 - 0.5*dt*b,
        -0.5*dt*c[:-1]
    ], offsets=[-1, 0, 1], shape=(m, m), format="csc")
    B = diags([
        0.5*dt*a[1:],
        1.0 + 0.5*dt*b,
        0.5*dt*c[:-1]
    ], offsets=[-1, 0, 1], shape=(m, m), format="csc")
    solve_A = factorized(A)

    V = np.maximum(S - strike, 0.0)
    for n in range(n_time):
        tau_n = n * dt
        tau_np1 = (n + 1) * dt
        low_n = 0.0
        low_np1 = 0.0
        high_n = s_max - strike * np.exp(-r * tau_n)
        high_np1 = s_max - strike * np.exp(-r * tau_np1)
        rhs = B @ V[1:-1]
        rhs[0] += 0.5*dt*a[0]*(low_n + low_np1)
        rhs[-1] += 0.5*dt*c[-1]*(high_n + high_np1)
        V[1:-1] = solve_A(rhs)
        V[0] = low_np1
        V[-1] = high_np1
    return S, V


def explicit_call(strike: float = 100.0, r: float = 0.05, sigma: float = 0.2,
                  T: float = 1.0, s_max: float = 400.0,
                  n_space: int = 100, n_time: int = 1000):
    """Forward-in-tau explicit Black-Scholes solver, useful for stability demonstrations."""
    S = np.linspace(0.0, s_max, n_space + 1)
    dS = S[1] - S[0]
    dt = T / n_time
    interior = S[1:-1]
    a, b, c = _operator_coefficients(interior, r, sigma, dS)
    V = np.maximum(S - strike, 0.0)
    max_abs_history = [float(np.max(np.abs(V)))]
    for n in range(n_time):
        tau = (n + 1) * dt
        old = V.copy()
        V[1:-1] = old[1:-1] + dt * (a*old[:-2] + b*old[1:-1] + c*old[2:])
        V[0] = 0.0
        V[-1] = s_max - strike * np.exp(-r * tau)
        max_abs_history.append(float(np.max(np.abs(V))))
        if not np.all(np.isfinite(V)) or np.max(np.abs(V)) > 1e12:
            break
    return S, V, np.asarray(max_abs_history)


def fd_error_metrics(S: np.ndarray, V: np.ndarray, strike: float, r: float,
                     sigma: float, T: float, eval_low: float = 50.0,
                     eval_high: float = 150.0):
    mask = (S >= eval_low) & (S <= eval_high)
    truth = black_scholes_call(S[mask], strike, r, sigma, T)
    err = V[mask] - truth
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "max_abs_error": float(np.max(np.abs(err))),
    }
