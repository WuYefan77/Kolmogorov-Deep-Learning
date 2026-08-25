import numpy as np
from src.sde import gbm_exact_terminal, em_moment_exact, gbm_moment_exact
from src.monte_carlo import black_scholes_call
from src.finite_difference import crank_nicolson_call, fd_error_metrics


def test_gbm_exact_terminal_zero_vol():
    out = gbm_exact_terminal(100.0, 0.05, 0.0, 1.0, np.array([0.0, 1.0]))
    assert np.allclose(out, 100*np.exp(0.05))


def test_em_weak_error_shrinks():
    truth = gbm_moment_exact(100, 0.05, 0.2, 1.0, 2)
    e8 = abs(em_moment_exact(100, 0.05, 0.2, 1.0, 8, 2)-truth)
    e64 = abs(em_moment_exact(100, 0.05, 0.2, 1.0, 64, 2)-truth)
    assert e64 < e8


def test_black_scholes_reference_value():
    v = float(black_scholes_call(100, 100, 0.05, 0.2, 1.0))
    assert abs(v - 10.450583572185565) < 1e-10


def test_crank_nicolson_reasonable_accuracy():
    S, V = crank_nicolson_call(n_space=200, n_time=400)
    m = fd_error_metrics(S, V, 100, 0.05, 0.2, 1.0)
    assert m["mae"] < 0.05
    assert m["max_abs_error"] < 0.2
