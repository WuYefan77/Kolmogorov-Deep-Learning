import numpy as np

from src.finite_difference import crank_nicolson_call, fd_error_metrics
from src.monte_carlo import black_scholes_call, mc_call_price
from src.neural_solver import error_metrics, qmc_reference_prices
from src.payoffs import basket_call_payoff, call_payoff, max_call_payoff
from src.sde import em_moment_exact, gbm_exact_terminal, gbm_moment_exact


def test_gbm_exact_terminal_zero_vol():
    out = gbm_exact_terminal(100.0, 0.05, 0.0, 1.0, np.array([0.0, 1.0]))
    assert np.allclose(out, 100 * np.exp(0.05))


def test_em_weak_error_shrinks():
    truth = gbm_moment_exact(100, 0.05, 0.2, 1.0, 2)
    e8 = abs(em_moment_exact(100, 0.05, 0.2, 1.0, 8, 2) - truth)
    e64 = abs(em_moment_exact(100, 0.05, 0.2, 1.0, 64, 2) - truth)
    assert e64 < e8


def test_black_scholes_reference_value():
    value = float(black_scholes_call(100, 100, 0.05, 0.2, 1.0))
    assert abs(value - 10.450583572185565) < 1e-10


def test_antithetic_mc_is_consistent_with_black_scholes():
    estimate, standard_error = mc_call_price(
        100, 100, 0.05, 0.2, 1.0, n_paths=20_000, seed=42, antithetic=True
    )
    truth = float(black_scholes_call(100, 100, 0.05, 0.2, 1.0))
    assert abs(estimate - truth) < 5 * standard_error
    assert standard_error > 0


def test_crank_nicolson_reasonable_accuracy():
    asset_grid, values = crank_nicolson_call(n_space=200, n_time=400)
    metrics = fd_error_metrics(asset_grid, values, 100, 0.05, 0.2, 1.0)
    assert metrics["mae"] < 0.05
    assert metrics["max_abs_error"] < 0.2


def test_payoff_helpers():
    states = np.array([[80.0, 120.0], [110.0, 130.0]])
    assert np.allclose(call_payoff(states, 100.0), [[0.0, 20.0], [10.0, 30.0]])
    assert np.allclose(basket_call_payoff(states, 100.0), [0.0, 20.0])
    assert np.allclose(max_call_payoff(states, 100.0), [20.0, 30.0])


def test_qmc_reference_matches_one_dimensional_closed_form():
    states = np.array([[80.0], [100.0], [120.0]])
    qmc = qmc_reference_prices(
        states,
        payoff="call",
        strike=100.0,
        r=0.05,
        sigma=0.2,
        T=1.0,
        n_paths=8192,
        seed=123,
    )
    exact = black_scholes_call(states[:, 0], 100.0, 0.05, 0.2, 1.0)
    assert np.max(np.abs(qmc - exact)) < 0.1


def test_error_metrics_are_zero_for_identical_arrays():
    metrics = error_metrics(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
    assert all(value == 0.0 for value in metrics.values())
