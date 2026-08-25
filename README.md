# Deep Learning Approximation of Kolmogorov PDEs

An independent numerical machine-learning project connecting stochastic differential equations, Feynman–Kac representations, Monte Carlo, finite differences, and neural conditional-expectation regression.

## Project question

For the risk-neutral geometric Brownian motion

$$
dS_t=rS_t\,dt+\sigma S_t\,dW_t,
$$

the Black–Scholes call value is

$$
V(0,S)=e^{-rT}\,\mathbb{E}[(S_T-K)^+\mid S_0=S].
$$

Instead of estimating this expectation separately at every initial state, the project samples initial conditions and stochastic terminal payoffs, then fits a neural network by squared loss. The population minimizer is the conditional expectation, so the network learns an amortized approximation of the whole solution surface.

## Main findings

| Experiment | Reference result |
|---|---:|
| Euler–Maruyama strong-order slope | 0.507 |
| Euler–Maruyama weak-order slopes | 0.998 and 0.996 |
| Monte Carlo RMSE slope | -0.500 |
| Crank–Nicolson MAE on $S\in[50,150]$ | $9.53\times10^{-4}$ |
| One-dimensional neural relative $L^2$ error | 0.337% |
| 50D basket neural relative $L^2$ error | 1.28% |
| 50D max-call neural relative $L^2$ error | 2.05% |

The results support a deliberately qualified conclusion: classical finite differences remain preferable for this one-dimensional benchmark, while neural Feynman–Kac regression becomes useful as an amortized solution-map approximation when repeated queries and higher-dimensional state spaces make tensor grids impractical.

![Neural conditional-expectation approximation](figures/06_neural_conditional_expectation.png)

![Error versus dimension](figures/08_error_vs_dimension_max.png)

## What is implemented

1. **SDE discretization** — Euler–Maruyama versus exact GBM, with coupled strong and moment-based weak convergence checks.
2. **Feynman–Kac Monte Carlo** — pointwise pricing, confidence intervals, antithetic sampling, and empirical $N^{-1/2}$ convergence.
3. **Finite differences** — an explicit stability demonstration and a Crank–Nicolson Black–Scholes solver.
4. **Neural Feynman–Kac** — online stochastic labels and an MLP approximation of $S\mapsto V(0,S)$.
5. **High-dimensional scaling** — arithmetic-basket and max-call payoffs for $d=1,2,5,10,20,50$, including accuracy and amortized runtime comparisons.

## Repository layout

```text
Kolmogorov-Deep-Learning/
├── notebooks/
│   ├── 01_sde_simulation.ipynb
│   ├── 02_monte_carlo.ipynb
│   ├── 03_black_scholes_pde.ipynb
│   ├── 04_neural_feynman_kac.ipynb
│   └── 05_high_dimensional_basket.ipynb
├── src/
│   ├── sde.py
│   ├── monte_carlo.py
│   ├── finite_difference.py
│   ├── neural_solver.py
│   └── payoffs.py
├── tests/test_core.py
├── figures/
├── results/
├── report/
├── run_experiments.py
└── requirements.txt
