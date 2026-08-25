# Deep Learning Approximation of Kolmogorov PDEs

An SF2525-style numerical project connecting stochastic differential equations, Feynman–Kac, Monte Carlo, finite differences, and neural conditional-expectation regression.

## Project question

For the risk-neutral GBM

\[
dS_t=rS_t\,dt+\sigma S_t\,dW_t,
\]

the Black–Scholes call value is

\[
V(0,S)=e^{-rT}\,\mathbb E[(S_T-K)^+\mid S_0=S].
\]

Instead of estimating the expectation separately at every initial state, sample initial conditions and terminal stochastic payoffs, then fit a neural network by squared loss. The population minimizer is the conditional expectation, so the network learns an *amortized approximation of the whole solution surface*.

## What is implemented

1. **SDE discretization** — Euler–Maruyama vs exact GBM, strong and weak convergence.
2. **Feynman–Kac Monte Carlo** — pointwise pricing, confidence intervals, empirical \(N^{-1/2}\) convergence.
3. **Finite differences** — explicit stability demonstration and Crank–Nicolson Black–Scholes solver.
4. **Neural Feynman–Kac** — online stochastic labels and an MLP approximation of \(S\mapsto V(0,S)\).
5. **High-dimensional scaling** — arithmetic-basket and max-call payoffs for \(d=1,2,5,10,20,50\), including accuracy, training cost, inference cost and amortization.

## Repository layout

```text
kth-sf2525-kolmogorov-deep-learning/
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
├── figures/
├── results/
├── report/
│   ├── report.md
│   └── mini_research_report.md
├── tests/test_core.py
├── run_experiments.py
└── requirements.txt
```

## Reproduce

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
python run_experiments.py
jupyter lab
```

All random seeds used by the reference run are fixed in `run_experiments.py`.

## Reports

- `report/report.md` — detailed coursework-style numerical report.
- `report/mini_research_report.md` — compact standalone research report emphasizing the scientific question, main results, interpretation, and limitations.

## Numerical design choices

- Exact GBM transitions are used when the purpose is to isolate Monte Carlo or neural-regression error from time-discretization error.
- Strong convergence couples Euler–Maruyama and the exact GBM with the **same Brownian path**.
- Weak convergence is measured using exact first/second moments of the Euler discretization, avoiding Monte Carlo noise obscuring the expected order-one bias.
- The high-dimensional neural benchmark uses antithetic labels, while reference values use scrambled Sobol integration.
- Neural-vs-MC runtime is reported as an **amortization question**, not as an unfair claim that a trained network is intrinsically cheaper than one Monte Carlo price.

## References

- C. Beck, S. Becker, P. Grohs, N. Jaafari, A. Jentzen, *Solving the Kolmogorov PDE by means of deep learning*, Journal of Scientific Computing 88 (2021); arXiv:1806.00421.
- KTH SF2525, *Computational Methods for Stochastic Differential Equations and Machine Learning*.
