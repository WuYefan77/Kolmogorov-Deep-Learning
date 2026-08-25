# Deep Learning Approximation of Kolmogorov PDEs

## A numerical study of SDE simulation, Feynman–Kac, finite differences, and amortized neural solvers

### Abstract

This project studies four numerical views of the same pricing problem: exact Black–Scholes evaluation, Monte Carlo simulation of the underlying stochastic differential equation, deterministic finite differences for the associated PDE, and neural Feynman–Kac regression. The objective is not to show that a neural network is universally superior. Instead, it asks when learning the conditional expectation as a function of the initial state is useful, especially when classical state-space grids become impossible in high dimension.

For geometric Brownian motion, Euler–Maruyama exhibits an empirical strong convergence slope of **0.507**, while weak errors in the first two moments have slopes **0.998** and **0.996**, matching the expected orders $1/2$ and $1$. Pointwise Monte Carlo pricing produces an RMSE slope of **-0.500**, matching $N^{-1/2}$. A Crank–Nicolson Black–Scholes solver reaches MAE **9.526e-04** on $S\in[50,150]$ at the finest grid. A neural Feynman–Kac solver trained only from stochastic terminal payoffs approximates the full one-dimensional price curve with relative $L^2$ error **0.34%**. In dimensions up to 50, relative $L^2$ error remains about **1.28%** for the arithmetic basket and **2.05%** for the harder max-call benchmark.

## 1. Mathematical setting

Consider the diffusion

$$dX_t=\mu(X_t)dt+\sigma(X_t)dW_t.$$

For a terminal payoff $g$, the backward Kolmogorov equation is

$$\partial_tu+\mu\cdot\nabla u+\tfrac12\operatorname{Tr}(\sigma\sigma^\top\nabla^2u)=0,\qquad u(T,x)=g(x),$$

with Feynman–Kac representation $u(0,x)=\mathbb E[g(X_T^x)]$ under the corresponding assumptions.

For risk-neutral geometric Brownian motion,

$$dS_t=rS_tdt+\sigma S_tdW_t,$$

and a European call with strike $K$ has

$$V(0,S)=e^{-rT}\mathbb E[(S_T-K)^+\mid S_0=S].$$

The associated Black–Scholes PDE contains the killing term $-rV$:

$$V_t+rSV_S+\tfrac12\sigma^2S^2V_{SS}-rV=0.$$

The neural idea is to sample an initial state $\xi$ and a stochastic discounted terminal payoff $Y$, and minimize

$$\mathbb E[(v_\theta(\xi)-Y)^2].$$

Over all square-integrable functions, the population minimizer is

$$v^*(x)=\mathbb E[Y\mid\xi=x]=V(0,x).$$

Thus simulation provides noisy labels while regression recovers the conditional mean surface.

## 2. Task 1 — Euler–Maruyama convergence

Euler–Maruyama for GBM is

$$S_{n+1}=S_n+rS_n\Delta t+\sigma S_n\sqrt{\Delta t}Z_n.$$

The exact terminal transition is

$$S_T=S_0\exp((r-\sigma^2/2)T+\sigma W_T).$$

For strong error, exact and discrete solutions are coupled using the same Brownian increments. The fitted RMS-error slope is **0.5072**, close to the theoretical $1/2$ order.

Weak convergence is evaluated through the first and second moments. Because the Euler multiplicative step is explicit, these discrete moments can be calculated exactly, avoiding a Monte Carlo noise floor. The fitted slopes are **0.9982** for $\phi(S)=S$ and **0.9958** for $\phi(S)=S^2$.

See `figures/01_strong_convergence.png` and `figures/02_weak_convergence.png`.

## 3. Task 2 — Feynman–Kac Monte Carlo

The estimator

$$\widehat V_N(S_0)=e^{-rT}\frac1N\sum_{i=1}^N(S_T^{(i)}-K)^+$$

is evaluated against the closed-form Black–Scholes value. The reference value at $S_0=K=100$, $r=0.05$, $\sigma=0.2$, $T=1$ is **10.450584**. Across repeated simulations, the fitted log-log RMSE slope is **-0.4999**, essentially the theoretical $-1/2$. At $N=100000$, the empirical RMSE across repetitions is **0.0461**.

This stage also demonstrates confidence intervals and antithetic sampling.

## 4. Task 3 — deterministic PDE benchmark

The PDE is evolved forward in time-to-maturity $\tau=T-t$. Two finite-difference roles are separated:

- an explicit scheme demonstrates the stability restriction and can become numerically unstable for an overly large time step;
- Crank–Nicolson provides the accurate deterministic benchmark.

At $n_S=400$ and $n_t=800$, the error on $S\in[50,150]$ is:

- MAE: **9.526e-04**;
- RMSE: **1.282e-03**;
- maximum absolute error: **2.498e-03**.

In one dimension, the deterministic PDE method is therefore extremely competitive: the grid produces the entire price curve with accuracy far beyond that of a modest pointwise Monte Carlo budget.

## 5. Task 4 — neural Feynman–Kac solver

Initial states are sampled uniformly from $[50,150]$. Each SGD batch generates new stochastic terminal states rather than reusing a fixed supervised dataset. Antithetic pairs $Z$ and $-Z$ reduce label variance without changing the conditional expectation target.

The trained MLP reaches:

- MAE: **0.0650**;
- RMSE: **0.0817**;
- relative $L^2$ error: **0.337%**;
- maximum absolute error: **0.2872**.

The key diagnostic is `figures/06_neural_conditional_expectation.png`: individual discounted payoffs form a highly noisy cloud, yet their conditional mean is the smooth Black–Scholes value curve, which the network recovers.

## 6. Task 5 — high-dimensional experiment

For independent GBMs, two terminal payoffs are considered:

$$g_{basket}(S)=\left(\frac1d\sum_{i=1}^dS_i-K\right)^+,$$

and

$$g_{max}(S)=(\max_iS_i-K)^+.$$

The second payoff is important because the arithmetic basket can become easier with dimension: averaging concentrates. A max payoff remains sensitive to extreme coordinates and is nonsmooth across switching surfaces.

### 6.1 Accuracy by dimension

| d | Basket rel. L2 | Basket MAE | Max rel. L2 | Max MAE |
|---:|---:|---:|---:|---:|
| 1 | 0.88% | 0.094 | 1.15% | 0.126 |
| 2 | 1.40% | 0.104 | 0.65% | 0.119 |
| 5 | 1.10% | 0.066 | 1.46% | 0.388 |
| 10 | 0.93% | 0.052 | 2.20% | 0.837 |
| 20 | 0.88% | 0.042 | 1.83% | 0.857 |
| 50 | 1.28% | 0.052 | 2.05% | 1.073 |

At $d=50$, the basket relative $L^2$ error is **1.28%**, while the max-call relative $L^2$ error is **2.05%**. The absolute max-call error grows more strongly because both the scale and geometric complexity of the payoff change with dimension.

These results do **not** prove dimension-independent complexity. They show that, for this bounded numerical experiment, a small dense neural network continues to approximate the Feynman–Kac map at useful accuracy in dimensions where a tensor-product finite-difference grid is not realistic.

### 6.2 Amortization

A trained network has an up-front optimization cost and a very small inference cost. Monte Carlo has essentially no training cost but pays a simulation cost for every new state. Therefore the appropriate comparison is

$$C_{NN}(Q)=C_{train}+Qc_{infer},\qquad C_{MC}(Q)=Qc_{MC}.$$

On the reference machine, the one-dimensional basket/call timing gives an estimated break-even of roughly **10563 queried states** for a 4096-path Monte Carlo benchmark. This number is explicitly machine- and implementation-dependent. The scientific point is the existence of an amortization trade-off, not the particular threshold.

## 7. What the experiment says

Three conclusions are robust.

First, the project reproduces the expected numerical-analysis laws before introducing machine learning: strong order $1/2$, weak order $1$, and Monte Carlo order $N^{-1/2}$ all appear cleanly. This matters because the neural experiment is then embedded in validated stochastic numerics rather than treated as an isolated ML demo.

Second, in one spatial dimension, classical finite differences are superior if the goal is simply to solve Black–Scholes accurately on a grid. The neural solver is not justified by 1D accuracy alone.

Third, the role of the neural method changes in high dimension. A tensor grid scales exponentially in dimension, while simulation remains feasible and the neural network converts repeated stochastic simulation into an amortized function approximation. The 50-dimensional experiments illustrate this mechanism, but they should not be read as a theorem that neural methods escape the curse of dimensionality for arbitrary PDEs.

## 8. Limitations

The high-dimensional reference values are numerical (scrambled Sobol integration), not analytic. The models use independent assets and constant coefficients. Hyperparameter search is intentionally minimal. Timing results depend on hardware, batching, language implementation, and the Monte Carlo budget. The neural benchmark also uses exact GBM transitions, so it isolates regression/integration error rather than combining it with Euler time-discretization error.

These restrictions are deliberate: the project is a numerical-methods study, not an attempt to reproduce every experiment of Beck et al.

## 9. Conclusion

The most important conceptual result is the change from pointwise estimation to conditional-expectation learning. Monte Carlo supplies stochastic labels, while squared-loss regression learns the whole initial-state-to-value map. In low dimension, deterministic PDE solvers remain hard to beat. In high dimension, the attraction of neural Feynman–Kac methods is that they preserve simulation-based tractability while producing a reusable approximation of the solution surface.

## References

1. C. Beck, S. Becker, P. Grohs, N. Jaafari, A. Jentzen, **Solving the Kolmogorov PDE by means of deep learning**, *Journal of Scientific Computing* 88 (2021), arXiv:1806.00421.
2. F. Black and M. Scholes, **The Pricing of Options and Corporate Liabilities**, *Journal of Political Economy* 81(3), 1973.
