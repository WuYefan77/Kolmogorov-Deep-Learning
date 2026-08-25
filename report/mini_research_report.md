# Neural Feynman–Kac Approximation of Kolmogorov PDEs

## A compact numerical study of stochastic simulation, finite differences, and amortized deep learning

### Abstract

This study investigates a simple but important computational question for backward Kolmogorov equations: when is it useful to replace repeated pointwise Monte Carlo evaluation by a neural network trained directly from stochastic simulation? The experiments begin with risk-neutral geometric Brownian motion, for which the Black–Scholes formula provides an exact benchmark, and progressively compare four views of the same problem: stochastic differential equation simulation, Feynman–Kac Monte Carlo, finite-difference PDE solution, and neural conditional-expectation regression. The numerical pipeline first validates classical convergence theory. Euler–Maruyama gives an empirical strong-order slope of 0.507 and weak-order slopes of 0.998 and 0.996, while Monte Carlo pricing exhibits the expected RMSE slope of -0.500. In one dimension, a Crank–Nicolson solver is substantially more accurate than the neural approximation, achieving an MAE of 9.53e-4 on the main pricing interval. The neural Feynman–Kac solver nevertheless learns the entire Black–Scholes price curve from noisy terminal-payoff labels with relative L2 error 0.337%. In dimensions up to 50, the same simulation-and-regression principle achieves relative L2 errors of 1.28% for an arithmetic-basket payoff and 2.05% for a max-call payoff. The results support a qualified conclusion: neural Feynman–Kac methods are not compelling replacements for classical one-dimensional PDE solvers, but they provide a practical way to amortize repeated stochastic simulation into a reusable high-dimensional solution map when tensor-product grids become infeasible.

## 1. Problem formulation

Consider the diffusion

\[
dX_t = \mu(X_t)\,dt + \sigma(X_t)\,dW_t,
\]

with terminal payoff \(g\). Under standard regularity conditions, the backward Kolmogorov equation

\[
\partial_t u + \mu\cdot\nabla u
+ \frac12\operatorname{Tr}(\sigma\sigma^\top\nabla^2u)=0,
\qquad u(T,x)=g(x),
\]

admits the Feynman–Kac representation

\[
u(0,x)=\mathbb E[g(X_T^x)].
\]

The conventional Monte Carlo interpretation is pointwise: fix an initial state \(x\), simulate many terminal states, and estimate the expectation. The neural formulation changes the computational object. Let \(\xi\) be a random initial condition and let \(Y\) be a stochastic terminal payoff generated conditionally on \(\xi\). Training a function \(v_\theta\) by

\[
\min_\theta \mathbb E\big[(v_\theta(\xi)-Y)^2\big]
\]

has the population minimizer

\[
v^*(x)=\mathbb E[Y\mid \xi=x].
\]

Thus stochastic simulation supplies noisy labels, while regression learns the conditional-expectation surface. The main computational question is therefore not whether a neural network can beat Monte Carlo at one point, but whether the up-front training cost is justified when the solution must be queried repeatedly over many initial conditions or in dimensions where grid PDE methods are impractical.

The principal benchmark is the risk-neutral Black–Scholes model

\[
dS_t=rS_t\,dt+\sigma S_t\,dW_t,
\]

with European-call payoff \((S_T-K)^+\). Its value is

\[
V(0,S)=e^{-rT}\mathbb E[(S_T-K)^+\mid S_0=S],
\]

and the closed-form Black–Scholes price gives exact ground truth in one dimension.

## 2. Numerical methodology

The study is organized so that the machine-learning experiment is built on verified stochastic numerics rather than treated as an isolated prediction task.

First, Euler–Maruyama is tested against the exact GBM transition using common Brownian paths. Strong error is measured in RMS norm, while weak error is measured through the first and second moments. This separates pathwise approximation from distributional approximation and permits direct comparison with the classical orders \(1/2\) and \(1\).

Second, Feynman–Kac Monte Carlo is used for pointwise option pricing. Repeated experiments across sample sizes estimate the empirical RMSE law and confidence intervals. The expected benchmark is the canonical \(N^{-1/2}\) convergence rate.

Third, the Black–Scholes PDE is solved on a one-dimensional spatial grid. An explicit finite-difference scheme is retained mainly to demonstrate stability restrictions, while Crank–Nicolson provides the accurate deterministic reference method.

Fourth, a multilayer perceptron is trained on stochastic terminal payoffs. Initial prices are sampled uniformly on \([50,150]\), and fresh stochastic labels are generated during optimization rather than stored as a fixed supervised dataset. Antithetic Gaussian pairs reduce label variance. The resulting network approximates the entire map \(S_0\mapsto V(0,S_0)\).

Finally, the state dimension is increased to \(d\in\{1,2,5,10,20,50\}\). Independent GBMs are used with two payoffs:

\[
g_{\mathrm{basket}}(S)=\left(\frac1d\sum_{i=1}^dS_i-K\right)^+,
\]

and

\[
g_{\max}(S)=(\max_i S_i-K)^+.
\]

The second benchmark is deliberately included because an arithmetic basket can become easier as dimension increases: averaging induces concentration. The max-call payoff remains more sensitive to coordinate extremes and switching geometry. High-dimensional reference values are computed numerically with scrambled Sobol integration.

## 3. Results

### 3.1 Classical convergence checks

The stochastic-numerics portion reproduces the expected asymptotic laws closely. The fitted Euler–Maruyama strong-error slope is **0.507**, consistent with strong order \(1/2\). Weak-error slopes are **0.998** for the first moment and **0.996** for the second moment, both essentially order one.

Monte Carlo pricing at \(S_0=K=100\), \(r=0.05\), \(\sigma=0.2\), and \(T=1\) gives a fitted log-log RMSE slope of **-0.500**. The exact Black–Scholes price is **10.450584**, and at \(N=100000\) paths the repeated-simulation RMSE is approximately **0.0461**. These checks establish that the implementation reproduces standard stochastic convergence behavior before neural approximation is introduced.

### 3.2 Deterministic PDE benchmark

For the one-dimensional Black–Scholes PDE, Crank–Nicolson is extremely effective. On the finest tested grid, the error over \(S\in[50,150]\) is:

- MAE: **9.53e-4**;
- RMSE: **1.28e-3**;
- maximum absolute error: **2.50e-3**.

This is an important negative result for any overly broad machine-learning claim. In one spatial dimension, a conventional PDE grid is inexpensive, accurate, interpretable, and difficult to improve upon for this task. The neural solver is therefore not justified by one-dimensional accuracy.

### 3.3 Neural conditional-expectation learning

The neural Feynman–Kac solver reaches:

- MAE: **0.0650**;
- RMSE: **0.0817**;
- relative L2 error: **0.337%**;
- maximum absolute error: **0.287**.

The central observation is qualitative as well as quantitative. Individual discounted terminal payoffs are highly noisy as a function of the initial state, but their conditional mean is smooth. Squared-loss training recovers this conditional mean without requiring deterministic labels. In this sense the method converts a simulator into an amortized numerical representation of the solution surface.

### 3.4 High-dimensional scaling

The high-dimensional results are summarized below.

| Dimension | Basket relative L2 | Basket MAE | Max-call relative L2 | Max-call MAE |
|---:|---:|---:|---:|---:|
| 1 | 0.88% | 0.094 | 1.15% | 0.126 |
| 2 | 1.40% | 0.104 | 0.65% | 0.119 |
| 5 | 1.10% | 0.066 | 1.46% | 0.388 |
| 10 | 0.93% | 0.052 | 2.20% | 0.837 |
| 20 | 0.88% | 0.042 | 1.83% | 0.857 |
| 50 | **1.28%** | **0.052** | **2.05%** | **1.073** |

The arithmetic basket remains comparatively easy, partly because averaging reduces variability as dimension grows. The max-call benchmark produces larger absolute errors and a less favorable relative-error profile, which is consistent with its more difficult geometry. Even so, useful approximation accuracy is retained at dimension 50 with a small dense network.

These results should not be interpreted as proof that neural solvers possess dimension-independent complexity. They show only that, for the tested class of diffusion-payoff problems, simulation remains feasible and regression can represent the resulting conditional-expectation map in dimensions where a tensor-product finite-difference discretization would be unrealistic.

## 4. Amortization and computational interpretation

The fairest comparison between Monte Carlo and a trained neural solver is based on repeated queries. If \(Q\) different initial conditions must be evaluated, then approximately

\[
C_{\mathrm{NN}}(Q)=C_{\mathrm{train}}+Q\,c_{\mathrm{infer}},
\qquad
C_{\mathrm{MC}}(Q)=Q\,c_{\mathrm{MC}}.
\]

On the reference machine, the one-dimensional benchmark using 4096 Monte Carlo paths per query gives an estimated break-even point of roughly **1.06e4 queried states**. This number is implementation- and hardware-dependent and is not itself a universal result. The relevant conclusion is structural: Monte Carlo has negligible setup cost but repeatedly pays for simulation, whereas the neural method pays a large up-front optimization cost and then evaluates the learned map cheaply.

This distinction explains where the neural formulation is potentially valuable. It is unattractive when only a small number of prices are needed and classical low-dimensional PDE methods are available. It becomes more plausible when many state-dependent evaluations are required, especially in moderate or high dimension.

## 5. Limitations

The experiment deliberately uses a controlled setting. Asset dynamics have constant coefficients and independent Brownian drivers. Exact GBM transitions are used in the Monte Carlo and neural stages, so regression error is isolated from SDE time-discretization error. High-dimensional ground truth is numerical rather than analytic, and the neural networks receive only minimal hyperparameter tuning. Runtime results depend on hardware, batching, software libraries, and the chosen Monte Carlo path budget.

The project therefore does not establish that neural Feynman–Kac methods dominate specialized numerical PDE techniques, nor that they eliminate the curse of dimensionality for general nonlinear equations. Its purpose is narrower: to demonstrate, within a reproducible numerical framework, how conditional-expectation regression connects stochastic simulation to reusable PDE solution approximations and why that connection becomes computationally interesting as state dimension grows.

## 6. Conclusion

The experiments provide a coherent numerical progression from classical SDE approximation to neural PDE approximation. Euler–Maruyama and Monte Carlo reproduce their expected convergence laws, and Crank–Nicolson confirms that classical deterministic methods remain the preferred solution in one dimension. Neural Feynman–Kac regression, however, successfully learns an entire value function from stochastic labels alone and remains accurate on selected 50-dimensional basket and max-call problems.

The resulting interpretation is deliberately conservative. The neural network is not a universally faster option pricer and is not needed for standard one-dimensional Black–Scholes. Its role is instead **amortized conditional-expectation approximation**: simulation generates local stochastic information, and the network compresses that information into a function that can be queried repeatedly across the state space. That is the computational mechanism through which deep learning can become useful for high-dimensional Kolmogorov PDEs.

## References

1. C. Beck, S. Becker, P. Grohs, N. Jaafari, and A. Jentzen, “Solving the Kolmogorov PDE by means of deep learning,” *Journal of Scientific Computing*, 88, 2021.
2. KTH Royal Institute of Technology, **SF2525 Computational Methods for Stochastic Differential Equations and Machine Learning**.
3. F. Black and M. Scholes, “The Pricing of Options and Corporate Liabilities,” *Journal of Political Economy*, 81(3), 1973.
