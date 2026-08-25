from __future__ import annotations
import math
import time
from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy.stats import norm


class NormalizedMLP(nn.Module):
    def __init__(self, input_dim: int, low: float, high: float,
                 hidden=(96, 96, 96), positive_output: bool = True):
        super().__init__()
        self.low = float(low)
        self.high = float(high)
        self.positive_output = positive_output
        layers = []
        d = input_dim
        for h in hidden:
            layers.extend([nn.Linear(d, h), nn.SiLU()])
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)
        nn.init.constant_(self.net[-1].bias, -2.0)

    def forward(self, x):
        x = 2.0 * (x - self.low) / (self.high - self.low) - 1.0
        y = self.net(x).squeeze(-1)
        if self.positive_output:
            y = torch.nn.functional.softplus(y)
        return y


@dataclass
class TrainResult:
    model: nn.Module
    history: pd.DataFrame
    train_seconds: float


def _payoff_torch(st: torch.Tensor, strike: float, payoff: str) -> torch.Tensor:
    if payoff == "basket":
        return torch.relu(st.mean(dim=1) - strike)
    if payoff == "max":
        return torch.relu(st.max(dim=1).values - strike)
    if payoff == "call":
        if st.shape[1] != 1:
            raise ValueError("call payoff expects dim=1")
        return torch.relu(st[:, 0] - strike)
    raise ValueError(f"Unknown payoff: {payoff}")


def _equicorrelated_normals(batch: int, dim: int, rho: float, device, generator):
    if rho == 0.0:
        return torch.randn((batch, dim), generator=generator, device=device)
    common = torch.randn((batch, 1), generator=generator, device=device)
    iid = torch.randn((batch, dim), generator=generator, device=device)
    return math.sqrt(rho) * common + math.sqrt(1.0-rho) * iid


def train_neural_feynman_kac(dim: int = 1, payoff: str = "call",
                             low: float = 50.0, high: float = 150.0,
                             strike: float = 100.0, r: float = 0.05,
                             sigma: float = 0.2, T: float = 1.0,
                             rho: float = 0.0, steps: int = 1500,
                             batch_size: int = 2048, lr: float = 1e-3,
                             hidden=(96, 96, 96), seed: int = 2026,
                             antithetic: bool = False,
                             device: str = "cpu") -> TrainResult:
    torch.manual_seed(seed + dim)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed + 17*dim)
    model = NormalizedMLP(dim, low, high, hidden=hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    discount = math.exp(-r*T)
    drift = (r - 0.5*sigma**2)*T
    vol = sigma*math.sqrt(T)
    rows = []
    t0 = time.perf_counter()
    model.train()
    for step in range(1, steps+1):
        x = low + (high-low) * torch.rand((batch_size, dim), generator=gen, device=device)
        z = _equicorrelated_normals(batch_size, dim, rho, device, gen)
        st = x * torch.exp(drift + vol*z)
        y = discount * _payoff_torch(st, strike, payoff)
        if antithetic:
            st2 = x * torch.exp(drift - vol*z)
            y2 = discount * _payoff_torch(st2, strike, payoff)
            y = 0.5*(y+y2)
        pred_norm = model(x)
        y_norm = y / strike
        loss = torch.mean((pred_norm - y_norm)**2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step == 1 or step % 50 == 0 or step == steps:
            rows.append({"step": step, "loss": float(loss.detach().cpu()),
                         "rmse_price_batch": float(strike*torch.sqrt(loss).detach().cpu())})
    elapsed = time.perf_counter() - t0
    return TrainResult(model=model, history=pd.DataFrame(rows), train_seconds=elapsed)


def predict_prices(model: nn.Module, x: np.ndarray, strike: float = 100.0,
                   batch_size: int = 8192) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[:, None]
    device = next(model.parameters()).device
    out = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[i:i+batch_size]).to(device)
            out.append((strike*model(xb)).cpu().numpy())
    return np.concatenate(out)


def qmc_reference_prices(x: np.ndarray, payoff: str, strike: float = 100.0,
                         r: float = 0.05, sigma: float = 0.2, T: float = 1.0,
                         rho: float = 0.0, n_paths: int = 8192,
                         seed: int = 123, chunk_q: int = 16) -> np.ndarray:
    """Scrambled Sobol reference integration for multi-asset GBM."""
    from scipy.stats import qmc
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    q, dim = x.shape
    m_power = int(np.ceil(np.log2(n_paths)))
    sobol = qmc.Sobol(d=dim + (1 if rho > 0 else 0), scramble=True, seed=seed + dim)
    u = sobol.random_base2(m_power)[:n_paths]
    u = np.clip(u, 1e-12, 1-1e-12)
    zraw = norm.ppf(u)
    if rho > 0:
        common = zraw[:, :1]
        iid = zraw[:, 1:]
        z = np.sqrt(rho)*common + np.sqrt(1-rho)*iid
    else:
        z = zraw
    factors = np.exp((r-0.5*sigma**2)*T + sigma*np.sqrt(T)*z)
    disc = np.exp(-r*T)
    ans = np.empty(q)
    for i in range(0, q, chunk_q):
        xb = x[i:i+chunk_q]
        st = xb[:, None, :] * factors[None, :, :]
        if payoff == "basket":
            val = np.maximum(st.mean(axis=2)-strike, 0.0)
        elif payoff == "max":
            val = np.maximum(st.max(axis=2)-strike, 0.0)
        elif payoff == "call":
            val = np.maximum(st[:, :, 0]-strike, 0.0)
        else:
            raise ValueError(payoff)
        ans[i:i+chunk_q] = disc*val.mean(axis=1)
    return ans


def error_metrics(pred: np.ndarray, ref: np.ndarray) -> dict:
    pred = np.asarray(pred); ref = np.asarray(ref)
    e = pred-ref
    denom = max(float(np.linalg.norm(ref)), 1e-12)
    return {
        "mae": float(np.mean(np.abs(e))),
        "rmse": float(np.sqrt(np.mean(e**2))),
        "relative_l2": float(np.linalg.norm(e)/denom),
        "max_abs_error": float(np.max(np.abs(e)))
    }


def benchmark_inference(model: nn.Module, dim: int, low: float, high: float,
                        strike: float = 100.0, n_queries: int = 100_000,
                        seed: int = 999) -> dict:
    rng = np.random.default_rng(seed+dim)
    x = rng.uniform(low, high, size=(n_queries, dim)).astype(np.float32)
    _ = predict_prices(model, x[:1000], strike=strike)
    t0 = time.perf_counter(); _ = predict_prices(model, x, strike=strike); elapsed = time.perf_counter()-t0
    return {"n_queries": n_queries, "elapsed_seconds": elapsed,
            "seconds_per_query": elapsed/n_queries}


def benchmark_mc_queries(dim: int, payoff: str, low: float, high: float,
                         strike: float = 100.0, r: float = 0.05,
                         sigma: float = 0.2, T: float = 1.0,
                         n_queries: int = 1024, n_paths: int = 4096,
                         seed: int = 777, chunk_q: int = 64) -> dict:
    rng = np.random.default_rng(seed+dim)
    x = rng.uniform(low, high, size=(n_queries, dim))
    z = rng.standard_normal((n_paths, dim))
    factors = np.exp((r-0.5*sigma**2)*T + sigma*np.sqrt(T)*z)
    disc = np.exp(-r*T)
    t0 = time.perf_counter()
    for i in range(0, n_queries, chunk_q):
        st = x[i:i+chunk_q, None, :] * factors[None, :, :]
        if payoff == "basket":
            vals = np.maximum(st.mean(axis=2)-strike, 0.0)
        elif payoff == "max":
            vals = np.maximum(st.max(axis=2)-strike, 0.0)
        else:
            vals = np.maximum(st[:, :, 0]-strike, 0.0)
        _ = disc*vals.mean(axis=1)
    elapsed = time.perf_counter()-t0
    return {"n_queries": n_queries, "n_paths": n_paths,
            "elapsed_seconds": elapsed, "seconds_per_query": elapsed/n_queries}
