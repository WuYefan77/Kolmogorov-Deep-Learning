from __future__ import annotations
import json, math, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from src.sde import convergence_tables
from src.monte_carlo import black_scholes_call, mc_convergence_experiment, mc_call_price
from src.finite_difference import crank_nicolson_call, explicit_call, fd_error_metrics
from src.neural_solver import (train_neural_feynman_kac, predict_prices, qmc_reference_prices,
                               error_metrics, benchmark_inference, benchmark_mc_queries)

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"; RES = ROOT / "results"
FIG.mkdir(exist_ok=True); RES.mkdir(exist_ok=True)

PARAMS = dict(strike=100.0, r=0.05, sigma=0.2, T=1.0)


def savefig(name):
    plt.tight_layout(); plt.savefig(FIG/name, dpi=180, bbox_inches="tight"); plt.close()


def task1():
    strong, weak = convergence_tables(n_paths=60_000)
    strong.to_csv(RES/"task1_strong_convergence.csv", index=False)
    weak.to_csv(RES/"task1_weak_convergence.csv", index=False)

    plt.figure(figsize=(6,4)); plt.loglog(strong.dt, strong.strong_rmse, "o-", label="Euler-Maruyama strong RMSE")
    anchor = strong.strong_rmse.iloc[-1]*(strong.dt/strong.dt.iloc[-1])**0.5
    plt.loglog(strong.dt, anchor, "--", label=r"reference slope $1/2$")
    plt.xlabel(r"$\Delta t$"); plt.ylabel("RMS strong error"); plt.legend(); plt.grid(True, which="both", alpha=.25)
    savefig("01_strong_convergence.png")

    w2 = weak[weak.moment_power==2]
    plt.figure(figsize=(6,4)); plt.loglog(w2.dt, w2.weak_abs_error, "o-", label=r"weak error for $\phi(S)=S^2$")
    anchor = w2.weak_abs_error.iloc[-1]*(w2.dt/w2.dt.iloc[-1])
    plt.loglog(w2.dt, anchor, "--", label="reference slope 1")
    plt.xlabel(r"$\Delta t$"); plt.ylabel("absolute weak error"); plt.legend(); plt.grid(True, which="both", alpha=.25)
    savefig("02_weak_convergence.png")
    return {"strong_slope": float(strong.fitted_slope.iloc[0]),
            "weak_slope_p1": float(weak[weak.moment_power==1].fitted_slope.iloc[0]),
            "weak_slope_p2": float(weak[weak.moment_power==2].fitted_slope.iloc[0])}


def task2():
    mc = mc_convergence_experiment(reps=60)
    mc.to_csv(RES/"task2_mc_convergence.csv", index=False)
    plt.figure(figsize=(6,4)); plt.loglog(mc.n_paths, mc.rmse, "o-", label="empirical MC RMSE")
    anchor = mc.rmse.iloc[0]*(mc.n_paths/mc.n_paths.iloc[0])**(-0.5)
    plt.loglog(mc.n_paths, anchor, "--", label=r"reference slope $-1/2$")
    plt.xlabel("number of paths N"); plt.ylabel("RMSE"); plt.legend(); plt.grid(True, which="both", alpha=.25)
    savefig("03_mc_convergence.png")
    est, se = mc_call_price(100, **PARAMS, n_paths=100_000, seed=42, antithetic=True)
    truth = float(black_scholes_call(100, **PARAMS))
    return {
        "mc_slope": float(mc.fitted_slope.iloc[0]),
        "bs_truth_at_100": truth,
        "mc_rmse_N100000": float(mc.loc[mc.n_paths == 100_000, "rmse"].iloc[0]),
    }


def task3():
    rows=[]
    last=None
    for ns in [50,100,200,400]:
        S,V=crank_nicolson_call(n_space=ns,n_time=2*ns, **PARAMS)
        met=fd_error_metrics(S,V,**PARAMS)
        rows.append({"n_space":ns,"n_time":2*ns,**met}); last=(S,V)
    fd=pd.DataFrame(rows); fd.to_csv(RES/"task3_fd_convergence.csv",index=False)
    S,V=last
    mask=(S>=40)&(S<=180)
    exact=black_scholes_call(S[mask],**PARAMS)
    plt.figure(figsize=(6,4)); plt.plot(S[mask],exact,label="Black-Scholes closed form"); plt.plot(S[mask],V[mask],"--",label="Crank-Nicolson")
    pts=np.linspace(50,150,16); vals=[]; errs=[]
    for j,x in enumerate(pts):
        e,se=mc_call_price(float(x),**PARAMS,n_paths=50_000,seed=100+j,antithetic=True); vals.append(e); errs.append(1.96*se)
    plt.errorbar(pts,vals,yerr=errs,fmt="o",ms=3,capsize=2,label="MC 95% CI")
    plt.xlabel("initial asset price S"); plt.ylabel("call value V(0,S)"); plt.legend(); plt.grid(True,alpha=.25)
    savefig("04_bs_fd_mc_comparison.png")

    _,_,stable_hist=explicit_call(n_space=100,n_time=2000,**PARAMS)
    _,_,unstable_hist=explicit_call(n_space=100,n_time=50,**PARAMS)
    plt.figure(figsize=(6,4)); plt.semilogy(np.arange(len(stable_hist)),stable_hist,label="stable: 2000 time steps"); plt.semilogy(np.arange(len(unstable_hist)),unstable_hist,label="unstable: 50 time steps")
    plt.xlabel("explicit time step"); plt.ylabel("max |V|"); plt.legend(); plt.grid(True,alpha=.25)
    savefig("05_explicit_stability.png")
    return {
        "fd_finest_mae": rows[-1]["mae"],
        "fd_finest_rmse": rows[-1]["rmse"],
        "fd_finest_max_abs": rows[-1]["max_abs_error"],
    }


def task4():
    tr=train_neural_feynman_kac(dim=1,payoff="call",low=50,high=150,steps=1800,batch_size=2048,
                                hidden=(96,96,96),seed=321,antithetic=True,**PARAMS)
    tr.history.to_csv(RES/"task4_training_history.csv",index=False)
    torch.save(tr.model.state_dict(), RES/"task4_neural_fk_state.pt")
    grid=np.linspace(50,150,401)[:,None]
    pred=predict_prices(tr.model,grid,PARAMS["strike"]); truth=black_scholes_call(grid[:,0],**PARAMS)
    met=error_metrics(pred,truth)
    pd.DataFrame({"S":grid[:,0],"neural":pred,"exact":truth}).to_csv(RES/"task4_solution_curve.csv",index=False)

    rng=np.random.default_rng(77); x=rng.uniform(50,150,800); z=rng.standard_normal(800)
    st=x*np.exp((PARAMS["r"]-.5*PARAMS["sigma"]**2)*PARAMS["T"]+PARAMS["sigma"]*np.sqrt(PARAMS["T"])*z)
    y=np.exp(-PARAMS["r"]*PARAMS["T"])*np.maximum(st-PARAMS["strike"],0)
    plt.figure(figsize=(6,4)); plt.scatter(x,y,s=7,alpha=.2,label="single-path stochastic labels"); plt.plot(grid[:,0],truth,label="conditional mean / exact BS"); plt.plot(grid[:,0],pred,"--",label="neural FK")
    plt.xlabel("initial asset price S"); plt.ylabel("discounted payoff / value"); plt.ylim(-2,110); plt.legend(); plt.grid(True,alpha=.2)
    savefig("06_neural_conditional_expectation.png")

    plt.figure(figsize=(6,4)); plt.semilogy(tr.history.step,tr.history.rmse_price_batch)
    plt.xlabel("SGD step"); plt.ylabel("batch RMSE in price units"); plt.grid(True,alpha=.25)
    savefig("07_neural_training_curve.png")
    return {f"neural_{k}": v for k, v in met.items()}, tr.model


def task5():
    dims=[1,2,5,10,20,50]
    rows=[]; models={}
    for payoff in ["basket","max"]:
        for d in dims:
            # d=1 basket/max are both a call, retained deliberately as a consistency anchor.
            steps=900 if d<=10 else 1100
            tr=train_neural_feynman_kac(dim=d,payoff=payoff,low=80,high=120,steps=steps,batch_size=1536,
                                        hidden=(96,96,96),seed=500+(0 if payoff=="basket" else 1000),
                                        antithetic=True,**PARAMS)
            rng=np.random.default_rng(1000+d+(0 if payoff=="basket" else 5000))
            x=rng.uniform(80,120,size=(64,d))
            ref=qmc_reference_prices(x,payoff=payoff,n_paths=4096,seed=900,**PARAMS)
            pred=predict_prices(tr.model,x,PARAMS["strike"])
            met=error_metrics(pred,ref)
            inf=benchmark_inference(tr.model,d,80,120,PARAMS["strike"],n_queries=50_000)
            mcbench=benchmark_mc_queries(d,payoff,80,120,**PARAMS,n_queries=128,n_paths=4096)
            denom=mcbench["seconds_per_query"]-inf["seconds_per_query"]
            qstar=tr.train_seconds/denom if denom>0 else np.inf
            row={"payoff":payoff,"dimension":d,"train_seconds":tr.train_seconds,
                 "nn_seconds_per_query":inf["seconds_per_query"],
                 "mc4096_seconds_per_query":mcbench["seconds_per_query"],
                 "break_even_queries_est":qstar,**met}
            rows.append(row); models[(payoff,d)]=tr.model
            pd.DataFrame(rows).to_csv(RES/"task5_dimension_scaling_partial.csv",index=False)
            print("high-d",row,flush=True)
    out=pd.DataFrame(rows); out.to_csv(RES/"task5_dimension_scaling.csv",index=False)

    for payoff in ["basket","max"]:
        sub=out[out.payoff==payoff]
        plt.figure(figsize=(6,4)); plt.plot(sub.dimension,sub.relative_l2,"o-")
        plt.xlabel("dimension d"); plt.ylabel("relative L2 error"); plt.title(f"Neural FK: {payoff} payoff"); plt.grid(True,alpha=.25)
        savefig(f"08_error_vs_dimension_{payoff}.png")

    # Amortized cost curve for the 1D basket/call anchor.
    row=out[(out.payoff=="basket")&(out.dimension==1)].iloc[0]
    q=np.unique(np.logspace(1,7,160).astype(int))
    nn=row.train_seconds+q*row.nn_seconds_per_query
    mc=q*row.mc4096_seconds_per_query
    plt.figure(figsize=(6,4)); plt.loglog(q,nn,label="Neural FK: train + inference"); plt.loglog(q,mc,label="MC: 4096 paths/query")
    if np.isfinite(row.break_even_queries_est): plt.axvline(row.break_even_queries_est,ls="--",label=f"estimated break-even {row.break_even_queries_est:,.0f}")
    plt.xlabel("number of queried initial states Q"); plt.ylabel("estimated total seconds"); plt.legend(); plt.grid(True,which="both",alpha=.25)
    savefig("09_amortized_runtime.png")
    return {
        "basket_d50_relative_l2": float(
            out[(out.payoff == "basket") & (out.dimension == 50)].relative_l2.iloc[0]
        ),
        "max_d50_relative_l2": float(
            out[(out.payoff == "max") & (out.dimension == 50)].relative_l2.iloc[0]
        ),
        "basket_1d_break_even_queries_est": float(row.break_even_queries_est),
    }


def main():
    torch.set_num_threads(max(1,min(4,torch.get_num_threads())))
    summary={}
    print("Task 1",flush=True); summary["task1"]=task1()
    print("Task 2",flush=True); summary["task2"]=task2()
    print("Task 3",flush=True); summary["task3"]=task3()
    print("Task 4",flush=True); s4,_=task4(); summary["task4"]=s4
    print("Task 5",flush=True); summary["task5"]=task5()
    (RES/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2),flush=True)

if __name__=="__main__": main()
