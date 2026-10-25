from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import numpy as np
import pandas as pd


ROOT=Path(__file__).resolve().parent
DIMS=[2,4,8,16]
N=20000
REPS=800
WINDOWS=[2,4,8,16,64]


def labels_for(regime,d,rng):
    if regime=="stationary_pure":return np.zeros(N,dtype=int)
    if regime=="iid_switch":return rng.integers(0,d,N)
    if regime=="burst_4":return np.repeat(rng.integers(0,d,(N+3)//4),4)[:N]
    if regime=="burst_16":return np.repeat(rng.integers(0,d,(N+15)//16),16)[:N]
    raise ValueError(regime)


def collision_sample(overlap,d,rng,size):
    p=(overlap+1)/(d+1)
    return (d+1)*(rng.random(size)<p)-1


def estimate_sync(labels,d,rng):
    return float(collision_sample(np.ones(N),d,rng,N).mean())


def estimate_pooled(labels,d,rng):
    i=rng.integers(0,N,N);j=rng.integers(0,N-1,N);j+=j>=i
    overlap=(labels[i]==labels[j]).astype(float)
    return float(collision_sample(overlap,d,rng,N).mean())


def estimate_window(labels,d,w,rng):
    starts=rng.integers(0,N-w+1,N);i=starts+rng.integers(0,w,N)
    offset=rng.integers(0,w-1,N);local=i-starts;jlocal=offset+(offset>=local);j=starts+jlocal
    overlap=(labels[i]==labels[j]).astype(float)
    return float(collision_sample(overlap,d,rng,N).mean())


def run_one(task):
    d,regime,rep,seed=task;rng=np.random.default_rng(seed)
    if regime=="maximally_mixed":
        truth=1/d
        # Two-copy collision probability for I/d is 1/d after inversion.
        estimates={"synchronized":float(collision_sample(np.full(N,1/d),d,rng,N).mean())}
        estimates["pooled"]=float(collision_sample(np.full(N,1/d),d,rng,N).mean())
        for w in WINDOWS:estimates[f"window_{w}"]=float(collision_sample(np.full(N,1/d),d,rng,N).mean())
    else:
        labels=labels_for(regime,d,rng);truth=1.0
        estimates={"synchronized":estimate_sync(labels,d,rng),"pooled":estimate_pooled(labels,d,rng)}
        for w in WINDOWS:estimates[f"window_{w}"]=estimate_window(labels,d,w,rng)
    return [{"dimension":d,"regime":regime,"rep":rep,"method":m,"truth":truth,
             "estimate":e,"sq_error":(e-truth)**2,"abs_error":abs(e-truth)} for m,e in estimates.items()]


def main():
    regimes=["stationary_pure","maximally_mixed","iid_switch","burst_4","burst_16"]
    master=np.random.default_rng(20261022);tasks=[]
    for d in DIMS:
        for regime in regimes:
            for rep in range(REPS):tasks.append((d,regime,rep,int(master.integers(2**31-1))))
    rows=[]
    with ProcessPoolExecutor(max_workers=8) as ex:
        for chunk in ex.map(run_one,tasks,chunksize=8):rows.extend(chunk)
    out=ROOT/"results";out.mkdir(exist_ok=True)
    df=pd.DataFrame(rows);df.to_csv(out/"raw.csv",index=False)
    agg=df.groupby(["dimension","regime","method"],as_index=False).agg(
        rmse=("sq_error",lambda x:np.sqrt(np.mean(x))),mae=("abs_error","mean"),bias=("estimate",lambda x:np.mean(x)-df.loc[x.index,"truth"].mean()))
    agg.to_csv(out/"aggregate.csv",index=False)
    hard=agg[agg.regime.isin(["iid_switch","burst_4","burst_16"])]
    comparisons=[]
    for (d,regime),g in hard.groupby(["dimension","regime"]):
        sync=float(g[g.method=="synchronized"].rmse.iloc[0])
        others=g[g.method!="synchronized"];best=others.loc[others.rmse.idxmin()]
        comparisons.append({"dimension":d,"regime":regime,"sync_rmse":sync,
                            "best_baseline":best.method,"baseline_rmse":best.rmse,
                            "relative_improvement":1-sync/best.rmse})
    comp=pd.DataFrame(comparisons);comp.to_csv(out/"comparisons.csv",index=False)
    easy=agg[agg.regime.isin(["stationary_pure","maximally_mixed"])]
    easy_regret=[]
    for (d,regime),g in easy.groupby(["dimension","regime"]):
        sync=float(g[g.method=="synchronized"].rmse.iloc[0]);best=float(g.rmse.min())
        easy_regret.append(sync/best-1)
    decision={"decision":"GO" if (comp.relative_improvement>=.20).all() and max(easy_regret)<=.15 else "NO_GO",
              "hard_regimes_all_over_20pct":bool((comp.relative_improvement>=.20).all()),
              "minimum_hard_improvement":float(comp.relative_improvement.min()),
              "maximum_easy_regret":float(max(easy_regret))}
    (out/"decision.json").write_text(json.dumps(decision,indent=2))
    print(comp.to_string(index=False));print(json.dumps(decision,indent=2))


if __name__=="__main__":main()

