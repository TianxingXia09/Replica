"""Deterministic reconstruction validation, not a recovery of missing outputs."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import t
from local_protocol import (probability_table, random_density, sample_pair_with_setting,
    local_overlap_kernel, subsystem_purities, all_subsystems)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'rebuild_results'

def ci(x):
    x = np.asarray(x)
    m = x.mean()
    h = t.ppf(.975, len(x)-1)*x.std(ddof=1)/np.sqrt(len(x))
    return [float(m-h), float(m+h)]

def feedback():
    n, N, reps = 3, 1000, 120
    rng = np.random.default_rng(2026102601)
    bank = [random_density(8, rng, rank=r) for r in [1,2,8]]
    sets = [tuple(range(k)) for k in [1,2,3]]
    tables = [probability_table(r,n) for r in bank]
    ps = np.array([subsystem_purities(r,sets,n) for r in bank])
    rows = []
    for rep in range(reps):
        rng = np.random.default_rng(2026102602+rep)
        label = 0
        total, target = np.zeros(3), np.zeros(3)
        for j in range(N):
            s,a,b = sample_pair_with_setting(tables[label],tables[label],n,rng)
            total += local_overlap_kernel(a,b,sets)
            target += ps[label]
            # The actual observed outcomes drive the next source state.
            label = int((label+int(a.sum())+2*int(b.sum())+j%2)%3)
        for k,e in enumerate((total-target)/N,1):
            rows.append(dict(rep=rep,size=k,error=e,rounds=N))
    df = pd.DataFrame(rows)
    df.to_csv(OUT/'feedback.csv',index=False)
    return {str(k):dict(bias=float(g.error.mean()),ci=ci(g.error))
            for k,g in df.groupby('size')}

def multibank():
    rows=[]
    for n in [3,4]:
        sets=all_subsystems(n,3)
        sizes=np.array([len(s) for s in sets])
        for bank_id in range(8):
            rng=np.random.default_rng(2026102700+100*n+bank_id)
            states=[random_density(2**n,rng,rank=r) for r in [1,2,2**n]]
            tables=[probability_table(r,n) for r in states]
            ps=np.array([subsystem_purities(r,sets,n) for r in states])
            for regime in ['stationary','alternating','switch']:
                for rep in range(12):
                    rng=np.random.default_rng(2026102800+10000*n+1000*bank_id+100*['stationary','alternating','switch'].index(regime)+rep)
                    labels=np.zeros(600,dtype=int) if regime=='stationary' else (np.arange(600)%2 if regime=='alternating' else rng.integers(0,3,600))
                    estimate=np.zeros(len(sets))
                    for label in labels:
                        _,a,b=sample_pair_with_setting(tables[label],tables[label],n,rng)
                        estimate+=local_overlap_kernel(a,b,sets)
                    error=estimate/600-ps[labels].mean(axis=0)
                    for k in [1,2,3]:
                        rows.append(dict(n=n,bank=bank_id,regime=regime,rep=rep,size=k,error=float(error[sizes==k].mean())))
    df=pd.DataFrame(rows)
    df.to_csv(OUT/'multibank.csv',index=False)
    rng=np.random.default_rng(2026102901)
    result=[]
    for key,g in df.groupby(['n','regime','size']):
        arrays=[g[g.bank==b].error.to_numpy() for b in range(8)]
        boot=[]
        for _ in range(3000):
            bs=rng.integers(0,8,8)
            boot.append(np.mean([rng.choice(arrays[b],12,replace=True).mean() for b in bs]))
        lo,hi=np.quantile(boot,[.025,.975])
        result.append(dict(n=key[0],regime=key[1],size=key[2],bias=g.error.mean(),ci_low=lo,ci_high=hi))
    out=pd.DataFrame(result)
    out.to_csv(OUT/'multibank_bootstrap.csv',index=False)
    return dict(cells=len(out),cover_zero=int(((out.ci_low<=0)&(out.ci_high>=0)).sum()),max_abs_bias=float(out.bias.abs().max()))

def statistics():
    raw=pd.read_csv(ROOT/'phase2_results/raw.csv')
    raw=raw[raw.rep>=20]
    keys=['n_qubits','paired_rounds','regime','size','rep']
    mse=raw.groupby(keys+['method']).sq_error.mean().unstack('method')
    methods=[x for x in mse.columns if x.startswith('shadow_u_')]
    rng=np.random.default_rng(2026103001)
    rows=[]
    for key,g in mse.groupby(level=keys[:-1]):
        for method in methods:
            d=(g[method]-g.synchronized_local_pauli).to_numpy()
            # Two-sided paired sign-flip test on repetition-averaged MSE.
            signs=rng.choice([-1,1],size=(19999,len(d)))
            p=(1+np.sum(np.abs((signs*d).mean(axis=1))>=abs(d.mean())))/20000
            rows.append(dict(zip(keys[:-1],key),method=method,advantage=d.mean(),p=p))
    df=pd.DataFrame(rows)
    order=np.argsort(df.p.to_numpy())
    q=np.minimum.accumulate((df.p.to_numpy()[order]*len(df)/np.arange(1,len(df)+1))[::-1])[::-1]
    df.loc[order,'q']=np.minimum(q,1)
    df.to_csv(OUT/'confirmatory_all.csv',index=False)
    selected=pd.read_csv(ROOT/'phase3_results/boundary_selected_windows.csv')
    selected=selected.merge(df,on=keys[:-1]+['method'],validate='one_to_one')
    selected['decision']=np.where(selected.q>=.05,'unresolved',np.where(selected.advantage>0,'synchronized','temporal'))
    selected.to_csv(OUT/'confirmatory_selected.csv',index=False)
    return dict(tests=len(df),selected=len(selected),decisions=selected.decision.value_counts().to_dict())

def scaling():
    rng=np.random.default_rng(2026103101)
    rows=[]
    for k in range(1,7):
        # Computational product-pure states: factor 2 with probability 2/3,
        # and -1 with probability 1/3 in shared random Pauli measurements.
        factors=np.where(rng.random((1000,2000,k))<2/3,2.,-1.)
        errors=factors.prod(axis=2).mean(axis=1)-1
        exact=np.sqrt((3**k-1)/2000)
        rows.append(dict(size=k,exact_rmse=exact,empirical_rmse=np.sqrt(np.mean(errors**2))))
    df=pd.DataFrame(rows)
    df.to_csv(OUT/'scaling.csv',index=False)
    return dict(max_relative_rmse_error=float((df.empirical_rmse/df.exact_rmse-1).abs().max()))

if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    result={}
    for name,fn in [('feedback',feedback),('multibank',multibank),('statistics',statistics),('scaling',scaling)]:
        result[name]=fn()
        (OUT/'summary.json').write_text(json.dumps(result,indent=2))
        print(name,json.dumps(result[name]),flush=True)
