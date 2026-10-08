"""Signal-propagation loss along cell-class merge trajectories, in wild type and
unc-31 mutants.

For each strain, observed pairs are occ3 > 0 with that strain's released
responses. Trajectories: cell classes of size > 1 merged one at a time in
random order (bio), or same-size groups drawn from unused neurons (random); 15
trajectories per arm (seed = rep). Loss is AUC(full R) - AUC(class-level
operator) among unsynapsed pairs, alpha = 0.1, averaged per k and reported at
the k nearest each value in a fixed grid, also as a percentage of the strain's
own above-chance AUC.

Inputs: conres.celegans, conres.randi.
Outputs: printed only.
Usage: python scripts/diagnostics/unc31_fragility_curve.py
"""
import sys, warnings
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from conres import celegans as ce, randi
from conres.paths import tee_stdout                   # noqa: E402
tee_stdout(__file__)
ALPHA=0.1
A, nodes = ce.build_matrix("all", neurons_only=True)
labels = ce.partitions(nodes)["class"]; n=len(nodes)
idx={x:i for i,x in enumerate(nodes)}

def data(strain,key):
    occ3,_=randi.build_occ3(nodes,strain)
    d=randi.load_responses(key); d=d[d.pre.isin(idx)&d.post.isin(idx)]
    hits={(idx[p],idx[q]) for p,q in zip(d.pre,d.post)}
    ii,jj=np.nonzero(occ3)
    y=np.array([1.0 if (j,i) in hits else 0.0 for i,j in zip(ii,jj)])
    return jj,ii,y

def traj(jj,ii,y,sub,rng,mode,a_sub):
    classes=[np.flatnonzero(labels==c) for c in np.unique(labels)]
    classes=[c for c in classes if c.size>1]; rng.shuffle(classes)
    cell=np.arange(n); free=list(range(n)); rng.shuffle(free); fs=set(free); out=[]
    for grp in classes:
        if mode=="bio":
            mem=[g for g in grp if g in fs]
            if len(mem)<2: continue
        else:
            take=min(grp.size,len(fs))
            if take<2: continue
            mem=[free.pop() for _ in range(take) if free]
        for m in mem: fs.discard(m)
        cell[mem]=mem[0]
        lab=np.unique(cell,return_inverse=True)[1]
        v=randi.class_leverage(A,lab,ALPHA)[jj,ii]
        out.append((int(np.unique(lab).size), a_sub-roc_auc_score(y[sub],v[sub])))
    return out

for strain,key,tag in (("wt","wildty","WILD TYPE"),("unc31","unc31","unc-31")):
    jj,ii,y=data(strain,key); sub=A[jj,ii]==0
    full=randi.resolvent(A,ALPHA)[jj,ii]; a_sub=roc_auc_score(y[sub],full[sub]); eff=a_sub-0.5
    curves={}
    for mode in ("bio","random"):
        acc={}
        for r in range(15):
            for k,l in traj(jj,ii,y,sub,np.random.Generator(np.random.PCG64(r)),mode,a_sub):
                acc.setdefault(k,[]).append(l)
        curves[mode]={k:np.mean(v) for k,v in acc.items()}
    print(f"\n{tag}   full-operator effect above chance {eff:+.4f}")
    print(f"  {'k':>5} {'%merged':>8} {'bio loss':>10} {'random':>10} {'bio %eff':>9} {'rand %eff':>10}")
    for k in (290,275,260,242,230,220,200,180,150,119):
        b=min(curves['bio'],key=lambda x:abs(x-k)); r=min(curves['random'],key=lambda x:abs(x-k))
        lb,lr=curves['bio'][b],curves['random'][r]
        print(f"  {k:>5} {100*(n-k)/n:>7.1f}% {lb:>+10.4f} {lr:>+10.4f} {100*lb/eff:>+8.0f}% {100*lr/eff:>+9.0f}%")
