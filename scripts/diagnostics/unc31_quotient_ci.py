"""Cell-class (k = 119) signal-propagation loss among unsynapsed pairs, with a
cluster-bootstrap interval, in wild type and unc-31 mutants.

Loss is AUC(full R) - AUC(class-level operator), alpha = 0.1, over unsynapsed
observed pairs. 2,000 resamples (seed 0) of the stimulated neuron's cell class,
restricted to unsynapsed pairs; resamples with a single outcome class are
skipped. Also reported: loss as a percentage of the strain's above-chance AUC
and P(loss > 0) over resamples.

Inputs: conres.celegans, conres.randi.
Outputs: printed only.
Usage: python scripts/diagnostics/unc31_quotient_ci.py
"""
import sys, warnings
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from conres import celegans as ce, randi
from conres.paths import tee_stdout                   # noqa: E402
tee_stdout(__file__)
ALPHA=0.1
A,nodes=ce.build_matrix("all",neurons_only=True)
labels=ce.partitions(nodes)["class"]; lab=np.unique(labels,return_inverse=True)[1]
idx={x:i for i,x in enumerate(nodes)}
full_op=randi.resolvent(A,ALPHA); quo=randi.class_leverage(A,lab,ALPHA)

def boot(strain,key,reps=2000,seed=0):
    occ3,_=randi.build_occ3(nodes,strain)
    d=randi.load_responses(key); d=d[d.pre.isin(idx)&d.post.isin(idx)]
    hits={(idx[p],idx[q]) for p,q in zip(d.pre,d.post)}
    ii,jj=np.nonzero(occ3)
    y=np.array([1.0 if (j,i) in hits else 0.0 for i,j in zip(ii,jj)])
    sub=A[jj,ii]==0
    f=full_op[jj,ii]; q=quo[jj,ii]
    # cluster bootstrap over classes of the stimulated neuron
    cls=labels[jj]; uq=np.unique(cls[sub]); rng=np.random.default_rng(seed)
    cmap={c:np.flatnonzero(sub&(cls==c)) for c in uq}
    obs_full=roc_auc_score(y[sub],f[sub]); obs=obs_full-roc_auc_score(y[sub],q[sub])
    out=[]
    for _ in range(reps):
        pick=rng.choice(uq,uq.size,replace=True)
        ix=np.concatenate([cmap[c] for c in pick])
        yy=y[ix]
        if yy.sum()==0 or yy.sum()==len(yy): continue
        out.append(roc_auc_score(yy,f[ix])-roc_auc_score(yy,q[ix]))
    out=np.array(out)
    return obs, obs_full-0.5, np.percentile(out,[2.5,97.5]), out

for strain,key,tag in (("wt","wildty","WILD TYPE"),("unc31","unc31","unc-31")):
    o,eff,ci,dist=boot(strain,key)
    print(f"{tag:10s} k=119 bio loss {o:+.4f}  95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]"
          f"   effect {eff:+.4f}   loss as %eff {100*o/eff:+.0f}%"
          f"   CI as %eff [{100*ci[0]/eff:+.0f}%, {100*ci[1]/eff:+.0f}%]")
    print(f"{'':10s}       P(loss>0) = {(dist>0).mean():.3f}, n_boot={len(dist)}")
