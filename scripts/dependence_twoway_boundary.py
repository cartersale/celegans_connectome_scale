"""Loss of operator-defined partitions at k = 12, 10, 9, ..., 3 with two-way
bootstrap intervals.

Methods: loss-guided agglomeration (envelope partitions from
results/coarse_trajectory_paths.npz), Ward and k-means (one initialization,
n_init=10, random_state=0) on operator profiles log1p(1e3 [R, R^T] / max R),
with R the path operator at alpha = 0.1. Loss = (full - quotient) / full, the
class-level path operator's shortfall as a fraction of the full operator's
performance, for signal propagation (all observed pairs) and activity
correlations. Intervals: 500 two-way bootstrap resamples (stimulated- and
responding-neuron cell classes resampled independently, pairs weighted by the
product of multiplicities; for correlations one class resample weights both
neurons), seed 0, with the same resamples for full and quotient scores.

Produces the Supplementary Information table on loss of operator-defined
partitions under two-way resampling.

Inputs: conres.observables.load, results/coarse_trajectory_paths.npz,
    complexity_curve.relabel.
Outputs: results/dependence_twoway_boundary.csv (method, k, loss and interval
    per observable).
Usage: run from the repository root (paths are relative):
    python scripts/dependence_twoway_boundary.py
"""
import sys; sys.path[:0]=['src','scripts']
import numpy as np, pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import KMeans
from conres import observables as ob, randi
from complexity_curve import relabel
D=ob.load(); A,cls,obs=D['A'],D['cls'],D['obs']; R=randi.resolvent(A,0.1)
g=np.load('results/coarse_trajectory_paths.npz')
Xop=np.log1p(np.hstack([R,R.T])/R.max()*1e3); tree=linkage(Xop,'ward')
rng=np.random.default_rng(0); ncls=cls.max()+1
def mult(pool):
    c=np.zeros(ncls); np.add.at(c,rng.choice(pool,pool.size),1); return c
O={o:obs[o] for o in ('incidence_all','correlation')}
src=np.unique(cls[O['incidence_all'][0]]); tgt=np.unique(cls[O['incidence_all'][1]])
W={o:[] for o in O}
for _ in range(500):
    cs,ct,cn=mult(src),mult(tgt),mult(np.arange(ncls))
    j,i,_,_=O['incidence_all']; W['incidence_all'].append(np.repeat(np.arange(j.size),(cs[cls[j]]*ct[cls[i]]).astype(int)))
    a,b,_,_=O['correlation']; W['correlation'].append(np.repeat(np.arange(a.size),(cn[cls[a]]*cn[cls[b]]).astype(int)))
fb={o:np.array([ob.score(kd,ob.predict(kd,R,j,i),v,s) for s in W[o]]) for o,(j,i,v,kd) in O.items()}
rows=[]
for method in ('loss-guided','Ward on operator','k-means on operator'):
    for k in (12,10,9,8,7,6,5,4,3):
        lab={'loss-guided':lambda:relabel(g[f'envelope_{k}']),'Ward on operator':lambda:relabel(fcluster(tree,t=k,criterion='maxclust')),
             'k-means on operator':lambda:relabel(KMeans(k,n_init=10,random_state=0).fit(Xop).labels_)}[method]()
        Q=randi.class_leverage(A,lab,0.1); r={'method':method,'k':k}
        for o,(j,i,v,kd) in O.items():
            p=ob.predict(kd,Q,j,i); full=ob.score(kd,ob.predict(kd,R,j,i),v)
            lb=np.array([(f-ob.score(kd,p,v,s))/f for f,s in zip(fb[o],W[o])])
            r[o]=(full-ob.score(kd,p,v))/full; r[o+'_lo'],r[o+'_hi']=np.percentile(lb,[2.5,97.5])
        rows.append(r); print(method,k,' '.join(f"{o} {r[o]:+.2f} [{r[o+'_lo']:+.2f},{r[o+'_hi']:+.2f}]{'*' if r[o+'_lo']>0 else ''}" for o in O),flush=True)
pd.DataFrame(rows).to_csv('results/dependence_twoway_boundary.csv',index=False)
