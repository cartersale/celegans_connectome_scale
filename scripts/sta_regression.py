"""Remove the stimulus-locked component from each trace by subtracting a
stimulus-triggered average, then score the operators on the residual
correlations.

The mean inter-stimulus interval (about 68 volumes, TPL) is shorter than the
stimulated neuron's evoked transient (stimulus_transient.py), so frame
excision cannot leave an unstimulated state. Instead, per recording and per
neuron, the template T[L] = mean over onsets s of X[s+L] - X[s], L < TPL, is
subtracted at every onset; all frames are kept. Where onsets are closer than
TPL, the template is subtracted twice in the overlap.

Arms, each scored by Spearman of S[i,j] + S[j,i] against the mean correlation of
pairs co-identified in >= 3 recordings: the full resolvent (alpha = 0.1), the
lifted class-level resolvent, and N_NULL size-matched random partitions
(two_observables.size_matched, seed 0). Run on the unaltered traces
("baseline (all frames)") and on the STA-regressed traces.

Inputs: C. elegans connectome (conres.celegans), exported Randi 2023 recordings.
Outputs: results/sta_regression.csv (arm, pairs, full, bio, random,
random_sd, bio_frac, random_frac, z). Produces the evoked-response-removal
result in the Supplementary Information.
"""
import sys, re, warnings
from pathlib import Path
import numpy as np
from scipy import stats
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from conres import celegans as ce, randi
from two_observables import size_matched
ALPHA=0.1; N_NULL=200
TPL=68     # template length = mean inter-stimulus interval

def correlations_stareg(nodes, remove=True):
    folder=randi.EXPORTED/"exported_data"; valid={n:k for k,n in enumerate(nodes)}
    n=len(nodes); ssum=np.zeros((n,n)); cnt=np.zeros((n,n))
    for f in sorted(folder.glob("*_labels.txt")):
        i=int(re.match(r"(\d+)_labels",f.name).group(1))
        labels=[ce.normalise_name(x.rstrip("\n").strip()) for x in f.open()]
        G=randi._gcamp(folder,i); sf=folder/f"{i}_stim_volume_i.txt"
        if G is None or G.shape[0]<100: continue
        alive=np.isfinite(G).all(axis=0)
        cols=[(k,valid[x]) for k,x in enumerate(labels) if x in valid and k<G.shape[1] and alive[k]]
        if len(cols)<2: continue
        X=G[:,[c for c,_ in cols]].astype(float).copy()
        if remove and sf.exists():
            stims=[int(x) for x in sf.open() if x.strip()]
            stims=[s for s in stims if 0<=s<X.shape[0]]
            if stims:
                # per-neuron STA template, then subtract it at every onset
                T=np.zeros((TPL,X.shape[1])); W=np.zeros(TPL)
                for s in stims:
                    L=min(TPL,X.shape[0]-s)
                    if L<=0: continue
                    T[:L]+=X[s:s+L]-X[s]; W[:L]+=1
                T/=np.maximum(W,1)[:,None]
                for s in stims:
                    L=min(TPL,X.shape[0]-s)
                    if L>0: X[s:s+L]-=T[:L]
        X=X-X.mean(0); sd=X.std(0); good=(sd>0)&np.isfinite(sd)
        if good.sum()<2: continue
        C=(X[:,good].T@X[:,good])/(X.shape[0]*np.outer(sd[good],sd[good]))
        idx=np.array([j for (_,j),g in zip(cols,good) if g])
        ssum[np.ix_(idx,idx)]+=C; cnt[np.ix_(idx,idx)]+=1.0
    with np.errstate(invalid="ignore"): R=ssum/cnt
    np.fill_diagonal(R,np.nan); np.fill_diagonal(cnt,0.0)
    return R,cnt

A,nodes=ce.build_matrix("all",neurons_only=True)
labels=ce.partitions(nodes)["class"]; lab=np.unique(labels,return_inverse=True)[1]
n=len(nodes); S=randi.resolvent(A,ALPHA); iu,ju=np.triu_indices(n,1)
Q=randi.resolvent(randi.quotient(A,lab),ALPHA)[np.ix_(lab,lab)]

ROWS=[]
def arm(R,cnt,tag):
    keep=(cnt[iu,ju]>=3)&np.isfinite(R[iu,ju]); a,b=iu[keep],ju[keep]; r=R[a,b]
    rho=lambda M: stats.spearmanr(M[a,b]+M[b,a],r).statistic
    full=rho(S); bio=rho(Q); rng=np.random.default_rng(0)
    rnd=np.array([rho(randi.resolvent(randi.quotient(A,rl),ALPHA)[np.ix_(rl,rl)])
                  for rl in (size_matched(lab,rng) for _ in range(N_NULL))])
    ROWS.append(dict(arm=tag.strip(), pairs=int(keep.sum()), full=full, bio=bio,
                     random=rnd.mean(), random_sd=rnd.std(),
                     bio_frac=(full-bio)/full, random_frac=(full-rnd.mean())/full,
                     z=(bio-rnd.mean())/rnd.std()))
    print(f"{tag:>26} {keep.sum():>7,} {full:>7.4f} {bio:>7.4f} {rnd.mean():>7.4f} "
          f"{(full-bio)/full:>+9.3f} {(full-rnd.mean())/full:>+10.3f} {(bio-rnd.mean())/rnd.std():>6.2f}")

print(f"{'arm':>26} {'pairs':>7} {'full':>7} {'bio':>7} {'rand':>7} {'bio frac':>9} {'rand frac':>10} {'z':>6}")
arm(*correlations_stareg(nodes,remove=False),"baseline (all frames)")
arm(*correlations_stareg(nodes,remove=True), "STA-regressed")

import csv
out=Path(__file__).resolve().parents[1] / "results/sta_regression.csv"
with out.open("w", newline="") as fh:
    w=csv.DictWriter(fh, fieldnames=list(ROWS[0])); w.writeheader(); w.writerows(ROWS)
print(f"\nwrote {out}")
