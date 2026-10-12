"""Sham excision: remove windows of the same length as stimulus excision at
random onsets, to separate removing evoked periods from removing frames.

For each recording, as many onsets as it has stimulations are drawn uniformly
and volumes s-2 .. s+WIN-1 (WIN = 40) are masked, the same window as
conres.randi.build_correlations(excise=40); overlapping windows mean the number
of masked frames is matched only approximately. Correlations are otherwise
computed as in build_correlations. For three sham seeds and for true excision,
the full resolvent (alpha = 0.1), the lifted class-level resolvent and N_NULL
size-matched random partitions (two_observables.size_matched, seed 0) are
scored by Spearman against the correlations of pairs co-identified in >= 3
recordings, with loss as a fraction of the full operator's rho.

Inputs: C. elegans connectome (conres.celegans), exported Randi 2023 recordings.
Outputs: stdout only (saved as results/sham_excision.log for the
stimulus-excision table in the Supplementary Information).
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
ALPHA=0.1; WIN=40; N_NULL=200

def correlations_sham(nodes, win, seed):
    folder=randi.EXPORTED/"exported_data"; valid={n:k for k,n in enumerate(nodes)}
    n=len(nodes); ssum=np.zeros((n,n)); cnt=np.zeros((n,n)); rng=np.random.default_rng(seed)
    for f in sorted(folder.glob("*_labels.txt")):
        i=int(re.match(r"(\d+)_labels",f.name).group(1))
        labels=[ce.normalise_name(x.rstrip("\n").strip()) for x in f.open()]
        G=randi._gcamp(folder,i)
        if G is None or G.shape[0]<100: continue
        sf=folder/f"{i}_stim_volume_i.txt"
        nstim=len([x for x in sf.open() if x.strip()]) if sf.exists() else 0
        alive=np.isfinite(G).all(axis=0)
        cols=[(k,valid[x]) for k,x in enumerate(labels) if x in valid and k<G.shape[1] and alive[k]]
        if len(cols)<2: continue
        mask=np.ones(G.shape[0],bool)
        for s in rng.integers(0,max(G.shape[0]-1,1),size=nstim):   # random onsets
            mask[max(0,s-2):s+win]=False
        X=G[np.ix_(mask,[c for c,_ in cols])]
        if X.shape[0]<100: continue
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
n=len(nodes); S=randi.resolvent(A,ALPHA)
iu,ju=np.triu_indices(n,1)

def arm(R,cnt):
    keep=(cnt[iu,ju]>=3)&np.isfinite(R[iu,ju]); a,b=iu[keep],ju[keep]; r=R[a,b]
    def rho(M): return stats.spearmanr(M[a,b]+M[b,a],r).statistic
    full=rho(S); bio=rho(randi.resolvent(randi.quotient(A,lab),ALPHA)[np.ix_(lab,lab)])
    rng=np.random.default_rng(0)
    rnd=[rho(randi.resolvent(randi.quotient(A,rl),ALPHA)[np.ix_(rl,rl)])
         for rl in (size_matched(lab,rng) for _ in range(N_NULL))]
    rnd=np.array(rnd); head=full  # headroom above chance (rho=0)
    return dict(npair=keep.sum(),full=full,bio=bio,rnd=rnd.mean(),
                bfrac=(full-bio)/head,rfrac=(full-rnd.mean())/head,
                z=(bio-rnd.mean())/rnd.std())

print(f"sham excision, window {WIN}+2 volumes at random onsets (3 seeds)\n")
print(f"{'arm':>22} {'pairs':>7} {'full':>7} {'bio':>7} {'rand':>7} {'bio frac':>9} {'rand frac':>10} {'z':>6}")
for sd in (0,1,2):
    R,c=correlations_sham(nodes,WIN,sd); a=arm(R,c)
    print(f"{'sham seed '+str(sd):>22} {a['npair']:>7,} {a['full']:>7.4f} {a['bio']:>7.4f} "
          f"{a['rnd']:>7.4f} {a['bfrac']:>+9.3f} {a['rfrac']:>+10.3f} {a['z']:>6.2f}")
R,c=randi.build_correlations(nodes,"wt",excise=WIN)
a=arm(R,c)
print(f"{'TRUE excise=40':>22} {a['npair']:>7,} {a['full']:>7.4f} {a['bio']:>7.4f} "
      f"{a['rnd']:>7.4f} {a['bfrac']:>+9.3f} {a['rfrac']:>+10.3f} {a['z']:>6.2f}")
