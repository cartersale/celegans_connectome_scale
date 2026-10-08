"""Stability of the label-free merge selection across the eight Witvliet 2021
developmental connectomes.

On the neurons reconstructed in all eight animals (ce.witvliet_common_neurons),
each animal's MERGES = 35 disjoint merges are selected greedily by the
symmetrized pairwise equitability defect (prospective_merges) from a
shared candidate pool (cross_animal_transfer.pool_mask). Reported, raw
and after multinomial thinning of each graph to 1,200 synapses (seed = animal
index): mean pairwise Jaccard of merge sets and ARI of the induced partitions
across animals, each against 200 pairs of random disjoint selections from the
pool; ARI of each selection with the cell classes; same-class merge rate
against the pool's chance rate.

Inputs: conres.celegans Witvliet loaders, prospective_merges,
    cross_animal_transfer.
Outputs: printed only.
Usage: python scripts/diagnostics/dev_stability.py
"""
import sys, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from conres import celegans as ce
from conres.paths import tee_stdout                   # noqa: E402
tee_stdout(__file__)
from prospective_merges import pairwise_defect, greedy_disjoint
from cross_animal_transfer import pool_mask
from sklearn.metrics import adjusted_rand_score

nodes=ce.witvliet_common_neurons("all"); labels=ce.partitions(nodes)["class"]
lab_bio=np.unique(labels,return_inverse=True)[1]; n=len(nodes)

# One candidate pool (the pooled-graph degree floor of cross_animal_transfer)
# and one merge count for every animal: Jaccard between selections is only a
# stability measure when every selection has the same size. MERGES is below
# what the pool can supply.
POOL=pool_mask([ce.witvliet_matrix(p,nodes,"all") for _,_,p in ce.witvliet_files()])
MERGES=35

def select(A,seed=0):
    g=(pairwise_defect(A)+pairwise_defect(A.T))/2.0
    return greedy_disjoint(g,MERGES,POOL)

def partition_from(sel):
    c=np.arange(n)
    for a,b in sel: c[b]=c[a]
    return np.unique(c,return_inverse=True)[1]

def thin(A,budget,rng):
    w=A.ravel(); t=w.sum()
    return A.copy() if t<=budget else rng.multinomial(budget,w/t).reshape(A.shape).astype(float)

files=ce.witvliet_files()
for tag,budget in (("RAW",None),("THINNED@1200",1200)):
    sels=[]
    for i,stage,p in files:
        A=ce.witvliet_matrix(p,nodes,"all")
        if budget: A=thin(A,budget,np.random.default_rng(i))
        sels.append(select(A))
    S=[{frozenset(x) for x in s} for s in sels]
    J=[len(S[a]&S[b])/len(S[a]|S[b]) for a in range(8) for b in range(a+1,8)]
    P=[partition_from(s) for s in sels]
    ARI=[adjusted_rand_score(P[a],P[b]) for a in range(8) for b in range(a+1,8)]
    ARIbio=[adjusted_rand_score(p_,lab_bio) for p_ in P]
    # Chance baseline: random disjoint pairs drawn from the same pool as the
    # selections, so chance overlap is on the same footing.
    rng=np.random.default_rng(0); Jc=[]; Ac=[]; poolidx=np.flatnonzero(POOL)
    for _ in range(200):
        rs=[]
        for _ in range(2):
            perm=rng.permutation(poolidx)
            rs.append([(int(perm[2*k]),int(perm[2*k+1])) for k in range(MERGES)])
        A_,B_={frozenset(x) for x in rs[0]},{frozenset(x) for x in rs[1]}
        Jc.append(len(A_&B_)/len(A_|B_))
        Ac.append(adjusted_rand_score(partition_from(rs[0]),partition_from(rs[1])))
    print(f"\n{tag}")
    print(f"  merge-set Jaccard across animals   {np.mean(J):.3f}  (chance {np.mean(Jc):.4f}"
          f" -> {np.mean(J)/max(np.mean(Jc),1e-9):.0f}x)")
    print(f"  partition ARI across animals       {np.mean(ARI):.3f}  (chance {np.mean(Ac):.4f})")
    print(f"  ARI of each selection vs biology   {np.mean(ARIbio):.3f}"
          f"  (range {np.min(ARIbio):.3f}-{np.max(ARIbio):.3f})")
    sc=[np.mean([lab_bio[a]==lab_bio[b] for a,b in s]) for s in sels]
    cs=np.bincount(np.unique(lab_bio[POOL],return_inverse=True)[1])
    ch=(cs*(cs-1)//2).sum()/(POOL.sum()*(POOL.sum()-1)/2)
    print(f"  same-class rate per animal         {np.mean(sc)*100:.1f}% "
          f"(range {np.min(sc)*100:.1f}-{np.max(sc)*100:.1f}%), sd {np.std(sc)*100:.1f}pp"
          f"  [pool chance {ch*100:.1f}%, {np.mean(sc)/ch:.0f}x]")
    print(f"  merges per animal                  {[len(s) for s in sels]}")
