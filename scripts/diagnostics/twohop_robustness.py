"""Cell-class sufficiency for the two-step path count in place of the path operator.

On wild-type observed pairs (occ3 > 0, Randi 2023 responses, Cook 2019
connectome), compares the path operator R (alpha = 0.1) and the binary
two-step path count B @ B at full resolution and on the cell-class quotient
(k = 119). The quotient two-step predictor is Q_B @ Q_B with
Q_B = randi.quotient(B, classes), read back at each pair's classes. Loss is the
AUC shortfall among unsynapsed pairs; z against 30 size-matched random
partitions (seed 0).

Inputs: conres.celegans, conres.randi.
Outputs: printed only.
Usage: python scripts/diagnostics/twohop_robustness.py
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

ALPHA = 0.1
A, nodes = ce.build_matrix("all", neurons_only=True)
labels = ce.partitions(nodes)["class"]
occ3, _ = randi.build_occ3(nodes, "wt")
idx = {n: i for i, n in enumerate(nodes)}
d = randi.load_responses("wildty"); d = d[d.pre.isin(idx) & d.post.isin(idx)]
hits = {(idx[p], idx[q]) for p, q in zip(d.pre, d.post)}
ii, jj = np.nonzero(occ3)
y = np.array([1.0 if (j, i) in hits else 0.0 for i, j in zip(ii, jj)])
direct = A[jj, ii]; sub = direct == 0
B = (A > 0).astype(float)

def two_hop_full():
    return (B @ B)[jj, ii]

def two_hop_quotient(lab):
    """Two-hop count aggregated on the quotient, read back at each pair's cells."""
    QB = randi.quotient(B, lab)
    return (QB @ QB)[np.ix_(lab, lab)][jj, ii]

def resolv_quotient(lab):
    return randi.class_leverage(A, lab, ALPHA)[jj, ii]

# ---- full-resolution baselines
f_res = randi.resolvent(A, ALPHA)[jj, ii]
f_two = two_hop_full()
print("FULL RESOLUTION           all pairs   no-synapse")
print(f"  resolvent               {roc_auc_score(y, f_res):.4f}      {roc_auc_score(y[sub], f_res[sub]):.4f}")
print(f"  two-hop count           {roc_auc_score(y, f_two):.4f}      {roc_auc_score(y[sub], f_two[sub]):.4f}")

# ---- class partition (k=119)
lab = np.unique(labels, return_inverse=True)[1]
q_res, q_two = resolv_quotient(lab), two_hop_quotient(lab)
print("\nCLASS QUOTIENT (k=119, 2.54x)")
print(f"  resolvent  quotient     {roc_auc_score(y, q_res):.4f}      {roc_auc_score(y[sub], q_res[sub]):.4f}")
print(f"  two-hop    quotient     {roc_auc_score(y, q_two):.4f}      {roc_auc_score(y[sub], q_two[sub]):.4f}")
print(f"\n  LOSS (full - quotient), no-synapse subgroup")
l_res = roc_auc_score(y[sub], f_res[sub]) - roc_auc_score(y[sub], q_res[sub])
l_two = roc_auc_score(y[sub], f_two[sub]) - roc_auc_score(y[sub], q_two[sub])
print(f"    resolvent  {l_res:+.4f}")
print(f"    two-hop    {l_two:+.4f}")

# ---- size-matched random partitions at k=119, same size profile
sizes = np.bincount(lab)
rng = np.random.default_rng(0)
lr, lt = [], []
for r in range(30):
    perm = rng.permutation(len(nodes))
    rlab = np.zeros(len(nodes), int); pos = 0
    for c, s in enumerate(sizes):
        rlab[perm[pos:pos+s]] = c; pos += s
    lr.append(roc_auc_score(y[sub], f_res[sub]) - roc_auc_score(y[sub], resolv_quotient(rlab)[sub]))
    lt.append(roc_auc_score(y[sub], f_two[sub]) - roc_auc_score(y[sub], two_hop_quotient(rlab)[sub]))
print(f"\n  SIZE-MATCHED RANDOM at k=119 (30 reps, mean +/- sd)")
print(f"    resolvent  {np.mean(lr):+.4f} +/- {np.std(lr):.4f}   z(bio) = {(l_res-np.mean(lr))/np.std(lr):+.2f}")
print(f"    two-hop    {np.mean(lt):+.4f} +/- {np.std(lt):.4f}   z(bio) = {(l_two-np.mean(lt))/np.std(lt):+.2f}")
