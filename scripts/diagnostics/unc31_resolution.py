"""Cell-class (k = 119) signal-propagation loss against size-matched random
partitions, in wild type and unc-31 mutants.

Per strain: AUC of the full path operator R (alpha = 0.1) over all and
unsynapsed observed pairs; loss AUC(R) - AUC(class-level operator) among
unsynapsed pairs for the cell classes and for 50 size-matched random partitions
(seed 0), with z for the cell classes. Losses are also given as a percentage of
the strain's own above-chance AUC, since the two strains have different
effects to lose. Only k = 119 is scored; unc31_fragility_curve.py scores the
merge trajectories.

Inputs: conres.celegans, conres.randi.
Outputs: printed only.
Usage: python scripts/diagnostics/unc31_resolution.py
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
lab_bio = np.unique(labels, return_inverse=True)[1]
idx = {n: i for i, n in enumerate(nodes)}

def strain_data(strain, resp_key):
    occ3, _ = randi.build_occ3(nodes, strain)
    d = randi.load_responses(resp_key)
    d = d[d.pre.isin(idx) & d.post.isin(idx)]
    hits = {(idx[p], idx[q]) for p, q in zip(d.pre, d.post)}
    ii, jj = np.nonzero(occ3)
    y = np.array([1.0 if (j, i) in hits else 0.0 for i, j in zip(ii, jj)])
    return jj, ii, y

def report(strain, resp_key, tag):
    jj, ii, y = strain_data(strain, resp_key)
    direct = A[jj, ii]; sub = direct == 0
    full = randi.resolvent(A, ALPHA)[jj, ii]
    a_all, a_sub = roc_auc_score(y, full), roc_auc_score(y[sub], full[sub])
    eff_sub = a_sub - 0.5
    q = randi.class_leverage(A, lab_bio, ALPHA)[jj, ii]
    q_sub = roc_auc_score(y[sub], q[sub])
    loss_bio = a_sub - q_sub

    sizes = np.bincount(lab_bio); rng = np.random.default_rng(0); L = []
    for _ in range(50):
        perm = rng.permutation(len(nodes)); rl = np.zeros(len(nodes), int); pos = 0
        for c, s in enumerate(sizes):
            rl[perm[pos:pos+s]] = c; pos += s
        L.append(a_sub - roc_auc_score(y[sub], randi.class_leverage(A, rl, ALPHA)[sub.nonzero()[0]*0 + np.arange(len(y))[sub]] if False else randi.class_leverage(A, rl, ALPHA)[jj, ii][sub]))
    L = np.array(L)
    z = (loss_bio - L.mean()) / L.std()

    print(f"\n{tag}")
    print(f"  observed pairs {len(y):,}  responses {int(y.sum()):,} ({y.mean()*100:.1f}%)"
          f"   no-synapse {int(sub.sum()):,} ({int(y[sub].sum()):,})")
    print(f"  full operator      all {a_all:.4f}   no-synapse {a_sub:.4f}   (effect above chance {eff_sub:+.4f})")
    print(f"  class quotient     no-synapse {q_sub:.4f}")
    print(f"  BIO loss           {loss_bio:+.4f}   = {100*loss_bio/eff_sub:+.1f}% of the strain's own effect")
    print(f"  RANDOM loss k=119  {L.mean():+.4f} +/- {L.std():.4f}"
          f"   = {100*L.mean()/eff_sub:+.1f}% of effect     z(bio) = {z:+.2f}")
    return dict(eff=eff_sub, bio=loss_bio, rnd=L.mean(), rsd=L.std(), z=z)

wt  = report("wt",    "wildty", "WILD TYPE")
unc = report("unc31", "unc31",  "unc-31")
print("\nSUMMARY")
print(f"  effect available to lose:  WT {wt['eff']:+.4f}   unc-31 {unc['eff']:+.4f}")
print(f"  biological arm lossless?   WT {wt['bio']:+.4f}   unc-31 {unc['bio']:+.4f}")
print(f"  random arm destroys:       WT {100*wt['rnd']/wt['eff']:.0f}%   unc-31 {100*unc['rnd']/unc['eff']:.0f}% of own effect")
