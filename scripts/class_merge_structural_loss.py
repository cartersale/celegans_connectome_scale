"""Score the C. elegans cell-class merge trajectory with both the functional AUC
metric and the fly's structural metric, against a size-matched random
trajectory.

The worm's functional loss is an AUC shortfall against measured Randi 2023
responses; the fly's is 1 - Spearman(quotient operator, full operator), which
needs no measurements. Running both on the same worm trajectories separates the
effect of the metric from the effect of scale.

Trajectory: per replicate (seed + rep), the multi-neuron cell classes are
merged one at a time in a random order. The bio arm merges each class's members;
the random arm instead merges the same number of neurons drawn from a random
permutation of not-yet-merged neurons. Every step is scored, with the quotient
resolvent recomputed (alpha = 0.1):

  loss_auc_all     AUC(full) - AUC(quotient), all occ3-observed pairs
  loss_auc_nosyn   the same on unsynapsed observed pairs
  loss_struct_occ  1 - Spearman(quotient, full) over occ3-observed pairs
  loss_struct_all  1 - Spearman(quotient, full) over all ordered pairs

Inputs: Cook 2019 adjacency and cell classes (conres.celegans), Randi 2023
wild-type responses (conres.randi).
Outputs: results/class_merge_structural_loss.csv (mode, rep, k and the four losses);
a summary table on stdout.
Usage: python class_merge_structural_loss.py [--n-rep 20] [--seed 0]
"""
from __future__ import annotations

import argparse
import csv
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from conres import celegans as ce, randi          # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ALPHA = 0.1


def setup():
    A, nodes = ce.build_matrix("all", neurons_only=True)
    labels = ce.partitions(nodes)["class"]
    occ3, _ = randi.build_occ3(nodes, "wt")
    idx = {n: i for i, n in enumerate(nodes)}
    d = randi.load_responses("wildty")
    d = d[d.pre.isin(idx) & d.post.isin(idx)]
    hits = {(idx[p], idx[q]) for p, q in zip(d.pre, d.post)}
    ii, jj = np.nonzero(occ3)
    y = np.array([1.0 if (j, i) in hits else 0.0 for i, j in zip(ii, jj)])
    n = A.shape[0]
    aj, ai = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    off = aj.ravel() != ai.ravel()
    return A, nodes, labels, jj, ii, y, aj.ravel()[off], ai.ravel()[off]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-rep", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    A, nodes, labels, jj, ii, y, allj, alli = setup()
    n = A.shape[0]
    sub = A[jj, ii] == 0
    R = randi.resolvent(A, ALPHA)
    full_occ, full_all = R[jj, ii], R[allj, alli]
    auc_all_full = roc_auc_score(y, full_occ)
    auc_sub_full = roc_auc_score(y[sub], full_occ[sub])
    print(f"worm: {n} neurons, {int(np.unique(labels).size)} classes, alpha={ALPHA}")
    print(f"full operator AUC: all {auc_all_full:.4f}  no-synapse {auc_sub_full:.4f}")
    print(f"pair sets: occ3-observed {len(full_occ):,}   all ordered {len(full_all):,}\n")

    classes = [np.flatnonzero(labels == c) for c in np.unique(labels)]
    classes = [c for c in classes if c.size > 1]

    rows = []
    for mode in ("bio", "random"):
        for r in range(args.n_rep):
            rng = np.random.Generator(np.random.PCG64(args.seed + r))
            order = list(range(len(classes)))
            rng.shuffle(order)
            cell = np.arange(n)
            free = list(rng.permutation(n))
            free_set = set(free)
            for oi in order:
                grp = classes[oi]
                if mode == "bio":
                    members = [g for g in grp if g in free_set]
                else:
                    take = min(grp.size, len(free_set))
                    members = [free.pop() for _ in range(take) if free]
                if len(members) < 2:
                    continue
                for m in members:
                    free_set.discard(m)
                cell[members] = members[0]
                lab = np.unique(cell, return_inverse=True)[1]
                k = int(lab.max() + 1)
                RQ = randi.resolvent(randi.quotient(A, lab), ALPHA)
                qv_occ = RQ[lab[jj], lab[ii]]
                qv_all = RQ[lab[allj], lab[alli]]
                rows.append({
                    "mode": mode, "rep": r, "k": k,
                    "loss_auc_all": auc_all_full - roc_auc_score(y, qv_occ),
                    "loss_auc_nosyn": auc_sub_full - roc_auc_score(y[sub], qv_occ[sub]),
                    "loss_struct_occ": 1 - stats.spearmanr(qv_occ, full_occ).statistic,
                    "loss_struct_all": 1 - stats.spearmanr(qv_all, full_all).statistic,
                })
            print(f"  {mode:6s} rep {r+1}/{args.n_rep}", end="\r", flush=True)
        print()

    out = ROOT / "results/class_merge_structural_loss.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)\n")

    import collections
    agg = collections.defaultdict(list)
    for d in rows:
        agg[(d["mode"], d["k"])].append(d)
    print(f"{'k':>5} {'compr':>6} | {'bio AUC':>9} {'rand AUC':>9} | "
          f"{'bio STRUCT':>11} {'rand STRUCT':>12}")
    print("-" * 62)
    for k in (290, 275, 260, 242, 220, 200, 180, 160, 140, 119):
        b, q = agg.get(("bio", k)), agg.get(("random", k))
        if not b or not q:
            continue
        print(f"{k:>5} {n/k:>5.2f}x | "
              f"{np.mean([x['loss_auc_nosyn'] for x in b]):>+9.4f} "
              f"{np.mean([x['loss_auc_nosyn'] for x in q]):>+9.4f} | "
              f"{np.mean([x['loss_struct_occ'] for x in b]):>+11.4f} "
              f"{np.mean([x['loss_struct_occ'] for x in q]):>+12.4f}")
    print("\n  fly reference at comparable compression (structural metric):")
    print("    2.5x  bio 0.295  random 0.742      1.4x  bio 0.164  random 0.499")


if __name__ == "__main__":
    main()
