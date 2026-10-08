"""Compare the functional cost of class-pure and class-heterogeneous cells in
the defect-greedy partition at a fixed merge depth.

The partition is defect_greedy from label_free_merge_trajectory.py at --merges
merges. Each multi-member cell is labelled pure (all members share a cell class)
or heterogeneous, and its cost is measured two ways:

  isolated   AUC loss of the partition in which only that cell is merged and
             every other neuron is a singleton
  marginal   full-partition loss minus the loss after un-merging that cell
             into singletons, i.e. its cost in the context of the other cells

Loss is the drop in AUC for predicting Randi 2023 responses among observed pairs
without a direct synapse, lifted quotient resolvent (alpha = 0.1) against the
full resolvent. Larger cells are more likely to span classes, so the primary
comparison is restricted to size-2 cells (one-sided Mann-Whitney,
heterogeneous > pure); all multi-member cells and an OLS regression of cost on
size and heterogeneity are reported alongside.

Inputs: C. elegans connectome (conres.celegans), Randi 2023 wild-type responses
and observation counts (conres.randi); imports label_free_merge_trajectory.
Outputs: results/heterogeneity_cost_<merges>.csv (one row per multi-member cell).
Usage: heterogeneity_cost.py [--merges 180]   (--seed is accepted but unused)
"""
from __future__ import annotations

import argparse
import collections
import csv
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from conres import celegans as ce, randi          # noqa: E402
from label_free_merge_trajectory import defect_greedy  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ALPHA = 0.1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merges", type=int, default=180)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    A, nodes = ce.build_matrix("all", neurons_only=True)
    labels = ce.partitions(nodes)["class"]
    mapping = ce.assign_classes(nodes)
    n = A.shape[0]
    idx = {x: i for i, x in enumerate(nodes)}

    occ3, _ = randi.build_occ3(nodes, "wt")
    d = randi.load_responses("wildty")
    d = d[d.pre.isin(idx) & d.post.isin(idx)]
    hits = {(idx[p], idx[q]) for p, q in zip(d.pre, d.post)}
    ii, jj = np.nonzero(occ3)
    y = np.array([1.0 if (j, i) in hits else 0.0 for i, j in zip(ii, jj)])
    R = randi.resolvent(A, ALPHA)
    sub = A[jj, ii] == 0
    ys, fs = y[sub], R[jj, ii][sub]
    auc_full = roc_auc_score(ys, fs)

    def loss_of(lab):
        RQ = randi.resolvent(randi.quotient(A, lab), ALPHA)
        q = RQ[np.ix_(lab, lab)][jj, ii][sub]
        return auc_full - roc_auc_score(ys, q)

    dl = defect_greedy(A, args.merges)
    full_loss = loss_of(dl)
    cells = [np.flatnonzero(dl == c) for c in range(dl.max() + 1)]
    multi = [m for m in cells if m.size > 1]
    print(f"defect-greedy at {args.merges} merges: k={dl.max()+1}, "
          f"{len(multi)} multi-member cells, full-partition loss {full_loss:+.4f}")

    rows = []
    for m in multi:
        classes = {mapping[nodes[i]] for i in m}
        pure = len(classes) == 1
        lab_iso = np.arange(n)
        lab_iso[m] = m[0]
        lab_iso = np.unique(lab_iso, return_inverse=True)[1]
        lab_marg = dl.copy()
        lab_marg[m] = np.arange(n, n + m.size)
        lab_marg = np.unique(lab_marg, return_inverse=True)[1]
        rows.append({"size": int(m.size), "n_classes": len(classes),
                     "pure": int(pure),
                     "isolated_cost": round(loss_of(lab_iso), 5),
                     "marginal_cost": round(full_loss - loss_of(lab_marg), 5),
                     "members": "|".join(nodes[i] for i in m)})

    sizes = collections.Counter(r["size"] for r in rows)
    print(f"  cell sizes: {dict(sorted(sizes.items()))}")
    print(f"  pure {sum(r['pure'] for r in rows)} / heterogeneous "
          f"{sum(1-r['pure'] for r in rows)}\n")

    def report(sel, title):
        p = [r for r in sel if r["pure"]]
        h = [r for r in sel if not r["pure"]]
        if len(p) < 3 or len(h) < 3:
            print(f"  {title}: too few cells (pure {len(p)}, het {len(h)})")
            return
        print(f"  {title}   pure n={len(p)}, heterogeneous n={len(h)}")
        for key in ("isolated_cost", "marginal_cost"):
            a = np.array([r[key] for r in p]); b = np.array([r[key] for r in h])
            u = stats.mannwhitneyu(b, a, alternative="greater")   # H1: heterogeneous > pure
            print(f"    {key:>14}: pure {a.mean():+.5f}   heterogeneous {b.mean():+.5f}"
                  f"   diff {b.mean()-a.mean():+.5f}   one-sided p={u.pvalue:.3f}")

    print("=== PRIMARY: size-2 cells only (pure and heterogeneous differ in nothing else) ===")
    report([r for r in rows if r["size"] == 2], "size 2")

    print("\n=== all multi-member cells, size not controlled ===")
    report(rows, "all")

    print("\n=== size-controlled regression across all cells ===")
    X = np.column_stack([[r["size"] for r in rows],
                         [1 - r["pure"] for r in rows],
                         np.ones(len(rows))])
    for key in ("isolated_cost", "marginal_cost"):
        yv = np.array([r[key] for r in rows])
        beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
        resid = yv - X @ beta
        dof = len(yv) - X.shape[1]
        se = np.sqrt(np.sum(resid ** 2) / dof * np.linalg.inv(X.T @ X)[1, 1])
        t = beta[1] / se
        print(f"  {key:>14}: heterogeneity coefficient {beta[1]:+.5f} "
              f"(SE {se:.5f}, t={t:+.2f}, p={2*(1-stats.t.cdf(abs(t), dof)):.3f})"
              f"   size coefficient {beta[0]:+.5f}")

    with (ROOT / f"results/heterogeneity_cost_{args.merges}.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\nwrote results/heterogeneity_cost_{args.merges}.csv")


if __name__ == "__main__":
    main()
