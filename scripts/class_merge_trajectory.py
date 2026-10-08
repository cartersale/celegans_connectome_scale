"""Sweep matched class and random merge trajectories from 302 neurons to 119 groups and record the propagation-AUC loss.

Two arms per replicate, sharing one shuffled class order:
  bio     multi-member classes are collapsed whole, one class per step, in
          random order
  random  at each step a group of the same size as that step's class is
          collapsed, drawn from the neurons not yet placed in a merged group
So after each step both arms have the same k and the same group-size profile.
k decreases by (class size - 1) per step, so a replicate visits only the k
values its class order produces, and no row is written for k = 302.

Loss = AUC(full resolvent) - AUC(quotient resolvent lifted to neuron pairs,
randi.class_leverage), ALPHA = 0.1, predicting Randi et al. 2023 wild-type
response incidence over the pairs observed in occ3. Reported for all observed
pairs (loss_all) and for pairs with no synapse from stimulated to responder
(loss_nosyn).

Usage: python scripts/class_merge_trajectory.py [--n-rep 30] [--seed 0]
    Replicate r uses PCG64(seed + r) for both arms.
Inputs: connectome via conres.celegans; occ3 and responses via conres.randi.
Outputs: results/class_merge_trajectory.csv (mode, rep, k, n_merged, loss_all,
    loss_nosyn). `setup` is imported by diagnostics/metric_comparability.py.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
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
    return A, nodes, labels, jj, ii, y


def trajectory(A, labels, jj, ii, y, sub, rng, mode: str):
    """One merge trajectory. Yields (k, n_merged, loss_all, loss_sub) per step."""
    n = A.shape[0]
    classes = [np.flatnonzero(labels == c) for c in np.unique(labels)]
    classes = [c for c in classes if c.size > 1]
    rng.shuffle(classes)

    cell = np.arange(n)                     # current partition, starts all-singleton
    free = list(range(n))                   # neurons not yet placed in a merged cell
    rng.shuffle(free)
    free_set = set(free)
    out = []
    for grp in classes:
        if mode == "bio":
            members = [g for g in grp if g in free_set]
            if len(members) < 2:
                continue
        else:
            take = min(grp.size, len(free_set))
            if take < 2:
                continue
            members = [free.pop() for _ in range(take) if free]
        for m in members:
            free_set.discard(m)
        cell[members] = members[0]          # collapse onto one representative
        lab = np.unique(cell, return_inverse=True)[1]
        v = randi.class_leverage(A, lab, ALPHA)[jj, ii]
        out.append((int(np.unique(lab).size), int(n - np.unique(lab).size),
                    roc_auc_score(y, v), roc_auc_score(y[sub], v[sub])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-rep", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    A, nodes, labels, jj, ii, y = setup()
    direct = A[jj, ii]
    sub = direct == 0
    full = randi.resolvent(A, ALPHA)[jj, ii]
    auc_all = roc_auc_score(y, full)
    auc_sub = roc_auc_score(y[sub], full[sub])
    print(f"full operator: all pairs {auc_all:.4f}   no-synapse {auc_sub:.4f}")
    print(f"pairs {len(y):,} ({int(y.sum()):,} propagating); no-synapse {int(sub.sum()):,} "
          f"({int(y[sub].sum()):,})")
    print(f"biological partition: k=119 over {len(nodes)} neurons\n")

    rows = []
    for mode in ("bio", "random"):
        for r in range(args.n_rep):
            rng = np.random.Generator(np.random.PCG64(args.seed + r))
            for k, nm, a_all, a_sub in trajectory(A, labels, jj, ii, y, sub, rng, mode):
                rows.append((mode, r, k, nm, auc_all - a_all, auc_sub - a_sub))
            print(f"  {mode:6s} rep {r + 1}/{args.n_rep}", end="\r", flush=True)
        print()

    import csv
    out = ROOT / "results/class_merge_trajectory.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mode", "rep", "k", "n_merged", "loss_all", "loss_nosyn"])
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)")

    import collections
    arr = collections.defaultdict(list)
    for mode, r, k, nm, la, ls in rows:
        arr[(mode, k)].append((la, ls))
    print(f"\n{'k':>5} {'merged':>7} {'bio loss (no-syn)':>19} {'random loss (no-syn)':>22} "
          f"{'% of effect lost':>17}")
    effect = auc_sub - 0.5
    for k in (290, 275, 260, 242, 220, 200, 180, 160, 140, 119):
        b = arr.get(("bio", k)); q = arr.get(("random", k))
        if not b or not q:
            continue
        bl = np.mean([x[1] for x in b]); ql = np.mean([x[1] for x in q])
        print(f"{k:>5} {302 - k:>7} {bl:>+19.4f} {ql:>+22.4f} {100 * ql / effect:>16.0f}%")
    print(f"\n  full-operator effect above chance (no-synapse) = {effect:.4f}")


if __name__ == "__main__":
    main()
