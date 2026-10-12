"""Two loss metrics on the same C. elegans merge trajectories: AUC shortfall
against measured responses and 1 - Spearman against the full path operator.

The fly has no propagation measurement, so its structural loss is
self-referential (quotient vs full operator); the worm loss used elsewhere is
against measured responses. Here both are computed on identical worm
trajectories:

  trajectory   cell classes of size > 1 merged one at a time in random order
               (bio), or groups of the same sizes drawn at random from unused
               neurons (random); --n-rep trajectories per arm, seed = rep
  loss_auc     AUC(full R) - AUC(class-level operator) over unsynapsed observed
               pairs
  loss_self    1 - Spearman(class-level operator, R) over all observed pairs

alpha and the observed pair set come from class_merge_trajectory.setup.
Compression is n/k, as for the fly. The printed fly reference values are the
rungs of results/fly_fragility_alpha0.1.csv with compression <= 2.5; the
printed table shows only k values that a trajectory hits exactly.

Inputs: class_merge_trajectory.setup, conres.randi,
    results/fly_fragility_alpha0.1.csv.
Outputs: results/metric_comparability.csv (mode, rep, k, compression,
    loss_auc_vs_measured, loss_self_vs_operator).
Usage: python scripts/diagnostics/metric_comparability.py [--n-rep 15]
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
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conres import randi          # noqa: E402
from conres.paths import tee_stdout                   # noqa: E402
tee_stdout(__file__)
from class_merge_trajectory import setup, ALPHA      # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def run(A, labels, jj, ii, y, sub, full, auc_sub, rng, mode):
    n = A.shape[0]
    classes = [np.flatnonzero(labels == c) for c in np.unique(labels)]
    classes = [c for c in classes if c.size > 1]
    rng.shuffle(classes)
    cell = np.arange(n)
    free = list(range(n)); rng.shuffle(free); free_set = set(free)
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
        cell[members] = members[0]
        lab = np.unique(cell, return_inverse=True)[1]
        v = randi.class_leverage(A, lab, ALPHA)[jj, ii]
        k = int(np.unique(lab).size)
        loss_auc = auc_sub - roc_auc_score(y[sub], v[sub])
        loss_self = 1.0 - float(stats.spearmanr(v, full).statistic)
        out.append((k, n / k, loss_auc, loss_self))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-rep", type=int, default=15)
    args = ap.parse_args()
    A, nodes, labels, jj, ii, y = setup()
    sub = A[jj, ii] == 0
    full = randi.resolvent(A, ALPHA)[jj, ii]
    auc_sub = roc_auc_score(y[sub], full[sub])

    rows = []
    for mode in ("bio", "random"):
        for r in range(args.n_rep):
            rng = np.random.Generator(np.random.PCG64(r))
            for k, comp, la, ls in run(A, labels, jj, ii, y, sub, full, auc_sub, rng, mode):
                rows.append((mode, r, k, comp, la, ls))
            print(f"  {mode} {r+1}/{args.n_rep}", end="\r", flush=True)
        print()

    out = ROOT / "results/metric_comparability.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(
            ["mode", "rep", "k", "compression", "loss_auc_vs_measured", "loss_self_vs_operator"])
        w.writerows(rows)

    import collections
    agg = collections.defaultdict(list)
    for mode, r, k, comp, la, ls in rows:
        agg[(mode, k)].append((la, ls))
    print(f"\nWORM, both metrics on identical trajectories")
    print(f"{'k':>5} {'compr':>7} | {'AUC loss vs measured':>22} | {'1-rho vs operator':>20}")
    print(f"{'':>5} {'':>7} | {'bio':>10} {'random':>11} | {'bio':>9} {'random':>10}")
    print("-" * 76)
    for k in (290, 260, 242, 220, 200, 180, 160, 140, 119):
        b, q = agg.get(("bio", k)), agg.get(("random", k))
        if not b or not q:
            continue
        print(f"{k:>5} {302/k:>6.2f}x | {np.mean([x[0] for x in b]):>+10.4f} "
              f"{np.mean([x[0] for x in q]):>+11.4f} | {np.mean([x[1] for x in b]):>9.4f} "
              f"{np.mean([x[1] for x in q]):>10.4f}")
    print(f"\nFLY at comparable compression (self-referential metric, same definition):")
    with (ROOT / "results/fly_fragility_alpha0.1.csv").open() as fh:
        for r in csv.DictReader(fh):
            if float(r["compression"]) <= 2.5:
                print(f"  {float(r['compression']):.2f}x : bio {float(r['bio_loss']):.4f}  "
                      f"random {float(r['rand_loss_mean']):.4f}")
    print(f"\nworm max compression is {302/119:.2f}x, so the overlap with the fly is 1x-2.5x")


if __name__ == "__main__":
    main()
