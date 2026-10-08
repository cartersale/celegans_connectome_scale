"""Score incidence AUC and correlation Spearman along class-merge trajectories
and matched random trajectories.

Each replicate shuffles the multi-member cell classes and collapses them one at
a time; after each step the lifted class-level resolvent (alpha = 0.1) is scored
on both observables from two_observables.setup. The random arm uses the
same class order (same seed per replicate) but fills each cell with randomly
chosen unmerged neurons, so k and the cell-size profile match the class arm at
every step. Losses are expressed as fractions of the full operator's headroom:
(AUC_full - AUC) / (AUC_full - 0.5) and (rho_full - rho) / rho_full.

Inputs: as two_observables.py (imports its setup; no detrending).
Outputs: results/two_observable_trajectory.csv (mode, rep, k, auc, rho,
frac_lost_incidence, frac_lost_correlation); a summary at selected k to stdout.
Usage: two_observable_trajectory.py [--n-rep 15] [--min-animals 3]
       [--excise 0] [--seed 0]
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from conres import randi                            # noqa: E402
from two_observables import setup              # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ALPHA = 0.1


def trajectory(dat, rng, mode: str):
    A, labels = dat["A"], dat["labels"]
    n = A.shape[0]
    classes = [np.flatnonzero(labels == c) for c in np.unique(labels)]
    classes = [c for c in classes if c.size > 1]
    rng.shuffle(classes)

    cell = np.arange(n)
    free = list(range(n))
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
        cell[members] = members[0]
        lab = np.unique(cell, return_inverse=True)[1]
        S = randi.class_leverage(A, lab, ALPHA)
        auc = roc_auc_score(dat["y"], S[dat["jj"], dat["ii"]])
        rho = stats.spearmanr(S[dat["iu"], dat["ju"]] + S[dat["ju"], dat["iu"]],
                              dat["r"]).statistic
        out.append((int(lab.max() + 1), auc, rho))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-rep", type=int, default=15)
    ap.add_argument("--min-animals", type=int, default=3)
    ap.add_argument("--excise", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dat = setup(args.min_animals, args.excise, False)
    Sfull = randi.resolvent(dat["A"], ALPHA)
    auc0 = roc_auc_score(dat["y"], Sfull[dat["jj"], dat["ii"]])
    rho0 = stats.spearmanr(Sfull[dat["iu"], dat["ju"]] + Sfull[dat["ju"], dat["iu"]],
                           dat["r"]).statistic
    print(f"full operator: incidence AUC {auc0:.4f}   correlation rho {rho0:.4f}")
    print(f"headroom: incidence {auc0 - 0.5:.4f}   correlation {rho0:.4f}\n")

    rows = []
    for mode in ("bio", "random"):
        for rep in range(args.n_rep):
            rng = np.random.Generator(np.random.PCG64(args.seed + rep))
            for k, a, r_ in trajectory(dat, rng, mode):
                rows.append((mode, rep, k, a, r_,
                             (auc0 - a) / (auc0 - 0.5), (rho0 - r_) / rho0))
            print(f"  {mode:6s} rep {rep + 1}/{args.n_rep}", end="\r", flush=True)
        print()

    out = ROOT / "results/two_observable_trajectory.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mode", "rep", "k", "auc", "rho", "frac_lost_incidence",
                    "frac_lost_correlation"])
        w.writerows(rows)

    import collections
    agg = collections.defaultdict(list)
    for mode, rep, k, a, r_, fa, fr in rows:
        agg[(mode, k)].append((fa, fr))
    print(f"\n{'k':>5} {'compress':>9} | {'bio incid':>10} {'bio corr':>9} | "
          f"{'rand incid':>11} {'rand corr':>10}")
    for k in (290, 275, 260, 242, 220, 200, 180, 160, 140, 119):
        b, q = agg.get(("bio", k)), agg.get(("random", k))
        if not b or not q:
            continue
        print(f"{k:>5} {302 / k:>8.2f}x | {np.mean([x[0] for x in b]):>+10.3f} "
              f"{np.mean([x[1] for x in b]):>+9.3f} | "
              f"{np.mean([x[0] for x in q]):>+11.3f} {np.mean([x[1] for x in q]):>+10.3f}")
    print(f"\nwrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
