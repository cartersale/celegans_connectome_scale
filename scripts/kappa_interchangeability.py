"""Agreement of evoked-response sets between stimulated neurons of the same cell
class, as Cohen's kappa, against pairs from different classes.

For two stimulated neurons j1, j2 the comparison set is the neurons (other than
j1, j2) recorded during stimulation of both (occ3 > 0 for both), so
observability is held fixed; at least --min-shared such neurons are required.
Each member contributes a binary vector of detected responses over that set,
and kappa corrects for each member's detection rate. Within-class: every
co-class pair of stimulated neurons. Between-class: --n-between random pairs of
stimulated neurons from different classes (sampled with replacement, at most
50,000 draws).

At each response filter (min responses per member over the shared set: 0, 3,
5), reported:
  pair-level      Mann-Whitney of within- vs between-class kappa
  class-level     Mann-Whitney of class-mean kappa vs between-class pairs, so
                  that a large class (VA contributes up to 66 pairs) counts once
  permutation     class membership reassigned among the stimulated neurons in
                  the filter's classes, class sizes fixed (--n-perm); statistic
                  is the mean of class-mean kappa
  leave-out       class-mean kappa without the three classes contributing most
                  pairs
  metric guard    conres.metric_guard.guard_binary_similarity (500
                  permutations) on 60 sampled within-class pairs; fractions
                  flagged FLOOR and SPARSE

Inputs: conres.celegans (build_matrix, assign_classes), conres.randi
    (build_occ3, Randi 2023 wild-type responses).
Outputs: printed only.
Usage: python scripts/kappa_interchangeability.py [--min-shared 20]
    [--n-between 400] [--n-perm 2000] [--seed 0]
"""
from __future__ import annotations

import argparse
import collections
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy import stats

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from conres import celegans as ce, randi                          # noqa: E402
from conres.metric_guard import guard_binary_similarity           # noqa: E402


def kappa(d1, d2):
    po = (d1 == d2).mean()
    p1, p2 = d1.mean(), d2.mean()
    pe = p1 * p2 + (1 - p1) * (1 - p2)
    return (po - pe) / (1 - pe) if pe < 1 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-shared", type=int, default=20)
    ap.add_argument("--n-between", type=int, default=400)
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    A, nodes = ce.build_matrix("all", neurons_only=True)
    mapping = ce.assign_classes(nodes)
    idx = {n: i for i, n in enumerate(nodes)}
    occ3, _ = randi.build_occ3(nodes, "wt")
    d = randi.load_responses("wildty")
    d = d[d.pre.isin(idx) & d.post.isin(idx)]
    det = np.zeros_like(occ3, dtype=bool)
    for p, q in zip(d.pre, d.post):
        det[idx[q], idx[p]] = True                  # [responder, stimulated]

    stim = [j for j in range(len(nodes)) if occ3[:, j].sum() > 0]
    cls_of = {j: mapping[nodes[j]] for j in stim}

    def compare(j1, j2):
        S = np.flatnonzero((occ3[:, j1] > 0) & (occ3[:, j2] > 0))
        S = S[(S != j1) & (S != j2)]
        if S.size < args.min_shared:
            return None
        d1, d2 = det[S, j1], det[S, j2]
        return {"n_shared": S.size, "kappa": kappa(d1, d2),
                "min_resp": int(min(d1.sum(), d2.sum())), "d1": d1, "d2": d2}

    by_class = collections.defaultdict(list)
    for j in stim:
        by_class[cls_of[j]].append(j)
    within_all = []
    for c, mem in by_class.items():
        for a in range(len(mem)):
            for b in range(a + 1, len(mem)):
                r = compare(mem[a], mem[b])
                if r:
                    within_all.append({"cls": c, "size": len(mem), **r})

    rng = np.random.default_rng(args.seed)
    between_all, tries = [], 0
    while len(between_all) < args.n_between and tries < 50000:
        tries += 1
        j1, j2 = rng.choice(stim, 2, replace=False)
        if cls_of[j1] == cls_of[j2]:
            continue
        r = compare(j1, j2)
        if r:
            between_all.append(r)

    # guard on a sample of the actual vectors
    def guard_rate(rows, n=60):
        pick = rng.choice(len(rows), min(n, len(rows)), replace=False)
        flags = [guard_binary_similarity(rows[i]["d1"], rows[i]["d2"], n_perm=500)["flags"]
                 for i in pick]
        floor = np.mean([any(f.startswith("FLOOR") for f in fl) for fl in flags])
        sparse = np.mean([any(f.startswith("SPARSE") for f in fl) for fl in flags])
        return floor, sparse

    print(f"{len(stim)} stimulated neurons in {len(by_class)} classes; "
          f"min {args.min_shared} shared observed responders per comparison")
    print(f"\n{'filter':>28} {'classes':>8} {'pairs':>6} {'within':>8} {'between':>8} "
          f"{'pair MW p':>10} {'class-mean MW p':>16} {'perm p':>8} {'guard FLOOR/SPARSE':>20}")
    for m in (0, 3, 5):
        W = [r for r in within_all if r["min_resp"] >= m]
        B = [r for r in between_all if r["min_resp"] >= m]
        wk = np.array([r["kappa"] for r in W])
        bk = np.array([r["kappa"] for r in B])
        classes = sorted({r["cls"] for r in W})
        cmean = np.array([np.nanmean([r["kappa"] for r in W if r["cls"] == c]) for c in classes])
        p_pair = stats.mannwhitneyu(wk[np.isfinite(wk)], bk[np.isfinite(bk)]).pvalue
        p_cls = stats.mannwhitneyu(cmean[np.isfinite(cmean)], bk[np.isfinite(bk)]).pvalue

        # permutation: reassign class membership among the stimulated neurons that
        # appear in this filter's within-class pairs, class sizes fixed; statistic
        # is the mean of class-mean kappa, the same one-value-per-class estimand
        neurons = sorted({j for c in classes for j in by_class[c]})
        true_lab = np.array([cls_of[j] for j in neurons])
        cache = {}

        def pair_k(a, b):
            key = (min(a, b), max(a, b))
            if key not in cache:
                r = compare(*key)
                cache[key] = r["kappa"] if (r and r["min_resp"] >= m) else np.nan
            return cache[key]

        def stat(lab):
            vals = []
            for c in np.unique(lab):
                mem = [neurons[i] for i in np.flatnonzero(lab == c)]
                ks = [pair_k(mem[a], mem[b]) for a in range(len(mem)) for b in range(a + 1, len(mem))]
                ks = [x for x in ks if np.isfinite(x)]
                if ks:
                    vals.append(np.mean(ks))
            return np.mean(vals) if vals else np.nan

        obs = stat(true_lab)
        prng = np.random.default_rng(args.seed + m)
        null = np.array([stat(prng.permutation(true_lab)) for _ in range(args.n_perm)])
        null = null[np.isfinite(null)]
        p_perm = (1 + np.sum(null >= obs)) / (1 + null.size)
        fl, sp = guard_rate(W)
        label = {0: "none", 3: "both drive >= 3 responses", 5: "both drive >= 5 responses"}[m]
        print(f"{label:>28} {len(classes):>8} {len(W):>6} {np.nanmean(wk):>8.3f} "
              f"{np.nanmean(bk):>8.3f} {p_pair:>10.1e} {p_cls:>16.1e} {p_perm:>8.4f} "
              f"{fl:>9.0%}/{sp:.0%}")
        print(f"{'':>28} class-mean kappa {np.nanmean(cmean):.3f}; permutation null "
              f"{null.mean():.3f} [{np.percentile(null, 2.5):.3f}, {np.percentile(null, 97.5):.3f}]"
              f" (n={null.size}); between-class pairs in filter: {len(B)}")

        # which classes carry it
        sizes = collections.Counter(r["cls"] for r in W)
        top = sizes.most_common(3)
        print(f"{'':>28} largest contributors: " +
              ", ".join(f"{c} {n} pairs (mean kappa {np.nanmean([r['kappa'] for r in W if r['cls'] == c]):.3f})"
                        for c, n in top))
        keep = [c for c in classes if c not in {t[0] for t in top}]
        cm2 = np.array([np.nanmean([r["kappa"] for r in W if r["cls"] == c]) for c in keep])
        print(f"{'':>28} without those three classes: class-mean kappa {np.nanmean(cm2):.3f}, "
              f"MW vs between p {stats.mannwhitneyu(cm2[np.isfinite(cm2)], bk[np.isfinite(bk)]).pvalue:.1e}")


if __name__ == "__main__":
    main()
