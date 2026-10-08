"""Infer a defect-selected merge set in each of the eight Witvliet 2021
connectomes and score it, unchanged, on every other animal.

For each animal, the --n-merge lowest-defect disjoint pairs are chosen with
pairwise_defect / greedy_disjoint from prospective_merges.py (mean of out-
and in-direction defects of the all-synapse matrix). The 8x8 matrix L[a, b] is
the structural loss of animal a's partition evaluated on animal b: 1 - Spearman
between the lifted quotient resolvent and the full resolvent (alpha = 0.1) over
off-diagonal entries. The diagonal is scored on the graph the partition was
chosen from and is the within-animal reference.

Fixed across animals and arms:
  - node set: neurons reconstructed in all eight datasets
    (conres.celegans.witvliet_common_neurons);
  - candidate pool: neurons at or above median binary degree in the summed
    graph of all eight animals;
  - every partition is exactly --n-merge disjoint pairs, so k = n - n_merge and
    the cell-size profile are identical across arms.
Synapse count differs severalfold across the series and structural loss depends
on density, so arms are compared only within a test animal (column).

Per test animal b: within = L[b, b]; transfer = mean of L[a, b] over a != b;
biological = mean over --n-null draws of within-class disjoint pairs from the
pool; random = mean over --n-null random disjoint pairs from the pool;
retention = (random - transfer) / (random - within). Also reported: transfer
with the neurons selected in each training animal randomly re-paired (5 draws
per training animal), a within-class arm built by degree_matched_bio, mean
pooled degree of merged neurons, Jaccard agreement of merge sets between
animals, same-class rate against the pool chance rate, and the Spearman of
column-centred transfer loss against developmental-stage distance.

Inputs: Witvliet 2021 connectomes (conres.celegans.witvliet_files/_matrix);
imports prospective_merges.
Outputs: results/cross_animal_transfer_m<n_merge>.csv with rows
(kind, train, test, value): kind "loss" for the 8x8 matrix, and within,
transfer, biological, random, retention per test animal. The re-paired and
degree_matched_bio results go to stdout only.
Usage: cross_animal_transfer.py [--n-merge 30] [--n-null 50] [--seed 0] [--exclude-glia]
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

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from conres import celegans as ce, randi                      # noqa: E402
from prospective_merges import (pairwise_defect,          # noqa: E402
                                     greedy_disjoint, labels_from_pairs)

ALPHA = 0.1


def pool_mask(mats: list[np.ndarray]) -> np.ndarray:
    """Neurons at or above median binary in+out degree in the summed graph.

    Near-empty rows are trivially similar and favour any merge criterion, hence
    the floor; computing it on the summed graph gives every animal the same pool.
    """
    S = sum(mats)
    deg = (S > 0).sum(1) + (S > 0).sum(0)
    return deg >= np.median(deg)


def struct_loss(A: np.ndarray, lab: np.ndarray, full: np.ndarray,
                oj: np.ndarray, oi: np.ndarray) -> float:
    """1 - Spearman between the lifted quotient resolvent and `full` (the full
    resolvent at the off-diagonal entries oj, oi)."""
    RQ = randi.resolvent(randi.quotient(A, lab), ALPHA)
    return 1.0 - stats.spearmanr(RQ[np.ix_(lab, lab)][oj, oi], full).statistic


def random_pairs(pool: np.ndarray, m: int, rng) -> list[tuple[int, int]]:
    p = rng.permutation(pool)
    return [(int(p[2 * i]), int(p[2 * i + 1])) for i in range(m)]


def biological_pairs(pool: np.ndarray, lab_bio: np.ndarray, m: int, rng
                     ) -> list[tuple[int, int]] | None:
    """m random disjoint within-class pairs from `pool`, or None if the pool
    holds fewer than m."""
    byc = collections.defaultdict(list)
    for i in pool:
        byc[lab_bio[i]].append(int(i))
    cand = []
    for members in byc.values():
        v = rng.permutation(members)
        cand += [(int(v[2 * i]), int(v[2 * i + 1])) for i in range(len(v) // 2)]
    if len(cand) < m:
        return None
    return [cand[i] for i in rng.permutation(len(cand))[:m]]


def degree_matched_bio(pool, lab_bio, sel, deg, m, rng):
    """Within-class disjoint pairs from `pool`, intended to match the degree
    profile of the defect-selected pairs. As written, `target` and `want` are
    unused and the function returns the m random within-class pairs with the
    highest summed degree `deg`."""
    target = np.sort([deg[x] for pr in sel[rng.integers(len(sel))] for x in pr])
    byc = collections.defaultdict(list)
    for i in pool:
        byc[lab_bio[i]].append(int(i))
    cand = []
    for members in byc.values():
        v = rng.permutation(members)
        cand += [(int(v[2 * i]), int(v[2 * i + 1])) for i in range(len(v) // 2)]
    if len(cand) < m:
        return None
    key = np.array([deg[a] + deg[b] for a, b in cand], float)
    want = target.reshape(-1, 2).sum(1)
    want = np.sort(want)[::-1][:m]
    order = np.argsort(-key)
    return [cand[i] for i in order[:m]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-merge", type=int, default=30)
    ap.add_argument("--n-null", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude-glia", action="store_true",
                    help="drop the five glia from the node set; output gets a _noglia suffix")
    args = ap.parse_args()
    m = args.n_merge

    nodes = ce.witvliet_common_neurons("all", exclude_glia=args.exclude_glia)
    n = len(nodes)
    lab_bio = np.unique(ce.partitions(nodes)["class"], return_inverse=True)[1]
    files = ce.witvliet_files()
    mats = [ce.witvliet_matrix(p, nodes, "all") for _, _, p in files]
    stages = [f"{i}{s[0]}" for i, s, _ in files]
    pool = np.flatnonzero(pool_mask(mats))

    off = ~np.eye(n, dtype=bool)
    oj, oi = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    oj, oi = oj[off], oi[off]

    print(f"{n} neurons in all eight datasets; candidate pool {pool.size}; "
          f"{m} disjoint merges per partition (k = {n - m})")
    print(f"synapses: " + "  ".join(f"{s}={A.sum():.0f}" for s, A in zip(stages, mats)))

    sel, labs = [], []
    allowed = np.zeros(n, bool); allowed[pool] = True
    for A in mats:
        g = (pairwise_defect(A) + pairwise_defect(A.T)) / 2.0
        s = greedy_disjoint(g, m, allowed)
        sel.append(s)
        labs.append(labels_from_pairs(n, s))

    # rows = partition inferred in, cols = evaluated on
    full = [randi.resolvent(A, ALPHA)[oj, oi] for A in mats]
    L = np.array([[struct_loss(mats[b], labs[a], full[b], oj, oi)
                   for b in range(8)] for a in range(8)])

    print(f"\nstructural loss, rows = partition inferred in, cols = evaluated on")
    print("      " + "".join(f"{s:>8s}" for s in stages))
    for a in range(8):
        print(f"{stages[a]:>5s} " + "".join(
            (f"[{L[a, b]:.3f}]" if a == b else f"{L[a, b]:>8.3f}") for b in range(8)))

    rng = np.random.Generator(np.random.PCG64(args.seed))
    rows = []
    print(f"\n{'test':>5} {'within':>8} {'transfer':>9} {'biolog.':>8} {'random':>8} "
          f"{'retention':>10}")
    for b in range(8):
        within = L[b, b]
        transfer = np.mean([L[a, b] for a in range(8) if a != b])
        bio = np.mean([struct_loss(mats[b], labels_from_pairs(n, biological_pairs(
            pool, lab_bio, m, rng)), full[b], oj, oi) for _ in range(args.n_null)])
        ran = np.mean([struct_loss(mats[b], labels_from_pairs(n, random_pairs(
            pool, m, rng)), full[b], oj, oi) for _ in range(args.n_null)])
        ret = (ran - transfer) / (ran - within) if ran > within else np.nan
        rows.append((stages[b], within, transfer, bio, ran, ret))
        print(f"{stages[b]:>5s} {within:>8.3f} {transfer:>9.3f} {bio:>8.3f} {ran:>8.3f} "
              f"{ret:>9.1%}")
    arr = np.array([r[1:] for r in rows], float)
    print(f"{'mean':>5} " + "".join(f"{v:>8.3f}" if i < 4 else f"{v:>10.1%}"
                                    for i, v in enumerate(arr.mean(0))))

    # Re-pairing the neurons each training animal selected separates the effect
    # of which neurons are merged from the effect of how they are paired.
    print(f"\n{'test':>5} {'transfer':>9} {'re-paired':>10} {'bio(deg-m)':>11}")
    deg_pool = {}
    for b in range(8):
        Ab = mats[b]
        d = (Ab > 0).sum(1) + (Ab > 0).sum(0)
        deg_pool[b] = d
    rep_rows = []
    for b in range(8):
        transfer = np.mean([L[a, b] for a in range(8) if a != b])
        rep = []
        for a in range(8):
            if a == b:
                continue
            touched = np.array(sorted({x for pr in sel[a] for x in pr}))
            for _ in range(5):
                rep.append(struct_loss(mats[b], labels_from_pairs(
                    n, random_pairs(touched, m, rng)), full[b], oj, oi))
        bd = np.mean([struct_loss(mats[b], labels_from_pairs(n, degree_matched_bio(
            pool, lab_bio, sel, deg_pool[b], m, rng)), full[b], oj, oi)
            for _ in range(args.n_null)])
        rep_rows.append((stages[b], transfer, np.mean(rep), bd))
        print(f"{stages[b]:>5s} {transfer:>9.3f} {np.mean(rep):>10.3f} {bd:>11.3f}")
    ra = np.array([r[1:] for r in rep_rows], float)
    print(f"{'mean':>5} " + "".join(f"{v:>9.3f}" if i == 0 else f"{v:>10.3f}"
                                    for i, v in enumerate(ra.mean(0))))

    # degree of the neurons each arm touches, on the pooled graph
    Sp = sum(mats)
    dg = (Sp > 0).sum(1) + (Sp > 0).sum(0)
    inf_deg = np.mean([dg[x] for s_ in sel for pr in s_ for x in pr])
    bio_deg = np.mean([dg[x] for _ in range(20)
                       for pr in biological_pairs(pool, lab_bio, m, rng) for x in pr])
    print(f"\nmean pooled degree of merged neurons: inferred {inf_deg:.1f}, "
          f"within-class {bio_deg:.1f}, whole pool {dg[pool].mean():.1f}")

    setsel = [{frozenset(p) for p in s} for s in sel]
    jac = [len(setsel[a] & setsel[b]) / len(setsel[a] | setsel[b])
           for a in range(8) for b in range(a + 1, 8)]
    same = [np.mean([lab_bio[x] == lab_bio[y] for x, y in s]) for s in sel]
    cs = np.bincount(lab_bio[pool])
    chance = (cs * (cs - 1) // 2).sum() / (pool.size * (pool.size - 1) / 2)
    print(f"\nmerge-set agreement between animals: Jaccard {np.mean(jac):.3f} "
          f"[{np.min(jac):.3f}, {np.max(jac):.3f}]")
    print(f"same-class rate of selected merges: {np.mean(same):.1%} "
          f"against a pool chance rate of {chance:.1%} ({np.mean(same)/chance:.1f}x)")

    # transfer loss centred on each test column's mean (diagonal included)
    d = np.array([abs(a - b) for a in range(8) for b in range(8) if a != b], float)
    v = np.array([L[a, b] for a in range(8) for b in range(8) if a != b], float)
    col = np.array([L[:, b].mean() for a in range(8) for b in range(8) if a != b])
    r_ = stats.spearmanr(d, v - col)
    print(f"transfer loss vs |stage difference| (column-centred): "
          f"rho {r_.statistic:+.3f}, p {r_.pvalue:.3f}")

    out = ROOT / f"results/cross_animal_transfer_m{m}{'_noglia' if args.exclude_glia else ''}.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kind", "train", "test", "value"])
        for a in range(8):
            for b in range(8):
                w.writerow(["loss", stages[a], stages[b], f"{L[a, b]:.6f}"])
        for st, wi, tr, bi, ra, re in rows:
            for k, v in (("within", wi), ("transfer", tr), ("biological", bi),
                         ("random", ra), ("retention", re)):
                w.writerow([k, "", st, f"{v:.6f}"])
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
