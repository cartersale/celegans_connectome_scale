"""Apply defect-selected merge sets inferred in each Witvliet 2021 connectome to
the Cook 2019 adult node set and score adult propagation prediction (Randi 2023).

The candidate pool is the cross_animal_transfer.py pool (--pool floor:
neurons at or above median degree in the summed Witvliet graph) or all common
Witvliet neurons (--pool all), restricted to neurons present in Cook and mapped
to Cook indices. Every arm merges --n-merge disjoint pairs from this pool, so k
and the cell-size profile are the same across arms.

Arms:
  Cook adult      pairwise_defect / greedy_disjoint (prospective_merges.py)
                  run on the Cook all-synapse matrix
  each Witvliet   the same criterion run on that animal, pairs mapped to Cook
  Cook, thinned   the criterion run on Cook with the pool-pool subgraph
                  resampled (multinomial) to each Witvliet animal's total synapse
                  count; 5 resamples per animal, averaged over animals
  biological      random within-class disjoint pairs from the pool, --n-null draws
  random          random disjoint pairs from the pool, --n-null draws
Each partition is scored with the lifted class-level resolvent (alpha = 0.1):
AUC on all observed Randi pairs, on pairs with an endpoint in the pool (one
evaluation set for every arm), and on pairs without a direct synapse, plus
structural loss 1 - Spearman(lifted, full) over off-diagonal entries. Headroom
lost is (AUC_full - AUC) / (AUC_full - 0.5). "L1 only" averages the first four
Witvliet datasets, assumed to be the L1 animals in witvliet_files() order.

Inputs: Cook 2019 connectome and Witvliet 2021 connectomes (conres.celegans),
Randi 2023 wild-type responses and observation counts (conres.randi); imports
prospective_merges and cross_animal_transfer.
Outputs: results/functional_transfer_<pool>_m<n_merge>.csv (arm, auc,
auc_inpool, auc_nosyn, struct_loss) and the printed table, with the standard
deviation of the two null arms, in results/functional_transfer_<pool>_m<n_merge>.log.
Usage: functional_transfer.py [--n-merge 30] [--pool floor|all]
       [--n-null 50] [--seed 0]
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
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from conres import celegans as ce, randi                       # noqa: E402
from conres.paths import _Tee                                  # noqa: E402
from prospective_merges import (pairwise_defect,           # noqa: E402
                                     greedy_disjoint, labels_from_pairs)
from cross_animal_transfer import pool_mask               # noqa: E402

ALPHA = 0.1


def infer(A: np.ndarray, allowed: np.ndarray, m: int) -> list[tuple[int, int]]:
    g = (pairwise_defect(A) + pairwise_defect(A.T)) / 2.0
    return greedy_disjoint(g, m, allowed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-merge", type=int, default=30)
    ap.add_argument("--pool", choices=("floor", "all"), default="floor",
                    help="'floor': neurons at or above median degree in the summed "
                         "Witvliet graph; 'all': every common Witvliet neuron")
    ap.add_argument("--n-null", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    m = args.n_merge
    log = ROOT / f"results/functional_transfer_{args.pool}_m{m}.log"
    sys.stdout = _Tee(sys.__stdout__, log.open("w", buffering=1))

    A, nodes = ce.build_matrix("all", neurons_only=True)
    n = len(nodes)
    idx = {x: i for i, x in enumerate(nodes)}
    cls = ce.assign_classes(nodes)
    occ3, _ = randi.build_occ3(nodes, "wt")
    d = randi.load_responses("wildty")
    d = d[d.pre.isin(idx) & d.post.isin(idx)]
    hits = {(idx[p], idx[q]) for p, q in zip(d.pre, d.post)}
    ii, jj = np.nonzero(occ3)
    y = np.array([1.0 if (j, i) in hits else 0.0 for i, j in zip(ii, jj)])
    nosyn = A[jj, ii] == 0

    wn = ce.witvliet_common_neurons("all")
    wmats = [ce.witvliet_matrix(p, wn, "all") for _, _, p in ce.witvliet_files()]
    stages = [f"{i}{s[0]}" for i, s, _ in ce.witvliet_files()]
    wpool = (np.flatnonzero(pool_mask(wmats)) if args.pool == "floor"
             else np.arange(len(wn)))
    lift = {i: idx[wn[i]] for i in wpool if wn[i] in idx}      # witvliet -> cook
    pool_cook = np.array(sorted(lift.values()))
    wallowed = np.zeros(len(wn), bool)
    wallowed[list(lift)] = True

    pmask = np.zeros(n, bool)
    pmask[pool_cook] = True
    inpool = pmask[ii] | pmask[jj]           # fixed evaluation set, identical for all arms

    Sfull = randi.resolvent(A, ALPHA)
    auc0 = roc_auc_score(y, Sfull[jj, ii])
    auc0_ns = roc_auc_score(y[nosyn], Sfull[jj, ii][nosyn])
    off = ~np.eye(n, dtype=bool)
    oj, oi = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    oj, oi = oj[off], oi[off]
    sfull = Sfull[oj, oi]

    print(f"Cook adult: {n} neurons; lifted candidate pool {pool_cook.size}; "
          f"{m} disjoint merges (k = {n - m})")
    print(f"Randi: {len(y):,} observed ordered pairs, {int(y.sum()):,} propagating; "
          f"no-synapse subgroup {int(nosyn.sum()):,}")
    auc0_p = roc_auc_score(y[inpool], Sfull[jj, ii][inpool])
    print(f"full operator: AUC {auc0:.4f} (pool-touching {auc0_p:.4f}, "
          f"no-synapse {auc0_ns:.4f}); pool-touching pairs {int(inpool.sum()):,}\n")

    def score(pairs):
        """(AUC all pairs, AUC pool-touching pairs, AUC no-synapse pairs,
        structural loss) for the partition made of `pairs`.

        Merging a minority of neurons changes the operator mostly near the merged
        neurons, which the all-pairs AUC dilutes. The pool-touching set is
        defined by the candidate pool rather than by each arm's own merges so
        that every arm is scored on the same pairs."""
        lab = labels_from_pairs(n, pairs)
        S = randi.class_leverage(A, lab, ALPHA)
        v = S[jj, ii]
        RQ = S[oj, oi]
        return (roc_auc_score(y, v), roc_auc_score(y[inpool], v[inpool]),
                roc_auc_score(y[nosyn], v[nosyn]),
                1.0 - stats.spearmanr(RQ, sfull).statistic)

    rng = np.random.Generator(np.random.PCG64(args.seed))

    def random_pairs():
        p = rng.permutation(pool_cook)
        return [(int(p[2 * i]), int(p[2 * i + 1])) for i in range(m)]

    def biological_pairs():
        byc = collections.defaultdict(list)
        for i in pool_cook:
            byc[cls[nodes[i]]].append(int(i))
        cand = []
        for v in byc.values():
            q = rng.permutation(v)
            cand += [(int(q[2 * i]), int(q[2 * i + 1])) for i in range(len(q) // 2)]
        return [cand[i] for i in rng.permutation(len(cand))[:m]]

    rows = []
    print(f"{'inferred in':>14} {'AUC':>8} {'in-pool':>9} {'no-syn':>8} {'struct':>8} "
          f"{'headroom lost':>15}")

    cook_allowed = np.zeros(n, bool)
    cook_allowed[pool_cook] = True
    sel_cook = infer(A, cook_allowed, m)
    a, tc, ns, st = score(sel_cook)
    print(f"{'Cook adult':>14} {a:>8.4f} {tc:>9.4f} {ns:>8.4f} {st:>8.4f} "
          f"{(auc0 - a) / (auc0 - 0.5):>14.1%}")
    rows.append(("cook", a, tc, ns, st))

    lifted = []
    for st_name, Aw in zip(stages, wmats):
        pairs = [(lift[x], lift[y_]) for x, y_ in infer(Aw, wallowed, m)]
        a, tc, ns, s_ = score(pairs)
        lifted.append((st_name, a, tc, ns, s_, pairs))
        print(f"{st_name:>14} {a:>8.4f} {tc:>9.4f} {ns:>8.4f} {s_:>8.4f} "
              f"{(auc0 - a) / (auc0 - 0.5):>14.1%}")
        rows.append((st_name, a, tc, ns, s_))
    arr = np.array([[x[1], x[2], x[3], x[4]] for x in lifted])
    mm = arr.mean(0)
    print(f"{'mean L1..adult':>14} {mm[0]:>8.4f} {mm[1]:>9.4f} {mm[2]:>8.4f} "
          f"{mm[3]:>8.4f} {(auc0 - mm[0]) / (auc0 - 0.5):>14.1%}")
    l1 = arr[:4].mean(0)
    print(f"{'L1 only (1-4)':>14} {l1[0]:>8.4f} {l1[1]:>9.4f} {l1[2]:>8.4f} "
          f"{l1[3]:>8.4f} {(auc0 - l1[0]) / (auc0 - 0.5):>14.1%}")

    # Witvliet graphs are much sparser than Cook; inferring on Cook thinned to
    # each animal's synapse count isolates the effect of density on the criterion.
    def thin(M, budget, r):
        w = M.ravel()
        tot = w.sum()
        if tot <= budget:
            return M.copy()
        return r.multinomial(budget, w / tot).reshape(M.shape).astype(float)

    Asub = A * np.outer(pmask, pmask)          # thin only the pool's own subgraph
    thin_rows = []
    for st_name, Aw in zip(stages, wmats):
        acc = []
        for t in range(5):
            r = np.random.Generator(np.random.PCG64(args.seed + 100 * t))
            At = thin(Asub, Aw.sum(), r) + A * ~np.outer(pmask, pmask)
            acc.append(score(infer(At, cook_allowed, m)))
        thin_rows.append(np.mean(acc, axis=0))
    tm = np.mean(thin_rows, axis=0)
    print(f"{'Cook, thinned':>14} {tm[0]:>8.4f} {tm[1]:>9.4f} {tm[2]:>8.4f} {tm[3]:>8.4f} "
          f"{(auc0 - tm[0]) / (auc0 - 0.5):>14.1%}   (to each animal's budget)")
    rows.append(("cook_thinned", *tm))

    for name, gen in (("biological", biological_pairs), ("random", random_pairs)):
        acc = np.array([score(gen()) for _ in range(args.n_null)])
        mu, sd = acc.mean(0), acc.std(0)
        print(f"{name:>14} {mu[0]:>8.4f} {mu[1]:>9.4f} {mu[2]:>8.4f} {mu[3]:>8.4f} "
              f"{(auc0 - mu[0]) / (auc0 - 0.5):>14.1%}   (sd {sd[0]:.4f}/{sd[1]:.4f})")
        rows.append((name, mu[0], mu[1], mu[2], mu[3]))
        if name == "random":
            rand_auc, rand_sd = mu[0], sd[0]
            rand_t, rand_tsd = mu[1], sd[1]
        else:
            bio_auc, bio_t = mu[0], mu[1]

    print(f"\nfunctional headroom relative to the matched random arm:")
    for nm, a, t in (("Cook-inferred", rows[0][1], rows[0][2]),
                     ("lifted, mean", mm[0], mm[1]),
                     ("lifted, L1 only", l1[0], l1[1]),
                     ("biological", bio_auc, bio_t)):
        print(f"  {nm:>16}: all pairs {(a - rand_auc) / rand_sd:+5.1f} sd, "
              f"in-pool pairs {(t - rand_t) / rand_tsd:+5.1f} sd above random")

    r_ = stats.spearmanr([x[4] for x in lifted], [x[1] for x in lifted])
    print(f"\nacross the eight lifted partitions, structural loss vs functional AUC: "
          f"rho {r_.statistic:+.3f}, p {r_.pvalue:.3f}")
    same = [np.mean([cls[nodes[a_]] == cls[nodes[b_]] for a_, b_ in x[5]]) for x in lifted]
    cs = np.bincount(np.unique([cls[nodes[i]] for i in pool_cook],
                               return_inverse=True)[1])
    ch = (cs * (cs - 1) // 2).sum() / (pool_cook.size * (pool_cook.size - 1) / 2)
    print(f"lifted merges that are same-class in the adult: {np.mean(same):.1%} "
          f"(L1 only {np.mean(same[:4]):.1%}) against chance {ch:.1%}")

    out = ROOT / f"results/functional_transfer_{args.pool}_m{m}.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "auc", "auc_inpool", "auc_nosyn", "struct_loss"])
        for r in rows:
            w.writerow([r[0]] + [f"{v:.6f}" for v in r[1:]])
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
