"""Structural and functional loss of the worm path operator as a function of
partition size k, for biological, connectivity-greedy, loss-greedy, spatial and
random partitions of the Cook adult connectome, and the resulting k_eps.

k_eps is read from the fine end downward,

    k_eps = the smallest k such that L(k') <= eps at every k' >= k,

so a noisy curve that dips back under eps after exceeding it does not count
(the literal min{k : L(k) <= eps} would be set by a single lucky rung).

Arms, each scored at every k it reaches:

  bio            the 119 cell classes merged one neuron at a time in random class
                 order (--n-bio trajectories); defined only for k >= 119
  ward           connectivity-only Ward agglomeration (the criterion of
                 `label_free_merge_trajectory.py`), run from singletons to --k-min
  envelope       loss-greedy agglomeration: at each step the --shortlist cheapest
                 Ward candidates are scored by the structural loss (all ordered
                 pairs) and the best is taken. It optimises the metric it is
                 scored on, so it is a heuristic lower envelope, optimistic by
                 construction.
  spatial        k-means on z-scored soma position (lineage-blind), at K_GRID
  random_<arm>   each arm's labels permuted (--n-rand draws), holding its
                 cell-size profile, at K_GRID

Losses, with the quotient operator lifted back to neurons (alpha = 0.1):
  struct_all     1 - Spearman with the full resolvent over all ordered pairs
  struct_occ     the same over occ3-observed pairs
  func_all       AUC shortfall on the Randi wild-type incidence, all observed pairs
  func_nosyn     the same on pairs with no direct synapse
  *_frac         shortfall as a fraction of the full operator's headroom above 0.5
No partition is built from functional data.

At k = 119 the bio, ward and envelope partitions are compared with 95% intervals:
functional loss by a cluster bootstrap over stimulated class (seed 7), structural
loss by a bootstrap over source neurons (seed 8, at most 300 resamples).

`anal` and `sph` are muscles in the Cook node set and have no soma position; the
spatial arm places them at the mean of the three somata with the largest y.

Inputs:  Cook connectome and Randi atlas via conres; worm soma positions.
Outputs: results/complexity_curve.csv  (arm, k, rep, losses)
         results/complexity_curve_k_eps.csv             (k_eps per metric, arm, eps)
         results/complexity_curve_paths.npz      (ward_<k>, envelope_<k> labels, nodes;
                                              read by fig16, fig17, fig18)
Usage:   python scripts/complexity_curve.py [--k-min 20] [--shortlist 30]
             [--n-bio 20] [--n-rand 10] [--n-boot 1000] [--seed 0]
"""
from __future__ import annotations

import argparse
import csv
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from conres import celegans as ce, randi, worm               # noqa: E402
from label_free_merge_trajectory import cell_profiles, ward_costs  # noqa: E402

ALPHA = 0.1
K_GRID = (290, 275, 260, 242, 220, 200, 182, 160, 140, 119, 100, 80, 60, 40, 30, 20)
EPS_STRUCT = (0.05, 0.10, 0.20, 0.30)
EPS_FUNC = (0.05, 0.10, 0.25, 0.50)          # fractions of no-synapse headroom


def relabel(lab):
    return np.unique(lab, return_inverse=True)[1]


def ward_matrix(A, lab):
    Qo, sz = cell_profiles(A, lab)
    Qi, _ = cell_profiles(A.T, lab)
    return (ward_costs(Qo, sz) + ward_costs(Qi, sz)) / 2.0


def greedy_path(A, k_min, loss_fn=None, shortlist=1):
    """Agglomerate from singletons to k_min; return {k: labels}.

    Without loss_fn this is plain Ward agglomeration. With loss_fn, the
    `shortlist` cheapest Ward merges are proposed and loss_fn picks among them."""
    n = A.shape[0]
    lab = np.arange(n)
    path = {n: lab.copy()}
    while lab.max() + 1 > k_min:
        C = ward_matrix(A, lab)
        iu = np.triu_indices_from(C, 1)
        cost = C[iu]
        order = np.argsort(cost)[:shortlist]
        best, best_l = None, np.inf
        for o in order:
            a, b = iu[0][o], iu[1][o]
            cand = relabel(np.where(lab == b, a, lab))
            if loss_fn is None:
                best = cand
                break
            l_ = loss_fn(cand)
            if l_ < best_l:
                best, best_l = cand, l_
        lab = best
        path[int(lab.max() + 1)] = lab.copy()
        print(f"    k = {lab.max() + 1:3d}", end="\r", flush=True)
    print()
    return path


def bio_path(labels, rng):
    """Classes merged one neuron at a time in random class order; {k: labels}."""
    n = labels.size
    classes = [np.flatnonzero(labels == c) for c in np.unique(labels)]
    classes = [c for c in classes if c.size > 1]
    order = rng.permutation(len(classes))
    cell = np.arange(n)
    path = {n: relabel(cell)}
    for oi in order:
        grp = rng.permutation(classes[oi])
        for m in grp[1:]:
            cell[cell == cell[m]] = cell[grp[0]]
            lab = relabel(cell)
            path[int(lab.max() + 1)] = lab
    return path


def spatial_labels(xyz, k, seed=0):
    X = (xyz - xyz.mean(0)) / xyz.std(0)
    return relabel(KMeans(n_clusters=k, n_init=4, random_state=seed).fit(X).labels_)


def k_eps(ks, losses, eps):
    """Smallest k such that loss <= eps at every k' >= k; None if the finest k fails."""
    ok = None
    for k, l_ in sorted(zip(ks, losses), key=lambda t: -t[0]):
        if not np.isfinite(l_) or l_ > eps:
            break
        ok = k
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k-min", type=int, default=20)
    ap.add_argument("--shortlist", type=int, default=30)
    ap.add_argument("--n-bio", type=int, default=20)
    ap.add_argument("--n-rand", type=int, default=10)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    A, nodes = ce.build_matrix("all", neurons_only=True)
    n = A.shape[0]
    labels = relabel(ce.partitions(nodes)["class"])
    idx = {x: i for i, x in enumerate(nodes)}
    occ3, _ = randi.build_occ3(nodes, "wt")
    d = randi.load_responses("wildty")
    d = d[d.pre.isin(idx) & d.post.isin(idx)]
    hits = {(idx[p], idx[q]) for p, q in zip(d.pre, d.post)}
    ii, jj = np.nonzero(occ3)
    y = np.array([1.0 if (j, i) in hits else 0.0 for i, j in zip(ii, jj)])
    ns = A[jj, ii] == 0
    off = ~np.eye(n, dtype=bool)
    aj, ai = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    oj, oi = aj[off], ai[off]

    R = randi.resolvent(A, ALPHA)
    f_all, f_occ = R[oj, oi], R[jj, ii]
    auc0, auc0_ns = roc_auc_score(y, f_occ), roc_auc_score(y[ns], f_occ[ns])
    head, head_ns = auc0 - 0.5, auc0_ns - 0.5

    def lifted(lab):
        return randi.resolvent(randi.quotient(A, lab), ALPHA)[np.ix_(lab, lab)]

    def struct_all(lab):
        return 1 - stats.spearmanr(lifted(lab)[oj, oi], f_all).statistic

    def score(lab):
        S = lifted(lab)
        q = S[jj, ii]
        la, ln = auc0 - roc_auc_score(y, q), auc0_ns - roc_auc_score(y[ns], q[ns])
        return {"struct_all": 1 - stats.spearmanr(S[oj, oi], f_all).statistic,
                "struct_occ": 1 - stats.spearmanr(q, f_occ).statistic,
                "func_all": la, "func_nosyn": ln,
                "func_all_frac": la / head, "func_nosyn_frac": ln / head_ns}

    xyz = worm.soma_xyz(nodes)
    miss = ~np.isfinite(xyz).all(1)
    post = np.argsort(np.where(miss, -np.inf, xyz[:, 1]))[-3:]
    xyz[miss] = xyz[post].mean(0)

    print(f"worm: {n} cells, {labels.max() + 1} classes; full operator AUC {auc0:.4f}, "
          f"no-synapse {auc0_ns:.4f}; {int(miss.sum())} cells without soma imputed")

    rng = np.random.default_rng(args.seed)
    print("  ward-greedy path")
    ward = greedy_path(A, args.k_min)
    print(f"  loss-greedy path (shortlist {args.shortlist})")
    env = greedy_path(A, args.k_min, loss_fn=struct_all, shortlist=args.shortlist)

    rows = []

    def emit(arm, k, rep, lab):
        rows.append({"arm": arm, "k": k, "rep": rep, **score(lab)})

    for k in sorted(ward, reverse=True):
        emit("ward", k, 0, ward[k])
        emit("envelope", k, 0, env[k])
    print("  scored greedy paths")
    for r in range(args.n_bio):
        for k, lab in bio_path(labels, np.random.default_rng(args.seed + r)).items():
            emit("bio", k, r, lab)
    print("  scored biological trajectories")
    for k in K_GRID:
        if k < args.k_min:
            continue
        sp = spatial_labels(xyz, k, args.seed)
        emit("spatial", k, 0, sp)
        srcs = [("ward", ward[k]), ("envelope", env[k]), ("spatial", sp)]
        if k >= labels.max() + 1:
            srcs.append(("bio", bio_path(labels, np.random.default_rng(args.seed))[k]))
        for src, lab in srcs:
            for b in range(args.n_rand):
                emit(f"random_{src}", k, b, rng.permutation(lab))
    print("  scored spatial and random arms")

    out = ROOT / "results/complexity_curve.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)")

    import pandas as pd
    df = pd.DataFrame(rows)
    mean = df.groupby(["arm", "k"]).mean(numeric_only=True).reset_index()

    def at(arm, k, col):
        v = mean[(mean.arm == arm) & (mean.k == k)][col]
        return float(v.iloc[0]) if len(v) else np.nan

    arms = ("bio", "ward", "envelope", "spatial", "random_bio", "random_ward",
            "random_envelope", "random_spatial")
    for col, title in (("struct_all", "structural loss, all ordered pairs"),
                       ("struct_occ", "structural loss, occ3 pairs"),
                       ("func_nosyn_frac", "functional loss, fraction of no-synapse headroom")):
        print(f"\n{title}")
        print(f"{'k':>5} {'N/k':>6} " + " ".join(f"{a:>15}" for a in arms))
        for k in K_GRID:
            if k < args.k_min:
                continue
            print(f"{k:>5} {n / k:>6.2f} " + " ".join(
                f"{at(a, k, col):>15.4f}" for a in arms))

    # -- k_eps per arm and metric, on the densest curve each arm has
    print("\nk_eps = smallest k keeping loss <= eps at every coarser-to-finer step above it")
    keps_rows = []
    for col, grid in (("struct_all", EPS_STRUCT), ("struct_occ", EPS_STRUCT),
                      ("func_nosyn_frac", EPS_FUNC), ("func_all_frac", EPS_FUNC)):
        for arm in ("bio", "ward", "envelope", "spatial", "random_bio", "random_ward",
                    "random_spatial"):
            sub = mean[mean.arm == arm].sort_values("k", ascending=False)
            if arm.startswith("random") or arm == "spatial":
                sub = pd.concat([pd.DataFrame({"k": [n], col: [0.0]}), sub])
            floor = int(sub.k.min())
            vals = []
            for e in grid:
                ke = k_eps(sub.k.to_numpy(), sub[col].to_numpy(), e)
                vals.append(ke)
                keps_rows.append({"metric": col, "arm": arm, "eps": e,
                                  "k_eps": ke if ke is not None else np.nan,
                                  "floor": floor,
                                  "reached_floor": ke == floor})
            print(f"  {col:>16} {arm:>15}: " + "  ".join(
                f"eps {e:.2f} -> {('<=' + str(v)) if v == floor else v}"
                for e, v in zip(grid, vals)))
    out_k = ROOT / "results/complexity_curve_k_eps.csv"
    pd.DataFrame(keps_rows).to_csv(out_k, index=False)
    print(f"wrote {out_k}")

    # -- bio, ward and envelope compared at k = number of classes, with intervals
    k = int(labels.max() + 1)
    if k not in ward:
        return
    parts = {"bio": labels, "ward": ward[k], "envelope": env[k]}
    lifts = {a: lifted(l_) for a, l_ in parts.items()}
    rng_b = np.random.default_rng(7)
    g = labels[jj]
    uq = np.unique(g)
    where = {c: np.flatnonzero(g == c) for c in uq}
    boots = {a: [] for a in parts}
    diffs = {"bio-envelope": [], "bio-ward": []}
    for _ in range(args.n_boot):
        s = np.concatenate([where[c] for c in rng_b.choice(uq, uq.size, replace=True)])
        s = s[ns[s]]
        if len(np.unique(y[s])) < 2:
            continue
        a_full = roc_auc_score(y[s], f_occ[s])
        v = {a: a_full - roc_auc_score(y[s], lifts[a][jj, ii][s]) for a in parts}
        for a in parts:
            boots[a].append(v[a])
        diffs["bio-envelope"].append(v["bio"] - v["envelope"])
        diffs["bio-ward"].append(v["bio"] - v["ward"])
    src = np.unique(oj)
    wsrc = {j: np.flatnonzero(oj == j) for j in src}
    sboot = {"bio-envelope": [], "bio-ward": []}
    q_all = {a: lifts[a][oj, oi] for a in parts}
    rng_s = np.random.default_rng(8)
    for _ in range(min(args.n_boot, 300)):
        s = np.concatenate([wsrc[j] for j in rng_s.choice(src, src.size, replace=True)])
        sv = {a: 1 - stats.spearmanr(q_all[a][s], f_all[s]).statistic for a in parts}
        sboot["bio-envelope"].append(sv["bio"] - sv["envelope"])
        sboot["bio-ward"].append(sv["bio"] - sv["ward"])
    ci = lambda v: f"[{np.percentile(v, 2.5):+.4f}, {np.percentile(v, 97.5):+.4f}]"
    print(f"\nidea 6 at k = {k} (N/k = {n / k:.2f})")
    print(f"{'arm':>10} {'struct all':>11} {'struct occ3':>12} {'func no-syn':>12} "
          f"{'95% CI':>20} {'headroom lost':>14} {'same-class co-cell pairs':>26}")
    mapping = labels
    for a, l_ in parts.items():
        sc = score(l_)
        same = tot = 0
        for c in range(l_.max() + 1):
            m = np.flatnonzero(l_ == c)
            if m.size > 1:
                eq = mapping[m][:, None] == mapping[m][None, :]
                same += (eq.sum() - m.size) // 2
                tot += m.size * (m.size - 1) // 2
        sizes = np.bincount(l_)
        print(f"{a:>10} {sc['struct_all']:>11.4f} {sc['struct_occ']:>12.4f} "
              f"{sc['func_nosyn']:>+12.4f} {ci(boots[a]):>20} "
              f"{sc['func_nosyn_frac']:>13.1%} {same:>6}/{tot:<6} = {same / max(tot, 1):>5.1%}"
              f"   largest cell {sizes.max()}")
    for key in diffs:
        print(f"  {key}: functional {ci(diffs[key])} (stimulated-class bootstrap); "
              f"structural (all pairs) {ci(sboot[key])} (source-neuron bootstrap)")
    np.savez_compressed(ROOT / "results/complexity_curve_paths.npz",
                        **{f"ward_{k_}": v for k_, v in ward.items()},
                        **{f"envelope_{k_}": v for k_, v in env.items()},
                        nodes=np.array(nodes))


if __name__ == "__main__":
    main()
