"""Repeat the key C. elegans results with the class-level path operator
normalized by the full connectome's spectral radius instead of the quotient's own.

The worm analyses build the class-level operator on the quotient
Q = D^-1 H^T A H and normalize by Q's own spectral radius,

    own:    R_Q = (I - alpha Q / rho_Q)^-1 - I,

whereas the fly analyses normalize by the spectral radius of the full
connectome,

    fixed:  R_Q = (I - alpha Q / rho_A)^-1 - I.

For an exactly equitable partition A H = H Q, so the Perron root of A is an
eigenvalue of Q and rho_Q = rho_A; the two differ only as far as the partition
departs from equitability. alpha = 0.1 throughout.

  ratios     rho_Q / rho_A for cell classes, lineage-only cell classes
             (supp_checks.lineage_only), the merge criterion, loss-guided k = 20
             and k = 9, Ward and k-means k = 9; each against 50 label
             permutations of the same partition
  1          cell classes and lineage-only cell classes under both
             normalizations: fraction of the full operator's performance for
             every observable, and structural loss (1 - Spearman with R over
             observed pairs and over all ordered pairs), against --n-rand
             size-matched random partitions (label permutations)
  2          the merge criterion: 60 disjoint merges selected greedily by the
             symmetrized pairwise equitability defect among neurons at or above
             median total degree (prospective_merges), against --n-rand
             sets of 60 random disjoint pairs from the same neurons
  3          operator-defined boundary under each normalization: loss at
             K_GRID = 20 ... 4 for loss-guided agglomeration, Ward and k-means
             (--n-seed initializations) on operator profiles. Under "own" the
             loss-guided partitions are read from
             results/coarse_trajectory_paths.npz; under "fixed" they are rebuilt
             (greedy_path, Ward shortlist 30 to k = 20, then exhaustive) with
             the fixed normalization in the structural-loss objective. Ward and
             k-means partitions do not depend on the normalization. Detection
             uses the lower bound of a --n-boot cluster-bootstrap interval for
             ob.POWERED observables, with the boundary rule of
             alpha_sensitivity.py.

Produces the Supplementary Information section and table on spectral
normalization (C. elegans part).

Inputs: conres.observables.load, results/coarse_trajectory_paths.npz,
    complexity_curve, prospective_merges, supp_checks.
Outputs: results/normalization_rho_ratios.csv, results/normalization_partitions.csv,
    results/normalization_boundary.csv.
Usage: python scripts/normalization_control.py [--n-rand 100] [--n-boot 300] [--n-seed 3]
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from conres import observables as ob, randi   # noqa: E402
from complexity_curve import greedy_path, relabel   # noqa: E402
from prospective_merges import greedy_disjoint, labels_from_pairs, pairwise_defect   # noqa: E402
from supp_checks import lineage_only   # noqa: E402

ALPHA = 0.1
K_GRID = (20, 15, 12, 10, 9, 8, 7, 6, 5, 4)


def spectral_radius(M):
    return float(np.max(np.abs(np.linalg.eigvals(M))))


def class_op(A, lab, rho=None, alpha=ALPHA):
    """Class-level path operator; rho=None uses the quotient's own spectral radius."""
    Q = randi.quotient(A, lab)
    r = spectral_radius(Q) if rho is None else rho
    RQ = np.linalg.inv(np.eye(len(Q)) - alpha * Q / r) - np.eye(len(Q))
    return RQ[np.ix_(lab, lab)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-rand", type=int, default=100)
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--n-seed", type=int, default=3)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    D = ob.load()
    A, nodes, cls, obs = D["A"], D["nodes"], D["cls"], D["obs"]
    n = len(nodes)
    rhoA = spectral_radius(A)
    R = randi.resolvent(A, ALPHA)
    boots = ob.boot_indices(obs, cls, args.n_boot)
    full = {o: ob.score(kd, ob.predict(kd, R, j, i), v) for o, (j, i, v, kd) in obs.items()}
    fb = {o: np.array([ob.score(kd, ob.predict(kd, R, j, i), v, s) for s in boots[o]])
          for o, (j, i, v, kd) in obs.items()}
    j0, i0, _, _ = obs["incidence_all"]
    off = ~np.eye(n, dtype=bool)
    oj, oi = np.nonzero(off)
    rf_all = stats.rankdata(R[oj, oi])
    norms = {"own": None, "fixed": rhoA}

    def frac(lab, rho):
        M = class_op(A, lab, rho)
        out = {o: ob.score(kd, ob.predict(kd, M, j, i), v) / full[o] for o, (j, i, v, kd) in obs.items()}
        out["structural_observed"] = 1 - stats.spearmanr(M[j0, i0], R[j0, i0]).statistic
        out["structural_all"] = 1 - stats.spearmanr(M[oj, oi], rf_all).statistic
        return out

    # ------------------------------------------------ spectral radius ratios
    greedy = np.load(ROOT / "results/coarse_trajectory_paths.npz")
    Xop = np.log1p(np.hstack([R, R.T]) / R.max() * 1e3)
    tree = linkage(Xop, method="ward")
    deg = (A > 0).sum(1) + (A > 0).sum(0)
    allowed = deg >= np.median(deg)
    g = (pairwise_defect(A) + pairwise_defect(A.T)) / 2.0
    crit = labels_from_pairs(n, greedy_disjoint(g, 60, allowed))
    cand = np.flatnonzero(allowed)

    def random_disjoint():
        perm = rng.permutation(cand)[:120]
        return labels_from_pairs(n, [(int(perm[2 * q]), int(perm[2 * q + 1])) for q in range(60)])

    named = {"cell classes": cls, "lineage-only cell classes": lineage_only(nodes),
             "merge criterion, 60 merges": crit,
             "loss-guided k=20": relabel(greedy["envelope_20"]),
             "loss-guided k=9": relabel(greedy["envelope_9"]),
             "Ward on operator k=9": relabel(fcluster(tree, t=9, criterion="maxclust")),
             "k-means on operator k=9": relabel(KMeans(9, n_init=10, random_state=0).fit(Xop).labels_)}
    rat = []
    print(f"rho_A = {rhoA:.3f}")
    for name, lab in named.items():
        r = spectral_radius(randi.quotient(A, lab)) / rhoA
        rnd = [spectral_radius(randi.quotient(A, rng.permutation(lab))) / rhoA for _ in range(50)]
        rat.append({"partition": name, "k": int(lab.max() + 1), "ratio": r,
                    "random_ratio_mean": np.mean(rnd), "random_ratio_lo": np.percentile(rnd, 2.5),
                    "random_ratio_hi": np.percentile(rnd, 97.5)})
        print(f"  rho_Q / rho_A  {name:>28} (k={lab.max() + 1:3d}): {r:.3f};  size-matched random "
              f"{np.mean(rnd):.3f} [{np.percentile(rnd, 2.5):.3f}, {np.percentile(rnd, 97.5):.3f}]")
    pd.DataFrame(rat).to_csv(ROOT / "results/normalization_rho_ratios.csv", index=False)

    # ------------------------------------------- 1-2. cell classes, lineage-only, criterion
    rows = []
    for name in ("cell classes", "lineage-only cell classes", "merge criterion, 60 merges"):
        lab = named[name]
        for norm, rho in norms.items():
            f = frac(lab, rho)
            rl = [random_disjoint() if name.startswith("merge") else rng.permutation(lab)
                  for _ in range(args.n_rand)]
            rf = [frac(x, rho) for x in rl]
            for key in f:
                vals = np.array([x[key] for x in rf])
                rows.append({"partition": name, "normalization": norm, "measure": key,
                             "value": f[key], "random_mean": vals.mean(), "random_sd": vals.std(),
                             "z": (f[key] - vals.mean()) / vals.std()})
            print(f"\n  {name} [{norm} normalization]: " + ", ".join(
                f"{k} {f[k]:.3f} (random {np.mean([x[k] for x in rf]):.3f})" for k in f), flush=True)
    pd.DataFrame(rows).to_csv(ROOT / "results/normalization_partitions.csv", index=False)

    # ------------------------------------------------ 3. operator-defined boundary
    def struct_loss_fixed(lab):
        rq = stats.rankdata(class_op(A, lab, rhoA)[oj, oi])
        return float(1 - stats.pearsonr(rq, rf_all).statistic)

    print("\n  rebuilding loss-guided agglomeration under the fixed normalization ...", flush=True)
    lg_fixed = greedy_path(A, 20, loss_fn=struct_loss_fixed, shortlist=30)
    lab = relabel(lg_fixed[20])
    while lab.max() + 1 > min(K_GRID):
        k = lab.max() + 1
        best, best_l = None, np.inf
        for a in range(k):
            for b in range(a + 1, k):
                c = relabel(np.where(lab == b, a, lab))
                l_ = struct_loss_fixed(c)
                if l_ < best_l:
                    best, best_l = c, l_
        lab = best
        lg_fixed[int(lab.max() + 1)] = lab.copy()

    def build(method, k, norm):
        if method == "loss-guided":
            return [relabel(greedy[f"envelope_{k}"]) if norm == "own" else relabel(lg_fixed[k])]
        if method == "Ward on operator":
            return [relabel(fcluster(tree, t=k, criterion="maxclust"))]
        return [relabel(KMeans(k, n_init=10, random_state=s).fit(Xop).labels_) for s in range(args.n_seed)]

    curves = []
    for norm, rho in norms.items():
        for method in ("loss-guided", "Ward on operator", "k-means on operator"):
            for k in K_GRID:
                labs = build(method, k, norm)
                r = {"normalization": norm, "method": method, "k": k}
                for o in ob.POWERED:
                    j, i, v, kd = obs[o]
                    ls = []
                    for q, lb in enumerate(labs):
                        p = ob.predict(kd, class_op(A, lb, rho), j, i)
                        ls.append((full[o] - ob.score(kd, p, v)) / full[o])
                        if q == 0:
                            bl = np.array([(f - ob.score(kd, p, v, s)) / f for f, s in zip(fb[o], boots[o])])
                            r[o + "_lo"], r[o + "_hi"] = np.nanpercentile(bl, [2.5, 97.5])
                    r[o] = float(np.mean(ls))
                curves.append(r)
            d = pd.DataFrame([c for c in curves if c["normalization"] == norm and c["method"] == method])
            det = d.set_index("k")[[o + "_lo" for o in ob.POWERED]].max(axis=1) > 0
            ks = sorted(det.index, reverse=True)
            boundary = next((k for q, k in enumerate(ks) if all(det[kk] for kk in ks[q:])), None)
            l9 = d.set_index("k").loc[9]
            print(f"  [{norm}] {method:>20}: sustained loss from k = {boundary}; k = 9 loss: propagation "
                  f"{l9['incidence_all']:+.2f} [{l9['incidence_all_lo']:+.2f}, {l9['incidence_all_hi']:+.2f}], "
                  f"correlation {l9['correlation']:+.2f} [{l9['correlation_lo']:+.2f}, "
                  f"{l9['correlation_hi']:+.2f}]", flush=True)
    pd.DataFrame(curves).to_csv(ROOT / "results/normalization_boundary.csv", index=False)
    print("\nwrote results/normalization_{rho_ratios,partitions,boundary}.csv")


if __name__ == "__main__":
    main()
