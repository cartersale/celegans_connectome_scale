"""Functional loss of the class-level path operator for six activity-blind
grouping methods on a common grid of small k, plus three anatomical reference
partitions, and the k at which loss first becomes detectable for each method.

Grid: k = 20, 15, 12, 10, 9, 8, 7, 6, 5, 4, 3, 2.

Methods:
  loss-greedy          envelope_<k> from results/coarse_trajectory_paths.npz (nested)
  Ward on operator     Ward clustering of log-scaled rows and columns of R (nested;
                       fcluster maxclust)
  k-means on operator  the same profiles, k-means, --n-seed seeds
  k-means on wiring    rows and columns of log(1 + A), k-means, --n-seed seeds
  spectral             spectral clustering of (A + A^T) / max, --n-seed seeds
  spatial              k-means on soma position, --n-seed seeds
Anatomical references: conres.subsystems divisions, subsystems and cell types.

At each (method, k), for every observable in conres.observables: loss
(s_full - s) / s_full averaged over seeds with the across-seed range; a paired
stimulated-class cluster-bootstrap interval (--n-boot) from seed 0; the mean
loss of --n-rand label permutations of the seed-0 partition; and the largest tie
block of the seed-0 predictor, since at small k the read-back has at most k^2
distinct values.

Transitions use the POWERED observables (incidence over all pairs, correlation):
"first" is the largest k at which either shows detectable loss (lower bound > 0);
"sustained" is the largest k such that loss is detectable at that k and every
smaller k on the grid.

Produces the grouping-method comparison table in the Supplementary Information.

Inputs:  conres.observables.load(); results/coarse_trajectory_paths.npz;
         spatial_kmeans and relabel from `block_decomposition.py`.
Outputs: results/grouping_methods.csv
Usage:   python scripts/grouping_methods.py [--n-seed 5] [--n-boot 300] [--n-rand 20]
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import KMeans, SpectralClustering

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from conres import observables as ob, randi, subsystems as ss        # noqa: E402
from conres.metric_guard import guard_predictor                      # noqa: E402
from block_decomposition import spatial_kmeans, relabel        # noqa: E402

ALPHA = 0.1
K_GRID = (20, 15, 12, 10, 9, 8, 7, 6, 5, 4, 3, 2)


def method_builders(A, nodes, R, n_seed):
    """Return builders(k), yielding (method, [labels per seed]) for the six methods.

    Deterministic methods (loss-greedy, Ward) give one partition; the k-means and
    spectral methods give one per seed 0..n_seed-1.
    """
    Xop = np.log1p(np.hstack([R, R.T]) / R.max() * 1e3)
    Xw = np.log1p(np.hstack([A, A.T]))
    W = A + A.T
    aff = W / W.max()
    ward_tree = linkage(Xop, method="ward")
    greedy = np.load(ROOT / "results/coarse_trajectory_paths.npz")

    def builders(k):
        yield "loss-greedy", [relabel(greedy[f"envelope_{k}"])]
        yield "Ward on operator", [relabel(fcluster(ward_tree, t=k, criterion="maxclust"))]
        yield "k-means on operator", [relabel(KMeans(k, n_init=10, random_state=s).fit(Xop).labels_)
                                      for s in range(n_seed)]
        yield "k-means on wiring", [relabel(KMeans(k, n_init=10, random_state=s).fit(Xw).labels_)
                                    for s in range(n_seed)]
        yield "spectral", [relabel(SpectralClustering(k, affinity="precomputed", random_state=s,
                                                      assign_labels="kmeans").fit(aff).labels_)
                           for s in range(n_seed)]
        yield "spatial", [spatial_kmeans(nodes, k, s) for s in range(n_seed)]

    return builders


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seed", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--n-rand", type=int, default=20)
    args = ap.parse_args()

    D = ob.load()
    A, nodes, cls, obs = D["A"], D["nodes"], D["cls"], D["obs"]
    R = randi.resolvent(A, ALPHA)
    boots = ob.boot_indices(obs, cls, args.n_boot)
    full = {o: ob.score(k, ob.predict(k, R, j, i), v) for o, (j, i, v, k) in obs.items()}
    full_boot = {o: np.array([ob.score(k, ob.predict(k, R, j, i), v, s) for s in boots[o]])
                 for o, (j, i, v, k) in obs.items()}

    builders = method_builders(A, nodes, R, args.n_seed)

    def evaluate(method, k, labs, rows):
        per_seed = []
        for lab in labs:
            Q = randi.class_leverage(A, lab, ALPHA)
            per_seed.append({o: (full[o] - ob.score(kd, ob.predict(kd, Q, j, i), v)) / full[o]
                             for o, (j, i, v, kd) in obs.items()})
        Q0 = randi.class_leverage(A, labs[0], ALPHA)
        rng = np.random.default_rng(k)
        rq = [randi.class_leverage(A, rng.permutation(labs[0]), ALPHA) for _ in range(args.n_rand)]
        for o, (j, i, v, kd) in obs.items():
            p0 = ob.predict(kd, Q0, j, i)
            lb = np.array([(fb - ob.score(kd, p0, v, s)) / fb
                           for fb, s in zip(full_boot[o], boots[o])])
            vals = [ps[o] for ps in per_seed]
            rows.append({"method": method, "k": int(labs[0].max() + 1), "observable": o,
                         "loss": float(np.mean(vals)), "seed_min": float(np.min(vals)),
                         "seed_max": float(np.max(vals)),
                         "lo": float(np.nanpercentile(lb, 2.5)),
                         "hi": float(np.nanpercentile(lb, 97.5)),
                         "random": float(np.mean([(full[o] - ob.score(kd, ob.predict(kd, M, j, i), v))
                                                  / full[o] for M in rq])),
                         "ties": guard_predictor(v if kd == "auc" else np.zeros(len(v)),
                                                 p0)["largest_tie_block"],
                         "largest_cell": int(np.bincount(labs[0]).max())})

    rows = []
    for k in K_GRID:
        for method, labs in builders(k):
            evaluate(method, k, labs, rows)
        print(f"  k = {k} done", flush=True)
    for level in ("divisions", "subsystems", "types"):
        lab = ss.partition(nodes, level)[0]
        evaluate(f"anatomy: {level}", int(lab.max() + 1), [lab], rows)
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results/grouping_methods.csv", index=False)

    methods = list(dict.fromkeys(df.method))
    for o in obs:
        print(f"\n== {o}: loss as fraction of full operator (mean over seeds), '*' detectable "
              f"(seed-0 interval), random mean for loss-greedy in last column")
        print(f"{'k':>4} " + " ".join(f"{m[:18]:>19}" for m in methods if not m.startswith("anatomy"))
              + f" {'random':>8}")
        for k in K_GRID:
            cells = []
            for m in methods:
                if m.startswith("anatomy"):
                    continue
                r = df[(df.method == m) & (df.k == k) & (df.observable == o)]
                if r.empty:
                    cells.append(f"{'':>19}")
                    continue
                r = r.iloc[0]
                cells.append(f"{r.loss:+.2f}{'*' if r.lo > 0 else ' '}[{r.lo:+.2f},{r.hi:+.2f}]")
            rnd = df[(df.method == "loss-greedy") & (df.k == k) & (df.observable == o)]["random"]
            print(f"{k:>4} " + " ".join(f"{c:>19}" for c in cells)
                  + f" {rnd.iloc[0] if len(rnd) else np.nan:>8.2f}")
        for _, r in df[df.method.str.startswith("anatomy") & (df.observable == o)].iterrows():
            print(f"     {r.method} (k = {r.k}): {r.loss:+.2f}{'*' if r.lo > 0 else ' '}"
                  f"[{r.lo:+.2f},{r.hi:+.2f}]  random {r.random:.2f}")

    print("\n== transitions (powered observables: incidence over all pairs, correlation)")
    for m in methods:
        if m.startswith("anatomy"):
            continue
        sub = df[(df.method == m) & df.observable.isin(ob.POWERED)]
        det = sub.groupby("k").lo.max() > 0          # detectable in either powered observable
        ks = sorted(det.index, reverse=True)
        first = next((k for k in ks if det[k]), None)
        sustained = None
        for k in ks:
            if all(det[k2] for k2 in ks if k2 <= k):
                sustained = k
                break
        ties = sub[sub.k == first].ties.max() if first else np.nan
        print(f"  {m:>20}: first detectable loss at k = {first}; sustained from k = {sustained}; "
              f"largest tie block at first loss {ties:.2f}")
    print("\nwrote results/grouping_methods.csv")


if __name__ == "__main__":
    main()
