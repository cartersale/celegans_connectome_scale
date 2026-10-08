"""Split the full worm path operator into block means and residual for a set of
coarse partitions, and score each part against every observable; then compare
spatial and operator-defined partitions at k = 9.

For a partition P,

    R = R_block + R_residual,   R_block = PRP (group-pair means, diagonal excluded;
                                               conres.observables.block_means)

Scored for each partition and observable:

  block          R_block alone, also as a fraction of the full operator with a
                 stimulated-class bootstrap interval
  residual       R - R_block, pooled over all pairs (mixes blocks with different
                 base rates)
  within-block   R scored inside each (source group, target group) block of
                 >= 30 pairs and pair-weighted; a block-constant predictor scores
                 nothing here by construction, so this measures neuron-level
                 ranking information finer than P. Intervals (first 200
                 resamples) for partitions with k <= 20.
  quotient       the class-level path operator on P (randi.class_leverage)

Partitions: envelope_9 from results/coarse_trajectory_paths.npz (loss-greedy, k = 9),
k-means on log-scaled rows and columns of R (k = 9), k-means on soma position
(k = 9), the a priori subsystems, divisions and cell types of conres.subsystems,
the cell classes, and --n-rand permutations of the loss-greedy partition (block
predictor only).

The spatial comparison matches spatial k = 9 groups to each of loss-greedy,
operator k-means and subsystems by Hungarian assignment on the contingency
table, reports ARI and the fraction of neurons in matched groups, scores
POWERED observables separately on pairs with both endpoints matched and on the
rest, scores the meet (intersection) partition, and scores each partition's
block predictor inside the other's blocks.

Inputs:  conres.observables.load(); results/coarse_trajectory_paths.npz.
Outputs: results/block_decomposition.csv
Usage:   python scripts/block_decomposition.py [--n-boot 500] [--n-rand 20]
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ari

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from conres import observables as ob, randi, subsystems as ss, worm   # noqa: E402

ALPHA = 0.1


def relabel(l):
    return np.unique(l, return_inverse=True)[1]


def operator_kmeans(R, k, seed=0):
    X = np.log1p(np.hstack([R, R.T]) / R.max() * 1e3)
    return relabel(KMeans(k, n_init=10, random_state=seed).fit(X).labels_)


def spatial_kmeans(nodes, k, seed=0):
    """k-means on z-scored soma position; cells without a soma (the muscles anal,
    sph) are placed at the mean of the three somata with the largest y."""
    xyz = worm.soma_xyz(nodes)
    m = ~np.isfinite(xyz).all(1)
    post = np.argsort(np.where(m, -np.inf, xyz[:, 1]))[-3:]
    xyz[m] = xyz[post].mean(0)
    X = (xyz - xyz.mean(0)) / xyz.std(0)
    return relabel(KMeans(k, n_init=10, random_state=seed).fit(X).labels_)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--n-rand", type=int, default=20)
    args = ap.parse_args()

    D = ob.load()
    A, nodes, cls, obs = D["A"], D["nodes"], D["cls"], D["obs"]
    R = randi.resolvent(A, ALPHA)
    boots = ob.boot_indices(obs, cls, args.n_boot)
    full = {o: ob.score(k, ob.predict(k, R, j, i), v) for o, (j, i, v, k) in obs.items()}
    full_boot = {o: np.array([ob.score(k, ob.predict(k, R, j, i), v, s) for s in boots[o]])
                 for o, (j, i, v, k) in obs.items()}

    greedy9 = relabel(np.load(ROOT / "results/coarse_trajectory_paths.npz")["envelope_9"])
    parts = {
        "loss-greedy k=9": greedy9,
        "operator k-means k=9": operator_kmeans(R, 9),
        "spatial k-means k=9": spatial_kmeans(nodes, 9),
        "subsystems (a priori)": ss.partition(nodes, "subsystems")[0],
        "divisions (a priori)": ss.partition(nodes, "divisions")[0],
        "cell types (a priori)": ss.partition(nodes, "types")[0],
        "cell classes": cls,
    }

    rows = []
    print(f"full operator: " + "  ".join(f"{o} {full[o]:.4f}" for o in obs))
    for pname, lab in parts.items():
        k = int(lab.max() + 1)
        B = ob.block_means(R, lab)
        Rres = R - B
        Q = randi.class_leverage(A, lab, ALPHA)
        for o, (j, i, v, kind) in obs.items():
            pb, pr, pq = (ob.predict(kind, M, j, i) for M in (B, Rres, Q))
            pf = ob.predict(kind, R, j, i)
            sb, sr, sq = ob.score(kind, pb, v), ob.score(kind, pr, v), ob.score(kind, pq, v)
            wb = ob.within_block(kind, pf, v, lab[j], lab[i])
            fb = full_boot[o]
            bb = np.array([ob.score(kind, pb, v, s) for s in boots[o]])
            rb = np.array([ob.score(kind, pr, v, s) for s in boots[o]])
            row = {"partition": pname, "k": k, "observable": o, "full": full[o],
                   "block": sb, "block_frac": sb / full[o],
                   "block_frac_lo": np.nanpercentile(bb / fb, 2.5),
                   "block_frac_hi": np.nanpercentile(bb / fb, 97.5),
                   "residual": sr, "residual_lo": np.nanpercentile(rb, 2.5),
                   "residual_hi": np.nanpercentile(rb, 97.5),
                   "within_block": wb, "quotient": sq, "quotient_frac": sq / full[o]}
            rows.append(row)
        if pname == "loss-greedy k=9":
            rnd = []
            rng = np.random.default_rng(0)
            for _ in range(args.n_rand):
                rl = rng.permutation(lab)
                Br = ob.block_means(R, rl)
                rnd.append({o: ob.score(kd, ob.predict(kd, Br, j, i), v) / full[o]
                            for o, (j, i, v, kd) in obs.items()})
            for o in obs:
                vals = [r[o] for r in rnd]
                rows.append({"partition": "random (size-matched to loss-greedy k=9)", "k": k,
                             "observable": o, "full": full[o],
                             "block": np.mean(vals) * full[o], "block_frac": np.mean(vals),
                             "block_frac_lo": np.percentile(vals, 2.5),
                             "block_frac_hi": np.percentile(vals, 97.5)})
        print(f"  scored {pname}", flush=True)

    # within-block intervals for the coarse partitions, from the same
    # stimulated-class resamples
    for r in rows:
        if r["partition"] in parts and r["k"] <= 20:
            lab = parts[r["partition"]]
            j, i, v, kind = obs[r["observable"]]
            pf = ob.predict(kind, R, j, i)
            wb = [ob.within_block(kind, pf[s], v[s], lab[j[s]], lab[i[s]])
                  for s in boots[r["observable"]][:200]]
            r["within_block_lo"], r["within_block_hi"] = np.nanpercentile(wb, [2.5, 97.5])

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results/block_decomposition.csv", index=False)

    for o in obs:
        print(f"\n== {o}  (full operator {full[o]:.4f}; performance above chance, fraction "
              f"of full in brackets)")
        print(f"{'partition':>42} {'k':>4} {'block [95% CI of fraction]':>34} {'residual':>22} "
              f"{'within-block':>24} {'quotient':>14}")
        for _, r in df[df.observable == o].iterrows():
            res = (f"{r.residual:+.4f} [{r.residual_lo:+.3f},{r.residual_hi:+.3f}]"
                   if pd.notna(r.get("residual")) else "")
            wbk = (f"{r.within_block:+.4f}" + (f" [{r.within_block_lo:+.3f},{r.within_block_hi:+.3f}]"
                                               if pd.notna(r.get("within_block_lo")) else "")
                   if pd.notna(r.get("within_block")) else "")
            q = f"{r.quotient:+.4f}" if pd.notna(r.get("quotient")) else ""
            print(f"{r.partition:>42} {int(r.k):>4} {r.block:+.4f} ({r.block_frac:.2f} "
                  f"[{r.block_frac_lo:.2f},{r.block_frac_hi:.2f}]) {res:>24} {wbk:>26} {q:>12}")

    # ---------------------------------------------------------------- spatial vs operator
    print("\n== spatial versus operator partitions (k = 9)")
    spa = parts["spatial k-means k=9"]
    for oname in ("loss-greedy k=9", "operator k-means k=9", "subsystems (a priori)"):
        op = parts[oname]
        C = np.zeros((spa.max() + 1, op.max() + 1))
        np.add.at(C, (spa, op), 1)
        rr, cc = linear_sum_assignment(-C)
        match = dict(zip(rr, cc))
        agree = np.array([match.get(s, -1) == o for s, o in zip(spa, op)])
        print(f"\n  spatial vs {oname}: ARI {ari(spa, op):.3f}; "
              f"{agree.mean():.0%} of neurons in matched groups")
        Bs, Bo = ob.block_means(R, spa), ob.block_means(R, op)
        meet = relabel(spa * (op.max() + 1) + op)
        Bm = ob.block_means(R, meet)
        for o in ob.POWERED:
            j, i, v, kind = obs[o]
            both = agree[j] & agree[i]
            line = []
            for nm, M in (("full", R), ("spatial", Bs), ("operator", Bo), ("meet", Bm)):
                p = ob.predict(kind, M, j, i)
                line.append(f"{nm} {ob.score(kind, p, v, both) / ob.score(kind, ob.predict(kind, R, j, i), v, both):.2f}"
                            f"/{ob.score(kind, p, v, ~both) / ob.score(kind, ob.predict(kind, R, j, i), v, ~both):.2f}")
            p_o_in_s = ob.within_block(kind, ob.predict(kind, Bo, j, i), v, spa[j], spa[i])
            p_s_in_o = ob.within_block(kind, ob.predict(kind, Bs, j, i), v, op[j], op[i])
            print(f"    {o:>14}: fraction of full, agreeing pairs ({both.mean():.0%}) / "
                  f"disagreeing pairs: " + "  ".join(line))
            print(f"    {'':>14}  meet partition k={meet.max() + 1}: "
                  f"{ob.score(kind, ob.predict(kind, Bm, j, i), v) / full[o]:.2f} of full; "
                  f"operator blocks inside spatial blocks {p_o_in_s:+.4f}; "
                  f"spatial blocks inside operator blocks {p_s_in_o:+.4f}")
        if oname == "loss-greedy k=9":
            names = np.array(nodes)
            dis = np.flatnonzero(~agree)
            types = ss.cell_types(nodes)
            print(f"    disagreeing neurons ({dis.size}): " +
                  ", ".join(f"{names[x]}[{types[x][:12]}]" for x in dis[:40]))
    print("\nwrote results/block_decomposition.csv")


if __name__ == "__main__":
    main()
