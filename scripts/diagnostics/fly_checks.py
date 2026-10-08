"""Two diagnostics on the FlyWire fragility curve: a spatial null for the Tier B
rungs and a paired bootstrap of the two Tier A rungs.

Tiers follow preregistration/fly_subsplit_rule.md: Tier A is hemilineage
(optionally x side); Tier B sub-splits Tier A cells by Ward on soma position.

Spatial null. Tier B cells are spatially compact, and nearby neurons share
partners, so a compact but lineage-blind null is scored at matched k: k-means
(MiniBatchKMeans, seed 0) on z-scored soma position over the whole in-scope set.
k-means rather than Ward because a global Ward over ~32k neurons needs a ~4 GB
condensed distance matrix. Scored at f = 0.02, 0.05, 0.10, 0.20, 0.40 against
the biological partition and three size-matched random partitions (seeds 500+b).

Tier A monotonicity. Loss for hemilineage and hemilineage x side with 95%
intervals from 300 bootstrap resamples of sources, and the paired difference
(500 resamples; both partitions scored on identical sources). A negative
difference means the finer partition loses more.

Loss = 1 - Spearman(quotient values, full-operator values), alpha = 0.1, rows
normalized by the full connectome's spectral radius; 200 degree-stratified
sources x 400 targets (seed 0), sampled as in fly_fragility.py.

Inputs: FlyWire in-scope adjacency and metadata via conres.flywire; imports
pair_sample and rung_loss from fly_fragility.py.
Outputs: stdout only.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.cluster import MiniBatchKMeans

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conres import flywire as fw          # noqa: E402
from conres.paths import tee_stdout                   # noqa: E402
tee_stdout(__file__)
import fly_fragility as F            # noqa: E402

ALPHA = 0.1


def spatial_null(meta, k, seed=0):
    xyz = meta[["soma_x", "soma_y", "soma_z"]].to_numpy(float)
    xyz = (xyz - xyz.mean(0)) / xyz.std(0)
    km = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=3,
                         batch_size=4096).fit(xyz)
    return km.labels_.astype(np.int64)


def loss_with_ci(A, lab, sources, tgt, full_vals, rho, n_boot=300, seed=0):
    Q = fw.quotient(A, lab)
    rows = fw.propagation_rows(Q, lab[sources], ALPHA, rho=rho)
    qv = np.stack([rows[r][lab[tgt[r]]] for r in range(len(sources))])
    fv = full_vals.reshape(qv.shape)
    point = 1 - stats.spearmanr(qv.ravel(), fv.ravel()).statistic
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        pick = rng.integers(0, qv.shape[0], qv.shape[0])   # resample sources
        bs.append(1 - stats.spearmanr(qv[pick].ravel(), fv[pick].ravel()).statistic)
    bs = np.array(bs)
    return point, np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def paired_diff(A, lab_a, lab_b, sources, tgt, full_vals, rho, n_boot=500, seed=0):
    """Bootstrap the paired difference in loss between two partitions.

    Both are scored on identical sources, so sources are resampled jointly and
    differenced; returns (mean, 2.5th, 97.5th percentile).
    """
    def vals(lab):
        rows = fw.propagation_rows(fw.quotient(A, lab), lab[sources], ALPHA, rho=rho)
        return np.stack([rows[r][lab[tgt[r]]] for r in range(len(sources))])
    qa, qb = vals(lab_a), vals(lab_b)
    fv = full_vals.reshape(qa.shape)
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n_boot):
        p = rng.integers(0, qa.shape[0], qa.shape[0])
        la = 1 - stats.spearmanr(qa[p].ravel(), fv[p].ravel()).statistic
        lb = 1 - stats.spearmanr(qb[p].ravel(), fv[p].ravel()).statistic
        d.append(la - lb)
    d = np.array(d)
    return d.mean(), np.percentile(d, 2.5), np.percentile(d, 97.5)


def main():
    meta = fw.load_scope()
    A = fw.load_adjacency(meta)
    n = A.shape[0]
    rho = fw.spectral_radius(A)
    sources, per_dec = fw.stratified_sources(A, 200, 0)
    rng = np.random.default_rng(0)
    tgt = F.pair_sample(n, sources, 400, rng)
    full_rows = fw.propagation_rows(A, sources, ALPHA, rho=rho)
    full_vals = np.stack([full_rows[r][tgt[r]] for r in range(len(sources))]).ravel()
    print(f"in-scope {n:,} neurons, rho={rho:.0f}, {len(sources)} sources, "
          f"{len(sources)*400:,} pairs\n")

    print("== CHECK 2: Tier A monotonicity, bootstrapped over sources ==")
    sides = meta["side"].value_counts()
    print(f"  side values: {dict(sides)}")
    for name, ws in (("hemilineage", False), ("hemilineage x side", True)):
        lab = np.asarray(fw.tier_a(meta, ws), np.int64)
        k = int(lab.max() + 1)
        sz = np.bincount(lab)
        p, lo, hi = loss_with_ci(A, lab, sources, tgt, full_vals, rho)
        print(f"  {name:20s} k={k:4d} {n/k:6.1f}x  loss {p:.4f} [{lo:.4f}, {hi:.4f}]"
              f"   cells of size 1: {(sz==1).sum()}, median size {np.median(sz):.0f}")

    la = np.asarray(fw.tier_a(meta, False), np.int64)
    lb = np.asarray(fw.tier_a(meta, True), np.int64)
    m, lo, hi = paired_diff(A, la, lb, sources, tgt, full_vals, rho)
    print(f"  paired difference (hemilineage - hemilineage x side): "
          f"{m:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    print("  negative => the finer partition loses MORE, i.e. a real inversion")

    print("\n== CHECK 1: spatially compact but lineage-blind null, Tier B rungs ==")
    print(f"  {'k':>6} {'compr':>7} {'bio (Ward/lineage)':>20} {'spatial null':>14} "
          f"{'size-matched random':>20}")
    for f in (0.02, 0.05, 0.10, 0.20, 0.40):
        lab = np.asarray(fw.subsplit(meta, f), np.int64)
        k = int(lab.max() + 1)
        bio = F.rung_loss(A, lab, sources, tgt, full_vals, ALPHA, rho)
        spl = F.rung_loss(A, spatial_null(meta, k), sources, tgt, full_vals, ALPHA, rho)
        rl = [F.rung_loss(A, fw.size_matched_random(lab, np.random.default_rng(500 + b)),
                          sources, tgt, full_vals, ALPHA, rho) for b in range(3)]
        print(f"  {k:>6} {n/k:>6.1f}x {bio:>20.4f} {spl:>14.4f} {np.mean(rl):>20.4f}",
              flush=True)
    print("\n  if the spatial null matches bio, the Tier B gap is geometry not development")


if __name__ == "__main__":
    main()
