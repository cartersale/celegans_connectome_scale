"""Structural loss of the FlyWire path operator at every rung of the biological
hierarchy, for the biological, spatial and size-matched random partitions, and
the compression N/k each arm tolerates within eps.

Structural only: there is no brain-wide propagation measurement in the fly.
Rungs are the two hemilineage partitions (tier A, with and without side) and
the nested sub-split ladder over fw.F_GRID (tier B). Loss is
`fly_fragility.rung_loss` on the same source/target sample as
`fly_fragility.py` (200 stratified sources, seed 0, x 400 targets,
alpha = 0.1), so the bio column must match that script's output.

Arms:
  bio        the rung's own labels
  spatial    MiniBatchKMeans on z-scored soma position at the rung's k (as in
             `fly_checks.py` / `fly_spatial_null.py`); loss 0 at k >= N, since
             a near-identity partition is near-lossless whatever defined it
  random     fw.size_matched_random, --n-rand draws (seeds 1000 + b)

The summary reports, per rung, bio / random and the geometry share
(random - spatial) / (random - bio), and for each eps the largest compression
N/k keeping loss <= eps at every finer tier B rung. With a discrete ladder the
threshold is resolved only to the bracketing pair of rungs and is printed so.

Results are appended rung by rung and finished rungs are skipped on restart.

Inputs:  FlyWire scope and adjacency via conres.flywire.
Outputs: results/fly_complexity_curve.csv
Usage:   python scripts/fly_complexity_curve.py [--n-rand 5] [--rungs all|name,name]
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from conres import flywire as fw                 # noqa: E402
import fly_fragility as F                   # noqa: E402

ALPHA = 0.1
OUT = ROOT / "results/fly_complexity_curve.csv"
FIELDS = ["rung", "tier", "k", "compression", "bio", "spatial",
          "random_mean", "random_sd", "seconds"]


def spatial_labels(meta, k: int, seed: int = 0) -> np.ndarray:
    n = len(meta)
    if k >= n:
        return np.arange(n)
    X = meta[["soma_x", "soma_y", "soma_z"]].to_numpy(float)
    X = (X - X.mean(0)) / X.std(0)
    km = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=3,
                         batch_size=max(4096, min(n, 2 * k))).fit(X)
    return np.unique(km.labels_, return_inverse=True)[1].astype(np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-rand", type=int, default=5)
    ap.add_argument("--rungs", default="all",
                    help="comma-separated rung names, or 'all'")
    args = ap.parse_args()

    meta = fw.load_scope()
    A = fw.load_adjacency(meta)
    n = A.shape[0]
    rho = fw.spectral_radius(A)
    sources, _ = fw.stratified_sources(A, 200, 0)
    tgt = F.pair_sample(n, sources, 400, np.random.default_rng(0))
    full_rows = fw.propagation_rows(A, sources, ALPHA, rho=rho)
    full_vals = np.stack([full_rows[r][tgt[r]] for r in range(len(sources))]).ravel()
    print(f"fly: {n:,} neurons, rho {rho:.1f}, {len(full_vals):,} scored pairs", flush=True)

    rungs = [("tierA_hemilineage", fw.tier_a(meta, False), "A"),
             ("tierA_hemilineage_side", fw.tier_a(meta, True), "A")]
    rungs += [(f"tierB_f{f:.2f}", fw.subsplit(meta, f), "B") for f in fw.F_GRID]
    want = None if args.rungs == "all" else set(args.rungs.split(","))

    done = set()
    if OUT.exists():
        with OUT.open() as fh:
            done = {r["rung"] for r in csv.DictReader(fh)}
    else:
        with OUT.open("w", newline="") as fh:
            csv.DictWriter(fh, fieldnames=FIELDS).writeheader()

    loss = lambda lab: F.rung_loss(A, np.asarray(lab, np.int64), sources, tgt,
                                   full_vals, ALPHA, rho)
    for name, lab, tier in rungs:
        if (want and name not in want) or name in done:
            continue
        t0 = time.time()
        lab = np.asarray(lab, np.int64)
        k = int(lab.max() + 1)
        bio = loss(lab)
        spa = loss(spatial_labels(meta, k)) if k < n else 0.0
        rnd = np.array([loss(fw.size_matched_random(lab, np.random.default_rng(1000 + b)))
                        for b in range(args.n_rand)], float)
        row = {"rung": name, "tier": tier, "k": k, "compression": round(n / k, 3),
               "bio": round(float(bio), 5), "spatial": round(float(spa), 5),
               "random_mean": round(float(np.nanmean(rnd)), 5),
               "random_sd": round(float(np.nanstd(rnd, ddof=1)), 5),
               "seconds": round(time.time() - t0, 1)}
        with OUT.open("a", newline="") as fh:
            csv.DictWriter(fh, fieldnames=FIELDS).writerow(row)
        print(f"  {name:24s} k={k:6d} {n / k:7.2f}x  bio {bio:.4f}  spatial {spa:.4f}  "
              f"random {row['random_mean']:.4f} +- {row['random_sd']:.4f}  "
              f"({row['seconds']:.0f}s)", flush=True)

    summarise(n)


def summarise(n: int) -> None:
    import pandas as pd
    df = pd.read_csv(OUT).sort_values("k", ascending=False)
    print(f"\n{'k':>7} {'N/k':>7} {'bio':>7} {'spatial':>8} {'random':>7} "
          f"{'bio/rand':>9} {'geometry share':>15}")
    for _, r in df.iterrows():
        gap = r.random_mean - r.bio
        geo = (r.random_mean - r.spatial) / gap if gap > 0 else np.nan
        ratio = r.bio / r.random_mean if r.random_mean > 0 else np.nan
        print(f"{r.k:>7d} {r.compression:>7.2f} {r.bio:>7.4f} {r.spatial:>8.4f} "
              f"{r.random_mean:>7.4f} {ratio:>9.3f} {geo:>14.0%}")
    # only tier B is a nested ladder; tier A rungs are a different construction
    ladder = df[df.tier == "B"].sort_values("k", ascending=False)
    print("\nlargest compression keeping loss <= eps at every finer Tier B rung")
    for arm, col in (("bio", "bio"), ("spatial", "spatial"), ("random", "random_mean")):
        parts = []
        for eps in (0.05, 0.10, 0.20, 0.30, 0.50):
            last, nxt = None, None
            for _, r in ladder.iterrows():
                if r[col] <= eps:
                    last = r
                else:
                    nxt = r
                    break
            if last is None:
                parts.append(f"eps {eps:.2f}: none")
            elif nxt is None:
                parts.append(f"eps {eps:.2f}: >= {last.compression:.1f}x")
            else:
                parts.append(f"eps {eps:.2f}: {last.compression:.2f}x-{nxt.compression:.2f}x")
        print(f"  {arm:>8}  " + "   ".join(parts))


if __name__ == "__main__":
    main()
