"""Fly structural loss of hemilineage subdivisions against a spatially compact,
lineage-blind null at matched k.

Hemilineage subdivisions (fw.subsplit) split hemilineages by Ward on soma
position, so their cells are spatially compact, and nearby neurons share
partners. The size-matched random null is spatially arbitrary; the spatial
null is MiniBatchKMeans (n_init=3, seed 0) on z-scored soma position over the
whole in-scope set, ignoring hemilineage boundaries, at the rung's k (annotation
position where the soma position is missing).

Loss is fly_fragility.rung_loss (1 - Spearman between quotient and full
operator, alpha = 0.1, full-connectome rho) on 200 out-degree-stratified
sources x 400 targets (seed 0). Printed "% of gap that is geometry" is
(random - spatial) / (random - bio).

The rungs are f = 0.02, 0.05, 0.10, 0.20, 0.40 by default.

Inputs: conres.flywire, fly_fragility (pair_sample, rung_loss).
Outputs: results/fly_spatial_null.csv (k, compression, bio, spatial_null, random).
Usage: python scripts/diagnostics/fly_spatial_null.py [f ...]
"""
from __future__ import annotations

import csv
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conres import flywire as fw                      # noqa: E402
from conres.paths import tee_stdout                   # noqa: E402
tee_stdout(__file__)
import fly_fragility as F                        # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
ALPHA = 0.1


def spatial_labels(meta, k, seed=0):
    """Spatially compact, lineage-blind cells at matched k."""
    X = meta[["soma_x", "soma_y", "soma_z"]].to_numpy(float)
    bad = ~np.isfinite(X).all(axis=1)
    if bad.any():                       # fall back to the annotation position
        X[bad] = meta.loc[bad, ["pos_x", "pos_y", "pos_z"]].to_numpy(float)
    X = (X - np.nanmean(X, axis=0)) / np.nanstd(X, axis=0)
    X = np.nan_to_num(X)
    km = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=3,
                         batch_size=4096).fit(X)
    return km.labels_


def main():
    fs = [float(x) for x in sys.argv[1:]] or [0.02, 0.05, 0.10, 0.20, 0.40]
    meta = fw.load_scope()
    A = fw.load_adjacency(meta)
    n = A.shape[0]
    rho = fw.spectral_radius(A)
    sources, _ = fw.stratified_sources(A, 200, 0)
    rng = np.random.default_rng(0)
    tgt = F.pair_sample(n, sources, 400, rng)
    full_rows = fw.propagation_rows(A, sources, ALPHA, rho=rho)
    full_vals = np.stack([full_rows[r][tgt[r]] for r in range(len(sources))]).ravel()
    print(f"in-scope {n:,} neurons, rho={rho:.0f}, {len(sources)} sources, "
          f"{len(full_vals):,} pairs\n")
    print(f"{'k':>7} {'compr':>7} {'bio (Ward/lineage)':>20} {'spatial null':>14} "
          f"{'size-matched random':>20} {'% of gap that is geometry':>27}")
    rows = []
    for f in fs:
        lab = fw.subsplit(meta, f)
        k = int(lab.max() + 1)
        bio = F.rung_loss(A, lab, sources, tgt, full_vals, ALPHA, rho)
        spa = F.rung_loss(A, spatial_labels(meta, k), sources, tgt, full_vals, ALPHA, rho)
        rnd = F.rung_loss(A, fw.size_matched_random(lab, rng), sources, tgt,
                          full_vals, ALPHA, rho)
        frac = 100 * (rnd - spa) / (rnd - bio) if rnd > bio else np.nan
        rows.append({"k": k, "compression": round(n / k, 2), "bio": round(bio, 4),
                     "spatial_null": round(spa, 4), "random": round(rnd, 4)})
        print(f"{k:>7} {n/k:>6.1f}x {bio:>20.4f} {spa:>14.4f} {rnd:>20.4f} "
              f"{frac:>26.0f}%", flush=True)
    out = ROOT / "results/fly_spatial_null.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out}")
    print("  last column = (random - spatial null) / (random - bio)")


if __name__ == "__main__":
    main()
