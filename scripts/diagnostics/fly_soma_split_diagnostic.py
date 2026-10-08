"""Fly structural loss at k = 12,935 (f = 0.40, 2.5x compression) for three
groupings: the hemilineage soma-position subdivision, curated cell_type, and a
size-matched random partition.

The hemilineage subdivision (fw.subsplit) is the pre-registered,
connectivity-blind rule. FlyWire cell_type is partly defined from connectivity,
so it is used only as a ceiling: how much lower the loss can go with a curated
grouping at the same k. Loss is fly_fragility.rung_loss on 200
out-degree-stratified sources x 400 targets (seed 0), alpha = 0.1. The printed
worm reference values are fixed constants, not recomputed here.

Inputs: conres.flywire, fly_fragility.
Outputs: printed only.
Usage: python scripts/diagnostics/fly_soma_split_diagnostic.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conres import flywire as fw                      # noqa: E402
from conres.paths import tee_stdout                   # noqa: E402
tee_stdout(__file__)
import fly_fragility as F                        # noqa: E402

ALPHA = 0.1
TARGET_K = 12935          # the f=0.40 rung, 2.5x compression


def celltype_at_k(meta: pd.DataFrame, target_k: int, rng) -> np.ndarray:
    """Groups from curated cell_type x side (unannotated neurons as singletons),
    adjusted toward target_k.

    Too fine: k - target_k disjoint random pairs of groups are merged. Too
    coarse: groups are halved at random, largest first, each at most once, so
    target_k is not reached if more than k splits are needed.
    """
    fallback = pd.Series("__" + meta.index.astype(str), index=meta.index)
    keys = meta["cell_type"].astype("object").fillna(fallback)
    keys = keys.astype(str) + "|" + meta["side"].astype(str)
    lab = pd.factorize(keys)[0]
    k = lab.max() + 1
    if k > target_k:                       # too fine: merge random type pairs
        order = rng.permutation(k)
        merge_map = np.arange(k)
        need = k - target_k
        for a in range(0, 2 * need, 2):
            merge_map[order[a + 1]] = merge_map[order[a]]
        lab = pd.factorize(merge_map[lab])[0]
    elif k < target_k:                     # too coarse: split largest groups
        lab = lab.copy()
        nxt = k
        sizes = np.bincount(lab)
        for g in np.argsort(-sizes):
            if nxt >= target_k:
                break
            idx = np.flatnonzero(lab == g)
            if idx.size < 2:
                continue
            half = rng.permutation(idx)[: idx.size // 2]
            lab[half] = nxt
            nxt += 1
        lab = pd.factorize(lab)[0]
    return lab


def main():
    n_src, n_tgt = 200, 400
    meta = fw.load_scope()
    A = fw.load_adjacency(meta)
    n = A.shape[0]
    rho = fw.spectral_radius(A)
    sources, counts = fw.stratified_sources(A, n_src, seed=0)
    rng = np.random.Generator(np.random.PCG64(0))
    tgt = F.pair_sample(n, sources, n_tgt, rng)
    fullrows = fw.propagation_rows(A, sources, ALPHA, rho)
    # tgt is (n_src, n_tgt): row r holds the targets sampled for source r
    full_vals = np.stack([fullrows[r][tgt[r]] for r in range(len(sources))]).ravel()
    print(f"scope {n:,} neurons; sources {len(sources)}; target pairs {len(full_vals):,}")
    print(f"out-degree strata realised counts: {list(counts)}")
    empty = int((np.asarray(counts) == 0).sum())
    if empty:
        print(f"  WARNING: {empty} empty strata -- degree is discrete here; "
              f"equal-count strata would be better than equal-quantile")

    print(f"\nAt k={TARGET_K:,} (2.5x compression):")
    tb = fw.subsplit(meta, 0.40)
    for name, lab in (("Tier B soma-position (pre-registered)", tb),
                      ("cell_type ceiling (contaminated)", celltype_at_k(meta, TARGET_K, rng)),
                      ("size-matched random", fw.size_matched_random(tb, rng))):
        loss = F.rung_loss(A, lab, sources, tgt, full_vals, ALPHA, rho)
        print(f"  {name:38s} k={lab.max()+1:6d}  loss {loss:.4f}")
    print(f"\n  worm biological at 2.54x: 0.1218   worm random at 2.54x: 0.7066")
    print("\n  If the ceiling lands near the worm's 0.122, the residual gap is the")
    print("  blind Tier B rule. If it stays near 0.295, it is a real species difference.")


if __name__ == "__main__":
    main()
