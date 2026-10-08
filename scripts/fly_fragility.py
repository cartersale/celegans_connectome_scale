"""Score biological FlyWire partitions from hemilineage down to singletons against the full-resolution propagation operator, with
size-matched random nulls at every rung.

Rungs: Tier A is hemilineage and hemilineage x side; Tier B is fw.subsplit at
each f in fw.F_GRID (side, then Ward on soma position within each hemilineage x
side group). Rule, scope, tiers and nulls are fixed in
preregistration/fly_subsplit_rule.md.

There is no brain-wide causal propagation measurement in the fly, so the
reference is the full-resolution operator itself: loss = 1 - Spearman(quotient
values, full-operator values) over a fixed sample of ordered pairs, identical
across arms and rungs. The quotient Q[a, b] is the mean out-weight from a member
of a into b; propagation rows are (I - alpha A/rho)^-1 - I. By default rho is
the full connectome's spectral radius; --own-rho uses each quotient's own.

Sources: n_src stratified by out-degree decile; n_tgt targets per source drawn
without replacement (seed). Random nulls: n_rand label permutations, seeds
1000+b.

Inputs: FlyWire in-scope adjacency and metadata via conres.flywire.
Outputs: results/fly_fragility_alpha{alpha}[_ownrho].csv (rung, tier, k,
compression, bio_loss, rand_loss_mean, rand_loss_sd).
Usage: python fly_fragility.py [--alpha 0.1] [--n-src 200] [--n-tgt 400]
       [--n-rand 5] [--seed 0] [--own-rho]
"""
from __future__ import annotations

import argparse
import csv
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy import stats

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from conres import flywire as fw          # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def pair_sample(n_nodes, sources, n_tgt, rng):
    """Fixed target sample per source; identical pairs scored at every rung."""
    tgt = np.stack([rng.choice(n_nodes, n_tgt, replace=False) for _ in sources])
    return tgt


def rung_loss(A, labels, sources, tgt, full_vals, alpha, rho_scale):
    """1 - Spearman between quotient and full-operator values on the fixed pairs;
    NaN if fewer than 100 finite pairs or the quotient values are constant."""
    Q = fw.quotient(A, labels)
    cells = labels[sources]
    rows = fw.propagation_rows(Q, cells, alpha, rho=rho_scale)
    qv = np.stack([rows[r][labels[tgt[r]]] for r in range(len(sources))]).ravel()
    ok = np.isfinite(qv) & np.isfinite(full_vals)
    if ok.sum() < 100 or np.std(qv[ok]) == 0:
        return np.nan
    return 1.0 - float(stats.spearmanr(qv[ok], full_vals[ok]).statistic)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--n-src", type=int, default=200)
    ap.add_argument("--n-tgt", type=int, default=400)
    ap.add_argument("--n-rand", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--own-rho", action="store_true",
                    help="normalize each class-level operator by its own quotient's spectral "
                         "radius (the worm convention) instead of the full connectome's")
    args = ap.parse_args()

    meta = fw.load_scope()
    A = fw.load_adjacency(meta)
    n = A.shape[0]
    rho = fw.spectral_radius(A)
    print(f"in-scope: {n:,} central-brain neurons, {A.nnz:,} connections, rho={rho:.1f}")

    sources, per_dec = fw.stratified_sources(A, args.n_src, args.seed)
    rng = np.random.default_rng(args.seed)
    tgt = pair_sample(n, sources, args.n_tgt, rng)
    print(f"sources: {len(sources)} stratified by out-degree decile, "
          f"per-decile counts {per_dec}")
    print(f"pairs scored: {len(sources) * args.n_tgt:,}  alpha={args.alpha}\n")

    full_rows = fw.propagation_rows(A, sources, args.alpha, rho=rho)
    full_vals = np.stack([full_rows[r][tgt[r]] for r in range(len(sources))]).ravel()

    rungs = [("tierA_hemilineage", fw.tier_a(meta, False), "A"),
             ("tierA_hemilineage_side", fw.tier_a(meta, True), "A")]
    for f in fw.F_GRID:
        rungs.append((f"tierB_f{f:.2f}", fw.subsplit(meta, f), "B"))

    q_rho = None if args.own_rho else rho
    rows = []
    for name, lab, tier in rungs:
        lab = np.asarray(lab, np.int64)
        k = int(lab.max() + 1)
        bio = rung_loss(A, lab, sources, tgt, full_vals, args.alpha, q_rho)
        rnd = []
        for b in range(args.n_rand):
            rl = fw.size_matched_random(lab, np.random.default_rng(1000 + b))
            rnd.append(rung_loss(A, rl, sources, tgt, full_vals, args.alpha, q_rho))
        rnd = np.array(rnd, float)
        rows.append({"rung": name, "tier": tier, "k": k,
                     "compression": round(n / k, 2),
                     "bio_loss": round(float(bio), 5),
                     "rand_loss_mean": round(float(np.nanmean(rnd)), 5),
                     "rand_loss_sd": round(float(np.nanstd(rnd, ddof=1)), 5)})
        print(f"  {name:22s} tier {tier}  k={k:6d} {n/k:7.1f}x   "
              f"bio loss {bio:+.4f}   random {np.nanmean(rnd):+.4f} "
              f"+- {np.nanstd(rnd, ddof=1):.4f}", flush=True)

    out = ROOT / f"results/fly_fragility_alpha{args.alpha}{'_ownrho' if args.own_rho else ''}.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
