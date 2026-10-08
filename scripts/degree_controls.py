"""Test whether path-operator prediction of signal propagation reduces to neuronal degree.

Two controls, scored by AUC for detected responses (Randi 2023, wild type) over
observed pairs (occ3 > 0), across all observed pairs and across unsynapsed
pairs (A[s, r] = 0):

  degree predictors   in-degree of the responding neuron, sum_i B[i, r], and the
                      product out-degree(s) x in-degree(r), the two-step path
                      count expected under random wiring with the same degrees;
                      B = (A > 0) is the binary connection matrix.
  rewired surrogates  N_SURR = 100 degree-preserving surrogates of A. Each is
                      built by SWAPS_PER_EDGE x E attempted directed double-edge
                      swaps (i -> j, k -> l becomes i -> l, k -> j), with each
                      connection's strength carried by its source end, and swaps
                      that would create a self-connection or duplicate connection
                      rejected; every neuron keeps its exact in- and out-degree.
                      The path operator (alpha = 0.1) is recomputed on each
                      surrogate. Unsynapsed pairs are always defined from the
                      real A, so every surrogate is scored on the same pairs.
                      z = (real - surrogate mean) / surrogate SD (ddof = 1).

Inputs:  Cook connectome (conres.celegans); responses and observation counts
         (conres.randi).
Outputs: results/degree_controls.csv (predictor, pairs, auc, with surrogate
         mean, sd and z on the path-operator rows),
         results/degree_controls_surrogates.csv (one AUC per surrogate).
Usage:   python scripts/degree_controls.py [--n-surr 100] [--seed 0]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from conres import celegans as ce, randi                # noqa: E402

ALPHA = 0.1
SWAPS_PER_EDGE = 10


def rewire(A: np.ndarray, rng: np.random.Generator,
           swaps_per_edge: int = SWAPS_PER_EDGE) -> np.ndarray:
    """Directed double-edge swap surrogate preserving every in- and out-degree.

    Weights stay with the source end of each connection, so out-strength is
    preserved and in-strength is not. Existing self-connections can be moved
    but no new ones are created.
    """
    u, v = np.nonzero(A)
    w = A[u, v].copy()
    E = len(u)
    have = set(zip(u.tolist(), v.tolist()))
    for _ in range(swaps_per_edge * E):
        i, j = rng.integers(0, E, 2)
        a, b, c, d = u[i], v[i], u[j], v[j]
        if a == d or c == b or (a, d) in have or (c, b) in have:
            continue
        have -= {(a, b), (c, d)}
        have |= {(a, d), (c, b)}
        v[i], v[j] = d, b
    out = np.zeros_like(A)
    out[u, v] = w
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-surr", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    A, nodes = ce.build_matrix("all", neurons_only=True)
    idx = {n: i for i, n in enumerate(nodes)}
    occ3, _ = randi.build_occ3(nodes, "wt")
    resp, stim = np.nonzero(occ3)                       # occ3[responder, stimulated]
    d = randi.load_responses("wildty")
    hits = {(idx[p], idx[q]) for p, q in zip(d.pre, d.post) if p in idx and q in idx}
    y = np.array([(s, r) in hits for s, r in zip(stim, resp)], dtype=float)
    nosyn = A[stim, resp] == 0
    sets = {"all observed": np.ones_like(nosyn), "unsynapsed": nosyn}
    print(f"{len(y)} observed pairs, {int(y.sum())} responses; "
          f"{nosyn.sum()} unsynapsed, {int(y[nosyn].sum())} responses")

    B = (A > 0).astype(float)
    indeg, outdeg = B.sum(0), B.sum(1)
    preds = {
        "path operator": randi.resolvent(A, ALPHA)[stim, resp],
        "in-degree of responder": indeg[resp],
        "out-degree x in-degree": outdeg[stim] * indeg[resp],
    }
    rows = []
    for name, p in preds.items():
        for pairs, m in sets.items():
            rows.append(dict(predictor=name, pairs=pairs, auc=roc_auc_score(y[m], p[m])))

    rng = np.random.default_rng(args.seed)
    surr = []
    for b in range(args.n_surr):
        S = randi.resolvent(rewire(A, rng), ALPHA)[stim, resp]
        surr.append({pairs: roc_auc_score(y[m], S[m]) for pairs, m in sets.items()})
        if (b + 1) % 20 == 0:
            print(f"  {b + 1}/{args.n_surr} surrogates", flush=True)
    surr = pd.DataFrame(surr)
    for r in rows:
        if r["predictor"] == "path operator":
            s = surr[r["pairs"]]
            r.update(surr_mean=s.mean(), surr_sd=s.std(ddof=1),
                     z=(r["auc"] - s.mean()) / s.std(ddof=1))

    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results/degree_controls.csv", index=False)
    surr.to_csv(ROOT / "results/degree_controls_surrogates.csv", index_label="surrogate")
    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 120):
        print(out.to_string(index=False))
    print("wrote results/degree_controls.csv, results/degree_controls_surrogates.csv")


if __name__ == "__main__":
    main()
