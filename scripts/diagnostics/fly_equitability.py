"""Equitability defect gamma of the FlyWire hemilineage partitions against
size-matched random partitions.

gamma = ||(I-P) A P||_F / ||A P||_F, with P the projector onto cell-constant
functions. P is n x n for the full central brain and is never formed: with H
the indicator matrix and M = A H (H^T H)^-1,

    A P          = M H^T
    P (A P)      = H N H^T,   N = (H^T H)^-1 H^T M
    ||X H^T||_F^2 = sum_c n_c ||X[:, c]||^2

so both norms come from n x k factors (this is the outgoing defect). Dense
M is n x k, which is tractable for hemilineage and hemilineage x side
(fw.tier_a) but not for the fine hemilineage subdivisions.

Null: n_null label permutations (default 30), z with ddof = 1. The printed
worm reference line is a fixed constant, not recomputed here.

Inputs: conres.flywire (load_scope, load_adjacency, tier_a).
Outputs: printed only.
Usage: python scripts/diagnostics/fly_equitability.py [n_null]
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import scipy.sparse as sp

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from conres import flywire as fw                      # noqa: E402
from conres.paths import tee_stdout                   # noqa: E402
tee_stdout(__file__)


def gamma_sparse(A: sp.csr_matrix, labels: np.ndarray) -> float:
    n = A.shape[0]
    used, lab = np.unique(labels, return_inverse=True)
    k = used.size
    H = sp.csr_matrix((np.ones(n), (np.arange(n), lab)), shape=(n, k))
    sizes = np.asarray(H.sum(axis=0)).ravel()
    M = np.asarray((A @ H).todense()) / np.maximum(sizes, 1)      # n x k
    N = (H.T @ M) / np.maximum(sizes, 1)[:, None]                 # k x k
    R = M - N[lab, :]                                             # (M - H N)
    num = float((sizes * (R ** 2).sum(axis=0)).sum())
    den = float((sizes * (M ** 2).sum(axis=0)).sum())
    return float(np.sqrt(num / den)) if den > 0 else float("nan")


def main():
    n_null = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    meta = fw.load_scope()
    A = fw.load_adjacency(meta)
    rng = np.random.Generator(np.random.PCG64(0))
    print(f"fly central brain: {A.shape[0]:,} neurons, {A.nnz:,} connections\n")
    print(f"{'partition':>26} {'k':>7} {'compr':>8} {'gamma':>8} "
          f"{'random gamma':>16} {'z':>8}")
    print("-" * 80)
    for name, lab in (("hemilineage", fw.tier_a(meta, False)),
                      ("hemilineage x side", fw.tier_a(meta, True))):
        g = gamma_sparse(A, lab)
        nl = np.array([gamma_sparse(A, rng.permutation(lab)) for _ in range(n_null)])
        z = (g - nl.mean()) / nl.std(ddof=1)
        print(f"{name:>26} {lab.max()+1:>7} {A.shape[0]/(lab.max()+1):>7.0f}x "
              f"{g:>8.3f} {nl.mean():>9.3f}+-{nl.std(ddof=1):.3f} {z:>8.2f}")
    print(f"\n  worm reference, class partition: k=119, 2.5x, gamma 0.287, z -32.3")
    print("  (z is the comparable quantity across species; gamma alone scales with k)")
    print("\n  A fly z near zero would mean hemilineages are no more equitable than")
    print("  chance, which would explain the residual gap. A large negative z would")
    print("  mean they ARE equitable and the residual needs another explanation.")


if __name__ == "__main__":
    main()
