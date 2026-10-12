"""Measure, on the C. elegans connectome, each implementation difference between
the worm structural loss (class_merge_structural_loss.py) and the fly
structural loss (fly_fragility.py).

Differences checked:

  1 pair set     worm: every ordered pair (or every occ3-observed pair);
                 fly: 200 out-degree-stratified sources x 400 random targets
  2 self-pairs   worm excludes i == j; the fly target draw does not exclude the
                 source
  3 operator     worm: dense exact (I - alpha A / rho)^-1 - I;
                 fly: truncated Neumann series (fw.propagation_rows)
  4 rho          worm: max |eigenvalue| (dense); fly: power iteration on |A|
  q quotient     randi.quotient vs fw.quotient

Both code paths are run on the same matrix, and the structural loss of the
cell classes (k = 119, alpha = 0.1) is printed under each pair-selection rule.
The fly code's masking of non-finite values is not exercised.

The run fails if the two spectral radii, operators or quotients differ by more
than RHO_TOL (relative), OP_TOL or QUOT_TOL (absolute).

Inputs: conres.celegans, conres.randi, conres.flywire.
Outputs: results/diagnostics/verify_metric_parity.log.
Usage: python scripts/diagnostics/verify_metric_parity.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy import stats

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from conres import celegans as ce, randi, flywire as fw     # noqa: E402
from conres.paths import tee_stdout                   # noqa: E402
tee_stdout(__file__)

ALPHA = 0.1
RHO_TOL, OP_TOL, QUOT_TOL = 1e-6, 1e-6, 1e-9


def main():
    A, nodes = ce.build_matrix("all", neurons_only=True)
    labels = ce.partitions(nodes)["class"]
    n = A.shape[0]
    As = sp.csr_matrix(A)

    print(f"worm: n={n}, k={int(np.unique(labels).size)}, alpha={ALPHA}\n")

    # ---- 4. rho
    rho_dense = float(np.max(np.abs(np.linalg.eigvals(A))))
    rho_power = fw.spectral_radius(As)
    print(f"[4] rho   dense eigvals {rho_dense:.6f}   power iteration on |A| "
          f"{rho_power:.6f}   rel diff {abs(rho_dense-rho_power)/rho_dense:.2e}")

    # ---- 3. operator
    R_dense = randi.resolvent(A, ALPHA)
    src_all = np.arange(n)
    R_neu = fw.propagation_rows(As, src_all, ALPHA, rho=rho_dense)
    off = ~np.eye(n, dtype=bool)
    op_diff = np.abs(R_dense - R_neu)[off].max()
    print(f"[3] operator  max abs diff {op_diff:.3e}   "
          f"Spearman {stats.spearmanr(R_dense[off], R_neu[off]).statistic:.8f}")
    assert abs(rho_dense - rho_power) / rho_dense < RHO_TOL, "spectral radii disagree"
    assert op_diff < OP_TOL, "truncated series departs from the exact inverse"

    # ---- quotient definitions
    Qd = randi.quotient(A, labels)
    Qs = fw.quotient(As, labels).toarray()
    print(f"[q] quotient  max abs diff {np.abs(Qd - Qs).max():.3e}")
    assert np.abs(Qd - Qs).max() < QUOT_TOL, "quotient definitions disagree"

    # ---- loss under each pair-selection rule
    RQ = randi.resolvent(Qd, ALPHA)
    qv_full = RQ[np.ix_(labels, labels)]

    def loss(mask_j, mask_i):
        return 1 - stats.spearmanr(qv_full[mask_j, mask_i], R_dense[mask_j, mask_i]).statistic

    aj, ai = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    offj, offi = aj[off], ai[off]
    print(f"\n[1/2] loss at k=119 under different pair-selection rules")
    print(f"   all ordered pairs, self excluded  (n={len(offj):,})      "
          f"{loss(offj, offi):.4f}   <- pair set of the worm structural loss")
    dj, di = np.diag_indices(n)
    allj, alli = aj.ravel(), ai.ravel()
    print(f"   all ordered pairs, self INCLUDED  (n={len(allj):,})      "
          f"{loss(allj, alli):.4f}   <- fly-style, self not excluded")

    occ3, _ = randi.build_occ3(nodes, "wt")
    ii, jj = np.nonzero(occ3)
    print(f"   occ3-observed pairs only          (n={len(jj):,})      "
          f"{loss(jj, ii):.4f}")

    # fly-style stratified source sample, all targets
    rng = np.random.default_rng(0)
    for n_src in (100, 200):
        s, cnt = fw.stratified_sources(As, n_src, 0)
        sj = np.repeat(s, n)
        si = np.tile(np.arange(n), len(s))
        keep = sj != si
        print(f"   stratified {len(s):3d} sources x all targets, self excl "
              f"(n={keep.sum():,})  {loss(sj[keep], si[keep]):.4f}")

    # fly-style: stratified sources x sampled targets, self NOT excluded
    s, _ = fw.stratified_sources(As, 200, 0)
    t = np.stack([rng.choice(n, min(400, n), replace=False) for _ in s])
    sj = np.repeat(s, t.shape[1]); si = t.ravel()
    print(f"   stratified {len(s)} sources x {t.shape[1]} targets, fly rule "
          f"(n={len(sj):,})  {loss(sj, si):.4f}   <- closest to the fly path")


if __name__ == "__main__":
    main()
