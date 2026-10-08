"""Equitability defect of the adult cell-class partition against size-matched random partitions.

gamma (conres.equitability) is computed for the 119 cell classes on the Cook
all-synapse matrix A, outgoing (M = A) and incoming (M = A^T), and compared with
N_NULL = 1000 label permutations of the class partition (same class sizes,
seed 0) on the outgoing defect; z = (gamma - null mean) / null SD.

Inputs:  Cook connectome via conres.celegans.
Outputs: results/class_gamma.csv (gamma_out, gamma_in, null_mean, null_sd, z).
Usage:   python scripts/class_gamma.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from conres import celegans as ce                       # noqa: E402
from conres.equitability import gamma                   # noqa: E402

N_NULL = 1000


def main() -> None:
    A, nodes = ce.build_matrix("all", neurons_only=True)
    lab = np.unique(ce.partitions(nodes)["class"], return_inverse=True)[1]
    rng = np.random.default_rng(0)
    null = np.array([gamma(A, rng.permutation(lab)) for _ in range(N_NULL)])
    g_out, g_in = gamma(A, lab), gamma(A.T, lab)
    z = (g_out - null.mean()) / null.std()
    print(f"gamma out {g_out:.3f} (in {g_in:.3f}); null {null.mean():.3f} "
          f"+/- {null.std():.3f}; z = {z:.1f}")
    pd.DataFrame([dict(gamma_out=g_out, gamma_in=g_in, null_mean=null.mean(),
                       null_sd=null.std(), z=z)]).to_csv(ROOT / "results/class_gamma.csv", index=False)


if __name__ == "__main__":
    main()
