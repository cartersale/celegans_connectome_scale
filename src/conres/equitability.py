"""Equitability defect of a partition.

For adjacency A and partition with one-hot membership H, M = A H / |b| is each
neuron's mean weight onto each group. The partition is equitable (out-direction)
when M is constant within every group. gamma is the size-weighted relative
squared deviation of M from its group means:

    gamma = sqrt( sum_b |b| ||M[:, b] - M_bar[g(.), b]||^2 / sum_b |b| ||M[:, b]||^2 )

which is 0 for an equitable partition. Pass A.T for the in-direction.
"""
from __future__ import annotations

import numpy as np


def gamma(A: np.ndarray, lab: np.ndarray) -> float:
    """Equitability defect of partition `lab` on A (outgoing; pass A.T for incoming)."""
    n = A.shape[0]
    used, l = np.unique(lab, return_inverse=True)
    H = np.zeros((n, used.size)); H[np.arange(n), l] = 1.0
    sz = H.sum(0)
    M = (A @ H) / sz
    N = (H.T @ M) / sz[:, None]
    R = M - N[l, :]
    return float(np.sqrt((sz * (R ** 2).sum(0)).sum() / (sz * (M ** 2).sum(0)).sum()))
