"""Regression tests for the node sets, cell classes, equitability defect and quotient.

The node-set tests read data/raw/ and are skipped when it is absent.
Usage: pytest tests/
"""
from __future__ import annotations

import collections

import numpy as np
import pytest

from conres import celegans as ce, randi
from conres.equitability import gamma

needs_cook = pytest.mark.skipif(not (ce.RAW / "herm_full_edgelist.csv").exists(),
                                reason="Cook 2019 edge list not in data/raw/celegans")
needs_witvliet = pytest.mark.skipif(not list(ce.RAW.glob("witvliet_2020_*.xlsx")),
                                    reason="Witvliet 2021 spreadsheets not in data/raw/celegans")
needs_randi = pytest.mark.skipif(not (randi.EXPORTED / "exported_data").exists(),
                                 reason="Randi 2023 exports not in data/raw/randi/ex")


def planted(sizes=(4, 3, 5, 2), seed=0):
    """A matrix on which the partition with these group sizes is exactly equitable.

    Every neuron of group a sends the same weight B[a, b] to each neuron of
    group b, so its total onto group b is |b| B[a, b] for every member of a.
    """
    rng = np.random.default_rng(seed)
    lab = np.repeat(np.arange(len(sizes)), sizes)
    B = rng.exponential(1.0, (len(sizes), len(sizes)))
    return B[np.ix_(lab, lab)], lab


@needs_cook
def test_cook_node_set_and_classes():
    A, nodes = ce.build_matrix("all", neurons_only=True)
    assert A.shape == (302, 302)
    # 300 neurons plus the two muscles the release names without a muscle prefix
    assert {"anal", "sph"} <= set(nodes)
    sizes = collections.Counter(ce.assign_classes(nodes).values())
    assert len(sizes) == 119
    assert sum(v > 1 for v in sizes.values()) == 91
    assert np.unique(ce.partitions(nodes)["class"]).size == 119


@needs_witvliet
def test_witvliet_common_set():
    cells = ce.witvliet_common_neurons("all")
    assert len(cells) == 166
    assert sum(ce.is_glia(c) for c in cells) == 5
    assert len(ce.witvliet_common_neurons("all", exclude_glia=True)) == 161


@needs_cook
@needs_randi
def test_observed_pairs():
    A, nodes = ce.build_matrix("all", neurons_only=True)
    occ3, _ = randi.build_occ3(nodes, "wt")
    ii, jj = np.nonzero(occ3)
    assert len(ii) == 28989
    assert int((A[jj, ii] == 0).sum()) == 26537


def test_gamma_zero_on_planted_partition():
    A, lab = planted()
    assert gamma(A, lab) == pytest.approx(0.0, abs=1e-12)
    assert gamma(A.T, lab) == pytest.approx(0.0, abs=1e-12)
    assert gamma(A, np.random.default_rng(1).permutation(lab)) > 0.05


def test_quotient_identity_on_planted_partition():
    A, lab = planted()
    k = lab.max() + 1
    H = np.zeros((len(lab), k))
    H[np.arange(len(lab)), lab] = 1.0
    Q = randi.quotient(A, lab)
    # equitable: A H = H Q, so A and Q share the spectral radius
    assert np.allclose(A @ H, H @ Q)
    rho = lambda M: np.max(np.abs(np.linalg.eigvals(M)))
    assert rho(Q) == pytest.approx(rho(A))
    # and the class-level operator reproduces group sums of the full operator
    R, RQ = randi.resolvent(A, 0.1), randi.resolvent(Q, 0.1)
    assert np.allclose(R @ H, H @ RQ)


def test_quotient_is_mean_total_weight():
    rng = np.random.default_rng(2)
    A = rng.poisson(1.0, (9, 9)).astype(float)
    lab = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2])
    Q = randi.quotient(A, lab)
    assert Q[0, 2] == pytest.approx(A[np.ix_(lab == 0, lab == 2)].sum() / 3)
    assert randi.class_leverage(A, lab).shape == A.shape
