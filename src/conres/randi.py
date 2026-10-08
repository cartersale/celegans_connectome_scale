"""Randi et al. (2023) signal-propagation data and the path operators scored against it.

Data (data/raw/randi/):
  randi_funconn_<strain>.json   released atlas: one row per ordered pair
                                (stimulated `pre` -> responding `post`) with a
                                detected response, and its signed amplitude.
                                Pairs observed without a detected response are
                                not in this file.
  ex/exported_data[_<strain>]/  per-recording exports: <i>_labels.txt (neuron
                                ID per ROI), <i>_stim_neurons.txt (ROI index of
                                each stimulation), <i>_stim_volume_i.txt
                                (volume index of each stimulation),
                                <i>_gcamp.txt (volumes x ROIs). Used to rebuild
                                the set of observed pairs and the correlations.

Operators: `resolvent` is the path operator R = (I - alpha A / rho)^-1 - I;
`quotient` and `class_leverage` give its class-level version, built on the
quotient matrix and read back at neuron resolution.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from conres.paths import REPO_ROOT
from conres import celegans as ce

RAW = REPO_ROOT / "data/raw/randi"
CE_RAW = REPO_ROOT / "data/raw/celegans"


def load_responses(strain: str = "wildty") -> pd.DataFrame:
    """Measured pairs. `value` is the signed response, `mag` its magnitude."""
    f = RAW / f"randi_funconn_{strain}.json"
    d = pd.DataFrame(json.load(f.open()))
    d["pre"] = d["pre"].map(ce.normalise_name)
    d["post"] = d["post"].map(ce.normalise_name)
    d = d.rename(columns={"synapses": "value"})
    d["mag"] = d["value"].abs()
    return d[["pre", "post", "value", "mag"]]


def load_extrasynaptic(kind: str, nodes: list[str]) -> np.ndarray:
    """Bentley et al. 2016 monoamine ('MA') or neuropeptide ('NP') network on `nodes`.

    A[i, j] counts distinct ligand-receptor entries from i to j in
    data/raw/celegans/edgelist_<kind>.csv; names outside `nodes` are dropped.
    """
    f = CE_RAW / f"edgelist_{kind}.csv"
    d = pd.read_csv(f, header=None, names=["pre", "post", "transmitter", "receptor"])
    d["pre"] = d["pre"].map(ce.normalise_name)
    d["post"] = d["post"].map(ce.normalise_name)
    idx = {n: i for i, n in enumerate(nodes)}
    A = np.zeros((len(nodes), len(nodes)))
    for s, t in zip(d.pre, d.post):
        if s in idx and t in idx:
            A[idx[s], idx[t]] += 1.0        # multiplicity = distinct ligand/receptor pairs
    return A


def resolvent(A: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """Path operator R = (I - alpha A / rho)^-1 - I = sum_{l>=1} (alpha A / rho)^l.

    rho is the spectral radius of A, so the series converges for 0 < alpha < 1.
    R[i, j] sums all walks i -> j with weight (alpha/rho)^length; the identity
    (length-zero walk) is removed. Returns zeros if rho is 0.
    """
    rho = np.max(np.abs(np.linalg.eigvals(A)))
    if rho <= 0:
        return np.zeros_like(A)
    R = np.linalg.inv(np.eye(len(A)) - alpha * A / rho) - np.eye(len(A))
    return R


def quotient(A: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Quotient matrix Q = (H^T A H) / |a|, with H the one-hot membership matrix.

    Q[a, b] is the mean, over members of group a, of their total weight onto
    group b. Groups are the distinct values of `labels`, in sorted order.
    """
    n = A.shape[0]
    used, lab = np.unique(labels, return_inverse=True)
    H = np.zeros((n, used.size))
    H[np.arange(n), lab] = 1.0
    return (H.T @ A @ H) / H.sum(axis=0)[:, None]


def class_leverage(A: np.ndarray, labels: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """Class-level path operator at neuron resolution: R_Q[g(i), g(j)].

    R_Q = resolvent(quotient(A, labels), alpha), normalised by the spectral
    radius of Q. Every pair drawn from the same two groups gets the same value.

    `labels` must be 0..k-1 integer codes: they index R_Q directly.
    """
    RQ = resolvent(quotient(A, labels), alpha)
    return RQ[np.ix_(labels, labels)]


# ------------------------------------------------------- observation matrix

EXPORTED = RAW / "ex"


def _recordings(folder: Path, keep: set[int] | None = None):
    """Yield (ROI labels, stimulated ROI indices) per recording in an export folder.

    `keep` restricts to a subset of recording indices. Recordings without a
    stim_neurons file are skipped.
    """
    import re as _re
    for f in sorted(folder.glob("*_labels.txt")):
        if keep is not None and int(_re.match(r"(\d+)_labels", f.name).group(1)) not in keep:
            continue
        i = _re.match(r"(\d+)_labels", f.name).group(1)
        sf = folder / f"{i}_stim_neurons.txt"
        if not sf.exists():
            continue
        labels = [ln.rstrip("\n").strip() for ln in f.open()]
        stims = [int(x) for x in sf.open() if x.strip()]
        yield labels, stims


def build_occ3(nodes: list[str], strain: str = "wt",
               keep: set[int] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Observation counts per ordered pair, rebuilt from the per-recording exports.

    occ3[i, j] = number of stimulations of neuron j in recordings where neuron
    i was also identified (i != j). A pair is "observed" when occ3 > 0; the
    count also serves as the exposure covariate.

    ROI labels are used only when they exactly match a name in `nodes`. The
    exports also contain glia (AMsoL), annotation notes ('merge', 'smthng
    else'), bare ROI numbers and partial identifications (IL2V, AWCOFF); none
    match a connectome node and none appear in the released atlas.

    Returns (occ3, occ3): both elements are the same array.
    """
    folder = EXPORTED / ("exported_data" if strain == "wt" else f"exported_data_{strain}")
    valid = {n: k for k, n in enumerate(nodes)}
    n = len(nodes)
    occ3 = np.zeros((n, n))
    for labels, stims in _recordings(folder, keep):
        present = sorted({valid[v] for v in labels if v in valid})
        if not present:
            continue
        for s in stims:
            if not (0 <= s < len(labels)):
                continue
            src = labels[s]
            if src not in valid:
                continue
            j = valid[src]
            for i in present:
                if i != j:
                    occ3[i, j] += 1.0
    return occ3, occ3


def n_recordings(strain: str = "wt") -> int:
    folder = EXPORTED / ("exported_data" if strain == "wt" else f"exported_data_{strain}")
    return len(list(folder.glob("*_labels.txt")))


# ------------------------------------------------- correlation observable

def _gcamp(folder: Path, i: int) -> np.ndarray:
    return np.loadtxt(folder / f"{i}_gcamp.txt")


def build_correlations(nodes: list[str], strain: str = "wt",
                       excise: int = 0, detrend: bool = False,
                       keep: set[int] | None = None
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Mean within-recording Pearson correlation per neuron pair, and its recording count.

    For each recording: ROI labels are name-normalised and matched to `nodes`;
    traces that are entirely NaN are dropped; recordings with fewer than 100
    volumes (before or after excision) are skipped. r is computed over the
    remaining volumes and averaged over the recordings in which both neurons
    were identified. Returns (R, cnt), symmetric, with nan / 0 on the diagonal
    and nan where a pair was never co-identified.

    The recordings contain the stimulation protocol, so the correlations are
    measured on the same frames as the evoked responses. `excise` > 0 masks,
    for each stimulation volume s, volumes s-2 .. s+excise-1. `detrend` removes
    each trace's least-squares linear trend before correlating.

    `keep` restricts to a subset of recording indices.
    """
    folder = EXPORTED / ("exported_data" if strain == "wt" else f"exported_data_{strain}")
    valid = {n: k for k, n in enumerate(nodes)}
    n = len(nodes)
    ssum = np.zeros((n, n))
    cnt = np.zeros((n, n))
    import re as _re
    for f in sorted(folder.glob("*_labels.txt")):
        i = int(_re.match(r"(\d+)_labels", f.name).group(1))
        if keep is not None and i not in keep:
            continue
        labels = [ln.rstrip("\n").strip() for ln in f.open()]
        labels = [ce.normalise_name(x) for x in labels]
        G = _gcamp(folder, i)
        if G.shape[0] < 100:
            continue
        # NaNs in the exports occur as whole columns (a labelled neuron that was
        # never tracked), never as scattered samples, so dropping the column
        # leaves the rest of the recording intact.
        alive = np.isfinite(G).all(axis=0)
        cols = [(k, valid[x]) for k, x in enumerate(labels)
                if x in valid and k < G.shape[1] and alive[k]]
        if len(cols) < 2:
            continue
        mask = np.ones(G.shape[0], bool)
        if excise:
            sf = folder / f"{i}_stim_volume_i.txt"
            if sf.exists():
                for s in (int(x) for x in sf.open() if x.strip()):
                    mask[max(0, s - 2):s + excise] = False
        X = G[np.ix_(mask, [c for c, _ in cols])]
        if X.shape[0] < 100:
            continue
        if detrend:
            t = np.arange(X.shape[0], dtype=float)
            t -= t.mean()
            X = X - np.outer(t, (t @ X) / (t @ t))
        X = X - X.mean(0)
        sd = X.std(0)
        good = (sd > 0) & np.isfinite(sd)
        if good.sum() < 2:
            continue
        C = (X[:, good].T @ X[:, good]) / (X.shape[0] * np.outer(sd[good], sd[good]))
        idx = np.array([j for (_, j), g in zip(cols, good) if g])
        ssum[np.ix_(idx, idx)] += C
        cnt[np.ix_(idx, idx)] += 1.0
    with np.errstate(invalid="ignore"):
        R = ssum / cnt
    np.fill_diagonal(R, np.nan)
    np.fill_diagonal(cnt, 0.0)
    return R, cnt
