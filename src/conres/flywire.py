"""FlyWire (v783) central-brain connectome and the pre-registered partition ladder.

Scope, provenance tiers (A: hemilineage, hemilineage x side; B: soma-position
sub-splits) and the sub-splitting rule are fixed in
preregistration/fly_subsplit_rule.md and implemented here as written; none of it
is tuned against preservation results.

Inputs (data/raw/flywire): neuron_annotations.tsv,
proofread_connections_783.feather.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.cluster.hierarchy import fcluster, linkage

RAW = Path(__file__).resolve().parents[2] / "data/raw/flywire"

# the pre-registered ladder parameter grid; f = 1 is all-singletons
F_GRID = (0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.55, 0.70, 0.85, 1.00)
LINEAGE_COL = "ito_lee_hemilineage"


def load_scope() -> pd.DataFrame:
    """Central-brain neurons with an Ito-Lee hemilineage and a soma position."""
    d = pd.read_csv(RAW / "neuron_annotations.tsv", sep="\t", low_memory=False)
    d = d[(d.super_class == "central") & d[LINEAGE_COL].notna()]
    d = d[d[["soma_x", "soma_y", "soma_z"]].notna().all(axis=1)]
    return d.reset_index(drop=True)


def load_adjacency(meta: pd.DataFrame) -> sp.csr_matrix:
    """Weighted directed adjacency A[j, i] = synapses from j to i, in meta order."""
    c = pd.read_feather(RAW / "proofread_connections_783.feather")
    cols = {x.lower(): x for x in c.columns}
    pre = cols.get("pre_pt_root_id") or cols.get("pre_root_id") or cols.get("pre")
    post = cols.get("post_pt_root_id") or cols.get("post_root_id") or cols.get("post")
    wt = cols.get("syn_count") or cols.get("count") or cols.get("weight")
    idx = pd.Series(np.arange(len(meta)), index=meta.root_id.to_numpy())
    j = idx.reindex(c[pre].to_numpy()).to_numpy()
    i = idx.reindex(c[post].to_numpy()).to_numpy()
    w = c[wt].to_numpy(float)
    keep = ~(np.isnan(j) | np.isnan(i))
    n = len(meta)
    return sp.csr_matrix((w[keep], (j[keep].astype(np.int64), i[keep].astype(np.int64))),
                         shape=(n, n))


def spectral_radius(A: sp.csr_matrix, n_iter: int = 200) -> float:
    """Power iteration on |A|; adequate for scaling the operator."""
    M = abs(A)
    x = np.random.default_rng(0).random(A.shape[0])
    x /= np.linalg.norm(x)
    lam = 0.0
    for _ in range(n_iter):
        y = M @ x
        nrm = np.linalg.norm(y)
        if nrm == 0:
            return 0.0
        x = y / nrm
        lam = nrm
    return float(lam)


def propagation_rows(A: sp.csr_matrix, sources: np.ndarray, alpha: float,
                     rho: float | None = None, tol: float = 1e-6) -> np.ndarray:
    """Rows of the resolvent (I - alpha*A/rho)^-1 - I for the given sources.

    Neumann series sum_{t>=1} (alpha*A/rho)^t, truncated at the first t with
    alpha^t < tol (or earlier once a term's max entry falls below tol*1e-3),
    using sparse matvecs only. For alpha >= 1 the series does not converge and
    a fixed depth of 50 is used.
    """
    if rho is None:
        rho = spectral_radius(A)
    if rho <= 0:
        return np.zeros((len(sources), A.shape[0]))
    At = (A / rho).T.tocsr()
    depth = max(1, int(np.ceil(np.log(tol) / np.log(alpha)))) if alpha < 1 else 50
    out = np.zeros((len(sources), A.shape[0]))
    for r, s in enumerate(sources):
        x = np.zeros(A.shape[0]); x[s] = 1.0
        acc = np.zeros(A.shape[0])
        for t in range(1, depth + 1):
            x = alpha * (At @ x)
            acc += x
            if np.abs(x).max() < tol * 1e-3:
                break
        out[r] = acc
    return out


def stratified_sources(A: sp.csr_matrix, n_src: int, seed: int = 0
                       ) -> tuple[np.ndarray, list[int]]:
    """Sample sources stratified by out-degree decile, equal count per decile.

    Propagation columns vary a lot in density; a uniform sample would be
    dominated by low-out-degree neurons.
    """
    outdeg = np.asarray((A > 0).sum(axis=1)).ravel()
    rng = np.random.default_rng(seed)
    edges = np.quantile(outdeg, np.linspace(0, 1, 11))
    per = max(1, n_src // 10)
    picked, counts = [], []
    for b in range(10):
        lo, hi = edges[b], edges[b + 1]
        m = np.flatnonzero((outdeg >= lo) & (outdeg <= hi if b == 9 else outdeg < hi))
        m = m[outdeg[m] > 0]
        take = min(per, len(m))
        picked.append(rng.choice(m, take, replace=False))
        counts.append(int(take))
    return np.concatenate(picked), counts


# ------------------------------------------------------------------ partitions

def tier_a(meta: pd.DataFrame, with_side: bool) -> np.ndarray:
    """Tier A labels: hemilineage, or hemilineage x side."""
    keys = meta[LINEAGE_COL].astype(str)
    if with_side:
        keys = keys + "|" + meta["side"].astype(str)
    return pd.factorize(keys)[0]


def subsplit(meta: pd.DataFrame, f: float) -> np.ndarray:
    """Tier B labels: split each (hemilineage, side) group of size n into
    max(1, round(f*n)) clusters by Ward linkage on soma position z-scored once
    over the whole in-scope set."""
    xyz = meta[["soma_x", "soma_y", "soma_z"]].to_numpy(float)
    xyz = (xyz - xyz.mean(0)) / xyz.std(0)          # one global scaling
    base = tier_a(meta, with_side=True)
    labels = np.full(len(meta), -1, dtype=np.int64)
    nxt = 0
    for g in np.unique(base):
        m = np.flatnonzero(base == g)
        n = m.size
        want = max(1, int(round(f * n)))
        if n == 1 or want >= n:
            labels[m] = np.arange(nxt, nxt + n); nxt += n
            continue
        if want == 1:
            labels[m] = nxt; nxt += 1
            continue
        cl = fcluster(linkage(xyz[m], method="ward", metric="euclidean"),
                      t=want, criterion="maxclust")
        labels[m] = nxt + cl - 1
        nxt += cl.max()
    return labels


def size_matched_random(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Identical cell-size profile, neurons randomly assigned."""
    return rng.permutation(labels)


def quotient(A: sp.csr_matrix, labels: np.ndarray) -> sp.csr_matrix:
    """Q[a,b] = mean out-weight from a member of cell a into cell b."""
    n = A.shape[0]
    k = int(labels.max() + 1)
    H = sp.csr_matrix((np.ones(n), (np.arange(n), labels)), shape=(n, k))
    sizes = np.asarray(H.sum(axis=0)).ravel()
    Q = (H.T @ A @ H).tocsr()
    return sp.diags(1.0 / np.maximum(sizes, 1)) @ Q
