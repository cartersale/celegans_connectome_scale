"""MICrONS minnie65 (v1300) axon-proofread cells, their type labels and edges.

Scope: cells with `axon_cleaned == "t"`. The propagation operator walks along
outgoing edges, so a cell whose axon is not proofread has a structurally empty
output row; such cells are usable as synaptic targets but not as sources.

Two graphs are built from the same edge table:
  wide rows   out-rows of the proofread cells over every postsynaptic partner in
              the volume (`load_wide_rows`). Used only for merge criteria, which
              depend only on the rows of the two cells being merged.
  closed      square adjacency among the proofread cells (`load_closed`), the
              graph that is quotiented and scored.

Labels are population categories (m-types with tens to hundreds of members), so
the chance rate that two random cells share a label bounds any enrichment ratio;
`chance_rate` returns that bound.

Label provenance: `cell_type_source` and `mtype_source` are kept so that label
sets assigned using connectivity can be excluded. For inhibitory cells,
`aibs_metamodel_mtypes_v661_v2` and `allen_column_mtypes_v2` used connectivity
(`CIRCULAR_FOR_INHIBITORY`); excitatory labels under those sources did not.

Inputs (data/raw/microns): aibs_cell_info.csv.gz (+ _header.csv),
connections_with_nuclei.csv.gz (+ _header.csv).
Outputs: edges_pre_proofread.feather, a cache of every edge whose presynaptic
cell is axon-proofread (`build_edge_cache`).
"""
from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

RAW = Path(__file__).resolve().parents[2] / "data/raw/microns"

CELL_INFO = RAW / "aibs_cell_info.csv.gz"
CELL_INFO_HEADER = RAW / "aibs_cell_info_header.csv"
CONNECTIONS = RAW / "connections_with_nuclei.csv.gz"
EDGE_CACHE = RAW / "edges_pre_proofread.feather"

# label sources that used connectivity in assignment, for inhibitory cells
CIRCULAR_FOR_INHIBITORY = ("aibs_metamodel_mtypes_v661_v2", "allen_column_mtypes_v2")


def _header(path: Path) -> list[str]:
    return [ln.split(",")[0] for ln in path.read_text().strip().split("\n")]


def load_cells(axon_proofread: bool = True) -> pd.DataFrame:
    """v1300 cell info, by default restricted to axon-proofread cells."""
    d = pd.read_csv(CELL_INFO, names=_header(CELL_INFO_HEADER), low_memory=False)
    d = d[d.pt_root_id != 0]
    if axon_proofread:
        d = d[d.axon_cleaned == "t"]
    return d.reset_index(drop=True)


def label_arm(cells: pd.DataFrame, column: str, source: str | None = None,
              broad_type: str | None = None) -> pd.DataFrame:
    """One labelling arm: a label column, optionally filtered by provenance.

    `source` restricts to a single `{column}_source` value; `broad_type`
    restricts to "excitatory" or "inhibitory".
    """
    d = cells[cells[column].notna()]
    if source is not None:
        d = d[d[f"{column}_source"] == source]
    if broad_type is not None:
        d = d[d.broad_type == broad_type]
    return d.reset_index(drop=True)


def chance_rate(labels: np.ndarray | pd.Series) -> tuple[float, float]:
    """P(two random distinct neurons share a label), and its reciprocal.

    The reciprocal is the enrichment ratio attained if every selected merge
    joins two same-label neurons, i.e. the ceiling for that label set.
    """
    v = pd.Series(labels).dropna()
    n = len(v)
    c = v.value_counts().to_numpy(float)
    p = float((c * (c - 1)).sum() / (n * (n - 1)))
    return p, (1.0 / p if p > 0 else np.inf)


# ------------------------------------------------------------------ edges

def _stream_edges(keep_pre: set[int], chunk: int = 2_000_000):
    """Yield chunks of the neuron-level connection table with pre in keep_pre."""
    cols = _header(RAW / "connections_with_nuclei_header.csv")
    with gzip.open(CONNECTIONS, "rt") as fh:
        for c in pd.read_csv(fh, names=cols, chunksize=chunk, low_memory=False):
            c = c[c.pre_pt_root_id.isin(keep_pre) & (c.post_pt_root_id != 0)]
            if len(c):
                yield c[["pre_pt_root_id", "post_pt_root_id", "n_syn"]]


def build_edge_cache(cells: pd.DataFrame, force: bool = False) -> pd.DataFrame:
    """Edges with presynaptic cell in `cells`, from one streaming pass, cached.

    Every label arm is a subset of the axon-proofread set, so one cache built
    from that set serves all arms. The cache is returned whenever the file
    exists, regardless of which `cells` are passed; rebuild with `force=True`.
    """
    if EDGE_CACHE.exists() and not force:
        return pd.read_feather(EDGE_CACHE)
    src = {int(x) for x in cells.pt_root_id}
    parts = list(_stream_edges(src))
    e = (pd.concat(parts, ignore_index=True) if parts
         else pd.DataFrame(columns=["pre_pt_root_id", "post_pt_root_id", "n_syn"]))
    EDGE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    e.to_feather(EDGE_CACHE)
    return e


def load_wide_rows(cells: pd.DataFrame, edges: pd.DataFrame
                   ) -> tuple[sp.csr_matrix, np.ndarray]:
    """Out-rows of `cells` over every postsynaptic partner in the volume.

    Returns (W, target_roots) with W[i, t] = synapses from cells.iloc[i] onto
    target_roots[t]. Used for the merge criterion only; it is not a closed
    operator and cannot be quotiented.
    """
    roots = cells.pt_root_id.to_numpy()
    e = edges[edges.pre_pt_root_id.isin(set(int(x) for x in roots))]
    tgt_roots = np.sort(e.post_pt_root_id.unique())
    ri = pd.Series(np.arange(len(roots)), index=roots)
    ci = pd.Series(np.arange(len(tgt_roots)), index=tgt_roots)
    i = ri.reindex(e.pre_pt_root_id.to_numpy()).to_numpy()
    j = ci.reindex(e.post_pt_root_id.to_numpy()).to_numpy()
    W = sp.csr_matrix((e.n_syn.to_numpy(float), (i, j)),
                      shape=(len(roots), len(tgt_roots)))
    return W, tgt_roots


def load_closed(cells: pd.DataFrame, edges: pd.DataFrame) -> sp.csr_matrix:
    """Square adjacency among `cells`, A[j, i] = synapses from j to i.

    Orientation matches `conres.flywire.load_adjacency`, so the propagation and
    quotient machinery carries over unchanged.
    """
    roots = cells.pt_root_id.to_numpy()
    s = set(int(x) for x in roots)
    e = edges[edges.pre_pt_root_id.isin(s) & edges.post_pt_root_id.isin(s)]
    n = len(roots)
    idx = pd.Series(np.arange(n), index=roots)
    j = idx.reindex(e.pre_pt_root_id.to_numpy()).to_numpy()
    i = idx.reindex(e.post_pt_root_id.to_numpy()).to_numpy()
    return sp.csr_matrix((e.n_syn.to_numpy(float), (j, i)), shape=(n, n))


def degrees(W: sp.csr_matrix, A: sp.csr_matrix) -> dict[str, np.ndarray]:
    """Degree measures for the degree-floor sweep.

    `wide_out`: number of distinct postsynaptic partners over the whole volume.
    `closed`: in- plus out-degree within the closed proofread subgraph.
    """
    return {"wide_out": np.asarray((W > 0).sum(axis=1)).ravel(),
            "closed": (np.asarray((A > 0).sum(axis=1)).ravel()
                       + np.asarray((A > 0).sum(axis=0)).ravel())}
