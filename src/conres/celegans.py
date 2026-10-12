"""C. elegans connectomes, cell-name normalisation and neuron-class labels.

Adult: Cook et al. 2019 hermaphrodite full edge list (chemical and gap
junction), from the OpenWorm ConnectomeToolbox mirror, read from
data/raw/celegans/herm_full_edgelist.csv. `build_matrix` returns a dense
weighted adjacency A[i, j] = synapse count from i to j and the sorted node
names.

Development: the eight Witvliet et al. 2021 datasets
(data/raw/celegans/witvliet_2020_<n> <stage>.xlsx), loaded on a caller-supplied
node set by `witvliet_matrix`; `witvliet_common_neurons` gives the neurons
present in all eight.

Classes are inferred from names alone (`assign_classes`): numbered ventral-cord
families collapse to their family, and a positional suffix (L/R, D/V and their
combinations) is stripped only when the mirror-partner cells it implies exist in
the same roster.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from conres.paths import REPO_ROOT

RAW = REPO_ROOT / "data/raw/celegans"

# The ventral-cord motor neuron families are the only ones whose trailing digits
# are member indices rather than part of the class name: VA1..VA12 are one class,
# but M1..M5 and I1..I6 are five and six distinct pharyngeal classes.
NUMBERED_FAMILIES = ("VA", "VB", "VC", "VD", "DA", "DB", "DD", "AS")
# Positional suffixes: left/right, dorsal/ventral, and the four-way combinations.
POSITIONAL = ("DL", "DR", "VL", "VR", "L", "R", "D", "V")

# Case matters: the non-neuronal names are lower-case initial (marginal cells
# mc1DL, muscles pm3D, dBWML1), while MCL and MCR are pharyngeal NEURONS. A
# case-insensitive match here silently drops them from the connectome.
#
# Cook and Witvliet name the body-wall muscles differently -- dBWML1 against
# BWM-DL01 -- so a pattern written against one release silently admits 17 muscles
# from the other. Both spellings are matched here.
MUSCLE_RE = re.compile(r"^(dBWM|vBWM|BWM|pm|vm|um|mc|mu|g[0-9]|exc|hyp|int)")
GLIA_RE = re.compile(r"^([A-Z]+sh|[A-Z]+so|GLR)")


def normalise_name(name: str) -> str:
    """Strip padding and zero-padded indices: 'AS01 ' -> 'AS1'."""
    name = str(name).strip()
    m = re.match(r"^([A-Za-z]+)0+([0-9]+)$", name)
    return f"{m.group(1)}{m.group(2)}" if m else name


MIRROR = {"L": "R", "R": "L", "DL": "DR", "DR": "DL",
          "VL": "VR", "VR": "VL", "D": "D", "V": "V"}


def _candidate_splits(name: str) -> list[tuple[str, str]]:
    """Every way the name could be read as base + positional suffix.

    Deliberately ambiguous: RMDL is both RMD+L and RM+DL, and only looking at
    which other cells exist can say which reading is the real one.
    """
    out = []
    for suf in POSITIONAL:
        if name.endswith(suf) and len(name) - len(suf) >= 2:
            out.append((name[: -len(suf)], suf))
    return out


def _is_symmetry_group(suffixes: set[str]) -> bool:
    """True if every suffix has its mirror partner present (at least two suffixes).

    {L,R}, {DL,DR,VL,VR} and {D,V,L,R} (RME) pass. The spurious base 'AV'
    collected from AVDL, AVDR and AVL gives {DL,DR,L}, where L has no R, so it
    fails: AVL is its own neuron and AVD its own class.
    """
    if not suffixes <= set(MIRROR) or len(suffixes) < 2:
        return False
    return all(MIRROR[x] in suffixes for x in suffixes)


def assign_classes(nodes: list[str]) -> dict[str, str]:
    """Map each (normalised) cell name to its class, resolved against the whole roster.

    Where two readings both give valid groups (RMD with six members vs RMDD
    with two) the larger group wins, since the finer split comes from reading
    part of the class name as a suffix. Unclaimed cells are singleton classes.
    """
    nodes = [normalise_name(n) for n in nodes]
    cls: dict[str, str] = {}
    cand: dict[str, dict[str, str]] = {}
    for n in nodes:
        m = re.match(r"^([A-Za-z]+)([0-9]+)$", n)
        if m and m.group(1).upper() in NUMBERED_FAMILIES:
            cls[n] = m.group(1).upper()
            continue
        for base, suf in _candidate_splits(n):
            cand.setdefault(base, {})[suf] = n

    groups = [(base, mem) for base, mem in cand.items() if _is_symmetry_group(set(mem))]
    groups.sort(key=lambda bm: (-len(bm[1]), -len(bm[0])))
    for base, members in groups:
        free = {suf: n for suf, n in members.items() if n not in cls}
        if len(free) > 1 and _is_symmetry_group(set(free)):
            for n in free.values():
                cls[n] = base
    for n in nodes:                      # anything unclaimed is its own class
        cls.setdefault(n, n)
    return cls


def neuron_class(name: str) -> str:
    """Single-name convenience wrapper; prefer assign_classes for a roster."""
    return assign_classes([name])[normalise_name(name)]


def is_muscle(name: str) -> bool:
    return bool(MUSCLE_RE.match(normalise_name(name)))


def bilateral_pair(name: str) -> str | None:
    """Name with a trailing L/R removed, or None if it has none.

    Purely lexical: the partner cell is not checked for. `partitions` turns
    None into a singleton.
    """
    name = normalise_name(name)
    if name.endswith(("L", "R")) and len(name) >= 4:
        return name[:-1]
    return None


def load_edges(kind: str = "all") -> pd.DataFrame:
    """Cook 2019 hermaphrodite edge list. kind in {chemical, electrical, all}."""
    e = pd.read_csv(RAW / "herm_full_edgelist.csv")
    e.columns = [c.strip() for c in e.columns]
    for c in ("Source", "Target", "Type"):
        e[c] = e[c].astype(str).map(normalise_name) if c != "Type" else e[c].astype(str).str.strip()
    e["Weight"] = pd.to_numeric(e["Weight"], errors="coerce").fillna(0.0)
    if kind != "all":
        e = e[e["Type"] == kind]
    return e.reset_index(drop=True)


def build_matrix(kind: str = "chemical", *, neurons_only: bool = True
                 ) -> tuple[np.ndarray, list[str]]:
    """Weighted adjacency A[i, j] = synapses from i to j.

    kind='all' sums the edge list as given. The Cook release lists gap
    junctions in both orientations (all but five of 2,157 entries), so they
    enter A almost exactly symmetrically. kind='electrical' additionally
    symmetrises with max(A, A^T).
    """
    e = load_edges(kind)
    nodes = sorted(set(e.Source) | set(e.Target))
    if neurons_only:
        nodes = [n for n in nodes if not is_muscle(n)]
        e = e[e.Source.isin(nodes) & e.Target.isin(nodes)]
    idx = {n: i for i, n in enumerate(nodes)}
    A = np.zeros((len(nodes), len(nodes)))
    for s, t, w in zip(e.Source, e.Target, e.Weight):
        A[idx[s], idx[t]] += w
    if kind == "electrical":
        A = np.maximum(A, A.T)
    return A, nodes


def layer_matrix(nodes: list[str], kind: str) -> np.ndarray:
    """Adjacency of one contact type ('chemical' or 'electrical') on a given node list.

    Entries are summed as listed, with no symmetrisation, so the chemical and
    electrical layers on the build_matrix('all') node list add up to that matrix.
    """
    idx = {n: i for i, n in enumerate(nodes)}
    A = np.zeros((len(nodes), len(nodes)))
    e = load_edges(kind)
    for s, t, w in zip(e.Source, e.Target, e.Weight):
        if s in idx and t in idx:
            A[idx[s], idx[t]] += w
    return A


def partitions(nodes: list[str]) -> dict[str, np.ndarray]:
    """Integer label vectors over `nodes`: 'class' (assign_classes) and 'bilateral'."""
    out: dict[str, np.ndarray] = {}

    def encode(keys: list[str]) -> np.ndarray:
        uniq = {k: i for i, k in enumerate(sorted(set(keys)))}
        return np.array([uniq[k] for k in keys], dtype=int)

    cls = assign_classes(nodes)
    out["class"] = encode([cls[normalise_name(n)] for n in nodes])
    out["bilateral"] = encode([bilateral_pair(n) or f"__{n}" for n in nodes])
    return out


def summary() -> None:
    A, nodes = build_matrix("chemical", neurons_only=True)
    mapping = assign_classes(nodes)
    cls = [mapping[n] for n in nodes]
    print(f"neurons in edge list : {len(nodes)}")
    print(f"distinct classes     : {len(set(cls))}")
    sizes = pd.Series(cls).value_counts()
    print(f"class size profile   : {dict(sizes.value_counts().sort_index())}  (size: how many classes)")
    print(f"largest classes      : {dict(sizes.head(8))}")
    print(f"singleton classes    : {(sizes == 1).sum()}")


# --------------------------------------------------------------- development

WITVLIET = "witvliet_2020"


def witvliet_files() -> list[tuple[int, str, Path]]:
    """The eight developmental connectomes, ordered birth -> adult."""
    out = []
    for p in RAW.glob(f"{WITVLIET}_*.xlsx"):
        m = re.match(rf"{WITVLIET}_(\d+) (\w+)", p.name)
        if m:
            out.append((int(m.group(1)), m.group(2), p))
    return sorted(out)


def load_witvliet(path: Path, kind: str = "all") -> pd.DataFrame:
    d = pd.read_excel(path)
    d["pre"] = d["pre"].map(normalise_name)
    d["post"] = d["post"].map(normalise_name)
    d["synapses"] = pd.to_numeric(d["synapses"], errors="coerce").fillna(0.0)
    if kind != "all":
        d = d[d["type"] == kind]
    return d.reset_index(drop=True)


def is_glia(name: str) -> bool:
    """Sheath, socket and GLR glia (CEPshDL, AMsoL, GLRDR, ...)."""
    return bool(GLIA_RE.match(normalise_name(name)))


def witvliet_common_neurons(kind: str = "all", exclude_glia: bool = False) -> list[str]:
    """Sorted cells (muscles excluded) present in all eight Witvliet datasets.

    A fixed node set keeps the partition, its size profile and k identical
    across developmental stages. The set includes five glia (four CEP sheath
    cells and GLRDR) unless `exclude_glia`.
    """
    sets = []
    for _, _, p in witvliet_files():
        d = load_witvliet(p, kind)
        sets.append({n for n in set(d.pre) | set(d.post)
                     if not is_muscle(n) and not (exclude_glia and is_glia(n))})
    return sorted(set.intersection(*sets))


def witvliet_matrix(path: Path, nodes: list[str], kind: str = "all") -> np.ndarray:
    """Weighted adjacency on a fixed node set; A[i, j] = synapses from i to j."""
    d = load_witvliet(path, kind)
    idx = {n: i for i, n in enumerate(nodes)}
    d = d[d.pre.isin(idx) & d.post.isin(idx)]
    A = np.zeros((len(nodes), len(nodes)))
    for s, t, w in zip(d.pre, d.post, d.synapses):
        A[idx[s], idx[t]] += w
    if kind == "electrical":
        A = np.maximum(A, A.T)
    return A


if __name__ == "__main__":
    summary()
