"""Worm anatomy in micrometres, for drawing on the animal.

All coordinates are columns (ML, AP, DV) in um, in the OpenWorm NeuroML frame.

Sources:
  * WormBase VirtualWorm model (Grove and Sternberg, Caltech; public domain),
    data/raw/virtualworm/Virtual_Worm_February_2012.blend, recovered from the
    Internet Archive. Supplies the cuticle, pharynx and intestine meshes.
  * OpenWorm NeuroML cells, data/raw/openworm_neuroml/<cell>.cell.nml. Supplies
    neuron somata and neurites.

Reading the .blend (Blender 2.49 format, via the optional blender-asset-tracer):
mesh vertices are in local space and are multiplied by their object's obmat,
read from raw bytes because the library's array_index indexes a block's struct
array rather than a field's array. Blender world units are mapped to um by
SCALE and AP_OFFSET, fitted by matching the cuticle's anterior-posterior extent
to the NeuroML neurite cloud; the ML and DV axes then align with no further
fitting. The meshes are cached to anatomy_um.npz on first use.
"""
from __future__ import annotations

import json
import re
import struct

import numpy as np

from conres.paths import REPO_ROOT

BLEND = REPO_ROOT / "data/raw/virtualworm/Virtual_Worm_February_2012.blend"
CACHE = REPO_ROOT / "data/raw/virtualworm/anatomy_um.npz"
SOMA_CACHE = REPO_ROOT / "data/raw/virtualworm/soma_um.json"
NML = REPO_ROOT / "data/raw/openworm_neuroml"
PARTS = {"cuticle": "MECuticle", "pharynx": "MEPharynx", "intestine": "MEIntestine"}

# blend world units -> micrometres, in the NeuroML frame
SCALE, AP_OFFSET = 97.813, -4.21


def _extract() -> dict[str, np.ndarray]:
    from blender_asset_tracer import blendfile          # optional, only on cache miss

    bf = blendfile.open_cached(BLEND)
    psize = bf.header.pointer_size
    meshes = {b.get((b"id", b"name"), as_str=True): b for b in bf.blocks if b.code == b"ME"}
    by_addr = {b.addr_old: n for n, b in meshes.items()}

    def raw_floats(block, name: bytes, n: int) -> np.ndarray:
        _, off = block.dna_type.field_from_path(psize, name)
        fh = block.bfile.fileobj
        fh.seek(block.file_offset + off)
        return np.array(struct.unpack("<%df" % n, fh.read(4 * n)))

    obmat = {}
    for b in bf.blocks:
        if b.code == b"OB":
            nm = by_addr.get(b.get(b"data", 0))
            if nm in PARTS.values():
                obmat[nm] = raw_floats(b, b"obmat", 16).reshape(4, 4)

    out = {}
    for part, mesh in PARTS.items():
        mv = meshes[mesh].get_pointer(b"mvert")
        fh = mv.bfile.fileobj
        fh.seek(mv.file_offset)
        stride = mv.size // mv.count                    # MVert: co[3]f, no[3]h, 6 chars
        raw = np.frombuffer(fh.read(mv.size), dtype=np.uint8).reshape(mv.count, stride)
        local = raw[:, :12].copy().view(np.float32).reshape(mv.count, 3).astype(float)
        M = obmat[mesh]
        with np.errstate(all="ignore"):        # float32 denormals in the mesh
            world = local @ M[:3, :3] + M[3, :3]
        world *= SCALE
        world[:, 1] += AP_OFFSET
        out[part] = world                               # columns: (ML, AP, DV) in um
    return out


def anatomy() -> dict[str, np.ndarray]:
    """Cuticle, pharynx and intestine vertices in micrometres, columns (ML, AP, DV)."""
    if CACHE.exists():
        z = np.load(CACHE)
        return {k: z[k] for k in z.files}
    out = _extract()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, **out)
    return out


def silhouette(part: str = "cuticle", view: str = "lateral", nbin: int | None = None,
               smooth: int = 5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Outline of a mesh in one projection, as (ap, lower, upper).

    Slices the mesh along the AP axis into `nbin` bins, takes the min and max
    off-axis coordinate in each, and smooths midline and half-width with a
    `smooth`-point moving average. No taper or margin is added.

    view: 'lateral' takes dorsoventral extent, 'dorsal' takes mediolateral.

    The default bin count is coarser for 'lateral' because the animal is posed
    in a sinusoid: in lateral projection an axis-aligned slice cuts the tube
    obliquely wherever the body is steep, and with ~1000 cuticle vertices fine
    bins are noisy.
    """
    if nbin is None:
        nbin = 45 if view == "lateral" else 190
    V = anatomy()[part]
    ap = V[:, 1]
    off = V[:, 2] if view == "lateral" else V[:, 0]
    edges = np.linspace(ap.min(), ap.max(), nbin + 1)
    s, lo, hi = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (ap >= a) & (ap < b)
        if m.sum() < 2:
            continue
        s.append((a + b) / 2)
        lo.append(off[m].min())
        hi.append(off[m].max())
    s, lo, hi = map(np.asarray, (s, lo, hi))
    if smooth > 1:
        k = np.ones(smooth) / smooth
        pad = lambda v: np.convolve(np.pad(v, smooth, mode="edge"), k, "same")[smooth:-smooth]
        mid, half = pad((lo + hi) / 2), pad((hi - lo) / 2)
        lo, hi = mid - half, mid + half
    return s, lo, hi


# NeuroML morphology: <segment> carries an optional proximal and a distal point,
# a segment without a proximal one continuing from its parent's distal end
_SOMA_RE = re.compile(r'name="Seg0_soma_0">\s*<proximal x="([-\d.eE]+)" '
                      r'y="([-\d.eE]+)" z="([-\d.eE]+)"')
_PT_RE = re.compile(r'<(proximal|distal) x="([-\d.eE]+)" y="([-\d.eE]+)" z="([-\d.eE]+)"')


def soma_xyz(nodes: list[str]) -> np.ndarray:
    """Soma position per cell, columns (ML, AP, DV) in micrometres.

    Taken from the NeuroML soma where a cell file exists. NeuroML carries
    neurons only, so other cells (sheath cells, glia) fall back to
    data/raw/virtualworm/soma_um.json, written by scripts/export_soma_um.py.
    Cells in neither source are nan.
    """
    mesh = json.loads(SOMA_CACHE.read_text()) if SOMA_CACHE.exists() else {}
    out = np.full((len(nodes), 3), np.nan)
    for k, n in enumerate(nodes):
        f = NML / f"{n}.cell.nml"
        m = _SOMA_RE.search(f.read_text()) if f.exists() else None
        if m:
            out[k] = [float(m.group(1)), float(m.group(2)), float(m.group(3))]
        elif n in mesh:
            out[k] = mesh[n]
    return out


def neurites(nodes: list[str]) -> np.ndarray:
    """Every morphology segment of the named cells, as (n, 2, 3) endpoint pairs
    in micrometres, columns (ML, AP, DV).

    NeuroML only; cells without a NeuroML file are skipped silently.
    """
    out = []
    for n in nodes:
        f = NML / f"{n}.cell.nml"
        if not f.exists():
            continue
        cur = None
        for kind, x, y, z in _PT_RE.findall(f.read_text()):
            pt = (float(x), float(y), float(z))
            if kind == "distal" and cur is not None:
                out.append([cur, pt])
            cur = pt
    return np.array(out) if out else np.empty((0, 2, 3))


_DIST_RE = re.compile(r'<distal x="(-?[\d.eE+-]+)" y="(-?[\d.eE+-]+)" z="(-?[\d.eE+-]+)"')


def soma_ap(nodes: list[str]) -> np.ndarray:
    """Anterior-posterior position (um) of the first NeuroML <distal> point per cell.

    This is the soma segment's end point, not the soma centre used by
    `soma_xyz`. Cells without a NeuroML file are nan.
    """
    ap = np.full(len(nodes), np.nan)
    where = {n: k for k, n in enumerate(nodes)}
    for f in NML.glob("*.cell.nml"):
        name = f.name.split(".")[0]
        if name in where:
            m = _DIST_RE.search(f.read_text())
            if m:
                ap[where[name]] = float(m.group(2))
    return ap
