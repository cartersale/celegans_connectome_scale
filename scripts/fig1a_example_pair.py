"""Draw the example pair: a lateral-view worm with a call-out of the URYDL -> IL1DL -> URADL circuit.

Left: the body outline is the envelope of the reconstructed neurite cloud
(1st/99th percentile of dorsoventral z per AP bin, smoothed, tapered at both
ends), since the dorsal and ventral cords run just under the cuticle. The
neurites are drawn as one rasterized layer clipped to that outline.
Morphologies are the VirtualWorm reconstruction (Grove & Sternberg,
WormBase/Caltech, public domain) as converted to NeuroML2 by OpenWorm.

Right: the circuit is drawn as a graph rather than as morphology because the
three somata sit close together at the anterior tip and their processes
superimpose in every projection. The neuron names and synapse counts (3, 10)
are hard-coded.

Inputs:  data/raw/openworm_neuroml/*.cell.nml.
Outputs: figures/fig1/fig1a_example_pair.svg, plus a PNG preview in $CONRES_CHECK_DIR (if set).
"""
from __future__ import annotations

import re
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon

ROOT = Path(__file__).resolve().parents[1]
NML = ROOT / "data/raw/openworm_neuroml"
OUT = ROOT / "figures/fig1"
CHECKDIR = Path(os.environ["CONRES_CHECK_DIR"]) if os.environ.get("CONRES_CHECK_DIR") else None

# Arial; SVG text stays as editable text (svg.fonttype none), PDF embeds TrueType.
plt.rcParams.update({
    "font.family": "Arial",
    "svg.fonttype": "none",
    "mathtext.fontset": "custom",   # keep alpha, rho and superscripts in Arial
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
    "pdf.fonttype": 42,
})

STIM, MID, RESP = "#2a5fd6", "#2e9e52", "#d1332e"
GREY, FAINT = "#8b9196", "#c9ced2"
PT = re.compile(r'<(?:proximal|distal) x="(-?[\d.eE+-]+)" y="(-?[\d.eE+-]+)" z="(-?[\d.eE+-]+)"')
SEG = re.compile(r'<segment id="(\d+)"[^>]*>(.*?)</segment>', re.S)
PROX = re.compile(r'<proximal x="(-?[\d.eE+-]+)" y="(-?[\d.eE+-]+)" z="(-?[\d.eE+-]+)"')
DIST = re.compile(r'<distal x="(-?[\d.eE+-]+)" y="(-?[\d.eE+-]+)" z="(-?[\d.eE+-]+)"')
PAR = re.compile(r'<parent segment="(\d+)"')


def neurites() -> tuple[np.ndarray, np.ndarray]:
    """Every reconstructed segment as a (proximal, distal) pair, plus all points."""
    segs = []
    for f in NML.glob("*.cell.nml"):
        tip = {}
        for sid, body in SEG.findall(f.read_text()):
            d = np.array([float(v) for v in DIST.search(body).groups()])
            m, par = PROX.search(body), PAR.search(body)
            if m is not None:
                pv = np.array([float(v) for v in m.groups()])
            elif par is not None and par.group(1) in tip:
                pv = tip[par.group(1)]
            else:
                pv = d
            tip[sid] = d
            segs.append((pv, d))
    S = np.asarray(segs)
    return S, S.reshape(-1, 3)


def body_outline(P: np.ndarray, nbin: int = 120, smooth: int = 9):
    """Smoothed dorsal and ventral margins of the neurite cloud, tapered to a
    point at nose and tail so the silhouette closes the way the animal does."""
    y, z = P[:, 1], P[:, 2]
    edges = np.linspace(y.min(), y.max(), nbin + 1)
    s, lo, hi = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (y >= a) & (y < b)
        if m.sum() < 3:
            continue
        s.append((a + b) / 2)
        lo.append(np.percentile(z[m], 1.0))
        hi.append(np.percentile(z[m], 99.0))
    s, lo, hi = map(np.asarray, (s, lo, hi))

    k = np.ones(smooth) / smooth
    pad = lambda v: np.convolve(np.pad(v, smooth, mode="edge"), k, "same")[smooth:-smooth]
    mid, half = pad((lo + hi) / 2), pad((hi - lo) / 2)

    # square-root taper over the anterior 12% and posterior 30% of the length
    u = (s - s.min()) / np.ptp(s)
    taper = np.minimum(np.sqrt(np.clip(u / 0.12, 0, 1)), np.sqrt(np.clip((1 - u) / 0.30, 0, 1)))
    half = half * np.clip(taper, 0.02, 1.0)
    return s, mid - half, mid + half


def main() -> None:
    S, P = neurites()
    s, ventral, dorsal = body_outline(P)

    fig = plt.figure(figsize=(5.2, 1.85))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(0, 520)
    ax.set_ylim(0, 185)

    # ---------------------------------------------------------------- animal
    # map the reconstruction into the left third of the panel, anterior left
    wx0, wx1, wcy, wscale = 12.0, 198.0, 96.0, 0.58
    fx = lambda v: wx0 + (v - s.min()) / np.ptp(s) * (wx1 - wx0)
    fy = lambda v: wcy + v * wscale

    verts = np.concatenate([
        np.column_stack([fx(s), fy(dorsal)]),
        np.column_stack([fx(s[::-1]), fy(ventral[::-1])]),
    ])
    silhouette = Polygon(verts, closed=True, facecolor="#eceeef",
                         edgecolor="#6f767c", lw=0.8, zorder=1)
    ax.add_patch(silhouette)

    # The reconstruction, faint, clipped to the body it defines. ~16k segments
    # as vector paths are too heavy for an editor, so this one layer is
    # rasterized at 600 dpi; everything else stays vector.
    lc = LineCollection(
        np.stack([np.column_stack([fx(S[:, k, 1]), fy(S[:, k, 2])]) for k in (0, 1)], axis=1),
        colors="#9aa2a9", linewidths=0.18, alpha=0.75, zorder=2, rasterized=True)
    lc.set_clip_path(silhouette)
    ax.add_collection(lc)

    # locator on the nerve-ring / anterior sensory region
    hx, hy = fx(-284.0), fy(52.0)
    lw_, lh_ = 13.0, 13.0
    ax.add_patch(plt.Rectangle((hx - lw_, hy - lh_), 2 * lw_, 2 * lh_, fill=False,
                               ec="#3f464c", lw=0.7, ls=(0, (2.2, 1.6)), zorder=6))
    ax.plot([hx], [hy], "o", ms=2.0, color=STIM, zorder=7)

    ax.annotate("anterior", xy=(fx(s.min()) + 2, fy(-30)), fontsize=5.6, color=GREY,
                style="italic", ha="left", va="top")
    ax.text(14, 8, "lateral view, dorsal up", fontsize=5.6, color=GREY,
            style="italic", ha="left", va="bottom")

    # ------------------------------------------------------------- call-out
    bx0, by0, bw, bh = 214.0, 10.0, 296.0, 165.0
    box = FancyBboxPatch((bx0, by0), bw, bh, boxstyle="round,pad=0,rounding_size=7",
                         fc="#ffffff", ec="#3f464c", lw=0.8, zorder=3)
    ax.add_patch(box)
    for dy, ty in ((lh_, by0 + bh), (-lh_, by0)):
        ax.plot([hx + lw_, bx0], [hy + dy, ty],
                color="#3f464c", lw=0.6, ls=(0, (2.2, 1.6)), zorder=2)

    # node positions inside the box
    ns, nm, nr = (266.0, 68.0), (362.0, 124.0), (458.0, 68.0)
    R = 19.0

    # faint stubs standing for the rest of each neuron's connectivity
    rng = np.random.default_rng(4)
    for (cx, cy) in (ns, nm, nr):
        for ang in rng.uniform(0, 2 * np.pi, 5):
            ax.plot([cx + R * np.cos(ang), cx + (R + 17) * np.cos(ang)],
                    [cy + R * np.sin(ang), cy + (R + 17) * np.sin(ang)],
                    color=FAINT, lw=0.8, zorder=4, solid_capstyle="round")

    def arrow(a, b, color, lw, ls="-", shrink=R + 2.5, z=6):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=8,
                                     color=color, lw=lw, linestyle=ls,
                                     shrinkA=shrink, shrinkB=shrink, zorder=z))

    arrow(ns, nm, "#22282d", 1.5)
    arrow(nm, nr, "#22282d", 1.5)
    arrow(ns, nr, GREY, 1.1, ls=(0, (3.2, 2.2)))

    ax.text(303, 107, "3 synapses", fontsize=6.0, color="#22282d",
            ha="center", va="center", rotation=31)
    ax.text(421, 107, "10 synapses", fontsize=6.0, color="#22282d",
            ha="center", va="center", rotation=-31)

    # the absent connection
    xc, yc, h = 362.0, 68.0, 5.4
    ax.plot([xc - h, xc + h], [yc - h, yc + h], color=GREY, lw=1.6, zorder=7)
    ax.plot([xc - h, xc + h], [yc + h, yc - h], color=GREY, lw=1.6, zorder=7)
    ax.text(xc, yc - 15, "no direct synapse", fontsize=6.0, color=GREY,
            ha="center", va="top")

    for (cx, cy), col, name, role in (
        (ns, STIM, "URYDL", "stimulated"),
        (nm, MID, "IL1DL", "not recorded"),
        (nr, RESP, "URADL", "responded"),
    ):
        ax.add_patch(Circle((cx, cy), R, fc=col, ec="none", alpha=0.22, zorder=5))
        ax.add_patch(Circle((cx, cy), R, fc="none", ec=col, lw=1.5, zorder=6))
        ax.text(cx, cy + 1.5, name, fontsize=6.6, color=col, ha="center",
                va="center", weight="bold", zorder=7)
        off = 28 if (cx, cy) == nm else -28
        ax.text(cx, cy + off, role, fontsize=6.0, color=col if off < 0 else GREY,
                ha="center", va="center", zorder=7)

    ax.text(bx0 + 11, by0 + bh - 12, "one two-synapse path, no direct connection",
            fontsize=6.2, color="#22282d", ha="left", va="center")

    OUT.mkdir(parents=True, exist_ok=True)
    # dpi applies to the rasterized neurite layer only; everything else is vector
    fig.savefig(OUT / "fig1a_example_pair.svg", facecolor="white", dpi=600)
    if CHECKDIR is not None:
        fig.savefig(CHECKDIR / "check_fig1A_example_pair.png", dpi=400, facecolor="white")
    print("wrote", OUT / "fig1a_example_pair.svg")


if __name__ == "__main__":
    main()
