"""Draw the VD class on the whole animal: members with neurites, then collapsed to one node.

Two worms, dorsal view, anterior at left, true proportions. Top, the VD motor
neurons at their reconstructed soma positions with their neurites. Bottom, the
single quotient node at the mean soma position of the members. Standalone panel
showing that one class can span most of the ventral cord.

Projection is (anterior-posterior, mediolateral) = (y, x). The NeuroML z coordinate is not consistent as a
dorsoventral axis along the cord (VD somas differ in z by tens of units), so no
lateral view is drawn.

Body outline is the neurite-cloud envelope with both ends tapered, from the
VirtualWorm reconstruction (Grove and Sternberg, WormBase) as converted to
NeuroML2 by OpenWorm.

Inputs: connectome and class names via conres.celegans;
    data/raw/openworm_neuroml/*.cell.nml.
Outputs: figures/fig2/fig2_vd_anatomy.svg (plus a PNG check render in $CONRES_CHECK_DIR).
"""
from __future__ import annotations

import glob
import re
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch, Polygon

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from conres import celegans as ce, figstyle as fs          # noqa: E402

OUT = ROOT / "figures/fig2"
NML = ROOT / "data/raw/openworm_neuroml"

CLS = "VD"
VDC, INK, GREY, GHOST = "#8a5fa8", fs.INK, fs.GREY, "#c9ced2"
SEG = re.compile(r'<segment id="(\d+)"[^>]*>(.*?)</segment>', re.S)
PARENT = re.compile(r'<parent segment="(\d+)"')
PROX = re.compile(r'<proximal x="(-?[\d.eE+-]+)" y="(-?[\d.eE+-]+)" z="(-?[\d.eE+-]+)"')
DIST = re.compile(r'<distal x="(-?[\d.eE+-]+)" y="(-?[\d.eE+-]+)" z="(-?[\d.eE+-]+)"')
PT = re.compile(r'<(?:proximal|distal) x="(-?[\d.eE+-]+)" y="(-?[\d.eE+-]+)" z="(-?[\d.eE+-]+)"')


def reconstruction() -> tuple[np.ndarray, dict[str, tuple[float, float]]]:
    """All neurite points, and each soma, as (anterior-posterior, mediolateral)."""
    pts, soma = [], {}
    for f in glob.glob(str(NML / "*.cell.nml")):
        t = Path(f).read_text()
        pts += [[float(v) for v in m] for m in PT.findall(t)]
        m = DIST.search(t)
        if m:
            v = [float(q) for q in m.groups()]
            soma[Path(f).name.split(".")[0]] = (v[1], v[0])
    return np.asarray(pts), soma


def neurites(name: str) -> list[np.ndarray]:
    """One cell's segments as (anterior-posterior, mediolateral) line segments.

    A segment without its own <proximal> starts where its parent ended, which is
    what makes the arbor connected rather than a cloud of floating sticks.
    """
    text = (NML / f"{name}.cell.nml").read_text()
    ends: dict[int, np.ndarray] = {}
    out = []
    for sid, body in SEG.findall(text):
        d = DIST.search(body)
        if not d:
            continue
        end = np.array([float(d.group(2)), float(d.group(1))])     # (AP, ML)
        p = PROX.search(body)
        if p:
            start = np.array([float(p.group(2)), float(p.group(1))])
        else:
            par = PARENT.search(body)
            start = ends.get(int(par.group(1))) if par else None
            if start is None:
                ends[int(sid)] = end
                continue
        ends[int(sid)] = end
        out.append(np.stack([start, end]))
    return out


def outline(P: np.ndarray, nbin: int = 150, smooth: int = 17):
    """Dorsal silhouette: the neurite envelope per slice, both ends tapered."""
    ap, ml = P[:, 1], P[:, 0]
    edges = np.linspace(ap.min(), ap.max(), nbin + 1)
    s, lo, hi = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (ap >= a) & (ap < b)
        if m.sum() < 3:
            continue
        s.append((a + b) / 2)
        lo.append(np.percentile(ml[m], 1.0))
        hi.append(np.percentile(ml[m], 99.0))
    s, lo, hi = map(np.asarray, (s, lo, hi))
    k = np.ones(smooth) / smooth
    pad = lambda v: np.convolve(np.pad(v, smooth, mode="edge"), k, "same")[smooth:-smooth]
    mid, half = pad((lo + hi) / 2), pad((hi - lo) / 2)
    u = (s - s.min()) / np.ptp(s)
    taper = np.minimum(np.clip(np.sqrt(np.clip(u / 0.055, 0, 1)), 0.03, 1.0),
                       np.clip(np.sqrt(np.clip((1 - u) / 0.045, 0, 1)), 0.05, 1.0))
    return s, mid - half * taper, mid + half * taper


def main() -> None:
    fs.use()
    A, nodes = ce.build_matrix("all", neurons_only=True)
    cname = ce.assign_classes(nodes)
    P, soma = reconstruction()
    s, lo, hi = outline(P)

    members = sorted([n for n in nodes if cname.get(n) == CLS],
                     key=lambda n: soma[n][0])
    xy = np.array([soma[n] for n in members])
    others = np.array([soma[n] for n in nodes if n in soma and cname.get(n) != CLS])
    centre = xy.mean(axis=0)
    span = xy[:, 0].max() - xy[:, 0].min()
    print(f"{CLS}: {len(members)} cells, {members[0]} to {members[-1]}, "
          f"soma span {span:.0f} um of a {s.max() - s.min():.0f} um animal")

    # At true proportions 6 pt type is taller than the animal's half width, so
    # labels sit in figure coordinates between the strips, never inside one.
    fig = plt.figure(figsize=(7.1, 2.65))
    XPAD, YLIM, X0, W = 8.0, 32.0, 0.012, 0.93
    H = W * 7.1 * (2 * YLIM) / (s.max() - s.min() + 2 * XPAD) / 2.65

    def worm(y0):
        ax = fig.add_axes([X0, y0, W, H])
        ax.set_axis_off()
        ax.set_aspect("equal")
        ax.set_xlim(s.min() - XPAD, s.max() + XPAD)
        ax.set_ylim(-YLIM, YLIM)
        body = Polygon(np.concatenate([np.column_stack([s, hi]),
                                       np.column_stack([s[::-1], lo[::-1]])]),
                       closed=True, facecolor="#f4f6f7", edgecolor="#6f767c",
                       lw=0.7, zorder=1)
        ax.add_patch(body)
        ax.scatter(others[:, 0], others[:, 1], s=2.6, c=GHOST, lw=0, zorder=2)
        return ax, body

    def heading(y, label, sub):
        fig.text(X0, y, label, fontsize=7.0, color=INK, ha="left", va="bottom",
                 weight="bold")
        fig.text(X0 + 0.145, y, sub, fontsize=6.0, color=GREY, ha="left", va="bottom")

    top_y, bot_y = 0.655, 0.235
    axt, body_t = worm(top_y)
    segs = [q for n in members for q in neurites(n)]
    lc = LineCollection(segs, colors=VDC, linewidths=0.45, alpha=0.75, zorder=3)
    lc.set_clip_path(body_t)
    axt.add_collection(lc)
    axt.scatter(xy[:, 0], xy[:, 1], s=11, c=VDC, lw=0, zorder=5)

    # a bracket under the animal carries the extent, which no in-body label can
    xa, xb, yb = xy[0, 0], xy[-1, 0], -YLIM + 4.0
    axt.plot([xa, xa, xb, xb], [yb + 3.0, yb, yb, yb + 3.0], color=VDC, lw=0.7,
             solid_capstyle="butt", clip_on=False, zorder=6)

    axb, _ = worm(bot_y)
    axb.scatter([centre[0]], [centre[1]], s=52, c=VDC, lw=0, alpha=0.30, zorder=4)
    axb.scatter([centre[0]], [centre[1]], s=16, c=VDC, lw=0, zorder=5)

    heading(top_y + H + 0.055, f"{len(members)} {CLS} neurons",
            "at their reconstructed soma positions, with their neurites")
    heading(bot_y + H + 0.055, f"1 {CLS} node", "at the mean soma position")

    fig.text(X0 + W * 0.5, top_y - 0.055,
             f"{members[0]} … {members[-1]}, spanning {span:.0f} µm",
             fontsize=6.0, color=VDC, ha="center", va="center")
    fig.text(X0, top_y - 0.055, "anterior at left, dorsal view, true proportions",
             fontsize=5.8, color=GREY, ha="left", va="center", style="italic")

    fig.add_artist(FancyArrowPatch((X0 + W * 0.5, bot_y + H + 0.115),
                                   (X0 + W * 0.5, bot_y + H + 0.045),
                                   transform=fig.transFigure, arrowstyle="-|>",
                                   mutation_scale=8, color=INK, lw=1.2))
    fig.text(X0 + W * 0.5 + 0.012, bot_y + H + 0.080, "collapse", fontsize=6.6,
             color=INK, ha="left", va="center", style="italic")

    x0 = s.max() - 100.0
    axb.plot([x0, x0 + 100.0], [-YLIM + 3.0] * 2, color=INK, lw=0.9,
             solid_capstyle="butt", clip_on=False)
    fig.text(X0 + W * 0.955, bot_y - 0.050, "100 µm", fontsize=5.6, color=INK,
             ha="center", va="center")

    fig.text(X0, 0.065,
             f"{CLS} spans {span:.0f} µm of ventral cord — most of the animal — and "
             f"becomes one node. Class merging is not only joining neighbours.",
             fontsize=6.0, color=INK, ha="left", va="center")

    fs.save(fig, OUT / "fig2_vd_anatomy.svg", check=fs.check_path("check_fig2b_anatomy.png"))


if __name__ == "__main__":
    main()
