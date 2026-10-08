"""Draw the developmental series: each Witvliet 2021 dataset's synapses on a fixed
anatomy, and their quotient onto the biological cell classes.

One column per animal: L1 (Witvliet dataset 1), L2 (5), L3 (6), adult (8), on the
cells reconstructed in all eight datasets (ce.witvliet_common_neurons).

top     the whole animal, dorsal view, with the magnified region boxed. Dorsal
        because the VirtualWorm model is posed in a sinusoid in the lateral
        plane, so from above the body runs straight, the magnified region is
        axis-aligned and left-right pairs are mirrored across the midline.
middle  the head carrying that dataset's synapses, drawn between somata, over
        the OpenWorm NeuroML morphology of the same cells (rasterised). Anatomy
        is one adult model in every column; only the synapses change. The body
        axis is broken at BREAK and everything posterior is compressed by SQUASH
        along the long axis only, so mediolateral positions stay true. Drawn in
        body coordinates and rotated 90 degrees onto the page, nose up.
bottom  the quotient (randi.quotient) onto the biological cell classes, in one
        circular layout shared by all columns: classes in arcs by neuron type,
        anterior to posterior within each arc.

Edge width and opacity use one absolute scale across all four columns in both
graph rows (sqrt of weight over the global maximum). Synapse count grows across
the series, which reflects sampling depth as well as age; the density-controlled
comparison is in fig4c_dev_loss.py.

Geometry comes from conres.worm (VirtualWorm cuticle and pharynx surfaces,
OpenWorm NeuroML somata and morphology), in micrometres in one frame. Neuron
types and the polar layout come from circle_layout.py.

Inputs: Witvliet 2021 datasets via conres.celegans; conres.worm geometry.
Outputs: figures/fig4/fig4a_dev_series.svg.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PathCollection
from matplotlib.path import Path as MplPath
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from conres import celegans as ce, randi, worm, figstyle as fs      # noqa: E402
from circle_layout import TYPE_COL, TYPES, neuron_type, polar       # noqa: E402

OUT = ROOT / "figures/fig4"

STAGES = [(1, "L1"), (5, "L2"), (6, "L3"), (8, "adult")]
NOSE = -294.0            # anterior end of the magnified region, um
BREAK = -222.0           # body axis broken here; posterior compressed by SQUASH
SQUASH, GAP = 0.028, 0.0
HALF = 30.0              # mediolateral half-height of the magnified region
def _band_halfwidth():
    """Body half-width where the axis is broken, read off the silhouette, so the
    compressed length joins the head with no step in the outline."""
    ap, lo, hi = worm.silhouette("cuticle", "dorsal", nbin=60, smooth=13)
    k = int(np.argmin(np.abs(ap - BREAK)))
    return (hi[k] - lo[k]) / 2.0


BAND = _band_halfwidth()
BOW = 0.09               # how far a synapse arc bows off the chord
FS = 6.5
BODY, BODY_EDGE = "#f3f5f6", "#d8dde0"
PHARYNX = "#e7ebed"
NEURITE = (0.42, 0.46, 0.49, 0.22)


def squash(ap):
    """Map AP position (um) to the broken axis: true scale anterior of BREAK,
    compressed by SQUASH posterior of it, preserving order."""
    ap = np.asarray(ap, float)
    return np.where(ap <= BREAK, ap, BREAK + GAP + (ap - BREAK) * SQUASH)


def outline(part, nbin, smooth, upto=None):
    ap, lo, hi = worm.silhouette(part, "dorsal", nbin=nbin, smooth=smooth)
    if upto is not None:
        m = ap <= upto
        ap, lo, hi = ap[m], lo[m], hi[m]
    return np.r_[ap, ap[::-1]], np.r_[hi, lo[::-1]]


def body_with_break():
    """Body outline: the head at true scale continuing into the compressed
    posterior as one outline. The compressed part has constant half-width BAND,
    the body's half-width at the break.
    """
    ap, lo, hi = worm.silhouette("cuticle", "dorsal", nbin=60, smooth=13)
    m = ap <= BREAK
    tail = np.array([BREAK + GAP, squash(448)])
    x = np.r_[ap[m], tail, tail[::-1], ap[m][::-1]]
    y = np.r_[hi[m], BAND, BAND, -BAND, -BAND, lo[m][::-1]]
    return x, y


def quad(p, c, q):
    """One quadratic Bezier path, kept as a true curve rather than a polyline."""
    return MplPath([p, c, q], [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3])


def zigzag(ax, x, half, trans, teeth=13, amp=1.0):
    """A sawtooth across the body: the cut where the long axis is compressed."""
    t = np.linspace(-half, half, teeth * 2 + 1)
    return ax.plot(x + amp * (-1.0) ** np.arange(t.size), t, lw=0.55,
                   color="#b9c0c5", solid_joinstyle="miter", zorder=3,
                   transform=trans)


def ring_ap(segs, lo=-290.0, hi=-240.0, nbin=60):
    """AP position (um) in [lo, hi] crossed by the most neurite segments; used to
    place the nerve-ring label."""
    a = segs[:, :, 0]
    cross = a.min(1), a.max(1)
    grid = np.linspace(lo, hi, nbin)
    n = [((cross[0] <= g) & (cross[1] >= g)).sum() for g in grid]
    return float(grid[int(np.argmax(n))])


def locator(ax, box=True):
    """The whole animal, dorsal, with the magnified region marked."""
    x, y = outline("cuticle", 60, 13)
    ax.fill(x, y, facecolor=BODY, edgecolor=BODY_EDGE, lw=0.6, zorder=1)
    px, py = outline("pharynx", 40, 7)
    ax.fill(px, py, facecolor=PHARYNX, edgecolor="none", zorder=2)
    if box:
        ax.add_patch(Rectangle((NOSE, -HALF), BREAK - NOSE, 2 * HALF, fill=False,
                               ec=fs.INK, lw=0.6, zorder=3))
    ax.plot([348, 448], [40, 40], lw=0.9, color=fs.GREY, solid_capstyle="butt")
    ax.text(398, 46, "100 µm", fontsize=FS - 1.5, color=fs.GREY, ha="center", va="bottom")
    ax.set_xlim(-360, 460); ax.set_ylim(-44, 62)
    ax.set_aspect("equal"); ax.axis("off")


def head(ax, A, xy, ntype, wmax, dmax, segs, ring, label=False):
    """The head, with this dataset's synapses drawn between somata.

    Drawn in body coordinates (AP along x, ML along y) and rotated a quarter
    turn onto the page, nose up.
    """
    rot = Affine2D().rotate_deg(-90) + ax.transData
    x, y = body_with_break()
    ax.fill(x, y, facecolor=BODY, edgecolor=BODY_EDGE, lw=0.6, zorder=0, transform=rot)
    px, py = outline("pharynx", 40, 7)
    ax.fill(px, py, facecolor=PHARYNX, edgecolor="none", zorder=0, transform=rot)

    # the only rasterised artist in the panel; the morphology has thousands of
    # segments
    lc = LineCollection(segs, colors=[NEURITE], linewidths=0.20, zorder=1,
                        transform=rot)
    lc.set_rasterized(True)
    ax.add_collection(lc)

    a, b = np.nonzero(A)
    keep = a != b
    a, b = a[keep], b[keep]
    s = np.sqrt(A[a, b] / wmax)
    paths, cols, widths = [], [], []
    for k in np.argsort(s):
        p, q = xy[a[k]], xy[b[k]]
        d = q - p
        paths.append(quad(p, (p + q) / 2 + BOW * np.array([-d[1], d[0]]), q))
        cols.append((*matplotlib.colors.to_rgb(TYPE_COL[ntype[a[k]]]), min(1.0, 0.07 + 0.62 * s[k])))
        widths.append(0.20 + 0.90 * s[k])
    ax.add_collection(PathCollection(paths, facecolors="none", edgecolors=cols,
                                     linewidths=widths, capstyle="round", zorder=2,
                                     transform=rot))

    deg = (A > 0).sum(0) + (A > 0).sum(1)
    ax.scatter(*xy.T, s=1.0 + 10.0 * deg / dmax, c=[TYPE_COL[t] for t in ntype],
               lw=0.15, edgecolors="white", zorder=3, transform=rot)

    zigzag(ax, BREAK, BAND, rot)         # the cut, drawn through a continuous body
    if label:
        ax.annotate("nerve ring", xy=(-13, -ring), xytext=(-HALF - 6, -ring),
                    fontsize=FS - 1.0, color=fs.INK, style="italic", ha="right",
                    va="center", annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", lw=0.4, color=fs.GREY,
                                    shrinkA=2, shrinkB=2))
        ax.text(-BAND - 3, -BREAK - 5, "670 µm of body,\ncompressed 30×",
                fontsize=FS - 1.5, color=fs.GREY, style="italic",
                ha="right", va="top")
    ax.set_xlim(-HALF, HALF); ax.set_ylim(-squash(448) - 4, -NOSE + 2)
    ax.set_aspect("equal"); ax.axis("off")


def arcs(group_type, group_ap, gap=3):
    """Type arcs, anterior to posterior within each, so a class sits in the same
    place in all four columns."""
    n = len(group_type)
    slots = n + gap * len(TYPES)
    theta, s = np.empty(n), 0
    for t in range(len(TYPES)):
        idx = np.flatnonzero(group_type == t)
        idx = idx[np.argsort(group_ap[idx])]
        theta[idx] = 2 * np.pi * (s + np.arange(idx.size)) / slots
        s += idx.size + gap
    return np.pi / 2 - theta


def quotient_circle(ax, Q, ctype, xy, wmax, tension=0.55):
    a, b = np.nonzero(Q)
    keep = a != b
    a, b = a[keep], b[keep]
    s = np.sqrt(Q[a, b] / wmax)
    paths, cols, widths = [], [], []
    for k in np.argsort(s):
        p, q = xy[a[k]], xy[b[k]]
        paths.append(quad(p, (p + q) / 2 * (1.0 - tension), q))
        cols.append((*matplotlib.colors.to_rgb(TYPE_COL[ctype[a[k]]]), min(1.0, 0.07 + 0.70 * s[k])))
        widths.append(1.25 * (0.35 + 0.65 * s[k]))
    ax.add_collection(PathCollection(paths, facecolors="none", edgecolors=cols,
                                     linewidths=widths, capstyle="round"))
    ax.scatter(*xy.T, s=4.0, c=[TYPE_COL[t] for t in ctype], lw=0, zorder=3)
    ax.set_xlim(-1.14, 1.14); ax.set_ylim(-1.14, 1.14)
    ax.set_aspect("equal"); ax.axis("off")


def type_key(fig, x, y, pad=0.008):
    """The four types, laid out left to right with measured widths."""
    fig.canvas.draw()
    for t, label in enumerate(TYPES):
        txt = fig.text(x, y, label, fontsize=FS - 0.5, color=TYPE_COL[t],
                       ha="left", va="center")
        x += txt.get_window_extent(fig.canvas.get_renderer()).width / fig.bbox.width + pad


def main() -> None:
    fs.use()
    nodes = ce.witvliet_common_neurons("all")
    files = {i: p for i, _, p in ce.witvliet_files()}
    mats = {i: ce.witvliet_matrix(files[i], nodes, "all") for i, _ in STAGES}

    S = worm.soma_xyz(nodes)                            # (ML, AP, DV) um
    xy = np.column_stack([squash(S[:, 1]), S[:, 0]])    # (AP, ML), axis broken
    segs = worm.neurites(nodes)[:, :, [1, 0]]           # (AP, ML) per endpoint
    segs = segs[(segs[:, :, 0] <= BREAK).all(1)]        # head only; see head()
    ntype = neuron_type(nodes)
    ring = ring_ap(segs)

    # biological classes, one fixed circular layout shared by every column
    mapping = ce.assign_classes(nodes)
    cls, lab = np.unique([mapping[ce.normalise_name(n)] for n in nodes], return_inverse=True)
    ctype = np.array([np.bincount(ntype[lab == c], minlength=4).argmax()
                      for c in range(cls.size)])
    cap = np.array([np.nanmean(S[lab == c, 1]) for c in range(cls.size)])
    xy_cls = polar(arcs(ctype, cap))

    quots = {i: randi.quotient(mats[i], lab) for i, _ in STAGES}
    wmax = max(m.max() for m in mats.values())
    qmax = max(q.max() for q in quots.values())
    dmax = max(((m > 0).sum(0) + (m > 0).sum(1)).max() for m in mats.values())
    print(f"{len(nodes)} cells, {cls.size} classes, {len(segs)} morphology segments; "
          f"{(S[:, 1] > BREAK).sum()} cells behind the break; nerve ring at {ring:.0f} um")

    fig = plt.figure(figsize=(7.2, 5.0))
    locator(fig.add_axes([0.26, 0.885, 0.64, 0.10]))
    type_key(fig, 0.105, 0.035)
    gs = fig.add_gridspec(2, 4, height_ratios=[1.60, 1.00],
                          left=0.105, right=0.985, top=0.845, bottom=0.075,
                          wspace=0.04, hspace=0.04)

    for col, (i, name) in enumerate(STAGES):
        A, Q = mats[i], quots[i]
        ax = fig.add_subplot(gs[0, col])
        head(ax, A, xy, ntype, wmax, dmax, segs, ring, label=(col == 0))
        ax.text(0.5, 1.055, name, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=FS + 2.5, color=fs.INK)
        ax.text(0.5, 1.012, f"dataset {i}  ·  {int(A.sum()):,} synapses",
                transform=ax.transAxes, ha="center", va="bottom",
                fontsize=FS - 1.0, color=fs.GREY)

        ax = fig.add_subplot(gs[1, col])
        quotient_circle(ax, Q, ctype, xy_cls, qmax)
        if col == 0:
            ax.text(-0.02, 0.5, "71 classes,\none fixed\nlayout", transform=ax.transAxes,
                    ha="right", va="center", fontsize=FS, color=fs.INK, style="italic")

    fig.text(0.62, 0.035, "sampling depth   1,272 → 6,845 synapses",
             ha="center", va="center", fontsize=FS, color=fs.GREY)
    fs.save(fig, OUT / "fig4a_dev_series.svg", check=fs.check_path("check_fig4a.png"), dpi=600)


if __name__ == "__main__":
    main()
