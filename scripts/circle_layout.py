"""Circular-layout and edge-bundling helpers for the connectome circle plots.

Nodes sit in arcs by neuron type (sensory, interneuron, motor, other; from the
Randi cells.json type field), with GAP empty slots between arcs; `layout`
orders groups within an arc by the anterior-posterior positions it is given.

Edges are hierarchically bundled (Holten 2006): each is a clamped B-spline
through the path neuron -> class -> type -> centre -> type -> class -> neuron,
straightened by BETA. Colour marks the source's type; opacity and width scale
with sqrt(weight / panel maximum).

Used by fig2_class_resolution.py and fig4a_dev_series.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib.collections import LineCollection
from scipy.interpolate import BSpline

ROOT = Path(__file__).resolve().parents[1]

TYPES = ["Sensory", "Interneuron", "Motor", "Other"]
TYPE_COL = ["#d0843a", "#3f7fb8", "#8a5fa8", "#9aa0a6"]
R_CLASS, R_TYPE = 0.78, 0.42
BETA = 0.85
GAP = 6          # empty node slots between arcs
N_PTS = 40


def neuron_type(nodes):
    cells = {c["name"]: c["type"] for c in json.load((ROOT / "data/raw/randi/cells.json").open())}
    first = {"s": 0, "i": 1, "m": 2}
    return np.array([first.get(cells.get(n, "")[:1], 3) for n in nodes])


def layout(group_type, group_ap):
    """Angles for groups placed in type arcs, AP order within each arc."""
    n = len(group_type)
    slots = n + GAP * len(TYPES)
    theta, s = np.empty(n), 0
    for t in range(len(TYPES)):
        idx = np.flatnonzero(group_type == t)
        for i in idx[np.argsort(group_ap[idx])]:
            theta[i] = np.pi / 2 - 2 * np.pi * s / slots
            s += 1
        s += GAP
    return theta


def polar(theta, r=1.0):
    return np.column_stack([r * np.cos(theta), r * np.sin(theta)])


def mean_angle(theta):
    return np.angle(np.exp(1j * theta).mean())


def spline(ctrl):
    """Clamped B-spline through control polygon `ctrl`, straightened by BETA."""
    m = len(ctrl)
    line = ctrl[0] + np.linspace(0, 1, m)[:, None] * (ctrl[-1] - ctrl[0])
    ctrl = BETA * ctrl + (1 - BETA) * line
    k = min(3, m - 1)
    knots = np.r_[np.zeros(k), np.linspace(0, 1, m - k + 1), np.ones(k)]
    return BSpline(knots, ctrl, k)(np.linspace(0, 1, N_PTS))


def bundled(ax, W, node_xy, path_of, src_type, lw, a_max, palette=TYPE_COL):
    """Draw every nonzero W[a, b] as a bundled spline, weakest first."""
    a, b = np.nonzero(W)
    keep = a != b
    a, b = a[keep], b[keep]
    s = np.sqrt(W[a, b] / W[a, b].max())
    lines, cols, widths = [], [], []
    for k in np.argsort(s):
        ctrl = np.vstack([node_xy[a[k]], *path_of(a[k], b[k]), node_xy[b[k]]])
        lines.append(spline(ctrl))
        cols.append((*matplotlib.colors.to_rgb(palette[src_type[a[k]]]), a_max * (0.08 + 0.92 * s[k])))
        widths.append(lw * (0.4 + 0.6 * s[k]))
    lc = LineCollection(lines, colors=cols, linewidths=widths, capstyle="round")
    lc.set_rasterized(True)
    ax.add_collection(lc)


def hierarchy_path(inner, inner_xy, gtype, type_xy):
    """Control points between two nodes: [inner group], type, [centre], type, [inner group]."""
    def path(u, v):
        if inner is not None and inner[u] == inner[v]:
            return [inner_xy[inner[u]]]
        pts = [inner_xy[inner[u]]] if inner is not None else []
        tu, tv = gtype[u], gtype[v]
        pts.append(type_xy[tu])
        if tu != tv:
            pts += [np.zeros(2), type_xy[tv]]
        if inner is not None:
            pts.append(inner_xy[inner[v]])
        return pts
    return path


def rim(ax, theta, gtype, sizes):
    xy = polar(theta, 1.0)
    ax.scatter(*xy.T, s=sizes, c=[TYPE_COL[t] for t in gtype], lw=0, zorder=3)


def labels(ax, theta, gtype):
    for t, name in enumerate(TYPES):
        idx = np.flatnonzero(gtype == t)
        if not idx.size:
            continue
        a = mean_angle(theta[idx])
        c, s = np.cos(a), np.sin(a)
        ha = "left" if c > 0.3 else "right" if c < -0.3 else "center"
        ax.text(1.1 * c, 1.1 * s, name, color=TYPE_COL[t], fontsize=6.5, ha=ha, va="center")


def frame(ax):
    ax.set_xlim(-1.55, 1.55); ax.set_ylim(-1.3, 1.3)
    ax.set_aspect("equal"); ax.axis("off")
