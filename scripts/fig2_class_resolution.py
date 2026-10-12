"""Draw the full and class-quotient propagation operators on one circular layout, with the merge-trajectory summary panels.

Top row, three circles on identical geometry:

a  302 neurons on the rim, the anatomical wiring bundled behind, and the
   N_ROUTE strongest predicted routes of the full operator drawn as chords.
   Routes are restricted to pairs with no synapse in either direction.
b  the same rim with each class drawn as one band and one collapsed node; the
   neuron-level wiring is ghosted under the class-quotient wiring. A callout
   magnifies the largest class and its within-class block of A.
c  the same chords, rescored by the class-quotient operator lifted back to
   neuron pairs, drawn over panel a's chords in grey.

Chord emphasis in a and c is percentile rank within that operator's own
no-synapse pairs, not raw magnitude, because the two operators have different
scales and the AUC scoring is rank-based.

Bottom row, from saved sweeps:

d  propagation effect retained along the matched class and random merge
   trajectories, 302 -> 119, as % of the headroom (mean and 2.5-97.5 percentile band)
e  at k = 119, operator-structure loss and propagation-prediction loss per arm
f  at k = 119, loss for propagation and for activity correlation per arm

Operator: (I - alpha*A/rho)^-1 - I with alpha = 0.1 (randi.resolvent). The
class quotient is randi.quotient on the 119 cell classes. The headroom is the
full operator's AUC above 0.5 on observed pairs with no synapse from stimulated
to responder (pair set from class_merge_structural_loss.setup).

Inputs: connectome via conres.celegans; soma positions via conres.worm.soma_ap;
    layout and bundling helpers from circle_layout;
    results/class_merge_structural_loss.csv, results/two_observable_trajectory.csv.
Outputs: figures/fig2/fig2_class_resolution.svg (plus a PNG check render in $CONRES_CHECK_DIR).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from conres import celegans as ce, randi, figstyle as fs            # noqa: E402
from conres.worm import soma_ap                                     # noqa: E402
from circle_layout import (TYPES, TYPE_COL, R_CLASS, R_TYPE,        # noqa: E402
                           neuron_type, layout, polar, mean_angle,
                           bundled, hierarchy_path)
from class_merge_structural_loss import setup as merge_setup        # noqa: E402

OUT = ROOT / "figures/fig2"

ALPHA = 0.1          # same discounting as the fragility sweep
N_ROUTE = 70         # chords drawn
K_BEND = 0.30        # chord control-point radius; 0 is a straight line
N_ARROW = 12         # strongest routes that get an arrowhead
FS = 5.9
DPI = 600            # resolution of the rasterized wiring layers
BIO, RND = fs.CLASS, fs.GREY
GHOST = "#dfe3e6"
WIRE = "#7d858b"     # wiring ink; at 600 dpi the strands resolve individually
BAND_R, NODE_R = 1.05, 1.05
ROT = -2.20          # rotation that brings the called-out class down to the box


# --------------------------------------------------------------- chords

def chord(p, q, k=K_BEND, n=48):
    """Cubic Bezier from p to q pulled toward the centre."""
    ctrl = np.array([p, k * p, k * q, q])
    t = np.linspace(0, 1, n)[:, None]
    return ((1 - t) ** 3 * ctrl[0] + 3 * (1 - t) ** 2 * t * ctrl[1]
            + 3 * (1 - t) * t ** 2 * ctrl[2] + t ** 3 * ctrl[3])


def draw_routes(ax, xy, pairs, rank, src_type, *, grey=False, arrows=0):
    """Chords for `pairs`, emphasis from percentile rank in [0, 1]."""
    lines, cols, widths = [], [], []
    for (a, b), r in zip(pairs, rank):
        lines.append(chord(xy[a], xy[b]))
        base = (0.62, 0.66, 0.69) if grey else matplotlib.colors.to_rgb(TYPE_COL[src_type[a]])
        cols.append((*base, (0.40 + 0.50 * r) if grey else (0.22 + 0.72 * r)))
        widths.append((1.75 if grey else 1.15) * (0.30 + 1.05 * r))
    ax.add_collection(LineCollection(lines, colors=cols, linewidths=widths,
                                     capstyle="round", zorder=2 if grey else 3))
    for k in np.argsort(rank)[::-1][:arrows]:              # heads on the strongest
        pts = lines[k]
        ax.add_patch(FancyArrowPatch(pts[-6], pts[-1], arrowstyle="-|>",
                                     mutation_scale=4.4, lw=0,
                                     color=TYPE_COL[src_type[pairs[k][0]]],
                                     alpha=0.9, zorder=4))


# --------------------------------------------------------------- rim furniture

def rim(ax, theta, gtype, sizes, *, ghost=False, r=1.0):
    xy = polar(theta, r)
    c = GHOST if ghost else [TYPE_COL[t] for t in gtype]
    ax.scatter(*xy.T, s=sizes, c=c, lw=0, zorder=5 if not ghost else 1)


def bands(ax, theta_n, lab, ctype, members):
    """One arc per class, spanning its members, with the collapsed node on it."""
    segs, cols, widths = [], [], []
    for c, m in enumerate(members):
        t = theta_n[m]
        lo, hi = t.min(), t.max()
        pad = 0.45 * (2 * np.pi / (len(theta_n) + 6 * len(TYPES)))
        a = np.linspace(lo - pad, hi + pad, 12)
        segs.append(polar(a, BAND_R))
        cols.append(TYPE_COL[ctype[c]])
        widths.append(3.0)
    ax.add_collection(LineCollection(segs, colors=cols, linewidths=widths,
                                     capstyle="round", alpha=0.8, zorder=4))


def callout(ax, name, Asub, theta, names, col, anchor, box=(-1.34, -0.40, -1.30, -1.035)):
    """Zoom box on one class: its members in rim order and its block of A.

    Members are sorted by rim angle (anterior to posterior), not by the
    alphabetical node order, and the within-class edges are drawn bowed toward
    the circle's interior like the main-panel chords.
    """
    x0, x1, y0, y1 = box
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle="round,pad=0,rounding_size=0.03",
                                fc="white", ec="#c9ced2", lw=0.5, zorder=7))

    ring = 0.055
    for cx in (x0, x1):
        corner = np.array([cx, y1])
        u = anchor - corner
        u = u / np.linalg.norm(u)
        ax.plot(*np.column_stack([corner, anchor - ring * u]), color="#c9ced2",
                lw=0.5, zorder=6, solid_capstyle="butt")
    ax.add_patch(Circle(anchor, ring, fc="none", ec=col, lw=0.9, zorder=8))

    order = np.argsort(-theta)                   # rim order, anterior to posterior
    Asub = Asub[np.ix_(order, order)]
    m = len(order)

    # shallow wedge; edge bow grows with the rim distance between the two cells
    xa, xb = x0 + 0.085, x1 - 0.325
    base, sag = (y0 + y1) / 2 + 0.012, 0.038
    u = np.linspace(0, 1, m)
    xy = np.column_stack([xa + u * (xb - xa), base + sag * (2 * u - 1) ** 2])

    a, b = np.nonzero(Asub > 0)
    segs = []
    for i, j in zip(a, b):
        if i == j:
            continue
        lift = 0.20 * np.sqrt(abs(j - i) / m)
        ctrl = np.array([xy[i], (xy[i] + xy[j]) / 2 + [0, lift], xy[j]])
        t = np.linspace(0, 1, 24)[:, None]
        segs.append((1 - t) ** 2 * ctrl[0] + 2 * (1 - t) * t * ctrl[1] + t ** 2 * ctrl[2])
    ax.add_collection(LineCollection(segs, colors="#9aa1a6", linewidths=0.5,
                                     capstyle="round", zorder=8))
    ax.scatter(*xy.T, s=6.5, c=col, lw=0, zorder=9)

    # label the anterior-most and posterior-most members
    for k in (0, m - 1):
        ax.text(xy[k, 0], xy[k, 1] - 0.038, names[order[k]], fontsize=4.6,
                color=fs.GREY, ha="center", va="top", zorder=9)

    xe = x1 - 0.235
    ymid = (y0 + y1) / 2
    ax.add_patch(FancyArrowPatch((xe, ymid), (xe + 0.105, ymid), arrowstyle="-|>",
                                 mutation_scale=4.5, lw=0.7, color=fs.GREY, zorder=9))
    ax.scatter([xe + 0.185], [ymid], s=30, c=col, lw=0.3, edgecolors="white",
               zorder=9)
    ax.text(x0, y0 - 0.05,
            f"{name}: {m} cells and the {len(segs)} edges among them\n"
            f"become one node, one row of the quotient",
            fontsize=5.4, color=fs.INK, ha="left", va="top", linespacing=1.45,
            zorder=9)


def type_labels(ax, theta, gtype, r=1.14):
    """Arc labels set along the rim rather than radially outward.

    A radial label on a near-horizontal arc runs off the panel: the text grows
    outward faster than the panel can widen, since its size is fixed in points.
    Set tangentially it stays inside the circle's own bounding box at any
    rotation, which matters here because the layout is rotated to bring the
    called-out class down to its box.
    """
    for t, name in enumerate(TYPES):
        idx = np.flatnonzero(gtype == t)
        if not idx.size:
            continue
        ang = mean_angle(theta[idx])
        # run along the tangent, flipping to the opposite tangent whenever that
        # would set the text upside down
        rot = np.degrees(ang) + 90
        if np.cos(np.radians(rot)) < 0:
            rot -= 180
        ax.text(r * np.cos(ang), r * np.sin(ang), name, color=TYPE_COL[t],
                fontsize=5.6, ha="center", va="center", rotation=rot,
                rotation_mode="anchor")


def band(d, col, scale=1.0):
    g = (scale * d.set_index("k")[col]).groupby("k")
    return g.mean(), g.quantile(0.025), g.quantile(0.975)


def panel_d(ax, bm, headroom):
    """Predictive headroom retained along the two matched merge trajectories."""
    for mode, c, lbl in (("random", RND, "Random merges"), ("bio", BIO, "Classes")):
        m, lo, hi = band(bm[bm["mode"] == mode], "loss_auc_nosyn", -100 / headroom)
        m, lo, hi = m + 100, lo + 100, hi + 100
        ax.fill_between(m.index, lo, hi, color=c, alpha=0.18, lw=0)
        ax.plot(m.index, m, color=c, lw=1.2, label=lbl)
    ax.axhline(100, color=fs.FAINT, lw=0.6, ls=(0, (3, 2)), zorder=0)
    ax.axhline(0, color=fs.INK, lw=0.5)
    ax.set_xlim(302, 119)
    ax.set_xticks([300, 250, 200, 150, 119])
    ax.set_ylim(-12, 118)
    ax.set_yticks([0, 50, 100])
    ax.set_xlabel("Resolution (groups, k)", fontsize=FS)
    ax.set_ylabel("Propagation effect retained (%)", fontsize=FS)
    ax.text(288, 104, "Classes", color=BIO, fontsize=FS)
    ax.text(196, 40, "Random", color=RND, fontsize=FS)
    ax.text(297, -8, "full operator's advantage gone", color=fs.GREY, fontsize=5.2)
    fs.style(ax)


def panel_e(ax, bm, headroom):
    """At k = 119: what each arm costs in operator structure, and in prediction."""
    end = bm[bm.k == 119]
    vals, errs = {}, {}
    for mode in ("bio", "random"):
        d = end[end["mode"] == mode]
        vals[mode] = [100 * d.loss_struct_occ.mean(), 100 * d.loss_auc_nosyn.mean() / headroom]
        errs[mode] = [100 * d.loss_struct_occ.std(), 100 * d.loss_auc_nosyn.std() / headroom]
    x = np.arange(2)
    for k, (mode, c) in enumerate((("bio", BIO), ("random", RND))):
        xx = x + (k - 0.5) * 0.36
        ax.bar(xx, vals[mode], 0.33, color=c, lw=0, zorder=2)
        ax.errorbar(xx, vals[mode], yerr=errs[mode], fmt="none", ecolor=fs.INK,
                    elinewidth=0.6, capsize=1.6, capthick=0.6, zorder=3)
        for xi, v in zip(xx, vals[mode]):
            ax.text(xi, v + (4 if v >= 0 else -4), f"{v:.0f}", fontsize=FS, color=c,
                    ha="center", va="bottom" if v >= 0 else "top", weight="bold")
    ax.axhline(0, color=fs.INK, lw=0.5)
    for yy, lbl, c in ((110, "Classes", BIO), (98, "Random merges", RND)):
        ax.text(-0.42, yy, lbl, color=c, fontsize=FS, weight="bold", va="center")
    ax.set_xticks(x)
    ax.set_xticklabels(["Operator\nstructure", "Propagation\nprediction"], fontsize=FS)
    ax.set_ylim(-16, 116)
    ax.set_yticks([0, 50, 100])
    ax.set_ylabel("Lost at k = 119 (%)", fontsize=FS)
    ax.set_xlim(-0.55, 1.55)
    fs.style(ax)
    ax.tick_params(axis="x", length=0)


def panel_f(ax, ob):
    """The same endpoint, scored against a second, independent observable."""
    end = ob[ob.k == 119]
    cols = ["frac_lost_incidence", "frac_lost_correlation"]
    x = np.arange(2)
    stat = {m: (100 * end[end["mode"] == m][cols].mean().values,
                100 * end[end["mode"] == m][cols].std().values)
            for m in ("bio", "random")}
    for xi in x:
        ax.plot([xi, xi], [stat["bio"][0][xi], stat["random"][0][xi]],
                color=fs.FAINT, lw=1.0, zorder=1)
    for mode, c in (("random", RND), ("bio", BIO)):
        m, e = stat[mode]
        ax.errorbar(x, m, yerr=e, fmt="o", ms=4.2, color=c, ecolor=c,
                    elinewidth=0.7, capsize=1.8, capthick=0.7, lw=0, zorder=3)
    ax.axhline(0, color=fs.INK, lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(["Propagation\n(directed, causal)",
                        "Correlation\n(undirected)"], fontsize=FS)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-16, 96)
    ax.set_yticks([0, 25, 50, 75])
    ax.set_ylabel("Effect lost at k = 119 (%)", fontsize=FS)
    ax.text(0.06, stat["random"][0][0] + 9, "Random", color=RND, fontsize=FS)
    ax.text(0.06, stat["bio"][0][0] + 9, "Classes", color=BIO, fontsize=FS)
    fs.style(ax)
    ax.tick_params(axis="x", length=0)


def frame(ax, letter, title, sub):
    ax.set_xlim(-1.42, 1.42)
    ax.set_ylim(-1.62, 1.86)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(-1.36, 1.84, letter, fontsize=8, weight="bold", color=fs.INK,
            ha="left", va="top")
    ax.text(-1.20, 1.84, title, fontsize=7.0, weight="bold", color=fs.INK,
            ha="left", va="top")
    ax.text(-1.20, 1.695, sub, fontsize=FS, color=fs.GREY, ha="left", va="top",
            linespacing=1.45)


# --------------------------------------------------------------- main

def main() -> None:
    fs.use()
    # bundled() rasterizes its splines; at the default 100 dpi they pixelate in the SVG
    plt.rcParams["savefig.dpi"] = DPI
    A, nodes = ce.build_matrix("all", neurons_only=True)
    n = len(nodes)
    lab = np.unique(ce.partitions(nodes)["class"], return_inverse=True)[1]
    ntype = neuron_type(nodes)
    ap = soma_ap(nodes)
    ap = np.where(np.isnan(ap), np.nanmean(ap), ap)

    members = [np.flatnonzero(lab == c) for c in range(lab.max() + 1)]
    ctype = np.array([ntype[m[0]] for m in members])
    cap = np.array([ap[m].mean() for m in members])

    # one geometry for all three panels, rotated so the class the callout opens
    # lands at the lower left, a short leader away from its box
    big = int(np.argmax([len(m) for m in members]))
    theta_n = layout(ntype, cap[lab] + 1e-6 * ap)
    theta_n = theta_n + (ROT - mean_angle(theta_n[members[big]]))
    theta_c = np.array([mean_angle(theta_n[m]) for m in members])
    xy_n = polar(theta_n)
    type_xy = np.array([polar(np.array([mean_angle(theta_n[ntype == t])]), R_TYPE)[0]
                        for t in range(len(TYPES))])
    path_n = hierarchy_path(lab, polar(theta_c, R_CLASS), ntype, type_xy)
    path_q = hierarchy_path(np.arange(len(members)), polar(theta_c, R_CLASS),
                            ctype, type_xy)

    # operators: full, and the 119-class quotient read back at each neuron pair
    R_full = randi.resolvent(A, ALPHA)
    Q = randi.quotient(A, lab)
    R_cls = randi.resolvent(Q, ALPHA)[np.ix_(lab, lab)]

    off = ~np.eye(n, dtype=bool)
    nosyn = off & (A == 0) & (A.T == 0)
    rank = lambda M, mask: (M[mask].argsort().argsort() + 1) / mask.sum()

    r_full, r_cls = np.zeros((n, n)), np.zeros((n, n))
    r_full[nosyn], r_cls[nosyn] = rank(R_full, nosyn), rank(R_cls, nosyn)

    ii, jj = np.nonzero(nosyn)
    top = np.argsort(R_full[ii, jj])[::-1][:N_ROUTE]
    pairs = [(int(ii[k]), int(jj[k])) for k in top]
    e_full = np.array([r_full[a, b] for a, b in pairs])
    e_cls = np.array([r_cls[a, b] for a, b in pairs])

    rho = spearmanr(R_full[nosyn], R_cls[nosyn]).statistic
    kept = float(np.mean(e_cls > np.percentile(r_full[nosyn], 90)))
    n_within = int(((A > 0) & off & (lab[:, None] == lab[None, :])).sum())
    print(f"no-synapse pairs {nosyn.sum()}; spearman(full, class) = {rho:.2f}; "
          f"{100 * kept:.0f}% of drawn routes still in the class operator's top decile; "
          f"edges {int((A > 0).sum())} -> {int((Q > 0).sum())}, {n_within} within-class")

    bm = pd.read_csv(ROOT / "results/class_merge_structural_loss.csv")
    ob = pd.read_csv(ROOT / "results/two_observable_trajectory.csv")
    _, _, _, oj, oi, y, _, _ = merge_setup()
    sub = A[oj, oi] == 0
    headroom = roc_auc_score(y[sub], R_full[oj, oi][sub]) - 0.5
    print(f"full operator AUC above chance on no-synapse pairs: {headroom:.4f}")
    end = bm[bm.k == 119].groupby("mode")["loss_auc_nosyn"].mean() / headroom
    lost_bio, lost_rnd = max(0.0, 100 * end["bio"]), 100 * end["random"]
    st = 100 * bm[bm.k == 119].groupby("mode")["loss_struct_occ"].mean()
    print(f"k=119  effect lost: classes {lost_bio:.0f}%, random {lost_rnd:.0f}%; "
          f"operator structure lost: classes {st['bio']:.0f}%, random {st['random']:.0f}%")

    fig = plt.figure(figsize=(7.1, 5.35))
    gs_top = fig.add_gridspec(1, 3, left=0.005, right=0.995, top=0.995,
                              bottom=0.415, wspace=0.02)
    gs_bot = fig.add_gridspec(1, 3, left=0.075, right=0.985, top=0.370,
                              bottom=0.075, wspace=0.46)
    axa, axb, axc = (fig.add_subplot(gs_top[0, i]) for i in range(3))
    axd, axe, axf = (fig.add_subplot(gs_bot[0, i]) for i in range(3))

    # -- a: full resolution -------------------------------------------------
    bundled(axa, A, xy_n, path_n, ntype, 0.22, 0.105, palette=[WIRE] * 4)
    draw_routes(axa, xy_n, pairs, e_full, ntype, arrows=N_ARROW)
    rim(axa, theta_n, ntype, 2.6)
    type_labels(axa, theta_n, ntype)
    frame(axa, "a", "Full resolution",
          f"{n} neurons, {int((A > 0).sum())} wiring edges (grey)\n"
          f"top {N_ROUTE} predicted routes, none with a synapse")

    # -- b: coarse-grain ----------------------------------------------------
    bundled(axb, A, xy_n, path_n, ntype, 0.20, 0.050, palette=[WIRE] * 4)
    rim(axb, theta_n, ntype, 1.8, ghost=True)
    bundled(axb, Q, polar(theta_c, NODE_R), path_q, ctype, 0.26, 0.24,
            palette=[WIRE] * 4)
    bands(axb, theta_n, lab, ctype, members)
    qxy = polar(theta_c, NODE_R)
    axb.scatter(*qxy.T, s=3.0 * np.sqrt([len(m) for m in members]),
                c=[TYPE_COL[t] for t in ctype], lw=0.15, edgecolors="white", zorder=6)
    type_labels(axb, theta_c, ctype)
    frame(axb, "b", "Coarse-grain",
          f"{len(members)} classes, {int((Q > 0).sum())} edges\n"
          f"ghosted: {n_within} within-class edges, and which\nmember of a class an edge came from")

    # callout on the largest class
    cname = ce.assign_classes(nodes)
    callout(axb, cname[nodes[members[big][0]]],
            A[np.ix_(members[big], members[big])], theta_n[members[big]],
            [nodes[i] for i in members[big]], TYPE_COL[ctype[big]],
            polar(np.array([theta_c[big]]), NODE_R)[0])

    # -- c: the same routes under the class-level operator -------------------
    draw_routes(axc, xy_n, pairs, e_full, ntype, grey=True)
    draw_routes(axc, xy_n, pairs, e_cls, ntype, arrows=N_ARROW)
    rim(axc, theta_n, ntype, 2.6)
    type_labels(axc, theta_n, ntype)
    frame(axc, "c", "Prediction survives",
          f"the same routes, rescored on the classes\n"
          f"ρ = {rho:.2f} over all {nosyn.sum():,} no-synapse pairs\n"
          f"AUC on those pairs: classes lose {lost_bio:.0f}% of the\n"
          f"effect, matched random merges lose {lost_rnd:.0f}%")

    for y, col, lw, txt, tc in ((-1.26, "#9fa5a9", 2.0, "full resolution", fs.GREY),
                                (-1.40, TYPE_COL[0], 1.2, "class level", fs.INK)):
        axc.plot([-1.32, -1.18], [y, y], color=col, lw=lw, solid_capstyle="round")
        axc.text(-1.14, y, txt, fontsize=5.6, color=tc, va="center")

    panel_d(axd, bm, headroom)
    panel_e(axe, bm, headroom)
    panel_f(axf, ob)
    for ax, letter in zip((axd, axe, axf), "def"):
        ax.text(-0.22, 1.14, letter, transform=ax.transAxes, fontsize=8,
                weight="bold", color=fs.INK, va="top")

    fs.save(fig, OUT / "fig2_class_resolution.svg", check=fs.check_path("check_fig2_circular.png"))


if __name__ == "__main__":
    main()
