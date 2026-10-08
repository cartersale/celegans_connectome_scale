"""Draw Fig. 3 (panels A-D): connectivity-defined coarse-grainings across scales.

A  alluvial of the nine loss-guided classes (G1-G9 by size) onto the canonical
   subsystems
B  % of full prediction kept against k = 20..2 for loss-guided agglomeration,
   Ward and k-means on operator profiles, k-means on direct connectivity,
   spectral clustering and soma position, with size-matched random partitions as
   a band; signal propagation and activity correlation
C  increase in predictive loss (Delta AUC on unsynapsed pairs, alpha = 0.1) for
   sets of 30 disjoint merges: subsets of the 60 defect-selected merges, random
   disjoint pairs, and random different-class pairs (100 sets per arm, degree
   floor 29, seed 0); one-sided Mann-Whitney tests of selected against each
D  2x2 block: canonical hierarchy (neurons, cell classes, cell types,
   subsystems, divisions) scored against the full operator with the nine classes
   at k = 9; class-pair means kept, by partition and observable; within-class-pair
   performance; whole animal against the nerve ring

The nine classes are coloured by the division of their dominant canonical
subsystem (sensory, interneuron, motor, pharynx), with lightness separating
classes within a division.

Inputs: results/grouping_methods.csv (B, D),
results/coarse_trajectory_paths.npz (key envelope_9; the nine classes),
results/block_decomposition.csv and results/coarse_transfer.csv (D),
Randi 2023 responses via conres.randi, conres.observables.load(). Panel C imports
pairwise_defect, greedy_disjoint and labels_from_pairs from
prospective_merges.py and recomputes the merges.
Outputs: figures/fig3/figure_3.svg, results/fig3c_merge_sets.csv,
results/fig3a_alluvial.csv.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from conres import celegans as ce, figstyle as fs, observables as ob, randi, subsystems as ss  # noqa: E402

OUT = ROOT / "figures/fig3"
RES = ROOT / "results"
INK, GREY, FAINT = fs.INK, fs.GREY, fs.FAINT
LIGHT = "#d9dde0"
CLASS = "#2e9e52"
FS = 6.5

# nine operator-defined classes: colour by the division of the dominant subsystem
GROUP_COL = {1: "#e08214", 8: "#f0b36e", 4: "#9c5a0e", 0: "#4a7fd1", 2: "#9bb9ea",
             5: "#1f4a8f", 6: "#1b7837", 3: "#74b784", 7: "#b8467a"}
SHORT = {"amphid and gas sensory": "amphid sensory",
         "cephalic and labial sensory": "labial sensory",
         "body and tail sensory": "body/tail sensory",
         "layer 1 interneuron": "layer 1 interneurons",
         "layer 2 interneuron": "layer 2 interneurons",
         "layer 3 and category 4 interneuron": "layer 3 interneurons",
         "head and sublateral motor": "head motor",
         "body motor": "ventral cord motor",
         "pharynx": "pharynx", "muscle": "muscle"}


def _host(host, figsize):
    """Panels draw into a subfigure of the assembled figure when given one, and
    into a figure of their own otherwise; the second return value says whether
    this panel owns the file it would save."""
    return (plt.figure(figsize=figsize), True) if host is None else (host, False)


def nine_groups(nodes):
    g = np.unique(np.load(RES / "coarse_trajectory_paths.npz")["envelope_9"], return_inverse=True)[1]
    sub, names = ss.partition(nodes, "subsystems")
    info = {}
    for c in range(9):
        m = g == c
        cnt = np.bincount(sub[m], minlength=len(names))
        o = np.argsort(-cnt)
        p1 = cnt[o[0]] / m.sum()
        # naming rule, fixed before drawing: dominant subsystem if >= 75% of the
        # class, otherwise the top two
        if p1 >= 0.75:
            label = SHORT[names[o[0]]]
        else:
            a, b = SHORT[names[o[0]]], SHORT[names[o[1]]]
            label = f"{a} + {b}"
        info[c] = dict(label=label, n=int(m.sum()), purity=p1, counts=cnt)
    return g, sub, names, info


# --------------------------------------------------------------------- A
def panel_merge_criterion(A, nodes, host=None):
    """Score sets of 30 disjoint merges among neurons with degree >= 29, as the
    increase in predictive loss (AUC on unsynapsed pairs) relative to no merge.
    Single merges are too small to separate, so merges are compared in sets:

      selected         30 of the 60 merges chosen by the merge criterion, resampled
      random           30 random disjoint pairs
      different-class  30 random disjoint pairs whose members are in different
                       cell classes

    The selected dots are subsets of one chosen set of 60, so their spread shows
    which merges are sampled; the other two arms are independent draws.
    Writes results/fig3c_merge_sets.csv."""
    from scipy import stats
    from sklearn.metrics import roc_auc_score
    from prospective_merges import pairwise_defect, greedy_disjoint, labels_from_pairs
    n = A.shape[0]
    mapping = ce.assign_classes(nodes)
    idx = {x: i for i, x in enumerate(nodes)}
    occ3, _ = randi.build_occ3(nodes, "wt")
    d = randi.load_responses("wildty")
    d = d[d.pre.isin(idx) & d.post.isin(idx)]
    hits = {(idx[p], idx[q]) for p, q in zip(d.pre, d.post)}
    ii, jj = np.nonzero(occ3)
    y = np.array([1.0 if (j, i) in hits else 0.0 for i, j in zip(ii, jj)])
    ns = A[jj, ii] == 0
    R = randi.resolvent(A, 0.1)
    a0 = roc_auc_score(y[ns], R[jj, ii][ns])
    deg = (A > 0).sum(0) + (A > 0).sum(1)
    allowed = deg >= 29
    pool = np.flatnonzero(allowed)
    sel = greedy_disjoint((pairwise_defect(A) + pairwise_defect(A.T)) / 2.0, 60, allowed)
    rng = np.random.default_rng(0)
    M, N_SET = 30, 100

    def disjoint(cond):
        while True:
            p = rng.permutation(pool)
            out, used = [], set()
            for a in p:
                if a in used:
                    continue
                for b in p:
                    if b != a and b not in used and cond(a, b):
                        out.append((a, b)); used |= {a, b}
                        break
                if len(out) == M:
                    return out

    def loss(pairs):
        q = randi.class_leverage(A, labels_from_pairs(n, pairs), 0.1)[jj, ii]
        return a0 - roc_auc_score(y[ns], q[ns])

    arms = {
        "selected\nmerges": [loss([sel[i] for i in rng.choice(60, M, replace=False)])
                              for _ in range(N_SET)],
        "random\nmerges": [loss(disjoint(lambda a, b: True)) for _ in range(N_SET)],
        "different-\nclass merges": [loss(disjoint(lambda a, b: mapping[nodes[a]] != mapping[nodes[b]]))
                                     for _ in range(N_SET)],
    }
    keys = list(arms)
    pd.DataFrame({k.replace("\n", " "): v for k, v in arms.items()}).to_csv(
        RES / "fig3c_merge_sets.csv", index=False)
    pv = [stats.mannwhitneyu(arms[keys[0]], arms[k], alternative="less").pvalue for k in keys[1:]]
    for k in keys:
        print(f"A: {k.replace(chr(10), ' '):>24} {np.mean(arms[k]):+.4f} +- {np.std(arms[k]):.4f}")
    print(f"A: selected < random p {pv[0]:.1e}; selected < different-class p {pv[1]:.1e}")

    cols = ["#2a5fd6", "#8b9196", "#d1332e"]
    fig, own = _host(host, (2.3, 2.2))
    ax = fig.add_axes([0.27, 0.20, 0.70, 0.64])
    jr = np.random.default_rng(1)
    for x, (k, c) in enumerate(zip(keys, cols)):
        v = np.array(arms[k])
        ax.plot(x + jr.uniform(-0.18, 0.18, v.size), v, "o", ms=2.6, color=c, alpha=0.45,
                mew=0, zorder=2)
        m, sd = v.mean(), v.std()
        ax.plot([x, x], [m - sd, m + sd], color=INK, lw=2.2, solid_capstyle="butt", zorder=3)
        ax.plot(x, m, "o", ms=4.2, color=INK, zorder=4)
    ax.axhline(0, color=GREY, lw=0.6, zorder=0)

    def ptxt(p):
        e = int(np.floor(np.log10(p)))
        return f"$p < 10^{{{e + 1}}}$"
    top = max(max(v) for v in arms.values())
    for (x1, x2), p_, hgt in (((0, 1), pv[0], top + 0.004), ((0, 2), pv[1], top + 0.011)):
        ax.plot([x1, x2], [hgt, hgt], color=INK, lw=0.6)
        ax.text((x1 + x2) / 2, hgt + 0.0008, ptxt(p_), ha="center", va="bottom", fontsize=FS - 1)
    ax.set_xticks(range(3), keys, fontsize=FS - 1)
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylabel("increase in predictive loss\n(ΔAUC relative to no merge)", fontsize=FS,
                  linespacing=1.3)
    fs.style(ax)
    ax.tick_params(axis="x", length=0)
    if own:
        fs.save(fig, OUT / "fig3c_merge_criterion.svg", check=fs.check_path("check_fig3A.png"))
        plt.close(fig)


# --------------------------------------------------------------------- B
# categorical palette slots 1-6 in fixed order; line style and marker repeat
# the identity
METHODS = [("loss-greedy", "loss-guided agglomeration", "#2a78d6", "-", None),
           ("Ward on operator", "Ward, operator profiles", "#eb6834", (0, (3, 1.6)), None),
           ("k-means on operator", "k-means, operator profiles", "#1baf7a", (0, (1, 1.2)), None),
           ("k-means on wiring", "direct connectivity", "#eda100", "-", "s"),
           ("spectral", "spectral (adjacency)", "#e87ba4", "-", "^"),
           ("spatial", "soma position", "#008300", "-", "o")]


def panel_resolution(host=None):
    d = pd.read_csv(RES / "grouping_methods.csv")
    fig, own = _host(host, (4.6, 2.4))
    axes = [fig.add_axes([0.095, 0.33, 0.39, 0.60]), fig.add_axes([0.595, 0.33, 0.39, 0.60])]
    for ax, o, title in ((axes[0], "incidence_all", "signal propagation"),
                         (axes[1], "correlation", "activity correlation")):
        ax.axvline(9, color=INK, lw=0.7, ls=(0, (2.5, 2)), zorder=0.5)
        rnd = d[(d.method == "loss-greedy") & (d.observable == o)].sort_values("k")
        ax.fill_between(rnd.k, 0, 100 * (1 - rnd.random), color=LIGHT, lw=0, zorder=0)
        for key, name, col, ls, mk in METHODS:
            s = d[(d.method == key) & (d.observable == o)].sort_values("k")
            y = 100 * (1 - s.loss)
            if key == "loss-greedy":
                ax.fill_between(s.k, 100 * (1 - s.hi), 100 * (1 - s.lo), color=col,
                                alpha=0.12, lw=0, zorder=1)
            ax.plot(s.k, y, color=col, lw=1.1, ls=ls, zorder=3)
            if mk:
                ax.plot(s.k, y, mk, ms=2.6, mfc="white", mec=col, mew=0.8, ls="none", zorder=4)
            if ax is axes[0]:
                ax.plot([], [], color=col, lw=1.1, ls=ls, marker=mk, ms=2.6, mfc="white",
                        mec=col, label=name)
        ax.axhline(100, color=INK, lw=0.5, zorder=0)
        ax.set_xlim(20.3, 1.7)
        ax.set_ylim(-5, 125)
        ax.set_yticks([0, 50, 100])
        ax.set_ylabel(f"{title}\n(% of full kept)", fontsize=FS, linespacing=1.3)
        fs.style(ax)
    for ax in axes:
        ax.text(8.8, 119, "k = 9", fontsize=FS - 1.5, color=INK, ha="left", va="center")
        ax.set_xticks([20, 15, 12, 10, 9, 8, 6, 4, 2])
        ax.set_xlabel("number of classes, k", fontsize=FS)
    axes[0].plot([], [], color=LIGHT, lw=5, label="size-matched random")
    axes[0].legend(frameon=False, fontsize=FS - 1.5, loc="upper left", ncol=4,
                   bbox_to_anchor=(-0.02, -0.30), handlelength=2.2, labelspacing=0.45,
                   columnspacing=1.2)
    if own:
        fs.save(fig, OUT / "fig3b_prediction_vs_resolution.svg", check=fs.check_path("check_fig3B.png"))
        plt.close(fig)


# --------------------------------------------------------------------- C
# short key entries under each division heading: dominant subsystem first, the
# second subsystem in parentheses when the class is mixed (same rule as info)
BODY, BODY_EDGE, PHARYNX = "#f3f5f6", "#d8dde0", "#e7ebed"


# canonical subsystems share the division hues of the nine classes, so both
# panels can be read against one another
SUBSYS_COL = {"amphid and gas sensory": "#e08214", "cephalic and labial sensory": "#f0b36e",
              "body and tail sensory": "#9c5a0e", "layer 1 interneuron": "#1f4a8f",
              "layer 2 interneuron": "#4a7fd1", "layer 3 and category 4 interneuron": "#9bb9ea",
              "head and sublateral motor": "#74b784", "body motor": "#1b7837",
              "pharynx": "#b8467a", "muscle": "#b9c0c5"}


def panel_alluvial(nodes, g, sub, names, info, host=None):
    """Alluvial from the nine connectivity-defined classes to the canonical
    subsystems; ribbon width is number of neurons.

    The left-hand classes are named G1-G9 in order of size rather than with
    anatomical names, since they are defined by connectivity alone. Writes the
    class-by-subsystem counts to results/fig3a_alluvial.csv."""
    from matplotlib.path import Path as MplPath
    from matplotlib.patches import PathPatch
    k, m = 9, len(names)
    C = np.zeros((k, m))
    np.add.at(C, (g, sub), 1)
    left = list(np.argsort(-C.sum(1)))               # G1 = largest class
    gname = {c: f"G{i + 1}" for i, c in enumerate(left)}
    # colour follows each class's dominant subsystem, assigned after the classes
    # were built
    gcol = {c: GROUP_COL[c] for c in left}
    right = sorted(range(m), key=lambda j: np.average([left.index(c) for c in range(k)],
                                                      weights=C[:, j] + 1e-9))
    pd.DataFrame({"class": [gname[c] for c in left], "n": [int(C[c].sum()) for c in left],
                  **{SHORT[names[j]]: [int(C[c, j]) for c in left] for j in right}}
                 ).to_csv(RES / "fig3a_alluvial.csv", index=False)
    GAP, H = 0.012, 1.0

    def layout(order, totals):
        span = H - GAP * (len(order) - 1)
        y, pos = 1.0, {}
        for i in order:
            h = span * totals[i] / totals.sum()
            pos[i] = (y - h, y)
            y -= h + GAP
        return pos
    lp = layout(left, C.sum(1))
    rp = layout(right, C.sum(0))
    fig, own = _host(host, (4.4, 3.5))
    ax = fig.add_axes([0.20, 0.035, 0.50, 0.925])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.0); ax.axis("off")
    cursor_l = {c: lp[c][1] for c in left}
    cursor_r = {j: rp[j][1] for j in right}
    for c in left:
        for j in right:
            v = C[c, j]
            if not v:
                continue
            hl = (lp[c][1] - lp[c][0]) * v / C[c].sum()
            hr = (rp[j][1] - rp[j][0]) * v / C[:, j].sum()
            y0, y1 = cursor_l[c], cursor_l[c] - hl
            z0, z1 = cursor_r[j], cursor_r[j] - hr
            cursor_l[c] -= hl
            cursor_r[j] -= hr
            verts = [(0.08, y0), (0.5, y0), (0.5, z0), (0.92, z0),
                     (0.92, z1), (0.5, z1), (0.5, y1), (0.08, y1), (0.08, y0)]
            codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
                     MplPath.LINETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
                     MplPath.CLOSEPOLY]
            ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=gcol[c], lw=0,
                                   alpha=0.4, zorder=1))
    for c in left:
        ax.add_patch(plt.Rectangle((0.04, lp[c][0]), 0.04, lp[c][1] - lp[c][0],
                                   color=gcol[c], lw=0, zorder=3))
        ax.text(0.02, (lp[c][0] + lp[c][1]) / 2, f"{gname[c]}\n({info[c]['n']})",
                fontsize=FS - 1.5, ha="right", va="center", color=INK, linespacing=1.1)
    # small bars would collide, so labels are spread to a minimum spacing and
    # joined to their bar with a leader
    MIN_SEP = 0.072
    ys, prev = {}, None
    for j in reversed(right):                     # bottom to top
        y = (rp[j][0] + rp[j][1]) / 2
        if prev is not None and y < prev + MIN_SEP:
            y = prev + MIN_SEP
        ys[j], prev = y, y
    for j in right:
        ax.add_patch(plt.Rectangle((0.92, rp[j][0]), 0.04, rp[j][1] - rp[j][0],
                                   color=SUBSYS_COL[names[j]], lw=0, zorder=3))
        yc = (rp[j][0] + rp[j][1]) / 2
        if abs(ys[j] - yc) > 1e-6:
            ax.plot([0.965, 0.995], [yc, ys[j]], color=FAINT, lw=0.5, zorder=2, clip_on=False)
        ax.text(1.0, ys[j], f"{SHORT[names[j]]} ({int(C[:, j].sum())})",
                fontsize=FS - 1.5, ha="left", va="center", color=INK)
    if own:
        fs.save(fig, OUT / "fig3a_alluvial.svg", check=fs.check_path("check_fig3C_alluvial.png"))
        plt.close(fig)


# --------------------------------------------------------------------- D
def panel_hierarchy(D, nodes, host=None):
    """2x2 block of resolution and scale panels.

    ax_h  (top left) canonical hierarchy (neurons, cell classes, cell types, subsystems,
          divisions) scored against the full path operator, with 95% intervals
          from 300 bootstrap resamples and the nine loss-guided classes at k = 9
    ax_b  (bottom left) class-pair means: % of the full operator's prediction kept when every
          pair is replaced by its class-pair mean, by partition and observable
    ax_w  (bottom right) within-class-pair performance for the same two observables
    ax_s  (top right) the nine classes on the whole animal against the nerve ring

    Loss is (score_full - score_class) / score_full."""
    d20 = pd.read_csv(RES / "grouping_methods.csv")
    A, obs, cls = D["A"], D["obs"], D["cls"]
    R = randi.resolvent(A, 0.1)
    boots = ob.boot_indices(obs, cls, 300)

    def scored(lab):
        """loss and interval for one partition, per observable"""
        Q = randi.class_leverage(A, lab, 0.1)
        out = {}
        for o in ("incidence_all", "correlation"):
            j, i, v, kd = obs[o]
            pf, pq = ob.predict(kd, R, j, i), ob.predict(kd, Q, j, i)
            full = ob.score(kd, pf, v)
            bs = [(ob.score(kd, pf, v, t) - ob.score(kd, pq, v, t)) / ob.score(kd, pf, v, t)
                  for t in boots[o]]
            out[o] = ((full - ob.score(kd, pq, v)) / full, *np.nanpercentile(bs, [2.5, 97.5]))
        return out

    types_lab = ss.partition(nodes, "types")[0]
    subs_lab = ss.partition(nodes, "subsystems")[0]
    divs_lab = ss.partition(nodes, "divisions")[0]
    levels = [("neurons", 302, None), ("cell classes", int(cls.max() + 1), cls),
              ("cell types", int(types_lab.max() + 1), types_lab),
              ("subsystems", int(subs_lab.max() + 1), subs_lab),
              ("divisions", int(divs_lab.max() + 1), divs_lab)]
    vals = {}
    for name, k, lab in levels:
        vals[name] = ({"incidence_all": (0.0, 0.0, 0.0), "correlation": (0.0, 0.0, 0.0)}
                      if lab is None else scored(lab))
        print(f"D: {name} (k={k}): " + "  ".join(
            f"{o} keeps {100 * (1 - vals[name][o][0]):.0f}%" for o in vals[name]))

    d = pd.read_csv(RES / "block_decomposition.csv")
    t = pd.read_csv(RES / "coarse_transfer.csv")
    parts = [("loss-greedy k=9", "loss-guided agglomeration", INK),
             ("subsystems (a priori)", "subsystems", CLASS),
             ("spatial k-means k=9", "soma position", GREY),
             ("random (size-matched to loss-greedy k=9)", "size-matched random", LIGHT)]
    bobs = [("incidence_all", "signal\npropagation"), ("incidence_nosyn", "unsynapsed\npairs"),
            ("correlation", "activity\ncorrelation"), ("amplitude", "response\namplitude")]

    # sized for its slot in the assembled figure so type is not rescaled
    fig, own = _host(host, (3.9, 3.3))
    ax_h = fig.add_axes([0.155, 0.605, 0.40, 0.355])
    ax_s = fig.add_axes([0.725, 0.605, 0.245, 0.355])
    ax_b = fig.add_axes([0.155, 0.265, 0.40, 0.235])
    ax_w = fig.add_axes([0.725, 0.205, 0.245, 0.295])

    # ---- ax_h (top left): the canonical hierarchy against resolution
    ks = [k for _, k, _ in levels]
    for o, mk, ls, name in (("incidence_all", "o", "-", "signal propagation"),
                            ("correlation", "s", (0, (2.4, 1.5)), "activity correlation")):
        y = np.array([100 * (1 - vals[n][o][0]) for n, _, _ in levels])
        loq = np.array([100 * (1 - vals[n][o][2]) for n, _, _ in levels])
        hiq = np.array([100 * (1 - vals[n][o][1]) for n, _, _ in levels])
        ax_h.plot(ks, y, color=CLASS, lw=1.0, ls=ls, zorder=2)
        ax_h.vlines(ks, loq, hiq, color=CLASS, lw=0.8, zorder=2)
        ax_h.plot(ks, y, mk, ms=3.6, mfc=CLASS if mk == "o" else "white", mec=CLASS, mew=0.9,
                  ls="none", zorder=3)
        ax_h.plot([], [], mk, ms=3.6, mfc=GREY if mk == "o" else "white", mec=GREY, mew=0.9,
                  ls=ls, color=GREY, label=name)
    r9 = {o: d20[(d20.method == "loss-greedy") & (d20.k == 9) & (d20.observable == o)].iloc[0]
          for o in ("incidence_all", "correlation")}
    for o, mk in (("incidence_all", "o"), ("correlation", "s")):
        ax_h.errorbar(9, 100 * (1 - r9[o].loss),
                      yerr=[[100 * (r9[o].hi - r9[o].loss)], [100 * (r9[o].loss - r9[o].lo)]],
                      fmt=mk, ms=3.4, mfc=INK if mk == "o" else "white", mec=INK,
                      ecolor=INK, elinewidth=0.7, capsize=0, zorder=4)
    OFF = {"neurons": (7, 2, "left", "center"), "cell classes": (0, -13, "center", "top"),
           "cell types": (0, -12, "center", "top"), "subsystems": (6, 7, "left", "bottom"),
           "divisions": (6, -6, "left", "top")}
    for n, k, _ in levels:
        dx, dy, ha, va = OFF[n]
        ax_h.annotate(f"{n} ({k})", xy=(k, 100 * (1 - vals[n]["incidence_all"][0])),
                      xytext=(dx, dy), textcoords="offset points", fontsize=FS - 1.8,
                      color=GREY, ha=ha, va=va)
    ax_h.set_xscale("log")
    ax_h.set_xlim(3, 900)
    ax_h.set_ylim(0, 132)
    ax_h.set_yticks([0, 50, 100])
    ax_h.axhline(100, color=INK, lw=0.5, zorder=0)
    ax_h.set_xlabel("number of classes, k", fontsize=FS)
    ax_h.set_ylabel("% of full kept", fontsize=FS)
    ax_h.legend(frameon=False, fontsize=FS - 1.8, loc="lower right", handlelength=1.8,
                labelspacing=0.35)
    fs.style(ax_h)

    # ---- ax_b (bottom left): class-pair means
    w = 0.18
    for pi, (key, name, col) in enumerate(parts):
        for oi, (o, _) in enumerate(bobs):
            r = d[(d.partition == key) & (d.observable == o)].iloc[0]
            x = oi + (pi - 1.5) * w
            ax_b.bar(x, 100 * r.block_frac, width=w * 0.9, color=col, lw=0,
                     label=name if oi == 0 else None)
            ax_b.vlines(x, 100 * r.block_frac_lo, 100 * r.block_frac_hi, color=INK, lw=0.6)
    ax_b.axhline(100, color=INK, lw=0.5)
    # four two-word names under a 1.6 in axis: upright they collide, so tilt them
    ax_b.set_xticks(range(len(bobs)), [o[1].replace("\n", " ") for o in bobs],
                    fontsize=FS - 1.5, rotation=22, ha="right", rotation_mode="anchor")
    ax_b.set_ylabel("% of full kept", fontsize=FS)
    ax_b.set_ylim(-20, 175)
    ax_b.set_yticks([0, 50, 100, 150])
    ax_b.legend(frameon=False, fontsize=FS - 2.0, ncol=2, handlelength=0.9,
                columnspacing=0.8, labelspacing=0.25, handletextpad=0.4,
                loc="upper left", bbox_to_anchor=(-0.02, -0.62))
    fs.style(ax_b)

    # ---- ax_w (bottom right): what is left inside a class pair
    for pi, (key, name, col) in enumerate(parts[:2]):
        for oi, o in enumerate(("incidence_all", "correlation")):
            r = d[(d.partition == key) & (d.observable == o)].iloc[0]
            x = oi + (pi - 0.5) * 0.36
            ax_w.bar(x, r.within_block, width=0.32, color=col, lw=0)
            ax_w.vlines(x, r.within_block_lo, r.within_block_hi, color=INK, lw=0.6)
    ax_w.axhline(0, color=INK, lw=0.5)
    ax_w.set_xticks([0, 1], ["signal\npropagation", "activity\ncorrelation"], fontsize=FS - 2.5)
    ax_w.set_xlim(-0.55, 1.55)
    ax_w.set_ylim(0, 0.19)
    ax_w.set_ylabel("within class pairs\n(AUC − 0.5, or ρ)", fontsize=FS, linespacing=1.3)
    fs.style(ax_w)

    # ---- ax_s (top right): whole animal against the nerve ring
    whole = {o: d[(d.partition == "loss-greedy k=9") & (d.observable == o)].iloc[0].block_frac
             for o in ("incidence_all", "correlation")}
    head = t[(t.method == "loss-greedy") & (t.k == 9) & (t.arm == "Cook, same neurons")].iloc[0]
    for o, mk, ls in (("incidence_all", "o", "-"), ("correlation", "s", (0, (2.4, 1.5)))):
        ax_s.plot([0, 1], [100 * whole[o], 100 * head[o]], color=INK, lw=1.0, ls=ls)
        ax_s.plot([0, 1], [100 * whole[o], 100 * head[o]], mk, ms=3.6,
                  mfc=INK if mk == "o" else "white", mec=INK, mew=0.9, ls="none")
    ax_s.axhline(100, color=INK, lw=0.5)
    ax_s.set_xticks([0, 1], ["whole\nanimal (302)", "nerve\nring (161)"], fontsize=FS - 2.0)
    ax_s.set_xlim(-0.35, 1.35)
    ax_s.set_ylim(0, 130)
    ax_s.set_yticks([0, 50, 100])
    ax_s.set_ylabel("% of full kept", fontsize=FS)
    fs.style(ax_s)

    if own:
        fs.save(fig, OUT / "fig3d_scales.svg", check=fs.check_path("check_fig3DE.png"))
        plt.close(fig)


def assemble(D, nodes, g, sub, names, info):
    """Assemble panels A-D into figures/fig3/figure_3.svg."""
    fig = plt.figure(figsize=(7.1, 6.2))
    sfs = fig.subfigures(2, 2, width_ratios=[0.335, 0.665], height_ratios=[0.40, 0.60],
                         wspace=0.0, hspace=0.0)
    for sf in sfs.ravel():
        sf.set_facecolor("none")
    panel_merge_criterion(D["A"], nodes, host=sfs[0, 0])
    panel_resolution(host=sfs[0, 1])
    panel_alluvial(nodes, g, sub, names, info, host=sfs[1, 0])
    panel_hierarchy(D, nodes, host=sfs[1, 1])
    for sf, lab in zip(sfs.ravel(), "CBAD"):
        sf.text(0.012, 0.975, lab, fontsize=FS + 2.5, color=INK, weight="bold",
                ha="left", va="top")
    fs.save(fig, OUT / "figure_3.svg", check=fs.check_path("check_figure_3.png"))
    plt.close(fig)


def main():
    fs.use()
    D = ob.load()
    nodes = D["nodes"]
    g, sub, names, info = nine_groups(nodes)
    for c in sorted(info):
        print(f"  class {c}: {info[c]['label']} (n={info[c]['n']}, purity {info[c]['purity']:.0%})")
    assemble(D, nodes, g, sub, names, info)


if __name__ == "__main__":
    main()
