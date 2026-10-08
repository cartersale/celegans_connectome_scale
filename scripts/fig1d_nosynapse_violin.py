"""Compare walk-operator scores of responding and non-responding pairs with no direct synapse.

Both subpanels use observed ordered pairs (occ3 > 0) with A[pre, post] = 0,
label = response released; walk operator = resolvent at alpha = 0.1.

  fig1d_nosynapse_violin  log10 operator score by group, with IQR and median, the
                     ratio of group medians and a one-sided Mann-Whitney U
                     (responders greater). Values are floored at FLOOR before
                     the log, since pairs with no path return a numerical zero.

  fig1_nosynapse_roc     ROC and AUC, with a 95% percentile interval from a cluster
                     bootstrap over stimulated-neuron class (BOOT = 2000,
                     seed 0). Class is the resampling unit because both
                     members of a class are stimulated in the same animals.

Inputs:  conres.celegans, conres.randi.
Outputs: figures/fig1/fig1d_nosynapse_violin.svg, fig1_nosynapse_roc.svg, plus PNG
         previews in $CONRES_CHECK_DIR.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from conres import celegans as ce, randi  # noqa: E402
from conres import figstyle as fs                       # noqa: E402

OUT = ROOT / "figures/fig1"

plt.rcParams.update({
    "font.family": "Arial",
    "svg.fonttype": "none",
    "mathtext.fontset": "custom",   # keep alpha, rho and superscripts in Arial
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
    "pdf.fonttype": 42,
})

INK, GREY, FAINT = "#22282d", "#8b9196", "#c9ced2"
YES, NO, WALK = "#d1332e", "#8b9196", "#2a5fd6"
ALPHA = 0.1
FLOOR = 1e-19          # the operator's numerical zero, kept on the axis
BOOT = 2000


def sci(x: float) -> str:
    """Format a p-value as mathtext m x 10^e (mantissa rounded to an integer)."""
    e = int(np.floor(np.log10(x)))
    return f"{x / 10 ** e:.0f} \\times 10^{{{e}}}"


def style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.7)
        ax.spines[side].set_color(INK)
    ax.tick_params(labelsize=6.0, colors=INK, length=2.4, width=0.7, pad=1.6)


def save(fig, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.svg", facecolor="white")
    fs.save_check(fig, f"check_{stem}.png")
    plt.close(fig)
    print("wrote", OUT / f"{stem}.svg")


def main() -> None:
    A, nodes = ce.build_matrix("all", neurons_only=True)
    idx = {n: i for i, n in enumerate(nodes)}
    R = randi.resolvent(A, ALPHA)
    occ3, _ = randi.build_occ3(nodes, "wt")
    d = randi.load_responses("wildty")
    d = d[d.pre.isin(idx) & d.post.isin(idx)]

    ii, jj = np.nonzero(occ3)
    hits = {(idx[p], idx[q]) for p, q in zip(d.pre, d.post)}
    y = np.array([1.0 if (b_, a_) in hits else 0.0 for a_, b_ in zip(ii, jj)])
    sub = A[jj, ii] == 0
    v, y, grp = R[jj, ii][sub], y[sub], ce.partitions(nodes)["class"][jj][sub]
    lv = np.log10(np.maximum(v, FLOOR))

    auc = roc_auc_score(y, v)
    mw = stats.mannwhitneyu(v[y == 1], v[y == 0], alternative="greater")
    med = (np.median(v[y == 1]), np.median(v[y == 0]))

    # Cluster bootstrap over stimulated class: both members of a class are
    # stimulated in the same animals, so pairs are not independent within one.
    uq = np.unique(grp)
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(BOOT):
        m = np.concatenate([np.flatnonzero(grp == g)
                            for g in rng.choice(uq, uq.size, replace=True)])
        if len(np.unique(y[m])) > 1:
            boot.append(roc_auc_score(y[m], v[m]))
    lo, hi = np.percentile(boot, [2.5, 97.5])

    # ------------------------------------------------------------------ violin
    fig = plt.figure(figsize=(2.35, 2.05))
    ax = fig.add_axes([0.26, 0.235, 0.70, 0.665])
    groups = ((lv[y == 1], YES, "functionally\nconnected"),
              (lv[y == 0], NO, "not\nconnected"))
    # n goes into the tick label so it cannot collide with the violin tails
    ticklabels = [f"{lbl}\nn = {g.size:,}" for g, _, lbl in groups]

    parts = ax.violinplot([g for g, _, _ in groups], positions=[0, 1],
                          widths=0.78, showextrema=False, showmedians=False)
    for body, (_, col, _) in zip(parts["bodies"], groups):
        body.set_facecolor(col)
        body.set_edgecolor(col)
        body.set_alpha(0.30)
        body.set_linewidth(0.8)

    for k, (g, col, _) in enumerate(groups):
        q1, q2, q3 = np.percentile(g, [25, 50, 75])
        ax.plot([k, k], [q1, q3], color=col, lw=2.2, solid_capstyle="round", zorder=4)
        ax.plot([k - 0.115, k + 0.115], [q2, q2], color=col, lw=1.3, zorder=5)

    ax.set_xlim(-0.62, 1.62)
    ax.set_ylim(-17, -2.4)
    ax.set_xticks([0, 1], ticklabels)
    ax.set_yticks([-15, -12, -9, -6, -3],
                  ["$10^{-15}$", "$10^{-12}$", "$10^{-9}$", "$10^{-6}$", "$10^{-3}$"])
    ax.set_ylabel(f"Walk operator ($\\alpha$ = {ALPHA})", fontsize=6.4, color=WALK,
                  labelpad=1.5)
    style(ax)
    ax.spines["left"].set_color(WALK)
    ax.tick_params(axis="y", colors=WALK)
    # after style(), which would otherwise repaint every tick label INK
    ax.tick_params(axis="x", length=0, pad=2.5)
    for lbl, (_, col, _) in zip(ax.get_xticklabels(), groups):
        lbl.set_color(col)
        lbl.set_fontsize(6.2)
        lbl.set_linespacing(1.3)

    # the comparison, drawn once
    top = -3.05
    ax.plot([0, 0, 1, 1], [top - 0.45, top, top, top - 0.45], color=INK, lw=0.7,
            clip_on=False)
    ax.text(0.5, top + 0.25, f"{med[0] / med[1]:.1f}$\\times$ higher median, "
                             f"$p = {sci(mw.pvalue)}$",
            fontsize=6.0, color=INK, ha="center", va="bottom")
    ax.text(0.02, 0.02, "no-synapse pairs only", transform=ax.transAxes,
            fontsize=5.8, color=GREY, ha="left", va="bottom", style="italic")
    save(fig, "fig1d_nosynapse_violin")

    # --------------------------------------------------------------------- ROC
    fig = plt.figure(figsize=(2.35, 2.05))
    ax = fig.add_axes([0.24, 0.20, 0.72, 0.70])
    ax.plot([0, 1], [0, 1], ls=(0, (3.4, 2.4)), color=GREY, lw=0.9, zorder=2)
    fpr, tpr, _ = roc_curve(y, v)
    k = np.unique(np.linspace(0, len(fpr) - 1, min(600, len(fpr))).astype(int))
    ax.plot(fpr[k], tpr[k], color=WALK, lw=1.5, zorder=4, solid_joinstyle="round")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 0.5, 1.0], ["0", "0.5", "1.0"])
    ax.set_yticks([0, 0.5, 1.0], ["0", "0.5", "1.0"])
    ax.set_aspect("equal")
    ax.set_xlabel("False positive rate", fontsize=6.4, color=INK, labelpad=1.5)
    ax.set_ylabel("True positive rate", fontsize=6.4, color=INK, labelpad=1.5)
    style(ax)

    ax.plot([0.40, 0.52], [0.315, 0.315], color=WALK, lw=1.5,
            transform=ax.transAxes, clip_on=False)
    ax.text(0.555, 0.315, f"Walk operator\n(AUC = {auc:.3f})", transform=ax.transAxes,
            fontsize=6.2, color=WALK, ha="left", va="center", linespacing=1.25)
    ax.plot([0.40, 0.52], [0.145, 0.145], color=GREY, lw=0.9, ls=(0, (3.4, 2.4)),
            transform=ax.transAxes, clip_on=False)
    ax.text(0.555, 0.145, "Chance\n(AUC = 0.500)", transform=ax.transAxes,
            fontsize=6.2, color=GREY, ha="left", va="center", linespacing=1.25)
    ax.text(0.02, 0.98, f"{len(y):,} pairs, no direct synapse\n"
                        f"{int(y.sum()):,} functionally connected\n"
                        f"95% CI [{lo:.3f}, {hi:.3f}], {uq.size} classes",
            transform=ax.transAxes, fontsize=5.9, color=GREY, ha="left", va="top",
            linespacing=1.35)
    save(fig, "fig1_nosynapse_roc")

    print(f"  medians {med[0]:.3g} vs {med[1]:.3g} ({med[0]/med[1]:.1f}x), "
          f"p = {mw.pvalue:.2e}")
    print(f"  AUC {auc:.4f}, cluster bootstrap 95% CI [{lo:.3f}, {hi:.3f}]")


if __name__ == "__main__":
    main()
