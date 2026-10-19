"""Draw direct synaptic weight and the path operator against response magnitude and occurrence.

Two SVGs, each carrying both predictors:

  fig1_weight_vs_response  Spearman rho of each predictor against response
                            magnitude, over released responders with a direct
                            synapse (A > 0). Log-log scatter with a log-log
                            least-squares line; two x axes, since synapse
                            counts and operator values share no units.

  fig1b_roc            ROC and AUC over every observed ordered pair
                            (occ3 > 0), label = response released.

Response magnitudes are |value| from the released funconn atlas, in atlas units;
the release does not state a dF/F calibration.

The path operator is the resolvent at alpha = 0.1.

Inputs:  conres.celegans, conres.randi.
Outputs: figures/fig1/fig1_weight_vs_response.svg, fig1b_roc.svg, plus PNG
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
PATH = "#2a5fd6"          # the path operator, in every panel
ALPHA = 0.1


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
    print("wrote", OUT / f"{stem}.svg")


def main() -> None:
    A, nodes = ce.build_matrix("all", neurons_only=True)
    idx = {n: i for i, n in enumerate(nodes)}
    occ3, _ = randi.build_occ3(nodes, "wt")
    d = randi.load_responses("wildty")
    d = d[d.pre.isin(idx) & d.post.isin(idx)]

    # ------------------------------------------------- scatter, connected pairs
    R = randi.resolvent(A, ALPHA)
    j = d.pre.map(idx).to_numpy()
    i = d.post.map(idx).to_numpy()
    mag = d["mag"].to_numpy(float)
    w, r = A[j, i], R[j, i]
    con = w > 0
    y = mag[con]
    rho_w = stats.spearmanr(w[con], y)
    rho_r = stats.spearmanr(r[con], y)

    fig = plt.figure(figsize=(2.35, 2.05))
    ax = fig.add_axes([0.24, 0.20, 0.72, 0.60])      # direct weight, bottom axis
    tw = ax.twiny()                                   # path operator, top axis

    def cloud(a, xv, color, marker, dash, fill, z):
        """Scatter plus log-log fit line; returns the slope. Dash phases are
        offset so that coincident fit lines stay separately visible."""
        a.plot(xv, y, marker, ms=2.3, mfc=fill, mec=color, mew=0.5, alpha=0.55,
               ls="none", zorder=z)
        b, c = np.polyfit(np.log10(xv), np.log10(y), 1)
        lo, hi = np.log10(a.get_xlim())
        xs = np.logspace(lo, hi, 2)
        a.plot(xs, 10 ** (c + b * np.log10(xs)), ls=dash, color=color, lw=1.2,
               zorder=6, dash_capstyle="butt")
        return b

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.7, 130)
    ax.set_ylim(0.7, 300)
    tw.set_xscale("log")
    tw.set_xlim(3e-4, 9e-2)

    slope_r = cloud(tw, r[con], PATH, "^", (0, (3.6, 3.6)), "none", 3)
    slope_w = cloud(ax, w[con], INK, "o", (3.6, (3.6, 3.6)), INK, 2)

    ax.set_xticks([1, 10, 100])
    ax.set_yticks([1, 10, 100])
    tw.set_xticks([1e-3, 1e-2])
    tw.minorticks_off()
    ax.set_xlabel("Direct synaptic weight (synapses)", fontsize=6.4, color=INK,
                  labelpad=1.5)
    ax.set_ylabel("Measured response\n(atlas units)", fontsize=6.4, color=INK,
                  labelpad=1.5, linespacing=1.3)
    tw.set_xlabel("Path operator ($\\alpha$ = 0.1)", fontsize=6.4, color=PATH,
                  labelpad=2.0)
    style(ax)
    for side in ("left", "right", "bottom"):
        tw.spines[side].set_visible(False)
    tw.spines["top"].set_linewidth(0.7)
    tw.spines["top"].set_color(PATH)
    tw.tick_params(labelsize=6.0, colors=PATH, length=2.4, width=0.7, pad=1.6)

    for k, (col, mk, fill, rr) in enumerate(((INK, "o", INK, rho_w),
                                             (PATH, "^", "none", rho_r))):
        yk = 0.145 - 0.095 * k
        ax.plot([0.605], [yk], mk, ms=2.6, mfc=fill, mec=col, mew=0.6,
                transform=ax.transAxes, clip_on=False)
        ax.text(0.655, yk, f"$\\rho$ = {rr.statistic:.3f}", transform=ax.transAxes,
                fontsize=6.3, color=col, ha="left", va="center")
    ax.text(0.03, 0.975, f"n = {con.sum()} connected pairs, both $p$ > 0.8",
            transform=ax.transAxes, fontsize=6.0, color=GREY, ha="left",
            va="top")
    save(fig, "fig1_weight_vs_response")
    plt.close(fig)
    print(f"  direct rho = {rho_w.statistic:+.4f} (p={rho_w.pvalue:.3f}, slope {slope_w:+.3f}); "
          f"path rho = {rho_r.statistic:+.4f} (p={rho_r.pvalue:.3f}, slope {slope_r:+.3f}); "
          f"n = {con.sum()}")

    # ------------------------------------------------------ ROC, observed pairs
    ii, jj = np.nonzero(occ3)
    hits = {(idx[p], idx[q]) for p, q in zip(d.pre, d.post)}
    lab = np.array([1.0 if (b_, a_) in hits else 0.0 for a_, b_ in zip(ii, jj)])

    fig = plt.figure(figsize=(2.35, 2.05))
    ax = fig.add_axes([0.24, 0.20, 0.72, 0.70])
    ax.plot([0, 1], [0, 1], ls=(0, (3.4, 2.4)), color=GREY, lw=0.9, zorder=2)

    aucs = {}
    for name, v, color, z in (("path operator", R[jj, ii], PATH, 4),
                              ("direct weight", A[jj, ii], INK, 3)):
        fpr, tpr, _ = roc_curve(lab, v)
        keep = np.unique(np.linspace(0, len(fpr) - 1, min(600, len(fpr))).astype(int))
        aucs[name] = roc_auc_score(lab, v)
        ax.plot(fpr[keep], tpr[keep], color=color, lw=1.4, zorder=z,
                solid_joinstyle="round")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 0.5, 1.0], ["0", "0.5", "1.0"])
    ax.set_yticks([0, 0.5, 1.0], ["0", "0.5", "1.0"])
    ax.set_aspect("equal")
    ax.set_xlabel("False positive rate", fontsize=6.4, color=INK, labelpad=1.5)
    ax.set_ylabel("True positive rate", fontsize=6.4, color=INK, labelpad=1.5)
    style(ax)

    legend = (("Path operator", aucs["path operator"], PATH, 1.4, "-"),
              ("Direct weight", aucs["direct weight"], INK, 1.4, "-"),
              ("Chance", 0.5, GREY, 0.9, (0, (3.4, 2.4))))
    for k, (name, auc, color, lw, ls) in enumerate(legend):
        yk = 0.375 - 0.150 * k
        ax.plot([0.40, 0.52], [yk, yk], color=color, lw=lw, ls=ls,
                transform=ax.transAxes, clip_on=False)
        ax.text(0.555, yk, f"{name}\n(AUC = {auc:.3f})", transform=ax.transAxes,
                fontsize=6.2, color=color, ha="left", va="center", linespacing=1.25)
    ax.text(0.02, 0.98, f"{len(lab):,} observed pairs\n{int(lab.sum()):,} responses",
            transform=ax.transAxes, fontsize=6.0, color=GREY, ha="left", va="top",
            linespacing=1.3)
    save(fig, "fig1b_roc")
    plt.close(fig)
    print(f"  AUC direct {aucs['direct weight']:.4f}, path {aucs['path operator']:.4f}, "
          f"on {len(lab)} observed pairs / {int(lab.sum())} responses")


if __name__ == "__main__":
    main()
