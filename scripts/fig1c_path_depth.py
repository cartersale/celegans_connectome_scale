"""Score the path operator truncated at K steps for response occurrence and magnitude.

The predictor is

    S_K = sum_{k=1..K} (alpha A / rho)^k,    K = 1..KMAX (6), alpha = 0.1,

so K = 1 is rescaled adjacency; the rightmost point ("full") is the untruncated
resolvent (I - alpha A/rho)^-1 - I from conres.randi.

  occurrence  AUC over every observed ordered pair (occ3 > 0), label = response
              released (top panel).
  magnitude   Spearman rho against released response magnitude over responders
              (bottom panel).

Each is computed for all pairs and for the no-direct-synapse subset (A = 0).
The two estimands are stacked on separate axes, each with its own null line
(AUC 0.5, rho 0), rather than drawn on twin axes of one frame.

Bands are 95% percentile intervals from a cluster bootstrap over stimulated-
neuron class (BOOT = 500, seed 0). One set of resamples is shared across depths,
so the intervals support comparison along a curve, not between curves.
Resamples with <= 10 pairs or a constant predictor are skipped.

Inputs:  conres.celegans, conres.randi, conres.figstyle.
Outputs: figures/fig1/fig1c_path_depth.svg, plus a PNG preview in $CONRES_CHECK_DIR.
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
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from conres import celegans as ce, figstyle as fs, randi  # noqa: E402

OUT = ROOT / "figures/fig1"

fs.use()
INK, GREY = fs.INK, fs.GREY
ALL, SUB = "#22282d", "#2a5fd6"              # every pair, and the no-synapse subset
ALPHA, KMAX, BOOT = 0.1, 6, 500


def main() -> None:
    A, nodes = ce.build_matrix("all", neurons_only=True)
    idx = {n: i for i, n in enumerate(nodes)}
    lab = ce.partitions(nodes)["class"]
    rho_spec = np.max(np.abs(np.linalg.eigvals(A)))
    N = ALPHA * A / rho_spec

    occ3, _ = randi.build_occ3(nodes, "wt")
    d = randi.load_responses("wildty")
    d = d[d.pre.isin(idx) & d.post.isin(idx)]

    # occurrence: every observed ordered pair
    ii, jj = np.nonzero(occ3)                       # i responder, j stimulus
    hits = {(idx[p], idx[q]) for p, q in zip(d.pre, d.post)}
    y = np.array([1.0 if (b_, a_) in hits else 0.0 for a_, b_ in zip(ii, jj)])
    g_occ = lab[jj]
    sub_occ = A[jj, ii] == 0

    # magnitude: the responders, with the released atlas value
    mj = d.pre.map(idx).to_numpy()
    mi = d.post.map(idx).to_numpy()
    mag = d["mag"].to_numpy(float)
    g_mag = lab[mj]
    sub_mag = A[mj, mi] == 0

    S, P = np.zeros_like(A), np.eye(len(A))
    p_occ, p_mag = [], []
    for _ in range(KMAX):
        P = P @ N
        S = S + P
        p_occ.append(S[jj, ii].copy())
        p_mag.append(S[mj, mi].copy())
    R = randi.resolvent(A, ALPHA)
    p_occ.append(R[jj, ii])
    p_mag.append(R[mj, mi])

    def draws(grp):
        uq = np.unique(grp)
        rng = np.random.default_rng(0)
        return [np.concatenate([np.flatnonzero(grp == g)
                                for g in rng.choice(uq, uq.size, replace=True)])
                for _ in range(BOOT)]

    dr_occ, dr_mag = draws(g_occ), draws(g_mag)

    def curve(preds, mask, score, drs, n_class):
        pt, lo, hi = [], [], []
        for v in preds:
            pt.append(score(v[mask], mask))
            b = [score(v[s], s) for s in
                 (dr[mask[dr]] for dr in drs) if s.size > 10 and np.ptp(v[s]) > 0]
            b = [x for x in b if np.isfinite(x)]
            lo.append(np.percentile(b, 2.5) if b else pt[-1])
            hi.append(np.percentile(b, 97.5) if b else pt[-1])
        return np.array(pt), np.array(lo), np.array(hi), n_class

    auc = lambda v, m: (roc_auc_score(y[m], v) if len(np.unique(y[m])) > 1 else np.nan)
    spr = lambda v, m: stats.spearmanr(v, mag[m]).statistic

    ones_o = np.ones_like(sub_occ)
    ones_m = np.ones_like(sub_mag)
    series = {
        "occ_all": curve(p_occ, ones_o, auc, dr_occ, np.unique(g_occ).size),
        "occ_sub": curve(p_occ, sub_occ, auc, dr_occ, np.unique(g_occ[sub_occ]).size),
        "mag_all": curve(p_mag, ones_m, spr, dr_mag, np.unique(g_mag).size),
        "mag_sub": curve(p_mag, sub_mag, spr, dr_mag, np.unique(g_mag[sub_mag]).size),
    }
    for k, (pt, lo, hi, nc) in series.items():
        print(f"  {k}: " + "  ".join(f"K{i+1}={p:+.3f}" for i, p in enumerate(pt[:-1]))
              + f"  full={pt[-1]:+.3f}   [{nc} classes]")

    # ------------------------------------------------------------------- plot
    xs = np.arange(1, KMAX + 1, dtype=float)
    xfull = KMAX + 1.7
    fig = plt.figure(figsize=(2.75, 2.75))
    axo = fig.add_axes([0.235, 0.560, 0.735, 0.345])
    axm = fig.add_axes([0.235, 0.135, 0.735, 0.345])

    def draw(ax, key, col, mk):
        pt, lo, hi, _ = series[key]
        ok = np.isfinite(pt)
        ax.fill_between(xs[ok[:-1]], lo[:-1][ok[:-1]], hi[:-1][ok[:-1]],
                        color=col, alpha=0.13, lw=0, zorder=2)
        ax.plot([xs[-1], xfull], pt[-2:], color=col, lw=1.0, ls=(0, (2.2, 1.8)), zorder=3)
        ax.plot(xs[ok[:-1]], pt[:-1][ok[:-1]], color=col, lw=1.3, zorder=4)
        ax.plot([xfull, xfull], [lo[-1], hi[-1]], color=col, lw=1.0, zorder=3)
        ax.plot(np.append(xs, xfull)[ok], pt[ok], mk, ms=2.8, mfc="white", mec=col,
                mew=1.0, ls="none", zorder=5)

    for ax, null in ((axo, 0.5), (axm, 0.0)):
        ax.axhline(null, color=GREY, lw=0.8, ls=(0, (3.4, 2.4)), zorder=1)
        ax.set_xlim(0.5, xfull + 0.5)
        fs.style(ax)
        for dx in (-0.13, 0.02):
            ax.plot([KMAX + 0.75 + dx, KMAX + 0.95 + dx], [-0.012, 0.012],
                    transform=ax.get_xaxis_transform(), color=INK, lw=0.7,
                    clip_on=False, zorder=6)

    draw(axo, "occ_all", ALL, "o")
    draw(axo, "occ_sub", SUB, "^")
    draw(axm, "mag_all", ALL, "o")
    draw(axm, "mag_sub", SUB, "^")

    axo.set_ylim(0.47, 0.685)
    axo.set_yticks([0.50, 0.55, 0.60, 0.65])
    axo.set_xticks(list(xs) + [xfull], [""] * (KMAX + 1))
    axo.set_ylabel("AUC (response\nprediction)", fontsize=6.4, color=INK,
                   labelpad=1.5, linespacing=1.35)

    axm.set_ylim(-0.10, 0.26)
    axm.set_yticks([0.0, 0.1, 0.2])
    axm.set_xticks(list(xs) + [xfull], [f"{int(k)}" for k in xs] + ["full"])
    axm.set_ylabel("$\\rho$, how large\nthe response", fontsize=6.4, color=INK,
                   labelpad=1.5, linespacing=1.35)
    axm.set_xlabel("Longest path included ($K$ steps)", fontsize=6.4, color=INK,
                   labelpad=1.5)

    for k, (col, mk, name) in enumerate(((ALL, "o", "all pairs"),
                                         (SUB, "^", "no direct synapse"))):
        yk = 0.955 - 0.105 * k
        axo.plot([0.055, 0.135], [yk, yk], color=col, lw=1.3,
                 transform=axo.transAxes, zorder=7)
        axo.plot([0.095], [yk], mk, ms=2.8, mfc="white", mec=col, mew=1.0,
                 transform=axo.transAxes, zorder=8)
        axo.text(0.165, yk, name, transform=axo.transAxes, fontsize=5.9, color=col,
                 ha="left", va="center", zorder=8)

    fs.save(fig, OUT / "fig1c_path_depth.svg", check=fs.check_path("check_fig1E.png"))


if __name__ == "__main__":
    main()
