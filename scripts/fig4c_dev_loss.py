"""Plot structural loss of the biological cell-class quotient against a
size-matched random partition across the eight Witvliet 2021 datasets, raw and
after thinning.

Loss is 1 - Spearman between the quotient operator read back at neuron pairs and
the full-resolution operator on the same graph (0 preserves the operator's
ordering exactly). The random arm has identical k and cell-size profile.

Loss depends on synapse density, which grows across the series, so the series is
drawn twice: dashed lines are raw; solid lines with +-1 sd bands are after
multinomial thinning to a common budget (BUDGET = 1,200 synapses, THIN_REPS = 10
in dev_sufficiency.py). The annotation is the Spearman correlation of the
thinned bio-random gap (s_gap) with dataset index.

Inputs: results/dev_sufficiency.csv (written by dev_sufficiency.py; columns kind,
i, stage, syn, s_bio, s_bio_sd, s_rand, s_rand_sd, s_gap).
Outputs: figures/fig4/fig4c_dev_loss.svg.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from conres import figstyle as fs                                  # noqa: E402

OUT = ROOT / "figures/fig4"
DATA = ROOT / "results/dev_sufficiency.csv"

BIO, RAND = "#2e9e52", "#8b9196"
FS = 6.5


def main() -> None:
    fs.use()
    d = pd.read_csv(DATA)
    thin = d[d.kind == "thinned"].sort_values("i")
    raw = d[d.kind == "raw"].sort_values("i")
    x = np.arange(len(thin))

    rho, p = stats.spearmanr(thin.i, thin.s_gap)
    print(f"thinned gap vs age: rho = {rho:+.3f}, p = {p:.3f}   "
          f"(raw bio loss rho = {stats.spearmanr(raw.i, raw.s_bio).statistic:+.3f})")

    fig, ax = plt.subplots(figsize=(2.2, 2.35))
    fig.subplots_adjust(left=0.17, right=0.99, top=0.88, bottom=0.20)

    for arm, col, lab in ((("s_rand", "s_rand_sd"), RAND, "size-matched random"),
                          (("s_bio", "s_bio_sd"), BIO, "biological classes")):
        m, sd = thin[arm[0]].to_numpy(), thin[arm[1]].to_numpy()
        ax.fill_between(x, m - sd, m + sd, color=col, alpha=0.18, lw=0)
        ax.plot(x, m, color=col, lw=1.3, marker="o", ms=3.0, mew=0, label=lab, zorder=3)
        ax.plot(x, raw[arm[0]].to_numpy(), color=col, lw=0.8, dashes=(2.5, 2),
                alpha=0.55, zorder=2)

    k = 4
    ax.annotate("", xy=(k, thin.s_bio.iloc[k]), xytext=(k, thin.s_rand.iloc[k]),
                arrowprops=dict(arrowstyle="<->", lw=0.6, color=fs.INK, shrinkA=0, shrinkB=0))
    ax.text(k + 0.18, (thin.s_bio.iloc[k] + thin.s_rand.iloc[k]) / 2,
            f"gap does not widen\nρ = {rho:+.2f}, p = {p:.2f}", fontsize=FS - 1.0,
            color=fs.INK, ha="left", va="center")
    ax.text(len(x) - 1.1, raw.s_bio.iloc[-1] - 0.028, "raw (density-confounded)",
            fontsize=FS - 1.5, color=fs.GREY, style="italic", ha="right", va="top")

    fs.style(ax)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}\n{int(n):,}" for s, n in zip(thin.stage, thin.syn)],
                       fontsize=FS - 1.5)
    ax.set_xlim(-0.4, len(x) - 0.6)
    ax.set_ylim(0, 0.76)
    ax.set_ylabel("structural loss of the quotient", fontsize=FS)
    ax.set_xlabel("dataset  (stage, synapses)", fontsize=FS, labelpad=1.0)
    ax.text(0.0, 1.055, "thinned to 1,200 synapses, 10 replicates ± 1 sd",
            transform=ax.transAxes, fontsize=FS - 1.0, color=fs.GREY)
    ax.legend(fontsize=FS - 1.0, frameon=False, loc="center left",
              bbox_to_anchor=(0.01, 0.62), handlelength=1.4, borderpad=0.1,
              labelspacing=0.35)
    fs.save(fig, OUT / "fig4c_dev_loss.svg", check=fs.check_path("check_fig4b.png"))


if __name__ == "__main__":
    main()
