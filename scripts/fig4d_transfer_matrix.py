"""Draw the cross-animal transfer matrix and the per-animal transfer arms for
merge sets inferred in one Witvliet 2021 dataset and scored on another.

c  8x8 matrix: rows are the animal a merge set was inferred in, columns the
   animal it was scored on. Cells are structural loss divided by the column's
   size-matched random loss (same k and cell-size profile, same graph), so 0 is
   lossless and 1 is chance. Dividing by each column's own chance level makes
   columns comparable despite differing synapse density. The diagonal (boxed) is
   the within-animal optimum, scored on the graph it was selected from, and is
   optimistic by construction. The colour scale spans the full interval 0-1.
d  four arms per test animal, all on that animal's graph: within-animal optimum,
   mean of the seven transferred partitions, biological pairs, and random pairs.
   Retention is (random - transferred) / (random - within), as computed in
   cross_animal_transfer.py.

Density is not controlled across columns, so every comparison is within a
column.

Inputs: results/cross_animal_transfer_m30.csv from
cross_animal_transfer.py (cells common to all eight datasets, one fixed
candidate pool, 30 disjoint merges per partition; rows kind in loss, within,
transfer, biological, random, retention).
Outputs: figures/fig4/fig4d_transfer_matrix.svg, figures/fig4/fig4_transfer_arms.svg.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from conres import figstyle as fs                                  # noqa: E402

OUT = ROOT / "figures/fig4"
DATA = ROOT / "results/cross_animal_transfer_m30.csv"

ANIMALS = ["1L", "2L", "3L", "4L", "5L", "6L", "7a", "8a"]
TICK = {"1L": "L1-1", "2L": "L1-2", "3L": "L1-3", "4L": "L1-4",
        "5L": "L2", "6L": "L3", "7a": "A1", "8a": "A2"}
BIO, RAND, WITHIN, TRANS = "#2e9e52", "#8b9196", "#22282d", "#2a5fd6"
LOSS = LinearSegmentedColormap.from_list("loss", ["#ffffff", "#cfe0f2", "#5e8fc7", "#1f3d66"])
FS = 6.5


def load():
    d = pd.read_csv(DATA)
    M = np.full((8, 8), np.nan)
    for _, r in d[d.kind == "loss"].iterrows():
        M[ANIMALS.index(r.train), ANIMALS.index(r.test)] = r.value
    arms = {k: np.array([d[(d.kind == k) & (d.test == a)].value.iloc[0] for a in ANIMALS])
            for k in ("within", "transfer", "biological", "random", "retention")}
    return M, arms


def matrix(M, arms):
    R = M / arms["random"][None, :]          # loss as a fraction of chance, per column
    bio = float(np.mean(arms["biological"] / arms["random"]))
    off = ~np.eye(8, dtype=bool)
    print(f"as a fraction of chance: diagonal {np.diag(R).mean():.3f}, "
          f"off-diagonal {R[off].mean():.3f} ({R[off].min():.3f}-{R[off].max():.3f}), "
          f"biological {bio:.3f}")

    # the colour scale spans the whole interval the ratio is defined on, 0 to
    # chance, rather than the data range; per-cell numbers carry the detail
    lo, hi = 0.0, 1.0
    fig, ax = plt.subplots(figsize=(3.4, 2.75))
    fig.subplots_adjust(left=0.146, right=0.735, top=0.775, bottom=0.145)
    im = ax.imshow(R, cmap=LOSS, vmin=lo, vmax=hi, interpolation="nearest")
    for i in range(8):
        for j in range(8):
            ax.text(j, i, f"{R[i, j]:.2f}".lstrip("0"), ha="center", va="center",
                    fontsize=FS - 2.3, zorder=4,
                    color="white" if R[i, j] > 0.72 else fs.INK)
    for i in range(8):
        ax.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, ec=fs.INK,
                               lw=0.9, zorder=3))
    ax.set_xticks(range(8)); ax.set_yticks(range(8))
    ax.set_xticklabels([TICK[a] for a in ANIMALS], fontsize=FS - 1.5)
    ax.set_yticklabels([TICK[a] for a in ANIMALS], fontsize=FS - 1.5)
    ax.tick_params(length=0, pad=1.5, colors=fs.INK)
    for sp in ax.spines.values():
        sp.set_linewidth(0.5); sp.set_color(fs.FAINT)
    ax.set_xlabel("scored on", fontsize=FS, labelpad=2.0)
    ax.set_ylabel("inferred in", fontsize=FS, labelpad=2.0)
    ax.text(0.0, 1.155, "transferred partitions, as a fraction of chance",
            transform=ax.transAxes, fontsize=FS, color=fs.INK)
    ax.text(0.0, 1.048, "1.00 = a size-matched random partition on the same graph;\n"
            "boxed = the within-animal optimum, scored on its own graph",
            transform=ax.transAxes, fontsize=FS - 1.8, color=fs.GREY, va="bottom",
            linespacing=1.45)
    cax = fig.add_axes([0.755, 0.145, 0.027, 0.555])
    cb = fig.colorbar(im, cax=cax)
    cb.outline.set_linewidth(0.5); cb.outline.set_edgecolor(fs.FAINT)
    cb.set_ticks([])
    yax = cb.ax.get_yaxis_transform()
    worst = float(R[off].max())
    for y, col, lab in ((1.00, fs.INK, "1.00  chance"),
                        (worst, fs.INK, f"{worst:.2f}".lstrip("0") + "  worst transfer"),
                        (bio, BIO, f"{bio:.2f}".lstrip("0") + "  cell classes"),
                        (0.00, fs.GREY, "0  lossless")):
        cb.ax.plot([0, 1.0], [y, y], color=col, lw=1.0, clip_on=False, zorder=5,
                   transform=yax)
        cb.ax.text(1.45, y, lab, fontsize=FS - 1.8, color=col, ha="left", va="center",
                   transform=yax)
    fs.save(fig, OUT / "fig4d_transfer_matrix.svg", check=fs.check_path("check_fig4c.png"))


def arms_panel(arms):
    x = np.arange(8)
    fig, ax = plt.subplots(figsize=(2.2, 2.35))
    fig.subplots_adjust(left=0.18, right=0.995, top=0.88, bottom=0.20)
    series = (("random", RAND, "size-matched random", "o"),
              ("transfer", TRANS, "transferred from another animal", "o"),
              ("within", WITHIN, "within-animal optimum (ceiling)", "o"),
              ("biological", BIO, "cell classes", "o"))
    for key, col, lab, mk in series:
        ax.plot(x, arms[key], color=col, lw=1.2, marker=mk, ms=3.0, mew=0,
                label=lab, zorder=3)
    ax.fill_between(x, arms["within"], arms["transfer"], color=TRANS, alpha=0.14, lw=0)

    r = arms["retention"]
    ax.text(0.5, 0.13, f"transfer retains {100 * r.mean():.1f}% of the within-animal\n"
            f"advantage over random ({100 * r.min():.0f}–{100 * r.max():.0f}% by animal)",
            fontsize=FS - 1.0, color=TRANS, ha="left", va="center")

    fs.style(ax)
    ax.set_xticks(x)
    ax.set_xticklabels([TICK[a] for a in ANIMALS], fontsize=FS - 1.5)
    ax.set_xlim(-0.4, 7.4); ax.set_ylim(0, 0.30)
    ax.set_ylabel("structural loss", fontsize=FS)
    ax.set_xlabel("animal the partition was scored on", fontsize=FS, labelpad=1.0)
    ax.legend(fontsize=FS - 1.0, frameon=False, loc="upper left",
              bbox_to_anchor=(0.005, 1.03), handlelength=1.4, borderpad=0.1,
              labelspacing=0.3)
    fs.save(fig, OUT / "fig4_transfer_arms.svg", check=fs.check_path("check_fig4d.png"))


def main() -> None:
    fs.use()
    M, arms = load()
    off = ~np.eye(8, dtype=bool)
    print(f"within {np.diag(M).mean():.4f}   transferred {M[off].mean():.4f}   "
          f"biological {arms['biological'].mean():.4f}   random {arms['random'].mean():.4f}")
    print(f"retention {100 * arms['retention'].mean():.1f}%  "
          f"({100 * arms['retention'].min():.0f}-{100 * arms['retention'].max():.0f}%)   "
          f"off-diagonal range {M[off].min():.3f}-{M[off].max():.3f}")
    matrix(M, arms)
    arms_panel(arms)


if __name__ == "__main__":
    main()
