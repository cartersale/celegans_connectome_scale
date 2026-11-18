"""Structural loss of the class-level path operator for the six grouping methods of
Fig. 3B, on the same grid of small k, with size-matched random partitions.

Partitions are those of `grouping_methods.py` (method_builders, same --n-seed).
For each partition the class-level operator R_Q = randi.class_leverage(A, labels,
alpha = 0.1) is compared with the full operator R = randi.resolvent(A, 0.1):

  struct_all   1 - Spearman(R_Q, R) over all ordered off-diagonal pairs
  struct_occ   the same over the occ3-observed pairs (the incidence_all pair set
               of conres.observables)

Loss is averaged over seeds, with the across-seed range. The random column is the
mean loss of --n-rand label permutations of the seed-0 partition (rng seeded with
k, as in `grouping_methods.py`). A partition that puts every pair of a set into
one block pair makes R_Q constant there; Spearman is then undefined and the loss
is NaN (spectral, k = 2, measured pairs), which the panel leaves undrawn.

Inputs:  conres.observables.load(); results/coarse_trajectory_paths.npz.
Outputs: results/grouping_methods_structural.csv (method, k, measure, loss,
         seed_min, seed_max, random); figures/supp/structure_vs_resolution.svg.
Usage:   python scripts/grouping_methods_structural.py [--n-seed 5] [--n-rand 20]
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from conres import figstyle as fs, observables as ob, randi             # noqa: E402
from grouping_methods import ALPHA, K_GRID, method_builders             # noqa: E402
from fig3_coarse_grainings import FS, INK, LIGHT, METHODS                # noqa: E402

MEASURES = (("struct_all", "operator structure, all pairs"),
            ("struct_occ", "operator structure, measured pairs"))


def compute(n_seed, n_rand):
    D = ob.load()
    A, nodes = D["A"], D["nodes"]
    n = A.shape[0]
    R = randi.resolvent(A, ALPHA)
    off = ~np.eye(n, dtype=bool)
    oj, oi = np.nonzero(off)
    j, i = D["obs"]["incidence_all"][:2]
    pairs = {"struct_all": (oj, oi), "struct_occ": (j, i)}
    builders = method_builders(A, nodes, R, n_seed)

    def losses(lab):
        Q = randi.class_leverage(A, lab, ALPHA)
        return {m: 1 - stats.spearmanr(Q[pj, pi], R[pj, pi]).statistic
                for m, (pj, pi) in pairs.items()}

    rows = []
    for k in K_GRID:
        for method, labs in builders(k):
            per_seed = [losses(lab) for lab in labs]
            rng = np.random.default_rng(k)
            rnd = [losses(rng.permutation(labs[0])) for _ in range(n_rand)]
            for m in pairs:
                vals = [ps[m] for ps in per_seed]
                rows.append({"method": method, "k": k, "measure": m,
                             "loss": float(np.mean(vals)), "seed_min": float(np.min(vals)),
                             "seed_max": float(np.max(vals)),
                             "random": float(np.mean([r[m] for r in rnd]))})
        print(f"  k = {k} done", flush=True)
    return pd.DataFrame(rows)


def draw(d, path):
    fig = plt.figure(figsize=(4.6, 2.4))
    axes = [fig.add_axes([0.095, 0.33, 0.39, 0.60]), fig.add_axes([0.595, 0.33, 0.39, 0.60])]
    for ax, (m, title) in zip(axes, MEASURES):
        ax.axvline(9, color=INK, lw=0.7, ls=(0, (2.5, 2)), zorder=0.5)
        rnd = d[(d.method == "loss-greedy") & (d.measure == m)].sort_values("k")
        ax.fill_between(rnd.k, 0, 100 * (1 - rnd.random), color=LIGHT, lw=0, zorder=0)
        for key, name, col, ls, mk in METHODS:
            s = d[(d.method == key) & (d.measure == m)].sort_values("k")
            y = 100 * (1 - s.loss)
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
    fs.save(fig, path, check=fs.check_path("check_structure_vs_resolution.png"))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seed", type=int, default=5)
    ap.add_argument("--n-rand", type=int, default=20)
    args = ap.parse_args()
    fs.use()

    d = compute(args.n_seed, args.n_rand)
    d.to_csv(ROOT / "results/grouping_methods_structural.csv", index=False)
    for m, title in MEASURES:
        t = d[d.measure == m].pivot(index="k", columns="method", values="loss")
        t["random"] = d[(d.measure == m) & (d.method == "loss-greedy")].set_index("k").random
        print(f"\n== {title}: loss, 1 - Spearman(R_Q, R) (mean over seeds)")
        print(t.loc[list(K_GRID), [x[0] for x in METHODS] + ["random"]].round(3).to_string())
    draw(d, ROOT / "figures/supp/structure_vs_resolution.svg")
    print("\nwrote results/grouping_methods_structural.csv, figures/supp/structure_vs_resolution.svg")


if __name__ == "__main__":
    main()
