"""Plot pairwise transfer efficiency against merge-set overlap for every pair of
Witvliet 2021 datasets, with an exact Mantel test.

For each ordered pair (inferred in a, scored on b) the transfer efficiency is

    (L_random,b - L_ab) / (L_random,b - L_bb),

the fraction of b's within-animal advantage over a size-matched random partition
that a's partition keeps on b's graph (1 = as good as b's own, 0 = random). This
is the retention of fig4d_transfer_matrix.py before averaging over donors. Overlap is
the Jaccard index between the two animals' merge sets (shared / distinct merges).

Each unordered pair is drawn once at the mean of its two efficiencies, with a
vertical segment spanning them, plus a least-squares line. The 28 pairs are not
independent (each animal appears in seven), so the Spearman correlation is
tested by a two-sided Mantel test over all 8! = 40,320 permutations of animal
identities, giving an exact p value.

Merge sets are recomputed here with the criterion and pool of
cross_animal_transfer.py (pairwise_defect and greedy_disjoint from
prospective_merges.py, pool_mask from cross_animal_transfer.py, 30
disjoint merges). As a consistency check against that script, the mean Jaccard
must be 0.215 +- 0.0015 or the script exits.

Inputs: results/cross_animal_transfer_m30.csv (loss matrix and random
arm); Witvliet 2021 datasets via conres.celegans.
Outputs: results/fig4e_transfer_overlap.csv (a, b, jaccard, eff_a_to_b,
eff_b_to_a, eff_mean), figures/fig4/fig4e_transfer_overlap.svg.
"""
from __future__ import annotations

import itertools
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
sys.path.insert(0, str(ROOT / "scripts"))
from conres import celegans as ce, figstyle as fs                          # noqa: E402
from prospective_merges import pairwise_defect, greedy_disjoint       # noqa: E402
from cross_animal_transfer import pool_mask                          # noqa: E402

OUT = ROOT / "figures/fig4"
DATA = ROOT / "results/cross_animal_transfer_m30.csv"
ANIMALS = ["1L", "2L", "3L", "4L", "5L", "6L", "7a", "8a"]
TRANS, RAND = "#2a5fd6", "#8b9196"
FS = 6.5
M_MERGE = 30


def merge_sets():
    nodes = ce.witvliet_common_neurons("all")
    mats = [ce.witvliet_matrix(p, nodes, "all") for _, _, p in ce.witvliet_files()]
    pool = np.flatnonzero(pool_mask(mats))
    allowed = np.zeros(len(nodes), bool)
    allowed[pool] = True
    sets = [{frozenset(p) for p in greedy_disjoint(
        (pairwise_defect(A) + pairwise_defect(A.T)) / 2.0, M_MERGE, allowed)} for A in mats]
    J = np.full((8, 8), np.nan)
    for a, b in itertools.combinations(range(8), 2):
        J[a, b] = J[b, a] = len(sets[a] & sets[b]) / len(sets[a] | sets[b])
    return J


def efficiency():
    d = pd.read_csv(DATA)
    L = np.full((8, 8), np.nan)
    for _, r in d[d.kind == "loss"].iterrows():
        L[ANIMALS.index(r.train), ANIMALS.index(r.test)] = r.value
    rnd = np.array([d[(d.kind == "random") & (d.test == a)].value.iloc[0] for a in ANIMALS])
    E = (rnd[None, :] - L) / (rnd[None, :] - np.diag(L)[None, :])
    np.fill_diagonal(E, np.nan)
    return E


def mantel_exact(J, S, iu):
    """Spearman between upper triangles, animal identities permuted exhaustively."""
    x = J[iu]
    obs = stats.spearmanr(x, S[iu]).statistic
    null = []
    for p in itertools.permutations(range(8)):
        p = np.array(p)
        null.append(stats.spearmanr(x, S[np.ix_(p, p)][iu]).statistic)
    null = np.array(null)
    return obs, (np.sum(np.abs(null) >= abs(obs) - 1e-12)) / null.size


def main() -> None:
    J = merge_sets()
    iu = np.triu_indices(8, 1)
    mean_j = float(np.mean(J[iu]))
    print(f"merge-set Jaccard: mean {mean_j:.3f} [{J[iu].min():.3f}, {J[iu].max():.3f}]")
    if abs(mean_j - 0.215) > 0.0015:
        raise SystemExit("merge sets do not reproduce cross_animal_transfer.py (mean Jaccard 0.215)")

    E = efficiency()
    off = ~np.eye(8, dtype=bool)
    S = (E + E.T) / 2.0
    print(f"pairwise transfer efficiency: mean {np.nanmean(E[off]):.3f} "
          f"[{np.nanmin(E[off]):.3f}, {np.nanmax(E[off]):.3f}] over 56 ordered pairs")
    rho, p = mantel_exact(J, S, iu)
    slope, icpt = np.polyfit(J[iu], S[iu], 1)
    print(f"Spearman rho (pair-mean efficiency vs Jaccard) {rho:+.3f}, exact Mantel p {p:.4f}")
    print(f"linear fit: efficiency = {icpt:.3f} + {slope:.3f} x Jaccard "
          f"(efficiency at zero overlap {icpt:.2f})")
    lo_pair = np.argmin(J[iu])
    print(f"lowest-overlap pair {ANIMALS[iu[0][lo_pair]]}-{ANIMALS[iu[1][lo_pair]]}: "
          f"Jaccard {J[iu][lo_pair]:.3f}, efficiency {E[iu[0][lo_pair], iu[1][lo_pair]]:.2f} / "
          f"{E[iu[1][lo_pair], iu[0][lo_pair]]:.2f}")

    pd.DataFrame({"a": [ANIMALS[i] for i in iu[0]], "b": [ANIMALS[j] for j in iu[1]],
                  "jaccard": J[iu], "eff_a_to_b": E[iu], "eff_b_to_a": E.T[iu],
                  "eff_mean": S[iu]}).to_csv(ROOT / "results/fig4e_transfer_overlap.csv", index=False)

    # ------------------------------------------------------------------ panel
    fs.use()
    fig, ax = plt.subplots(figsize=(2.2, 2.35))
    fig.subplots_adjust(left=0.2, right=0.97, top=0.88, bottom=0.2)
    x = J[iu]
    lo = np.minimum(E[iu], E.T[iu])
    hi = np.maximum(E[iu], E.T[iu])
    ax.vlines(x, lo, hi, color=TRANS, lw=0.8, alpha=0.45, zorder=2)
    ax.plot(x, S[iu], "o", ms=3.2, mfc=TRANS, mec="white", mew=0.5, zorder=3)
    xs = np.array([0, 0.4])
    ax.plot(xs, icpt + slope * xs, color=TRANS, lw=0.9, ls=(0, (3, 2)), zorder=1)
    ax.axhline(1, color=fs.INK, lw=0.6, zorder=0)
    ax.axhline(0, color=RAND, lw=0.8, ls=(0, (3.4, 2.4)), zorder=0)
    ax.text(0.005, 1.02, "within-animal partition", fontsize=FS - 1.6, color=fs.INK,
            ha="left", va="bottom")
    ax.text(0.395, 0.02, "size-matched random", fontsize=FS - 1.6, color=RAND,
            ha="right", va="bottom")
    ax.text(0.02, 0.28, f"Spearman $\\rho$ = {rho:+.2f}\nexact Mantel p = {p:.3f}",
            fontsize=FS - 1.2, color=fs.INK, ha="left", va="center", linespacing=1.4)
    fs.style(ax)
    ax.set_xlim(0, 0.4)
    ax.set_ylim(-0.08, 1.15)
    ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4])
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_xlabel("shared merges (Jaccard index)", fontsize=FS)
    ax.set_ylabel("transfer efficiency", fontsize=FS)
    ax.set_title("pairs of animals", fontsize=FS, color=fs.INK, pad=3, loc="left")
    fs.save(fig, OUT / "fig4e_transfer_overlap.svg", check=fs.check_path("check_fig4f.png"))


if __name__ == "__main__":
    main()
