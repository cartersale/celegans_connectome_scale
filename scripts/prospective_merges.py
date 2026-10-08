"""Score every pairwise merge by its equitability defect and compare
defect-selected, class-pair and random merge sets at a matched size profile.

Defect of a single merge. For the partition that merges only {a,b} and leaves
every other neuron a singleton, P = H (H^T H)^-1 H^T is the identity except for
a 2x2 averaging block, so (I-P)X has row a = (X[a,:]-X[b,:])/2, row b its
negation, and zeros elsewhere. With MP = M with columns a and b each replaced by
their mean,

    ||(I-P) M P||_F^2 = || (MP)[a,:] - (MP)[b,:] ||^2 / 2

i.e. the row difference of M with the entries in columns a and b replaced by the
averaged values. The denominator ||MP||_F^2 likewise follows from column norms
and the column Gram matrix. Both vectorise via Gram matrices, so the defect
gamma = ||(I-P)MP||_F / ||MP||_F is computed exactly for all n(n-1)/2 pairs.
The score is the mean of the out-direction (M = A) and in-direction (M = A^T)
defects of the all-synapse matrix A; it uses connectivity only, no labels.

Arms, each with N_MERGE = 60 merges into cells of size two (k = n - 60):
  defect-selected  the 60 lowest-defect disjoint pairs, chosen greedily
  class pairs      60 pairs drawn from the size-2 cell classes, --n-rand draws
                   (labelled "biological (bilateral)" in the output)
  random disjoint  60 random disjoint pairs, --n-rand draws
With --min-degree, neurons whose binary in+out degree is below the floor are
excluded from all arms.

Each partition is scored by (i) the drop in AUC for predicting Randi 2023
responses among observed pairs without a direct synapse, using the lifted
quotient resolvent (alpha = 0.1) against the full resolvent, and (ii) structural
loss, 1 - Spearman between lifted quotient and full resolvent over off-diagonal
entries. Low-degree neurons have near-empty, hence similar, rows, so the script
also reports the degrees of the selected pairs and the fraction of selected
merges that join two neurons of the same cell class, against the probability
that two distinct candidates share a cell class.

Inputs: C. elegans connectome (conres.celegans), Randi 2023 wild-type responses
and observation counts (conres.randi).
Outputs: results/prospective_merges.csv (arm, rep, auc_loss, struct_loss),
overwritten on every run regardless of --min-degree; the summary goes to stdout.
Produces the degree-floor table in the Supplementary Information
(run with --min-degree 0, 20, 29, 40; stdout saved as
results/prospective_merges_floor*.log).
Usage: prospective_merges.py [--min-degree 0] [--n-rand 30] [--seed 0]
"""
from __future__ import annotations

import argparse
import csv
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from conres import celegans as ce, randi          # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ALPHA = 0.1
N_MERGE = 60


def pairwise_defect(M: np.ndarray) -> np.ndarray:
    """g[a, b] = exact defect ||(I-P) M P||_F / ||M P||_F of merging only {a, b};
    the diagonal is set to inf."""
    n = M.shape[0]
    sq = (M ** 2).sum(1)
    G = M @ M.T
    rowdiff2 = sq[:, None] + sq[None, :] - 2 * G          # ||row_a - row_b||^2
    d = np.diag(M)
    Maa = d[:, None] + np.zeros((1, n))                   # M[a,a]
    Mab = M                                              # M[a,b]
    Mba = M.T                                            # M[b,a]
    Mbb = np.zeros((n, 1)) + d[None, :]                   # M[b,b]
    raw = (Maa - Mba) ** 2 + (Mab - Mbb) ** 2             # contribution of cols a,b
    val = (Maa + Mab - Mba - Mbb) / 2.0                   # corrected shared entry
    num2 = np.maximum(rowdiff2 - raw + 2 * val ** 2, 0.0) / 2.0
    # denominator ||MP||_F^2, which varies slightly with the pair
    colsq = (M ** 2).sum(0)
    tot = (M ** 2).sum()
    cross = M.T @ M
    mean_sq = (colsq[:, None] + colsq[None, :] + 2 * cross) / 4.0   # ||(c_a+c_b)/2||^2
    den2 = np.maximum(tot - colsq[:, None] - colsq[None, :] + 2 * mean_sq, 1e-12)
    g = np.sqrt(num2 / den2)
    np.fill_diagonal(g, np.inf)
    return g


def greedy_disjoint(score: np.ndarray, k: int, allowed: np.ndarray) -> list[tuple[int, int]]:
    """Lowest-scoring disjoint pairs, greedily."""
    n = score.shape[0]
    order = np.argsort(score, axis=None)
    used = np.zeros(n, bool)
    used[~allowed] = True
    out = []
    for flat in order:
        a, b = divmod(int(flat), n)
        if a >= b or used[a] or used[b] or not np.isfinite(score[a, b]):
            continue
        used[a] = used[b] = True
        out.append((a, b))
        if len(out) == k:
            break
    return out


def labels_from_pairs(n: int, pairs: list[tuple[int, int]]) -> np.ndarray:
    cell = np.arange(n)
    for a, b in pairs:
        cell[b] = cell[a]
    return np.unique(cell, return_inverse=True)[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-rand", type=int, default=30)
    ap.add_argument("--min-degree", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    A, nodes = ce.build_matrix("all", neurons_only=True)
    labels = ce.partitions(nodes)["class"]
    mapping = ce.assign_classes(nodes)
    n = A.shape[0]
    idx = {x: i for i, x in enumerate(nodes)}

    occ3, _ = randi.build_occ3(nodes, "wt")
    d = randi.load_responses("wildty")
    d = d[d.pre.isin(idx) & d.post.isin(idx)]
    hits = {(idx[p], idx[q]) for p, q in zip(d.pre, d.post)}
    ii, jj = np.nonzero(occ3)
    y = np.array([1.0 if (j, i) in hits else 0.0 for i, j in zip(ii, jj)])
    R = randi.resolvent(A, ALPHA)
    sub = A[jj, ii] == 0
    auc_full = roc_auc_score(y[sub], R[jj, ii][sub])
    off = ~np.eye(n, dtype=bool)
    aj, ai = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    offj, offi = aj[off], ai[off]
    struct_full = R[offj, offi]

    deg = ((A > 0).sum(1) + (A > 0).sum(0))
    allowed = deg >= args.min_degree
    print(f"n={n}, k after {N_MERGE} merges = {n - N_MERGE}")
    print(f"full operator: AUC(no-synapse) {auc_full:.4f}   "
          f"min-degree floor {args.min_degree} keeps {allowed.sum()} neurons\n")

    g = (pairwise_defect(A) + pairwise_defect(A.T)) / 2.0
    sel = greedy_disjoint(g, N_MERGE, allowed)

    # class-pair arm: every cell class of exactly two neurons
    bipairs = []
    for c in np.unique(labels):
        m = np.flatnonzero(labels == c)
        if m.size == 2 and allowed[m].all():
            bipairs.append((int(m[0]), int(m[1])))

    def score(lab):
        RQ = randi.resolvent(randi.quotient(A, lab), ALPHA)
        q = RQ[np.ix_(lab, lab)]
        a = auc_full - roc_auc_score(y[sub], q[jj, ii][sub])
        s = 1 - stats.spearmanr(q[offj, offi], struct_full).statistic
        return a, s

    rng = np.random.default_rng(args.seed)
    arms = {}
    arms["defect-selected"] = [score(labels_from_pairs(n, sel))]
    arms["biological (bilateral)"] = [
        score(labels_from_pairs(n, [bipairs[i] for i in
              rng.choice(len(bipairs), min(N_MERGE, len(bipairs)), replace=False)]))
        for _ in range(args.n_rand)]
    pool = np.flatnonzero(allowed)
    arms["random disjoint"] = []
    for _ in range(args.n_rand):
        pick = rng.permutation(pool)[:2 * N_MERGE]
        arms["random disjoint"].append(
            score(labels_from_pairs(n, list(zip(pick[::2], pick[1::2])))))

    print(f"{'arm':>24} {'AUC loss':>22} {'structural loss':>22}")
    print("-" * 72)
    for name, vals in arms.items():
        a = np.array([v[0] for v in vals]); s = np.array([v[1] for v in vals])
        if len(vals) == 1:
            print(f"{name:>24} {a[0]:>+22.4f} {s[0]:>22.4f}")
        else:
            print(f"{name:>24} {a.mean():>+12.4f} +-{a.std(ddof=1):.4f}"
                  f" {s.mean():>13.4f} +-{s.std(ddof=1):.4f}")

    same = sum(1 for a, b in sel if mapping[nodes[a]] == mapping[nodes[b]])
    print(f"\n  of the {len(sel)} defect-selected merges, {same} join two neurons of the "
          f"SAME biological class ({100*same/len(sel):.0f}%)")
    # Probability that two distinct candidates share a cell class; the
    # expected same-class rate of random disjoint pairs from the pool.
    sizes = np.array(list(Counter(mapping[nodes[i]] for i in pool).values()))
    chance = (sizes * (sizes - 1)).sum() / (pool.size * (pool.size - 1))
    print(f"  chance rate for two distinct candidates: {100*chance:.2f}% "
          f"({same / len(sel) / chance:.0f}x enrichment)")

    # low-degree neurons have near-identical (near-empty) rows
    sel_deg = np.array([deg[a] + deg[b] for a, b in sel])
    bio_deg = np.array([deg[a] + deg[b] for a, b in bipairs])
    print(f"\n  summed degree of merged pairs: defect-selected median {np.median(sel_deg):.0f}"
          f"   biological median {np.median(bio_deg):.0f}   all-neuron median {2*np.median(deg):.0f}")
    print(f"  defect-selected pairs where BOTH neurons are below median degree: "
          f"{sum(1 for a,b in sel if deg[a]<np.median(deg) and deg[b]<np.median(deg))}/{len(sel)}")

    with (ROOT / "results/prospective_merges.csv").open("w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["arm", "rep", "auc_loss", "struct_loss"])
        for name, vals in arms.items():
            for i, (a, s) in enumerate(vals):
                w.writerow([name, i, round(a, 5), round(s, 5)])
    print(f"\nwrote results/prospective_merges.csv")


if __name__ == "__main__":
    main()
