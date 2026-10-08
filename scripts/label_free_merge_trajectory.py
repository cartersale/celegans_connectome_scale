"""Run a label-free agglomerative merge criterion to 60, 120 and 180 merges and
compare its propagation-prediction loss with merging cell classes and with
size-matched random partitions.

Beyond n/2 merges disjoint pairwise selection is impossible, so the criterion
merges cells, not only neurons. For the current partition with cell-profile
matrix Q = (H^T A H) / n_a (mean member out-flow into each cell) and cell sizes
n_a, the cost of merging cells a and b is the Ward-style weighted distance

    sqrt( n_a n_b / (n_a + n_b) ) * || Q[a,:] - Q[b,:] ||  /  ||Q||_F

averaged over the out- (A) and in- (A^T) directions. The weight makes merging two
large cells cost more than merging two small ones. At each step the cheapest
merge is taken and Q is recomputed. For two singletons this is the row distance
of A without the column-averaging correction of the exact defect in
prospective_merges.py.

Arms at each merge count:
  defect-greedy       the criterion above, connectivity only
  biological classes  whole cell classes merged in one random order until the
                      merge count is reached (the last class may be partial)
Loss is the drop in AUC for predicting Randi 2023 responses among observed
pairs without a direct synapse, lifted quotient resolvent (alpha = 0.1) against
the full resolvent. Each arm is compared with --n-rand permutations of its own
labels (seeds 900+b), so the null has the same cell-size profile. Intervals are
a cluster bootstrap over the class of the stimulated neuron (seed 7, --n-boot
draws), identical for every arm. Also reports the fraction of co-cell pairs in
the defect-greedy partition that share a cell class.

Inputs: C. elegans connectome (conres.celegans), Randi 2023 wild-type responses
and observation counts (conres.randi).
Outputs: results/label_free_merge_trajectory.csv (n_merge, arm, k, loss, lo, hi,
null_mean, null_sd).
Usage: label_free_merge_trajectory.py [--merges 60,120,180] [--n-rand 20]
       [--n-boot 1000] [--seed 0]
"""
from __future__ import annotations

import argparse
import csv
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from conres import celegans as ce, randi          # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ALPHA = 0.1


def cell_profiles(A, lab):
    k = int(lab.max() + 1)
    n = A.shape[0]
    H = np.zeros((n, k))
    H[np.arange(n), lab] = 1.0
    sizes = H.sum(0)
    return (H.T @ A @ H) / sizes[:, None], sizes


def ward_costs(Q, sizes):
    """Weighted pairwise profile distance between cells; inf on the diagonal."""
    sq = (Q ** 2).sum(1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2 * Q @ Q.T, 0.0)
    w = np.outer(sizes, sizes) / (sizes[:, None] + sizes[None, :])
    c = np.sqrt(w * d2) / max(np.linalg.norm(Q), 1e-12)
    np.fill_diagonal(c, np.inf)
    return c


def defect_greedy(A, n_merge, allowed_mask=None):
    """Agglomerative merging by the Ward-style cost for `n_merge` steps; cells
    containing a neuron outside `allowed_mask` are never merged."""
    n = A.shape[0]
    lab = np.arange(n)
    if allowed_mask is not None:
        pass
    for _ in range(n_merge):
        rel = np.unique(lab, return_inverse=True)[1]
        Qo, sz = cell_profiles(A, rel)
        Qi, _ = cell_profiles(A.T, rel)
        C = (ward_costs(Qo, sz) + ward_costs(Qi, sz)) / 2.0
        if allowed_mask is not None:
            blocked = np.array([not allowed_mask[np.flatnonzero(rel == c)].all()
                                for c in range(rel.max() + 1)])
            C[blocked, :] = np.inf
            C[:, blocked] = np.inf
        if not np.isfinite(C).any():
            break
        a, b = np.unravel_index(np.argmin(C), C.shape)
        lab = np.where(rel == b, a, rel)
        lab = np.unique(lab, return_inverse=True)[1]
    return np.unique(lab, return_inverse=True)[1]


def bio_trajectory(A, labels, n_merge, rng):
    """Merge whole cell classes, in random order, until `n_merge` merges are made."""
    n = A.shape[0]
    classes = [np.flatnonzero(labels == c) for c in np.unique(labels)]
    classes = [c for c in classes if c.size > 1]
    rng.shuffle(classes)
    cell = np.arange(n)
    merged = 0
    for grp in classes:
        for m in grp[1:]:
            cell[m] = grp[0]
            merged += 1
            if merged >= n_merge:
                return np.unique(cell, return_inverse=True)[1]
    return np.unique(cell, return_inverse=True)[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merges", default="60,120,180")
    ap.add_argument("--n-rand", type=int, default=20)
    ap.add_argument("--n-boot", type=int, default=1000)
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
    ys, fs, gs = y[sub], R[jj, ii][sub], labels[jj][sub]
    auc_full = roc_auc_score(ys, fs)

    def loss_point(lab):
        """AUC loss without a bootstrap interval, for the permutation nulls."""
        RQ = randi.resolvent(randi.quotient(A, lab), ALPHA)
        q = RQ[np.ix_(lab, lab)][jj, ii][sub]
        return auc_full - roc_auc_score(ys, q)

    def loss_ci(lab):
        RQ = randi.resolvent(randi.quotient(A, lab), ALPHA)
        q = RQ[np.ix_(lab, lab)][jj, ii][sub]
        point = auc_full - roc_auc_score(ys, q)
        rng = np.random.default_rng(7)
        uq = np.unique(gs); where = {g: np.flatnonzero(gs == g) for g in uq}
        bs = []
        for _ in range(args.n_boot):
            k = np.concatenate([where[g] for g in rng.choice(uq, uq.size, replace=True)])
            if len(np.unique(ys[k])) > 1:
                bs.append(roc_auc_score(ys[k], fs[k]) - roc_auc_score(ys[k], q[k]))
        bs = np.array(bs)
        return point, np.percentile(bs, 2.5), np.percentile(bs, 97.5)

    print(f"n={n}, full-operator AUC (no-synapse) {auc_full:.4f}")
    print(f"intervals = cluster bootstrap over {len(np.unique(gs))} stimulated-neuron "
          f"classes, identical for every arm\n")

    rng = np.random.default_rng(args.seed)
    rows = []
    for nm in [int(x) for x in args.merges.split(",")]:
        k_target = n - nm
        print(f"=== {nm} merges  (k = {k_target}) ===")
        dl = defect_greedy(A, nm)
        arms = [("defect-greedy (no labels)", dl)]
        bl = bio_trajectory(A, labels, nm, rng)
        arms.append(("biological classes", bl))
        for name, lab in arms:
            kk = int(lab.max() + 1)
            p, lo, hi = loss_ci(lab)
            nulls = []
            for b in range(args.n_rand):
                nulls.append(loss_point(np.random.default_rng(900 + b).permutation(lab)))
            nulls = np.array(nulls)
            print(f"  {name:>26} k={kk:3d}  loss {p:+.4f} [{lo:+.4f}, {hi:+.4f}]"
                  f"   its size-matched random {nulls.mean():+.4f} +-{nulls.std(ddof=1):.4f}")
            rows.append({"n_merge": nm, "arm": name, "k": kk, "loss": round(p, 5),
                         "lo": round(lo, 5), "hi": round(hi, 5),
                         "null_mean": round(float(nulls.mean()), 5),
                         "null_sd": round(float(nulls.std(ddof=1)), 5)})
        same = 0; tot = 0
        for c in range(dl.max() + 1):
            m = np.flatnonzero(dl == c)
            if m.size < 2:
                continue
            for a in range(m.size):
                for b in range(a + 1, m.size):
                    tot += 1
                    same += mapping[nodes[m[a]]] == mapping[nodes[m[b]]]
        print(f"  {'':>26} co-cell pairs sharing a biological class: "
              f"{same}/{tot} = {100*same/max(tot,1):.0f}%")
        print()

    with (ROOT / "results/label_free_merge_trajectory.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("wrote results/label_free_merge_trajectory.csv")


if __name__ == "__main__":
    main()
