"""Diagnostics for metrics that degenerate on sparse binary data.

Three failure modes are checked:
  * a rank correlation dominated by one large tie block in the predictor;
  * per-fold model fitting at a low base rate, where pooled out-of-fold
    probabilities stop forming a single ranking;
  * overlap similarities (Jaccard) between sparse binary vectors, where zero
    overlap is the most likely outcome under chance.

Each function returns a dict of summary statistics and a list of `flags`
(["none"] if nothing is flagged). Run on the vectors that enter the metric:

    from conres.metric_guard import guard_binary_similarity, guard_predictor
    print(guard_binary_similarity(v1, v2))
    print(guard_predictor(y, score))
"""
from __future__ import annotations

import numpy as np


def guard_predictor(y, score) -> dict:
    """Diagnostics for a continuous predictor of a binary outcome.

    Flags a tie block covering > 50% of the predictor, a base rate < 10%, and
    fewer than 10 distinct predictor values.
    """
    y = np.asarray(y, float)
    s = np.asarray(score, float)
    vals, counts = np.unique(s, return_counts=True)
    tie_frac = counts.max() / s.size
    base = y.mean()
    out = {
        "n": int(s.size),
        "base_rate": round(float(base), 4),
        "largest_tie_block": round(float(tie_frac), 4),
        "distinct_values": int(vals.size),
    }
    flags = []
    if tie_frac > 0.5:
        flags.append(f"TIE BLOCK {tie_frac:.0%} of the predictor is one value -- "
                     "rank correlations are dominated by it; prefer AUC or a "
                     "within-nonzero analysis, and never compare across "
                     "predictors with different tie structure")
    if base < 0.10:
        flags.append(f"LOW BASE RATE {base:.1%} -- do not fit per-fold models and "
                     "pool their probabilities into one AUC; use the model-free "
                     "AUC for a single monotone predictor")
    if vals.size < 10:
        flags.append(f"COARSE {vals.size} distinct values -- ties drive the estimate")
    out["flags"] = flags or ["none"]
    return out


def guard_binary_similarity(d1, d2, n_perm: int = 2000, seed: int = 0) -> dict:
    """Diagnostics for a similarity between two sparse binary vectors.

    Reports Jaccard, its mean under `n_perm` independent random placements with
    the observed counts, Cohen's kappa, and the chance probability of zero
    overlap. Flags a zero-overlap probability > 0.2 (a Jaccard of 0 is then
    uninformative; use kappa) and min count < 5.
    """
    d1 = np.asarray(d1).astype(bool)
    d2 = np.asarray(d2).astype(bool)
    n = d1.size
    k1, k2 = int(d1.sum()), int(d2.sum())
    rng = np.random.default_rng(seed)

    def jac(a, b):
        u = (a | b).sum()
        return (a & b).sum() / u if u else np.nan

    obs = jac(d1, d2)
    null = np.empty(n_perm)
    zeros = 0
    for i in range(n_perm):
        a = np.zeros(n, bool); a[rng.choice(n, k1, replace=False)] = True
        b = np.zeros(n, bool); b[rng.choice(n, k2, replace=False)] = True
        null[i] = jac(a, b)
        zeros += (a & b).sum() == 0
    po = (d1 == d2).mean()
    p1, p2 = d1.mean(), d2.mean()
    pe = p1 * p2 + (1 - p1) * (1 - p2)
    kappa = (po - pe) / (1 - pe) if pe < 1 else np.nan

    out = {
        "n": n, "k1": k1, "k2": k2,
        "jaccard": round(float(obs), 4),
        "jaccard_null_mean": round(float(np.nanmean(null)), 4),
        "kappa": round(float(kappa), 4),
        "p_chance_zero_overlap": round(zeros / n_perm, 4),
    }
    flags = []
    if zeros / n_perm > 0.2:
        flags.append(f"FLOOR zero overlap happens {zeros/n_perm:.0%} of the time by "
                     "chance -- a Jaccard of 0 does not mean dissimilar. Use kappa, "
                     "which corrects for chance agreement given these marginals")
    if min(k1, k2) < 5:
        flags.append(f"SPARSE min(k)={min(k1,k2)} -- the similarity is not estimable; "
                     "filter on a minimum count before ranking units by it")
    out["flags"] = flags or ["none"]
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 150
    print("-- a pair like I1/BAG: 4% detection, few responses each")
    a = np.zeros(n, bool); a[rng.choice(n, 4, replace=False)] = True
    b = np.zeros(n, bool); b[rng.choice(n, 5, replace=False)] = True
    for k, v in guard_binary_similarity(a, b).items():
        print(f"   {k}: {v}")
    print("-- a predictor like syn_direct: 80% exact zeros, 4% base rate")
    s = np.where(rng.random(5000) < 0.2, rng.random(5000), 0.0)
    y = (rng.random(5000) < 0.04).astype(float)
    for k, v in guard_predictor(y, s).items():
        print(f"   {k}: {v}")
