"""Functional observables and the scoring functions shared by the coarse-scale analyses.

`load` returns, per observable, (source index, target index, value, metric):

  incidence_all    detected response j -> i over every observed pair; AUC
  incidence_nosyn  the same, restricted to pairs with no direct synapse j -> i
  correlation      mean across-recording Pearson r (>= `min_animals`
                   recordings); Spearman against S[j, i] + S[i, j]
  amplitude        |stimulus-triggered average| over the response window, for
                   observed pairs with >= `min_events` stimulations; Spearman
  magnitude        released atlas magnitude among detected responses; Spearman

Scores are performance above chance (AUC - 0.5, or Spearman rho), so a loss is
a fraction of each observable's own headroom.

The pair sets are built by `setup` in scripts/two_observables.py and the
stimulus-triggered averages by `build_sta` in
scripts/observable_resolution.py, which are imported at call time.
"""
from __future__ import annotations

import sys

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

from conres import celegans as ce, randi
from conres.paths import REPO_ROOT

OBSERVABLES = ("incidence_all", "incidence_nosyn", "correlation", "amplitude", "magnitude")
POWERED = ("incidence_all", "correlation")    # narrowest intervals; they resolve boundaries


def load(min_animals: int = 3, min_events: int = 2) -> dict:
    """Dict with Cook adjacency `A`, `nodes`, class codes `cls` and observables `obs`."""
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from two_observables import setup as setup_corr          # noqa: E402
    from observable_resolution import build_sta, PRE, WIN   # noqa: E402

    dat = setup_corr(min_animals, 0, False)
    A, nodes = dat["A"], dat["nodes"]
    idx = {x: i for i, x in enumerate(nodes)}
    ii, jj, y = dat["ii"], dat["jj"], dat["y"]
    ns = A[jj, ii] == 0
    acc, cnt = build_sta(nodes)
    sta = acc / np.maximum(cnt, 1)[..., None]
    amp = np.abs(sta[:, :, PRE + WIN[0]:PRE + WIN[1]].mean(-1))
    d = randi.load_responses("wildty")
    d = d[d.pre.isin(idx) & d.post.isin(idx)]
    rj, ri = d.pre.map(idx).to_numpy(), d.post.map(idx).to_numpy()
    ev = cnt[ii, jj] >= min_events
    obs = {
        "incidence_all": (jj, ii, y, "auc"),
        "incidence_nosyn": (jj[ns], ii[ns], y[ns], "auc"),
        "correlation": (dat["iu"], dat["ju"], dat["r"], "corr"),
        "amplitude": (jj[ev], ii[ev], amp[ii[ev], jj[ev]], "rho"),
        "magnitude": (rj, ri, d["mag"].to_numpy(float), "rho"),
    }
    cls = np.unique(ce.partitions(nodes)["class"], return_inverse=True)[1]
    return dict(A=A, nodes=nodes, cls=cls, obs=obs)


def predict(kind: str, S: np.ndarray, j: np.ndarray, i: np.ndarray) -> np.ndarray:
    """S[j, i], symmetrised to S[j, i] + S[i, j] for the correlation metric."""
    return S[j, i] + S[i, j] if kind == "corr" else S[j, i]


def score(kind: str, p: np.ndarray, v: np.ndarray, sel=slice(None)) -> float:
    """Performance above chance: AUC - 0.5, or Spearman rho."""
    p, v = p[sel], v[sel]
    if kind == "auc":
        return roc_auc_score(v, p) - 0.5 if len(np.unique(v)) > 1 else np.nan
    return float(stats.spearmanr(p, v).statistic)


def boot_indices(obs: dict, cls: np.ndarray, n_boot: int, seed: int = 0) -> dict:
    """`n_boot` cluster-bootstrap index arrays per observable.

    Clusters are the classes of the first index of each pair: the stimulated
    neuron, or for correlation the lower-indexed neuron. Classes are drawn with
    replacement and all their pairs included.
    """
    rng = np.random.default_rng(seed)
    out = {}
    for o, (j, _, _, _) in obs.items():
        g = cls[j]
        where = {c: np.flatnonzero(g == c) for c in np.unique(g)}
        uq = np.array(list(where))
        out[o] = [np.concatenate([where[c] for c in rng.choice(uq, uq.size)])
                  for _ in range(n_boot)]
    return out


def block_means(S: np.ndarray, lab: np.ndarray) -> np.ndarray:
    """Block-mean of S at neuron resolution, diagonal excluded.

    Each pair (i, j) gets the mean of S over all off-diagonal pairs in the
    block (lab[i], lab[j]); empty blocks are 0. `lab` must be 0..k-1 codes.
    """
    n = S.shape[0]
    k = int(lab.max() + 1)
    H = np.zeros((n, k))
    H[np.arange(n), lab] = 1.0
    Z = S.copy()
    np.fill_diagonal(Z, 0.0)
    tot = H.T @ Z @ H
    cnt = H.T @ (np.ones((n, n)) - np.eye(n)) @ H
    M = np.divide(tot, cnt, out=np.zeros_like(tot), where=cnt > 0)
    return M[np.ix_(lab, lab)]


def within_block(kind: str, p: np.ndarray, v: np.ndarray, gj: np.ndarray, gi: np.ndarray,
                 min_pairs: int = 30) -> float:
    """Pair-weighted mean of `score` computed separately within each (gj, gi) block.

    Blocks with fewer than `min_pairs` pairs or an undefined score are skipped.
    Any block-constant predictor scores 0 here, so the result measures ranking
    information finer than the partition.
    """
    key = gj * (gi.max() + 1) + gi
    tot, wsum = 0.0, 0
    for b in np.unique(key):
        m = np.flatnonzero(key == b)
        if m.size < min_pairs:
            continue
        s = score(kind, p, v, m)
        if np.isfinite(s):
            tot += s * m.size
            wsum += m.size
    return tot / wsum if wsum else np.nan
