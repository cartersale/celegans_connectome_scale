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

`pair_sets` builds the incidence and correlation pair sets, and `build_sta` the
stimulus-triggered averages behind amplitude and latency (cached in
results/sta_cache.npz).
"""
from __future__ import annotations

import re

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

from conres import celegans as ce, randi
from conres.paths import REPO_ROOT

OBSERVABLES = ("incidence_all", "incidence_nosyn", "correlation", "amplitude", "magnitude")
POWERED = ("incidence_all", "correlation")    # narrowest intervals; they resolve boundaries

# stimulus-triggered averages: volumes kept before and after onset (2 Hz), and
# the response window [WIN[0], WIN[1]) in volumes after onset
PRE, POST, WIN = 10, 60, (2, 40)
STA_CACHE = REPO_ROOT / "results/sta_cache.npz"


def pair_sets(min_animals: int, excise: int, detrend: bool) -> dict:
    """Incidence and correlation pair sets on the Cook node set (Randi 2023 wild type).

    Incidence: ordered pairs (ii responds, jj stimulated) with occ3 > 0, label
    `y` = 1 for a detected response, `direct` = A[jj, ii]. Correlation:
    unordered pairs (iu, ju) co-identified in >= `min_animals` recordings, with
    mean within-recording Pearson `r` and recording count `n_animals`; `excise`
    and `detrend` are passed to randi.build_correlations. `common` and
    `dcommon` mark, in each set, the pairs present in both.
    """
    A, nodes = ce.build_matrix("all", neurons_only=True)
    labels = ce.partitions(nodes)["class"]
    n = len(nodes)

    occ3, _ = randi.build_occ3(nodes, "wt")
    idx = {m: i for i, m in enumerate(nodes)}
    d = randi.load_responses("wildty")
    d = d[d.pre.isin(idx) & d.post.isin(idx)]
    hits = {(idx[p], idx[q]) for p, q in zip(d.pre, d.post)}
    ii, jj = np.nonzero(occ3)                       # i responds, j stimulated
    y = np.array([1.0 if (j, i) in hits else 0.0 for i, j in zip(ii, jj)])

    R, cnt = randi.build_correlations(nodes, "wt", excise=excise, detrend=detrend)
    iu, ju = np.triu_indices(n, 1)
    keep = (cnt[iu, ju] >= min_animals) & np.isfinite(R[iu, ju])
    iu, ju = iu[keep], ju[keep]
    r = R[iu, ju]

    # unordered pairs present in both observables, indexed both ways
    und = {(min(a, b), max(a, b)) for a, b in zip(ii, jj)}
    common = np.array([(a, b) in und for a, b in zip(iu, ju)])
    cset = {(a, b) for a, b, k in zip(iu, ju, common) if k}
    dcommon = np.array([(min(a, b), max(a, b)) in cset for a, b in zip(ii, jj)])
    return dict(A=A, nodes=nodes, labels=labels, ii=ii, jj=jj, y=y,
                iu=iu, ju=ju, r=r, common=common, dcommon=dcommon,
                direct=A[jj, ii], n_animals=cnt[iu, ju])


def build_sta(nodes: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Per ordered pair (i responds, j stimulated): summed z-trace and event count.

    For each stimulation the responder's trace from PRE volumes before to POST
    after onset is baseline-subtracted by its pre-stimulus mean and divided by
    that neuron's whole-recording SD. Recordings shorter than 100 volumes,
    traces with any non-finite sample and events without a full window are
    skipped. Returns (acc, cnt): acc[i, j] is the summed trace (length
    PRE + POST + 1) and cnt[i, j] the number of events. The result is cached
    in STA_CACHE and reused when the node list matches.
    """
    if STA_CACHE.exists():
        z = np.load(STA_CACHE)
        if list(z["nodes"]) == list(nodes):
            return z["acc"], z["cnt"]
    folder = randi.EXPORTED / "exported_data"
    valid = {x: k for k, x in enumerate(nodes)}
    n = len(nodes)
    L = PRE + POST + 1
    acc = np.zeros((n, n, L), np.float32)
    cnt = np.zeros((n, n))
    for f in sorted(folder.glob("*_labels.txt")):
        r = int(re.match(r"(\d+)_labels", f.name).group(1))
        labels = [ce.normalise_name(x.rstrip("\n").strip()) for x in f.open()]
        sv, sn = folder / f"{r}_stim_volume_i.txt", folder / f"{r}_stim_neurons.txt"
        if not (sv.exists() and sn.exists()):
            continue
        G = randi._gcamp(folder, r)
        if G.shape[0] < 100:
            continue
        alive = np.isfinite(G).all(axis=0)
        cols = [(c, valid[x]) for c, x in enumerate(labels)
                if x in valid and c < G.shape[1] and alive[c]]
        if len(cols) < 2:
            continue
        ci = np.array([c for c, _ in cols])
        ni = np.array([k for _, k in cols])
        X = G[:, ci]
        sd = np.maximum(X.std(0), 1e-9)
        stims = [int(x) for x in sv.open() if x.strip()]
        sneu = [int(x) for x in sn.open() if x.strip()]
        for s, c in zip(stims, sneu):
            if not (0 <= c < len(labels)) or labels[c] not in valid:
                continue
            if s - PRE < 0 or s + POST >= X.shape[0]:
                continue
            j = valid[labels[c]]
            seg = X[s - PRE:s + POST + 1]
            z = (seg - seg[:PRE].mean(0)) / sd
            keep = ni != j
            acc[ni[keep], j] += z[:, keep].T
            cnt[ni[keep], j] += 1
    np.savez_compressed(STA_CACHE, acc=acc, cnt=cnt, nodes=np.array(nodes))
    return acc, cnt

def load(min_animals: int = 3, min_events: int = 2) -> dict:
    """Dict with Cook adjacency `A`, `nodes`, class codes `cls` and observables `obs`."""
    dat = pair_sets(min_animals, 0, False)
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
