"""Simulated annealing on the structural loss at k = 119 (the number of worm
cell classes), started from biology, from the greedy paths and from random
partitions, to test whether the cell-class partition is near a structural optimum.

Loss is 1 - Spearman between the class-level path operator (read back to
neurons) and the full resolvent over all ordered pairs, alpha = 0.1. Two search
spaces:

  free      any cell sizes; a move relocates one neuron to another cell and may
            not empty a cell
  profile   biology's exact cell-size multiset; a move swaps two neurons in
            different cells, so which neurons are grouped is varied while the
            size profile is held fixed

Proposals are half uniform and half informed: the partner neuron is one of the
moving neuron's 15 nearest neighbours by cosine similarity of log1p(row, column)
connectivity profiles. Temperature falls geometrically from --t0 to --t1 over
--n-iter proposals, and the best state seen is kept. Chains per space: biology at
t0, biology at 5 t0, --n-random random starts at 5 t0, and (free space only) the
envelope and ward partitions at k = 119 from `complexity_curve.py`, at t0.

The search optimises the loss it is scored on, so a best partition below biology
is a certificate that biology is not optimal; failing to beat biology is bounded
by how consistently independent chains converge, hence the spread and pairwise
ARI across random-start chains are reported.

Before annealing, --n-local single legal moves from biology are sampled in each
space and the fraction that lower the loss is reported. Each final partition is
also scored by adjusted Rand index to biology, the fraction of co-cell pairs
that share a class, and its functional loss (AUC shortfall on unsynapsed occ3
pairs) with a stimulated-class cluster-bootstrap interval (500 resamples, seed 3).

Inputs:  Cook connectome and Randi atlas via conres; results/complexity_curve_paths.npz.
Outputs: results/global_search.csv  (one row per chain)
         results/global_search.npz  (best labels and trace per chain;
                                           names with ':' replaced by '__')
Usage:   python scripts/global_search.py [--n-iter 40000] [--n-random 4]
             [--t0 2e-3] [--t1 2e-5] [--workers 12] [--n-local 2000] [--seed 0]
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
from scipy import stats

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from conres import celegans as ce, randi          # noqa: E402

ALPHA = 0.1
_G = {}


def _setup():
    if _G:
        return _G
    A, nodes = ce.build_matrix("all", neurons_only=True)
    n = A.shape[0]
    off = ~np.eye(n, dtype=bool)
    oj, oi = np.nonzero(off)
    f = randi.resolvent(A, ALPHA)[oj, oi]
    rf = stats.rankdata(f)
    rf = rf - rf.mean()
    rf /= np.linalg.norm(rf)
    X = np.log1p(np.hstack([A, A.T]))
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    sim = X @ X.T
    np.fill_diagonal(sim, -np.inf)
    nbr = np.argsort(-sim, axis=1)[:, :15]
    bio = np.unique(ce.partitions(nodes)["class"], return_inverse=True)[1]
    _G.update(A=A, nodes=nodes, n=n, oj=oj, oi=oi, rf=rf, nbr=nbr, bio=bio)
    return _G


def loss(lab: np.ndarray) -> float:
    g = _setup()
    q = randi.class_leverage(g["A"], lab, ALPHA)[g["oj"], g["oi"]]
    rq = stats.rankdata(q)
    rq -= rq.mean()
    return float(1.0 - rq @ g["rf"] / np.linalg.norm(rq))


def propose(lab, sizes, space, rng):
    """Return a new label vector, or None if the draw was not a legal move."""
    g = _setup()
    n = g["n"]
    i = int(rng.integers(n))
    if rng.random() < 0.5:
        j = int(g["nbr"][i, rng.integers(g["nbr"].shape[1])])
    else:
        j = int(rng.integers(n))
    a, b = lab[i], lab[j]
    if a == b:
        return None
    new = lab.copy()
    if space == "free":
        if sizes[a] < 2:
            return None
        new[i] = b
    else:
        # swapping i and j leaves both cell sizes unchanged
        new[i], new[j] = b, a
    return new


def anneal(job):
    name, space, start, n_iter, t0, t1, seed = job
    _setup()
    rng = np.random.default_rng(seed)
    lab = start.copy()
    sizes = np.bincount(lab)
    cur = loss(lab)
    best, best_lab = cur, lab.copy()
    trace = [(0, cur)]
    acc = 0
    t_start = time.time()
    for it in range(1, n_iter + 1):
        T = t0 * (t1 / t0) ** (it / n_iter)
        new = propose(lab, sizes, space, rng)
        if new is None:
            continue
        l_ = loss(new)
        if l_ <= cur or rng.random() < np.exp((cur - l_) / T):
            if space == "free":
                sizes[lab[np.flatnonzero(new != lab)[0]]] -= 1
                sizes[new[np.flatnonzero(new != lab)[0]]] += 1
            lab, cur = new, l_
            acc += 1
            if cur < best:
                best, best_lab = cur, lab.copy()
        if it % 1000 == 0:
            trace.append((it, best))
    return {"name": name, "space": space, "seed": seed, "start_loss": trace[0][1],
            "best": best, "labels": best_lab, "trace": trace,
            "accept": acc / n_iter, "minutes": (time.time() - t_start) / 60}


def random_start(k, n, rng, profile=None):
    if profile is not None:
        return rng.permutation(profile)
    lab = np.concatenate([np.arange(k), rng.integers(0, k, n - k)])
    return rng.permutation(lab)


def local_check(bio, n_draws, seed):
    """Sample single legal moves from biology; how many lower the loss?"""
    rng = np.random.default_rng(seed)
    base = loss(bio)
    out = {}
    for space in ("free", "profile"):
        sizes = np.bincount(bio)
        deltas = []
        while len(deltas) < n_draws:
            new = propose(bio, sizes, space, rng)
            if new is not None:
                deltas.append(loss(new) - base)
        out[space] = np.array(deltas)
    return base, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-iter", type=int, default=40000)
    ap.add_argument("--n-random", type=int, default=4)
    ap.add_argument("--t0", type=float, default=2e-3)
    ap.add_argument("--t1", type=float, default=2e-5)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--n-local", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    g = _setup()
    n, bio = g["n"], g["bio"]
    k = int(bio.max() + 1)
    base, deltas = local_check(bio, args.n_local, args.seed)
    print(f"biological partition: k = {k}, structural loss {base:.4f}")
    for space, d in deltas.items():
        print(f"  local check ({space}): {np.mean(d < 0):.1%} of {d.size} sampled single "
              f"moves improve it; best {d.min():+.5f}, median {np.median(d):+.5f}", flush=True)

    paths = np.load(ROOT / "results/complexity_curve_paths.npz")
    rng = np.random.default_rng(args.seed + 1)
    jobs = []
    s = args.seed
    for space in ("free", "profile"):
        prof = bio if space == "profile" else None
        jobs.append((f"{space}:bio", space, bio, args.n_iter, args.t0, args.t1, s)); s += 1
        jobs.append((f"{space}:bio_hot", space, bio, args.n_iter, 5 * args.t0, args.t1, s)); s += 1
        if space == "free":
            for nm in ("envelope", "ward"):
                jobs.append((f"free:{nm}", space, paths[f"{nm}_{k}"], args.n_iter,
                             args.t0, args.t1, s)); s += 1
        for r in range(args.n_random):
            jobs.append((f"{space}:random{r}", space, random_start(k, n, rng, prof),
                         args.n_iter, 5 * args.t0, args.t1, s)); s += 1
    print(f"\n{len(jobs)} annealing chains, {args.n_iter:,} proposals each, "
          f"{args.workers} workers", flush=True)

    from multiprocessing import get_context
    with get_context("spawn").Pool(args.workers) as pool:
        res = []
        for r in pool.imap_unordered(anneal, jobs):
            res.append(r)
            print(f"  {r['name']:>20}: {r['start_loss']:.4f} -> {r['best']:.4f}  "
                  f"(accept {r['accept']:.1%}, {r['minutes']:.1f} min)", flush=True)
    res.sort(key=lambda r: r["name"])
    summarise(res, bio, base, g, args)


def summarise(res, bio, base, g, args):
    from sklearn.metrics import adjusted_rand_score as ari
    from sklearn.metrics import roc_auc_score
    nodes, A = g["nodes"], g["A"]
    n = g["n"]

    def same_class_rate(lab):
        same = tot = 0
        for c in range(lab.max() + 1):
            m = np.flatnonzero(lab == c)
            if m.size > 1:
                eq = bio[m][:, None] == bio[m][None, :]
                same += (eq.sum() - m.size) // 2
                tot += m.size * (m.size - 1) // 2
        return same / max(tot, 1)

    idx = {x: i for i, x in enumerate(nodes)}
    occ3, _ = randi.build_occ3(nodes, "wt")
    d = randi.load_responses("wildty")
    d = d[d.pre.isin(idx) & d.post.isin(idx)]
    hits = {(idx[p], idx[q]) for p, q in zip(d.pre, d.post)}
    ii, jj = np.nonzero(occ3)
    y = np.array([1.0 if (j, i) in hits else 0.0 for i, j in zip(ii, jj)])
    ns = A[jj, ii] == 0
    R = randi.resolvent(A, ALPHA)[jj, ii]
    auc0 = roc_auc_score(y[ns], R[ns])
    gcls = bio[jj][ns]
    uq = np.unique(gcls)
    where = {c: np.flatnonzero(gcls == c) for c in uq}

    def func(lab, n_boot=500):
        q = randi.class_leverage(A, lab, ALPHA)[jj, ii][ns]
        point = auc0 - roc_auc_score(y[ns], q)
        rng = np.random.default_rng(3)
        bs = []
        for _ in range(n_boot):
            s = np.concatenate([where[c] for c in rng.choice(uq, uq.size)])
            bs.append(roc_auc_score(y[ns][s], R[ns][s]) - roc_auc_score(y[ns][s], q[s]))
        return point, np.percentile(bs, 2.5), np.percentile(bs, 97.5)

    print(f"\n{'chain':>20} {'start':>7} {'best':>7} {'vs bio':>8} {'ARI to bio':>11} "
          f"{'same-class':>11} {'largest cell':>13} {'func no-syn [95% CI]':>28}")
    rows = []
    fb = func(bio)
    print(f"{'biology (reference)':>20} {base:>7.4f} {base:>7.4f} {0:>+8.4f} {1:>11.3f} "
          f"{same_class_rate(bio):>11.1%} {np.bincount(bio).max():>13d} "
          f"{fb[0]:>+9.4f} [{fb[1]:+.4f}, {fb[2]:+.4f}]")
    for r in res:
        lab = r["labels"]
        f_ = func(lab)
        row = {"chain": r["name"], "space": r["space"], "start": r["start_loss"],
               "best": r["best"], "vs_bio": r["best"] - base,
               "ari_bio": ari(bio, lab), "same_class": same_class_rate(lab),
               "largest_cell": int(np.bincount(lab).max()),
               "func_nosyn": f_[0], "func_lo": f_[1], "func_hi": f_[2],
               "accept": r["accept"]}
        rows.append(row)
        print(f"{r['name']:>20} {row['start']:>7.4f} {row['best']:>7.4f} "
              f"{row['vs_bio']:>+8.4f} {row['ari_bio']:>11.3f} {row['same_class']:>11.1%} "
              f"{row['largest_cell']:>13d} {f_[0]:>+9.4f} [{f_[1]:+.4f}, {f_[2]:+.4f}]")

    for space in ("free", "profile"):
        rr = [r for r in res if r["space"] == space and "random" in r["name"]]
        if len(rr) < 2:
            continue
        losses = np.array([r["best"] for r in rr])
        pair = [ari(a["labels"], b["labels"]) for i_, a in enumerate(rr) for b in rr[i_ + 1:]]
        print(f"\n{space}: random-start chains reach {losses.mean():.4f} +- "
              f"{losses.std(ddof=1):.4f} (range {losses.min():.4f}-{losses.max():.4f}); "
              f"biology {base:.4f}")
        print(f"  pairwise ARI between random-start optima: mean {np.mean(pair):.3f} "
              f"[{np.min(pair):.3f}, {np.max(pair):.3f}]")

    out = ROOT / "results/global_search.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    np.savez_compressed(ROOT / "results/global_search.npz",
                        **{r["name"].replace(":", "__"): r["labels"] for r in res},
                        **{r["name"].replace(":", "__") + "__trace": np.array(r["trace"])
                           for r in res})
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
