"""Continue the loss-greedy path below k = 20 to --k-min and score structural
and functional loss, with paired bootstrap intervals and tie diagnostics, at
every k.

The path starts from envelope_20 in results/complexity_curve_paths.npz, so every
partition here is nested in that path. Below 20 there are at most 190
candidate merges per step, so every one is scored by structural loss (1 -
Spearman over all ordered pairs, alpha = 0.1) and the best is taken; no Ward
shortlist is used.

At every k:
  structural     1 - Spearman over all ordered pairs and over occ3-observed pairs;
                 random reference from the first 10 permutations
  functional     incidence (unsynapsed occ3 pairs, and all occ3 pairs; AUC - 0.5),
                 activity correlation, response amplitude (pairs with >= 2
                 events) and response magnitude (Spearman), each as loss
                 = (s_full - s) / s_full with a paired stimulated-class cluster
                 bootstrap (--n-boot) and the mean over --n-rand size-matched
                 permutations
  ties           with k classes the read-back takes at most k^2 distinct values,
                 so the largest tie block and distinct-value count of each
                 predictor are recorded. A dominant tie block degrades rank
                 correlations; AUC gives ties half credit and degrades more
                 gracefully, but the distinct-value count still bounds resolution.

Loss is detectable where the lower bound of its paired interval exceeds 0.

Inputs:  results/complexity_curve_paths.npz; observables as in
         `observable_resolution.py` (imports build_sta, setup_corr, PRE, WIN).
Outputs: results/coarse_trajectory_paths.npz  (envelope_<k> labels, k = 20..k_min)
         results/coarse_trajectory.csv
Usage:   python scripts/coarse_trajectory.py [--k-min 2] [--n-rand 50] [--n-boot 500] [--seed 0]
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from conres import randi                                            # noqa: E402
from conres.metric_guard import guard_predictor                     # noqa: E402
from observable_resolution import build_sta, setup_corr, PRE, WIN  # noqa: E402

ALPHA = 0.1


def relabel(lab):
    return np.unique(lab, return_inverse=True)[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k-min", type=int, default=2)
    ap.add_argument("--n-rand", type=int, default=50)
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dat = setup_corr(3, 0, False)
    A, nodes = dat["A"], dat["nodes"]
    n = A.shape[0]
    idx = {x: i for i, x in enumerate(nodes)}
    cls = relabel(dat["labels"])
    ii, jj, y = dat["ii"], dat["jj"], dat["y"]
    ns = A[jj, ii] == 0
    off = ~np.eye(n, dtype=bool)
    oj, oi = np.nonzero(off)

    acc, cnt = build_sta(nodes)
    sta = acc / np.maximum(cnt, 1)[..., None]
    amp = np.abs(sta[:, :, PRE + WIN[0]:PRE + WIN[1]].mean(-1))
    d = randi.load_responses("wildty")
    d = d[d.pre.isin(idx) & d.post.isin(idx)]
    rj, ri = d.pre.map(idx).to_numpy(), d.post.map(idx).to_numpy()
    ev = cnt[ii, jj] >= 2

    obs = {
        "incidence_nosyn": (jj[ns], ii[ns], y[ns], "auc"),
        "incidence_all": (jj, ii, y, "auc"),
        "correlation": (dat["iu"], dat["ju"], dat["r"], "corr"),
        "amplitude": (jj[ev], ii[ev], amp[ii[ev], jj[ev]], "rho"),
        "magnitude": (rj, ri, d["mag"].to_numpy(float), "rho"),
    }

    def pred(kind, S, j, i):
        return S[j, i] + S[i, j] if kind == "corr" else S[j, i]

    def score(kind, p, v, s=slice(None)):
        if kind == "auc":
            return roc_auc_score(v[s], p[s]) - 0.5
        return stats.spearmanr(p[s], v[s]).statistic

    R = randi.resolvent(A, ALPHA)
    f_all, f_occ = R[oj, oi], R[jj, ii]
    rf = stats.rankdata(f_all)
    rf = (rf - rf.mean()) / np.linalg.norm(rf - rf.mean())

    def lifted(lab):
        return randi.class_leverage(A, lab, ALPHA)

    def struct_all(lab):
        rq = stats.rankdata(lifted(lab)[oj, oi])
        rq -= rq.mean()
        return float(1 - rq @ rf / np.linalg.norm(rq))

    full = {o: score(k, pred(k, R, j, i), v) for o, (j, i, v, k) in obs.items()}
    boots = {}
    rng = np.random.default_rng(args.seed)
    for o, (j, i, v, k) in obs.items():
        g = cls[j]
        where = {c: np.flatnonzero(g == c) for c in np.unique(g)}
        uq = np.array(list(where))
        boots[o] = [np.concatenate([where[c] for c in rng.choice(uq, uq.size)])
                    for _ in range(args.n_boot)]
    full_boot = {o: np.array([score(k, pred(k, R, j, i), v, s) for s in boots[o]])
                 for o, (j, i, v, k) in obs.items()}
    print("full operator: " + "  ".join(f"{o} {full[o]:.4f}" for o in obs))

    # -- continue the envelope path from k = 20, scoring every merge
    paths = np.load(ROOT / "results/complexity_curve_paths.npz")
    lab = relabel(paths["envelope_20"])
    path = {20: lab.copy()}
    while lab.max() + 1 > args.k_min:
        k = lab.max() + 1
        best, best_l = None, np.inf
        for a in range(k):
            for b in range(a + 1, k):
                cand = relabel(np.where(lab == b, a, lab))
                l_ = struct_all(cand)
                if l_ < best_l:
                    best, best_l = cand, l_
        lab = best
        path[int(lab.max() + 1)] = lab.copy()
        print(f"  k = {lab.max() + 1:2d}  structural loss {best_l:.4f}", flush=True)
    np.savez_compressed(ROOT / "results/coarse_trajectory_paths.npz",
                        **{f"envelope_{k}": v for k, v in path.items()})

    rows = []
    for k in sorted(path, reverse=True):
        lab = path[k]
        S = lifted(lab)
        sizes = np.bincount(lab)
        row = {"k": k, "largest_cell": int(sizes.max()), "singletons": int((sizes == 1).sum()),
               "struct_all": struct_all(lab),
               "struct_occ": 1 - stats.spearmanr(S[jj, ii], f_occ).statistic}
        rrng = np.random.default_rng(1000 + k)
        nulls = [lifted(rrng.permutation(lab)) for _ in range(args.n_rand)]
        row["struct_all_random"] = np.mean([
            1 - stats.spearmanr(Sn[oj, oi], f_all).statistic for Sn in nulls[:10]])
        for o, (j, i, v, kind) in obs.items():
            p = pred(kind, S, j, i)
            sc = score(kind, p, v)
            loss = (full[o] - sc) / full[o]
            lb = np.array([(fb - score(kind, p, v, s)) / fb
                           for fb, s in zip(full_boot[o], boots[o])])
            g = guard_predictor(v if kind == "auc" else np.zeros(len(v)), p)
            rl = [(full[o] - score(kind, pred(kind, Sn, j, i), v)) / full[o] for Sn in nulls]
            row.update({f"{o}_score": sc, f"{o}_loss": loss,
                        f"{o}_lo": np.nanpercentile(lb, 2.5),
                        f"{o}_hi": np.nanpercentile(lb, 97.5),
                        f"{o}_ties": g["largest_tie_block"],
                        f"{o}_distinct": g["distinct_values"],
                        f"{o}_random": float(np.mean(rl))})
        rows.append(row)
        print(f"  scored k = {k}", end="\r", flush=True)
    print()
    df = pd.DataFrame(rows)
    out = ROOT / "results/coarse_trajectory.csv"
    df.to_csv(out, index=False)

    print(f"\n{'k':>3} {'N/k':>6} {'largest':>8} {'struct':>7} {'rand':>6} | "
          + " | ".join(f"{o:^30}" for o in obs))
    print(f"{'':>3} {'':>6} {'cell':>8} {'(all)':>7} {'':>6} | "
          + " | ".join(f"{'loss [95% CI]  rand  ties':^30}" for _ in obs))
    for _, r in df.iterrows():
        cells = []
        for o in obs:
            star = "*" if r[f"{o}_lo"] > 0 else " "
            cells.append(f"{r[f'{o}_loss']:+.2f}[{r[f'{o}_lo']:+.2f},{r[f'{o}_hi']:+.2f}]{star}"
                         f" {r[f'{o}_random']:.2f} {r[f'{o}_ties']:.2f}")
        print(f"{int(r.k):>3} {n / r.k:>6.1f} {int(r.largest_cell):>8} {r.struct_all:>7.3f} "
              f"{r.struct_all_random:>6.3f} | " + " | ".join(f"{c:^30}" for c in cells))

    print("\nfirst k (coarsening from 20) with detectable functional loss")
    for o in obs:
        hit = df[df[f"{o}_lo"] > 0].sort_values("k", ascending=False)
        print(f"  {o:>16}: {int(hit.k.iloc[0]) if len(hit) else 'none down to ' + str(args.k_min)}"
              f"   (largest tie block at k = {args.k_min}: "
              f"{df[df.k == df.k.min()][f'{o}_ties'].iloc[0]:.2f})")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
