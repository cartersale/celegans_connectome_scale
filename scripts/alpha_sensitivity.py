"""Rebuild and rescore every alpha-dependent result at each value of the path
discount alpha.

The path operator is R = (I - alpha A / rho)^-1 - I; the main analyses use
alpha = 0.1. At each alpha in ALPHAS = (0.01, 0.03, 0.1, 0.3, 0.5, 0.7, 0.9)
(override with --alphas) everything below is recomputed from R at that alpha:

  full operator      AUC - 0.5 (all observed pairs, unsynapsed pairs), Spearman
                     rho with activity correlations and response amplitude
  cell classes       loss of the class-level operator at the same alpha as a
                     fraction of full performance, and the mean loss over
                     --n-rand size-matched random partitions (label permutations)
  operator-defined   loss-guided agglomeration (greedy_path with a Ward
                     shortlist of 30 down to k = 20, then exhaustive pairwise
                     merges minimizing structural loss over all ordered pairs),
                     Ward and k-means (--n-seed initializations, n_init=10) on
                     operator profiles log1p(1e3 [R, R^T] / max R); scored at
                     K_GRID = 20, 15, 12, 10, 9, 8, 7, 6, 5, 4. For k-means the
                     point loss is the mean over seeds and the interval is from
                     seed 0 only.
  boundary           the largest k from which loss in signal propagation (all
                     pairs) or activity correlations (ob.POWERED) is detectable,
                     i.e. the lower bound of the cluster-bootstrap interval
                     (--n-boot resamples over stimulated-neuron classes) is
                     above 0, at every coarser k scored; None if no such k
  agreement          ARI of each method's k = 9 partition with the same method
                     at alpha = 0.1, and with the canonical subsystems
                     (conres.subsystems)

The alpha-free two-step binary path count AUC is printed once for reference.

Produces the Supplementary Information section and table on sensitivity to alpha.

Inputs: conres.observables.load, conres.subsystems, complexity_curve
    (greedy_path, relabel).
Outputs: results/alpha_sensitivity_curves.csv (alpha, method, k, loss per
    observable with intervals for POWERED), results/alpha_sensitivity_summary.csv
    (per alpha: full performance, cell-class and random loss, boundary and
    k = 9 loss per method, ARIs).
Usage: python scripts/alpha_sensitivity.py [--n-boot 200] [--n-seed 3]
    [--n-rand 20] [--alphas ...]
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ari

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from conres import observables as ob, randi, subsystems as ss   # noqa: E402
from complexity_curve import greedy_path, relabel   # noqa: E402

ALPHAS = (0.01, 0.03, 0.1, 0.3, 0.5, 0.7, 0.9)
K_GRID = (20, 15, 12, 10, 9, 8, 7, 6, 5, 4)
POWERED = ob.POWERED


def loss_guided(A, alpha, k_min):
    n = A.shape[0]
    off = ~np.eye(n, dtype=bool)
    oj, oi = np.nonzero(off)
    R = randi.resolvent(A, alpha)
    rf = stats.rankdata(R[oj, oi])
    rf = (rf - rf.mean()) / np.linalg.norm(rf - rf.mean())

    def struct_all(lab):
        rq = stats.rankdata(randi.class_leverage(A, lab, alpha)[oj, oi])
        rq -= rq.mean()
        return float(1 - rq @ rf / np.linalg.norm(rq))

    path = greedy_path(A, 20, loss_fn=struct_all, shortlist=30)
    lab = relabel(path[20])
    while lab.max() + 1 > k_min:
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
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--n-seed", type=int, default=3)
    ap.add_argument("--n-rand", type=int, default=20)
    ap.add_argument("--alphas", type=float, nargs="*", default=ALPHAS)
    args = ap.parse_args()

    D = ob.load()
    A, nodes, cls, obs = D["A"], D["nodes"], D["cls"], D["obs"]
    obs = {o: obs[o] for o in ("incidence_all", "incidence_nosyn", "correlation", "amplitude")}
    boots = ob.boot_indices(obs, cls, args.n_boot)
    sub = ss.partition(nodes, "subsystems")[0]
    B = (A > 0).astype(float)
    two = B @ B
    j, i, y, _ = obs["incidence_all"]
    print(f"two-step path count (alpha-free): AUC {ob.score('auc', two[j, i], y) + 0.5:.3f}")

    rows, parts_at, summary = [], {}, []
    for alpha in args.alphas:
        rng = np.random.default_rng(0)
        R = randi.resolvent(A, alpha)
        full = {o: ob.score(kd, ob.predict(kd, R, j_, i_), v) for o, (j_, i_, v, kd) in obs.items()}
        fb = {o: np.array([ob.score(kd, ob.predict(kd, R, j_, i_), v, s) for s in boots[o]])
              for o, (j_, i_, v, kd) in obs.items()}

        def frac_loss(lab, with_ci=True):
            Q = randi.class_leverage(A, lab, alpha)
            out = {}
            for o, (j_, i_, v, kd) in obs.items():
                p = ob.predict(kd, Q, j_, i_)
                out[o] = (full[o] - ob.score(kd, p, v)) / full[o]
                if with_ci and o in POWERED:
                    lb = np.array([(f - ob.score(kd, p, v, s)) / f for f, s in zip(fb[o], boots[o])])
                    out[o + "_lo"], out[o + "_hi"] = np.nanpercentile(lb, [2.5, 97.5])
            return out

        cc = frac_loss(cls)
        rnd = [frac_loss(rng.permutation(cls), with_ci=False) for _ in range(args.n_rand)]
        s = {"alpha": alpha, **{f"full_{o}": full[o] for o in obs},
             **{f"classes_loss_{o}": cc[o] for o in obs},
             **{f"random_loss_{o}": np.mean([r[o] for r in rnd]) for o in obs}}
        print(f"\n== alpha {alpha}: full AUC all {full['incidence_all'] + 0.5:.3f}, unsynapsed "
              f"{full['incidence_nosyn'] + 0.5:.3f}, correlation rho {full['correlation']:.3f}, "
              f"amplitude rho {full['amplitude']:.3f}")
        print("   cell classes loss: " + ", ".join(f"{o} {cc[o]:+.2f}" for o in obs)
              + "   random: " + ", ".join(f"{o} {s[f'random_loss_{o}']:.2f}" for o in obs), flush=True)

        print("   building loss-guided path ...", flush=True)
        lg = loss_guided(A, alpha, min(K_GRID))
        Xop = np.log1p(np.hstack([R, R.T]) / R.max() * 1e3)
        tree = linkage(Xop, method="ward")
        builders = {
            "loss-guided": lambda k: [relabel(lg[k])],
            "Ward on operator": lambda k: [relabel(fcluster(tree, t=k, criterion="maxclust"))],
            "k-means on operator": lambda k: [relabel(KMeans(k, n_init=10, random_state=sd).fit(Xop).labels_)
                                              for sd in range(args.n_seed)],
        }
        for method, build in builders.items():
            for k in K_GRID:
                labs = build(k)
                per = [frac_loss(l, with_ci=(q == 0)) for q, l in enumerate(labs)]
                r = {"alpha": alpha, "method": method, "k": k}
                for o in obs:
                    r[o] = np.mean([p[o] for p in per])
                for o in POWERED:
                    r[o + "_lo"], r[o + "_hi"] = per[0][o + "_lo"], per[0][o + "_hi"]
                rows.append(r)
                if k == 9:
                    parts_at[(alpha, method)] = labs[0]
            d = pd.DataFrame([r for r in rows if r["alpha"] == alpha and r["method"] == method])
            det = (d.set_index("k")[[o + "_lo" for o in POWERED]].max(axis=1) > 0)
            ks = sorted(det.index, reverse=True)
            boundary = None
            for q, k in enumerate(ks):
                if all(det[kk] for kk in ks[q:]):
                    boundary = k
                    break
            l9 = d.set_index("k").loc[9]
            s[f"{method} boundary"] = boundary
            s[f"{method} loss9_all"] = l9["incidence_all"]
            s[f"{method} loss9_corr"] = l9["correlation"]
            print(f"   {method:>20}: sustained loss from k = {boundary}; loss at k = 9: signal "
                  f"propagation {l9['incidence_all']:+.2f} [{l9['incidence_all_lo']:+.2f}, "
                  f"{l9['incidence_all_hi']:+.2f}], correlation {l9['correlation']:+.2f} "
                  f"[{l9['correlation_lo']:+.2f}, {l9['correlation_hi']:+.2f}]", flush=True)
        summary.append(s)

    for s in summary:
        for method in ("loss-guided", "Ward on operator", "k-means on operator"):
            lab = parts_at[(s["alpha"], method)]
            ref = parts_at.get((0.1, method))
            s[f"{method} ARI9_vs_alpha0.1"] = ari(lab, ref) if ref is not None else np.nan
            s[f"{method} ARI9_vs_subsystems"] = ari(lab, sub)

    pd.DataFrame(rows).to_csv(ROOT / "results/alpha_sensitivity_curves.csv", index=False)
    S = pd.DataFrame(summary)
    S.to_csv(ROOT / "results/alpha_sensitivity_summary.csv", index=False)
    print("\n== summary")
    print(S.T.to_string(float_format=lambda x: f"{x:.3f}"))
    print("\nwrote results/alpha_sensitivity_{curves,summary}.csv")


if __name__ == "__main__":
    main()
