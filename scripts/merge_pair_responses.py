"""Test whether neuron pairs that connectivity marks as interchangeable evoke
similar brain-wide responses when stimulated, beyond degree, soma position and
generic wiring similarity.

Interchangeability of a pair {a, b} is the negated single-merge equitability
defect of prospective_merges.py (mean of the out- and in-direction defects of
the all-synapse matrix A), so it uses connectivity only. Response similarity is
Cohen's kappa on detected responses, computed as in kappa_interchangeability.py:
over the neurons (other than a, b) recorded during stimulation of both
(occ3 > 0 for both), at least --min-shared of them.

Universe: every unordered pair of stimulated neurons meeting --min-shared, both
members at or above --min-degree (binary in+out degree), and both members
evoking at least m detected responses over the shared set, for each m in
--min-resp. Pairs where neither member evokes a response have undefined kappa
and are dropped.

Covariate sets, entered cumulatively as ranks (partial Spearman):
  estimability  log1p response counts of the two members (sum and |difference|),
                number of shared responders, log mean observation count. Kappa
                rises with response counts for any pair, so every model has these.
  degree        log binary degree, sum and |difference|
  space         Euclidean soma distance (conres.worm.soma_xyz, micrometres)
  wiring        Pearson r of the two members' log1p(out-row, in-column)
                profiles with entries for a and b removed (generic connectivity
                similarity), and log1p of direct synapses between a and b: when
                a drives b, stimulating a also evokes b's responders.
  class         same-cell-class indicator
The "wiring" model asks whether the defect ranking carries information that
profile correlation does not. The "class" model and the cross-class subset ask
whether the association is more than the known within-class agreement. The
same-class indicator under the estimability set is printed as a positive
control: it is the within-class agreement of kappa_interchangeability.py.

Matched comparison: "predicted" pairs are the lowest --top fraction of defect in
the universe. Within strata (quintiles; terciles for the joint stratification)
of estimability crossed with each control, the mean kappa of predicted minus
other pairs, averaged over strata weighted by predicted count.

The --n-merge disjoint pairs selected greedily at the same degree floor (the
prospective_merges.py selection) are reported as an illustrative subset.

Intervals: two-way neuron bootstrap for dyadic data. Neurons are resampled with
replacement and each pair weighted by the product of its two members'
multiplicities (--n-boot draws, seed --seed). Ranks are fixed at the full
sample; the bootstrap reweights a fixed-rank weighted regression.

Inputs: C. elegans connectome (conres.celegans), Randi 2023 wild-type responses
and observation counts (conres.randi), soma positions (conres.worm).
Outputs: results/merge_pair_responses.csv (one row per pair in the m = 0
universe); the summary goes to stdout.
Usage: merge_pair_responses.py [--min-degree 0] [--min-shared 20]
       [--min-resp 0,3] [--top 0.05] [--n-merge 60] [--n-boot 1000] [--seed 0]
"""
from __future__ import annotations

import argparse
import csv
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy import stats

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from conres import celegans as ce, randi, worm                    # noqa: E402
from prospective_merges import pairwise_defect, greedy_disjoint   # noqa: E402

COVARIATE_SETS = [
    ("none", []),
    ("estimability", ["resp_sum", "resp_diff", "n_shared", "log_expo"]),
    ("+ degree", ["deg_sum", "deg_diff"]),
    ("+ space", ["soma_dist"]),
    ("+ wiring", ["profile_r", "direct"]),
    ("+ class", ["same_class"]),
]


def kappa(d1, d2):
    po = (d1 == d2).mean()
    p1, p2 = d1.mean(), d2.mean()
    pe = p1 * p2 + (1 - p1) * (1 - p2)
    return (po - pe) / (1 - pe) if pe < 1 else np.nan


def profile_r(L, a, b):
    """Correlation of the concatenated log1p out/in profiles of a and b, with
    the entries for a and b removed so self- and mutual synapses do not count."""
    keep = np.ones(L.shape[0], bool)
    keep[[a, b]] = False
    x = np.concatenate([L[a, keep], L[keep, a]])
    y = np.concatenate([L[b, keep], L[keep, b]])
    if x.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def wls_resid(y, X, w):
    Xw = X * w[:, None]
    beta = np.linalg.lstsq(Xw.T @ X, Xw.T @ y, rcond=None)[0]
    return y - X @ beta


def wcorr(x, y, w):
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    cov = np.average((x - mx) * (y - my), weights=w)
    return cov / np.sqrt(np.average((x - mx) ** 2, weights=w)
                         * np.average((y - my) ** 2, weights=w))


def partial(x, y, Z, w):
    """Weighted correlation of x and y after weighted regression of both on Z."""
    X = np.column_stack([np.ones(len(x))] + ([Z] if Z.size else []))
    return wcorr(wls_resid(x, X, w), wls_resid(y, X, w), w)


def matched_diff(k, pred, strata, w):
    """Mean over strata (weighted by predicted count) of mean kappa of predicted
    minus other pairs; strata with no pair on one side are skipped."""
    num = den = 0.0
    for s in np.unique(strata):
        m = strata == s
        wp, wo = w[m & pred], w[m & ~pred]
        if wp.sum() == 0 or wo.sum() == 0:
            continue
        diff = np.average(k[m & pred], weights=wp) - np.average(k[m & ~pred], weights=wo)
        num += wp.sum() * diff
        den += wp.sum()
    return num / den if den else np.nan


def qbin(x, q):
    edges = np.quantile(x, np.linspace(0, 1, q + 1)[1:-1])
    return np.searchsorted(edges, x, side="right")


def boot_ci(stat, pa, pb, n_nodes, n_boot, seed):
    """Point estimate and 95% interval of stat(w) under the two-way neuron bootstrap."""
    rng = np.random.default_rng(seed)
    point = stat(np.ones(len(pa)))
    draws = []
    for _ in range(n_boot):
        mult = rng.multinomial(n_nodes, np.full(n_nodes, 1 / n_nodes))
        w = (mult[pa] * mult[pb]).astype(float)
        if w.sum() == 0:
            continue
        draws.append(stat(w))
    draws = np.array(draws)
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return point, lo, hi, float(np.mean(draws <= 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-degree", type=int, default=0)
    ap.add_argument("--min-shared", type=int, default=20)
    ap.add_argument("--min-resp", default="0,3")
    ap.add_argument("--top", type=float, default=0.05)
    ap.add_argument("--n-merge", type=int, default=60)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    A, nodes = ce.build_matrix("all", neurons_only=True)
    mapping = ce.assign_classes(nodes)
    n = len(nodes)
    idx = {x: i for i, x in enumerate(nodes)}
    occ3, _ = randi.build_occ3(nodes, "wt")
    d = randi.load_responses("wildty")
    d = d[d.pre.isin(idx) & d.post.isin(idx)]
    det = np.zeros_like(occ3, dtype=bool)
    for p, q in zip(d.pre, d.post):
        det[idx[q], idx[p]] = True                  # [responder, stimulated]

    deg = (A > 0).sum(1) + (A > 0).sum(0)
    g = (pairwise_defect(A) + pairwise_defect(A.T)) / 2.0
    xyz = worm.soma_xyz(nodes)
    L = np.log1p(A)
    stim = (occ3.sum(0) > 0) & (deg >= args.min_degree)

    rows = []
    cand = np.flatnonzero(stim)
    for ia, a in enumerate(cand):
        for b in cand[ia + 1:]:
            S = (occ3[:, a] > 0) & (occ3[:, b] > 0)
            S[[a, b]] = False
            if S.sum() < args.min_shared:
                continue
            da, db = det[S, a], det[S, b]
            ra, rb = int(da.sum()), int(db.sum())
            rows.append(dict(
                a=nodes[a], b=nodes[b], ia=a, ib=b, defect=g[a, b], kappa=kappa(da, db),
                resp_a=ra, resp_b=rb,
                resp_sum=np.log1p(ra) + np.log1p(rb),
                resp_diff=abs(np.log1p(ra) - np.log1p(rb)),
                n_shared=int(S.sum()),
                log_expo=0.5 * np.log(occ3[S, a].mean() * occ3[S, b].mean()),
                deg_sum=np.log(deg[a]) + np.log(deg[b]),
                deg_diff=abs(np.log(deg[a]) - np.log(deg[b])),
                soma_dist=float(np.linalg.norm(xyz[a] - xyz[b])),
                profile_r=profile_r(L, a, b),
                direct=np.log1p(A[a, b] + A[b, a]),
                same_class=float(mapping[nodes[a]] == mapping[nodes[b]])))

    sel = greedy_disjoint(g, args.n_merge, deg >= args.min_degree)
    sel_set = {frozenset(p) for p in sel}
    for r in rows:
        r["selected"] = int(frozenset((r["ia"], r["ib"])) in sel_set)

    with (ROOT / "results/merge_pair_responses.csv").open("w", newline="") as fh:
        cols = [c for c in rows[0] if c not in ("ia", "ib")]
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    n_nan_soma = int(np.isnan(xyz[cand]).any(1).sum())
    print(f"{n} neurons, {stim.sum()} stimulated at degree floor {args.min_degree}; "
          f"{len(rows)} pairs with >= {args.min_shared} shared observed responders; "
          f"{n_nan_soma} candidate neurons without soma position")
    print(f"two-way neuron bootstrap, {args.n_boot} draws; "
          f"p = fraction of draws <= 0\n")

    for m in [int(x) for x in args.min_resp.split(",")]:
        R = [r for r in rows if min(r["resp_a"], r["resp_b"]) >= m
             and np.isfinite(r["kappa"]) and np.isfinite(r["soma_dist"])]
        if len(R) < 50:
            print(f"min responses {m}: {len(R)} pairs, too few\n")
            continue
        col = lambda k: np.array([r[k] for r in R], float)       # noqa: E731
        pa, pb = col("ia").astype(int), col("ib").astype(int)
        k = col("kappa")
        rk = lambda v: stats.rankdata(v)                          # noqa: E731
        y, x = rk(k), rk(-col("defect"))
        same = col("same_class").astype(bool)
        n_nodes = n

        print(f"=== both members evoke >= {m} responses: {len(R)} pairs, "
              f"{len(np.unique(np.r_[pa, pb]))} neurons, {int(same.sum())} same-class; "
              f"mean kappa {k.mean():.3f}")
        print(f"{'partial Spearman, interchangeability vs kappa':<48}"
              f"{'rho':>8} {'95% CI':>18} {'p':>7}")
        used = []
        for name, add in COVARIATE_SETS:
            used = used + add
            Z = np.column_stack([rk(col(c)) for c in used]) if used else np.empty((len(R), 0))
            est = boot_ci(lambda w, Z=Z: partial(x, y, Z, w), pa, pb, n_nodes,
                          args.n_boot, args.seed)
            print(f"  {name:<46}{est[0]:>8.3f}  [{est[1]:>+.3f}, {est[2]:>+.3f}] {est[3]:>7.3f}")

        Zs = np.column_stack([rk(col(c)) for c in
                              COVARIATE_SETS[1][1] + COVARIATE_SETS[2][1] + COVARIATE_SETS[3][1]])
        est = boot_ci(lambda w: partial(rk(col("profile_r")), y, Zs, w), pa, pb, n_nodes,
                      args.n_boot, args.seed)
        print(f"  {'(reference) profile r, + space set':<46}{est[0]:>8.3f}  "
              f"[{est[1]:>+.3f}, {est[2]:>+.3f}] {est[3]:>7.3f}")

        Ze = np.column_stack([rk(col(c)) for c in COVARIATE_SETS[1][1]])
        est = boot_ci(lambda w: partial(rk(col("same_class")), y, Ze, w), pa, pb, n_nodes,
                      args.n_boot, args.seed)
        print(f"  {'(positive control) same class, estimability':<46}{est[0]:>8.3f}  "
              f"[{est[1]:>+.3f}, {est[2]:>+.3f}] {est[3]:>7.3f}")

        xc = ~same
        Zw = np.column_stack([rk(col(c)[xc]) for c in sum((c for _, c in COVARIATE_SETS[:5]), [])])
        est = boot_ci(lambda w: partial(rk(-col("defect")[xc]), rk(k[xc]), Zw, w),
                      pa[xc], pb[xc], n_nodes, args.n_boot, args.seed)
        print(f"  {'cross-class pairs only, + wiring set':<46}{est[0]:>8.3f}  "
              f"[{est[1]:>+.3f}, {est[2]:>+.3f}] {est[3]:>7.3f}  (n = {xc.sum()})")

        pred = col("defect") <= np.quantile(col("defect"), args.top)
        e5 = qbin(col("resp_sum"), 5)
        strata = {
            "random (estimability only)": e5,
            "degree-matched": e5 * 25 + qbin(col("deg_sum"), 5) * 5 + qbin(col("deg_diff"), 5),
            "spatially matched": e5 * 5 + qbin(col("soma_dist"), 5),
            "wiring-similarity matched": e5 * 5 + qbin(col("profile_r"), 5),
            "all three, terciles": (qbin(col("resp_sum"), 3) * 81 + qbin(col("deg_sum"), 3) * 27
                                    + qbin(col("deg_diff"), 3) * 9
                                    + qbin(col("soma_dist"), 3) * 3 + qbin(col("profile_r"), 3)),
        }
        print(f"\n{'predicted (lowest ' + format(args.top, '.0%') + ' defect) minus matched, kappa':<48}"
              f"{'diff':>8} {'95% CI':>18} {'p':>7}")
        print(f"  ({pred.sum()} predicted pairs, {int((pred & same).sum())} same-class)")
        for name, s in strata.items():
            est = boot_ci(lambda w, s=s: matched_diff(k, pred, s, w), pa, pb, n_nodes,
                          args.n_boot, args.seed)
            print(f"  {name:<46}{est[0]:>8.3f}  [{est[1]:>+.3f}, {est[2]:>+.3f}] {est[3]:>7.3f}")
        s = strata["all three, terciles"]
        est = boot_ci(lambda w: matched_diff(k[xc], pred[xc], s[xc], w),
                      pa[xc], pb[xc], n_nodes, args.n_boot, args.seed)
        print(f"  {'all three, cross-class pairs only':<46}{est[0]:>8.3f}  "
              f"[{est[1]:>+.3f}, {est[2]:>+.3f}] {est[3]:>7.3f}")

        sl = col("selected").astype(bool)
        print(f"\n  disjoint {args.n_merge} selected merges in this universe: {sl.sum()} pairs, "
              f"mean kappa {k[sl].mean() if sl.any() else np.nan:.3f} "
              f"(universe {k.mean():.3f}; same-class {int((sl & same).sum())})\n")


if __name__ == "__main__":
    main()
