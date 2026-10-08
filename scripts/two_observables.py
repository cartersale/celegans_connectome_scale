"""Score the full path operator, the cell-class quotient and size-matched random
partitions on two observables: evoked response incidence and activity
correlation.

Observables (Randi 2023 wild type):
  incidence     ordered (stimulated j, responder i) pairs with occ3 > 0; label
                1 if a response was detected. Scored by AUC of S[j, i].
  correlation   mean within-recording Pearson r per unordered pair, kept if
                co-identified in >= --min-animals recordings. Scored by Spearman
                of S[i, j] + S[j, i] against r.
The correlations are computed on the same optogenetic recordings as the
responses (the release contains no unstimulated recordings), so the two
observables share frames. --excise drops that many volumes (2 Hz) after each
stimulation onset and --detrend detrends traces (see
conres.randi.build_correlations).

Operators: the full resolvent R (alpha = 0.1), the class-level resolvent lifted
to neurons (conres.randi.class_leverage), and --n-null random partitions with
the cell-class size profile. Results are reported on each observable's native
pair set and on the unordered pairs present in both. Losses are expressed as a
fraction of headroom, (full - arm) / (full - 0.5) for AUC and (full - arm) / full
for Spearman, because the two metrics are on different scales; the class
quotient is also placed in its null as z and the fraction of nulls >= it.

Inputs: C. elegans connectome (conres.celegans), Randi 2023 responses,
observation counts and traces (conres.randi).
Outputs: results/two_observables_<tag>.csv; tag defaults to
excise<N>[_detrend]. Rows are (scope, arm, AUC, rho, frac_inc, frac_corr) for
arms bio/random, and (scope, "z", z_inc, z_corr, p_inc, p_corr) for the z row.
Produces the stimulus-excision and detrending table in the Supplementary
Information (together with sham_excision.py).
Usage: two_observables.py [--min-animals 3] [--excise 0] [--detrend]
       [--n-null 200] [--seed 0] [--tag TAG]
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from conres import celegans as ce, randi          # noqa: E402
from conres.metric_guard import guard_predictor    # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ALPHA = 0.1


def size_matched(labels: np.ndarray, rng) -> np.ndarray:
    """A random partition with exactly the cell-size profile of `labels`."""
    sizes = np.bincount(labels)
    perm = rng.permutation(labels.size)
    out = np.empty(labels.size, int)
    p = 0
    for c, s in enumerate(sizes):
        out[perm[p:p + s]] = c
        p += s
    return out


def setup(min_animals: int, excise: int, detrend: bool):
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


def scores(S, dat, mask_d=None, mask_c=None):
    """(AUC on incidence, Spearman on correlation) for one operator S."""
    v = S[dat["jj"], dat["ii"]]                      # directed: j -> i
    y = dat["y"]
    if mask_d is not None:
        v, y = v[mask_d], y[mask_d]
    auc = roc_auc_score(y, v)
    w = S[dat["iu"], dat["ju"]] + S[dat["ju"], dat["iu"]]   # symmetrised
    r = dat["r"]
    if mask_c is not None:
        w, r = w[mask_c], r[mask_c]
    rho = stats.spearmanr(w, r).statistic
    return auc, rho


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-animals", type=int, default=3)
    ap.add_argument("--excise", type=int, default=0,
                    help="volumes dropped after each stimulus (2 Hz; 20 = 10 s)")
    ap.add_argument("--detrend", action="store_true")
    ap.add_argument("--n-null", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    dat = setup(args.min_animals, args.excise, args.detrend)
    A, labels = dat["A"], dat["labels"]
    k = np.unique(labels).size
    print(f"neurons {A.shape[0]}   biological cells {k}")
    print(f"incidence: {len(dat['y']):,} ordered pairs, {int(dat['y'].sum()):,} propagating "
          f"({dat['y'].mean():.1%})")
    print(f"correlation: {len(dat['r']):,} unordered pairs at >= {args.min_animals} animals "
          f"(median {int(np.median(dat['n_animals']))}), mean r {dat['r'].mean():+.3f}")
    print(f"shared pairs: {int(dat['common'].sum()):,} unordered / "
          f"{int(dat['dcommon'].sum()):,} ordered")
    if args.excise:
        print(f"stimulus excision: {args.excise} volumes ({args.excise / 2:.0f} s) after each stim")

    Sfull = randi.resolvent(A, ALPHA)
    g = guard_predictor(dat["y"], Sfull[dat["jj"], dat["ii"]])
    print(f"guard (incidence predictor): base {g['base_rate']:.3f}, "
          f"largest tie block {g['largest_tie_block']:.3f} -- {g['flags'][0][:60]}")

    rows = []
    for scope, md, mc in (("native", None, None),
                          ("shared", dat["dcommon"], dat["common"])):
        af, rf = scores(Sfull, dat, md, mc)
        Sbio = randi.class_leverage(A, labels, ALPHA)
        ab, rb = scores(Sbio, dat, md, mc)
        null = []
        rng = np.random.Generator(np.random.PCG64(args.seed))
        for _ in range(args.n_null):
            Sr = randi.class_leverage(A, size_matched(labels, rng), ALPHA)
            null.append(scores(Sr, dat, md, mc))
        null = np.array(null)

        print(f"\n=== {scope} pair set "
              f"(incidence n={len(dat['y']) if md is None else int(md.sum()):,}, "
              f"correlation n={len(dat['r']) if mc is None else int(mc.sum()):,})")
        print(f"{'':22s}{'incidence AUC':>16s}{'correlation rho':>18s}")
        print(f"{'full operator':22s}{af:>16.4f}{rf:>18.4f}")
        print(f"{'class quotient (k=%d)' % k:22s}{ab:>16.4f}{rb:>18.4f}")
        print(f"{'random, same sizes':22s}{null[:,0].mean():>16.4f}{null[:,1].mean():>18.4f}"
              f"   (sd {null[:,0].std():.4f} / {null[:,1].std():.4f})")

        for name, (a, r_) in (("bio", (ab, rb)), ("random", (null[:, 0].mean(), null[:, 1].mean()))):
            fa = (af - a) / (af - 0.5)
            fr = (rf - r_) / rf
            print(f"  {name:8s} fraction of headroom lost:  incidence {fa:+.3f}   "
                  f"correlation {fr:+.3f}")
            rows.append((scope, name, a, r_, fa, fr))
        pa = (null[:, 0] >= ab).mean()
        pr = (null[:, 1] >= rb).mean()
        za = (ab - null[:, 0].mean()) / null[:, 0].std()
        zr = (rb - null[:, 1].mean()) / null[:, 1].std()
        print(f"  biology vs size-matched random:  incidence z={za:+.2f} (p={pa:.3f})   "
              f"correlation z={zr:+.2f} (p={pr:.3f})")
        rows.append((scope, "z", za, zr, pa, pr))

    import csv
    tag = args.tag or (f"excise{args.excise}" + ("_detrend" if args.detrend else ""))
    out = ROOT / f"results/two_observables_{tag}.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scope", "arm", "auc_or_z", "rho_or_zr", "frac_inc_or_p", "frac_corr_or_p"])
        w.writerows(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
