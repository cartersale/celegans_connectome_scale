"""Signal-propagation AUC of each structural predictor conditional on observation count.

A response is detected after however many stimulations a pair received, so
detection depends on the observation count O = occ3[responder, stimulated].
Quartile stratification leaves much of that dependence inside the top quartile,
so three stricter controls are run on the same pairs and predictors
(observation count, direct connection strength, two-step binary path count,
path operator R at alpha = 0.1, class-level path operator on cell classes), for
all observed pairs and for unsynapsed pairs:

1. Exact stratification. Each responding pair is compared only with
   non-responding pairs of identical O. The pooled statistic is the per-stratum
   Mann-Whitney AUC weighted by n_pos * n_neg, i.e. the probability that a
   responding pair outranks a non-responding pair with the same O; strata with
   no responders or no non-responders drop out. Null: predictor values permuted
   within strata (--n-perm, p = (1 + #null >= obs) / (1 + n_perm)). Also
   reported: the quartile-stratified AUC, the raw AUC, and paired exact-O
   contrasts (path operator - direct; class-level - full).

2. Conditional logistic model. y ~ exposure + predictor, with exposure entered
   as one indicator per exact O for O < 40 and three pooled bins above
   (40-54, 55-74, >= 75). Predictors enter as standardized ranks. Likelihood
   ratio tests: path operator beyond exposure (both pair sets); path operator
   beyond exposure and direct connection strength (all observed pairs only).

3. Trial model. If each stimulation evokes a detectable response independently
   with probability q, P(detect | O) = 1 - (1 - q)^O, with logit q = a + b z and
   z the standardized rank of the path operator; b is tested by likelihood
   ratio. A second fit adds log O to logit q, so the independent-trial form is
   not forced to hold.

Cell-class sufficiency within exact strata: (class-level AUC - 0.5) /
(full AUC - 0.5), against --n-rand size-matched random partitions
(permutations of the cell-class labels).

Intervals are percentile intervals from the cluster bootstrap over
stimulated-neuron cell classes (conres.observables.boot_indices, --n-boot
resamples, seed 0). The logistic odds ratio and the trial-model coefficient use
only the first 200 of those resamples; logistic fits that fail to converge are
dropped from the interval.

Produces the Supplementary Information section and table on signal propagation
conditional on observation count.

Inputs: conres.observables.load (Cook 2019 adjacency, Randi 2023 wild-type
    responses), conres.randi.build_occ3.
Outputs: results/exposure_control.csv (per predictor and contrast),
    results/exposure_control_summary.csv (per pair set: O range, quartile
    ranges, Spearman(O, R), class fractions, logistic and trial fits).
Usage: python scripts/exposure_control.py [--n-boot 500] [--n-perm 1000] [--n-rand 100]
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from conres import observables as ob, randi   # noqa: E402

ALPHA = 0.1


def zrank(x):
    r = stats.rankdata(x)
    return (r - r.mean()) / r.std()


def strat_auc(p, y, O):
    """Probability a responding pair outranks a non-responding pair with the same O,
    minus 0.5 (pooled over strata, weighted by n_pos * n_neg)."""
    num = den = 0.0
    for o in np.unique(O):
        m = O == o
        yy = y[m]
        npos = yy.sum()
        nneg = yy.size - npos
        if npos == 0 or nneg == 0:
            continue
        w = npos * nneg
        num += roc_auc_score(yy, p[m]) * w
        den += w
    return num / den - 0.5 if den else np.nan


def informative(y, O):
    """Fraction of responding pairs whose stratum also has non-responding pairs."""
    keep = 0
    for o in np.unique(O):
        m = O == o
        if 0 < y[m].sum() < m.sum():
            keep += y[m].sum()
    return keep / y.sum()


def exposure_design(O):
    """One indicator per exact O below 40; the sparse tail pooled into 40-54, 55-74, >= 75."""
    b = np.where(O < 40, O, np.where(O < 55, 40, np.where(O < 75, 55, 75)))
    return pd.get_dummies(b.astype(int), prefix="O", drop_first=False).astype(float).to_numpy()


def logit_llf(y, X):
    return sm.Logit(y, X).fit(disp=0, method="bfgs", maxiter=2000).llf


def trial_nll(theta, y, O, Z):
    q = 1 / (1 + np.exp(-(Z @ theta)))
    p = 1 - np.power(1 - q, O)
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log1p(-p)).sum()


def trial_fit(y, O, Z):
    t0 = np.zeros(Z.shape[1])
    t0[0] = np.log(y.mean() / O.mean())
    r = optimize.minimize(trial_nll, t0, args=(y, O, Z), method="BFGS")
    return r.x, -r.fun


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--n-rand", type=int, default=100)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    D = ob.load()
    A, nodes, cls, obs = D["A"], D["nodes"], D["cls"], D["obs"]
    occ, _ = randi.build_occ3(nodes, "wt")
    R = randi.resolvent(A, ALPHA)
    Q = randi.class_leverage(A, cls, ALPHA)
    B = (A > 0).astype(float)
    two = B @ B
    boots = ob.boot_indices(obs, cls, args.n_boot)

    rows, summary = [], []
    for o in ("incidence_all", "incidence_nosyn"):
        j, i, y, _ = obs[o]
        O = occ[i, j]
        pred = {"observation count": O, "direct connection strength": A[j, i],
                "two-step path count": two[j, i], "path operator": R[j, i],
                "class-level path operator (cell classes)": Q[j, i]}
        if o == "incidence_nosyn":
            del pred["direct connection strength"]
        rho_O = stats.spearmanr(O, R[j, i]).statistic
        q = np.quantile(O, [.25, .5, .75])
        quart = np.digitize(O, q)
        spread = [f"{O[quart == k].min():.0f}-{O[quart == k].max():.0f}" for k in range(4)]
        print(f"\n== {o}: {y.size} pairs, {int(y.sum())} responses; O spans {O.min():.0f}-{O.max():.0f} "
              f"({np.unique(O).size} values); quartiles span {', '.join(spread)}")
        print(f"   Spearman(O, path operator) = {rho_O:+.3f}; responding pairs in informative exact "
              f"strata: {informative(y, O):.1%}")
        summary.append({"observable": o, "pairs": y.size, "responses": int(y.sum()),
                        "O_min": O.min(), "O_max": O.max(), "O_values": np.unique(O).size,
                        "quartile_ranges": "; ".join(spread), "rho_O_path": rho_O,
                        "informative_frac": informative(y, O)})

        for name, p in pred.items():
            raw = roc_auc_score(y, p) - 0.5
            qa = strat_auc(p, y, quart)
            ea = strat_auc(p, y, O)
            eb = np.array([strat_auc(p[s], y[s], O[s]) for s in boots[o]])
            r = {"observable": o, "predictor": name, "auc_raw": raw + 0.5,
                 "auc_quartile": qa + 0.5, "auc_exact": ea + 0.5,
                 "auc_exact_lo": np.nanpercentile(eb, 2.5) + 0.5,
                 "auc_exact_hi": np.nanpercentile(eb, 97.5) + 0.5}
            if name not in ("observation count",):
                # permutation within exact strata
                strata = [np.flatnonzero(O == v) for v in np.unique(O)]
                null = []
                for _ in range(args.n_perm):
                    pp = p.copy()
                    for s in strata:
                        pp[s] = p[rng.permutation(s)]
                    null.append(strat_auc(pp, y, O))
                null = np.array(null)
                r["perm_p"] = (1 + (null >= ea).sum()) / (1 + args.n_perm)
            rows.append(r)
            print(f"   {name:>42}: AUC raw {raw + 0.5:.3f}  within quartiles {qa + 0.5:.3f}  "
                  f"within exact O {ea + 0.5:.3f} [{r['auc_exact_lo']:.3f}, {r['auc_exact_hi']:.3f}]"
                  + (f"  perm p {r['perm_p']:.4f}" if "perm_p" in r else ""))

        # paired contrasts under exact stratification
        for a_, b_ in (("path operator", "direct connection strength"),
                       ("class-level path operator (cell classes)", "path operator")):
            if a_ not in pred or b_ not in pred:
                continue
            d = np.array([strat_auc(pred[a_][s], y[s], O[s]) - strat_auc(pred[b_][s], y[s], O[s])
                          for s in boots[o]])
            pt = strat_auc(pred[a_], y, O) - strat_auc(pred[b_], y, O)
            print(f"   exact-O contrast {a_} - {b_}: {pt:+.4f} "
                  f"[{np.nanpercentile(d, 2.5):+.4f}, {np.nanpercentile(d, 97.5):+.4f}]")
            rows.append({"observable": o, "predictor": f"contrast: {a_} - {b_}", "auc_exact": pt,
                         "auc_exact_lo": np.nanpercentile(d, 2.5), "auc_exact_hi": np.nanpercentile(d, 97.5)})

        # class-level sufficiency conditional on exposure, against random partitions
        full = strat_auc(R[j, i], y, O)
        rnd = []
        for _ in range(args.n_rand):
            Qr = randi.class_leverage(A, rng.permutation(cls), ALPHA)
            rnd.append(strat_auc(Qr[j, i], y, O))
        rnd = np.array(rnd)
        cl = strat_auc(Q[j, i], y, O)
        fb = np.array([strat_auc(Q[j, i][s], y[s], O[s]) / strat_auc(R[j, i][s], y[s], O[s])
                       for s in boots[o]])
        print(f"   cell classes within exact O: {cl / full:.2f} of full above chance "
              f"[{np.nanpercentile(fb, 2.5):.2f}, {np.nanpercentile(fb, 97.5):.2f}]; "
              f"random {rnd.mean() / full:.2f} [{np.percentile(rnd, 2.5) / full:.2f}, "
              f"{np.percentile(rnd, 97.5) / full:.2f}]; z {(cl - rnd.mean()) / rnd.std():+.1f}")
        summary[-1].update({"class_frac_exact": cl / full, "class_frac_lo": np.nanpercentile(fb, 2.5),
                            "class_frac_hi": np.nanpercentile(fb, 97.5),
                            "random_frac_exact": rnd.mean() / full,
                            "z_class_vs_random": (cl - rnd.mean()) / rnd.std()})

        # conditional logistic model with exact-O intercepts
        E = exposure_design(O)
        zp, zd = zrank(R[j, i]), zrank(A[j, i])
        l0 = logit_llf(y, E)
        l1 = logit_llf(y, np.column_stack([E, zp]))
        fit = sm.Logit(y, np.column_stack([E, zp])).fit(disp=0, method="bfgs", maxiter=2000)
        beta = fit.params[-1]
        bb = []
        for s in boots[o][:200]:
            Es = exposure_design(O[s])
            try:
                bb.append(sm.Logit(y[s], np.column_stack([Es, zrank(R[j, i][s])]))
                          .fit(disp=0, method="bfgs", maxiter=2000).params[-1])
            except Exception:
                pass
        lr1 = 2 * (l1 - l0)
        line = (f"   logistic, exact-O intercepts: path operator OR per s.d. of rank "
                f"{np.exp(beta):.2f} [{np.exp(np.percentile(bb, 2.5)):.2f}, "
                f"{np.exp(np.percentile(bb, 97.5)):.2f}], LR {lr1:.1f}, p {stats.chi2.sf(lr1, 1):.1e}")
        s_ = {"logit_OR": np.exp(beta), "logit_OR_lo": np.exp(np.percentile(bb, 2.5)),
              "logit_OR_hi": np.exp(np.percentile(bb, 97.5)), "logit_LR": lr1,
              "logit_p": stats.chi2.sf(lr1, 1)}
        if o == "incidence_all":
            l2 = logit_llf(y, np.column_stack([E, zd]))
            l3 = logit_llf(y, np.column_stack([E, zd, zp]))
            lr2 = 2 * (l3 - l2)
            line += f"; beyond direct connection strength LR {lr2:.1f}, p {stats.chi2.sf(lr2, 1):.1e}"
            s_.update({"logit_LR_beyond_direct": lr2, "logit_p_beyond_direct": stats.chi2.sf(lr2, 1)})
        print(line)

        # trial model
        one = np.ones_like(O)
        _, t0 = trial_fit(y, O, one[:, None])
        th1, t1 = trial_fit(y, O, np.column_stack([one, zp]))
        _, t2 = trial_fit(y, O, np.column_stack([one, np.log(O)]))
        th3, t3 = trial_fit(y, O, np.column_stack([one, np.log(O), zp]))
        tb = []
        for s in boots[o][:200]:
            th, _ = trial_fit(y[s], O[s], np.column_stack([np.ones(s.size), np.log(O[s]),
                                                           zrank(R[j, i][s])]))
            tb.append(th[-1])
        lrt1, lrt3 = 2 * (t1 - t0), 2 * (t3 - t2)
        print(f"   trial model 1-(1-q)^O: path operator log-odds per s.d. {th1[-1]:+.3f}, LR {lrt1:.1f} "
              f"p {stats.chi2.sf(lrt1, 1):.1e}; with log O in q: {th3[-1]:+.3f} "
              f"[{np.percentile(tb, 2.5):+.3f}, {np.percentile(tb, 97.5):+.3f}], LR {lrt3:.1f} "
              f"p {stats.chi2.sf(lrt3, 1):.1e}; log O coefficient {th3[1]:+.3f}")
        s_.update({"trial_b": th1[-1], "trial_LR": lrt1, "trial_b_logO": th3[-1],
                   "trial_b_logO_lo": np.percentile(tb, 2.5), "trial_b_logO_hi": np.percentile(tb, 97.5),
                   "trial_LR_logO": lrt3, "trial_logO_coef": th3[1]})
        summary[-1].update(s_)

    pd.DataFrame(rows).to_csv(ROOT / "results/exposure_control.csv", index=False)
    pd.DataFrame(summary).to_csv(ROOT / "results/exposure_control_summary.csv", index=False)
    print("\nwrote results/exposure_control{,_summary}.csv")


if __name__ == "__main__":
    main()
