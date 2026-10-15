"""Wild-type and unc-31 responses per stimulation, and the path-operator advantage
at matched numbers of stimulations.

Every stimulation event in the per-recording exports of both strains is kept as
one row per (event, responder). The windowed response of a responder is

    a = mean(F[s + WIN[0] : s + WIN[1]]) - mean(F[s - PRE : s])

for onset volume s (conres.observables PRE, WIN), in the units of the exported
traces. Recordings shorter than 100 volumes, traces with any non-finite sample
and events without a full window are skipped, as in conres.observables.build_sta.
The same quantity evaluated at every possible onset of a recording gives each
trace's reference distribution, with median `med`, scaled MAD `mad` and the 0.95
quantile `q95` of |a - med|.

Trial outcomes:
  occ      1 if |a - med| > q95 (the same exceedance rate for every trace by
           construction)
  m_zeta   |a - med| / mad
  m_sd     |a| / whole-recording SD of the trace
  m_dff    |a| / pre-stimulus mean F0, for F0 > 1
  m_raw    |a|
Magnitudes are capped at their pooled 0.999 quantile.

Pair covariates on the Cook 2019 connectome: path operator R at alpha = 0.1
(normal scores of its rank over trials), direct synapse (A[stim, resp] > 0),
directed hop count on the binarised connectome, and distance, standardised
log(1 + d) with d either the soma distance (conres.worm.soma_xyz) or the minimum
distance between any two morphology points of the cells (conres.worm.neurites).
Pairs lacking either distance are dropped.

  1  recordings with usable trials per strain and their medians of trace SD,
     first-difference noise, F and mad
  2  released-atlas response rate per strain among observed pairs with at most
     --o-split stimulations and with more
  3  mean of each trial outcome per strain and hop count (4 = four or more)
  4  trial model: outcome ~ recording fixed effects + path + distance + direct
     + genotype x (path + distance + direct), least squares; genotype = 1 for
     wild type, so the interaction terms are wild type minus unc-31. Intervals:
     --n-boot resamples of recordings within strain, and of the stimulated
     neuron's cell class
  5  per-recording fits of occ and m_zeta on path, soma distance and direct
     (recordings with >= 300 trials); medians per strain and Mann-Whitney p
  6  detection at matched stimulation counts. zeta = (a - med) / mad per trial;
     a pair's statistic is sum(zeta) / sqrt(m) over its m trials and the pair
     is detected when |statistic| exceeds a threshold (--thresholds). For pairs
     with trials in both strains, m = the smaller of the two counts and that
     many trials are drawn at random in each strain, --n-sub times. Reported per
     threshold: detection rate, AUC(R), AUC(R) - AUC(direct strength) per
     strain and the difference between strains, averaged over subsamples. The
     interval takes one class-bootstrap resample of the pairs per subsample, so
     it covers trial subsampling and pair sampling. The same statistics with
     all trials, and the AUC of |statistic| for the released wild-type atlas
     label, are printed for reference.

Inputs: conres.celegans.build_matrix, conres.randi (per-recording exports,
    build_occ3, load_responses), conres.worm (soma positions, morphologies).
Outputs: results/unc31_trial_model.csv (section, outcome, variant, term,
    value, lo, hi), results/unc31_trial_model.log.
Usage: python scripts/unc31_trial_model.py [--n-boot 500] [--n-sub 100]
    [--thresholds 2.5 3.5] [--o-split 5] [--seed 0]
"""
from __future__ import annotations

import argparse
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.sparse.csgraph import shortest_path
from scipy.spatial import cKDTree
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from conres import celegans as ce, randi, worm                     # noqa: E402
from conres.metric_guard import guard_predictor                    # noqa: E402
from conres.observables import PRE, POST, WIN                      # noqa: E402

ALPHA = 0.1
STRAINS = ("wt", "unc31")
OUTCOMES = ("occ", "m_zeta", "m_sd", "m_dff", "m_raw")
TERMS = ("path", "dist", "direct", "path:wt", "dist:wt", "direct:wt")


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for s in self.streams:
            s.write(text)

    def flush(self):
        for s in self.streams:
            s.flush()


def neurite_distance(nodes: list[str]) -> np.ndarray:
    """Minimum distance (um) between any morphology points of each pair of cells; nan without a morphology."""
    pts = [worm.neurites([m]).reshape(-1, 3) for m in nodes]
    trees = [cKDTree(p) if len(p) else None for p in pts]
    n = len(nodes)
    d = np.full((n, n), np.nan)
    for a in range(n):
        if trees[a] is None:
            continue
        for b in range(a + 1, n):
            if trees[b] is not None:
                d[a, b] = d[b, a] = trees[b].query(pts[a])[0].min()
    return d


def trial_table(idx: dict[str, int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One row per (stimulation event, responder) in both strains, and one row per recording."""
    rows, recs = [], []
    for strain in STRAINS:
        folder = randi.EXPORTED / ("exported_data" if strain == "wt" else f"exported_data_{strain}")
        for f in sorted(folder.glob("*_labels.txt")):
            r = int(re.match(r"(\d+)_labels", f.name).group(1))
            labels = [ce.normalise_name(x.rstrip("\n").strip()) for x in f.open()]
            sv, sn = folder / f"{r}_stim_volume_i.txt", folder / f"{r}_stim_neurons.txt"
            if not (sv.exists() and sn.exists()):
                continue
            G = randi._gcamp(folder, r)
            if G.ndim < 2 or G.shape[0] < 100:
                continue
            alive = np.isfinite(G).all(axis=0)
            cols = [(c, idx[x]) for c, x in enumerate(labels)
                    if x in idx and c < G.shape[1] and alive[c]]
            if len(cols) < 2:
                continue
            ci = np.array([c for c, _ in cols])
            ni = np.array([k for _, k in cols])
            X = G[:, ci]
            sd = np.maximum(X.std(0), 1e-9)
            dX = np.diff(X, axis=0)
            # SD of single-volume noise from first differences (variance 2 sigma^2)
            hf = 1.4826 * np.median(np.abs(dX - np.median(dX, 0)), 0) / np.sqrt(2)
            cs = np.vstack([np.zeros(X.shape[1]), np.cumsum(X, 0)])
            onset = np.arange(PRE, X.shape[0] - POST)
            base = (cs[onset] - cs[onset - PRE]) / PRE
            a_all = (cs[onset + WIN[1]] - cs[onset + WIN[0]]) / (WIN[1] - WIN[0]) - base
            med = np.median(a_all, 0)
            mad = np.maximum(1.4826 * np.median(np.abs(a_all - med), 0), 1e-9)
            q95 = np.quantile(np.abs(a_all - med), 0.95, axis=0)
            stims = [int(x) for x in sv.open() if x.strip()]
            sneu = [int(x) for x in sn.open() if x.strip()]
            rec = len(recs)
            n_trials = 0
            for s, c in zip(stims, sneu):
                if not (0 <= c < len(labels)) or labels[c] not in idx:
                    continue
                if s - PRE < 0 or s + POST >= X.shape[0]:
                    continue
                j = idx[labels[c]]
                a, f0 = a_all[s - PRE], base[s - PRE]
                keep = ni != j
                rows.append(pd.DataFrame(dict(
                    strain=strain, rec=rec, stim=j, resp=ni[keep], a=a[keep], f0=f0[keep],
                    sd=sd[keep], zeta=((a - med) / mad)[keep],
                    occ=(np.abs(a - med) > q95)[keep].astype(float))))
                n_trials += 1
            recs.append(dict(rec=rec, strain=strain, n_trials=n_trials, trace_sd=np.median(sd),
                             noise=np.median(hf), F=np.median(X), mad=np.median(mad)))
    return pd.concat(rows, ignore_index=True), pd.DataFrame(recs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--n-sub", type=int, default=100)
    ap.add_argument("--thresholds", type=float, nargs="+", default=[2.5, 3.5])
    ap.add_argument("--o-split", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    sys.stdout = _Tee(sys.__stdout__, (ROOT / "results/unc31_trial_model.log").open("w"))
    rng = np.random.default_rng(args.seed)
    out = []

    def add(section, outcome, variant, term, value, lo=np.nan, hi=np.nan):
        out.append(dict(section=section, outcome=outcome, variant=variant, term=term,
                        value=value, lo=lo, hi=hi))

    A, nodes = ce.build_matrix("all", neurons_only=True)
    idx = {m: i for i, m in enumerate(nodes)}
    cls = np.unique(ce.partitions(nodes)["class"], return_inverse=True)[1]
    R = randi.resolvent(A, ALPHA)
    hops = shortest_path((A > 0).astype(float), unweighted=True, directed=True)
    xyz = worm.soma_xyz(nodes)
    dsoma = np.linalg.norm(xyz[:, None] - xyz[None], axis=-1)
    dneur = neurite_distance(nodes)

    D, recs = trial_table(idx)
    j, i = D.stim.to_numpy(), D.resp.to_numpy()
    D["cls"], D["strength"], D["R"], D["hops"] = cls[j], A[j, i], R[j, i], hops[j, i]
    D["dsoma"], D["dneur"] = dsoma[j, i], dneur[j, i]
    D = D[np.isfinite(D.dsoma) & np.isfinite(D.dneur)].reset_index(drop=True)
    z = lambda v: (v - v.mean()) / v.std()                         # noqa: E731
    D["path"] = stats.norm.ppf((D.R.rank() - 0.5) / len(D))
    D["soma"], D["neurite"] = z(np.log1p(D.dsoma)), z(np.log1p(D.dneur))
    D["direct"] = (D.strength > 0).astype(float)
    D["wt"] = (D.strain == "wt").astype(float)
    cap = lambda v: np.minimum(v, np.nanquantile(v, 0.999))         # noqa: E731
    D["m_zeta"], D["m_sd"], D["m_raw"] = cap(D.zeta.abs()), cap((D.a / D.sd).abs()), cap(D.a.abs())
    D["m_dff"] = cap((D.a / D.f0.where(D.f0 > 1)).abs())

    # -- 1 recordings
    print("1  recordings with usable trials, trials, and per-recording medians")
    g = recs[recs.n_trials > 0].groupby("strain")
    tab = g[["trace_sd", "noise", "F", "mad"]].median()
    tab.insert(0, "trials", D.groupby("strain").size())
    tab.insert(0, "recordings", D.groupby("strain").rec.nunique())
    print(tab.loc[list(STRAINS)].round(3).to_string())
    for s in STRAINS:
        for c in tab.columns:
            add("recordings", c, "", s, tab.loc[s, c])

    # -- 2 released atlas at equal observation counts
    print(f"\n2  released-atlas response rate by observation count O (split at {args.o_split})")
    for s in STRAINS:
        d = randi.load_responses("wildty" if s == "wt" else s)
        d = d[d.pre.isin(idx) & d.post.isin(idx)]
        hits = {(idx[p], idx[q]) for p, q in zip(d.pre, d.post)}
        occ3, _ = randi.build_occ3(nodes, s)
        ii, jj = np.nonzero(occ3)
        y = np.array([(b, a) in hits for a, b in zip(ii, jj)], float)
        low = occ3[ii, jj] <= args.o_split
        for name, m in ((f"O<={args.o_split}", low), (f"O>{args.o_split}", ~low)):
            print(f"   {s:>6} {name:>5}: {int(m.sum()):6d} pairs, mean O {occ3[ii, jj][m].mean():5.2f}, "
                  f"rate {y[m].mean():.4f}")
            add("atlas_rate", "rate", name, s, y[m].mean())
            add("atlas_rate", "n_pairs", name, s, int(m.sum()))

    # -- 3 trial outcomes by hop count
    print("\n3  mean trial outcome by hop count (4 = four or more)")
    D["hop"] = np.minimum(D.hops, 4).astype(int)
    tab = D.groupby(["hop", "strain"])[list(OUTCOMES)].mean()
    print(tab.unstack().round(3).to_string())
    for (h, s), row in tab.iterrows():
        for o in OUTCOMES:
            add("by_hop", o, f"hop{h}", s, row[o])

    # -- 4 trial model
    def trial_model(outcome, dist, cluster):
        d = D[np.isfinite(D[outcome])]
        X = pd.DataFrame({"path": d.path, "dist": d[dist], "direct": d.direct})
        for t in ("path", "dist", "direct"):
            X[t + ":wt"] = X[t] * d.wt
        X["y"] = d[outcome]
        X = (X - X.groupby(d.rec).transform("mean")).to_numpy()       # recording fixed effects
        X, yv = X[:, :6], X[:, 6]
        code = np.unique(d[cluster], return_inverse=True)[1]
        XX = np.zeros((code.max() + 1, 6, 6))
        Xy = np.zeros((code.max() + 1, 6))
        for k in range(code.max() + 1):
            m = code == k
            XX[k], Xy[k] = X[m].T @ X[m], X[m].T @ yv[m]
        est = np.linalg.solve(XX.sum(0), Xy.sum(0))
        if cluster == "rec":                                          # resample within strain
            is_wt = d.groupby(code).wt.first().to_numpy() == 1
            strata = [np.flatnonzero(is_wt), np.flatnonzero(~is_wt)]
        else:
            strata = [np.arange(code.max() + 1)]
        draws = []
        for _ in range(args.n_boot):
            s = np.concatenate([rng.choice(st, st.size) for st in strata])
            draws.append(np.linalg.solve(XX[s].sum(0), Xy[s].sum(0)))
        return est, *np.percentile(draws, [2.5, 97.5], axis=0)

    for dist, cluster in (("soma", "rec"), ("neurite", "rec"), ("soma", "cls")):
        print(f"\n4  trial model, {dist} distance, bootstrap over "
              f"{'recordings within strain' if cluster == 'rec' else 'stimulated-neuron classes'}")
        print(f"{'':>8} " + " ".join(f"{t:>27}" for t in TERMS))
        for o in OUTCOMES:
            est, lo, hi = trial_model(o, dist, cluster)
            print(f"{o:>8} " + " ".join(f"{e:+.4f} [{a:+.4f}, {b:+.4f}]" for e, a, b in zip(est, lo, hi)))
            for t, e, a, b in zip(TERMS, est, lo, hi):
                add("trial_model", o, f"{dist}_boot_{cluster}", t, e, a, b)

    # -- 5 per-recording fits
    print("\n5  per-recording fits (>= 300 trials): median wild type, median unc-31, Mann-Whitney p")
    fits = []
    for r, d in D.groupby("rec"):
        if len(d) < 300:
            continue
        X = np.column_stack([np.ones(len(d)), d.path, d.soma, d.direct])
        for o in ("occ", "m_zeta"):
            b = np.linalg.lstsq(X, d[o].to_numpy(), rcond=None)[0]
            fits.append(dict(strain=d.strain.iloc[0], outcome=o, intercept=b[0], path=b[1],
                             dist=b[2], direct=b[3]))
    fits = pd.DataFrame(fits)
    for o in ("occ", "m_zeta"):
        w = fits[(fits.outcome == o) & (fits.strain == "wt")]
        u = fits[(fits.outcome == o) & (fits.strain == "unc31")]
        print(f"   {o}: recordings {len(w)} / {len(u)}")
        for t in ("intercept", "path", "dist", "direct"):
            p = stats.mannwhitneyu(w[t], u[t]).pvalue
            print(f"      {t:>9} {w[t].median():+.4f} {u[t].median():+.4f}  p {p:.3f}")
            add("recording_fits", o, "median", f"{t}:wt_rec", w[t].median())
            add("recording_fits", o, "median", f"{t}:unc31_rec", u[t].median())
            add("recording_fits", o, "mannwhitney_p", t, p)

    # -- 6 detection at matched stimulation counts
    pair_cov = D.drop_duplicates(["stim", "resp"]).set_index(["stim", "resp"])[["cls", "strength", "R"]]

    def pair_stats(d):
        g = d.groupby(["stim", "resp"]).zeta.agg(["sum", "size"])
        return pd.DataFrame({"stat": (g["sum"] / np.sqrt(g["size"])).abs()}).join(pair_cov)

    def scores(t, thr):
        y = (t.stat > thr).astype(float)
        if y.nunique() < 2:
            return np.full(3, np.nan)
        auc = roc_auc_score(y, t.R)
        return np.array([y.mean(), auc, auc - roc_auc_score(y, t.strength)])

    print("\n6  detection rebuilt from trials")
    full = {s: pair_stats(D[D.strain == s]) for s in STRAINS}
    d = randi.load_responses("wildty")
    d = d[d.pre.isin(idx) & d.post.isin(idx)]
    hits = {(idx[p], idx[q]) for p, q in zip(d.pre, d.post)}
    y_atlas = np.array([k in hits for k in full["wt"].index], float)
    auc_atlas = roc_auc_score(y_atlas, full["wt"].stat)
    print(f"   AUC of |statistic| (all wild-type trials) for the released wild-type label: {auc_atlas:.4f}")
    add("matched_detection", "auc_vs_atlas", "all_trials", "wt", auc_atlas)

    n_tr = D.groupby(["strain", "stim", "resp"]).size().unstack(0).dropna()
    m = n_tr.min(axis=1).astype(int)
    print(f"   pairs with trials in both strains: {len(m)}; matched trials per pair: mean {m.mean():.2f}")
    add("matched_detection", "n_pairs", "shared", "both", len(m))
    add("matched_detection", "mean_trials", "matched", "both", m.mean())
    S = D.set_index(["stim", "resp"]).loc[m.index].reset_index()
    S["m"] = m.reindex(pd.MultiIndex.from_frame(S[["stim", "resp"]])).to_numpy()
    names = ("rate", "auc_path", "advantage")
    for thr in args.thresholds:
        for label, tabs in (("all pairs, all trials", full),
                            ("shared pairs, all trials", {s: full[s].loc[m.index] for s in STRAINS})):
            sc = {s: scores(tabs[s], thr) for s in STRAINS}
            print(f"   |statistic| > {thr}, {label}: " + "; ".join(
                f"{s} rate {sc[s][0]:.4f}, AUC(R) {sc[s][1]:.4f}, advantage {sc[s][2]:+.4f}" for s in STRAINS))
            for s in STRAINS:
                for nm, v in zip(names, sc[s]):
                    add("matched_detection", nm, f"thr{thr}_{label.replace(', ', '_').replace(' ', '_')}", s, v)
        for s in STRAINS:
            y = (full[s].stat > thr).astype(float)
            print(f"      guard {s}: {guard_predictor(y, full[s].R)['flags']}; direct tie block "
                  f"{guard_predictor(y, full[s].strength)['largest_tie_block']:.2f}")
    point = {thr: [] for thr in args.thresholds}
    boot = {thr: [] for thr in args.thresholds}
    classes = pair_cov.loc[m.index].cls.unique()
    for _ in range(args.n_sub):
        S["u"] = rng.random(len(S))
        K = S[S.groupby(["strain", "stim", "resp"]).u.rank(method="first") <= S.m]
        tabs = {s: pair_stats(K[K.strain == s]) for s in STRAINS}
        pick = rng.choice(classes, classes.size)
        res = {s: pd.concat([tabs[s][tabs[s].cls == c] for c in pick]) for s in STRAINS}
        for thr in args.thresholds:
            a = {s: scores(tabs[s], thr) for s in STRAINS}
            b = {s: scores(res[s], thr) for s in STRAINS}
            point[thr].append(np.concatenate([a["wt"], a["unc31"], [a["wt"][2] - a["unc31"][2]]]))
            boot[thr].append(np.concatenate([b["wt"], b["unc31"], [b["wt"][2] - b["unc31"][2]]]))
    cols = [f"{nm}:{s}" for s in STRAINS for nm in names] + ["advantage:wt-unc31"]
    for thr in args.thresholds:
        mu = np.nanmean(point[thr], 0)
        lo, hi = np.nanpercentile(boot[thr], [2.5, 97.5], axis=0)
        print(f"   |statistic| > {thr}, shared pairs, matched trials ({args.n_sub} subsamples):")
        for c, v, a, b in zip(cols, mu, lo, hi):
            print(f"      {c:>20} {v:+.4f} [{a:+.4f}, {b:+.4f}]")
            nm, s = c.split(":")
            add("matched_detection", nm, f"thr{thr}_shared_pairs_matched_trials", s, v, a, b)

    pd.DataFrame(out).to_csv(ROOT / "results/unc31_trial_model.csv", index=False)
    print("\nwrote results/unc31_trial_model.csv")


if __name__ == "__main__":
    main()
