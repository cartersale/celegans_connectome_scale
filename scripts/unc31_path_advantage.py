"""Path-operator advantage over direct connection strength in wild type and
unc-31 mutants, with sampling and detection-selectivity controls.

Observed pairs are all (stimulated j, responder i) with occ3[i, j] > 0 in each
strain; a pair is a response if it is in that strain's released response list.
Path operator R at alpha = 0.1 on the Cook 2019 connectome.

  1  per strain: recordings, observed pairs, responses, mean observation count,
     AUC of direct connection strength and of R, and the advantage
     AUC(R) - AUC(direct), with --n-boot cluster-bootstrap resamples over the
     stimulated neuron's cell class (resamples with a single outcome class are
     skipped). The between-strain difference is bootstrapped by pairing
     independent resamples of each strain; p = (1 + #diff <= 0) / (1 + n),
     two-sided = 2 x one-sided.
  2  AUC(R) - 0.5 among unsynapsed pairs, with its bootstrap interval
  3  ratio of above-chance AUC, (AUC(R) - 0.5) / (AUC(direct) - 0.5)
  4  matched sampling: --n-sub random subsets of wild-type recordings of the
     unc-31 recording count; observed pairs rebuilt from the subset, response
     labels taken from the full wild-type response list; subsets with fewer
     than 30 responses are skipped
  5  selectivity: response magnitudes compared by Mann-Whitney; wild-type
     responses restricted to the largest fraction (unc-31 rate / wild-type
     rate) by magnitude, advantage recomputed
  6  AUC of the extrasynaptic monoamine and neuropeptide networks (Bentley 2016)
  7  conres.metric_guard.guard_predictor on R and direct strength per strain;
     direct strength is tied at zero for most observed pairs

Produces the Supplementary Information section and table on the path-operator
advantage in unc-31 mutants.

Inputs: conres.celegans.build_matrix, conres.randi (build_occ3,
    load_responses for "wildty" and "unc31", load_extrasynaptic, n_recordings).
Outputs: results/unc31_path_advantage.csv (one row per strain, then one row
    "matched_wt_subsamples" holding every subsample's advantage); the
    remaining statistics are printed only.
Usage: python scripts/unc31_path_advantage.py [--n-boot 2000] [--n-sub 100] [--seed 0]
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
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from conres import celegans as ce, randi                          # noqa: E402
from conres.metric_guard import guard_predictor                    # noqa: E402

ALPHA = 0.1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-sub", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    A, nodes = ce.build_matrix("all", neurons_only=True)
    idx = {n: i for i, n in enumerate(nodes)}
    labels = np.unique(ce.partitions(nodes)["class"], return_inverse=True)[1]
    R = randi.resolvent(A, ALPHA)
    MA = randi.load_extrasynaptic("MA", nodes)
    NP = randi.load_extrasynaptic("NP", nodes)

    def responses(strain):
        d = randi.load_responses("wildty" if strain == "wt" else strain)
        d = d[d.pre.isin(idx) & d.post.isin(idx)]
        return d, {(idx[p], idx[q]) for p, q in zip(d.pre, d.post)}

    def pairs(occ3, hits):
        ii, jj = np.nonzero(occ3)
        y = np.array([1.0 if (j, i) in hits else 0.0 for i, j in zip(ii, jj)])
        return jj, ii, y

    def advantage(jj, ii, y, mask=None):
        m = slice(None) if mask is None else mask
        return roc_auc_score(y[m], R[jj, ii][m]) - roc_auc_score(y[m], A[jj, ii][m])

    rng = np.random.default_rng(args.seed)
    out = {}
    for strain in ("wt", "unc31"):
        d, hits = responses(strain)
        occ3, _ = randi.build_occ3(nodes, strain)
        jj, ii, y = pairs(occ3, hits)
        ns = A[jj, ii] == 0
        g = labels[jj]
        where = {c: np.flatnonzero(g == c) for c in np.unique(g)}
        uq = np.array(list(where))
        adv, adv_ns, rp, rd = [], [], [], []
        for _ in range(args.n_boot):
            s = np.concatenate([where[c] for c in rng.choice(uq, uq.size)])
            if len(np.unique(y[s])) < 2:
                continue
            adv.append(advantage(jj[s], ii[s], y[s]))
            sns = s[ns[s]]
            adv_ns.append(roc_auc_score(y[sns], R[jj[sns], ii[sns]]) - 0.5
                          if len(np.unique(y[sns])) > 1 else np.nan)
        out[strain] = dict(
            n_rec=randi.n_recordings(strain), n_pairs=len(y), n_resp=int(y.sum()),
            rate=y.mean(), expo=occ3[ii, jj].mean(), ns_n=int(ns.sum()), ns_resp=int(y[ns].sum()),
            auc_direct=roc_auc_score(y, A[jj, ii]), auc_path=roc_auc_score(y, R[jj, ii]),
            auc_path_ns=roc_auc_score(y[ns], R[jj, ii][ns]),
            auc_ma=roc_auc_score(y, MA[jj, ii]), auc_np=roc_auc_score(y, NP[jj, ii]),
            adv=advantage(jj, ii, y), adv_boot=np.array(adv), adv_ns_boot=np.array(adv_ns),
            guard_path=guard_predictor(y, R[jj, ii]), guard_direct=guard_predictor(y, A[jj, ii]),
            mag=d["mag"].to_numpy(float), d=d)

    print(f"{'':>34} {'wild type':>22} {'unc-31':>22}")
    for key, name in (("n_rec", "recordings"), ("n_pairs", "observed pairs"),
                      ("n_resp", "responses"), ("rate", "response rate"),
                      ("expo", "mean observation count"),
                      ("auc_direct", "AUC direct connection strength"),
                      ("auc_path", "AUC path operator"), ("auc_path_ns", "AUC path, unsynapsed pairs"),
                      ("auc_ma", "AUC monoamine network"), ("auc_np", "AUC neuropeptide network")):
        w, u = out["wt"][key], out["unc31"][key]
        fmt = (lambda v: f"{v:>22,}") if isinstance(w, (int, np.integer)) else (lambda v: f"{v:>22.4f}")
        print(f"{name:>34} {fmt(w)} {fmt(u)}")
    for s in ("wt", "unc31"):
        o = out[s]
        lo, hi = np.percentile(o["adv_boot"], [2.5, 97.5])
        nlo, nhi = np.nanpercentile(o["adv_ns_boot"], [2.5, 97.5])
        o["adv_ci"] = (lo, hi)
        print(f"{'path advantage ' + s:>34} {o['adv']:+.4f} [{lo:+.4f}, {hi:+.4f}]   "
              f"unsynapsed path above chance {o['auc_path_ns'] - 0.5:+.4f} [{nlo:+.4f}, {nhi:+.4f}]")
    n = min(len(out["wt"]["adv_boot"]), len(out["unc31"]["adv_boot"]))
    dd = out["wt"]["adv_boot"][:n] - out["unc31"]["adv_boot"][:n]
    ddp = out["wt"]["adv"] - out["unc31"]["adv"]
    p1 = (np.sum(dd <= 0) + 1) / (n + 1)
    print(f"{'difference of differences':>34} {ddp:+.4f} [{np.percentile(dd, 2.5):+.4f}, "
          f"{np.percentile(dd, 97.5):+.4f}]  one-sided p {p1:.4f}, two-sided {min(1, 2 * p1):.4f}")
    rw = (out["wt"]["auc_path"] - 0.5) / (out["wt"]["auc_direct"] - 0.5)
    ru = (out["unc31"]["auc_path"] - 0.5) / (out["unc31"]["auc_direct"] - 0.5)
    print(f"{'ratio of AUC above chance':>34} WT {rw:.2f}   unc-31 {ru:.2f}")
    for s in ("wt", "unc31"):
        print(f"  guard {s}: path {out[s]['guard_path']['flags']}; "
              f"direct tie block {out[s]['guard_direct']['largest_tie_block']:.2f}")

    # -- control A: matched sampling
    d_wt, hits_wt = responses("wt")
    n_wt, n_u = out["wt"]["n_rec"], out["unc31"]["n_rec"]
    srng = np.random.Generator(np.random.PCG64(args.seed))
    vals, expos = [], []
    for _ in range(args.n_sub):
        keep = set(srng.choice(n_wt, n_u, replace=False).tolist())
        occ = randi.build_occ3(nodes, "wt", keep)[0]
        jj, ii, y = pairs(occ, hits_wt)
        if y.sum() < 30:
            continue
        vals.append(advantage(jj, ii, y))
        expos.append(occ[ii, jj].mean())
    v = np.array(vals)
    print(f"\nmatched sampling: {len(v)} subsamples of {n_u} of {n_wt} WT recordings: advantage "
          f"{v.mean():+.4f} [{np.percentile(v, 2.5):+.4f}, {np.percentile(v, 97.5):+.4f}], "
          f"mean observation count {np.mean(expos):.1f} (unc-31 {out['unc31']['expo']:.1f}); "
          f"unc-31 at percentile {100 * (v <= out['unc31']['adv']).mean():.0f}")

    # -- control B: selectivity
    mw, mu = out["wt"]["mag"], out["unc31"]["mag"]
    print(f"selectivity: response magnitude median WT {np.median(mw):.1f}, unc-31 {np.median(mu):.1f}, "
          f"Mann-Whitney p {stats.mannwhitneyu(mw, mu).pvalue:.1e}")
    target = out["unc31"]["rate"] / out["wt"]["rate"]
    thr = np.quantile(mw, 1 - target)
    strong = {(idx[p], idx[q]) for p, q, m in zip(d_wt.pre, d_wt.post, d_wt["mag"]) if m >= thr}
    occ_wt = randi.build_occ3(nodes, "wt")[0]
    jj, ii, y = pairs(occ_wt, strong)
    print(f"  WT keeping only its top {100 * target:.0f}% of responses by magnitude: advantage "
          f"{advantage(jj, ii, y):+.4f} (full WT {out['wt']['adv']:+.4f}, unc-31 {out['unc31']['adv']:+.4f})")

    import csv
    with (ROOT / "results/unc31_path_advantage.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["strain", "auc_direct", "auc_path", "advantage", "lo", "hi", "auc_path_nosyn",
                    "auc_ma", "auc_np"])
        for s in ("wt", "unc31"):
            o = out[s]
            w.writerow([s, o["auc_direct"], o["auc_path"], o["adv"], *o["adv_ci"],
                        o["auc_path_ns"], o["auc_ma"], o["auc_np"]])
        w.writerow(["matched_wt_subsamples"] + list(v))
    print("\nwrote results/unc31_path_advantage.csv")


if __name__ == "__main__":
    main()
