"""95% intervals for the main path-operator and cell-class statistics under four
resampling schemes for the dependence among observed pairs.

Schemes (--n-boot resamples each, seed 0):

  source         stimulated-neuron cell classes resampled with replacement
                 (the scheme used in the main text). The correlation observable
                 is resampled over the lower-indexed neuron's class, as in
                 conres.observables.boot_indices.
  source x target  two-way (pigeonhole) bootstrap: stimulated- and
                 responding-neuron classes resampled independently, each pair
                 weighted by the product of its two classes' multiplicities
                 (Owen 2007). For the symmetric correlation observable one
                 resample of all cell classes weights both neurons of a pair.
  recording      the wild-type recordings resampled with replacement;
                 observation counts, the observed pair set and activity
                 correlations are rebuilt from per-recording matrices.
                 Detection labels are the atlas's and are held fixed, because
                 the detection criterion pools recordings and cannot be
                 recomputed per resample. Correlation pairs are kept if the two
                 neurons were recorded together in >= 3 distinct recordings.
  recording + source x target   both at once.

Statistics (path operator R at alpha = 0.1): AUC - 0.5 over all observed pairs,
unsynapsed pairs, and within exact observation counts (strat_auc from
exposure_control.py); AUC(R) - AUC(direct connection strength); the ratio of
median R between responding and non-responding unsynapsed pairs; Spearman rho
of R[a,b] + R[b,a] with activity correlations. For the class-level operator on
cell classes and on the loss-guided k = 9 and k = 8 partitions: fraction of the
full operator's above-chance performance (all pairs, unsynapsed, correlation),
and for cell classes the fraction minus the mean over --n-rand size-matched
random partitions (unsynapsed propagation, correlation). p_le_0 / p_le_1 are
the fractions of resamples at or below 0 (differences) or 1 (ratios).

Produces the Supplementary Information table of intervals under the four
resampling schemes.

Inputs: conres.observables.load, conres.randi (per-recording build_occ3 and
    build_correlations over data/raw/ex/exported_data), Randi 2023 wild-type
    responses, results/coarse_trajectory_paths.npz (loss-guided partitions,
    from coarse_trajectory.py).
Outputs: results/dependence_control.csv (scheme, statistic, point, lo, hi, sd,
    p_le_0, p_le_1).
Usage: python scripts/dependence_control.py [--n-boot 500] [--n-rand 20]
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
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from conres import observables as ob, randi   # noqa: E402
from exposure_control import strat_auc   # noqa: E402

ALPHA = 0.1


def auc(p, y):
    return roc_auc_score(y, p) - 0.5 if 0 < y.sum() < y.size else np.nan


def rho(p, v):
    return float(stats.spearmanr(p, v).statistic)


def expand(w):
    """Integer weights -> repeated index array."""
    return np.repeat(np.arange(w.size), w.astype(int))


def per_recording(nodes):
    folder = randi.EXPORTED / "exported_data"
    ids = sorted(int(re.match(r"(\d+)_labels", f.name).group(1)) for f in folder.glob("*_labels.txt"))
    occ, ssum, cnt = [], [], []
    for r in ids:
        o, _ = randi.build_occ3(nodes, "wt", keep={r})
        R, c = randi.build_correlations(nodes, "wt", keep={r})
        occ.append(o.astype(np.float32))
        ssum.append(np.nan_to_num(R * c).astype(np.float32))
        cnt.append(c.astype(np.float32))
    return ids, np.array(occ), np.array(ssum), np.array(cnt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--n-rand", type=int, default=20)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    D = ob.load()
    A, nodes, cls = D["A"], D["nodes"], D["cls"]
    n = len(nodes)
    R = randi.resolvent(A, ALPHA)
    greedy = np.load(ROOT / "results/coarse_trajectory_paths.npz")
    Qs = {"cell classes": randi.class_leverage(A, cls, ALPHA),
          "loss-guided k=9": randi.class_leverage(A, np.unique(greedy["envelope_9"], return_inverse=True)[1], ALPHA),
          "loss-guided k=8": randi.class_leverage(A, np.unique(greedy["envelope_8"], return_inverse=True)[1], ALPHA)}
    Qrand = [randi.class_leverage(A, rng.permutation(cls), ALPHA) for _ in range(args.n_rand)]

    d = randi.load_responses("wildty")
    idx = {m: k for k, m in enumerate(nodes)}
    d = d[d.pre.isin(idx) & d.post.isin(idx)]
    HIT = np.zeros((n, n), bool)
    HIT[d.post.map(idx), d.pre.map(idx)] = True        # HIT[i, j]: i responds to j

    print("building per-recording observation and correlation matrices ...", flush=True)
    ids, OCC, SS, CN = per_recording(nodes)
    print(f"  {len(ids)} recordings", flush=True)
    iu, ju = np.triu_indices(n, 1)

    def dataset(m):
        """Observed pairs and correlation pairs for recording multiplicities m."""
        O = np.tensordot(m, OCC, 1)
        ii, jj = np.nonzero(O)
        y = HIT[ii, jj].astype(float)
        ss, cc = np.tensordot(m, SS, 1), np.tensordot(m, CN, 1)
        distinct = np.tensordot((m > 0).astype(float), CN, 1)
        keep = distinct[iu, ju] >= 3          # recorded together in >= 3 distinct animals
        a, b = iu[keep], ju[keep]
        return dict(i=ii, j=jj, y=y, O=O[ii, jj], ca=a, cb=b, r=ss[a, b] / cc[a, b])

    def statistics(ds, wi=None, wc=None):
        """All statistics on a dataset; wi / wc are integer pair weights (None = 1)."""
        i, j, y, O = ds["i"], ds["j"], ds["y"], ds["O"]
        s = expand(wi) if wi is not None else np.arange(y.size)
        i, j, y, O = i[s], j[s], y[s], O[s]
        ns = A[j, i] == 0
        pR, pD = R[j, i], A[j, i]
        out = {"auc_all": auc(pR, y), "auc_nosyn": auc(pR[ns], y[ns]),
               "auc_exactO": strat_auc(pR, y, O),
               "path_minus_direct": auc(pR, y) - auc(pD, y)}
        pos, neg = pR[ns & (y == 1)], pR[ns & (y == 0)]
        out["median_ratio_nosyn"] = np.median(pos) / np.median(neg)
        a, b, r = ds["ca"], ds["cb"], ds["r"]
        t = expand(wc) if wc is not None else np.arange(r.size)
        a, b, r = a[t], b[t], r[t]
        out["rho_corr"] = rho(R[a, b] + R[b, a], r)
        for name, Q in Qs.items():
            out[f"{name} frac_all"] = auc(Q[j, i], y) / out["auc_all"]
            out[f"{name} frac_nosyn"] = auc(Q[j, i][ns], y[ns]) / out["auc_nosyn"]
            out[f"{name} frac_corr"] = rho(Q[a, b] + Q[b, a], r) / out["rho_corr"]
        rn = np.mean([auc(Q[j, i][ns], y[ns]) for Q in Qrand]) / out["auc_nosyn"]
        rc = np.mean([rho(Q[a, b] + Q[b, a], r) for Q in Qrand]) / out["rho_corr"]
        out["cell classes minus random, nosyn"] = out["cell classes frac_nosyn"] - rn
        out["cell classes minus random, corr"] = out["cell classes frac_corr"] - rc
        return out

    base_m = np.ones(len(ids))
    base = dataset(base_m)
    point = statistics(base)
    ncls = cls.max() + 1
    src_classes = np.unique(cls[base["j"]])
    tgt_classes = np.unique(cls[base["i"]])

    def class_mult(pool):
        c = np.zeros(ncls)
        np.add.at(c, rng.choice(pool, pool.size), 1)
        return c

    rows = []
    for scheme in ("source", "source x target", "recording", "recording + source x target"):
        vals = []
        for b in range(args.n_boot):
            if scheme.startswith("recording"):
                m = np.bincount(rng.integers(0, len(ids), len(ids)), minlength=len(ids)).astype(float)
                ds = dataset(m)
            else:
                ds = base
            if scheme == "source":
                cs = class_mult(src_classes)
                wi = cs[cls[ds["j"]]]
                wc = None
                # the correlation observable keeps its paper scheme: lower-indexed neuron's class
                cc = class_mult(np.unique(cls[ds["ca"]]))
                wc = cc[cls[ds["ca"]]]
            elif scheme.endswith("source x target"):
                cs, ct = class_mult(src_classes), class_mult(tgt_classes)
                wi = cs[cls[ds["j"]]] * ct[cls[ds["i"]]]
                cn = class_mult(np.arange(ncls))
                wc = cn[cls[ds["ca"]]] * cn[cls[ds["cb"]]]
            else:
                wi = wc = None
            vals.append(statistics(ds, wi, wc))
            if (b + 1) % 100 == 0:
                print(f"  {scheme}: {b + 1}/{args.n_boot}", flush=True)
        V = pd.DataFrame(vals)
        for k in point:
            v = V[k].to_numpy(float)
            lo, hi = np.nanpercentile(v, [2.5, 97.5])
            rows.append({"scheme": scheme, "statistic": k, "point": point[k], "lo": lo, "hi": hi,
                         "sd": np.nanstd(v), "p_le_0": np.mean(v <= 0) if "ratio" not in k and "frac" not in k else np.nan,
                         "p_le_1": np.mean(v <= 1) if "ratio" in k else np.nan})
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results/dependence_control.csv", index=False)
    wide = df.pivot(index="statistic", columns="scheme", values=["lo", "hi"])
    print("\n== point estimate and 95% interval under each resampling scheme")
    for k in point:
        line = f"{k:>36} {point[k]:+.3f}"
        for sc in ("source", "source x target", "recording", "recording + source x target"):
            r = df[(df.statistic == k) & (df.scheme == sc)].iloc[0]
            line += f"  | {sc[:14]:>14} [{r.lo:+.3f}, {r.hi:+.3f}]"
        print(line)
    print("\nwrote results/dependence_control.csv")


if __name__ == "__main__":
    main()
