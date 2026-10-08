"""Correlate per-class structural dispersion with the similarity of measured
response profiles between co-class neurons that were both stimulated.

Structural dispersion is the within-class RMS spread of out-flow profiles
(rows of A @ H, A = all-synapse matrix, H = class indicator), scaled by the RMS
of A @ H and divided by its mean over 200 label permutations (size-matched null).

For each pair of stimulated co-class neurons, the comparison is restricted to
responders observed for both members (occ3 > 0 in both columns, the two members
themselves excluded); pairs with fewer than --min-shared such responders are
dropped. Response vectors hold the detected magnitude, zero where the pair was
observed and no response was detected. Similarity is reported as Jaccard,
Spearman, cosine and Cohen's kappa on detection; kappa is primary because it
corrects for chance overlap given the two detection rates, which Jaccard does not.

Tests:
  - within-class vs between-class similarity (Mann-Whitney), with up to 400
    random between-class stimulated pairs subject to the same --min-resp filter;
  - class-level Spearman of dispersion vs mean similarity, plus a partial
    Spearman after rank-residualising both on mean exposure and response rate;
  - pair-level Spearman with a 2,000-draw cluster bootstrap over classes.

Inputs: C. elegans connectome (conres.celegans), Randi 2023 wild-type responses
and observation counts (conres.randi).
Outputs: results/exchangeability.csv (one row per within-class pair).
Usage: exchangeability.py [--min-shared 20] [--min-resp 0]
       [--seed 0]   (--n-perm is accepted but unused)
"""
from __future__ import annotations

import argparse
import collections
import csv
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy import stats

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from conres import celegans as ce, randi          # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def per_class_dispersion(A, labels, n_null=200, seed=0):
    """Within-class dispersion of out-flow profiles as a ratio to the mean over
    `n_null` label permutations. Lower means members are structurally more alike.
    Returned in the order of np.unique(labels)."""
    n = A.shape[0]

    def disp(lab):
        used, li = np.unique(lab, return_inverse=True)
        H = np.zeros((n, used.size))
        H[np.arange(n), li] = 1.0
        S = A @ H
        scale = np.sqrt((S ** 2).mean())
        out = np.full(used.size, np.nan)
        for c in range(used.size):
            rows = S[li == c]
            if len(rows) < 2:
                continue
            out[c] = np.sqrt(((rows - rows.mean(0)) ** 2).sum(1).mean()) / scale
        return out

    real = disp(labels)
    rng = np.random.default_rng(seed)
    null = np.array([disp(rng.permutation(labels)) for _ in range(n_null)])
    return real / np.nanmean(null, axis=0)


def profile_similarity(v1, v2):
    """Return (Jaccard, Spearman, cosine, Cohen's kappa) for two response vectors.

    Jaccard and kappa use detection (v > 0). With sparse detection, two neurons
    usually share zero detected responders by chance, which Jaccard cannot
    distinguish from dissimilarity; kappa corrects for chance agreement given the
    two marginal detection rates.
    """
    d1, d2 = v1 > 0, v2 > 0
    union = (d1 | d2).sum()
    jac = (d1 & d2).sum() / union if union else np.nan
    sp = stats.spearmanr(v1, v2).statistic if (v1.std() > 0 and v2.std() > 0) else np.nan
    nz = np.linalg.norm(v1) * np.linalg.norm(v2)
    cos = float(v1 @ v2 / nz) if nz > 0 else np.nan
    n = d1.size
    po = (d1 == d2).mean()
    p1, p2 = d1.mean(), d2.mean()
    pe = p1 * p2 + (1 - p1) * (1 - p2)
    kap = (po - pe) / (1 - pe) if pe < 1 else np.nan
    return jac, sp, cos, kap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-shared", type=int, default=20)
    ap.add_argument("--n-perm", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-resp", type=int, default=0,
                    help="require BOTH members to drive at least this many detected "
                         "responses among shared responders; below it the per-class "
                         "similarity is not estimable and only adds noise")
    args = ap.parse_args()

    A, nodes = ce.build_matrix("all", neurons_only=True)
    labels = ce.partitions(nodes)["class"]
    mapping = ce.assign_classes(nodes)
    idx = {n: i for i, n in enumerate(nodes)}
    occ3, _ = randi.build_occ3(nodes, "wt")

    d = randi.load_responses("wildty")
    d = d[d.pre.isin(idx) & d.post.isin(idx)]
    mag = np.zeros_like(occ3)
    for p, q, mv in zip(d.pre, d.post, d["mag"]):
        mag[idx[q], idx[p]] = mv            # [responder, stimulated]

    disp = per_class_dispersion(A, labels, seed=args.seed)
    class_names = sorted(set(mapping[n] for n in nodes))
    disp_by_name = {class_names[c]: disp[c] for c in range(len(class_names))}

    stim = [j for j in range(len(nodes)) if occ3[:, j].sum() > 0]
    by_class = collections.defaultdict(list)
    for j in stim:
        by_class[mapping[nodes[j]]].append(j)

    def compare(j1, j2):
        S = np.flatnonzero((occ3[:, j1] > 0) & (occ3[:, j2] > 0))
        S = S[(S != j1) & (S != j2)]
        if S.size < args.min_shared:
            return None
        v1, v2 = mag[S, j1], mag[S, j2]
        jac, sp, cos, kap = profile_similarity(v1, v2)
        expo = float(np.sqrt(occ3[S, j1].mean() * occ3[S, j2].mean()))
        rate = float(((v1 > 0).mean() + (v2 > 0).mean()) / 2)
        return dict(n_shared=int(S.size), jaccard=jac, spearman=sp, cosine=cos,
                    kappa=kap, n_resp=int((v1 > 0).sum() + (v2 > 0).sum()),
                    min_resp=int(min((v1 > 0).sum(), (v2 > 0).sum())),
                    exposure=expo, resp_rate=rate)

    rows = []
    for cname, members in by_class.items():
        if len(members) < 2 or not np.isfinite(disp_by_name.get(cname, np.nan)):
            continue
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                r = compare(members[a], members[b])
                if r and r["min_resp"] >= args.min_resp:
                    rows.append({"cls": cname, "size": len(members),
                                 "dispersion": disp_by_name[cname],
                                 "m1": nodes[members[a]], "m2": nodes[members[b]], **r})
    print(f"within-class stimulated pairs usable: {len(rows)} "
          f"across {len({r['cls'] for r in rows})} classes")

    # between-class baseline: random pairs of stimulated neurons
    rng = np.random.default_rng(args.seed)
    btw = []
    tries = 0
    while len(btw) < 400 and tries < 20000:
        tries += 1
        j1, j2 = rng.choice(stim, 2, replace=False)
        if mapping[nodes[j1]] == mapping[nodes[j2]]:
            continue
        r = compare(j1, j2)
        # the --min-resp filter applies here too: kappa rises with the number of
        # responses for any pair, so filtering one arm only biases the contrast
        if r and r["min_resp"] >= args.min_resp:
            btw.append(r)
    for key in ("jaccard", "spearman", "cosine", "kappa"):
        w = np.array([r[key] for r in rows], float)
        b = np.array([r[key] for r in btw], float)
        u = stats.mannwhitneyu(w[np.isfinite(w)], b[np.isfinite(b)])
        print(f"  {key:>9}: within-class {np.nanmean(w):.4f}   "
              f"between-class {np.nanmean(b):.4f}   Mann-Whitney p={u.pvalue:.2e}")

    print(f"\n=== gradient: does structural dispersion predict response similarity? ===")
    cl = collections.defaultdict(list)
    for r in rows:
        cl[r["cls"]].append(r)
    cls = sorted(cl)
    x = np.array([cl[c][0]["dispersion"] for c in cls])
    print(f"  {len(cls)} classes; dispersion range {x.min():.2f}-{x.max():.2f}")
    print(f"  {'metric':>9} {'rho':>7} {'p':>9} {'partial rho (expo, rate)':>26}")
    out_rows = []
    for key in ("jaccard", "spearman", "cosine", "kappa"):
        y = np.array([np.nanmean([r[key] for r in cl[c]]) for c in cls])
        e = np.array([np.nanmean([r["exposure"] for r in cl[c]]) for c in cls])
        q = np.array([np.nanmean([r["resp_rate"] for r in cl[c]]) for c in cls])
        ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(e) & np.isfinite(q)
        rho, p = stats.spearmanr(x[ok], y[ok])
        # partial: rank-residualise y and x on exposure and response rate
        def resid(v):
            Z = np.column_stack([stats.rankdata(e[ok]), stats.rankdata(q[ok]),
                                 np.ones(ok.sum())])
            beta, *_ = np.linalg.lstsq(Z, stats.rankdata(v), rcond=None)
            return stats.rankdata(v) - Z @ beta
        pr, pp = stats.spearmanr(resid(x[ok]), resid(y[ok]))
        print(f"  {key:>9} {rho:>+7.3f} {p:>9.4f}   {pr:>+10.3f}  p={pp:.4f}")
        out_rows.append({"metric": key, "n_classes": int(ok.sum()), "rho": rho,
                         "p": p, "partial_rho": pr, "partial_p": pp})

    # pair level; the bootstrap resamples whole classes because pairs within a
    # class share one dispersion value
    print(f"\n  pair level ({len(rows)} pairs), cluster bootstrap by class:")
    xs = np.array([r["dispersion"] for r in rows])
    for key in ("kappa", "cosine"):
        ys = np.array([r[key] for r in rows], float)
        ok = np.isfinite(xs) & np.isfinite(ys)
        rho = stats.spearmanr(xs[ok], ys[ok]).statistic
        groups = np.array([r["cls"] for r in rows])[ok]
        uq = np.unique(groups)
        where = {g: np.flatnonzero(groups == g) for g in uq}
        bs = []
        for _ in range(2000):
            pick = np.concatenate([where[g] for g in rng.choice(uq, uq.size, replace=True)])
            bs.append(stats.spearmanr(xs[ok][pick], ys[ok][pick]).statistic)
        bs = np.array(bs)
        print(f"  {key:>9} rho {rho:+.3f} [{np.percentile(bs,2.5):+.3f}, "
              f"{np.percentile(bs,97.5):+.3f}]")

    print("\n  most structurally exchangeable classes with both members stimulated:")
    for c in sorted(cls, key=lambda c: cl[c][0]["dispersion"])[:6]:
        r = cl[c][0]
        print(f"    {c:>5} disp {r['dispersion']:.2f}  kappa {np.nanmean([q['kappa'] for q in cl[c]]):+.3f}"
              f"  cosine {np.nanmean([q['cosine'] for q in cl[c]]):.3f}"
              f"  min_resp {min(q['min_resp'] for q in cl[c])}")
    print("  least:")
    for c in sorted(cls, key=lambda c: -cl[c][0]["dispersion"])[:6]:
        r = cl[c][0]
        print(f"    {c:>5} disp {r['dispersion']:.2f}  kappa {np.nanmean([q['kappa'] for q in cl[c]]):+.3f}"
              f"  cosine {np.nanmean([q['cosine'] for q in cl[c]]):.3f}"
              f"  min_resp {min(q['min_resp'] for q in cl[c])}")

    f = ROOT / "results/exchangeability.csv"
    with f.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {f}")


if __name__ == "__main__":
    main()
