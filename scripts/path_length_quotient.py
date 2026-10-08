"""Compare class and bilateral quotients with size-matched random partitions
for path operators of increasing length: A^1, A^2, A^3, A^4 and the resolvent R.

Each operator is built on the quotient matrix and read back at neuron pairs,
Op(Q)[g(i), g(j)], as conres.randi.class_leverage does for R. Powers use the
matrix divided by its own spectral radius; R uses alpha = 0.1. Losses per
operator and partition:

  struct_occ   1 - Spearman(read-back, full) over occ3-observed ordered pairs
  struct_all   1 - Spearman(read-back, full) over all off-diagonal ordered pairs
  cos_all      1 - cosine(read-back, full) over all off-diagonal ordered pairs
  func_all     AUC(full) - AUC(read-back) for Randi 2023 response incidence
  func_nosyn   the same on observed pairs without a direct synapse

and Delta = L_random - L_partition, with L_random the mean over --n-rand label
permutations. struct_occ is on the same pairs as the AUCs; struct_all is on the
pair set used for the fly and is not to be set beside the AUC. A^1 is mostly
exact zeros, so Spearman on it has a dominant tie block; the largest tie block
of each full operator is printed, and the tie-free cosine loss is carried so
that a step between A^1 and A^2 seen only in the rank loss can be recognised as
a tie artefact.

Partitions: cell classes, and bilateral homologue pairs (lineage-defined,
independent of wiring). Intervals on Delta and on the step in Delta between
consecutive operators come from a cluster bootstrap (--n-boot draws) over the
class of the stimulated neuron; the first --n-rand-boot nulls are re-scored in
every resample, and structural and functional Deltas share each resample.

Inputs: C. elegans connectome (conres.celegans), Randi 2023 wild-type responses
and observation counts (conres.randi).
Outputs: results/path_length_quotient.csv (partition, operator, metric,
bio, rand_mean, rand_sd, delta, z, headroom) and
results/path_length_bootstrap.csv (95% intervals per contrast).
Usage: path_length_quotient.py [--n-rand 200] [--n-boot 200]
       [--n-rand-boot 20] [--seed 0]
"""
from __future__ import annotations

import argparse
import csv
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from conres import celegans as ce, randi                  # noqa: E402
from conres.metric_guard import guard_predictor           # noqa: E402

ALPHA = 0.1
OPERATORS = ("A1", "A2", "A3", "A4", "R")


def operator(M: np.ndarray, name: str) -> np.ndarray:
    """A power of the spectrally normalised matrix, or the resolvent.

    Rank, cosine and AUC are invariant to overall scale, so the normalisation
    only keeps high powers numerically tame."""
    if name == "R":
        return randi.resolvent(M, ALPHA)
    rho = np.max(np.abs(np.linalg.eigvals(M)))
    N = M / rho if rho > 0 else M
    return np.linalg.matrix_power(N, int(name[1:]))


def readback(A: np.ndarray, lab: np.ndarray, name: str) -> np.ndarray:
    return operator(randi.quotient(A, lab), name)[np.ix_(lab, lab)]


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return float(stats.spearmanr(x, y).statistic)


def cosine(x: np.ndarray, y: np.ndarray) -> float:
    return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-rand", type=int, default=200)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--n-rand-boot", type=int, default=20,
                    help="nulls re-scored inside each bootstrap resample")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    A, nodes = ce.build_matrix("all", neurons_only=True)
    n = A.shape[0]
    idx = {x: i for i, x in enumerate(nodes)}
    parts = ce.partitions(nodes)
    parts = {k: np.unique(v, return_inverse=True)[1] for k, v in parts.items()}

    occ3, _ = randi.build_occ3(nodes, "wt")
    d = randi.load_responses("wildty")
    d = d[d.pre.isin(idx) & d.post.isin(idx)]
    hits = {(idx[p], idx[q]) for p, q in zip(d.pre, d.post)}
    ii, jj = np.nonzero(occ3)
    y = np.array([1.0 if (j, i) in hits else 0.0 for i, j in zip(ii, jj)])
    nosyn = A[jj, ii] == 0
    off = ~np.eye(n, dtype=bool)
    aj, ai = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    oj, oi = aj[off], ai[off]

    full = {o: operator(A, o) for o in OPERATORS}
    print(f"worm: {n} neurons; occ3 pairs {len(y):,} ({int(y.sum()):,} propagating), "
          f"no-synapse {int(nosyn.sum()):,}; all ordered pairs {len(oj):,}")
    for name, lab in parts.items():
        print(f"partition '{name}': k = {lab.max() + 1}")

    print(f"\n{'op':>3} {'tie block (all)':>16} {'tie block (occ3)':>17} "
          f"{'AUC all':>8} {'AUC no-syn':>11}")
    auc0, auc0_ns = {}, {}
    for o in OPERATORS:
        F = full[o]
        ga = guard_predictor(np.zeros(len(oj)), F[oj, oi])
        go = guard_predictor(y, F[jj, ii])
        auc0[o] = roc_auc_score(y, F[jj, ii])
        auc0_ns[o] = roc_auc_score(y[nosyn], F[jj, ii][nosyn])
        print(f"{o:>3} {ga['largest_tie_block']:>16.3f} {go['largest_tie_block']:>17.3f} "
              f"{auc0[o]:>8.4f} {auc0_ns[o]:>11.4f}")

    def score(Q: np.ndarray, o: str) -> dict:
        F = full[o]
        qa, fa = Q[oj, oi], F[oj, oi]
        qo = Q[jj, ii]
        return {
            "struct_occ": 1 - spearman(qo, F[jj, ii]),
            "struct_all": 1 - spearman(qa, fa),
            "cos_all": 1 - cosine(qa, fa),
            "func_all": auc0[o] - roc_auc_score(y, qo),
            "func_nosyn": auc0_ns[o] - roc_auc_score(y[nosyn], qo[nosyn]),
        }

    rows = []
    boot_rows = []
    keys = ("struct_occ", "struct_all", "cos_all", "func_all", "func_nosyn")
    for pname, lab in parts.items():
        rng = np.random.default_rng(args.seed)
        nulls = [rng.permutation(lab) for _ in range(args.n_rand)]
        bio_Q = {o: readback(A, lab, o) for o in OPERATORS}
        print(f"\n=== {pname} (k = {lab.max() + 1}), {args.n_rand} size-matched nulls ===")
        print(f"{'op':>3} | {'struct occ3: bio / rand / Delta':>34} | "
              f"{'cos all: bio / rand':>22} | {'AUC all: bio / rand / Delta':>33} | "
              f"{'AUC no-syn: bio / rand / Delta':>33} | {'struct all: bio / rand':>24}")
        for o in OPERATORS:
            b = score(bio_Q[o], o)
            r = [score(readback(A, nl, o), o) for nl in nulls]
            rm = {k: float(np.mean([x[k] for x in r])) for k in keys}
            rs = {k: float(np.std([x[k] for x in r], ddof=1)) for k in keys}
            for k in keys:
                rows.append({"partition": pname, "operator": o, "metric": k,
                             "bio": b[k], "rand_mean": rm[k], "rand_sd": rs[k],
                             "delta": rm[k] - b[k],
                             "z": (b[k] - rm[k]) / rs[k] if rs[k] > 0 else np.nan,
                             "headroom": (auc0_ns[o] - 0.5) if k == "func_nosyn"
                             else (auc0[o] - 0.5) if k == "func_all" else np.nan})
            print(f"{o:>3} | {b['struct_occ']:>10.4f} {rm['struct_occ']:>10.4f} "
                  f"{rm['struct_occ'] - b['struct_occ']:>+12.4f} | "
                  f"{b['cos_all']:>10.4f} {rm['cos_all']:>11.4f} | "
                  f"{b['func_all']:>+10.4f} {rm['func_all']:>+10.4f} "
                  f"{rm['func_all'] - b['func_all']:>+11.4f} | "
                  f"{b['func_nosyn']:>+10.4f} {rm['func_nosyn']:>+10.4f} "
                  f"{rm['func_nosyn'] - b['func_nosyn']:>+11.4f} | "
                  f"{b['struct_all']:>11.4f} {rm['struct_all']:>12.4f}")

        # Bootstrap units are stimulated-neuron classes. The nulls inside the
        # bootstrap are a fixed subset, so the interval reflects pair sampling
        # rather than null-draw variation.
        rng_b = np.random.default_rng(args.seed + 1)
        g = parts["class"][jj]
        uq = np.unique(g)
        where = {c: np.flatnonzero(g == c) for c in uq}
        sub_nulls = nulls[:args.n_rand_boot]
        vals = {o: {"bio": bio_Q[o][jj, ii],
                    "rand": [readback(A, nl, o)[jj, ii] for nl in sub_nulls],
                    "full": full[o][jj, ii]} for o in OPERATORS}
        draws = []
        for _ in range(args.n_boot):
            s = np.concatenate([where[c] for c in rng_b.choice(uq, uq.size, replace=True)])
            ns = s[nosyn[s]]
            if len(np.unique(y[ns])) < 2:
                continue
            rec = {}
            for o in OPERATORS:
                v = vals[o]
                d_func = []
                for m in (ns, s):
                    a_full = roc_auc_score(y[m], v["full"][m])
                    fb = a_full - roc_auc_score(y[m], v["bio"][m])
                    fr = np.mean([a_full - roc_auc_score(y[m], q[m]) for q in v["rand"]])
                    d_func.append(fr - fb)
                sb = 1 - spearman(v["bio"][s], v["full"][s])
                sr = np.mean([1 - spearman(q[s], v["full"][s]) for q in v["rand"]])
                rec[o] = (d_func[0], sr - sb, d_func[1])
            draws.append(rec)
        print(f"\n  95% cluster-bootstrap intervals ({len(draws)} resamples over "
              f"{uq.size} stimulated classes; {len(sub_nulls)} nulls per resample)")
        print(f"  {'':>10} {'Delta functional (all)':>26} {'Delta functional (no-syn)':>28} "
              f"{'Delta structural (occ3)':>26}")
        ci = lambda v: (np.percentile(v, 2.5), np.percentile(v, 97.5))
        steps = [(o, lambda dr, k, o=o: dr[o][k]) for o in OPERATORS]
        steps += [(f"{a_}->{b_}", lambda dr, k, a_=a_, b_=b_: dr[b_][k] - dr[a_][k])
                  for a_, b_ in zip(OPERATORS[:-1], OPERATORS[1:])]
        for lab_, get in steps:
            fa = ci([get(dr, 2) for dr in draws])
            fn = ci([get(dr, 0) for dr in draws])
            st = ci([get(dr, 1) for dr in draws])
            print(f"  {lab_:>10} [{fa[0]:+.4f}, {fa[1]:+.4f}]       "
                  f"[{fn[0]:+.4f}, {fn[1]:+.4f}]         [{st[0]:+.4f}, {st[1]:+.4f}]")
            boot_rows.append({"partition": pname, "contrast": lab_,
                              "func_all_lo": fa[0], "func_all_hi": fa[1],
                              "func_nosyn_lo": fn[0], "func_nosyn_hi": fn[1],
                              "struct_occ_lo": st[0], "struct_occ_hi": st[1]})

    out = ROOT / "results/path_length_quotient.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    out_b = ROOT / "results/path_length_bootstrap.csv"
    with out_b.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(boot_rows[0].keys()))
        w.writeheader()
        w.writerows(boot_rows)
    print(f"\nwrote {out}\nwrote {out_b}")


if __name__ == "__main__":
    main()
