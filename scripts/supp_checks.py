"""Lineage-only cell classes and the calibration of gamma.

1. Lineage-only cell classes: left/right homolog pairs (name stem shared with an
   L/R partner present in the node set) and the eight serially repeated
   ventral-cord families VENTRAL_CORD (numbered members of VA, VB, VC, VD, DA,
   DB, DD, AS), every other neuron in its own class. Left/right pairs alone are
   also scored. Each partition, and the cell classes, is scored with the
   class-level path operator (alpha = 0.1) on every observable as a fraction of
   the full operator's above-chance performance, with --n-boot cluster-bootstrap
   intervals (stimulated-neuron classes), against --n-rand size-matched random
   partitions (label permutations). Every lineage-only class lies within one
   cell class.

2. Calibration of gamma (conres.equitability.gamma, averaged over A and A^T, i.e.
   outgoing and incoming). Reference matrices at the same n and cell-class size
   distribution, --n-ref draws each: planted block structure H B H^T with
   B ~ Exponential(mean positive connectome weight) and zero diagonal
   (gamma = 0 by construction); the same with multiplicative within-class noise
   P * (1 + 0.5 N(0,1)); independent standard Gaussian entries; and the
   connectome with its off-diagonal entries shuffled. Also 500 size-matched
   random partitions of the real connectome.

Produces the Supplementary Information sections and tables on lineage-only cell
classes and on the calibration of gamma.

Inputs: conres.observables.load, conres.equitability.gamma.
Outputs: results/supp_lineage_only.csv, results/supp_gamma_calibration.csv.
Usage: python scripts/supp_checks.py [--n-rand 200] [--n-boot 500] [--n-ref 50]
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from conres import observables as ob, randi   # noqa: E402
from conres.equitability import gamma   # noqa: E402

ALPHA = 0.1
VENTRAL_CORD = ("VA", "VB", "VC", "VD", "DA", "DB", "DD", "AS")


def lr_key(n, names):
    """Stem shared with a left/right homolog present in the node set, else None."""
    if n[-1:] in ("L", "R") and n[:-1] + {"L": "R", "R": "L"}[n[-1]] in names:
        return n[:-1]
    return None


def lineage_only(nodes):
    """Labels: ventral-cord family for its numbered members, else the left/right
    stem, else a singleton."""
    names = set(nodes)
    keys = []
    for n in nodes:
        stem = n.rstrip("0123456789")
        if stem in VENTRAL_CORD and stem != n:
            keys.append(stem)
        else:
            keys.append(lr_key(n, names) or f"__{n}")
    return np.unique(keys, return_inverse=True)[1]


def bilateral_only(nodes):
    names = set(nodes)
    return np.unique([lr_key(n, names) or f"__{n}" for n in nodes], return_inverse=True)[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-rand", type=int, default=200)
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--n-ref", type=int, default=50)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    D = ob.load()
    A, nodes, cls, obs = D["A"], D["nodes"], D["cls"], D["obs"]
    R = randi.resolvent(A, ALPHA)
    boots = ob.boot_indices(obs, cls, args.n_boot)
    pf = {o: ob.predict(kd, R, j, i) for o, (j, i, v, kd) in obs.items()}
    full = {o: ob.score(kd, pf[o], v) for o, (j, i, v, kd) in obs.items()}

    # ----------------------------------------------------- 1. lineage-only classes
    parts = {"cell classes": cls, "lineage-only cell classes": lineage_only(nodes),
             "left/right pairs only": bilateral_only(nodes)}
    rows = []
    for pname, lab in parts.items():
        k = int(lab.max() + 1)
        Q = randi.class_leverage(A, lab, ALPHA)
        rnd_Q = [randi.class_leverage(A, rng.permutation(lab), ALPHA) for _ in range(args.n_rand)]
        for o, (j, i, v, kd) in obs.items():
            pq = ob.predict(kd, Q, j, i)
            sq = ob.score(kd, pq, v)
            fb = np.array([ob.score(kd, pq, v, s) / ob.score(kd, pf[o], v, s) for s in boots[o]])
            rs = np.array([ob.score(kd, ob.predict(kd, M, j, i), v) for M in rnd_Q])
            rows.append({"partition": pname, "k": k, "reduction": len(nodes) / k, "observable": o,
                         "full": full[o], "quotient": sq, "frac": sq / full[o],
                         "frac_lo": np.nanpercentile(fb, 2.5), "frac_hi": np.nanpercentile(fb, 97.5),
                         "random_frac": rs.mean() / full[o],
                         "random_frac_lo": np.percentile(rs, 2.5) / full[o],
                         "random_frac_hi": np.percentile(rs, 97.5) / full[o],
                         "z_vs_random": (sq - rs.mean()) / rs.std()})
        print(f"  scored {pname} (k={k})", flush=True)
    lin = pd.DataFrame(rows)
    lin.to_csv(ROOT / "results/supp_lineage_only.csv", index=False)
    print("\n== lineage-only cell classes: class-level path operator, fraction of full "
          f"operator's performance above chance ({args.n_rand} size-matched random partitions)")
    for _, r in lin.iterrows():
        print(f"  {r.partition:>26} k={r.k:>3} ({r.reduction:.2f}-fold)  {r.observable:>16}  "
              f"{r.frac:.2f} [{r.frac_lo:.2f}, {r.frac_hi:.2f}]  random {r.random_frac:.2f} "
              f"[{r.random_frac_lo:.2f}, {r.random_frac_hi:.2f}]  z {r.z_vs_random:+.1f}")

    # --------------------------------------------------------- 2. gamma calibration
    n = len(nodes)
    k = int(cls.max() + 1)
    H = np.zeros((n, k))
    H[np.arange(n), cls] = 1.0
    scale = A[A > 0].mean()
    ref = {"planted, no noise": [], "planted, 50% within-class noise": [],
           "structureless, Gaussian": [], "structureless, entry-shuffled connectome": []}
    off = ~np.eye(n, dtype=bool)
    for _ in range(args.n_ref):
        B = rng.exponential(scale, (k, k))
        P = H @ B @ H.T
        np.fill_diagonal(P, 0)
        N = P * (1 + 0.5 * rng.standard_normal((n, n)))
        np.fill_diagonal(N, 0)
        G = rng.standard_normal((n, n))
        np.fill_diagonal(G, 0)
        S = np.zeros_like(A)
        S[off] = rng.permutation(A[off])
        for key, M in zip(ref, (P, N, G, S)):
            ref[key].append((gamma(M, cls) + gamma(M.T, cls)) / 2)
    null = np.array([(gamma(A, p) + gamma(A.T, p)) / 2
                     for p in (rng.permutation(cls) for _ in range(500))])
    g_out, g_in = gamma(A, cls), gamma(A.T, cls)
    g = (g_out + g_in) / 2
    cal = [{"matrix": key, "gamma_mean": np.mean(v), "gamma_sd": np.std(v)} for key, v in ref.items()]
    cal += [{"matrix": "connectome, size-matched random partitions", "gamma_mean": null.mean(),
             "gamma_sd": null.std()},
            {"matrix": "connectome, cell classes", "gamma_mean": g, "gamma_sd": 0.0,
             "gamma_out": g_out, "gamma_in": g_in, "z_vs_random": (g - null.mean()) / null.std()}]
    cal = pd.DataFrame(cal)
    cal.to_csv(ROOT / "results/supp_gamma_calibration.csv", index=False)
    print(f"\n== gamma calibration at n={n}, cell-class sizes (k={k}); mean of outgoing and "
          f"incoming; {args.n_ref} reference draws")
    print(cal.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nwrote results/supp_lineage_only.csv, results/supp_gamma_calibration.csv")


if __name__ == "__main__":
    main()
