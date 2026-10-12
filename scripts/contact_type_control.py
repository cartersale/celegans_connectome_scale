"""Score the path operator built from chemical synapses only, from gap junctions
only, and from both, on the same node set and the same pairs.

For each contact type the Cook 2019 edge list is restricted to that type and
placed on the 302-node set of the summed matrix (conres.celegans.layer_matrix;
gap junctions symmetrised with max(A, A^T), as in build_matrix). Nodes with no
contact of the type keep empty rows and columns. For each matrix M:

  direct        M[stimulated, responder], symmetrised for correlation
  path          randi.resolvent(M, alpha = 0.1)
  class-level   randi.class_leverage(M, cell classes, 0.1)
  random        the same on --n-rand label permutations of the cell classes

are scored on three observables from conres.observables.load: response
incidence over all observed pairs, incidence over unsynapsed pairs, and
activity correlation. Unsynapsed pairs are defined from the summed matrix for
every contact type, so the pair sets do not change between rows. Scores are
performance above chance (AUC - 0.5, or Spearman rho); `kept` is the
class-level or random score as a fraction of the path score of the same matrix.

Intervals are 95% cluster bootstraps over the class of the stimulated neuron
(observables.boot_indices, --n-boot resamples, seed 0); differences and ratios
are formed within each resample.

Inputs: Cook 2019 edge list (conres.celegans), Randi 2023 responses, observation
    counts and traces (conres.observables.load).
Outputs: results/contact_type_control.csv (contact, observable, quantity,
    point, lo, hi) and results/contact_type_control.log.
Usage: python scripts/contact_type_control.py [--n-boot 500] [--n-rand 200] [--seed 0]
"""
from __future__ import annotations

import argparse
import csv
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from conres import celegans as ce, observables as ob, randi   # noqa: E402

ALPHA = 0.1
OBS = ("incidence_all", "incidence_nosyn", "correlation")
CONTACTS = ("all", "chemical", "electrical")


def matrix_on(nodes: list[str], kind: str) -> np.ndarray:
    """Adjacency of one contact type on a fixed node list; gap junctions symmetrised."""
    if kind == "all":
        return ce.layer_matrix(nodes, "chemical") + ce.layer_matrix(nodes, "electrical")
    A = ce.layer_matrix(nodes, kind)
    return np.maximum(A, A.T) if kind == "electrical" else A


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for s in self.streams:
            s.write(text)

    def flush(self):
        for s in self.streams:
            s.flush()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--n-rand", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    sys.stdout = _Tee(sys.__stdout__, (ROOT / "results/contact_type_control.log").open("w"))

    D = ob.load()
    A_all, nodes, cls, obs = D["A"], D["nodes"], D["cls"], D["obs"]
    boots = ob.boot_indices({o: obs[o] for o in OBS}, cls, args.n_boot, args.seed)
    rng = np.random.default_rng(args.seed)
    perms = [rng.permutation(cls) for _ in range(args.n_rand)]

    rows = []

    def add(contact, o, quantity, point, draws=None):
        lo, hi = (np.nanpercentile(draws, [2.5, 97.5]) if draws is not None else (np.nan, np.nan))
        rows.append({"contact": contact, "observable": o, "quantity": quantity,
                     "point": point, "lo": lo, "hi": hi})
        return point, lo, hi

    for contact in CONTACTS:
        M = matrix_on(nodes, contact)
        assert contact != "all" or np.array_equal(M, A_all)
        R = randi.resolvent(M, ALPHA)
        Q = randi.class_leverage(M, cls, ALPHA)
        Qr = [randi.class_leverage(M, p, ALPHA) for p in perms]
        print(f"\n=== {contact}: {int((M > 0).sum()):,} connections, total weight {M.sum():,.0f}, "
              f"{int(((M.sum(0) + M.sum(1)) > 0).sum())} of {len(nodes)} nodes connected ===")
        print(f"{'observable':>16} {'n':>7} | {'direct':>22} | {'path':>22} | "
              f"{'path - direct':>24} | {'classes kept':>20} | {'random kept':>12}")
        for o in OBS:
            j, i, v, kind = obs[o]
            sc = lambda S, sel=slice(None): ob.score(kind, ob.predict(kind, S, j, i), v, sel)
            b = boots[o]
            d0, p0, q0 = sc(M), sc(R), sc(Q)
            db = np.array([sc(M, s) for s in b])
            pb = np.array([sc(R, s) for s in b])
            qb = np.array([sc(Q, s) for s in b])
            r0 = np.array([sc(S) for S in Qr])
            d = add(contact, o, "direct", d0, db)
            p = add(contact, o, "path", p0, pb)
            g = add(contact, o, "path_minus_direct", p0 - d0, pb - db)
            k = add(contact, o, "classes_kept", q0 / p0, qb / pb)
            add(contact, o, "random_kept_mean", r0.mean() / p0)
            add(contact, o, "random_kept_sd", r0.std() / abs(p0))
            f = lambda t: f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]"
            print(f"{o:>16} {len(v):>7,} | {f(d):>22} | {f(p):>22} | {f(g):>24} | "
                  f"{k[0]:.2f} [{k[1]:.2f}, {k[2]:.2f}]".ljust(118)
                  + f" | {r0.mean() / p0:.2f} +- {r0.std() / abs(p0):.2f}")

    out = ROOT / "results/contact_type_control.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
