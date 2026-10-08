"""Between-animal agreement of coarse partitions built from each of the eight
Witvliet connectomes, and transfer of those partitions to the adult (Cook)
operator and Randi observables.

Stability. For each Witvliet connectome (166 common cells), partitions are
built from that connectome alone at k = 60, 30, 20, 9, 5 by loss-greedy
agglomeration (`complexity_curve.greedy_path`, shortlist 30, structural
loss) and by k-means on log-scaled operator profiles. Agreement between animals
is the mean pairwise adjusted Rand index (0 for unrelated partitions at any k),
with a permutation reference. Every stability number is repeated on connectomes
thinned to --budget synapses by multinomial resampling (--n-thin replicates),
since merge agreement depends on synapse density. The fine-scale reference is
the partition of 30 disjoint merges (k = 136) chosen by equitability defect from
the shared degree-filtered candidate pool (`prospective_merges.py`,
`cross_animal_transfer.pool_mask`).
Each partition is also compared by ARI with the same method on the Cook
connectome restricted to the shared neurons, and with the a priori subsystems.

Transfer. Each animal's partition at k = 20, 9, 5 is restricted by neuron name
to the Cook neurons it shares and scored on adult observables for pairs with
both endpoints in that set, using block means of the full Cook path operator
over the transferred groups (score as a fraction of the full operator's on the
same pairs). Neurons outside the set are never grouped, so nothing about the
adult connectome enters the partition. Comparators on the same pairs: the same
method run on Cook restricted to the shared neurons, --n-rand permutations of
that Cook partition, and the subsystem partition. A stimulated-class bootstrap
interval is given for the first animal only. The L1 mean averages the first
four animals in `ce.witvliet_files()` order.

Inputs:  Witvliet 2021 connectomes and Cook connectome via conres;
         conres.observables.load().
Outputs: results/coarse_stability.csv
         results/coarse_transfer.csv
Usage:   python scripts/coarse_transfer.py [--n-thin 3] [--budget 1200]
             [--n-rand 20] [--n-boot 300]
"""
from __future__ import annotations

import argparse
import contextlib
import io
import itertools
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ari

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from conres import celegans as ce, observables as ob, randi, subsystems as ss   # noqa: E402
from complexity_curve import greedy_path                                  # noqa: E402
from prospective_merges import pairwise_defect, greedy_disjoint, labels_from_pairs  # noqa: E402
from cross_animal_transfer import pool_mask                               # noqa: E402

ALPHA = 0.1
KS = (60, 30, 20, 9, 5)


def relabel(l):
    return np.unique(l, return_inverse=True)[1]


def build(M: np.ndarray, seed: int = 0) -> dict:
    """{(method, k): labels} from one connectome, activity never used."""
    n = M.shape[0]
    R = randi.resolvent(M, ALPHA)
    off = ~np.eye(n, dtype=bool)
    oj, oi = np.nonzero(off)
    rf = stats.rankdata(R[oj, oi])
    rf = (rf - rf.mean()) / np.linalg.norm(rf - rf.mean())

    def loss(lab):
        rq = stats.rankdata(randi.class_leverage(M, lab, ALPHA)[oj, oi])
        rq -= rq.mean()
        return float(1 - rq @ rf / np.linalg.norm(rq))

    with contextlib.redirect_stdout(io.StringIO()):
        path = greedy_path(M, min(KS), loss_fn=loss, shortlist=30)
    X = np.log1p(np.hstack([R, R.T]) / max(R.max(), 1e-12) * 1e3)
    out = {}
    for k in KS:
        out[("loss-greedy", k)] = relabel(path[k])
        out[("k-means on operator", k)] = relabel(KMeans(k, n_init=10, random_state=seed).fit(X).labels_)
    return out


def thin(M, budget, rng):
    w = M.ravel()
    return rng.multinomial(int(budget), w / w.sum()).reshape(M.shape).astype(float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-thin", type=int, default=3)
    ap.add_argument("--budget", type=int, default=1200)
    ap.add_argument("--n-rand", type=int, default=20)
    ap.add_argument("--n-boot", type=int, default=300)
    args = ap.parse_args()

    wn = ce.witvliet_common_neurons("all")
    files = ce.witvliet_files()
    mats = [ce.witvliet_matrix(p, wn, "all") for _, _, p in files]
    stages = [f"{i}{s[0]}" for i, s, _ in files]

    D = ob.load()
    A, nodes, cls, obs = D["A"], D["nodes"], D["cls"], D["obs"]
    cidx = {x: i for i, x in enumerate(nodes)}
    shared = [x for x in wn if x in cidx]
    wpos = np.array([wn.index(x) for x in shared])          # positions in Witvliet order
    cpos = np.array([cidx[x] for x in shared])              # positions in Cook order
    print(f"{len(wn)} Witvliet cells, {len(shared)} shared with Cook")

    # ------------------------------------------------------------ stability
    raw = [build(M) for M in mats]
    print("  built raw partitions", flush=True)
    rng = np.random.default_rng(0)
    thinned = [[build(thin(M, args.budget, rng), seed=t) for M in mats] for t in range(args.n_thin)]
    print("  built thinned partitions", flush=True)
    cook = build(A[np.ix_(cpos, cpos)])
    sub_lab = ss.partition(nodes, "subsystems")[0][cpos]

    pool = np.flatnonzero(pool_mask(mats))
    allowed = np.zeros(len(wn), bool)
    allowed[pool] = True
    merges = [labels_from_pairs(len(wn), greedy_disjoint(
        (pairwise_defect(M) + pairwise_defect(M.T)) / 2.0, 30, allowed)) for M in mats]
    fine_ari = np.mean([ari(merges[a], merges[b]) for a, b in itertools.combinations(range(8), 2)])

    rows = []
    print(f"\npairwise adjusted Rand index between the eight animals (mean [min, max])")
    print(f"  fine reference, 30 disjoint merges (k = 136): {fine_ari:.3f}")
    for method in ("loss-greedy", "k-means on operator"):
        for k in KS:
            pr = [ari(raw[a][(method, k)], raw[b][(method, k)]) for a, b in itertools.combinations(range(8), 2)]
            pt = [np.mean([ari(th[a][(method, k)], th[b][(method, k)])
                           for a, b in itertools.combinations(range(8), 2)]) for th in thinned]
            to_cook = [ari(raw[a][(method, k)][wpos], cook[(method, k)]) for a in range(8)]
            to_sub = [ari(raw[a][(method, k)][wpos], sub_lab) for a in range(8)]
            rnull = [ari(np.random.default_rng(s).permutation(raw[0][(method, k)]),
                         np.random.default_rng(s + 99).permutation(raw[1][(method, k)])) for s in range(50)]
            rows.append({"method": method, "k": k, "ari_raw": np.mean(pr), "ari_raw_min": np.min(pr),
                         "ari_raw_max": np.max(pr), "ari_thinned": np.mean(pt),
                         "ari_to_cook": np.mean(to_cook), "ari_to_subsystems": np.mean(to_sub),
                         "ari_random": np.mean(rnull)})
            r = rows[-1]
            print(f"  {method:>20} k={k:>3}: raw {r['ari_raw']:.3f} [{r['ari_raw_min']:.3f}, "
                  f"{r['ari_raw_max']:.3f}]  thinned {r['ari_thinned']:.3f}  to Cook {r['ari_to_cook']:.3f}  "
                  f"to subsystems {r['ari_to_subsystems']:.3f}  random {r['ari_random']:+.3f}")
    pd.DataFrame(rows).to_csv(ROOT / "results/coarse_stability.csv", index=False)

    # ------------------------------------------------------------ transfer
    R = randi.resolvent(A, ALPHA)
    Rs = R[np.ix_(cpos, cpos)]
    inset = np.full(len(nodes), -1)
    inset[cpos] = np.arange(len(cpos))
    tobs = {}
    for o, (j, i, v, kd) in obs.items():
        m = (inset[j] >= 0) & (inset[i] >= 0)
        tobs[o] = (inset[j[m]], inset[i[m]], v[m], kd)
    boots = ob.boot_indices(tobs, cls[cpos], args.n_boot)
    full = {o: ob.score(kd, ob.predict(kd, Rs, j, i), v) for o, (j, i, v, kd) in tobs.items()}
    print("\nadult observables on pairs within the shared set: " +
          "  ".join(f"{o} n={len(tobs[o][2]):,} full {full[o]:.4f}" for o in tobs))

    def frac(lab):
        B = ob.block_means(Rs, relabel(lab))
        return {o: ob.score(kd, ob.predict(kd, B, j, i), v) / full[o] for o, (j, i, v, kd) in tobs.items()}

    trows = []
    for method in ("loss-greedy", "k-means on operator"):
        for k in (20, 9, 5):
            arms = {f"{st} (transferred)": raw[a][(method, k)][wpos] for a, st in enumerate(stages)}
            arms["Cook, same neurons"] = cook[(method, k)]
            for name, lab in arms.items():
                trows.append({"method": method, "k": k, "arm": name, **frac(lab)})
            rr = [frac(np.random.default_rng(s).permutation(cook[(method, k)])) for s in range(args.n_rand)]
            trows.append({"method": method, "k": k, "arm": "random",
                          **{o: np.mean([x[o] for x in rr]) for o in tobs}})
            l1 = np.mean([[trows[-10 + a][o] for o in tobs] for a in range(4)], axis=0)
            # bootstrap interval for the first L1 animal's partition
            B1 = ob.block_means(Rs, relabel(raw[0][(method, k)][wpos]))
            ci = {}
            for o, (j, i, v, kd) in tobs.items():
                pf, pb = ob.predict(kd, Rs, j, i), ob.predict(kd, B1, j, i)
                fr = [ob.score(kd, pb, v, s) / ob.score(kd, pf, v, s) for s in boots[o]]
                ci[o] = np.nanpercentile(fr, [2.5, 97.5])
            print(f"\n  {method}, k = {k}: fraction of full adult operator's performance")
            print(f"    {'arm':>22} " + " ".join(f"{o:>16}" for o in tobs))
            for r in trows[-10:]:
                print(f"    {r['arm']:>22} " + " ".join(f"{r[o]:>16.2f}" for o in tobs))
            print(f"    {'L1 mean (animals 1-4)':>22} " + " ".join(f"{x:>16.2f}" for x in l1))
            print(f"    {'1L 95% CI':>22} " + " ".join(f"{'[%.2f, %.2f]' % tuple(ci[o]):>16}" for o in tobs))
    sub = {"method": "anatomy", "k": int(sub_lab.max() + 1), "arm": "subsystems (a priori)", **frac(sub_lab)}
    trows.append(sub)
    print(f"\n  subsystems (a priori) on the shared set: " +
          "  ".join(f"{o} {sub[o]:.2f}" for o in tobs))
    pd.DataFrame(trows).to_csv(ROOT / "results/coarse_transfer.csv", index=False)
    print("\nwrote results/coarse_stability.csv, results/coarse_transfer.csv")


if __name__ == "__main__":
    main()
