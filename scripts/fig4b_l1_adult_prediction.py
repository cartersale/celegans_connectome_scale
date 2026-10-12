"""Score merge sets inferred in L1, adult and thinned-adult connectomes by the
adult prediction of signal propagation they remove, and draw every replicate.

Every arm is a partition of the adult Cook 2019 connectome into the same number
of classes (M = 70 disjoint pairs among the Cook neurons shared with the
Witvliet 2021 node set, k = 232), scored as 100 * (AUC_full - AUC_partition) /
(AUC_full - 0.5) for Randi 2023 wild-type signal propagation (all observed
pairs, alpha = 0.1):

  classes   random within-cell-class pairs (50 draws)
  adult     merge criterion on the adult connectome
  L1        merge criterion on each of the four L1 connectomes, lifted onto the
            adult by neuron name
  thinned   merge criterion on the adult connectome after multinomial thinning
            of the pool submatrix to each of the eight developmental synapse
            counts (8 x 5 replicates, seeds SEED + 100*t; no thinning when the
            budget exceeds the total)
  random    random disjoint pairs from the same pool (50 draws)

Uses infer from functional_transfer.py with the same seeds and draw
order, keeping every draw. As a consistency check, each arm mean must match
the corresponding arm of results/functional_transfer_all_m70.csv
(functional_transfer.py --pool all --n-merge 70) to within 0.15 percentage
points, or the script exits.

Inputs: Cook 2019 adjacency, Witvliet 2021 datasets (conres.celegans), Randi
2023 responses (conres.randi), results/functional_transfer_all_m70.csv.
Outputs: results/fig4b_l1_adult_prediction.csv (arm, loss_pct, one row per
partition), figures/fig4/fig4b_l1_adult_prediction.svg.
"""
from __future__ import annotations

import collections
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from conres import celegans as ce, figstyle as fs, randi                   # noqa: E402
from prospective_merges import labels_from_pairs                      # noqa: E402
from functional_transfer import infer                                # noqa: E402

OUT = ROOT / "figures/fig4"
ALPHA, M, SEED, N_NULL = 0.1, 70, 0, 50
BIO, TRANS, INK, RAND = "#2e9e52", "#2a5fd6", "#22282d", "#8b9196"
FS = 6.5
REFERENCE = ROOT / "results/functional_transfer_all_m70.csv"
# arm here -> arms of functional_transfer.py averaged
REF_ARMS = {"adult": ["cook"], "L1": ["1L", "2L", "3L", "4L"], "thinned": ["cook_thinned"],
            "random": ["random"], "classes": ["biological"]}


def main() -> None:
    A, nodes = ce.build_matrix("all", neurons_only=True)
    n = len(nodes)
    idx = {x: i for i, x in enumerate(nodes)}
    cls = ce.assign_classes(nodes)
    occ3, _ = randi.build_occ3(nodes, "wt")
    d = randi.load_responses("wildty")
    d = d[d.pre.isin(idx) & d.post.isin(idx)]
    hits = {(idx[p], idx[q]) for p, q in zip(d.pre, d.post)}
    ii, jj = np.nonzero(occ3)
    y = np.array([1.0 if (j, i) in hits else 0.0 for i, j in zip(ii, jj)])
    auc0 = roc_auc_score(y, randi.resolvent(A, ALPHA)[jj, ii])

    wn = ce.witvliet_common_neurons("all")
    files = ce.witvliet_files()
    wmats = [ce.witvliet_matrix(p, wn, "all") for _, _, p in files]
    stages = [f"{i}{s[0]}" for i, s, _ in files]
    lift = {i: idx[wn[i]] for i in range(len(wn)) if wn[i] in idx}
    pool_cook = np.array(sorted(lift.values()))
    wallowed = np.zeros(len(wn), bool)
    wallowed[list(lift)] = True
    pmask = np.zeros(n, bool)
    pmask[pool_cook] = True

    def loss(pairs):
        v = randi.class_leverage(A, labels_from_pairs(n, pairs), ALPHA)[jj, ii]
        return 100 * (auc0 - roc_auc_score(y, v)) / (auc0 - 0.5)

    vals = {}
    vals["adult"] = [loss(infer(A, pmask, M))]
    per_stage = {st: loss([(lift[a], lift[b]) for a, b in infer(Aw, wallowed, M)])
                 for st, Aw in zip(stages, wmats)}
    vals["L1"] = [per_stage[s] for s in stages[:4]]

    def thin(Mx, budget, r):
        w = Mx.ravel()
        tot = w.sum()
        if tot <= budget:
            return Mx.copy()
        return r.multinomial(int(budget), w / tot).reshape(Mx.shape).astype(float)

    Asub = A * np.outer(pmask, pmask)
    thinned = []
    for st, Aw in zip(stages, wmats):
        for t in range(5):
            r = np.random.Generator(np.random.PCG64(SEED + 100 * t))
            At = thin(Asub, Aw.sum(), r) + A * ~np.outer(pmask, pmask)
            thinned.append((st, loss(infer(At, pmask, M))))
    vals["thinned"] = [v for _, v in thinned]

    rng = np.random.Generator(np.random.PCG64(SEED))

    def biological_pairs():
        byc = collections.defaultdict(list)
        for i in pool_cook:
            byc[cls[nodes[i]]].append(int(i))
        cand = []
        for v in byc.values():
            q = rng.permutation(v)
            cand += [(int(q[2 * i]), int(q[2 * i + 1])) for i in range(len(q) // 2)]
        return [cand[i] for i in rng.permutation(len(cand))[:M]]

    def random_pairs():
        p = rng.permutation(pool_cook)
        return [(int(p[2 * i]), int(p[2 * i + 1])) for i in range(M)]

    vals["classes"] = [loss(biological_pairs()) for _ in range(N_NULL)]
    vals["random"] = [loss(random_pairs()) for _ in range(N_NULL)]

    for k, v in vals.items():
        print(f"{k:>8}: mean {np.mean(v):+.2f}%  (n={len(v)}, range {np.min(v):+.1f} to {np.max(v):+.1f})")
    ref = pd.read_csv(REFERENCE).set_index("arm").auc
    expected = {k: 100 * (auc0 - ref[a].mean()) / (auc0 - 0.5) for k, a in REF_ARMS.items()}
    bad = {k: np.mean(v) for k, v in vals.items() if abs(np.mean(v) - expected[k]) > 0.15}
    if bad:
        raise SystemExit(f"does not reproduce functional_transfer.py: {bad}")
    print("L1 animals: " + "  ".join(f"{s} {per_stage[s]:.1f}%" for s in stages[:4]))

    rows = [{"arm": k, "loss_pct": x} for k, v in vals.items() for x in v]
    pd.DataFrame(rows).to_csv(ROOT / "results/fig4b_l1_adult_prediction.csv", index=False)

    # ------------------------------------------------------------------ panel
    fs.use()
    arms = [("classes", "cell classes", BIO, "o", True),
            ("adult", "inferred in the adult", INK, "o", True),
            ("L1", "inferred in L1 animals", TRANS, "o", True),
            ("thinned", "adult, thinned to\ndevelopmental synapse counts", INK, "o", False),
            ("random", "size-matched random", RAND, "o", True)]
    fig, ax = plt.subplots(figsize=(2.8, 2.35))
    fig.subplots_adjust(left=0.47, right=0.96, top=0.86, bottom=0.2)
    jit = np.random.default_rng(1)
    for row, (k, lab, col, mk, filled) in enumerate(arms):
        yv = len(arms) - 1 - row
        v = np.asarray(vals[k])
        yy = yv + (jit.uniform(-0.16, 0.16, v.size) if v.size > 1 else np.zeros(1))
        ax.plot(v, yy, mk, ms=2.4 if v.size > 8 else 3.4, alpha=0.5 if v.size > 8 else 1.0,
                mfc=col if filled else "white", mec=col, mew=0.7, ls="none", zorder=3)
        m = float(np.mean(v))
        ax.plot([m, m], [yv - 0.3, yv + 0.3], color=col, lw=1.6, zorder=4,
                solid_capstyle="butt")
        ax.text(m, yv + 0.36, f"{m:.1f}%", fontsize=FS - 1.4, color=col, ha="center",
                va="bottom", zorder=5)
    ax.axvline(0, color=INK, lw=0.5, zorder=1)
    ax.set_yticks(range(len(arms)), [a[1] for a in arms][::-1], fontsize=FS - 1.0)
    ax.tick_params(axis="y", length=0)
    fs.style(ax)
    ax.set_ylim(-0.6, len(arms) - 0.25)
    ax.set_xlim(-8, 32)
    ax.set_xticks([0, 10, 20, 30])
    ax.set_xlabel("adult predictive performance lost (%)", fontsize=FS)
    ax.set_title("adult signal propagation, k = 232", fontsize=FS, color=INK, loc="left", pad=3)
    fs.save(fig, OUT / "fig4b_l1_adult_prediction.svg", check=fs.check_path("check_fig4g.png"))


if __name__ == "__main__":
    main()
