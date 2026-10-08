"""Compare operator-defined and reference partitions of the worm with the Dvali
et al. (2025) head communities, by agreement and by the class-level path
operator.

Dvali, Seguin, Betzel & Leifer (PRX Life 3, 033021) fit a hierarchical SBM to
the 188-neuron head connectome (20 anatomical communities) and to the Randi
signal propagation network (6 signaling communities). Memberships are
transcribed in conres.dvali; the transcription should reproduce their reported
anatomy-signaling ARI of 0.105.

Part 1: agreement. Partitions built on the whole connectome (loss-greedy from
results/coarse_trajectory_paths.npz, Ward and k-means on log-scaled operator
profiles, at k = 6, 9, 20) are restricted to the head neurons and compared with
both Dvali partitions by ARI and AMI. The null shuffles labels within the
neuron set (--n-perm permutations; z and permutation p are for ARI only). Each
comparison is repeated with the pharyngeal neurons (conres.dvali.PHARYNX)
removed. Because Dvali's SBM was fit to the head alone, Ward and k-means are
also refit on the head ("head-fit": rows and columns of the whole-animal
operator restricted to the head). References: spatial k-means (k = 9),
subsystems, cell types, cell classes. Contingency tables are written for the
loss-greedy k = 9 partition.

Part 2: class-level path operator. A Dvali partition covers only the head, so
every other neuron is kept as its own class; every partition is treated the
same way, so rows differ only in how the head neurons are grouped. Scores are
reported on pairs with both neurons in the head, on head pairs without
pharyngeal neurons, and on all pairs, each as a fraction of the full operator on
the same pairs. Intervals are the spread over seeds or draws for multi-seed
partitions and a stimulated-class bootstrap (head pairs) otherwise. Random
partitions (--n-rand) are size-matched to the anatomical community sizes on the
head. The signaling communities were fit to the same atlas, so scoring them is
circular and is a reference only.

Produces the Dvali comparison tables in the Supplementary Information.

Inputs:  conres.observables.load(); conres.dvali; results/coarse_trajectory_paths.npz;
         relabel and spatial_kmeans from `block_decomposition.py`.
Outputs: results/dvali_agreement.csv
         results/dvali_contingency.csv
         results/dvali_quotient.csv
Usage:   python scripts/dvali_communities.py [--n-perm 2000] [--n-rand 50]
             [--n-boot 300] [--n-seed 5]
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score as ami, adjusted_rand_score as ari

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from conres import dvali, observables as ob, randi, subsystems as ss   # noqa: E402
from block_decomposition import relabel, spatial_kmeans   # noqa: E402

ALPHA = 0.1


def head_only(lab, head):
    """Keep `lab` on the head, give every other neuron its own class."""
    out = np.empty(lab.size, int)
    out[head] = relabel(lab[head])
    rest = np.flatnonzero(~head)
    out[rest] = out[head].max() + 1 + np.arange(rest.size)
    return out


def agreement(a, b, n_perm, rng):
    obs_ari, obs_ami = ari(a, b), ami(a, b)
    null = np.array([ari(rng.permutation(a), b) for _ in range(n_perm)])
    return obs_ari, obs_ami, (obs_ari - null.mean()) / null.std(), (1 + (null >= obs_ari).sum()) / (1 + n_perm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--n-rand", type=int, default=50)
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--n-seed", type=int, default=5)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    D = ob.load()
    A, nodes, cls, obs = D["A"], D["nodes"], D["cls"], D["obs"]
    R = randi.resolvent(A, ALPHA)
    anat, sig = dvali.labels(nodes, "anatomy"), dvali.labels(nodes, "signaling")
    head = anat >= 0
    pharynx = np.isin(nodes, dvali.PHARYNX)
    print(f"{head.sum()} common head neurons; {pharynx.sum()} pharyngeal")

    Xop = np.log1p(np.hstack([R, R.T]) / R.max() * 1e3)
    ward = linkage(Xop, method="ward")
    greedy = np.load(ROOT / "results/coarse_trajectory_paths.npz")
    ours = {}
    for k in (6, 9, 20):
        ours[f"loss-greedy k={k}"] = [relabel(greedy[f"envelope_{k}"])]
        ours[f"Ward on operator k={k}"] = [relabel(fcluster(ward, t=k, criterion="maxclust"))]
        ours[f"k-means on operator k={k}"] = [relabel(KMeans(k, n_init=10, random_state=s).fit(Xop).labels_)
                                              for s in range(args.n_seed)]
    H = np.flatnonzero(head)
    Xh = np.log1p(np.hstack([R[np.ix_(H, H)], R[np.ix_(H, H)].T]) / R.max() * 1e3)
    ward_h = linkage(Xh, method="ward")

    def lift(lh):
        out = np.full(len(nodes), -1)
        out[H] = lh
        return out

    for k in (6, 9, 20):
        ours[f"Ward on operator, head-fit k={k}"] = [lift(relabel(fcluster(ward_h, t=k, criterion="maxclust")))]
        ours[f"k-means on operator, head-fit k={k}"] = [
            lift(relabel(KMeans(k, n_init=10, random_state=s).fit(Xh).labels_)) for s in range(args.n_seed)]
    refs = {
        "spatial k-means k=9": [spatial_kmeans(nodes, 9, s) for s in range(args.n_seed)],
        "subsystems (a priori)": [ss.partition(nodes, "subsystems")[0]],
        "cell types (a priori)": [ss.partition(nodes, "types")[0]],
        "cell classes": [cls],
    }

    # ------------------------------------------------------------- part 1: agreement
    rows = []
    targets = {"Dvali anatomy (20)": anat, "Dvali signaling (6)": sig}
    subsets = {"head": head, "head minus pharynx": head & ~pharynx}
    for pname, labs in {**ours, **refs,
                        "Dvali anatomy (20)": [anat], "Dvali signaling (6)": [sig]}.items():
        for tname, t in targets.items():
            if pname == tname:
                continue
            for sname, sub in subsets.items():
                vals = np.array([agreement(lab[sub], t[sub], args.n_perm, rng) for lab in labs])
                rows.append({"partition": pname, "against": tname, "neurons": sname,
                             "n": int(sub.sum()),
                             "k_on_set": int(np.mean([np.unique(l[sub]).size for l in labs])),
                             "ARI": vals[:, 0].mean(), "ARI_sd_seeds": vals[:, 0].std(),
                             "AMI": vals[:, 1].mean(), "z_null": vals[:, 2].mean(),
                             "p_null": vals[:, 3].max()})
    agree = pd.DataFrame(rows)
    agree.to_csv(ROOT / "results/dvali_agreement.csv", index=False)
    print("\n== agreement with Dvali communities on the common head neurons "
          f"(null: {args.n_perm} label permutations; k-means averaged over {args.n_seed} seeds)")
    print(agree.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    tabs = []
    op9 = ours["loss-greedy k=9"][0]
    for tname, t in targets.items():
        key = dvali.ANATOMY if tname.startswith("Dvali anatomy") else dvali.SIGNALING
        ct = pd.crosstab(pd.Series(np.array(list(key))[t[head]], name=tname),
                         pd.Series(op9[head], name="loss-greedy k=9"))
        ct = ct.loc[list(key)]
        print(f"\n== {tname} (rows) x loss-greedy k=9 (columns), head neurons")
        print(ct.to_string())
        ct.insert(0, "table", tname)
        tabs.append(ct.reset_index().rename(columns={tname: "community"}))
    pd.concat(tabs).to_csv(ROOT / "results/dvali_contingency.csv", index=False)

    # -------------------------------------------- part 2: class-level path operator
    in_head = {o: head[j] & head[i] for o, (j, i, _, _) in obs.items()}
    hnp = head & ~pharynx
    in_hnp = {o: hnp[j] & hnp[i] for o, (j, i, _, _) in obs.items()}
    boots = ob.boot_indices(obs, cls, args.n_boot)
    full = {o: {s: ob.score(kd, ob.predict(kd, R, j, i), v, m)
                for s, m in (("head pairs", in_head[o]), ("all pairs", slice(None)))}
            for o, (j, i, v, kd) in obs.items()}
    pf_all = {o: ob.predict(kd, R, j, i) for o, (j, i, v, kd) in obs.items()}

    def frac_ci(o, p):
        j, i, v, kd = obs[o]
        out = []
        for s in boots[o]:
            s = s[in_head[o][s]]
            out.append(ob.score(kd, p, v, s) / ob.score(kd, pf_all[o], v, s))
        return np.nanpercentile(out, [2.5, 97.5])

    scored = {"Dvali anatomy (20)": [anat], "Dvali signaling (6) [circular]": [sig],
              **{k: v for k, v in ours.items() if k.endswith(("k=9", "k=20", "k=6"))},
              "subsystems (a priori)": refs["subsystems (a priori)"],
              "cell types (a priori)": refs["cell types (a priori)"],
              "cell classes": [cls]}
    sizes = np.bincount(anat[head])
    rnd = []
    for _ in range(args.n_rand):
        lab = np.zeros(len(nodes), int)
        lab[np.flatnonzero(head)[rng.permutation(head.sum())]] = np.repeat(np.arange(sizes.size), sizes)
        rnd.append(lab)
    scored["random (size-matched to Dvali anatomy)"] = rnd

    rows = []
    for pname, labs in scored.items():
        for o, (j, i, v, kd) in obs.items():
            per = []
            for lab in labs:
                L = head_only(lab, head)
                Q = randi.class_leverage(A, L, ALPHA)
                pq = ob.predict(kd, Q, j, i)
                per.append((ob.score(kd, pq, v, in_head[o]), ob.score(kd, pq, v), pq,
                            ob.score(kd, pq, v, in_hnp[o])))
            r = {"partition": pname, "observable": o,
                 "k_head": int(np.mean([np.unique(l[head]).size for l in labs])),
                 "n_head_pairs": int(in_head[o].sum()),
                 "full_head": full[o]["head pairs"], "full_all": full[o]["all pairs"],
                 "quotient_head": np.mean([p[0] for p in per]),
                 "quotient_all": np.mean([p[1] for p in per]),
                 "full_head_nophar": ob.score(kd, pf_all[o], v, in_hnp[o]),
                 "quotient_head_nophar": np.mean([p[3] for p in per])}
            r["frac_head"] = r["quotient_head"] / r["full_head"]
            r["frac_all"] = r["quotient_all"] / r["full_all"]
            r["frac_head_nophar"] = r["quotient_head_nophar"] / r["full_head_nophar"]
            if len(labs) > 1:
                fr = [p[0] / r["full_head"] for p in per]
                r["frac_head_lo"], r["frac_head_hi"] = np.percentile(fr, [2.5, 97.5])
                r["ci_kind"] = "spread over draws/seeds"
            else:
                r["frac_head_lo"], r["frac_head_hi"] = frac_ci(o, per[0][2])
                r["ci_kind"] = "stimulated-class bootstrap"
            rows.append(r)
        print(f"  scored {pname}", flush=True)
    sc = pd.DataFrame(rows)
    sc.to_csv(ROOT / "results/dvali_quotient.csv", index=False)
    for o in obs:
        d = sc[sc.observable == o]
        print(f"\n== {o}: class-level path operator, head neurons grouped, rest singletons. "
              f"Full operator {d.full_head.iloc[0]:+.4f} on {d.n_head_pairs.iloc[0]} head pairs, "
              f"{d.full_all.iloc[0]:+.4f} on all pairs")
        for _, r in d.iterrows():
            print(f"  {r.partition:>42} k_head {r.k_head:>3}  head pairs {r.quotient_head:+.4f} "
                  f"({r.frac_head:.2f} [{r.frac_head_lo:.2f}, {r.frac_head_hi:.2f}])  "
                  f"head minus pharynx {r.frac_head_nophar:.2f}  all pairs {r.frac_all:.2f}")
    print("\nwrote results/dvali_{agreement,contingency,quotient}.csv")


if __name__ == "__main__":
    main()
