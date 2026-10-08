"""Loss of five functional observables under coarse-graining of the worm path
operator, each as a fraction of its own full-operator headroom, along the
biological, loss-greedy and random partition families, with k_eps per observable.

Observables (i responds, j stimulated):

  incidence     does j -> i respond; AUC on the occ3-observed ordered pairs
  correlation   mean across-animal Pearson r on unordered pairs co-identified in
                >= 3 animals (from `two_observables.setup`); predictor
                S_ji + S_ij, Spearman
  amplitude     |mean stimulus-triggered z| of i after stimulation of j, on the
                occ3 pairs with >= --min-events events; Spearman
  magnitude     the released atlas magnitude among detected responders; Spearman
  latency       time to peak of the stimulus-triggered response among detected
                responders with >= --min-events events; predicted by the
                operator's mean walk length
                    sum_l l (N^l)_ji / sum_l (N^l)_ji,  N = alpha M / rho(M),
                truncated at l = 6 (pairs with no walk get 7); Spearman

Amplitude and latency are built from the raw traces (2 Hz). For each
stimulation event the responder's trace from PRE = 10 volumes before to
POST = 60 after onset is baseline-subtracted by its own pre-stimulus mean and
divided by that neuron's whole-recording SD; events are summed per ordered pair
and cached. Amplitude is |mean z| over volumes WIN = [2, 40) after onset;
latency is (argmax over volumes 1-60 of |mean z|) x 0.5 s. Recordings shorter
than 100 volumes and neurons with any non-finite sample are skipped.

Each loss is the fraction of headroom destroyed,

    frac = (s_full - s_quotient) / (s_full - s_null),   s_null = 0.5 (AUC), 0 (rho),

since AUC and Spearman are not commensurate. A full-operator headroom whose
bootstrap interval touches zero leaves nothing to lose, so the headroom interval
is printed first and k_eps should be read only where it excludes zero.

Partition families: bio (`complexity_curve.bio_path`, k >= 119, --n-bio
trajectories), envelope (the loss-greedy path in results/complexity_curve_paths.npz)
and random (--n-rand permutations of the bio partition at k >= 119, of the
envelope partition below). Intervals are cluster bootstraps over the stimulated
neuron's class (the first neuron's class for correlation); loss is detectable
where the lower bound of the paired interval exceeds zero.

Inputs:  Cook connectome, Randi traces and atlas via conres;
         results/complexity_curve_paths.npz.
Outputs: results/observable_resolution.csv  (arm, k, rep, observable, score, frac_lost)
         results/observable_resolution_detectable_loss.csv        (paired-bootstrap intervals)
         results/sta_cache.npz              (stimulus-triggered sums; reused
                                                   when the node list matches)
Usage:   python scripts/observable_resolution.py [--n-bio 10] [--n-rand 10]
             [--n-boot 300] [--min-events 2] [--seed 0]
"""
from __future__ import annotations

import argparse
import re
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from conres import celegans as ce, randi                     # noqa: E402
from conres.metric_guard import guard_predictor              # noqa: E402
from two_observables import setup as setup_corr         # noqa: E402
from complexity_curve import bio_path                  # noqa: E402

ALPHA = 0.1
PRE, POST, WIN = 10, 60, (2, 40)
K_GRID = (302, 290, 275, 260, 242, 220, 200, 182, 160, 140, 119, 100, 80, 60, 40, 30, 20)
OBS = ("incidence", "correlation", "amplitude", "magnitude", "latency")
CACHE = ROOT / "results/sta_cache.npz"


def build_sta(nodes: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Per ordered pair (i responds, j stimulated): summed z-trace and event count."""
    if CACHE.exists():
        z = np.load(CACHE)
        if list(z["nodes"]) == list(nodes):
            return z["acc"], z["cnt"]
    folder = randi.EXPORTED / "exported_data"
    valid = {x: k for k, x in enumerate(nodes)}
    n = len(nodes)
    L = PRE + POST + 1
    acc = np.zeros((n, n, L), np.float32)
    cnt = np.zeros((n, n))
    for f in sorted(folder.glob("*_labels.txt")):
        r = int(re.match(r"(\d+)_labels", f.name).group(1))
        labels = [ce.normalise_name(x.rstrip("\n").strip()) for x in f.open()]
        sv, sn = folder / f"{r}_stim_volume_i.txt", folder / f"{r}_stim_neurons.txt"
        if not (sv.exists() and sn.exists()):
            continue
        G = randi._gcamp(folder, r)
        if G.shape[0] < 100:
            continue
        alive = np.isfinite(G).all(axis=0)
        cols = [(c, valid[x]) for c, x in enumerate(labels)
                if x in valid and c < G.shape[1] and alive[c]]
        if len(cols) < 2:
            continue
        ci = np.array([c for c, _ in cols])
        ni = np.array([k for _, k in cols])
        X = G[:, ci]
        sd = np.maximum(X.std(0), 1e-9)
        stims = [int(x) for x in sv.open() if x.strip()]
        sneu = [int(x) for x in sn.open() if x.strip()]
        for s, c in zip(stims, sneu):
            if not (0 <= c < len(labels)) or labels[c] not in valid:
                continue
            if s - PRE < 0 or s + POST >= X.shape[0]:
                continue
            j = valid[labels[c]]
            seg = X[s - PRE:s + POST + 1]
            z = (seg - seg[:PRE].mean(0)) / sd
            keep = ni != j
            acc[ni[keep], j] += z[:, keep].T
            cnt[ni[keep], j] += 1
    np.savez_compressed(CACHE, acc=acc, cnt=cnt, nodes=np.array(nodes))
    return acc, cnt


def walk_length(M: np.ndarray, lmax: int = 6) -> np.ndarray:
    rho = np.max(np.abs(np.linalg.eigvals(M)))
    N = ALPHA * M / rho if rho > 0 else M
    P = np.eye(len(M))
    num = np.zeros_like(M)
    den = np.zeros_like(M)
    for l in range(1, lmax + 1):
        P = P @ N
        num += l * P
        den += P
    with np.errstate(invalid="ignore", divide="ignore"):
        W = num / den
    return np.where(den > 0, W, lmax + 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-bio", type=int, default=10)
    ap.add_argument("--n-rand", type=int, default=10)
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--min-events", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dat = setup_corr(3, 0, False)
    A, nodes = dat["A"], dat["nodes"]
    n = A.shape[0]
    bio = np.unique(dat["labels"], return_inverse=True)[1]
    idx = {x: i for i, x in enumerate(nodes)}
    ii, jj, y = dat["ii"], dat["jj"], dat["y"]

    acc, cnt = build_sta(nodes)
    sta = acc / np.maximum(cnt, 1)[..., None]
    t0 = PRE
    amp = np.abs(sta[:, :, t0 + WIN[0]:t0 + WIN[1]].mean(-1))
    lat = (np.argmax(np.abs(sta[:, :, t0 + 1:t0 + POST + 1]), axis=-1) + 1) * 0.5

    d = randi.load_responses("wildty")
    d = d[d.pre.isin(idx) & d.post.isin(idx)]
    rj, ri = d.pre.map(idx).to_numpy(), d.post.map(idx).to_numpy()
    mag = d["mag"].to_numpy(float)

    ev_occ = cnt[ii, jj] >= args.min_events
    sets = {
        "incidence": dict(j=jj, i=ii, v=y),
        "correlation": dict(j=dat["iu"], i=dat["ju"], v=dat["r"]),
        "amplitude": dict(j=jj[ev_occ], i=ii[ev_occ], v=amp[ii[ev_occ], jj[ev_occ]]),
        "magnitude": dict(j=rj, i=ri, v=mag),
    }
    ev_r = cnt[ri, rj] >= args.min_events
    sets["latency"] = dict(j=rj[ev_r], i=ri[ev_r], v=lat[ri[ev_r], rj[ev_r]])
    for o, s in sets.items():
        s["g"] = bio[s["j"]]
        s["where"] = {c: np.flatnonzero(s["g"] == c) for c in np.unique(s["g"])}

    def predictor(o, S, W):
        s = sets[o]
        if o == "correlation":
            return S[s["j"], s["i"]] + S[s["i"], s["j"]]
        if o == "latency":
            return W[s["j"], s["i"]]
        return S[s["j"], s["i"]]

    def metric(o, p, sub=None):
        v = sets[o]["v"]
        if sub is not None:
            p, v = p[sub], v[sub]
        if o == "incidence":
            return roc_auc_score(v, p) if len(np.unique(v)) > 1 else np.nan
        return stats.spearmanr(p, v).statistic

    null = {o: (0.5 if o == "incidence" else 0.0) for o in OBS}

    def operators(lab):
        Q = randi.quotient(A, lab)
        S = randi.resolvent(Q, ALPHA)[np.ix_(lab, lab)]
        W = walk_length(Q)[np.ix_(lab, lab)] if lab.max() + 1 < n else walk_length(A)
        return S, W

    S0, W0 = randi.resolvent(A, ALPHA), walk_length(A)
    full_pred = {o: predictor(o, S0, W0) for o in OBS}
    full = {o: metric(o, full_pred[o]) for o in OBS}

    # -- step 1: is there headroom at full resolution?
    print(f"{'observable':>12} {'pairs':>7} {'full score':>11} {'95% CI':>20}")
    rng = np.random.default_rng(args.seed)
    boots = {}
    for o in OBS:
        s = sets[o]
        uq = np.array(list(s["where"]))
        draws = [np.concatenate([s["where"][c] for c in rng.choice(uq, uq.size)])
                 for _ in range(args.n_boot)]
        boots[o] = draws
        bs = np.array([metric(o, full_pred[o], dr) for dr in draws])
        lo, hi = np.nanpercentile(bs, [2.5, 97.5])
        # tie structure of predictor and observable; the base-rate flag only
        # means something for the binary incidence outcome
        gp = guard_predictor(s["v"] if o == "incidence" else np.zeros(len(s["v"])),
                             full_pred[o])
        gv = guard_predictor(np.zeros(len(s["v"])), s["v"])
        s["headroom_ci"] = (lo - null[o], hi - null[o])
        print(f"{o:>12} {len(s['v']):>7,} {full[o]:>11.4f} [{lo:+.4f}, {hi:+.4f}] "
              f"  tie block: predictor {gp['largest_tie_block']:.3f}, "
              f"observable {gv['largest_tie_block']:.3f}")
    alt = stats.spearmanr(-S0[sets["latency"]["j"],
                          sets["latency"]["i"]], sets["latency"]["v"]).statistic
    print(f"  latency against -R instead of mean walk length: rho {alt:+.4f}")

    # -- step 2: loss curves
    paths = np.load(ROOT / "results/complexity_curve_paths.npz")
    rows = []

    def emit(arm, k, rep, lab):
        S, W = operators(lab)
        for o in OBS:
            sc = metric(o, predictor(o, S, W))
            head = full[o] - null[o]
            rows.append({"arm": arm, "k": k, "rep": rep, "observable": o, "score": sc,
                         "frac_lost": (full[o] - sc) / head if head != 0 else np.nan})

    for k in K_GRID:
        if k == n:
            continue
        emit("envelope", k, 0, paths[f"envelope_{k}"])
        src = paths[f"envelope_{k}"]
        if k >= bio.max() + 1:
            for r in range(args.n_bio):
                emit("bio", k, r, bio_path(bio, np.random.default_rng(args.seed + r))[k])
            src = bio_path(bio, np.random.default_rng(args.seed))[k]
        for b in range(args.n_rand):
            emit("random", k, b, np.random.default_rng(1000 + b).permutation(src))
        print(f"  k = {k} scored", end="\r", flush=True)
    print()

    import pandas as pd
    df = pd.DataFrame(rows)
    out = ROOT / "results/observable_resolution.csv"
    df.to_csv(out, index=False)
    mean = df.groupby(["arm", "k", "observable"]).frac_lost.mean().unstack("observable")

    for arm in ("bio", "envelope", "random"):
        print(f"\nfraction of each observable's headroom lost -- {arm}")
        print(f"{'k':>5} {'N/k':>6} " + " ".join(f"{o:>12}" for o in OBS))
        sub = mean.loc[arm].sort_index(ascending=False)
        for k, r in sub.iterrows():
            print(f"{k:>5} {n / k:>6.2f} " + " ".join(f"{r[o]:>+12.3f}" for o in OBS))

    # -- step 3: detectable loss, paired bootstrap at each k on the envelope and bio paths
    print("\npaired bootstrap of the loss fraction (point [95% CI]); '*' = detectable")
    det_rows = []
    for arm in ("bio", "envelope"):
        for k in K_GRID:
            if k == n or (arm == "bio" and k < bio.max() + 1):
                continue
            lab = (bio_path(bio, np.random.default_rng(args.seed))[k] if arm == "bio"
                   else paths[f"envelope_{k}"])
            S, W = operators(lab)
            cells = []
            for o in OBS:
                q = predictor(o, S, W)
                head = full[o] - null[o]
                bs = []
                for dr in boots[o][:args.n_boot]:
                    f_ = metric(o, full_pred[o], dr)
                    bs.append((f_ - metric(o, q, dr)) / head)
                lo, hi = np.nanpercentile(bs, [2.5, 97.5])
                pt = (full[o] - metric(o, q)) / head
                det_rows.append({"arm": arm, "k": k, "observable": o, "frac_lost": pt,
                                 "lo": lo, "hi": hi, "detectable": lo > 0})
                cells.append(f"{pt:+.2f}[{lo:+.2f},{hi:+.2f}]{'*' if lo > 0 else ' '}")
            print(f"  {arm:>8} k={k:>3}  " + "  ".join(f"{o[:4]} {c}" for o, c in zip(OBS, cells)),
                  flush=True)
    pd.DataFrame(det_rows).to_csv(ROOT / "results/observable_resolution_detectable_loss.csv", index=False)

    # -- step 4: k_eps per observable
    print("\nk_eps (smallest k with mean loss fraction <= eps at every finer k)")
    from complexity_curve import k_eps
    for arm in ("bio", "envelope", "random"):
        sub = mean.loc[arm]
        for o in OBS:
            ks = [n] + list(sub.index)
            vals = [0.0] + list(sub[o])
            floor = min(sub.index)
            res = []
            for e in (0.10, 0.25, 0.50):
                ke = k_eps(ks, vals, e)
                res.append(f"eps {e:.2f} -> {('<=' + str(ke)) if ke == floor else ke}")
            print(f"  {arm:>8} {o:>12}: " + "   ".join(res))
    dd = pd.DataFrame(det_rows)
    print("\nfirst k (coarsening from 302) at which loss becomes detectable")
    for arm in ("bio", "envelope"):
        for o in OBS:
            sub = dd[(dd.arm == arm) & (dd.observable == o)].sort_values("k", ascending=False)
            hit = sub[sub.detectable]
            print(f"  {arm:>8} {o:>12}: "
                  f"{int(hit.k.iloc[0]) if len(hit) else 'none down to ' + str(int(sub.k.min()))}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
