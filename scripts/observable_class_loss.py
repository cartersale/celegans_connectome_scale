"""Compute the fraction of predictive performance lost at the class partition (k = 119) for five observables, and plot it.

For each observable, loss = (full - quotient) / full, where full is the score of
the full operator and quotient the score of the operator built on the partition
and lifted back to neuron pairs. Arms: the 119 cell classes, and N_NULL = 200
label permutations of them (same group sizes; seed 0), summarised as mean and
2.5-97.5 percentile.

Observables and scores:
  Signal propagation    AUC - 0.5 of resolvent vs response incidence, on pairs
                        with no synapse from stimulated to responder
  Activity correlation  Spearman of S[j, i] + S[i, j] vs pairwise correlation
  Response amplitude    Spearman of resolvent vs |mean STA| in window WIN, pairs
                        with >= 2 stimulation events
  Response magnitude    Spearman of resolvent vs Randi response magnitude
  Response latency      Spearman of walk length vs STA peak time (frames x 0.5),
                        pairs with >= 2 events; computed and saved but drawn
                        only as a label, since the full operator has no signal
Operator: resolvent with ALPHA = 0.1; walk length from
observable_resolution.

Class-arm interval: cluster bootstrap over the class of the first-index neuron
(the stimulated neuron; for correlation, the first of the unordered pair),
N_BOOT = 1000 resamples, seed 7.

Inputs: setup_corr, build_sta, walk_length, PRE, WIN from
    observable_resolution (which also loads the STA cache); Randi
    wild-type responses via conres.randi.
Outputs: results/observable_class_loss.csv; figures/supp/observable_class_loss.{svg,pdf}.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from conres import figstyle as fs, randi                            # noqa: E402
from conres.observables import (pair_sets as setup_corr,            # noqa: E402
                                build_sta, PRE, WIN)
from observable_resolution import walk_length                       # noqa: E402

ALPHA, N_NULL, N_BOOT = 0.1, 200, 1000
BIO, RND = fs.CLASS, fs.GREY
FS = 6.5
OUT = ROOT / "figures/supp"


def main() -> None:
    dat = setup_corr(3, 0, False)
    A, nodes = dat["A"], dat["nodes"]
    idx = {x: i for i, x in enumerate(nodes)}
    lab = np.unique(dat["labels"], return_inverse=True)[1]
    ii, jj, y = dat["ii"], dat["jj"], dat["y"]
    ns = A[jj, ii] == 0

    acc, cnt = build_sta(nodes)
    sta = acc / np.maximum(cnt, 1)[..., None]
    amp = np.abs(sta[:, :, PRE + WIN[0]:PRE + WIN[1]].mean(-1))
    lat = (np.argmax(np.abs(sta[:, :, PRE + 1:]), axis=-1) + 1) * 0.5
    d = randi.load_responses("wildty")
    d = d[d.pre.isin(idx) & d.post.isin(idx)]
    rj, ri = d.pre.map(idx).to_numpy(), d.post.map(idx).to_numpy()
    ev = cnt[ii, jj] >= 2
    evr = cnt[ri, rj] >= 2

    # (label, stimulated/first index, responder/second index, values, kind)
    obs = [
        ("Signal propagation", jj[ns], ii[ns], y[ns], "auc"),
        ("Activity correlation", dat["iu"], dat["ju"], dat["r"], "corr"),
        ("Response amplitude", jj[ev], ii[ev], amp[ii[ev], jj[ev]], "rho"),
        ("Response magnitude", rj, ri, d["mag"].to_numpy(float), "rho"),
        ("Response latency", rj[evr], ri[evr], lat[ri[evr], rj[evr]], "lat"),
    ]

    def pred(kind, S, W, j, i):
        if kind == "corr":
            return S[j, i] + S[i, j]
        return W[j, i] if kind == "lat" else S[j, i]

    def score(kind, p, v, s=slice(None)):
        if kind == "auc":
            return roc_auc_score(v[s], p[s]) - 0.5
        return stats.spearmanr(p[s], v[s]).statistic

    def ops(l):
        Q = randi.quotient(A, l)
        return (randi.resolvent(Q, ALPHA)[np.ix_(l, l)],
                walk_length(Q)[np.ix_(l, l)])

    S0, W0 = randi.resolvent(A, ALPHA), walk_length(A)
    Sb, Wb = ops(lab)
    rng = np.random.default_rng(0)
    nulls = [ops(rng.permutation(lab)) for _ in range(N_NULL)]

    rows = []
    for name, j, i, v, kind in obs:
        pf, pb = pred(kind, S0, W0, j, i), pred(kind, Sb, Wb, j, i)
        full, bio = score(kind, pf, v), score(kind, pb, v)
        g = lab[j]
        where = {c: np.flatnonzero(g == c) for c in np.unique(g)}
        uq = np.array(list(where))
        br = np.random.default_rng(7)
        fb, lb = [], []
        for _ in range(N_BOOT):
            s = np.concatenate([where[c] for c in br.choice(uq, uq.size)])
            f_ = score(kind, pf, v, s)
            fb.append(f_)
            lb.append((f_ - score(kind, pb, v, s)) / f_ if f_ != 0 else np.nan)
        rl = np.array([(full - score(kind, pred(kind, S, W, j, i), v)) / full
                       for S, W in nulls])
        rnd_score = np.array([score(kind, pred(kind, S, W, j, i), v) for S, W in nulls])
        rows.append({"observable": name, "n": len(v), "full": full,
                     "full_lo": np.percentile(fb, 2.5), "full_hi": np.percentile(fb, 97.5),
                     "class": bio, "class_loss": (full - bio) / full,
                     "class_loss_lo": np.nanpercentile(lb, 2.5),
                     "class_loss_hi": np.nanpercentile(lb, 97.5),
                     "random_score_mean": rnd_score.mean(), "random_score_sd": rnd_score.std(ddof=1),
                     "random_loss": rl.mean(), "random_loss_lo": np.percentile(rl, 2.5),
                     "random_loss_hi": np.percentile(rl, 97.5)})
        r = rows[-1]
        print(f"{name:>22} n={r['n']:>6,}  full {full:.4f} [{r['full_lo']:.4f}, {r['full_hi']:.4f}]"
              f"  class {bio:.4f}  loss {r['class_loss']:+.3f} [{r['class_loss_lo']:+.3f}, "
              f"{r['class_loss_hi']:+.3f}]  random score {r['random_score_mean']:.4f} "
              f"+- {r['random_score_sd']:.4f}, loss {r['random_loss']:.3f}")
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results/observable_class_loss.csv", index=False)

    # ---------------------------------------------------------------- panel
    fs.use()
    fig, ax = plt.subplots(figsize=(2.45, 2.05))
    fig.subplots_adjust(left=0.38, right=0.97, top=0.86, bottom=0.20)
    shown = df[df.observable != "Response latency"].reset_index(drop=True)
    yy = np.arange(len(shown))[::-1] + 1
    for y_, (_, r) in zip(yy, shown.iterrows()):
        ax.plot([100 * r.random_loss_lo, 100 * r.random_loss_hi], [y_ - 0.14] * 2,
                color=RND, lw=1.0, solid_capstyle="round")
        ax.plot(100 * r.random_loss, y_ - 0.14, "o", ms=3.4, color=RND, mew=0)
        ax.plot([100 * r.class_loss_lo, 100 * r.class_loss_hi], [y_ + 0.14] * 2,
                color=BIO, lw=1.0, solid_capstyle="round")
        ax.plot(100 * r.class_loss, y_ + 0.14, "o", ms=3.4, color=BIO, mew=0)
    lat = df[df.observable == "Response latency"].iloc[0]
    ax.text(6, 0.05, "no signal at full resolution", fontsize=FS - 0.6,
            color=fs.GREY, va="center", ha="left")
    ax.axvline(0, color=fs.INK, lw=0.5)
    ax.set_yticks(list(yy) + [0.05], list(shown.observable) + ["Response latency"],
                  fontsize=FS)
    ax.set_ylim(-0.45, len(shown) + 0.55)
    ax.set_xlim(-40, 140)
    ax.set_xticks([0, 50, 100])
    ax.set_xlabel("Predictive performance lost (%)", fontsize=FS)
    fs.style(ax)
    ax.tick_params(axis="y", length=0)
    ax.get_yticklabels()[-1].set_color(fs.GREY)
    ax.text(0.30, 1.04, "Classes", color=BIO, fontsize=FS, ha="left", va="bottom",
            transform=ax.transAxes)
    ax.text(0.62, 1.04, "Random", color=RND, fontsize=FS, ha="left", va="bottom",
            transform=ax.transAxes)
    fs.save(fig, OUT / "observable_class_loss.svg")
    fig.savefig(OUT / "observable_class_loss.pdf")
    print("wrote results/observable_class_loss.csv and figures/supp/observable_class_loss.{svg,pdf}")


if __name__ == "__main__":
    main()
