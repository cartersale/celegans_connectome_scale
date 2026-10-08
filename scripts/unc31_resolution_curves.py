"""Propagation prediction retained along the class and random merge trajectories, 302 -> 119, in wild type and unc-31 mutants.

The trajectories are those of the Fig. 2d resolution curve
(class_merge_trajectory.trajectory): replicate r uses PCG64(seed + r) for both
arms, so the two strains are scored on identical partition sequences and differ
only in their observed pairs and response labels.

Per strain, observed pairs are occ3 > 0 in that strain's recordings and a pair is
a response if it is in that strain's released response list. Retained prediction
at k is

    100 * (AUC(R_Q) - 0.5) / (AUC(R) - 0.5)

with R the full path operator and R_Q the quotient operator lifted to neuron pairs
(alpha = 0.1), among unsynapsed pairs (`_nosyn`) and among all observed pairs
(`_all`). Normalising by each strain's own above-chance AUC puts the strains on one
scale even though unc-31 has roughly half the effect to lose; the cost is that the
unc-31 curve divides by a smaller number and is noisier.

The band across replicates reflects merge order only. Sampling uncertainty at
k = 119 is given by a cluster bootstrap over the stimulated neuron's cell class
(--n-boot resamples, unsynapsed pairs; resamples with a single outcome class are
skipped); it is printed and drawn as an error bar at the k = 119 end of each
class curve.

Inputs: connectome via conres.celegans; occ3 and responses via conres.randi.
Outputs: results/unc31_resolution_curves.csv (strain, mode, rep, k,
    retained_all, retained_nosyn); figures/supp/unc31_resolution_curves.svg.
Usage: python scripts/unc31_resolution_curves.py [--n-rep 30] [--n-boot 2000] [--seed 0]
"""
from __future__ import annotations

import argparse
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

from conres import celegans as ce, randi, figstyle as fs           # noqa: E402
from class_merge_trajectory import trajectory                      # noqa: E402

ALPHA = 0.1
STRAINS = (("wt", "wildty", "Wild type"), ("unc31", "unc31", "unc-31"))
FS = 5.9


def strain_pairs(nodes, strain, key):
    idx = {n: i for i, n in enumerate(nodes)}
    occ3, _ = randi.build_occ3(nodes, strain)
    d = randi.load_responses(key)
    d = d[d.pre.isin(idx) & d.post.isin(idx)]
    hits = {(idx[p], idx[q]) for p, q in zip(d.pre, d.post)}
    ii, jj = np.nonzero(occ3)
    y = np.array([1.0 if (j, i) in hits else 0.0 for i, j in zip(ii, jj)])
    return jj, ii, y


def retained(auc_q, auc_full):
    return 100 * (auc_q - 0.5) / (auc_full - 0.5)


def endpoint_boot(y, full, quo, cls, n_boot, rng):
    """Cluster-bootstrap distribution of retained % at k = 119, unsynapsed pairs."""
    where = {c: np.flatnonzero(cls == c) for c in np.unique(cls)}
    uq = np.array(list(where))
    out = []
    for _ in range(n_boot):
        s = np.concatenate([where[c] for c in rng.choice(uq, uq.size)])
        if len(np.unique(y[s])) < 2:
            continue
        out.append(retained(roc_auc_score(y[s], quo[s]), roc_auc_score(y[s], full[s])))
    return np.array(out)


def plot(df, boot, headline):
    fs.use()
    fig, axes = plt.subplots(1, 2, figsize=(4.9, 1.9), sharey=True)
    fig.subplots_adjust(left=0.11, right=0.98, top=0.86, bottom=0.21, wspace=0.12)
    for ax, (strain, _, tag) in zip(axes, STRAINS):
        for mode, c, lbl in (("random", fs.GREY, "Random"), ("bio", fs.CLASS, "Classes")):
            g = df[(df.strain == strain) & (df["mode"] == mode)].groupby("k")["retained_nosyn"]
            m, lo, hi = g.mean(), g.quantile(0.025), g.quantile(0.975)
            ax.fill_between(m.index, lo, hi, color=c, alpha=0.18, lw=0)
            ax.plot(m.index, m, color=c, lw=1.2, label=lbl)
        b_lo, b_hi = np.percentile(boot[strain], [2.5, 97.5])
        mid = headline[strain]
        # the unc-31 interval exceeds the axis; the bar is clipped and the bounds printed
        ax.errorbar([116], [mid], yerr=[[mid - b_lo], [b_hi - mid]], fmt="o", ms=2.6,
                    color=fs.CLASS, elinewidth=0.7, capsize=1.6, capthick=0.7)
        ax.axhline(100, color=fs.FAINT, lw=0.6, ls=(0, (3, 2)), zorder=0)
        ax.axhline(0, color=fs.INK, lw=0.5)
        ax.set_xlim(302, 112)
        ax.set_xticks([300, 250, 200, 150, 119])
        ax.set_ylim(-25, 125)
        ax.set_yticks([0, 50, 100])
        ax.set_xlabel("Resolution (groups, k)", fontsize=FS)
        ax.set_title(tag, fontsize=6.5, color=fs.INK, loc="left")
        ax.set_title(f"classes at k = 119: {mid:.0f}% [{b_lo:.0f}, {b_hi:.0f}]",
                     fontsize=5.2, color=fs.CLASS, loc="right")
        fs.style(ax)
    axes[0].set_ylabel("Propagation effect\nretained (%)", fontsize=FS)
    axes[0].legend(frameon=False, fontsize=FS, loc="center left", handlelength=1.4,
                   bbox_to_anchor=(0.0, 0.27))
    fs.save(fig, ROOT / "figures/supp/unc31_resolution_curves.svg",
            check=fs.check_path("check_unc31_resolution_curves.png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-rep", type=int, default=30)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    A, nodes = ce.build_matrix("all", neurons_only=True)
    labels = ce.partitions(nodes)["class"]
    lab = np.unique(labels, return_inverse=True)[1]
    R = randi.resolvent(A, ALPHA)
    RQ = randi.class_leverage(A, lab, ALPHA)

    rows, boot, headline = [], {}, {}
    for strain, key, tag in STRAINS:
        jj, ii, y = strain_pairs(nodes, strain, key)
        sub = A[jj, ii] == 0
        full = R[jj, ii]
        auc_all, auc_sub = roc_auc_score(y, full), roc_auc_score(y[sub], full[sub])
        print(f"{tag}: {len(y):,} pairs ({int(y.sum()):,} responses), "
              f"unsynapsed {int(sub.sum()):,} ({int(y[sub].sum()):,}); full-operator AUC "
              f"all {auc_all:.4f}, unsynapsed {auc_sub:.4f} (headroom {auc_sub - 0.5:+.4f})")

        for mode in ("bio", "random"):
            for r in range(args.n_rep):
                rng = np.random.Generator(np.random.PCG64(args.seed + r))
                for k, _, a_all, a_sub in trajectory(A, labels, jj, ii, y, sub, rng, mode):
                    rows.append((strain, mode, r, k, retained(a_all, auc_all),
                                 retained(a_sub, auc_sub)))

        q = RQ[jj, ii]
        headline[strain] = retained(roc_auc_score(y[sub], q[sub]), auc_sub)
        boot[strain] = endpoint_boot(y[sub], full[sub], q[sub], labels[jj][sub], args.n_boot,
                                     np.random.default_rng(args.seed))
        lo, hi = np.percentile(boot[strain], [2.5, 97.5])
        print(f"  classes at k=119, unsynapsed: {headline[strain]:.0f}% retained "
              f"[{lo:.0f}, {hi:.0f}] ({len(boot[strain])} resamples)")

    df = pd.DataFrame(rows, columns=["strain", "mode", "rep", "k", "retained_all",
                                     "retained_nosyn"])
    out = ROOT / "results/unc31_resolution_curves.csv"
    df.to_csv(out, index=False)
    print(f"wrote {out} ({len(df)} rows)")

    print(f"\n{'k':>5}  {'WT classes':>11} {'WT random':>10}  {'unc-31 classes':>15} {'unc-31 random':>14}"
          "   (% retained, unsynapsed, mean over replicates)")
    mean = df.groupby(["strain", "mode", "k"])["retained_nosyn"].mean()
    for k in (290, 275, 260, 242, 220, 200, 180, 160, 140, 119):
        try:
            v = [mean[(s, m, k)] for s in ("wt", "unc31") for m in ("bio", "random")]
        except KeyError:
            continue
        print(f"{k:>5}  {v[0]:>11.0f} {v[1]:>10.0f}  {v[2]:>15.0f} {v[3]:>14.0f}")

    plot(df, boot, headline)


if __name__ == "__main__":
    main()
