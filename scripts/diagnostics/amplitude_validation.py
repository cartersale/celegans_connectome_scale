"""Check the response-amplitude observable against the released atlas.

Amplitude is the absolute stimulus-triggered average over the response window
(conres.observables.build_sta, PRE, WIN), defined for observed pairs with at
least --min-events stimulations. Two comparisons with the Randi 2023 wild-type
atlas are printed:

  detection   AUC of amplitude for whether the pair is a detected response,
              over those observed pairs
  magnitude   Spearman correlation of amplitude with the released response
              magnitude, over detected responses with >= --min-events
              stimulations

Inputs: Cook connectome node list, Randi recordings and atlas via conres;
    results/sta_cache.npz when present.
Outputs: results/diagnostics/amplitude_validation.log.
Usage: python scripts/diagnostics/amplitude_validation.py [--min-events 2]
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from conres import randi                                # noqa: E402
from conres.paths import tee_stdout                     # noqa: E402
from conres.observables import (pair_sets as setup_corr,   # noqa: E402
                                build_sta, PRE, WIN)

tee_stdout(__file__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-events", type=int, default=2)
    args = ap.parse_args()

    dat = setup_corr(3, 0, False)
    nodes = dat["nodes"]
    idx = {x: i for i, x in enumerate(nodes)}
    ii, jj, y = dat["ii"], dat["jj"], dat["y"]

    acc, cnt = build_sta(nodes)
    sta = acc / np.maximum(cnt, 1)[..., None]
    amp = np.abs(sta[:, :, PRE + WIN[0]:PRE + WIN[1]].mean(-1))

    ev = cnt[ii, jj] >= args.min_events
    print(f"observed pairs {len(y):,}; with >= {args.min_events} stimulations "
          f"{int(ev.sum()):,}, of which {int(y[ev].sum()):,} detected")
    print(f"detection  AUC of amplitude = {roc_auc_score(y[ev], amp[ii[ev], jj[ev]]):.4f}")

    d = randi.load_responses("wildty")
    d = d[d.pre.isin(idx) & d.post.isin(idx)]
    rj, ri = d.pre.map(idx).to_numpy(), d.post.map(idx).to_numpy()
    mag = d["mag"].to_numpy(float)
    ev_r = cnt[ri, rj] >= args.min_events
    r = stats.spearmanr(amp[ri[ev_r], rj[ev_r]], mag[ev_r])
    print(f"magnitude  Spearman rho = {r.statistic:.4f} (p = {r.pvalue:.1e}) over "
          f"{int(ev_r.sum()):,} of {len(mag):,} released responses")


if __name__ == "__main__":
    main()
