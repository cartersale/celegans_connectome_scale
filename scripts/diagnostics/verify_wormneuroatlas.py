"""Cross-check of the Randi 2023 inputs and the signal-propagation AUCs
against the Randi lab's wormneuroatlas package.

This project reaches the Randi measurements through src/conres/randi.py (our
parse of the released funconn JSON and our reconstruction of the observation
denominator occ3 from the OSF recordings). wormneuroatlas has its own loaders,
detection call (q < QTHRESH from its kernel fits) and anatomical connectome
(White 1986 + Witvliet 2021), so it is an independent comparison. Reported:

  - neuron sets (AWC is matched by function in theirs, by side in ours)
  - observation denominator: observed-pair overlap and Spearman of exposure
    counts with occ3
  - anatomical connectome: edge Jaccard and weight Spearman against Cook 2019
  - direct-weight and path-operator AUCs (alpha = 0.1) on their responses,
    using their connectome and Cook 2019

Index conventions: their matrices are [target, source] (their occurrence matrix
matches occ3 far better as given than transposed, and their connectome matches
Cook 2019 far better transposed). Their connectome is
transposed here into our [pre, post] order; their occurrence and q matrices are
read as [responder, stimulus], the same convention as occ3.

The first table row ("ours") is recomputed from the released responses, occ3 and
Cook 2019.

Generating the dump: wormneuroatlas needs h5py, which has no working numpy ABI
in the project environment, so it runs in a separate venv:

    python -m venv /tmp/wna && /tmp/wna/bin/pip install wormneuroatlas
    /tmp/wna/bin/python - <<'PY'
    import wormneuroatlas as wa, numpy as np
    wa.WormBase = lambda *a, **k: None   # its live WormBase version check fails
    na = wa.NeuroAtlas(verbose=False)
    np.savez_compressed("data/raw/wormneuroatlas/wna_dump.npz",
                        ids=np.array(na.neuron_ids),
                        occ=na.get_signal_propagation_occurrence_matrix(),
                        smap=na.get_signal_propagation_map(),
                        q=na.get_signal_propagation_q(),
                        aconn=na.get_anatomical_connectome())
    PY

Inputs: data/raw/wormneuroatlas/wna_dump.npz, conres.celegans, conres.randi.
Outputs: results/wormneuroatlas_crosscheck.md.
Usage: python scripts/diagnostics/verify_wormneuroatlas.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from conres import celegans as ce, randi  # noqa: E402
from conres.paths import tee_stdout                   # noqa: E402
tee_stdout(__file__)

DUMP = ROOT / "data/raw/wormneuroatlas/wna_dump.npz"
OUT = ROOT / "results/wormneuroatlas_crosscheck.md"
ALPHA, QTHRESH = 0.1, 0.05
# they carry AWC by function, we carry it by side
AWC = {"AWCON": "AWCR", "AWCOFF": "AWCL"}


def main() -> None:
    z = np.load(DUMP, allow_pickle=True)
    ids = [str(x) for x in z["ids"]]
    A, nodes = ce.build_matrix("all", neurons_only=True)
    idx = {n: i for i, n in enumerate(nodes)}
    occ_ours, _ = randi.build_occ3(nodes, "wt")

    shared = [n for n in ids if AWC.get(n, n) in idx]
    ti = [ids.index(n) for n in shared]
    oi = [idx[AWC.get(n, n)] for n in shared]
    lines = [f"# wormneuroatlas cross-check\n",
             f"Their neuron set {len(ids)}, ours {len(nodes)}, shared {len(shared)}.",
             f"Only theirs: {sorted(set(ids) - {n for n in ids if AWC.get(n, n) in idx})}. "
             f"Only ours: {sorted(set(nodes) - {AWC.get(n, n) for n in ids})}.",
             "Neither set contains CANL/CANR.\n"]

    # ---- observation denominator: theirs vs our occ3
    Ot, Oo = z["occ"][np.ix_(ti, ti)], occ_ours[np.ix_(oi, oi)]
    m = (Ot > 0) | (Oo > 0)
    rho = stats.spearmanr(Ot[m], Oo[m]).statistic
    rho_T = stats.spearmanr(Ot.T[m], Oo[m]).statistic
    lines += ["## Observation denominator",
              f"- observed pairs: theirs {int((Ot>0).sum()):,}, ours {int((Oo>0).sum()):,}, "
              f"in both {int(((Ot>0)&(Oo>0)).sum()):,}",
              f"- exposure counts Spearman rho = **{rho:.3f}** (transposed {rho_T:.3f}, "
              f"which fixes their index order as [responder, stimulus])\n"]

    # ---- connectome: same experiment, different EM
    At, Ao = z["aconn"][np.ix_(ti, ti)].T, A[np.ix_(oi, oi)]
    b1, b2 = At > 0, Ao > 0
    lines += ["## Anatomical connectome",
              f"- theirs is White 1986 + Witvliet 2020/2021; ours is Cook 2019",
              f"- edges theirs {int(b1.sum()):,}, ours {int(b2.sum()):,}, "
              f"Jaccard **{np.logical_and(b1,b2).sum()/np.logical_or(b1,b2).sum():.3f}**",
              f"- weight Spearman on shared edges rho = "
              f"{stats.spearmanr(At[b1&b2], Ao[b1&b2]).statistic:.3f}\n"]

    # ---- AUCs recomputed on their responses
    occ, q = z["occ"], z["q"]
    obs = occ > 0
    resp = np.isfinite(q) & (q < QTHRESH) & obs
    ii, jj = np.nonzero(obs)
    y = resp[ii, jj].astype(float)

    Acook = np.zeros_like(z["aconn"], dtype=float)
    row = [idx.get(AWC.get(n, n)) for n in ids]
    sel = np.flatnonzero([r is not None for r in row])
    Acook[np.ix_(sel, sel)] = A[np.ix_([row[k] for k in sel], [row[k] for k in sel])]

    lines += [f"## AUCs under each connectome and response call",
              f"Their detection is q < {QTHRESH} from their own kernel fits, over "
              f"{int(obs.sum()):,} observed pairs and {int(resp.sum()):,} responses "
              f"({100*resp.sum()/obs.sum():.2f}%). Walk operator is the resolvent at "
              f"alpha = {ALPHA}.\n",
              "| connectome | responses | direct weight | walk operator | walk, no-synapse |",
              "|---|---|---|---|---|"]
    d = randi.load_responses("wildty")
    d = d[d.pre.isin(idx) & d.post.isin(idx)]
    resp_ours = np.zeros_like(occ_ours, dtype=bool)
    resp_ours[d.post.map(idx).to_numpy(), d.pre.map(idx).to_numpy()] = True
    io, jo = np.nonzero(occ_ours)
    yo = resp_ours[io, jo].astype(float)
    Ro = randi.resolvent(A, ALPHA)
    nso = A[jo, io] == 0
    rows = [("Cook 2019", "ours (released funconn + our occ3)",
             roc_auc_score(yo, A[jo, io]), roc_auc_score(yo, Ro[jo, io]),
             roc_auc_score(yo[nso], Ro[jo, io][nso]))]
    for tag, Amat in (("White + Witvliet", z["aconn"].T), ("Cook 2019", Acook)):
        R = randi.resolvent(Amat, ALPHA)
        direct, walk = Amat[jj, ii], R[jj, ii]
        sub = direct == 0
        rows.append((tag, "theirs (q < 0.05)", roc_auc_score(y, direct),
                     roc_auc_score(y, walk), roc_auc_score(y[sub], walk[sub])))
    for tag, src, a, b, c in rows:
        lines.append(f"| {tag} | {src} | {a:.4f} | {b:.4f} | {c:.4f} |")
    da, wa = [r[2] for r in rows], [r[3] for r in rows]
    lines += [f"\nDirect weight {min(da):.3f}-{max(da):.3f}, walk operator "
              f"{min(wa):.3f}-{max(wa):.3f}; walk minus direct "
              f"{min(b - a for a, b in zip(da, wa)):+.3f} to "
              f"{max(b - a for a, b in zip(da, wa)):+.3f} across the three combinations "
              "of connectome, response call and observation denominator.\n"]

    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
