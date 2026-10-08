"""Equitability, structural loss and merge recovery of the cell classes across the
eight Witvliet 2021 developmental connectomes.

The node set is fixed at the neurons reconstructed in all eight animals
(ce.witvliet_common_neurons), so k and the class-size profile are identical at
every stage. Per animal:

  gamma        outgoing equitability defect of the cell classes, ||A H - H N||
               relative to ||A H|| with size weighting, as z and ratio against
               N_GAMMA_NULL size-matched random partitions
  struct loss  1 - Spearman between the class-level path operator (alpha = 0.1,
               own rho_Q) and the full operator over all ordered pairs, against
               N_STRUCT_NULL size-matched random partitions
  merges       same-class rate of MERGES = 35 disjoint merges selected greedily
               by the symmetrized pairwise defect from a shared candidate pool,
               against the pool's chance rate

Synapse count grows several-fold across the series and every statistic
responds to density alone, so each is reported raw and after multinomial
thinning to BUDGET synapses (THIN_REPS replicates, averaged). BUDGET is set
below the sparsest graph so that every stage, including the sparsest, carries
thinning noise. Trends against developmental order are Spearman rho. Also
printed: Jaccard of the merge selections across animals (raw, and thinning
replicate 0).

Inputs: conres.celegans Witvliet loaders, prospective_merges,
    cross_animal_transfer.
Outputs: results/dev_sufficiency.csv (kind = raw | thinned, per animal), read
    by fig4c_dev_loss.py.
Usage: python scripts/dev_sufficiency.py [--exclude-glia]
    --exclude-glia removes the five glia (four CEP sheath cells, GLRDR) from
    the node set and writes results/dev_sufficiency_noglia.csv instead.
"""
import csv, sys, warnings
from pathlib import Path
import numpy as np
from scipy import stats
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from conres import celegans as ce, randi
from prospective_merges import pairwise_defect, greedy_disjoint
from cross_animal_transfer import pool_mask

ALPHA = 0.1
BUDGET = 1200          # below the sparsest graph's 1,272
THIN_REPS = 10
N_GAMMA_NULL = 100
N_STRUCT_NULL = 20

# --exclude-glia drops the five glia from the node set (results/dev_sufficiency_noglia.csv)
EXCLUDE_GLIA = "--exclude-glia" in sys.argv
nodes = ce.witvliet_common_neurons("all", exclude_glia=EXCLUDE_GLIA)
labels = ce.partitions(nodes)["class"]
lab_bio = np.unique(labels, return_inverse=True)[1]
n = len(nodes); sizes = np.bincount(lab_bio)
off = ~np.eye(n, dtype=bool)
oj, oi = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
oj, oi = oj[off], oi[off]
same_class_pairs = (sizes * (sizes - 1) // 2).sum()
chance_same = same_class_pairs / (n * (n - 1) / 2)

def gamma(A, lab):
    used, l = np.unique(lab, return_inverse=True)
    H = np.zeros((n, used.size)); H[np.arange(n), l] = 1.0
    sz = H.sum(0)
    M = (A @ H) / np.maximum(sz, 1)
    N = (H.T @ M) / np.maximum(sz, 1)[:, None]
    R = M - N[l, :]
    num = (sz * (R ** 2).sum(0)).sum(); den = (sz * (M ** 2).sum(0)).sum()
    return float(np.sqrt(num / den)) if den > 0 else np.nan

def rand_lab(rng):
    perm = rng.permutation(n); rl = np.zeros(n, int); p = 0
    for c, s in enumerate(sizes):
        rl[perm[p:p + s]] = c; p += s
    return rl

def struct_loss(A, lab, full):
    RQ = randi.resolvent(randi.quotient(A, lab), ALPHA)
    return 1.0 - stats.spearmanr(RQ[np.ix_(lab, lab)][oj, oi], full).statistic

# One candidate pool (the pooled-graph degree floor of cross_animal_transfer)
# and one merge count for every animal and every thinning replicate, so k and the
# candidate set are identical across the series. MERGES is below what the pool
# can supply.
POOL = None            # set in __main__ once the matrices are loaded
MERGES = 35


def merge_recovery(A, rng):
    g = (pairwise_defect(A) + pairwise_defect(A.T)) / 2.0
    sel = greedy_disjoint(g, MERGES, POOL)
    if not sel: return np.nan, np.nan, []
    same = np.mean([lab_bio[a] == lab_bio[b] for a, b in sel])
    keep = np.flatnonzero(POOL); sub = lab_bio[keep]
    cs = np.bincount(sub); ch = (cs * (cs - 1) // 2).sum() / (len(keep) * (len(keep) - 1) / 2)
    return same, ch, sel

def thin(A, budget, rng):
    w = A.ravel(); tot = w.sum()
    if tot <= budget: return A.copy()
    draw = rng.multinomial(budget, w / tot)
    return draw.reshape(A.shape).astype(float)

def measure(A, rng):
    full = randi.resolvent(A, ALPHA)[oj, oi]
    g_bio = gamma(A, lab_bio)
    gn = np.array([gamma(A, rand_lab(rng)) for _ in range(N_GAMMA_NULL)])
    s_bio = struct_loss(A, lab_bio, full)
    sn = np.array([struct_loss(A, rand_lab(rng), full) for _ in range(N_STRUCT_NULL)])
    same, ch, sel = merge_recovery(A, rng)
    return dict(gamma=g_bio, gamma_z=(g_bio - gn.mean()) / gn.std(),
                gamma_ratio=g_bio / gn.mean(),
                s_bio=s_bio, s_rand=sn.mean(), s_gap=sn.mean() - s_bio,
                same=same, chance=ch, enrich=same / ch if ch > 0 else np.nan, sel=sel)

files = ce.witvliet_files()
POOL = pool_mask([ce.witvliet_matrix(p, nodes, "all") for _, _, p in files])
print(f"{n} neurons, {len(np.unique(lab_bio))} classes, "
      f"{(sizes>1).sum()} multi-member; chance same-class rate {chance_same:.3f}; "
      f"shared pool {POOL.sum()}, {MERGES} merges\n")

rows_raw, rows_thin, sels_raw, sels_thin = [], [], [], []
for i, stage, p in files:
    A = ce.witvliet_matrix(p, nodes, "all")
    rng = np.random.default_rng(i)
    r = measure(A, rng); r.update(i=i, stage=stage, syn=A.sum()); rows_raw.append(r)
    sels_raw.append({frozenset(x) for x in r["sel"]})
    acc = []
    for t in range(THIN_REPS):
        At = thin(A, BUDGET, np.random.default_rng(1000 * i + t))
        acc.append(measure(At, np.random.default_rng(2000 * i + t)))
    m = {k: np.nanmean([a[k] for a in acc]) for k in
         ("gamma", "gamma_z", "gamma_ratio", "s_bio", "s_rand", "s_gap", "same", "chance", "enrich")}
    # spread over thinning replicates, so a panel can show what the control costs
    m.update({k + "_sd": np.nanstd([a[k] for a in acc]) for k in
              ("gamma", "gamma_z", "s_bio", "s_rand")})
    m.update(i=i, stage=stage, syn=A.sum()); rows_thin.append(m)
    sels_thin.append({frozenset(x) for x in acc[0]["sel"]})

def table(rows, title):
    print(f"\n{title}")
    print(f"  {'#':>2} {'stage':>6} {'syn':>6} | {'gamma':>7} {'z':>7} {'ratio':>6} | "
          f"{'bio loss':>8} {'rand':>6} {'gap':>6} | {'same%':>6} {'chance':>6} {'x':>5}")
    for r in rows:
        print(f"  {r['i']:>2} {r['stage']:>6} {r['syn']:>6.0f} | {r['gamma']:>7.4f} "
              f"{r['gamma_z']:>7.2f} {r['gamma_ratio']:>6.3f} | {r['s_bio']:>8.4f} "
              f"{r['s_rand']:>6.3f} {r['s_gap']:>6.3f} | {100*r['same']:>5.1f}% "
              f"{100*r['chance']:>5.1f}% {r['enrich']:>5.1f}")
    idx = np.array([r["i"] for r in rows], float)
    print("  trend vs age (Spearman rho, p):")
    for k in ("gamma", "gamma_z", "gamma_ratio", "s_bio", "s_gap", "enrich"):
        v = np.array([r[k] for r in rows], float)
        ok = np.isfinite(v)
        rho, pv = stats.spearmanr(idx[ok], v[ok])
        print(f"     {k:>12}  rho = {rho:+.3f}   p = {pv:.3f}")

table(rows_raw,  "RAW (density confounded -- do not interpret trends here)")
table(rows_thin, f"THINNED to {BUDGET} synapses ({THIN_REPS} reps averaged)")

# raw and thinned rows (thinned with their spread over replicates) for fig4c_dev_loss.py
OUT = Path(__file__).resolve().parents[1] / ("results/dev_sufficiency_noglia.csv" if EXCLUDE_GLIA
                                             else "results/dev_sufficiency.csv")
COLS = ["kind", "i", "stage", "syn", "gamma", "gamma_sd", "gamma_z", "gamma_z_sd",
        "s_bio", "s_bio_sd", "s_rand", "s_rand_sd", "s_gap", "same", "chance", "enrich"]
with OUT.open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=COLS, extrasaction="ignore")
    w.writeheader()
    for kind, rows in (("raw", rows_raw), ("thinned", rows_thin)):
        for r in rows:
            w.writerow({"kind": kind, **{k: v for k, v in r.items() if k in COLS}})
print("\nwrote", OUT)

print("\nSTABILITY of the label-free merge selection across animals")
for name, sels in (("raw", sels_raw), ("thinned", sels_thin)):
    J = [len(sels[a] & sels[b]) / len(sels[a] | sels[b])
         for a in range(8) for b in range(a + 1, 8)]
    adj = [len(sels[a] & sels[a+1]) / len(sels[a] | sels[a+1]) for a in range(7)]
    print(f"  {name:>7}: mean pairwise Jaccard {np.mean(J):.3f} "
          f"(min {np.min(J):.3f}, max {np.max(J):.3f}); adjacent-stage {np.mean(adj):.3f}")
