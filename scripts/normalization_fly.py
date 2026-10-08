"""Spectral radius of the class-level quotient relative to the full FlyWire
central-brain connectome, rho_Q / rho_A, for the lineage partitions.

Partitions: hemilineage, hemilineage x side (fw.tier_a), and hemilineage
subdivisions at f = 0.05, 0.2, 0.55 (fw.subsplit); each against 5 label
permutations. Spectral radii by power iteration (500 iterations). Companion to
normalization_control.py; the fly structural loss under the quotient's own
normalization is computed separately by fly_fragility.py --own-rho.

Inputs: conres.flywire (load_scope, load_adjacency).
Outputs: results/normalization_fly_rho_ratios.csv.
Usage: python scripts/normalization_fly.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from conres import flywire as fw   # noqa: E402

meta = fw.load_scope()
A = fw.load_adjacency(meta)
rhoA = fw.spectral_radius(A, n_iter=500)
rng = np.random.default_rng(0)
parts = {"hemilineage": fw.tier_a(meta, False), "hemilineage x side": fw.tier_a(meta, True)}
for f in (0.05, 0.2, 0.55):
    parts[f"hemilineage subdivisions f={f}"] = fw.subsplit(meta, f)
rows = []
print(f"fly central brain: n={A.shape[0]:,}, rho_A={rhoA:.2f}")
for name, lab in parts.items():
    lab = np.unique(lab, return_inverse=True)[1]
    r = fw.spectral_radius(fw.quotient(A, lab), n_iter=500) / rhoA
    rnd = [fw.spectral_radius(fw.quotient(A, rng.permutation(lab)), n_iter=500) / rhoA for _ in range(5)]
    rows.append({"partition": name, "k": int(lab.max() + 1), "ratio": r,
                 "random_mean": np.mean(rnd), "random_min": np.min(rnd), "random_max": np.max(rnd)})
    print(f"  {name:>34} k={lab.max() + 1:6d}: rho_Q/rho_A {r:.3f}; random {np.mean(rnd):.3f} "
          f"[{np.min(rnd):.3f}, {np.max(rnd):.3f}]", flush=True)
pd.DataFrame(rows).to_csv(ROOT / "results/normalization_fly_rho_ratios.csv", index=False)
