"""Compare the cell classes used here with the class field of the Randi atlas.

conres.celegans.assign_classes derives classes from neuron names. The atlas
cell table (data/raw/randi/cells.json) carries its own `class` per cell. The
two labelings are compared as partitions of the neurons present in both, so
differently spelled class names do not count as disagreements: the adjusted
Rand index is printed, with every class of either labeling that is not a
single class of the other.

Inputs: Cook node list (conres.celegans), data/raw/randi/cells.json.
Outputs: results/diagnostics/class_assignment_check.log.
Usage: python scripts/diagnostics/class_assignment_check.py
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from conres import celegans as ce                 # noqa: E402
from conres.paths import tee_stdout               # noqa: E402

tee_stdout(__file__)


def main() -> None:
    _, nodes = ce.build_matrix("all", neurons_only=True)
    ours = ce.assign_classes(nodes)
    atlas = {ce.normalise_name(c["name"]): c["class"]
             for c in json.load((ROOT / "data/raw/randi/cells.json").open())}

    shared = [n for n in map(ce.normalise_name, nodes) if n in atlas]
    missing = sorted(set(map(ce.normalise_name, nodes)) - set(shared))
    print(f"nodes {len(nodes)}; in the atlas cell table {len(shared)}; "
          f"not in it: {missing}")
    a = [ours[n] for n in shared]
    b = [atlas[n] for n in shared]
    print(f"classes on the shared cells: ours {len(set(a))}, atlas {len(set(b))}")
    print(f"adjusted Rand index = {adjusted_rand_score(a, b):.4f}")

    split = collections.defaultdict(set)
    merged = collections.defaultdict(set)
    for x, y in zip(a, b):
        split[x].add(y)
        merged[y].add(x)
    bad = [(k, sorted(v)) for k, v in split.items() if len(v) > 1]
    bad2 = [(k, sorted(v)) for k, v in merged.items() if len(v) > 1]
    print(f"our classes spanning several atlas classes: {bad or 'none'}")
    print(f"atlas classes spanning several of ours: {bad2 or 'none'}")
    sizes = collections.Counter(collections.Counter(ours.values()).values())
    multi = sum(v for k, v in sizes.items() if k > 1)
    print(f"all nodes: {len(set(ours.values()))} classes, {multi} with more than one member")


if __name__ == "__main__":
    main()
