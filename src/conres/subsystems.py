"""Annotation-based anatomical partitions of the C. elegans nervous system.

Source: WormAtlas cell list (data/raw/celegans/all_cell_info.csv), whose `Type`
column assigns every neuron one of 21 categories: sensory modality (amphid,
cephalic, mechanosensory, touch, phasmid, O2/CO2), interneuron layer (1-3,
category 4), motor neuron location (head, sublateral, ventral cord,
hermaphrodite-specific) and the pharyngeal system. No connectivity enters any
partition.

Three levels, each a many-to-one map of the type column, fixed before any
partition was scored:

  types        the 21 categories with the two "; ... in White et al., 1986"
               variants folded into their base type (19)
  subsystems   sensory split by modality group, interneurons by layer, motor
               neurons by head versus body, and the pharynx (9)
  divisions    sensory, interneuron, motor, pharyngeal (4)

The Cook node set also contains two muscles, `anal` and `sph`, which have no
neuronal type. They form their own group at every level, so the reported k is
one more than the counts above.
"""
from __future__ import annotations

import csv

import numpy as np

from conres import celegans as ce
from conres.paths import REPO_ROOT

INFO = REPO_ROOT / "data/raw/celegans/all_cell_info.csv"

SUBSYSTEM = {
    "Amphid": "amphid and gas sensory",
    "Amphid, nociceptive": "amphid and gas sensory",
    "O2, CO2, social signals, touch": "amphid and gas sensory",
    "Cephalic": "cephalic and labial sensory",
    "Mechanosensory": "body and tail sensory",
    "Touch": "body and tail sensory",
    "Phasmid": "body and tail sensory",
    "Layer 1 interneuron": "layer 1 interneuron",
    "Layer 2 interneuron": "layer 2 interneuron",
    "Layer 3 interneuron": "layer 3 and category 4 interneuron",
    "Category 4 interneuron": "layer 3 and category 4 interneuron",
    "Head motor neuron": "head and sublateral motor",
    "Sublateral motor neuron": "head and sublateral motor",
    "Ventral cord motor neuron": "body motor",
    "Hermaphrodite specific motor neuron": "body motor",
    "Pharyngeal interneuron": "pharynx",
    "Pharyngeal motor neuron": "pharynx",
    "Pharyngeal polymodal neuron": "pharynx",
    "Linker to pharynx": "pharynx",
}

DIVISION = {
    "amphid and gas sensory": "sensory",
    "cephalic and labial sensory": "sensory",
    "body and tail sensory": "sensory",
    "layer 1 interneuron": "interneuron",
    "layer 2 interneuron": "interneuron",
    "layer 3 and category 4 interneuron": "interneuron",
    "head and sublateral motor": "motor",
    "body motor": "motor",
    "pharynx": "pharynx",
}


def cell_types(nodes: list[str]) -> list[str]:
    """WormAtlas `Type` per node, text after ';' dropped; 'muscle' if absent."""
    with INFO.open() as fh:
        rows = list(csv.reader(fh))[1:]
    t = {ce.normalise_name(r[0]): r[1].split(";")[0].strip() for r in rows}
    return [t.get(n, "muscle") for n in nodes]


def partition(nodes: list[str], level: str) -> tuple[np.ndarray, list[str]]:
    """Integer labels and the group name of each label, for level in
    {'types', 'subsystems', 'divisions'}."""
    types = cell_types(nodes)
    if level == "types":
        keys = types
    else:
        sub = [SUBSYSTEM.get(t, "muscle") for t in types]
        keys = sub if level == "subsystems" else [DIVISION.get(s, "muscle") for s in sub]
    names = sorted(set(keys))
    lab = np.array([names.index(k) for k in keys])
    unknown = sorted({t for t in types if t != "muscle" and t not in SUBSYSTEM})
    if unknown and level != "types":
        raise ValueError(f"unmapped cell types: {unknown}")
    return lab, names
