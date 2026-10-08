"""Soma coordinates for the listed cells in the VirtualWorm model, in micrometres.

Runs under Blender, not the project interpreter:

    blender -b data/raw/virtualworm/Virtual_Worm_February_2012.blend \\
            --python scripts/export_soma_um.py -- <spec.json> [out.json]

spec.json holds {"neurons": [...]}: the cell names to locate. A soma is taken
as the densest vertex cluster of the cell's mesh (the vertex with most
neighbours within 0.05 Blender units, via a KD-tree), not the mesh centre:
a long process moves the bounding-box centre away from the cell body (AVF's
process runs the length of the ventral cord, so its box centres on mid-body
while its soma is in the retrovesicular ganglion). conres.worm reads the body
meshes directly from the blend file; the per-cell KD-tree search is done here,
once, and cached as JSON.

Output: {name: [ML, AP, DV]} in micrometres in the NeuroML frame (scale 97.813,
AP offset -4.21), the same conversion conres.worm applies to the body surface,
so cells and cuticle share one coordinate system. DV is in the posed model.
Default output path data/raw/virtualworm/soma_um.json (read by conres.worm);
names absent from the blend are printed as unmatched. OpenWorm's NeuroML
release omits glia and sheath cells; this file is the source for those.
"""
import json
import sys
from pathlib import Path

import bpy
ARGS = sys.argv[sys.argv.index("--") + 1:]
SPEC = Path(ARGS[0])
OUT = Path(ARGS[1]) if len(ARGS) > 1 else Path("data/raw/virtualworm/soma_um.json")

spec = json.loads(SPEC.read_text())
NEURONS = set(spec["neurons"])
CONTEXT = {"Pharynx", "Intestine"}


neuron_objs = {}
for o in bpy.data.objects:
    if o.type == "MESH" and o.name.split(".")[0] in NEURONS:
        neuron_objs[o.name.split(".")[0]] = o

missing = sorted(NEURONS - set(neuron_objs))
print(f"locating {len(neuron_objs)}/{len(NEURONS)} somata")
if missing:
    print("  unmatched:", missing)

from mathutils.kdtree import KDTree
bpy.context.view_layer.update()


def soma(obj, radius=0.05):
    """World-space vertex with the most neighbours within `radius`: the cell body."""
    vs = [obj.matrix_world @ v.co for v in obj.data.vertices]
    kd = KDTree(len(vs))
    for i, v in enumerate(vs):
        kd.insert(v, i)
    kd.balance()
    return max(vs, key=lambda v: len(kd.find_range(v, radius)))


# Blender units -> micrometres in the NeuroML frame (as in conres.worm)
SCALE, AP_OFFSET = 97.813, -4.21
anchors = {}
for name, o in neuron_objs.items():
    c = soma(o)
    anchors[name] = [round(c.x * SCALE, 3),                       # ML
                     round(c.y * SCALE + AP_OFFSET, 3),           # AP
                     round(c.z * SCALE, 3)]                       # DV (posed)
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(anchors, indent=1, sort_keys=True))
print("wrote", OUT, len(anchors), "somata")
