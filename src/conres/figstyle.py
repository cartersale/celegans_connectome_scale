"""Shared figure palette, rcParams and SVG export.

`use()` defaults to `svg.fonttype = "none"`, which writes text as editable
<text> elements in Arial, for panels assembled by hand in Illustrator. Such
files render correctly only where Arial is installed. `save(fig, path,
flat=True)` also writes `*_flat.svg` with glyphs converted to outlines, for a
self-contained file.

Check renders: when the environment variable CONRES_CHECK_DIR is set, scripts
also write 400-dpi PNG previews there (`check_path`, `save_check`); otherwise
none are written.
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

INK, GREY, FAINT, MASK = "#22282d", "#8b9196", "#c9ced2", "#dfe3e6"
STIM, MID, RESP = "#2a5fd6", "#2e9e52", "#d1332e"
LEFT, RIGHT, CLASS = "#2a5fd6", "#d1332e", "#2e9e52"
WALK = "#2a5fd6"

CHECK_DIR = Path(os.environ["CONRES_CHECK_DIR"]) if os.environ.get("CONRES_CHECK_DIR") else None


def check_path(name: str) -> Path | None:
    """Path for a PNG preview in CHECK_DIR, or None when previews are off."""
    if CHECK_DIR is None:
        return None
    CHECK_DIR.mkdir(parents=True, exist_ok=True)
    return CHECK_DIR / name


def save_check(fig, name: str, dpi: float = 400) -> None:
    """Write a PNG preview to CHECK_DIR if previews are on."""
    path = check_path(name)
    if path is not None:
        fig.savefig(path, dpi=dpi, facecolor="white")


def use(editable: bool = True) -> None:
    plt.rcParams.update({
        "font.family": "Arial",
        "svg.fonttype": "none" if editable else "path",
        # mathtext otherwise falls back to DejaVu Sans, which mismatches
        # Arial and may not exist on the machine doing the layout
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
        "svg.hashsalt": "conres",     # stable element ids across reruns
        "pdf.fonttype": 42,
        "path.simplify": True,
        "path.simplify_threshold": 0.1,
    })


def style(ax) -> None:
    """The house axes: two spines, hairline ticks."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.7)
        ax.spines[side].set_color(INK)
    ax.tick_params(labelsize=6.0, colors=INK, length=2.4, width=0.7, pad=1.6)


def save(fig, path: str | Path, flat: bool = False, check: str | Path | None = None,
         dpi: float | None = None) -> None:
    """Write the editable SVG, optionally an outlined twin, and a check render.

    `dpi` sets the resolution any rasterised artist is embedded at. An SVG is
    otherwise written at the figure's own 100 dpi, which is fine for pure line
    work and visibly soft for anything carrying `set_rasterized(True)`.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="white", dpi=dpi)
    if flat:
        with matplotlib.rc_context({"svg.fonttype": "path"}):
            fig.savefig(path.with_name(path.stem + "_flat.svg"), facecolor="white", dpi=dpi)
    if check is not None:
        fig.savefig(check, dpi=400, facecolor="white")
    print("wrote", path)
