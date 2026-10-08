"""Plot URADL's calcium response to one URYDL stimulation event.

A single event, not a trial average: URYDL was stimulated with URADL identified
in several events, and the response is carried by one of them, so an event
mean would sit near zero. Randi et al. detect the pair from their kernel fit
across events, not from a trial-averaged dF/F.

Event: recording 111, stimulation index 33 (asserted to be URYDL).
dF/F uses F0 = mean fluorescence over the 5 s before the stimulus; the window
shown is -5 to +26 s.

Inputs:  data/raw/randi/ex/exported_data/111_{gcamp,t,labels,stim_volume_i,stim_neurons}.txt.
Outputs: figures/fig1/fig1a_response_trace.svg, plus a PNG preview in $CONRES_CHECK_DIR (if set).
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EX = ROOT / "data/raw/randi/ex/exported_data"
OUT = ROOT / "figures/fig1"
CHECKDIR = Path(os.environ["CONRES_CHECK_DIR"]) if os.environ.get("CONRES_CHECK_DIR") else None

plt.rcParams.update({
    "font.family": "Arial",
    "svg.fonttype": "none",
    "mathtext.fontset": "custom",   # keep alpha, rho and superscripts in Arial
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
    "pdf.fonttype": 42,
})

STIM, RESP = "#2a5fd6", "#d1332e"
INK, GREY = "#22282d", "#8b9196"

REC, EVENT, POST = "111", 33, "URADL"
PRE_S, POST_S, BASE_S = 5.0, 26.0, 5.0


def main() -> None:
    g = np.loadtxt(EX / f"{REC}_gcamp.txt")
    t = np.loadtxt(EX / f"{REC}_t.txt")
    labels = [ln.rstrip("\n").strip() for ln in (EX / f"{REC}_labels.txt").open()]
    vols = [int(x) for x in (EX / f"{REC}_stim_volume_i.txt").open() if x.strip()]
    stims = [int(x) for x in (EX / f"{REC}_stim_neurons.txt").open() if x.strip()]
    assert labels[stims[EVENT]] == "URYDL", labels[stims[EVENT]]

    t0 = t[vols[EVENT]]
    tt = t - t0
    y = g[:, labels.index(POST)]
    f0 = np.nanmean(y[(tt >= -BASE_S) & (tt < 0)])
    dff = (y - f0) / f0

    m = (tt >= -PRE_S) & (tt <= POST_S)
    x, v = tt[m], dff[m]

    fig = plt.figure(figsize=(2.1, 1.95))
    ax = fig.add_axes([0.22, 0.28, 0.74, 0.54])

    ax.axhline(0, color=GREY, lw=0.7, ls=(0, (3.0, 2.4)), zorder=1)
    ax.plot(x, v, color=RESP, lw=1.1, zorder=3, solid_joinstyle="round")

    ax.set_xlim(-PRE_S, POST_S)
    ax.set_ylim(-1, 3)
    ax.set_xticks([0, 5, 10, 15, 20, 25])
    ax.set_yticks([-1, 0, 1, 2, 3])
    ax.set_xlabel("Time (s)", fontsize=6.5, color=INK, labelpad=2, x=0.66)
    ax.set_ylabel("$\\Delta$F/F", fontsize=6.5, color=INK, labelpad=2)
    ax.tick_params(labelsize=6.0, colors=INK, length=2.4, width=0.7, pad=1.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.7)
        ax.spines[side].set_color(INK)

    # the stimulus, marked below the tick labels so it does not sit on the 0
    blend = ax.get_xaxis_transform()          # x in data, y in axes fraction
    ax.annotate("", xy=(0, -0.135), xytext=(0, -0.235), xycoords=blend,
                textcoords=blend, annotation_clip=False,
                arrowprops=dict(arrowstyle="-|>", mutation_scale=6.5,
                                color=STIM, lw=1.0, shrinkA=0, shrinkB=0))
    ax.text(0, -0.26, "Stim\nURYDL", fontsize=6.0, color=STIM, ha="center",
            va="top", linespacing=1.2, transform=blend, clip_on=False)

    ax.set_title("Measured response in URADL", fontsize=6.8, color=RESP,
                 weight="bold", pad=9)
    ax.text(0.5, 1.03, "GCaMP, one stimulation event", fontsize=6.0, color=GREY,
            ha="center", va="bottom", transform=ax.transAxes)

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig1a_response_trace.svg", facecolor="white")
    if CHECKDIR is not None:
        fig.savefig(CHECKDIR / "check_fig1A_response_trace.png", dpi=400,
                    facecolor="white")
    print(f"peak dF/F = {np.nanmax(v):.2f} at t = {x[np.nanargmax(v)]:.1f} s; F0 = {f0:.2f}")
    print("wrote", OUT / "fig1a_response_trace.svg")


if __name__ == "__main__":
    main()
