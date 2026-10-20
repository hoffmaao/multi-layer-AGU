r"""Schematic of the two experimental designs, in the poster's style.

Top row: the poster's runs, a two-layer column whose one interface moves
to make the basal ice thicker.  Bottom row: the interfaces seeded once at
every one of those heights, five layers, with only the assignment of
basal-ice rheology changing.  The arrows in each layer of the bottom row
trace each layer's mean along-flow speed from ``experiments.py``: every
layer moves as a block.

    python plot_schematic.py [--wavelength 6000] [--nx 40]
"""

import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

import poster_style as ps
from experiments import RESULTS, BASAL_THICKNESSES, WAVELENGTH, seeded_fractions
from ismip_hom_d import THICKNESS
from plot_results import FIGURES, load, percent


def column(ax, x0, width, edges, basal_layers, interface_lw=1.2):
    """One ice column: layer ``edges`` (fractions, bottom to top, including
    0 and 1), the bottom ``basal_layers`` of them basal ice."""
    for l, (bottom, top) in enumerate(zip(edges[:-1], edges[1:])):
        color = ps.BASAL_ICE if l < basal_layers else ps.ICE
        ax.add_patch(Rectangle((x0, bottom), width, top - bottom,
                               facecolor=color, edgecolor="none"))
    for z in edges[1:-1]:
        ax.plot([x0, x0 + width], [z, z], color="black", lw=interface_lw)
    ax.plot([x0, x0], [0, 1], color="black", lw=1.5)
    ax.plot([x0 + width, x0 + width], [0, 1], color="black", lw=1.5)


def title_box(ax, x, y, text):
    ax.text(x, y, text, ha="center", va="center", fontsize=18,
            fontweight="bold",
            bbox=dict(boxstyle="square,pad=0.35", facecolor="white",
                      edgecolor="black", linewidth=1.5))


def spread(z, gap):
    """Label heights near ``z`` (ascending), at least ``gap`` apart."""
    out = []
    for value in z:
        out.append(max(value, out[-1] + gap) if out else value)
    return out


def bed(ax, x0, x1, depth=0.035):
    ax.add_patch(Rectangle((x0, -depth), x1 - x0, depth, facecolor=ps.BED,
                           edgecolor="black", lw=1.5))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--wavelength", type=float, default=WAVELENGTH)
    parser.add_argument("--nx", type=int, default=40)
    args = parser.parse_args()
    directory = RESULTS / f"L{args.wavelength:g}m_nx{args.nx}"

    heights = sorted(h / THICKNESS for h in BASAL_THICKNESSES)
    fractions = seeded_fractions(heights)
    seeded = np.concatenate([[0.0], np.cumsum(fractions)])
    seeded[-1] = 1.0
    count = len(heights)
    width = 1.0

    fig, axes = plt.subplots(2, 1, figsize=(15, 10.5))

    # Two-layer runs: the interface moves.
    ax = axes[0]
    for i, h in enumerate(heights):
        column(ax, i * width, width, [0.0, h, 1.0], basal_layers=1,
               interface_lw=1.5)
        ps.label(ax, (i + 0.5) * width, 0.5 * (1 + h), percent(1 - h))
        ps.label(ax, (i + 0.5) * width, -0.12, percent(h))
    bed(ax, 0, count * width)
    ax.text(count * width + 0.12, 0.5,
            "Two layers: the\ninterface moves\nwith the basal ice,\n"
            "so the vertical\nresolution\nchanges too.",
            fontsize=17, va="center", ha="left")
    title_box(ax, 0.5 * count * width, 1.0,
              "Poster: varying basal ice thickness")

    # Five fixed layers: only the rheology moves.
    ax = axes[1]
    speeds = [load(directory, "fixed", f"k{k}")["velocity"][:, :-1].mean(axis=1)
              for k in range(1, count + 1)]
    vmax = max(s.max() for s in speeds)
    for i, k in enumerate(range(1, count + 1)):
        x0 = i * width
        column(ax, x0, width, seeded, basal_layers=k)
        # Each layer moves as a block: a stepped profile of the layer speeds.
        u = x0 + 0.08 * width + 0.6 * width * speeds[i] / vmax
        ax.stairs(u, seeded, baseline=None, orientation="horizontal",
                  color=ps.LABEL_EDGE, lw=2.5)
        ax.plot([x0 + 0.08 * width] * 2, [0, 1], color=ps.LABEL_EDGE, lw=1,
                ls=":")
        for l in range(len(fractions)):
            if fractions[l] < 0.1:
                continue
            z = 0.5 * (seeded[l] + seeded[l + 1])
            ax.add_patch(FancyArrowPatch(
                (x0 + 0.08 * width, z), (u[l], z), arrowstyle="-|>",
                mutation_scale=16, lw=2.0, color=ps.LABEL_EDGE,
                shrinkA=0, shrinkB=0))
        ps.label(ax, x0 + 0.5 * width, -0.12, percent(heights[i]))
    bed(ax, 0, count * width)
    middles = 0.5 * (seeded[:-1] + seeded[1:])
    for l, z in enumerate(spread(middles, 0.06)):
        ax.annotate(f"L={len(fractions) - l}", xy=(0, middles[l]),
                    xytext=(-0.2, z), fontsize=15, ha="center", va="center",
                    arrowprops=dict(arrowstyle="-", color="black", lw=0.8))
    ax.text(count * width + 0.12, 0.5,
            "Five layers seeded\nat every basal ice\nheight; only the\n"
            "rheology of each\nlayer changes.\nProfile: mean\nlayer speed.",
            fontsize=17, va="center", ha="left")
    title_box(ax, 0.5 * count * width, 1.0,
              "This work: fixed layers, varying rheology")

    for ax in axes:
        ax.text(-0.2, -0.12, "Basal ice", fontsize=15, ha="center",
                va="center")
        ax.set_xlim(-0.42, count * width + 1.45)
        ax.set_ylim(-0.17, 1.06)
        ax.axis("off")

    fig.tight_layout(h_pad=2.5)
    path = FIGURES / "schematic.png"
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(path, dpi=ps.DPI)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
