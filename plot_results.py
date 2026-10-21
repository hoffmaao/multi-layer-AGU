r"""Figures for the runs of ``experiments.py``, in the poster's style.

Layers are numbered from the surface down, as on the poster: L = 1 is the
surface layer, the highest L the layer on the bed.  Panel (a) is the
along-flow velocity of the surface layer (solid) and of the bed layer
(dashed) on the centreline, (b) their difference, (c) their ratio.

    python plot_results.py [--wavelength 6000] [--nx 40]
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import poster_style as ps
from experiments import RESULTS, BASAL_THICKNESSES, WAVELENGTH
from ismip_hom_d import THICKNESS

FIGURES = Path(__file__).resolve().parent / "figures"


def load(directory, experiment, tag):
    with np.load(directory / f"{experiment}_{tag}.npz") as data:
        run = {key: data[key] for key in data.files}
    run["x"] = run["x_hat"] * float(run["wavelength"])
    run["surface"] = run["velocity"][-1]
    run["bed"] = run["velocity"][0]
    run["layers"] = run["velocity"].shape[0]
    return run


def percent(fraction):
    return f"{100 * fraction:.1f}%"


def three_panel(runs, colors, key_labels, path, key_title=None):
    """Runs listed in key order, top first."""
    layers = {run["layers"] for run in runs}
    bed = f"L={max(layers)}" if len(layers) == 1 else "L=bed"

    fig = plt.figure(figsize=(ps.FIGSIZE[0] + 1.6, ps.FIGSIZE[1] + 0.7))
    grid = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.22],
                            left=0.06, right=0.985, bottom=0.12, top=0.84,
                            wspace=0.32)
    axes = [fig.add_subplot(grid[0, i]) for i in range(3)]
    key = fig.add_subplot(grid[0, 3])

    ax = axes[0]
    for run, color in zip(runs, colors):
        ax.plot(run["x"], run["surface"], color=color, lw=3)
        ax.plot(run["x"], run["bed"], color=color, lw=3, ls="--")
    ax.set_title("(a) Horizontal Velocity")
    ax.set_ylabel("Velocity (m/yr)")

    ax = axes[1]
    shears = [run["surface"] - run["bed"] for run in runs]
    for i in np.argsort([-s.max() for s in shears]):
        color = colors[i]
        alpha = ps.FILL_ALPHA.get(color, 1.0)
        ax.fill_between(runs[i]["x"], 0, shears[i], color=color, alpha=alpha,
                        lw=0)
        ax.plot(runs[i]["x"], shears[i], color=color, lw=1.5)
    ax.set_title("(b) Vertical Shear")
    ax.set_ylabel(f"L=1 - {bed} (m/yr)")

    ax = axes[2]
    for run, color in zip(runs, colors):
        ax.plot(run["x"], run["surface"] / run["bed"], "o", color=color, ms=6)
    ax.set_title("(c) Surface-to-Basal Velocity Ratio")
    ax.set_ylabel(f"L=1 / {bed}")

    for ax in axes:
        ax.set_xlabel("Domain (m)")
        ax.set_xticks(np.arange(0, run["x"].max() + 1, 2000))
        ps.mark_slippery_spot(ax, x=0.75 * float(runs[0]["wavelength"]))

    ps.color_key(key, colors, key_labels)
    if key_title:
        key.set_title(key_title, fontsize=14)

    handles = [Line2D([], [], color="black", lw=3),
               Line2D([], [], color="black", lw=3, ls="--"),
               Line2D([], [], color=ps.SLIPPERY, marker="*", ms=18, ls="none")]
    names = ["Layer 1 (L=1)", f"Layer {bed[2:]} ({bed})", "Slippery spot"]
    fig.legend(handles, names, loc="upper center", ncol=3, frameon=True,
               edgecolor="black", fancybox=False, bbox_to_anchor=(0.47, 1.0),
               handlelength=2.5)
    fig.savefig(path, dpi=ps.DPI)
    plt.close(fig)
    print(f"wrote {path}")


def summary(directory, path):
    """Mean surface speed against basal ice thickness, for every design."""
    heights = sorted(h / THICKNESS for h in BASAL_THICKNESSES)
    metres = [h * THICKNESS for h in heights]

    def mean_surface(run):
        return run["surface"][:-1].mean()   # the last sample repeats x = 0

    fixed = [mean_surface(load(directory, "fixed", f"k{k + 1}"))
             for k in range(len(heights))]
    variable, poster, split = (
        [mean_surface(load(directory, e, f"h{h:.4f}")) for h in heights]
        for e in ("variable", "poster", "split"))
    uniform5 = mean_surface(load(directory, "fixed", "k0"))

    fig, ax = plt.subplots(figsize=(14, 6.5))
    ax.plot(metres, split, "o--", color=ps.UNIFORM, lw=3, ms=9,
            label="Uniform, two layers (interface moves)")
    ax.axhline(uniform5, color=ps.UNIFORM, lw=3,
               label="Uniform, five fixed layers")
    ax.plot(metres, poster, "s:", color=ps.THICKNESS_COLORS[0], lw=3, ms=9,
            label="Composite, two layers (poster closure)")
    ax.plot(metres, variable, "o--", color=ps.THICKNESS_COLORS[2], lw=3, ms=9,
            label="Composite, two layers (series closure)")
    ax.plot(metres, fixed, "o-", color=ps.THICKNESS_COLORS[3], lw=3, ms=9,
            label="Composite, five fixed layers")
    ax.set_xscale("log")
    ax.set_xticks(metres)
    ax.set_xticklabels([f"{m:g}" for m in metres])
    ax.minorticks_off()
    ax.set_xlabel("Basal ice thickness (m)")
    ax.set_ylabel("Mean surface velocity (m/yr)")
    ax.set_title("Surface (L=1) velocity vs. basal ice thickness")
    ax.legend(fontsize=13, loc="center left", bbox_to_anchor=(1.02, 0.5),
              frameon=True)
    fig.tight_layout()
    fig.savefig(path, dpi=ps.DPI)
    plt.close(fig)
    print(f"wrote {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--wavelength", type=float, default=WAVELENGTH)
    parser.add_argument("--nx", type=int, default=40)
    args = parser.parse_args()
    directory = RESULTS / f"L{args.wavelength:g}m_nx{args.nx}"
    FIGURES.mkdir(exist_ok=True)

    heights = sorted(h / THICKNESS for h in BASAL_THICKNESSES)
    thickness_labels = [percent(h) for h in heights]
    colors = list(ps.THICKNESS_COLORS)

    fixed = [load(directory, "fixed", f"k{k + 1}") for k in range(len(heights))]
    three_panel(fixed + [load(directory, "fixed", "k0")], colors + [ps.UNIFORM],
                thickness_labels + ["Uniform"],
                FIGURES / "fixed_layers.png")

    for experiment, name, palette, title in (
            ("variable", "two_layer_series", colors, None),
            ("poster", "two_layer_poster", colors, None),
            ("split", "two_layer_uniform", list(ps.UNIFORM_COLORS), "Interface")):
        runs = [load(directory, experiment, f"h{h:.4f}") for h in heights]
        three_panel(runs, palette, thickness_labels, FIGURES / f"{name}.png",
                    key_title=title)

    summary(directory, FIGURES / "surface_vs_thickness.png")


if __name__ == "__main__":
    main()
