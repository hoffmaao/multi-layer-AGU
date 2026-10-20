r"""The look of Wright et al.'s poster, so new figures sit beside its panels.

Colours are the matplotlib named colours the poster's figures use; the
label boxes and schematic fills are sampled from the poster itself.
"""

import matplotlib
import matplotlib.patches

matplotlib.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
})

#: Uniform vs. composite rheology (poster, top results row).
UNIFORM = "lime"
COMPOSITE = "darkgreen"
#: Basal ice thickness, thinnest to thickest (poster, bottom results row).
THICKNESS_COLORS = ("violet", "mediumorchid", "darkviolet", "indigo")
#: Uniform rheology with the interface moved, lowest interface first:
#: the poster's lime family, stopping short of its composite dark green.
UNIFORM_COLORS = ("palegreen", "lime", "limegreen", "forestgreen")
#: Fill opacity of the vertical-shear panels.
FILL_ALPHA = {UNIFORM: 0.9, COMPOSITE: 0.7}

SLIPPERY = "red"

#: Schematic fills (poster, "Uniform vs. Composite Rheology" panels).
ICE = "#cbe6f0"
BASAL_ICE = "#cfbe79"
BED = "#9a9a9a"

#: Label boxes.
LABEL_FACE = "#eaf3de"
LABEL_EDGE = "#042433"

FIGSIZE = (16.87, 6.9)          # the poster's results panels, at dpi 100
DPI = 200


def mark_slippery_spot(ax, x=4500.0, size=22):
    """The red star on the x axis at the friction minimum."""
    ax.plot([x], [0], marker="*", ms=size, color=SLIPPERY, ls="none",
            transform=ax.get_xaxis_transform(), clip_on=False, zorder=10)


def label(ax, x, y, text, fontsize=16, transform=None, **kwargs):
    """A text label in the poster's rounded, pale green box."""
    return ax.text(
        x, y, text, fontsize=fontsize, ha=kwargs.pop("ha", "center"),
        va=kwargs.pop("va", "center"),
        transform=transform if transform is not None else ax.transData,
        bbox=dict(boxstyle="round,pad=0.3,rounding_size=0.3",
                  facecolor=LABEL_FACE, edgecolor=LABEL_EDGE, linewidth=1.5),
        **kwargs)


def color_key(ax, colors, labels):
    """The poster's key: a column of colour blocks, first entry on top,
    each with its label in a box."""
    count = len(colors)
    for i, (color, text) in enumerate(zip(colors, labels)):
        top = 1.0 - i / count
        ax.add_patch(matplotlib.patches.Rectangle(
            (0.0, top - 1.0 / count), 1.0, 1.0 / count, facecolor=color,
            edgecolor="none", transform=ax.transAxes))
        label(ax, 0.5, top - 0.35 / count, text, fontsize=14,
              transform=ax.transAxes)
    ax.add_patch(matplotlib.patches.Rectangle(
        (0, 0), 1, 1, fill=False, edgecolor="black", lw=1.0,
        transform=ax.transAxes))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


__all__ = ["UNIFORM", "COMPOSITE", "THICKNESS_COLORS", "UNIFORM_COLORS", "FILL_ALPHA",
           "SLIPPERY", "ICE", "BASAL_ICE", "BED", "LABEL_FACE", "LABEL_EDGE",
           "FIGSIZE", "DPI", "mark_slippery_spot", "label", "color_key"]
