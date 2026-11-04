r"""Soft basal ice in ISMIP-HOM D, with the discretisation held fixed.

The poster runs (Wright et al., AGU 2026) used a two-layer column and moved
its one interface to make the soft basal layer thicker.  That changes two
things at once: the rheology of the ice near the bed, and where the model
resolves vertical shear.  Here the interfaces are seeded once, at every
basal-layer height the poster used, and only the assignment of rheology to
layers changes:

``fixed``     the seeded column, every seeded layer split into
              FIXED_SUBDIVISIONS equal layers (20 layers); the bottom k
              seeded layers soft, the rest cold, k = 0 .. len(INTERFACE_HEIGHTS).
``variable``  the poster's two-layer column, interface at each height,
              basal ice below and stratigraphic ice above, interfaces in
              series.
``poster``    the same, with each interface closed by the law of the
              layer below over the whole velocity jump, as the poster's
              runs were.
``split``     the two-layer columns with stratigraphic ice throughout:
              the response to moving the interface alone, with no
              rheology contrast.  ``fixed`` with k = 0 is its reference.
``refine``    the seeded column with every layer whole (5 layers) and
              split in 2 (10 layers), with no basal ice and with basal ice
              in the bottom 200 m: with ``fixed`` (20 layers), the vertical
              resolution check.  Runs only when asked for with
              ``--only refine``.
``reference`` the first-order (Blatter-Pattyn) solution of each column
              (``multilayer.first_order``, independent of the layered
              model), with no basal ice and with each basal-ice thickness:
              what the layered runs converge to.

The three two-layer sets also run at LIMIT_THICKNESSES, basal ice thinning
to 0.1 m: as it vanishes a column should recover the stratigraphic-only
answer (``split``), and ``variable`` does while ``poster`` does not.

Each run is saved to ``results/<experiment>_<tag>.npz``; a run already
on disk is skipped unless ``--force``.
"""

import argparse
import math
import time
from pathlib import Path

import numpy as np

from ismip_hom_d import (FlowLaw, solve, THICKNESS, WAVELENGTH, SLOPE_DEG,
                         BETA2_MEAN, BETA2_AMPLITUDE, PA_TO_MPA)


#: Basal ice thicknesses of the poster's runs [m] (Doyle et al. 2018 bracket
#: 8-22 m at Store), and the column they sit in.
BASAL_THICKNESSES = (200.0, 100.0, 22.0, 9.0)
#: Heights of the basal-ice top, as fractions of the ice thickness above the
#: bed: 33.3, 16.7, 3.7 and 1.5 %.
INTERFACE_HEIGHTS = tuple(h / THICKNESS for h in BASAL_THICKNESSES)
#: Thinner basal ice for the two-layer runs [m]: the limit of no basal ice.
LIMIT_THICKNESSES = (3.0, 1.0, 0.3, 0.1)

#: The poster's rheology table, MPa^-n yr^-1.
COLD = FlowLaw(n=1.8, A=100.0, name="stratigraphic")
SOFT = FlowLaw(n=4.0, A=1000.0, name="basal")
RESULTS = Path(__file__).resolve().parent / "results"
#: Equal layers every seeded layer is split into in the ``fixed`` set.  The
#: seeded layers alone (5) leave the top layer 400 m thick: speeds 15-25 %
#: too fast and the basal-ice effect 5 points too weak at 200 m; split in 4
#: they are within about 1 % of the first-order solution.
FIXED_SUBDIVISIONS = 4
#: Subdivisions of every seeded layer in the ``refine`` set.
REFINEMENTS = (1, 2)
#: Experiments run only when named with ``--only``.
OPT_IN = ("refine",)


def seeded_fractions(heights=INTERFACE_HEIGHTS):
    """Layer thickness fractions, bottom to top, for interfaces at
    ``heights`` (fractions of the thickness above the bed)."""
    z = sorted(set(float(h) for h in heights))
    if z[0] <= 0 or z[-1] >= 1:
        raise ValueError(f"interface heights must lie in (0, 1), got {heights}")
    edges = [0.0, *z, 1.0]
    return tuple(top - bottom for bottom, top in zip(edges[:-1], edges[1:]))


def two_layer_tag(metres):
    """Run tag of a two-layer column with ``metres`` of basal ice."""
    return f"b{metres:g}m"


def runs(heights=INTERFACE_HEIGHTS, soft=SOFT, cold=COLD):
    """Every run as (experiment, tag, fractions, laws, basal fraction)."""
    fractions = seeded_fractions(heights)
    tops = np.cumsum(fractions)
    out = []
    m = FIXED_SUBDIVISIONS
    fixed = tuple(f / m for f in fractions for _ in range(m))
    for k in range(len(fractions)):
        laws = (soft,) * (k * m) + (cold,) * (len(fixed) - k * m)
        out.append(("fixed", f"k{k}", fixed, laws,
                    float(tops[k - 1]) if k else 0.0, "series"))
    for metres in sorted(BASAL_THICKNESSES + LIMIT_THICKNESSES):
        h = metres / THICKNESS
        tag = two_layer_tag(metres)
        out.append(("variable", tag, (h, 1 - h), (soft, cold), float(h), "series"))
        out.append(("poster", tag, (h, 1 - h), (soft, cold), float(h), "below"))
        out.append(("split", tag, (h, 1 - h), (cold, cold), float(h), "series"))
    k_basal = int(np.searchsorted(tops, max(BASAL_THICKNESSES) / THICKNESS - 1e-12)) + 1
    for m in REFINEMENTS:
        refined = tuple(f / m for f in fractions for _ in range(m))
        for k in (0, k_basal * m):
            laws = (soft,) * k + (cold,) * (len(refined) - k)
            out.append(("refine", f"x{m}_k{k}", refined, laws,
                        float(tops[k_basal - 1]) if k else 0.0, "series"))
    return out


def reference_runs():
    """First-order references as (tag, basal fraction): no basal ice, and
    every basal-ice thickness of the layered runs."""
    return [("none", 0.0)] + [
        (two_layer_tag(m), m / THICKNESS)
        for m in sorted(BASAL_THICKNESSES + LIMIT_THICKNESSES)]


def solve_reference(basal_fraction, wavelength, soft=SOFT, cold=COLD,
                    n_sample=101):
    """First-order solution of the slab with basal ice below
    ``basal_fraction``: x_hat, and the velocity on a grid of height
    fractions zeta (rows) at each x_hat."""
    from firedrake import Constant, sin
    from multilayer import first_order
    w = 2 * math.pi / wavelength
    laws = ([(basal_fraction, soft.n, soft.A)] if basal_fraction else []) \
        + [(1.0, cold.n, cold.A)]
    solution = first_order.solve(
        wavelength, lambda x: Constant(THICKNESS) + 0 * x,
        lambda x: Constant(-THICKNESS) + 0 * x,
        math.tan(math.radians(SLOPE_DEG)), laws,
        beta2=lambda x: (BETA2_MEAN + BETA2_AMPLITUDE * sin(w * x)) * PA_TO_MPA)
    x_hat = np.linspace(0.0, 1.0, n_sample)
    zeta = np.unique(np.concatenate([np.linspace(0.0, 1.0, 601),
                                     np.linspace(0.0, 0.05, 251)]))
    return x_hat, zeta, np.array([solution.at(x_hat, z) for z in zeta])


def save(path, solution, basal_fraction, interface, wavelength, nx, seconds):
    np.savez(
        path,
        x_hat=solution.x_hat,
        velocity=solution.velocity,
        fractions=np.array(solution.fractions),
        n=np.array([law.n for law in solution.laws]),
        A=np.array([law.A for law in solution.laws]),
        law_names=np.array([law.name for law in solution.laws]),
        basal_fraction=basal_fraction,
        interface=interface,
        wavelength=wavelength,
        nx=nx,
        seconds=seconds,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--wavelength", type=float, default=WAVELENGTH,
                        help="domain length L [m] (default %(default)g)")
    parser.add_argument("--nx", type=int, default=40,
                        help="cells per side (default %(default)d)")
    parser.add_argument("--only", nargs="+",
                        choices=("fixed", "variable", "poster", "split", "refine",
                                 "reference"),
                        help="run only these experiments; 'refine' is slow "
                             "(10+ min per 20-layer run) and runs only when named")
    parser.add_argument("--force", action="store_true",
                        help="re-run results already on disk")
    parser.add_argument("--list", action="store_true",
                        help="print the runs and exit")
    args = parser.parse_args()

    out_dir = RESULTS / f"L{args.wavelength:g}m_nx{args.nx}"
    todo = [r for r in runs()
            if (r[0] in args.only if args.only else r[0] not in OPT_IN)]
    if args.list:
        for experiment, tag, fractions, laws, basal, interface in todo:
            layers = ", ".join(f"{f:.4f} {law.name}" for f, law in zip(fractions, laws))
            print(f"{experiment:9s} {tag:8s} {interface:6s} basal {basal:.4f} | {layers}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.only or "reference" in args.only:
        for tag, basal in reference_runs():
            path = out_dir / f"reference_{tag}.npz"
            if path.exists() and not args.force:
                print(f"skip  {path.name}", flush=True)
                continue
            start = time.perf_counter()
            x_hat, zeta, velocity = solve_reference(basal, args.wavelength)
            np.savez(path, x_hat=x_hat, zeta=zeta, velocity=velocity,
                     basal_fraction=basal, wavelength=args.wavelength)
            print(f"done  {path.name}: surface mean {velocity[-1][:-1].mean():8.3f} "
                  f"m/yr ({time.perf_counter() - start:.0f} s)", flush=True)
    for experiment, tag, fractions, laws, basal, interface in todo:
        path = out_dir / f"{experiment}_{tag}.npz"
        if path.exists() and not args.force:
            print(f"skip  {path.name}", flush=True)
            continue
        start = time.perf_counter()
        solution = solve(fractions, laws, wavelength=args.wavelength,
                         interface=interface, nx=args.nx)
        seconds = time.perf_counter() - start
        save(path, solution, basal, interface, args.wavelength, args.nx, seconds)
        print(f"done  {path.name}: surface mean {solution.surface.mean():8.3f} "
              f"max {solution.surface.max():8.3f} m/yr, base mean "
              f"{solution.base.mean():8.3f} m/yr ({seconds:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
