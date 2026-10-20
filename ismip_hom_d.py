r"""ISMIP-HOM Experiment D for a layered column with a rheology per layer.

The friction of Pattyn et al. (2008), experiment D: a slab of constant
thickness, periodic in x with side walls in y, sliding on a linear (m = 1)
bed whose drag varies as

    beta^2(x) = 1000 + 1000 sin(2 pi x / L)   [Pa yr m^-1]

so beta^2 vanishes at x = 3L/4, the poster's "slippery spot".  The defaults
are the poster's geometry (Wright et al.): L = 6 km, 600 m of ice, a 2.3
degree slope.

The momentum balance is the multilayer dual form of
`multilayer <https://github.com/hoffmaao/multi-layer>`_, assembled here from
its action terms.  Each layer has its own flow law (n, A), and every
interface is closed in series: the half-layer below shears under its law,
the half-layer above under its own, at the common interface stress
(``multilayer.model.minimization.interlayer_power`` with the ``_above``
arguments).  So a column whose layers all carry one law is exactly the
uniform model, wherever its interfaces sit, and a rheology change sits
exactly on an interface.  ``interface="below"`` instead closes each
interface with the law of the layer below over the whole velocity jump,
the multilayer default the poster's runs used.

Units are icepack2's: MPa, m, yr.
"""

import math
from dataclasses import dataclass

import numpy as np
import firedrake
from firedrake import (
    Constant, Function, sin, dx, derivative,
    NonlinearVariationalProblem, NonlinearVariationalSolver,
)
from icepack2.constants import ice_density as rho_I, gravity as g
from multilayer.model import minimization, utilities


WAVELENGTH = 6000.0                    # m
THICKNESS = 600.0                      # m
SLOPE_DEG = 2.3
BETA2_MEAN = 1000.0                    # Pa yr m^-1
BETA2_AMPLITUDE = 1000.0
PA_TO_MPA = 1e-6

#: Stress at which the continuation holds each layer's fluidity fixed while
#: its exponent ramps from 1 to its target, so the path is gentle.
TAU_REF = 0.1                          # MPa

SOLVER_PARAMETERS = {
    "snes_type": "newtonls",
    "snes_max_it": 200,
    "snes_linesearch_type": "bt",
    "snes_divergence_tolerance": -1,
    "snes_rtol": 1e-10,
    "snes_atol": 1e-12,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}


@dataclass(frozen=True)
class FlowLaw:
    """Glen-type law: strain rate = A tau^n, A in MPa^-n yr^-1."""
    n: float
    A: float
    name: str = ""

    def rate_factor_at(self, n):
        """The rate factor for exponent n that gives this law's strain rate
        at TAU_REF."""
        return self.A * TAU_REF ** (self.n - n)


@dataclass
class Solution:
    x_hat: np.ndarray        # centreline sample points, x / L
    velocity: np.ndarray     # (num_layers, len(x_hat)) along-flow speed, m/yr
    fractions: tuple         # layer thickness fractions, bottom to top
    laws: tuple              # FlowLaw per layer, bottom to top
    wavelength: float        # domain length L, m

    @property
    def x(self):
        return self.x_hat * self.wavelength

    @property
    def surface(self):
        return self.velocity[-1]

    @property
    def base(self):
        return self.velocity[0]


INTERFACES = ("series", "below")


def solve(fractions, laws, wavelength=WAVELENGTH, thickness=THICKNESS,
          slope_deg=SLOPE_DEG, interface="series",
          beta2_amplitude=BETA2_AMPLITUDE, membrane_exponent=None,
          nx=40, steps=12,
          n_sample=101):
    r"""Solve experiment D for a column of ``len(fractions)`` layers.

    Parameters
    ----------
    fractions : sequence of float
        Thickness fraction of each layer, bottom to top; sums to 1.
    laws : sequence of FlowLaw
        Flow law of each layer, bottom to top.
    wavelength : float
        Domain length L [m], one period of the friction.
    thickness : float
        Ice thickness [m].
    slope_deg : float
        Surface slope [degrees], applied as a uniform body force.
    interface : {"series", "below"}
        How each interface is closed (see the module docstring).
    beta2_amplitude : float
        Amplitude of the friction's sinusoid [Pa yr m^-1]; 0 makes the
        friction uniform, a laterally uniform slab with a closed-form
        layered solution.
    membrane_exponent : float, optional
        Close the membrane stress with this exponent (at each layer's
        stress-matched rate factor) instead of the layer's own; the shear
        closures keep the layer laws.  For the uniform slab only: there
        the membrane stress is zero, so it does not change the solution,
        but the n > 1 membrane block vanishes with it and the Jacobian is
        singular.
    nx : int
        Cells per side of the periodic L x L mesh.
    steps : int
        Continuation steps from n = 1 to the target exponents.
    """
    fractions = tuple(float(f) for f in fractions)
    laws = tuple(laws)
    num_layers = len(fractions)
    if len(laws) != num_layers:
        raise ValueError(f"{len(laws)} flow laws for {num_layers} layers")
    if not math.isclose(sum(fractions), 1.0, rel_tol=0, abs_tol=1e-12):
        raise ValueError(f"layer fractions sum to {sum(fractions)}, not 1")
    if interface not in INTERFACES:
        raise ValueError(f"interface must be one of {INTERFACES}, got {interface!r}")
    if min(fractions) <= 0:
        raise ValueError("every layer needs a positive thickness")

    mesh = firedrake.PeriodicRectangleMesh(nx, nx, wavelength, wavelength,
                                           direction="x")
    x, _ = firedrake.SpatialCoordinate(mesh)
    Q = firedrake.FunctionSpace(mesh, "CG", 1)

    H = Constant(thickness)
    h_layers = utilities.layer_thicknesses(H, num_layers, fractions=list(fractions))
    surface = Function(Q)              # flat: the slope enters as a body force
    tan_alpha = Constant(math.tan(math.radians(slope_deg)))

    omega = 2 * math.pi / wavelength
    beta2 = Function(Q).interpolate(
        (Constant(BETA2_MEAN) + Constant(beta2_amplitude) * sin(omega * x))
        * Constant(PA_TO_MPA))
    # beta^2 touches zero at x = 3L/4 (free slip there); keep K finite.
    K = Constant(1.0) / (beta2 + Constant(1e-10))
    m = Constant(1.0)

    n = [Constant(1.0) for _ in laws]
    A = [Constant(law.rate_factor_at(1.0)) for law in laws]
    if membrane_exponent is None:
        n_membrane, A_membrane = n, A
    else:
        n_membrane = [Constant(membrane_exponent) for _ in laws]
        A_membrane = [Constant(law.rate_factor_at(membrane_exponent))
                      for law in laws]

    Z = utilities.create_function_space(mesh, num_layers)
    z = Function(Z)
    for l in range(num_layers):
        z.sub(3 * l).interpolate(firedrake.as_vector([Constant(10.0), Constant(0.0)]))
    fields = utilities.split_fields(firedrake.split(z), num_layers)

    action = Constant(0.0) * dx(mesh)
    for l, layer in enumerate(fields):
        action += minimization.viscous_power(
            membrane_stress=layer["membrane_stress"], thickness=h_layers[l],
            flow_law_coefficient=A_membrane[l],
            flow_law_exponent=n_membrane[l])
        action += minimization.momentum_balance(
            velocity=layer["velocity"], membrane_stress=layer["membrane_stress"],
            thickness=h_layers[l], surface=surface,
            basal_stress=layer["interlayer_stress"] if l == 0 else None,
            stress_above=(fields[l + 1]["interlayer_stress"]
                          if l < num_layers - 1 else None),
            stress_below=layer["interlayer_stress"] if l > 0 else None)
        action += rho_I * g * h_layers[l] * tan_alpha * layer["velocity"][0] * dx

    action += minimization.friction_power(
        basal_stress=fields[0]["interlayer_stress"],
        sliding_coefficient=K, sliding_exponent=m)

    for l in range(1, num_layers):
        above = ({"flow_law_coefficient_above": A[l],
                  "flow_law_exponent_above": n[l]}
                 if interface == "series" else {})
        action += minimization.interlayer_power(
            interlayer_stress=fields[l]["interlayer_stress"],
            thickness_below=h_layers[l - 1], thickness_above=h_layers[l],
            flow_law_coefficient=A[l - 1], flow_law_exponent=n[l - 1],
            **above)

    F = derivative(action, z)
    bcs = [firedrake.DirichletBC(Z.sub(3 * l).sub(1), 0, (1, 2))
           for l in range(num_layers)]
    problem = NonlinearVariationalProblem(
        F, z, bcs, form_compiler_parameters={"quadrature_degree": 8})
    solver = NonlinearVariationalSolver(problem, solver_parameters=SOLVER_PARAMETERS)

    for lam in np.linspace(0.0, 1.0, steps):
        for l, law in enumerate(laws):
            n_l = 1.0 + lam * (law.n - 1.0)
            n[l].assign(n_l)
            A[l].assign(law.rate_factor_at(n_l))
        solver.solve()

    x_hat = np.linspace(0.0, 1.0, n_sample)
    points = [[xi * wavelength, 0.5 * wavelength] for xi in x_hat]
    subs = z.subfunctions
    velocity = np.array([
        [u[0] for u in subs[3 * l].at(points)] for l in range(num_layers)])
    return Solution(x_hat, velocity, fractions, laws, wavelength)
