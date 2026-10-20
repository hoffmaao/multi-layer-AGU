r"""The layered momentum balance against its closed form on a uniform slab.

With uniform friction the slab is laterally uniform, the membrane stresses
vanish, and each interface carries the weight of the ice above it:

    S(z) = rho g tan(alpha) (H - z).

The bottom layer slides at u_1 = S(0) / beta^2, and each interface adds the
shear of the two half-layers that meet there.  Glen's law in the poster's
convention, strain rate A tau^n with du/dz = 2 A tau^n, sheared over a
half-layer h/2, gives h A S^n per half-layer, so across the interface at
z_l between layers l and l+1

    series:  u_{l+1} - u_l = h_l A_l S^n_l + h_{l+1} A_{l+1} S^n_{l+1}
    below:   u_{l+1} - u_l = (h_l + h_{l+1}) A_l S^n_l

The solver has to reproduce these to solver tolerance, layer by layer.
The membrane stress is zero here, so the membrane closure is taken linear
(``membrane_exponent=1``): at M = 0 an n > 1 membrane block vanishes and
the Jacobian is singular.  The shear closures under test keep their laws.

    python test_slab.py      (or pytest test_slab.py)
"""

import math

import numpy as np
from icepack2.constants import ice_density as rho_I, gravity as g

from ismip_hom_d import (FlowLaw, solve, THICKNESS, SLOPE_DEG, BETA2_MEAN,
                         PA_TO_MPA)

BASAL = FlowLaw(4.0, 1000.0, "basal")


def layered(fractions, laws, interface):
    """Closed-form layer velocities, bottom to top."""
    tau = rho_I * g * math.tan(math.radians(SLOPE_DEG))       # MPa per m
    h = [f * THICKNESS for f in fractions]
    u = [tau * THICKNESS / (BETA2_MEAN * PA_TO_MPA)]
    z = 0.0
    for l in range(len(h) - 1):
        z += h[l]
        S = tau * (THICKNESS - z)
        below, above = laws[l], laws[l + 1]
        if interface == "series":
            jump = h[l] * below.A * S ** below.n + h[l + 1] * above.A * S ** above.n
        else:
            jump = (h[l] + h[l + 1]) * below.A * S ** below.n
        u.append(u[-1] + jump)
    return np.array(u)


def check(fractions, laws, interface, rtol=1e-6):
    # The slab has no membrane stress, so its membrane law is immaterial;
    # a linear one keeps the Jacobian non-singular at M = 0.
    solution = solve(fractions, laws, interface=interface, beta2_amplitude=0.0,
                     membrane_exponent=1.0, wavelength=6000.0, nx=6)
    computed = solution.velocity.mean(axis=1)
    spread = np.ptp(solution.velocity, axis=1).max()
    exact = layered(fractions, laws, interface)
    error = np.abs(computed - exact) / exact
    print(f"{interface:6s} {[law.name for law in laws]}")
    for l in range(len(fractions)):
        print(f"   layer {l + 1}: computed {computed[l]:10.3f}  exact "
              f"{exact[l]:10.3f}  rel. error {error[l]:.1e}")
    assert spread < 1e-6 * exact.max(), f"not laterally uniform: {spread}"
    assert error.max() < rtol, f"relative error {error.max():.2e}"


FIVE = (9 / 600, 13 / 600, 78 / 600, 100 / 600, 400 / 600)


def test_series_two_laws():
    upper = FlowLaw(1.8, 100.0, "n=1.8")
    check(FIVE, [BASAL] * 2 + [upper] * 3, "series")


def test_series_every_assignment():
    upper = FlowLaw(1.8, 100.0, "n=1.8")
    for k in (0, 1, 3, 5):
        check(FIVE, [BASAL] * k + [upper] * (5 - k), "series")


def test_below_two_laws():
    upper = FlowLaw(1.8, 100.0, "n=1.8")
    check((1 / 3, 2 / 3), [BASAL, upper], "below")


if __name__ == "__main__":
    test_series_two_laws()
    test_series_every_assignment()
    test_below_two_laws()
    print("all slab checks pass")
