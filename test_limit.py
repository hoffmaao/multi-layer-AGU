r"""The limit of vanishing basal ice in ISMIP-HOM D.

As the basal layer of a two-layer column thins to nothing, the column
should recover the same column of stratigraphic ice only (n = 1.8 over the
full thickness).  With interfaces in series the basal ice shears only its
own half-layer, so its effect on the surface velocity is proportional to
its thickness:

    || u_composite - u_stratigraphic || / || u_stratigraphic ||  ~  h_basal.

With the poster's closure (``interface="below"``) the basal law shears the
whole interface jump, so the gap stays finite however thin the basal ice.

    python test_limit.py      (or pytest test_limit.py)
"""

from functools import cache

import numpy as np

from experiments import SOFT, COLD
from ismip_hom_d import solve, THICKNESS

#: Coarser than the experiments (nx = 40) to keep the test quick; the
#: differences below change by about 2 % between the two.
NX = 20


@cache
def surface(metres, laws, interface):
    """Surface (L = 1) velocity of a two-layer column with ``metres`` of
    basal ice, the periodic duplicate at x = L dropped."""
    h = metres / THICKNESS
    return solve((h, 1 - h), laws, interface=interface, nx=NX).surface[:-1]


def difference(metres, interface):
    """Relative surface difference between basal and stratigraphic ice
    in the bottom layer."""
    composite = surface(metres, (SOFT, COLD), interface)
    reference = surface(metres, (COLD, COLD), interface)
    return np.linalg.norm(composite - reference) / np.linalg.norm(reference)


def test_series_recovers_stratigraphic():
    coarse, fine = difference(1.0, "series"), difference(0.1, "series")
    print(f"series: 1 m {coarse:.2e}, 0.1 m {fine:.2e}")
    assert fine < 2e-4
    # Ten times thinner basal ice, ten times smaller difference.
    assert 8 < coarse / fine < 12


def test_poster_closure_keeps_a_gap():
    gap = difference(0.1, "below")
    print(f"below:  0.1 m {gap:.2e}")
    assert gap > 0.4


if __name__ == "__main__":
    test_series_recovers_stratigraphic()
    test_poster_closure_keeps_a_gap()
    print("limit checks pass")
