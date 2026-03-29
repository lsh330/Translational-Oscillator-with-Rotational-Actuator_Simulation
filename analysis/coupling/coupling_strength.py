"""Coupling strength analysis for the TORA.

The off-diagonal mass matrix element me*cos(theta) determines
how strongly the rotor influences the cart.
    - theta = 0:    max coupling (me)
    - theta = pi/2: zero coupling (rotor spins freely)
"""

import numpy as np


def coupling_strength(theta, p):
    """Compute coupling strength along the trajectory.

    Parameters
    ----------
    theta : float64[N]  Rotor angle history.
    p     : float64[4]  Packed parameters.

    Returns
    -------
    result : dict  Keys: coupling, max_coupling, min_coupling, normalized.
    """
    me = p[1]
    coupling = me * np.cos(theta)

    return {
        "coupling": coupling,
        "max_coupling": me,
        "min_coupling": np.min(np.abs(coupling)),
        "normalized": coupling / me,
    }
