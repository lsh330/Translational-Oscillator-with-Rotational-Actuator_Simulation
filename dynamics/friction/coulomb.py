"""Coulomb and viscous friction model.

Combined friction force:
    f = c_viscous * v + F_coulomb * sign(v)

With Stribeck-like smooth transition near zero velocity:
    f = c_viscous * v + F_coulomb * tanh(v / v_stribeck)
"""

import numpy as np
from numba import njit


@njit(cache=True)
def coulomb_friction(velocity, F_coulomb, v_stribeck=0.01):
    """Compute Coulomb friction with smooth Stribeck transition.

    Parameters
    ----------
    velocity    : float  Velocity [m/s or rad/s].
    F_coulomb   : float  Coulomb friction force magnitude.
    v_stribeck  : float  Stribeck transition velocity (smoothing).

    Returns
    -------
    f : float  Friction force (opposes motion).
    """
    return F_coulomb * np.tanh(velocity / v_stribeck)
