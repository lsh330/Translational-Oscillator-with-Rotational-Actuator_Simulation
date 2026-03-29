"""Coriolis/centrifugal force vector for the TORA.

    C(q, dq) = [-me * theta_dot^2 * sin(theta),
                 0                               ]

The only nonzero term is the centrifugal force from the spinning
eccentric rotor acting on the cart.  There is no Coriolis force
on the rotor equation because d(I_eff)/dtheta = 0.
"""

import numpy as np
from numba import njit


@njit(cache=True)
def coriolis_vector(theta, theta_dot, p):
    """Compute the Coriolis/centrifugal vector.

    Parameters
    ----------
    theta     : float  Rotor angle [rad].
    theta_dot : float  Rotor angular velocity [rad/s].
    p         : float64[4]  Packed parameters.

    Returns
    -------
    C : float64[2]  Coriolis vector.
    """
    me = p[1]
    st = np.sin(theta)

    C = np.empty(2)
    C[0] = -me * theta_dot * theta_dot * st
    C[1] = 0.0
    return C
