"""Optional viscous damping for the TORA.

    D(dq) = [c_x * x_dot, c_theta * theta_dot]

Default damping coefficients are zero (ideal benchmark model).
Set c_x > 0 and c_theta > 0 for realistic simulation.
"""

import numpy as np
from numba import njit


@njit(cache=True)
def viscous_damping(x_dot, theta_dot, c_x, c_theta):
    """Compute viscous damping force vector.

    Parameters
    ----------
    x_dot     : float  Cart velocity.
    theta_dot : float  Rotor angular velocity.
    c_x       : float  Cart viscous damping coefficient [N·s/m].
    c_theta   : float  Rotor bearing damping coefficient [N·m·s/rad].

    Returns
    -------
    D : float64[2]  Damping force vector.
    """
    D = np.empty(2)
    D[0] = c_x * x_dot
    D[1] = c_theta * theta_dot
    return D
