"""Spring restoring force for the TORA.

    K(q) = [k * x,  0]

The linear spring connects the cart to the fixed wall.
No spring force acts on the rotor DOF.
"""

import numpy as np
from numba import njit


@njit(cache=True)
def spring_force(x, p):
    """Compute the spring force vector.

    Parameters
    ----------
    x : float  Cart displacement [m].
    p : float64[4]  Packed parameters.

    Returns
    -------
    K : float64[2]  Spring force vector.
    """
    k = p[3]

    K = np.empty(2)
    K[0] = k * x
    K[1] = 0.0
    return K
