"""Generalized force vector assembly for the TORA.

The TORA is underactuated: torque tau acts only on the rotor DOF.

    B * tau = [0, tau]^T
"""

import numpy as np
from numba import njit


@njit(cache=True)
def tau_assembly(tau):
    """Assemble the generalized force vector.

    Parameters
    ----------
    tau : float  Control torque on rotor [N*m].

    Returns
    -------
    f : float64[2]  Generalized force [0, tau].
    """
    f = np.empty(2)
    f[0] = 0.0
    f[1] = tau
    return f
