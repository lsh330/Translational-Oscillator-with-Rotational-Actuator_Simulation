"""Default R matrix for the TORA LQR cost function.

The R value is chosen using Bryson's rule to respect the physical
torque limit.  For tau_max = 0.1 N*m:

    R = 1 / tau_max^2 = 100

This ensures the LQR gain produces torques within the actuator's
capability for typical operating deviations, avoiding excessive
saturation that would degrade optimality.
"""

import numpy as np


def default_R(tau_max=0.1):
    """Return the 1x1 input cost matrix based on actuator limit.

    Uses Bryson's rule: R_ii = 1 / u_max_i^2.

    Parameters
    ----------
    tau_max : float  Maximum allowable torque [N*m].

    Returns
    -------
    R : float64[1,1]  Input cost matrix.
    """
    return np.array([[1.0 / (tau_max ** 2)]])
