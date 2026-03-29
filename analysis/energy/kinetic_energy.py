"""Kinetic energy computation for the TORA.

    T = 0.5 * dq^T M(q) dq
      = 0.5 * [Mt*xd^2 + 2*me*cos(theta)*xd*td + I_eff*td^2]
"""

import numpy as np


def kinetic_energy(x_dot, theta, theta_dot, p):
    """Compute kinetic energy time series.

    Parameters
    ----------
    x_dot, theta, theta_dot : float64[N]  Time histories.
    p : float64[4]  Packed parameters.

    Returns
    -------
    T : float64[N]  Kinetic energy.
    """
    Mt = p[0]
    me = p[1]
    I_eff = p[2]

    ct = np.cos(theta)
    T = 0.5 * (Mt * x_dot ** 2
               + 2.0 * me * ct * x_dot * theta_dot
               + I_eff * theta_dot ** 2)
    return T
