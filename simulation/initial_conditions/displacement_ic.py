"""Initial conditions for the TORA benchmark.

Standard IC: displace the cart from equilibrium and release.
"""

import numpy as np


def initial_displacement(x0=0.1):
    """Return IC vector [x0, 0, 0, 0].

    Parameters
    ----------
    x0 : float  Initial cart displacement [m].  Default 0.1 m.

    Returns
    -------
    z0 : float64[4]  State [x, theta, x_dot, theta_dot].
    """
    return np.array([x0, 0.0, 0.0, 0.0])


def initial_state(x0=0.1, theta0=0.0, x_dot0=0.0, theta_dot0=0.0):
    """Return general IC vector [x0, theta0, x_dot0, theta_dot0].

    Parameters
    ----------
    x0, theta0       : float  Initial positions.
    x_dot0, theta_dot0 : float  Initial velocities.

    Returns
    -------
    z0 : float64[4]
    """
    return np.array([x0, theta0, x_dot0, theta_dot0])
