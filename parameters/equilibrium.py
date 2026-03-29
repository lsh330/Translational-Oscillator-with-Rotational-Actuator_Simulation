"""Equilibrium state for the TORA system.

The TORA equilibrium is at the origin: cart centered (x=0),
rotor aligned (theta=0), all velocities zero.
"""

import numpy as np


def equilibrium() -> np.ndarray:
    """Return the upright equilibrium state [x, theta, x_dot, theta_dot]."""
    return np.zeros(4, dtype=np.float64)
