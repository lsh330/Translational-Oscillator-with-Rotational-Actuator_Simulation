"""Angle wrapping utilities for periodic coordinates."""

import numpy as np
from numba import njit


@njit(cache=True)
def wrap_to_pi(angle):
    """Wrap angle to [-pi, pi].

    Parameters
    ----------
    angle : float  Angle in radians.

    Returns
    -------
    wrapped : float  Angle in [-pi, pi].
    """
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


@njit(cache=True)
def angle_error(target, actual):
    """Compute shortest angular error from actual to target."""
    return wrap_to_pi(target - actual)
