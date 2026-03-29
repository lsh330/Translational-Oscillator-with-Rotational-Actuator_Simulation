"""Shared trigonometric helpers for TORA dynamics.

For the TORA system only sin(theta) and cos(theta) are needed
(unlike the triple pendulum which requires 9 trig values).
"""

import numpy as np
from numba import njit


@njit(cache=True)
def sincos(theta):
    """Return (sin(theta), cos(theta)) computed once."""
    return np.sin(theta), np.cos(theta)
