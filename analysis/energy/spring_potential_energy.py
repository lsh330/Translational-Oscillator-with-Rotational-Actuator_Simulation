"""Spring potential energy for the TORA.

    V = 0.5 * k * x^2
"""

import numpy as np


def spring_potential_energy(x, p):
    """Compute spring potential energy time series.

    Parameters
    ----------
    x : float64[N]  Cart displacement history.
    p : float64[4]  Packed parameters.

    Returns
    -------
    V : float64[N]  Potential energy.
    """
    k = p[3]
    return 0.5 * k * x ** 2
