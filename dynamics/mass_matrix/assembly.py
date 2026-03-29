"""2x2 configuration-dependent mass matrix for the TORA.

    M(theta) = [[Mt,          me*cos(theta)],
                [me*cos(theta), I_eff       ]]

where Mt = M+m, me = m*e, I_eff = I + m*e^2.
"""

import numpy as np
from numba import njit


@njit(cache=True)
def mass_matrix(theta, p):
    """Compute the 2x2 mass matrix.

    Parameters
    ----------
    theta : float  Rotor angle [rad].
    p     : float64[4]  Packed parameters [Mt, me, I_eff, k].

    Returns
    -------
    M : float64[2,2]  Symmetric positive-definite mass matrix.
    """
    Mt = p[0]
    me = p[1]
    I_eff = p[2]

    ct = np.cos(theta)
    coupling = me * ct

    M = np.empty((2, 2))
    M[0, 0] = Mt
    M[0, 1] = coupling
    M[1, 0] = coupling
    M[1, 1] = I_eff
    return M
