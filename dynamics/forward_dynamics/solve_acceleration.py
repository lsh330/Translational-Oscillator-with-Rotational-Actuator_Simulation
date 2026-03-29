"""Solve 2x2 linear system M * ddq = rhs via Cramer's rule.

For the TORA, M is always 2x2 with det(M) = Mt*I_eff - (me*cos(theta))^2 > 0,
so Cramer's rule is both efficient and numerically safe.
"""

import numpy as np
from numba import njit


@njit(cache=True)
def solve_acceleration(M, rhs):
    """Solve M * ddq = rhs for ddq using Cramer's rule.

    Parameters
    ----------
    M   : float64[2,2]  Mass matrix (symmetric, positive definite).
    rhs : float64[2]    Right-hand side vector.

    Returns
    -------
    ddq : float64[2]  Generalized accelerations [ddx, ddtheta].
    """
    det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]

    ddq = np.empty(2)
    ddq[0] = (M[1, 1] * rhs[0] - M[0, 1] * rhs[1]) / det
    ddq[1] = (M[0, 0] * rhs[1] - M[1, 0] * rhs[0]) / det
    return ddq
