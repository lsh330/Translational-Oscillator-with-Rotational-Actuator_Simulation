"""LQR gain matrix computation: K = R^{-1} B^T P."""

import numpy as np


def compute_K(B, R, P):
    """Compute the optimal state-feedback gain matrix.

    Parameters
    ----------
    B : (4,1)  Input matrix.
    R : (1,1)  Input cost matrix.
    P : (4,4)  CARE solution.

    Returns
    -------
    K : (1,4)  Gain matrix such that u = -K @ z.
    """
    return np.linalg.solve(R, B.T @ P)
