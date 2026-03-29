"""Pole analysis for the TORA."""

import numpy as np


def compute_poles(A, B=None, K=None):
    """Compute open-loop and (optionally) closed-loop poles.

    Returns
    -------
    result : dict  Keys: ol_poles, cl_poles (if K given).
    """
    ol_poles = np.linalg.eigvals(A)
    result = {"ol_poles": ol_poles}

    if K is not None and B is not None:
        A_cl = A - B @ K
        cl_poles = np.linalg.eigvals(A_cl)
        result["cl_poles"] = cl_poles

    return result
