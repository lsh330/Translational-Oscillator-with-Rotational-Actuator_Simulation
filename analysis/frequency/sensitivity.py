"""Sensitivity and complementary sensitivity functions."""

import numpy as np


def sensitivity_functions(L):
    """Compute S(jw) and T(jw) from L(jw).

    S = 1 / (1 + L)      sensitivity
    T = L / (1 + L)       complementary sensitivity

    Returns
    -------
    S, T : complex128[M]
    Ms, Mt : float  Peak magnitudes.
    """
    S = 1.0 / (1.0 + L)
    T = L / (1.0 + L)

    Ms = np.max(np.abs(S))
    Mt = np.max(np.abs(T))

    return S, T, Ms, Mt
