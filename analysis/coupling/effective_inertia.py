"""Effective inertia and mass matrix conditioning analysis.

    det(M) = Mt*I_eff - (me*cos(theta))^2

This determinant is always positive for the standard parameters,
but its variation with theta affects controllability and numerical
conditioning.
"""

import numpy as np


def effective_inertia(theta, p):
    """Analyze mass matrix determinant and condition number.

    Parameters
    ----------
    theta : float64[N]  Rotor angle history.
    p     : float64[4]  Packed parameters.

    Returns
    -------
    result : dict  Keys: det_M, cond_M, det_min, det_max.
    """
    Mt = p[0]
    me = p[1]
    I_eff = p[2]

    ct = np.cos(theta)
    coupling = me * ct

    det_M = Mt * I_eff - coupling ** 2

    # Condition number: max_eig / min_eig of M(theta)
    # For 2x2 symmetric: eigenvalues via trace and det
    trace_M = Mt + I_eff
    disc = np.sqrt(np.maximum(trace_M ** 2 - 4.0 * det_M, 0.0))
    eig_max = 0.5 * (trace_M + disc)
    eig_min = 0.5 * (trace_M - disc)
    cond_M = eig_max / np.maximum(eig_min, 1e-15)

    return {
        "det_M": det_M,
        "cond_M": cond_M,
        "det_min": np.min(det_M),
        "det_max": np.max(det_M),
    }
