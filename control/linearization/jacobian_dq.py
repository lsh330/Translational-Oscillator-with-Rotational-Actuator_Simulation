"""Numerical Jacobian d(ddq)/d(dq) via central differences."""

import numpy as np
from numba import njit


@njit(cache=True)
def jacobian_dq(q_eq, dq_eq, tau_eq, p, forward_dynamics_fn):
    """Compute d(ddq)/d(dq) using central finite differences.

    Returns
    -------
    Jdq : float64[2,2]  Jacobian of accelerations w.r.t. velocities.
    """
    eps = 1.0e-7
    n = 2
    Jdq = np.empty((n, n))

    for j in range(n):
        dq_plus = dq_eq.copy()
        dq_minus = dq_eq.copy()
        h = eps * max(1.0, abs(dq_eq[j]))
        dq_plus[j] += h
        dq_minus[j] -= h

        ddq_plus = forward_dynamics_fn(q_eq, dq_plus, tau_eq, p)
        ddq_minus = forward_dynamics_fn(q_eq, dq_minus, tau_eq, p)

        for i in range(n):
            Jdq[i, j] = (ddq_plus[i] - ddq_minus[i]) / (2.0 * h)

    return Jdq
