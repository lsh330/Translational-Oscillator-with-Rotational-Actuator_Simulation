"""Numerical Jacobian d(ddq)/dtau via central differences."""

import numpy as np
from numba import njit


@njit(cache=True)
def jacobian_u(q_eq, dq_eq, tau_eq, p, forward_dynamics_fn):
    """Compute d(ddq)/dtau using central finite differences.

    Returns
    -------
    Ju : float64[2,1]  Jacobian of accelerations w.r.t. input torque.
    """
    eps = 1.0e-7
    h = eps * max(1.0, abs(tau_eq))

    ddq_plus = forward_dynamics_fn(q_eq, dq_eq, tau_eq + h, p)
    ddq_minus = forward_dynamics_fn(q_eq, dq_eq, tau_eq - h, p)

    Ju = np.empty((2, 1))
    for i in range(2):
        Ju[i, 0] = (ddq_plus[i] - ddq_minus[i]) / (2.0 * h)

    return Ju
