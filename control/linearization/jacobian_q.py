"""Numerical Jacobian d(ddq)/dq via central differences."""

import numpy as np
from numba import njit


@njit(cache=True)
def jacobian_q(q_eq, dq_eq, tau_eq, p, forward_dynamics_fn):
    """Compute d(ddq)/dq using central finite differences.

    Parameters
    ----------
    q_eq, dq_eq : float64[2]  Equilibrium configuration and velocity.
    tau_eq      : float        Equilibrium torque.
    p           : float64[4]   Packed parameters.
    forward_dynamics_fn : callable  forward_dynamics(q, dq, tau, p) -> ddq.

    Returns
    -------
    Jq : float64[2,2]  Jacobian of accelerations w.r.t. coordinates.
    """
    eps = 1.0e-7
    n = 2
    Jq = np.empty((n, n))

    for j in range(n):
        q_plus = q_eq.copy()
        q_minus = q_eq.copy()
        h = eps * max(1.0, abs(q_eq[j]))
        q_plus[j] += h
        q_minus[j] -= h

        ddq_plus = forward_dynamics_fn(q_plus, dq_eq, tau_eq, p)
        ddq_minus = forward_dynamics_fn(q_minus, dq_eq, tau_eq, p)

        for i in range(n):
            Jq[i, j] = (ddq_plus[i] - ddq_minus[i]) / (2.0 * h)

    return Jq
