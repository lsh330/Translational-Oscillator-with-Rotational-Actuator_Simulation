"""All-in-one JIT-compiled numerical Jacobian + state-space assembly."""

import numpy as np
from numba import njit

from dynamics.forward_dynamics.forward_dynamics import forward_dynamics


@njit(cache=True)
def compute_numerical_state_space(q_eq, dq_eq, tau_eq, p):
    """Compute full numerical A (4x4) and B (4x1) via central differences.

    Combines jacobian_q, jacobian_dq, jacobian_u into a single pass,
    and assembles the state-space matrices directly.
    """
    eps = 1.0e-7
    n_q = 2

    A = np.zeros((4, 4))
    B = np.zeros((4, 1))

    # Top-right identity block
    A[0, 2] = 1.0
    A[1, 3] = 1.0

    # d(ddq)/dq -> A[2:4, 0:2]
    for j in range(n_q):
        q_plus = q_eq.copy()
        q_minus = q_eq.copy()
        h = eps * max(1.0, abs(q_eq[j]))
        q_plus[j] += h
        q_minus[j] -= h
        ddq_p = forward_dynamics(q_plus, dq_eq, tau_eq, p)
        ddq_m = forward_dynamics(q_minus, dq_eq, tau_eq, p)
        for i in range(n_q):
            A[i + 2, j] = (ddq_p[i] - ddq_m[i]) / (2.0 * h)

    # d(ddq)/d(dq) -> A[2:4, 2:4]
    for j in range(n_q):
        dq_plus = dq_eq.copy()
        dq_minus = dq_eq.copy()
        h = eps * max(1.0, abs(dq_eq[j]))
        dq_plus[j] += h
        dq_minus[j] -= h
        ddq_p = forward_dynamics(q_eq, dq_plus, tau_eq, p)
        ddq_m = forward_dynamics(q_eq, dq_minus, tau_eq, p)
        for i in range(n_q):
            A[i + 2, j + 2] = (ddq_p[i] - ddq_m[i]) / (2.0 * h)

    # d(ddq)/dtau -> B[2:4, 0]
    h = eps * max(1.0, abs(tau_eq))
    ddq_p = forward_dynamics(q_eq, dq_eq, tau_eq + h, p)
    ddq_m = forward_dynamics(q_eq, dq_eq, tau_eq - h, p)
    for i in range(n_q):
        B[i + 2, 0] = (ddq_p[i] - ddq_m[i]) / (2.0 * h)

    return A, B
