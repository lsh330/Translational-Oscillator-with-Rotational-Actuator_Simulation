"""Array-based forward dynamics for the TORA.

    ddq = M(q)^{-1} * (B*tau - C(q,dq) - K(q))

Used for linearization and Jacobian computation where array
operations are natural.  For the main simulation loop, use
forward_dynamics_fast.py instead.
"""

import numpy as np
from numba import njit


@njit(cache=True)
def forward_dynamics(q, dq, tau, p):
    """Compute generalized accelerations (array interface).

    Parameters
    ----------
    q   : float64[2]  Generalized coordinates [x, theta].
    dq  : float64[2]  Generalized velocities [x_dot, theta_dot].
    tau : float        Control torque on rotor [N*m].
    p   : float64[4]  Packed parameters [Mt, me, I_eff, k].

    Returns
    -------
    ddq : float64[2]  Accelerations [ddx, ddtheta].
    """
    Mt = p[0]
    me = p[1]
    I_eff = p[2]
    k = p[3]

    x = q[0]
    theta = q[1]
    theta_dot = dq[1]

    ct = np.cos(theta)
    st = np.sin(theta)

    # Mass matrix
    coupling = me * ct
    m00 = Mt
    m01 = coupling
    m11 = I_eff

    # RHS = B*tau - C - K
    rhs0 = me * theta_dot * theta_dot * st - k * x
    rhs1 = tau

    # Cramer's rule for 2x2
    det = m00 * m11 - m01 * m01
    ddq = np.empty(2)
    ddq[0] = (m11 * rhs0 - m01 * rhs1) / det
    ddq[1] = (m00 * rhs1 - m01 * rhs0) / det
    return ddq
