"""Zero-allocation scalar forward dynamics for the TORA.

All inputs and outputs are scalars — no array allocation inside the
hot loop.  Uses inline Cramer's rule for the 2x2 mass matrix inverse.
Single trig call (cos, sin) per evaluation.
"""

import numpy as np
from numba import njit


@njit(cache=True)
def forward_dynamics_fast(x, theta, x_dot, theta_dot, tau, p):
    """Compute accelerations using pure scalar arithmetic.

    Parameters
    ----------
    x, theta        : float  Generalized coordinates.
    x_dot, theta_dot: float  Generalized velocities.
    tau             : float  Rotor torque [N*m].
    p               : float64[4]  Packed parameters.

    Returns
    -------
    ddx, ddtheta : float  Generalized accelerations.
    """

    Mt = p[0]
    me = p[1]
    I_eff = p[2]
    k = p[3]

    ct = np.cos(theta)
    st = np.sin(theta)

    # Mass matrix elements
    coupling = me * ct

    # RHS = B*tau - C(q,dq) - K(q)
    rhs0 = me * theta_dot * theta_dot * st - k * x
    rhs1 = tau

    # 2x2 Cramer's rule: det = Mt*I_eff - (me*cos(theta))^2
    det = Mt * I_eff - coupling * coupling
    inv_det = 1.0 / det

    ddx = (I_eff * rhs0 - coupling * rhs1) * inv_det
    ddtheta = (Mt * rhs1 - coupling * rhs0) * inv_det

    return ddx, ddtheta
