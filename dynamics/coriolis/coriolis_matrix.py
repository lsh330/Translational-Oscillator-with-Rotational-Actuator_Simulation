"""Full 2x2 Coriolis matrix for the TORA.

The Coriolis matrix C(q, dq) satisfies:
    C(q, dq) * dq = coriolis_vector(q, dq)

And the fundamental passivity property:
    M_dot - 2*C  is skew-symmetric

This property is essential for passivity-based control design.

For the TORA with M = [[Mt, me*cos(theta)], [me*cos(theta), I_eff]]:

    dM/dt = [[0, -me*sin(theta)*theta_dot],
             [-me*sin(theta)*theta_dot, 0]]

Using Christoffel symbols, the unique C satisfying the skew-symmetric property:

    C = [[0, -me*sin(theta)*theta_dot],
         [0, 0]]

Verification: C * [x_dot, theta_dot]^T = [-me*sin(theta)*theta_dot^2, 0]^T  ✓
              M_dot - 2C = [[0, -me*sin(theta)*theta_dot],
                            [-me*sin(theta)*theta_dot, 0]]
                         - [[0, -2*me*sin(theta)*theta_dot],
                            [0, 0]]
                         = [[0, me*sin(theta)*theta_dot],
                            [-me*sin(theta)*theta_dot, 0]]  ← skew-symmetric ✓
"""

import numpy as np
from numba import njit


@njit(cache=True)
def coriolis_matrix(theta, theta_dot, p):
    """Compute the 2x2 Coriolis matrix C(q, dq).

    Parameters
    ----------
    theta     : float  Rotor angle [rad].
    theta_dot : float  Rotor angular velocity [rad/s].
    p         : float64[4]  Packed parameters.

    Returns
    -------
    C : float64[2,2]  Coriolis matrix satisfying M_dot - 2C = skew-symmetric.
    """
    me = p[1]
    st = np.sin(theta)

    C = np.zeros((2, 2))
    C[0, 1] = -me * st * theta_dot
    return C


@njit(cache=True)
def verify_skew_symmetric(theta, theta_dot, p):
    """Check that M_dot - 2C is skew-symmetric.

    Returns
    -------
    S : float64[2,2]  The matrix M_dot - 2C (should be skew-symmetric).
    err : float  Max |S + S^T| (should be near zero).
    """
    me = p[1]
    st = np.sin(theta)

    # M_dot
    Mdot = np.zeros((2, 2))
    Mdot[0, 1] = -me * st * theta_dot
    Mdot[1, 0] = -me * st * theta_dot

    # 2C
    C2 = np.zeros((2, 2))
    C2[0, 1] = -2.0 * me * st * theta_dot

    S = Mdot - C2
    err = 0.0
    for i in range(2):
        for j in range(2):
            err = max(err, abs(S[i, j] + S[j, i]))

    return S, err
