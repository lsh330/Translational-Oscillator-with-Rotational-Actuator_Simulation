"""Exact analytical Jacobians for the TORA at the equilibrium.

At (x, theta, x_dot, theta_dot) = (0, 0, 0, 0):
  - cos(theta) = 1, sin(theta) = 0
  - All velocity-dependent terms vanish
  - dM/dtheta = 0 at theta=0 (since d(cos)/dtheta|_0 = -sin(0) = 0)

This yields closed-form A and B matrices without any numerical
differentiation.
"""

import numpy as np


def analytical_A(p):
    """Compute the exact 4x4 state matrix at the equilibrium.

    State ordering: [x, theta, x_dot, theta_dot].

    A = [[0,  0,  1,  0],
         [0,  0,  0,  1],
         [a20, 0, 0,  0],
         [a30, 0, 0,  0]]

    where:
        det_M = Mt*I_eff - me^2
        a20   = -k*I_eff / det_M
        a30   =  k*me    / det_M
    """
    Mt = p[0]
    me = p[1]
    I_eff = p[2]
    k = p[3]

    det_M = Mt * I_eff - me * me

    A = np.zeros((4, 4))
    # Top-right: identity (dq/dt = dq)
    A[0, 2] = 1.0
    A[1, 3] = 1.0
    # Bottom-left: d(ddq)/dq
    A[2, 0] = -k * I_eff / det_M
    A[3, 0] = k * me / det_M
    return A


def analytical_B(p):
    """Compute the exact 4x1 input matrix at the equilibrium.

    B = [[0],
         [0],
         [-me / det_M],
         [Mt  / det_M]]
    """
    Mt = p[0]
    me = p[1]
    I_eff = p[2]

    det_M = Mt * I_eff - me * me

    B = np.zeros((4, 1))
    B[2, 0] = -me / det_M
    B[3, 0] = Mt / det_M
    return B
