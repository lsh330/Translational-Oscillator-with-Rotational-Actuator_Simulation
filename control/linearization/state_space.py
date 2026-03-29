"""Assemble full state-space matrices from sub-blocks.

State: z = [x, theta, x_dot, theta_dot]

    A = [[0_{2x2},   I_{2x2}],      B = [[0_{2x1}],
         [A_q,       A_dq   ]]           [B_u    ]]
"""

import numpy as np


def assemble_state_space(A_q, A_dq, B_u):
    """Build the 4x4 A and 4x1 B matrices from 2x2 / 2x1 blocks.

    Parameters
    ----------
    A_q  : float64[2,2]  d(ddq)/dq  block.
    A_dq : float64[2,2]  d(ddq)/d(dq) block.
    B_u  : float64[2,1]  d(ddq)/dtau block.

    Returns
    -------
    A : float64[4,4]
    B : float64[4,1]
    """
    A = np.zeros((4, 4))
    A[0, 2] = 1.0
    A[1, 3] = 1.0
    A[2:4, 0:2] = A_q
    A[2:4, 2:4] = A_dq

    B = np.zeros((4, 1))
    B[2:4, :] = B_u

    return A, B
