"""State derivative for ODE integration: dz/dt = f(z, tau)."""

import numpy as np
from numba import njit

from dynamics.forward_dynamics.forward_dynamics import forward_dynamics


@njit(cache=True)
def state_derivative(z, tau, p):
    """Compute dz/dt = [dq, ddq].

    Parameters
    ----------
    z   : float64[4]  State [x, theta, x_dot, theta_dot].
    tau : float        Rotor torque.
    p   : float64[4]  Packed parameters.

    Returns
    -------
    dzdt : float64[4]  State derivative.
    """
    q = z[:2]
    dq = z[2:]
    ddq = forward_dynamics(q, dq, tau, p)

    dzdt = np.empty(4)
    dzdt[0] = dq[0]
    dzdt[1] = dq[1]
    dzdt[2] = ddq[0]
    dzdt[3] = ddq[1]
    return dzdt
