"""Störmer-Verlet symplectic integrator for TORA.

Preserves the symplectic structure of the Hamiltonian system,
guaranteeing near-machine-precision energy conservation over
exponentially long integration times.

Algorithm (leapfrog / velocity Verlet):
    1. q_{n+1/2} = q_n + (dt/2) * dq_n
    2. dq_{n+1} = dq_n + dt * ddq(q_{n+1/2}, dq_n, tau)
    3. q_{n+1} = q_{n+1/2} + (dt/2) * dq_{n+1}
"""

import numpy as np
from numba import njit

from dynamics.forward_dynamics.forward_dynamics import forward_dynamics


@njit(cache=True)
def stormer_verlet_step(z, tau, p, dt):
    """Symplectic Störmer-Verlet integration step.

    Parameters
    ----------
    z   : float64[4]  State [x, theta, x_dot, theta_dot].
    tau : float        Rotor torque.
    p   : float64[4]  Packed parameters.
    dt  : float        Time step.

    Returns
    -------
    z_next : float64[4]  State after one symplectic step.
    """
    q = z[:2].copy()
    dq = z[2:].copy()

    # Half-step position
    q_half = q + 0.5 * dt * dq

    # Full-step velocity using acceleration at half-step position
    ddq = forward_dynamics(q_half, dq, tau, p)
    dq_new = dq + dt * ddq

    # Complete position step
    q_new = q_half + 0.5 * dt * dq_new

    z_next = np.empty(4)
    z_next[0] = q_new[0]
    z_next[1] = q_new[1]
    z_next[2] = dq_new[0]
    z_next[3] = dq_new[1]
    return z_next
