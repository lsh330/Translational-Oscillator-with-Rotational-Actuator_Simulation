"""Fourth-order Runge-Kutta integrators for the TORA.

Two versions:
    rk4_step      — array-based (for analysis / Jacobian computation)
    rk4_step_fast — scalar-based (for the main simulation loop)
"""

import numpy as np
from numba import njit

from simulation.integrator.state_derivative import state_derivative
from dynamics.forward_dynamics.forward_dynamics_fast import forward_dynamics_fast


@njit(cache=True)
def rk4_step(z, tau, p, dt):
    """Array-based RK4 step.

    Parameters
    ----------
    z   : float64[4]  Current state.
    tau : float        Torque (held constant over step).
    p   : float64[4]  Packed parameters.
    dt  : float        Time step.

    Returns
    -------
    z_next : float64[4]  State after one step.
    """
    k1 = state_derivative(z, tau, p)
    k2 = state_derivative(z + 0.5 * dt * k1, tau, p)
    k3 = state_derivative(z + 0.5 * dt * k2, tau, p)
    k4 = state_derivative(z + dt * k3, tau, p)

    z_next = z + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return z_next


@njit(cache=True)
def rk4_step_fast(x, theta, x_dot, theta_dot, tau, p, dt):
    """Zero-allocation scalar RK4 step.

    Returns
    -------
    x_n, theta_n, xd_n, td_n : float  Next state scalars.
    """
    # k1
    ddx1, ddth1 = forward_dynamics_fast(x, theta, x_dot, theta_dot, tau, p)
    dx1 = x_dot
    dth1 = theta_dot

    # k2
    x2 = x + 0.5 * dt * dx1
    th2 = theta + 0.5 * dt * dth1
    xd2 = x_dot + 0.5 * dt * ddx1
    td2 = theta_dot + 0.5 * dt * ddth1
    ddx2, ddth2 = forward_dynamics_fast(x2, th2, xd2, td2, tau, p)
    dx2 = xd2
    dth2 = td2

    # k3
    x3 = x + 0.5 * dt * dx2
    th3 = theta + 0.5 * dt * dth2
    xd3 = x_dot + 0.5 * dt * ddx2
    td3 = theta_dot + 0.5 * dt * ddth2
    ddx3, ddth3 = forward_dynamics_fast(x3, th3, xd3, td3, tau, p)
    dx3 = xd3
    dth3 = td3

    # k4
    x4 = x + dt * dx3
    th4 = theta + dt * dth3
    xd4 = x_dot + dt * ddx3
    td4 = theta_dot + dt * ddth3
    ddx4, ddth4 = forward_dynamics_fast(x4, th4, xd4, td4, tau, p)
    dx4 = xd4
    dth4 = td4

    s = dt / 6.0
    x_n = x + s * (dx1 + 2.0 * dx2 + 2.0 * dx3 + dx4)
    theta_n = theta + s * (dth1 + 2.0 * dth2 + 2.0 * dth3 + dth4)
    xd_n = x_dot + s * (ddx1 + 2.0 * ddx2 + 2.0 * ddx3 + ddx4)
    td_n = theta_dot + s * (ddth1 + 2.0 * ddth2 + 2.0 * ddth3 + ddth4)

    return x_n, theta_n, xd_n, td_n
