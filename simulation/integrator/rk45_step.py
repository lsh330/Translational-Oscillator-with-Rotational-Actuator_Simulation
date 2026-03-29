"""Dormand-Prince RK4(5) adaptive step integrator."""

import numpy as np
from numba import njit

from simulation.integrator.state_derivative import state_derivative

# Dormand-Prince coefficients
_A2 = 1.0 / 5.0
_A3 = 3.0 / 10.0
_A4 = 4.0 / 5.0
_A5 = 8.0 / 9.0

_SAFETY = 0.9
_MIN_FACTOR = 0.2
_MAX_FACTOR = 5.0


@njit(cache=True)
def rk45_step(z, tau, p, dt, atol=1e-8, rtol=1e-6):
    """Adaptive RK4(5) step with embedded error estimation.

    Parameters
    ----------
    z    : float64[4]  Current state.
    tau  : float        Torque.
    p    : float64[4]  Packed parameters.
    dt   : float        Proposed step size.
    atol : float        Absolute error tolerance.
    rtol : float        Relative error tolerance.

    Returns
    -------
    z_next : float64[4]  State after step.
    dt_new : float        Recommended next step size.
    """
    k1 = state_derivative(z, tau, p)
    k2 = state_derivative(z + dt * _A2 * k1, tau, p)
    k3 = state_derivative(z + dt * (3.0 / 40.0 * k1 + 9.0 / 40.0 * k2), tau, p)
    k4 = state_derivative(z + dt * (44.0 / 45.0 * k1 - 56.0 / 15.0 * k2 + 32.0 / 9.0 * k3), tau, p)
    k5 = state_derivative(z + dt * (19372.0 / 6561.0 * k1 - 25360.0 / 2187.0 * k2
                                     + 64448.0 / 6561.0 * k3 - 212.0 / 729.0 * k4), tau, p)
    k6 = state_derivative(z + dt * (9017.0 / 3168.0 * k1 - 355.0 / 33.0 * k2
                                     + 46732.0 / 5247.0 * k3 + 49.0 / 176.0 * k4
                                     - 5103.0 / 18656.0 * k5), tau, p)

    # 4th order solution
    z4 = z + dt * (35.0 / 384.0 * k1 + 500.0 / 1113.0 * k3
                   + 125.0 / 192.0 * k4 - 2187.0 / 6784.0 * k5
                   + 11.0 / 84.0 * k6)

    # 5th order for error estimation
    k7 = state_derivative(z4, tau, p)
    z5 = z + dt * (5179.0 / 57600.0 * k1 + 7571.0 / 16695.0 * k3
                   + 393.0 / 640.0 * k4 - 92097.0 / 339200.0 * k5
                   + 187.0 / 2100.0 * k6 + 1.0 / 40.0 * k7)

    # Error estimate
    err = np.abs(z5 - z4)
    scale = atol + rtol * np.maximum(np.abs(z), np.abs(z4))
    err_norm = np.max(err / scale)

    if err_norm < 1e-12:
        dt_new = dt * _MAX_FACTOR
    else:
        dt_new = dt * min(_MAX_FACTOR, max(_MIN_FACTOR, _SAFETY * err_norm ** (-0.2)))

    return z4, dt_new
