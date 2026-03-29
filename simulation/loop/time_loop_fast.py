"""JIT-compiled simulation loops for each controller type.

Each loop variant is a separate @njit function for maximum
performance (no Python-level dispatch inside the hot path).
"""

import numpy as np
from numba import njit

from simulation.integrator.rk4_step import rk4_step_fast
from simulation.loop.control_law import lqr_control
from control.energy_based import energy_based_control
from control.sliding_mode import sliding_mode_control, sliding_surface


@njit(cache=True)
def _run_loop_lqr(N, dt, x0, theta0, xd0, td0, K_flat, p,
                  disturbance, tau_max):
    """Zero-allocation LQR simulation loop."""
    x_hist = np.empty(N + 1)
    theta_hist = np.empty(N + 1)
    xd_hist = np.empty(N + 1)
    td_hist = np.empty(N + 1)
    u_hist = np.empty(N)
    u_raw = np.empty(N)
    sat_count = 0

    x, theta, xd, td = x0, theta0, xd0, td0
    x_hist[0] = x
    theta_hist[0] = theta
    xd_hist[0] = xd
    td_hist[0] = td

    for i in range(N):
        tau = lqr_control(x, theta, xd, td, K_flat)
        tau += disturbance[i]
        u_raw[i] = tau

        if tau > tau_max:
            tau = tau_max
            sat_count += 1
        elif tau < -tau_max:
            tau = -tau_max
            sat_count += 1
        u_hist[i] = tau

        x, theta, xd, td = rk4_step_fast(x, theta, xd, td, tau, p, dt)

        # NaN check
        if np.isnan(x) or np.isnan(theta):
            for j in range(i + 1, N + 1):
                x_hist[j] = np.nan
                theta_hist[j] = np.nan
                xd_hist[j] = np.nan
                td_hist[j] = np.nan
            break

        x_hist[i + 1] = x
        theta_hist[i + 1] = theta
        xd_hist[i + 1] = xd
        td_hist[i + 1] = td

    return x_hist, theta_hist, xd_hist, td_hist, u_hist, u_raw, sat_count


@njit(cache=True)
def _run_loop_energy(N, dt, x0, theta0, xd0, td0, p,
                     kp, kd, kc, disturbance, tau_max):
    """Zero-allocation energy-based controller simulation loop."""
    x_hist = np.empty(N + 1)
    theta_hist = np.empty(N + 1)
    xd_hist = np.empty(N + 1)
    td_hist = np.empty(N + 1)
    u_hist = np.empty(N)
    sat_count = 0

    x, theta, xd, td = x0, theta0, xd0, td0
    x_hist[0] = x
    theta_hist[0] = theta
    xd_hist[0] = xd
    td_hist[0] = td

    for i in range(N):
        tau = energy_based_control(x, theta, xd, td, p, kp, kd, kc)
        tau += disturbance[i]

        if tau > tau_max:
            tau = tau_max
            sat_count += 1
        elif tau < -tau_max:
            tau = -tau_max
            sat_count += 1
        u_hist[i] = tau

        x, theta, xd, td = rk4_step_fast(x, theta, xd, td, tau, p, dt)

        if np.isnan(x) or np.isnan(theta):
            for j in range(i + 1, N + 1):
                x_hist[j] = np.nan
                theta_hist[j] = np.nan
                xd_hist[j] = np.nan
                td_hist[j] = np.nan
            break

        x_hist[i + 1] = x
        theta_hist[i + 1] = theta
        xd_hist[i + 1] = xd
        td_hist[i + 1] = td

    return x_hist, theta_hist, xd_hist, td_hist, u_hist, sat_count


@njit(cache=True)
def _run_loop_smc(N, dt, x0, theta0, xd0, td0, p,
                  c1, c2, c3, eta, phi, disturbance, tau_max):
    """Zero-allocation SMC simulation loop."""
    x_hist = np.empty(N + 1)
    theta_hist = np.empty(N + 1)
    xd_hist = np.empty(N + 1)
    td_hist = np.empty(N + 1)
    u_hist = np.empty(N)
    s_hist = np.empty(N)
    sat_count = 0

    x, theta, xd, td = x0, theta0, xd0, td0
    x_hist[0] = x
    theta_hist[0] = theta
    xd_hist[0] = xd
    td_hist[0] = td

    for i in range(N):
        tau = sliding_mode_control(x, theta, xd, td, p,
                                   c1, c2, c3, eta, phi)
        tau += disturbance[i]
        s_hist[i] = sliding_surface(x, theta, xd, td, c1, c2, c3)

        if tau > tau_max:
            tau = tau_max
            sat_count += 1
        elif tau < -tau_max:
            tau = -tau_max
            sat_count += 1
        u_hist[i] = tau

        x, theta, xd, td = rk4_step_fast(x, theta, xd, td, tau, p, dt)

        if np.isnan(x) or np.isnan(theta):
            for j in range(i + 1, N + 1):
                x_hist[j] = np.nan
                theta_hist[j] = np.nan
                xd_hist[j] = np.nan
                td_hist[j] = np.nan
            break

        x_hist[i + 1] = x
        theta_hist[i + 1] = theta
        xd_hist[i + 1] = xd
        td_hist[i + 1] = td

    return x_hist, theta_hist, xd_hist, td_hist, u_hist, s_hist, sat_count
