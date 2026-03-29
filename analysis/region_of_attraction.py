"""Region of Attraction (ROA) estimation via Monte Carlo simulation.

For the TORA, the ROA is 2-dimensional in (x0, theta0) space,
allowing complete visualization as a filled contour plot.
"""

import numpy as np

from simulation.integrator.rk4_step import rk4_step_fast
from simulation.loop.control_law import lqr_control
from utils.logger import get_logger

_log = get_logger("tora.roa")


def estimate_roa(p, K, x_range=(-0.5, 0.5), theta_range=(-np.pi, np.pi),
                 nx=51, ntheta=51, t_horizon=10.0, dt=0.002,
                 convergence_tol=0.01):
    """Estimate the Region of Attraction via grid-based simulation.

    Parameters
    ----------
    p : float64[4]  Packed parameters.
    K : (1,4)        LQR gain matrix.
    x_range, theta_range : tuple  IC ranges.
    nx, ntheta : int  Grid resolution.
    t_horizon : float  Simulation horizon [s].
    dt : float  Time step [s].
    convergence_tol : float  State norm threshold for convergence.

    Returns
    -------
    result : dict  Keys: x_grid, theta_grid, success_map (nx, ntheta),
                          success_rate, boundary.
    """
    K_flat = K.flatten().astype(np.float64)
    N_steps = int(t_horizon / dt)

    x_vals = np.linspace(x_range[0], x_range[1], nx)
    theta_vals = np.linspace(theta_range[0], theta_range[1], ntheta)
    success_map = np.zeros((nx, ntheta))

    total = nx * ntheta
    converged = 0

    # Memory-efficient: only final state checked, no full trajectory stored.
    # For N=51x51=2601 trials at dt=0.002 for 10s (5000 steps),
    # this uses O(1) memory per trial instead of O(5000*4) per trial.
    for i, x0 in enumerate(x_vals):
        for j, th0 in enumerate(theta_vals):
            x, theta, xd, td = x0, th0, 0.0, 0.0
            stable = True

            for step in range(N_steps):
                tau = lqr_control(x, theta, xd, td, K_flat)
                tau = max(-0.5, min(0.5, tau))
                x, theta, xd, td = rk4_step_fast(x, theta, xd, td, tau, p, dt)

                if np.isnan(x) or abs(x) > 5.0 or abs(theta) > 10.0:
                    stable = False
                    break

            if stable and (abs(x) + abs(theta) + abs(xd) + abs(td)) < convergence_tol:
                success_map[i, j] = 1.0
                converged += 1

    success_rate = converged / total

    _log.info("ROA estimation: %d/%d converged (%.1f%%)",
              converged, total, 100 * success_rate)

    return {
        "x_grid": x_vals,
        "theta_grid": theta_vals,
        "success_map": success_map,
        "success_rate": success_rate,
    }
