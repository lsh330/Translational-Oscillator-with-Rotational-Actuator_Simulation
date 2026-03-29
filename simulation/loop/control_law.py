"""Inline control computation for the fast simulation loops."""

from numba import njit
import numpy as np


@njit(cache=True)
def lqr_control(x, theta, x_dot, theta_dot, K_flat):
    """Compute u = -K @ z using flat gain vector.

    Parameters
    ----------
    x, theta, x_dot, theta_dot : float  State.
    K_flat : float64[4]  Flattened gain vector K[0,:].

    Returns
    -------
    tau : float  Control torque.
    """
    return -(K_flat[0] * x + K_flat[1] * theta
             + K_flat[2] * x_dot + K_flat[3] * theta_dot)


# Utility: available for LQI or integral-action controllers
@njit(cache=True)
def lqr_control_antiwindup(x, theta, x_dot, theta_dot, K_flat,
                            tau_max, integrator_state):
    """LQR with back-calculation anti-windup.

    Parameters
    ----------
    integrator_state : float  Accumulated integrator error.

    Returns
    -------
    tau : float  Saturated control torque.
    integrator_state : float  Updated integrator state.
    """
    tau_raw = -(K_flat[0] * x + K_flat[1] * theta
                + K_flat[2] * x_dot + K_flat[3] * theta_dot)

    # Saturate
    if tau_raw > tau_max:
        tau_sat = tau_max
    elif tau_raw < -tau_max:
        tau_sat = -tau_max
    else:
        tau_sat = tau_raw

    # Back-calculation: track saturation error for integrator correction
    sat_error = tau_sat - tau_raw
    integrator_state = integrator_state + 0.1 * sat_error

    return tau_sat, integrator_state
