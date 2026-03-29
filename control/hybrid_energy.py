"""Hybrid energy-based controller with mode switching.

Phase 1 (Energy pumping): When total energy is far from zero,
    use aggressive energy transfer from cart to rotor.
Phase 2 (Capture): When energy is below threshold,
    switch to local PD-like stabilization.
Phase 3 (Fine regulation): When near equilibrium,
    use LQR-like gains for precision.
"""

import numpy as np
from numba import njit


@njit(cache=True)
def hybrid_energy_control(x, theta, x_dot, theta_dot, p, K_flat,
                           kp, kd, kc, energy_threshold=0.1):
    """Hybrid energy controller with automatic mode switching.

    Parameters
    ----------
    x, theta, x_dot, theta_dot : float  State.
    p       : float64[6]  Packed parameters.
    K_flat  : float64[4]  LQR gain for fine regulation.
    kp, kd, kc : float  Energy-based controller gains.
    energy_threshold : float  Energy below which to switch to LQR.

    Returns
    -------
    tau  : float  Control torque.
    mode : int    0=energy pumping, 1=capture, 2=fine regulation.
    """
    Mt = p[0]
    me = p[1]
    I_eff = p[2]
    k = p[3]

    # Compute total energy
    T = 0.5 * (Mt * x_dot**2 + 2*me*np.cos(theta)*x_dot*theta_dot + I_eff*theta_dot**2)
    V = 0.5 * k * x * x
    H = T + V

    state_norm = abs(x) + abs(theta) + abs(x_dot) + abs(theta_dot)

    if state_norm < 0.01:
        # Phase 3: Fine regulation — LQR
        tau = -(K_flat[0]*x + K_flat[1]*theta + K_flat[2]*x_dot + K_flat[3]*theta_dot)
        return tau, 2

    elif H < energy_threshold:
        # Phase 2: Capture — stronger PD
        tau = -2.0*kp*theta - 2.0*kd*theta_dot
        return tau, 1

    else:
        # Phase 1: Energy pumping — passivity-based
        p_theta = me * np.cos(theta) * x_dot + I_eff * theta_dot
        tau = -kp * theta - kd * theta_dot - kc * p_theta
        return tau, 0
