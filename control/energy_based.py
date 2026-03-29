"""Passivity-based energy shaping controller for the TORA.

Inspired by the approach of Jankovic, Fontaine & Kokotovic (1996),
"TORA Example: Cascade- and Passivity-Based Control Designs",
IEEE Trans. Control Systems Technology, 4(3), 292-297.

The controller structure exploits the TORA's Hamiltonian structure.
Default gains are computed from system physical parameters using a
principled heuristic (not a strict reproduction of the paper's
specific gain values).

Control law:
    tau = -kp * theta - kd * theta_dot - kc * p_theta

where p_theta = me*cos(theta)*x_dot + I_eff*theta_dot is the
angular momentum conjugate to theta.
"""

import numpy as np
from numba import njit


def default_energy_gains(p):
    """Compute gains for the passivity-based controller.

    Gain selection follows the energy shaping + damping injection
    framework of Jankovic et al. (1996):

        kp : Proportional gain on theta — sets the "virtual spring"
             stiffness for the rotor angle. Scaled to the system's
             natural frequency for balanced response.

        kd : Damping gain on theta_dot — provides direct dissipation.
             Set to achieve critical damping of the rotor subsystem.

        kc : Coupling momentum damping — the key term that transfers
             energy from the translational (cart) mode to the rotational
             (rotor) mode where it can be dissipated by kd.
             Scaled with omega_n to match the system's natural timescale.
    """
    Mt = p[0]
    me = p[1]
    I_eff = p[2]
    k = p[3]

    omega_n = np.sqrt(k / Mt)
    # Coupling parameter: epsilon = me / sqrt(Mt * I_eff)
    epsilon = me / np.sqrt(Mt * I_eff)

    # kp: virtual spring ~ spring constant scaled by inertia ratio
    kp = 2.0 * omega_n * I_eff

    # kd: critical damping of rotor subsystem
    kd = 2.0 * np.sqrt(kp * I_eff)

    # kc: coupling dissipation scaled by natural frequency
    kc = 4.0 * omega_n * I_eff / (1.0 + epsilon)

    return {
        "kp": kp,
        "kd": kd,
        "kc": kc,
    }


@njit(cache=True)
def energy_based_control(x, theta, x_dot, theta_dot, p, kp, kd, kc):
    """Compute passivity-based control torque.

    Parameters
    ----------
    x, theta, x_dot, theta_dot : float  State variables.
    p      : float64[4]  Packed parameters.
    kp, kd, kc : float  Controller gains.

    Returns
    -------
    tau : float  Rotor torque [N*m].
    """
    me = p[1]
    I_eff = p[2]

    # Angular momentum conjugate to theta
    p_theta = me * np.cos(theta) * x_dot + I_eff * theta_dot

    tau = -kp * theta - kd * theta_dot - kc * p_theta
    return tau


@njit(cache=True)
def energy_lyapunov(x, theta, x_dot, theta_dot, p, kp):
    """Evaluate the Lyapunov-like storage function for stability verification.

    V = 0.5*k*x^2 + p_theta^2 / (2*I_eff) + 0.5*kp*theta^2

    where p_theta = me*cos(theta)*x_dot + I_eff*theta_dot.

    Note: This is a Lyapunov-like storage function, not a physical energy.
    The kp*theta^2 term represents a virtual potential from the controller.
    V_dot <= 0 along trajectories under the passivity-based law.
    """
    me = p[1]
    I_eff = p[2]
    k = p[3]

    p_theta = me * np.cos(theta) * x_dot + I_eff * theta_dot

    V = 0.5 * k * x * x + 0.5 * p_theta * p_theta / I_eff + 0.5 * kp * theta * theta
    return V
