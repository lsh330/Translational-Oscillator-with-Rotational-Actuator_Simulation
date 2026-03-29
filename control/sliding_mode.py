"""Sliding mode controller (SMC) for the TORA with boundary layer.

Sliding surface:
    s = c1*x + c2*theta + c3*x_dot + theta_dot

The surface coefficients c1, c2, c3 are chosen so that s=0
defines a stable 3rd-order manifold in state space.

Control law:
    tau = tau_eq - eta * sat(s / phi)

where tau_eq is the equivalent control (keeps system on s=0),
eta is the switching gain (robustness margin), and phi is the
boundary layer thickness (chattering suppression).
"""

import numpy as np
from numba import njit


def default_smc_gains(p):
    """Compute SMC design parameters using pole-placement on sliding manifold.

    The sliding surface s = c1*x + c2*theta + c3*x_dot + theta_dot defines
    a 3rd-order manifold. When s=0, the reduced dynamics have poles at
    the roots of: lambda^3 + c3*lambda^2 + c2*lambda + c1 = 0

    We place these poles at -omega_d (triple real pole) where omega_d
    is chosen as a fraction of the natural frequency.
    """
    Mt = p[0]
    k = p[3]
    omega_n = np.sqrt(k / Mt)

    # Desired closed-loop bandwidth on sliding manifold
    omega_d = 0.3 * omega_n  # conservative: 30% of natural frequency

    # Triple pole at -omega_d: (s + omega_d)^3 = s^3 + 3*omega_d*s^2 + 3*omega_d^2*s + omega_d^3
    c3 = 3.0 * omega_d           # coefficient of x_dot
    c2 = 3.0 * omega_d ** 2      # coefficient of theta
    c1 = omega_d ** 3             # coefficient of x

    # Switching parameters scaled to system dynamics
    eta = 0.02 * k * 0.1         # ~2% of max spring force at x=0.1
    phi = 0.05                    # boundary layer width

    return {
        "c1": c1,
        "c2": c2,
        "c3": c3,
        "eta": eta,
        "phi": phi,
    }


@njit(cache=True)
def sliding_surface(x, theta, x_dot, theta_dot, c1, c2, c3):
    """Compute the sliding variable s."""
    return c1 * x + c2 * theta + c3 * x_dot + theta_dot


@njit(cache=True)
def sat(val, phi):
    """Saturation function: smoothed sign() within boundary layer phi."""
    if val > phi:
        return 1.0
    elif val < -phi:
        return -1.0
    else:
        return val / phi


@njit(cache=True)
def sliding_mode_control(x, theta, x_dot, theta_dot, p,
                         c1, c2, c3, eta, phi):
    """Compute SMC torque.

    Parameters
    ----------
    x, theta, x_dot, theta_dot : float  State variables.
    p        : float64[4]  Packed parameters.
    c1, c2, c3 : float  Sliding surface coefficients.
    eta      : float  Switching gain.
    phi      : float  Boundary layer thickness.

    Returns
    -------
    tau : float  Rotor torque [N*m].
    """
    Mt = p[0]
    me = p[1]
    I_eff = p[2]
    k = p[3]

    ct = np.cos(theta)
    st = np.sin(theta)
    coupling = me * ct

    # Determinant of mass matrix
    det_M = Mt * I_eff - coupling * coupling

    # Compute ddx in terms of tau (from EOM row 1):
    # ddx = (I_eff * rhs0 - coupling * tau) / det_M
    # where rhs0 = me*theta_dot^2*sin(theta) - k*x
    rhs0 = me * theta_dot * theta_dot * st - k * x

    # ds/dt = c1*x_dot + c2*theta_dot + c3*ddx + ddtheta
    # We need: ds/dt = 0 for equivalent control
    # ddtheta = (Mt*tau - coupling*rhs0) / det_M
    # c3*ddx + ddtheta = c3*(I_eff*rhs0 - coupling*tau)/det_M
    #                   + (Mt*tau - coupling*rhs0)/det_M
    # = [c3*I_eff*rhs0 - c3*coupling*tau + Mt*tau - coupling*rhs0] / det_M
    # = [(c3*I_eff - coupling)*rhs0 + (Mt - c3*coupling)*tau] / det_M

    # Set ds/dt = -c1*x_dot - c2*theta_dot to find tau_eq:
    sigma = -c1 * x_dot - c2 * theta_dot
    numer = sigma * det_M - (c3 * I_eff - coupling) * rhs0
    denom = Mt - c3 * coupling

    if abs(denom) < 1e-12:
        tau_eq = 0.0
    else:
        tau_eq = numer / denom

    # Sliding variable
    s = sliding_surface(x, theta, x_dot, theta_dot, c1, c2, c3)

    # Total control: equivalent + switching
    tau = tau_eq - eta * sat(s, phi)
    return tau
