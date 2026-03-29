"""Default Q matrix for the TORA LQR cost function.

State ordering: [x, theta, x_dot, theta_dot].

Both default_Q() and adaptive_Q() use Bryson's rule:
    Q_ii = 1 / (max_acceptable_deviation_i)^2

This ensures consistent physical scaling between state penalties
and the input penalty R = 1/tau_max^2.
"""

import numpy as np


def default_Q(x_max=0.1, theta_max=1.0, x_dot_max=1.0, theta_dot_max=10.0):
    """Return the default 4x4 state cost matrix via Bryson's rule.

    Parameters
    ----------
    x_max         : float  Max acceptable cart displacement [m].
    theta_max     : float  Max acceptable rotor angle [rad].
    x_dot_max     : float  Max acceptable cart velocity [m/s].
    theta_dot_max : float  Max acceptable rotor angular velocity [rad/s].

    Returns
    -------
    Q : float64[4,4]  State cost matrix.
    """
    return np.diag([
        1.0 / x_max ** 2,
        1.0 / theta_max ** 2,
        1.0 / x_dot_max ** 2,
        1.0 / theta_dot_max ** 2,
    ])


def adaptive_Q(p):
    """Bryson's rule with system-derived natural scales.

    Uses the system's physical parameters to set scale limits:
        x_max     = x0 = 0.1 m  (benchmark IC)
        theta_max = 1.0 rad     (moderate rotation)
        x_dot_max = x0 * omega_n
        td_max    = 2*pi * 5    (5 rev/s limit)
    """
    Mt = p[0]
    k = p[3]

    x_max = 0.1
    theta_max = 1.0
    omega_n = np.sqrt(k / Mt)
    x_dot_max = x_max * omega_n
    theta_dot_max = 2.0 * np.pi * 5.0

    return np.diag([
        1.0 / x_max ** 2,
        1.0 / theta_max ** 2,
        1.0 / x_dot_max ** 2,
        1.0 / theta_dot_max ** 2,
    ])
