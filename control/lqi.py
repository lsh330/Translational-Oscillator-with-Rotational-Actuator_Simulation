"""Linear Quadratic Integral (LQI) controller for TORA.

Augments the state with integral of cart displacement error
to achieve zero steady-state offset under constant disturbances.

Augmented state: z_aug = [x, theta, x_dot, theta_dot, integral_x]
"""

import numpy as np
from scipy.linalg import solve_continuous_are
from utils.logger import get_logger

_log = get_logger("tora.lqi")


def compute_lqi(p, tau_max=0.1):
    """Compute LQI gain for the augmented system.

    Parameters
    ----------
    p       : float64[6]  Packed parameters.
    tau_max : float  Torque saturation for Bryson's rule.

    Returns
    -------
    result : dict  Keys: K_aug (1,5), A_aug (5,5), B_aug (5,1).
    """
    from control.linearization.linearize import linearize

    A, B = linearize(p, method="analytical")

    # Augment: add integral of x (output = x = C @ z, C = [1,0,0,0])
    n = 4
    A_aug = np.zeros((n + 1, n + 1))
    A_aug[:n, :n] = A
    A_aug[n, 0] = 1.0  # d(int_x)/dt = x

    B_aug = np.zeros((n + 1, 1))
    B_aug[:n, :] = B

    # Cost matrices (Bryson's rule extended)
    Q_aug = np.diag([100.0, 1.0, 1.0, 0.01, 500.0])  # heavy integral penalty
    R = np.array([[1.0 / tau_max**2]])

    P = solve_continuous_are(A_aug, B_aug, Q_aug, R)
    K_aug = np.linalg.solve(R, B_aug.T @ P)

    poles = np.linalg.eigvals(A_aug - B_aug @ K_aug)
    _log.info("LQI poles: %s", np.array2string(poles, precision=4))

    return {
        "K_aug": K_aug,
        "A_aug": A_aug,
        "B_aug": B_aug,
        "P_aug": P,
        "Q_aug": Q_aug,
        "R": R,
        "poles_cl": poles,
    }
