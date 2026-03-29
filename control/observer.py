"""Luenberger state observer for TORA output feedback.

Estimates full state [x, theta, x_dot, theta_dot] from
measured outputs [x, theta] (position-only sensing).

Observer dynamics:
    z_hat_dot = A * z_hat + B * tau + L * (y - C * z_hat)

where L is the observer gain matrix chosen to place
observer poles faster than controller poles.
"""

import numpy as np
from scipy.linalg import place_poles
from numba import njit

from control.linearization.linearize import linearize
from utils.logger import get_logger

_log = get_logger("tora.observer")


def design_observer(p, pole_multiplier=3.0):
    """Design a Luenberger observer for the TORA.

    Places observer poles at pole_multiplier times the real part
    of the LQR closed-loop poles (faster estimation than control).

    Parameters
    ----------
    p               : float64[6]  Packed parameters.
    pole_multiplier : float  Observer poles = multiplier * controller poles.

    Returns
    -------
    result : dict  Keys: L (4,2), A, C, observer_poles.
    """
    A, B = linearize(p, method="analytical")

    # Output matrix: measure x and theta
    C = np.zeros((2, 4))
    C[0, 0] = 1.0  # x
    C[1, 1] = 1.0  # theta

    # Place observer poles (need to work with A^T, C^T for dual problem)
    # Desired observer poles — fast and well-damped
    omega_n = np.sqrt(p[3] / p[0])
    desired_poles = np.array([
        -pole_multiplier * omega_n + 1j * omega_n,
        -pole_multiplier * omega_n - 1j * omega_n,
        -pole_multiplier * omega_n * 1.5 + 1j * omega_n * 0.5,
        -pole_multiplier * omega_n * 1.5 - 1j * omega_n * 0.5,
    ])

    # Observer gain: L such that (A - LC) has desired poles
    # Dual: place poles of (A^T - C^T L^T) = place_poles(A^T, C^T, poles)
    result = place_poles(A.T, C.T, desired_poles)
    L = result.gain_matrix.T

    obs_poles = np.linalg.eigvals(A - L @ C)
    _log.info("Observer poles: %s", np.array2string(obs_poles, precision=4))

    return {
        "L": L,
        "A": A,
        "C": C,
        "observer_poles": obs_poles,
    }


@njit(cache=True)
def observer_update(z_hat, y_measured, tau, A_flat, B_flat, L_flat, C_flat, dt):
    """One-step observer update (Euler integration).

    Parameters
    ----------
    z_hat      : float64[4]  Current state estimate.
    y_measured : float64[2]  Measured [x, theta].
    tau        : float        Applied torque.
    A_flat     : float64[16]  Flattened 4x4 A matrix.
    B_flat     : float64[4]   Flattened 4x1 B matrix.
    L_flat     : float64[8]   Flattened 4x2 L matrix.
    C_flat     : float64[8]   Flattened 2x4 C matrix.
    dt         : float        Time step.

    Returns
    -------
    z_hat_new : float64[4]  Updated state estimate.
    """
    # Reshape flattened matrices
    A = A_flat.reshape(4, 4)
    B = B_flat.reshape(4, 1)
    L = L_flat.reshape(4, 2)
    C = C_flat.reshape(2, 4)

    # Innovation
    y_hat = C @ z_hat
    innovation = y_measured - y_hat

    # State derivative
    dz = A @ z_hat + B.flatten() * tau + L @ innovation

    # Euler step
    z_hat_new = z_hat + dt * dz
    return z_hat_new
