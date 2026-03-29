"""End-to-end LQR optimal control design for the TORA.

Workflow:
    1. Linearize about equilibrium -> A, B
    2. Form cost matrices Q, R
    3. Solve CARE -> P
    4. Compute gain K = R^{-1} B^T P
    5. Verify closed-loop stability
"""

import numpy as np

from control.linearization.linearize import linearize
from control.cost_matrices.default_Q import default_Q, adaptive_Q
from control.cost_matrices.default_R import default_R
from control.riccati.solve_care import solve_care
from control.gain_computation.compute_K import compute_K
from utils.logger import get_logger

_log = get_logger("tora.lqr")


def compute_lqr(p, use_adaptive_q=False, tau_max=0.1):
    """Compute LQR gain and return all design matrices.

    Parameters
    ----------
    p              : float64[4]  Packed parameters.
    use_adaptive_q : bool  Use Bryson's rule Q instead of default.
    tau_max        : float  Maximum torque for Bryson's rule R.

    Returns
    -------
    result : dict  Keys: K, A, B, P, Q, R, poles_cl.
    """
    A, B = linearize(p, method="analytical")

    Q = adaptive_Q(p) if use_adaptive_q else default_Q()
    R = default_R(tau_max)

    P = solve_care(A, B, Q, R)
    K = compute_K(B, R, P)

    # Closed-loop poles
    A_cl = A - B @ K
    poles_cl = np.linalg.eigvals(A_cl)

    # Verify stability
    max_real = np.max(poles_cl.real)
    if max_real >= 0:
        _log.warning("Closed-loop NOT stable! Max Re(pole) = %.4e", max_real)
    else:
        _log.info("LQR stable: max Re(pole) = %.4e", max_real)

    _log.info("LQR gain K = %s", np.array2string(K, precision=4))

    return {
        "K": K,
        "A": A,
        "B": B,
        "P": P,
        "Q": Q,
        "R": R,
        "poles_cl": poles_cl,
    }
