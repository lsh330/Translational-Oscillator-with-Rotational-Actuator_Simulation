"""Main linearization entry point: analytical with numerical fallback.

The TORA linearization at the equilibrium is fully analytical
(unlike the triple pendulum which needs numerical A_dq).
The numerical path is provided as a cross-check and fallback.
"""

import numpy as np

from control.linearization.analytical_jacobian import analytical_A, analytical_B
from control.linearization.jit_jacobians import compute_numerical_state_space
from utils.logger import get_logger

_log = get_logger("tora.linearize")


def linearize(p, method="analytical"):
    """Linearize the TORA about the equilibrium.

    Parameters
    ----------
    p      : float64[4]  Packed parameters.
    method : str  "analytical" (default) or "numerical".

    Returns
    -------
    A : float64[4,4]  State matrix.
    B : float64[4,1]  Input matrix.
    """
    if method == "analytical":
        A = analytical_A(p)
        B = analytical_B(p)
        _log.debug("Analytical linearization complete.")
        return A, B

    q_eq = np.zeros(2)
    dq_eq = np.zeros(2)
    tau_eq = 0.0
    A, B = compute_numerical_state_space(q_eq, dq_eq, tau_eq, p)
    _log.debug("Numerical linearization complete.")
    return A, B
