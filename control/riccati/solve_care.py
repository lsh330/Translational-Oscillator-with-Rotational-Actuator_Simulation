"""Continuous Algebraic Riccati Equation (CARE) solver with validation."""

import numpy as np
from scipy.linalg import solve_continuous_are

from utils.logger import get_logger

_log = get_logger("tora.riccati")


def solve_care(A, B, Q, R):
    """Solve the CARE: A'P + PA - PBR^{-1}B'P + Q = 0.

    Parameters
    ----------
    A : (4,4)  State matrix.
    B : (4,1)  Input matrix.
    Q : (4,4)  State cost (must be PSD).
    R : (1,1)  Input cost (must be PD).

    Returns
    -------
    P : (4,4)  Unique stabilizing solution.

    Raises
    ------
    ValueError  If the solution fails validation.
    """
    # Controllability check
    n = A.shape[0]
    C_mat = B.copy()
    AB = B.copy()
    for _ in range(n - 1):
        AB = A @ AB
        C_mat = np.hstack([C_mat, AB])
    rank = np.linalg.matrix_rank(C_mat)
    if rank < n:
        raise ValueError(f"System not controllable (rank={rank}, need {n})")

    P = solve_continuous_are(A, B, Q, R)

    # Validate P
    eigvals_P = np.linalg.eigvalsh(P)
    if np.any(eigvals_P < -1e-10):
        raise ValueError(f"P not positive semi-definite: min eigenvalue = {eigvals_P.min():.2e}")

    sym_err = np.max(np.abs(P - P.T))
    if sym_err > 1e-10:
        _log.warning("P symmetry error: %.2e (forcing symmetry)", sym_err)
        P = 0.5 * (P + P.T)

    # Riccati residual
    residual = A.T @ P + P @ A - P @ B @ np.linalg.solve(R, B.T @ P) + Q
    res_norm = np.max(np.abs(residual))
    if res_norm > 1e-6:
        _log.warning("CARE residual = %.2e (expected < 1e-6)", res_norm)

    _log.debug("CARE solved: P min eig=%.4e, residual=%.2e", eigvals_P.min(), res_norm)
    return P
