"""Open-loop frequency response L(jw) = K(jwI - A)^{-1}B."""

import numpy as np


def open_loop_response(A, B, K, omega):
    """Compute open-loop transfer function magnitude and phase.

    Parameters
    ----------
    A : (4,4)  State matrix.
    B : (4,1)  Input matrix.
    K : (1,4)  Gain matrix.
    omega : float64[M]  Frequency vector [rad/s].

    Returns
    -------
    mag : float64[M]  |L(jw)| in dB.
    phase : float64[M]  angle(L(jw)) in degrees.
    L : complex128[M]  Raw transfer function values.
    """
    I4 = np.eye(4)
    L = np.empty(len(omega), dtype=np.complex128)

    for i, w in enumerate(omega):
        resolvent = np.linalg.solve(1j * w * I4 - A, B)
        L[i] = (K @ resolvent)[0, 0]

    mag = 20.0 * np.log10(np.abs(L) + 1e-30)
    phase = np.degrees(np.angle(L))
    return mag, phase, L
