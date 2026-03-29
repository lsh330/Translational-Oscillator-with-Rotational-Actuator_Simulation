"""Closed-loop frequency response T(jw) = L(jw) / (1 + L(jw))."""

import numpy as np


def closed_loop_response(L, omega):
    """Compute closed-loop response from open-loop L(jw).

    Parameters
    ----------
    L     : complex128[M]  Open-loop transfer function.
    omega : float64[M]     Frequency vector.

    Returns
    -------
    mag : float64[M]  |T(jw)| in dB.
    phase : float64[M]  angle(T(jw)) in degrees.
    T : complex128[M]  Raw values.
    """
    T = L / (1.0 + L)
    mag = 20.0 * np.log10(np.abs(T) + 1e-30)
    phase = np.degrees(np.angle(T))
    return mag, phase, T
