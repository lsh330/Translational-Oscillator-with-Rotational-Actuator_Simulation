"""Gain and phase margin computation."""

import numpy as np


def stability_margins(omega, L):
    """Compute gain margin and phase margin.

    Returns
    -------
    result : dict  Keys: gain_margin_dB, phase_margin_deg,
                          wg (gain crossover freq), wp (phase crossover freq).
    """
    mag = np.abs(L)
    phase = np.unwrap(np.angle(L))

    # Gain crossover: |L| = 1 (0 dB)
    gain_crossover = np.where(np.diff(np.sign(mag - 1.0)))[0]
    if len(gain_crossover) > 0:
        idx = gain_crossover[0]
        # Interpolate
        frac = (1.0 - mag[idx]) / (mag[idx + 1] - mag[idx] + 1e-30)
        wg = omega[idx] + frac * (omega[idx + 1] - omega[idx])
        phase_at_gc = phase[idx] + frac * (phase[idx + 1] - phase[idx])
        phase_margin = 180.0 + np.degrees(phase_at_gc)
    else:
        wg = np.nan
        phase_margin = np.inf

    # Phase crossover: phase = -180 deg
    target = -np.pi
    phase_crossover = np.where(np.diff(np.sign(phase - target)))[0]
    if len(phase_crossover) > 0:
        idx = phase_crossover[0]
        frac = (target - phase[idx]) / (phase[idx + 1] - phase[idx] + 1e-30)
        wp = omega[idx] + frac * (omega[idx + 1] - omega[idx])
        mag_at_pc = mag[idx] + frac * (mag[idx + 1] - mag[idx])
        gain_margin = -20.0 * np.log10(mag_at_pc + 1e-30)
    else:
        wp = np.nan
        gain_margin = np.inf

    return {
        "gain_margin_dB": gain_margin,
        "phase_margin_deg": phase_margin,
        "wg": wg,
        "wp": wp,
    }
