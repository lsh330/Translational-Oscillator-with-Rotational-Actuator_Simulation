"""Butterworth low-pass filter for band-limiting noise."""

import numpy as np


def lowpass_filter(signal, dt, bandwidth):
    """Apply 4th-order Butterworth low-pass via FFT.

    Parameters
    ----------
    signal    : float64[N]  Input signal.
    dt        : float        Sampling period [s].
    bandwidth : float        Cutoff frequency [Hz].

    Returns
    -------
    filtered : float64[N]  Band-limited signal.
    """
    N = len(signal)
    freqs = np.fft.rfftfreq(N, d=dt)
    spectrum = np.fft.rfft(signal)

    # 4th-order Butterworth magnitude response
    H = 1.0 / np.sqrt(1.0 + (freqs / bandwidth) ** 8)
    H[0] = 0.0  # Remove DC

    filtered = np.fft.irfft(spectrum * H, n=N)
    return filtered
