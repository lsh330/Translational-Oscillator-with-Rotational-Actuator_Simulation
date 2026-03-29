"""RMS normalization of disturbance signals."""

import numpy as np


def normalize_rms(signal, target_rms):
    """Scale signal to have the specified RMS amplitude."""
    current_rms = np.sqrt(np.mean(signal ** 2))
    if current_rms < 1e-15:
        return signal
    return signal * (target_rms / current_rms)
