"""Generate band-limited torque disturbance for the TORA."""

import numpy as np

from simulation.disturbance.white_noise import white_noise
from simulation.disturbance.bandpass_filter import lowpass_filter
from simulation.disturbance.normalize import normalize_rms


def generate_disturbance(N, dt, amplitude=0.01, bandwidth=5.0, seed=42):
    """Generate a band-limited torque disturbance signal.

    Parameters
    ----------
    N         : int    Number of samples.
    dt        : float  Sampling period [s].
    amplitude : float  Target RMS torque disturbance [N*m].
    bandwidth : float  Cutoff frequency [Hz].
    seed      : int    Random seed.

    Returns
    -------
    d : float64[N]  Disturbance torque signal.
    """
    raw = white_noise(N, seed=seed)
    filtered = lowpass_filter(raw, dt, bandwidth)
    d = normalize_rms(filtered, amplitude)
    return d


def generate_multi_disturbance(N, dt, tau_amplitude=0.01, force_amplitude=0.0,
                                bandwidth=5.0, seed=42):
    """Generate multi-channel disturbance signals.

    Returns
    -------
    result : dict  Keys: torque_dist, force_dist.
    """
    rng_base = seed
    torque_dist = generate_disturbance(N, dt, tau_amplitude, bandwidth, rng_base)
    force_dist = generate_disturbance(N, dt, force_amplitude, bandwidth, rng_base + 1000)
    return {"torque_dist": torque_dist, "force_dist": force_dist}
