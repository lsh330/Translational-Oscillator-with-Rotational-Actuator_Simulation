"""Seeded Gaussian white noise generation."""

import numpy as np


def white_noise(N, seed=42):
    """Generate N samples of unit-variance Gaussian noise."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(N)
