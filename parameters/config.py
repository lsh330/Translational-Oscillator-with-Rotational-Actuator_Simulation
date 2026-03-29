"""Unified system configuration with validation."""

import numpy as np

from parameters.physical import PhysicalParams
from parameters.derived import DerivedParams, compute_derived
from parameters.packing import pack_params
from parameters.equilibrium import equilibrium as _equilibrium


class SystemConfig:
    """Top-level configuration: validates, derives, and packs parameters.

    Parameters
    ----------
    M, m, e, k, I : float
        Physical parameters (see PhysicalParams for units).
    """

    def __init__(self, M=1.3608, m=0.096, e=0.0592, k=186.3, I=0.0002175):
        for name, val in [("M", M), ("m", m), ("e", e), ("k", k), ("I", I)]:
            if not isinstance(val, (int, float)) or val <= 0:
                raise ValueError(f"{name} must be a positive number, got {val}")

        self.physical = PhysicalParams(M=M, m=m, e=e, k=k, I=I)
        self.derived: DerivedParams = compute_derived(self.physical)
        self._packed: np.ndarray = pack_params(self.derived)

    def pack(self) -> np.ndarray:
        """Return packed parameter vector for JIT functions."""
        return self._packed

    @property
    def equilibrium(self) -> np.ndarray:
        """Equilibrium state vector [x, theta, x_dot, theta_dot]."""
        return _equilibrium()

    def __repr__(self) -> str:
        pp = self.physical
        return (
            f"SystemConfig(M={pp.M}, m={pp.m}, e={pp.e}, "
            f"k={pp.k}, I={pp.I})"
        )
