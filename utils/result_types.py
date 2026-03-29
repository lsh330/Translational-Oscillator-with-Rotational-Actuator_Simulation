"""Typed result containers for simulation outputs."""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class SimulationResult:
    """Container for time-domain simulation output."""
    t: np.ndarray
    x: np.ndarray
    theta: np.ndarray
    x_dot: np.ndarray
    theta_dot: np.ndarray
    u: np.ndarray
    disturbance: np.ndarray
    sat_count: int
    controller: str
    u_raw: Optional[np.ndarray] = None
    s: Optional[np.ndarray] = None

    def to_dict(self):
        """Convert to dict for backward compatibility."""
        d = {
            "t": self.t, "x": self.x, "theta": self.theta,
            "x_dot": self.x_dot, "theta_dot": self.theta_dot,
            "u": self.u, "disturbance": self.disturbance,
            "sat_count": self.sat_count, "controller": self.controller,
        }
        if self.u_raw is not None:
            d["u_raw"] = self.u_raw
        if self.s is not None:
            d["s"] = self.s
        return d


@dataclass
class LQRResult:
    """Container for LQR design output."""
    K: np.ndarray
    A: np.ndarray
    B: np.ndarray
    P: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    poles_cl: np.ndarray


@dataclass
class ComparisonMetrics:
    """Performance metrics for controller comparison."""
    settling_time: float
    overshoot_pct: float
    control_effort: float
    final_state_norm: float
    integral_x2: float
    integral_theta2: float
    peak_theta: float
    peak_theta_dot: float
    peak_torque: float
    sat_fraction: float = 0.0
