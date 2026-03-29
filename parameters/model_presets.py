"""Model presets for different simulation fidelity levels.

Level A: Ideal benchmark (canonical TORA, no losses)
Level B: Engineering model (damping, actuator lag, friction)
"""

from parameters.config import SystemConfig


def ideal_benchmark():
    """Level A: Canonical TORA benchmark (Bupp-Bernstein-Coppola).

    No damping, no friction, no actuator dynamics.
    Pure textbook dynamics for control algorithm comparison.
    """
    return SystemConfig(
        M=1.3608, m=0.096, e=0.0592, k=186.3, I=0.0002175,
        c_x=0.0, c_theta=0.0,
    )


def engineering_model(damping_level="light"):
    """Level B: Engineering model with realistic losses.

    Parameters
    ----------
    damping_level : str  "light", "moderate", or "heavy".
    """
    damping = {
        "light":    {"c_x": 0.5,  "c_theta": 0.001},
        "moderate": {"c_x": 2.0,  "c_theta": 0.005},
        "heavy":    {"c_x": 5.0,  "c_theta": 0.01},
    }
    d = damping.get(damping_level, damping["light"])

    return SystemConfig(
        M=1.3608, m=0.096, e=0.0592, k=186.3, I=0.0002175,
        c_x=d["c_x"], c_theta=d["c_theta"],
    )
