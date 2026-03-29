"""Phase portrait data for the TORA.

The 2-DOF TORA allows clean 2D phase portraits:
    - Configuration space: (x, theta)
    - Velocity space: (x_dot, theta_dot)
"""

import numpy as np


def phase_portrait_data(sim_result):
    """Extract phase portrait data from simulation result.

    Returns
    -------
    result : dict  Keys: x, theta, x_dot, theta_dot, t (for coloring).
    """
    return {
        "x": sim_result["x"],
        "theta": sim_result["theta"],
        "x_dot": sim_result["x_dot"],
        "theta_dot": sim_result["theta_dot"],
        "t": sim_result["t"],
    }
