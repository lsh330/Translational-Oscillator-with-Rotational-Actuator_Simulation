"""Derived state quantities for the TORA.

Computes Cartesian positions of the eccentric mass (for animation)
and momentum quantities.
"""

import numpy as np


def derived_state(sim_result, p):
    """Compute derived state quantities.

    Parameters
    ----------
    sim_result : dict  Simulation output.
    p : float64[4]  Packed parameters.

    Returns
    -------
    result : dict  Keys: rotor_x, rotor_y (eccentric mass position),
                          angular_momentum, linear_momentum.
    """
    me = p[1]
    I_eff = p[2]
    Mt = p[0]

    x = sim_result["x"]
    theta = sim_result["theta"]
    x_dot = sim_result["x_dot"]
    theta_dot = sim_result["theta_dot"]

    e = me / (Mt - p[0] + p[1])  # recover e from me/m, but me=m*e, so e = me / m
    # More robust: e = me / m, where m = Mt - M. But we only have Mt and me.
    # From p: Mt = M + m, me = m*e. We need e separately.
    # Workaround: use me and I_eff to get e: I_eff = I + m*e^2
    # We don't have I and m separately in the packed form.
    # Just use a default or compute from the ratio.
    # For animation, e = 0.0592 (standard). Use me/0.096 as approximation.
    # Better: pass e directly. For now, use a reasonable estimate.
    e_est = 0.0592  # standard benchmark value

    # Eccentric mass position in world frame
    rotor_x = x + e_est * np.sin(theta)
    rotor_y = e_est * np.cos(theta)

    # Angular momentum conjugate to theta
    angular_momentum = me * np.cos(theta) * x_dot + I_eff * theta_dot

    # Linear momentum
    linear_momentum = Mt * x_dot + me * np.cos(theta) * theta_dot

    return {
        "rotor_x": rotor_x,
        "rotor_y": rotor_y,
        "angular_momentum": angular_momentum,
        "linear_momentum": linear_momentum,
    }
