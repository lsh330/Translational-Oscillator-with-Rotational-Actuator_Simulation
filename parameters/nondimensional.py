"""Non-dimensional TORA model parameters.

The TORA system can be non-dimensionalized using:
    Time scale:   t* = omega_n * t,  where omega_n = sqrt(k / Mt)
    Length scale:  x* = x / L_ref,   where L_ref = e (eccentricity)

The single coupling parameter that governs the dynamics:
    epsilon = me / sqrt(Mt * I_eff)

For the standard benchmark: epsilon ≈ 0.200

When epsilon → 0, the cart and rotor decouple completely.
When epsilon → 1, the system approaches a singular configuration.
"""

import numpy as np

from parameters.physical import PhysicalParams
from parameters.derived import compute_derived


def compute_nondimensional(pp: PhysicalParams) -> dict:
    """Compute non-dimensional parameters.

    Returns
    -------
    result : dict  Keys: epsilon, omega_n, T_natural, L_ref.
    """
    dp = compute_derived(pp)

    omega_n = np.sqrt(dp.k / dp.Mt)
    epsilon = dp.me / np.sqrt(dp.Mt * dp.I_eff)
    T_natural = 2.0 * np.pi / omega_n
    L_ref = pp.e

    return {
        "epsilon": epsilon,
        "omega_n": omega_n,
        "T_natural": T_natural,
        "L_ref": L_ref,
        "Mt": dp.Mt,
        "me": dp.me,
        "I_eff": dp.I_eff,
        "k": dp.k,
    }
