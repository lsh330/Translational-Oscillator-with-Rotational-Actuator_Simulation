"""Serialize derived parameters into a flat float64 array for Numba JIT.

Packing layout (6 elements):
    p[0] = Mt        total mass
    p[1] = me        mass-eccentricity product
    p[2] = I_eff     effective rotor inertia
    p[3] = k         spring constant
    p[4] = c_x       cart viscous damping coefficient
    p[5] = c_theta   rotor bearing damping coefficient
"""

import numpy as np

from parameters.derived import DerivedParams


def pack_params(dp: DerivedParams, c_x=0.0, c_theta=0.0) -> np.ndarray:
    """Pack derived parameters into a contiguous float64 vector."""
    return np.array([dp.Mt, dp.me, dp.I_eff, dp.k, c_x, c_theta], dtype=np.float64)


def unpack_params(p: np.ndarray) -> DerivedParams:
    """Reconstruct DerivedParams from a packed vector."""
    return DerivedParams(Mt=p[0], me=p[1], I_eff=p[2], k=p[3])
