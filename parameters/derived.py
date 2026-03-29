"""Derived parameters computed from physical quantities.

Derived set (4 values):
    Mt    = M + m           total translating mass [kg]
    me    = m * e           mass-eccentricity product [kg*m]
    I_eff = I + m * e^2     effective rotor inertia [kg*m^2]
    k     = k               spring constant (passed through) [N/m]
"""

from dataclasses import dataclass

from parameters.physical import PhysicalParams


@dataclass(frozen=True)
class DerivedParams:
    """Derived parameter set for dynamics computation."""

    Mt: float
    me: float
    I_eff: float
    k: float


def compute_derived(pp: PhysicalParams) -> DerivedParams:
    """Compute derived parameters from physical parameters."""
    return DerivedParams(
        Mt=pp.M + pp.m,
        me=pp.m * pp.e,
        I_eff=pp.I + pp.m * pp.e ** 2,
        k=pp.k,
    )
