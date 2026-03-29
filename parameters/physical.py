"""Physical parameters for the TORA system.

Standard benchmark values from Bupp, Bernstein & Coppola (1998),
"Experimental Implementation of Integrator Backstepping and Passive
Nonlinear Controllers on the RTAC Testbed",
Int. J. Robust and Nonlinear Control, 8(4/5), 435-457.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PhysicalParams:
    """Immutable container for TORA physical parameters.

    Attributes
    ----------
    M : float  Cart mass [kg].
    m : float  Eccentric rotor mass [kg].
    e : float  Rotor eccentricity [m].
    k : float  Spring constant [N/m].
    I : float  Rotor inertia about its center [kg*m^2].
    """

    M: float = 1.3608
    m: float = 0.096
    e: float = 0.0592
    k: float = 186.3
    I: float = 0.0002175
