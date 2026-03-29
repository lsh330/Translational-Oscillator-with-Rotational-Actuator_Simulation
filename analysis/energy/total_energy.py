"""Total energy (Hamiltonian) for the TORA.

    H = T + V = 0.5*dq^T*M(q)*dq + 0.5*k*x^2

For the uncontrolled system (tau=0), H is conserved.
"""

from analysis.energy.kinetic_energy import kinetic_energy
from analysis.energy.spring_potential_energy import spring_potential_energy


def total_energy(x, x_dot, theta, theta_dot, p):
    """Compute total energy time series.

    Returns
    -------
    H, T, V : float64[N]  Total, kinetic, potential energy.
    """
    T = kinetic_energy(x_dot, theta, theta_dot, p)
    V = spring_potential_energy(x, p)
    H = T + V
    return H, T, V
