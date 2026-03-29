"""Dynamics plots: states, energy, coupling, control torque.

Layout: 3x2 grid
    [cart displacement, rotor angle]
    [cart velocity,     rotor velocity]
    [energy,            coupling + control]
"""

import numpy as np
import matplotlib.pyplot as plt

from analysis.energy.total_energy import total_energy
from analysis.coupling.coupling_strength import coupling_strength
from visualization.common.colors import COLORS
from visualization.common.axis_style import apply_style


def show_dynamics_plots(sim_result, p):
    """Generate the 3x2 dynamics overview figure.

    Returns
    -------
    fig : matplotlib Figure.
    """
    t = sim_result["t"]
    x = sim_result["x"]
    theta = sim_result["theta"]
    xd = sim_result["x_dot"]
    td = sim_result["theta_dot"]
    u = sim_result["u"]

    H, T, V = total_energy(x, xd, theta, td, p)
    coupling = coupling_strength(theta, p)

    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    fig.suptitle("TORA Dynamics Overview", fontsize=14, fontweight="bold")

    # Cart displacement
    ax = axes[0, 0]
    ax.plot(t, x, color=COLORS["cart"], linewidth=0.8)
    ax.set_ylabel("x [m]")
    ax.set_title("Cart Displacement")
    apply_style(ax)

    # Rotor angle
    ax = axes[0, 1]
    ax.plot(t, theta, color=COLORS["rotor"], linewidth=0.8)
    ax.set_ylabel("θ [rad]")
    ax.set_title("Rotor Angle")
    apply_style(ax)

    # Cart velocity
    ax = axes[1, 0]
    ax.plot(t, xd, color=COLORS["cart"], linewidth=0.8)
    ax.set_ylabel("ẋ [m/s]")
    ax.set_title("Cart Velocity")
    apply_style(ax)

    # Rotor angular velocity
    ax = axes[1, 1]
    ax.plot(t, td, color=COLORS["rotor"], linewidth=0.8)
    ax.set_ylabel("θ̇ [rad/s]")
    ax.set_title("Rotor Angular Velocity")
    apply_style(ax)

    # Energy
    ax = axes[2, 0]
    ax.plot(t, H, color=COLORS["energy_total"], linewidth=1.0, label="Total H")
    ax.plot(t, T, color=COLORS["energy_kinetic"], linewidth=0.7, alpha=0.7, label="Kinetic T")
    ax.plot(t, V, color=COLORS["energy_potential"], linewidth=0.7, alpha=0.7, label="Potential V")
    ax.set_ylabel("Energy [J]")
    ax.set_xlabel("Time [s]")
    ax.set_title("Energy")
    ax.legend(fontsize=7, loc="upper right")
    apply_style(ax)

    # Coupling & Control — split into two clear subplots stacked vertically
    ax = axes[2, 1]
    t_u = t[:len(u)]

    # Plot coupling on primary axis with clear color
    ax.fill_between(t, coupling["normalized"], alpha=0.25, color=COLORS["coupling"])
    ax.plot(t, coupling["normalized"], color=COLORS["coupling"], linewidth=1.0,
            label="Coupling cos(θ)")
    ax.set_ylabel("Normalized Coupling", fontsize=9)
    ax.set_ylim(-1.2, 1.2)
    ax.legend(fontsize=7, loc="lower left")
    apply_style(ax, zero_line=True)

    # Overlay control torque on twin axis with distinct style
    ax2 = ax.twinx()
    ax2.plot(t_u, u, color=COLORS["control"], linewidth=0.8, alpha=0.85, label="τ [N·m]")
    ax2.set_ylabel("Torque τ [N·m]", fontsize=9, color=COLORS["control"])
    ax2.tick_params(axis="y", colors=COLORS["control"])
    ax2.legend(fontsize=7, loc="upper right")
    ax.set_xlabel("Time [s]")
    ax.set_title("Coupling Strength & Control Torque")

    fig.tight_layout()
    return fig
