"""Phase portrait plots for the TORA.

Layout: 1x2
    [(x, theta) configuration space, (x_dot, theta_dot) velocity space]

Trajectories are colored by time to show the spiral-in behavior.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from visualization.common.axis_style import apply_style


def _colored_line(x, y, c, ax, cmap="viridis", linewidth=0.8):
    """Plot a line colored by a third variable."""
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=cmap, linewidths=linewidth)
    lc.set_array(c[:-1])
    return ax.add_collection(lc)


def show_phase_plots(sim_result):
    """Generate the 1x2 phase portrait figure."""
    x = sim_result["x"]
    theta = sim_result["theta"]
    xd = sim_result["x_dot"]
    td = sim_result["theta_dot"]
    t = sim_result["t"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("TORA Phase Portraits", fontsize=14, fontweight="bold")

    # Configuration space (x, theta)
    ax = axes[0]
    lc = _colored_line(x, theta, t, ax)
    fig.colorbar(lc, ax=ax, label="Time [s]")
    ax.scatter([x[0]], [theta[0]], c="green", s=60, zorder=5, marker="o", label="Start")
    ax.scatter([x[-1]], [theta[-1]], c="red", s=60, zorder=5, marker="s", label="End")
    ax.scatter([0], [0], c="black", s=100, zorder=5, marker="+", linewidths=2, label="Equilibrium")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("θ [rad]")
    ax.set_title("Configuration Space")
    ax.legend(fontsize=7)
    apply_style(ax)
    ax.set_aspect("auto")

    # Velocity space (x_dot, theta_dot)
    ax = axes[1]
    lc = _colored_line(xd, td, t, ax)
    fig.colorbar(lc, ax=ax, label="Time [s]")
    ax.scatter([xd[0]], [td[0]], c="green", s=60, zorder=5, marker="o", label="Start")
    ax.scatter([xd[-1]], [td[-1]], c="red", s=60, zorder=5, marker="s", label="End")
    ax.scatter([0], [0], c="black", s=100, zorder=5, marker="+", linewidths=2)
    ax.set_xlabel("ẋ [m/s]")
    ax.set_ylabel("θ̇ [rad/s]")
    ax.set_title("Velocity Space")
    ax.legend(fontsize=7)
    apply_style(ax)
    ax.set_aspect("auto")

    fig.tight_layout()
    return fig
