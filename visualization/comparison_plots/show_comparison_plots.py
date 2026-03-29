"""4-controller comparison plots.

Layout: 2x2
    [cart displacement, rotor angle]
    [control torque,    performance metrics bar chart]
"""

import numpy as np
import matplotlib.pyplot as plt

from visualization.common.colors import COLORS, LABELS
from visualization.common.axis_style import apply_style


def show_comparison_plots(comparison_result):
    """Generate the 2x2 controller comparison figure."""
    t = comparison_result["t"]
    controllers = ["lqr", "energy", "smc"]
    if "ilqr" in comparison_result:
        controllers.append("ilqr")

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("Controller Comparison", fontsize=14, fontweight="bold")

    # Cart displacement
    ax = axes[0, 0]
    for name in controllers:
        data = comparison_result[name]
        t_plot = np.arange(len(data["z"])) * (t[1] - t[0])
        ax.plot(t_plot, data["z"][:, 0], color=COLORS[name], linewidth=0.8,
                label=LABELS.get(name, name))
    ax.set_ylabel("x [m]")
    ax.set_title("Cart Displacement")
    ax.legend(fontsize=7)
    apply_style(ax)

    # Rotor angle
    ax = axes[0, 1]
    for name in controllers:
        data = comparison_result[name]
        t_plot = np.arange(len(data["z"])) * (t[1] - t[0])
        ax.plot(t_plot, data["z"][:, 1], color=COLORS[name], linewidth=0.8,
                label=LABELS.get(name, name))
    ax.set_ylabel("θ [rad]")
    ax.set_title("Rotor Angle")
    ax.legend(fontsize=7)
    apply_style(ax)

    # Control torque
    ax = axes[1, 0]
    for name in controllers:
        data = comparison_result[name]
        t_plot = np.arange(len(data["u"])) * (t[1] - t[0])
        ax.plot(t_plot, data["u"], color=COLORS[name], linewidth=0.5, alpha=0.7,
                label=LABELS.get(name, name))
    ax.set_ylabel("Torque τ [N·m]")
    ax.set_xlabel("Time [s]")
    ax.set_title("Control Torque")
    ax.legend(fontsize=7)
    apply_style(ax)

    # Performance metrics bar chart
    ax = axes[1, 1]
    metrics = ["settling_time", "control_effort"]
    x_pos = np.arange(len(metrics))
    width = 0.8 / len(controllers)

    for idx, name in enumerate(controllers):
        m = comparison_result[name]["metrics"]
        vals = [m["settling_time"], m["control_effort"]]
        bars = ax.bar(x_pos + idx * width, vals, width,
                      color=COLORS[name], alpha=0.8, label=LABELS.get(name, name))

    ax.set_xticks(x_pos + width * (len(controllers) - 1) / 2)
    ax.set_xticklabels(["Settling [s]", "Effort [J]"])
    ax.set_title("Performance Metrics")
    ax.legend(fontsize=7)
    apply_style(ax, zero_line=False)

    fig.tight_layout()
    return fig
