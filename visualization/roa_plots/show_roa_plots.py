"""Region of Attraction visualization.

The TORA's 2-DOF configuration allows a clean 2D contour plot
of the ROA in (x0, theta0) space.
"""

import numpy as np
import matplotlib.pyplot as plt

from visualization.common.axis_style import apply_style


def show_roa_plots(roa_result):
    """Generate the ROA visualization figure."""
    x_grid = roa_result["x_grid"]
    theta_grid = roa_result["theta_grid"]
    success_map = roa_result["success_map"]
    rate = roa_result["success_rate"]

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    fig.suptitle(f"Region of Attraction (success rate: {100*rate:.1f}%)",
                 fontsize=14, fontweight="bold")

    X, TH = np.meshgrid(x_grid, theta_grid, indexing="ij")
    cf = ax.contourf(X, TH, success_map, levels=[0, 0.5, 1.0],
                     colors=["#FFCDD2", "#C8E6C9"], alpha=0.8)
    ax.contour(X, TH, success_map, levels=[0.5],
               colors=["black"], linewidths=2)

    ax.scatter([0], [0], c="black", s=100, marker="+", linewidths=2,
              label="Equilibrium", zorder=5)

    ax.set_xlabel("x₀ [m]")
    ax.set_ylabel("θ₀ [rad]")
    ax.set_title("LQR Region of Attraction")

    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#C8E6C9", edgecolor="black", label="Converged"),
        Patch(facecolor="#FFCDD2", edgecolor="black", label="Diverged"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="upper right")

    apply_style(ax)
    fig.tight_layout()
    return fig
