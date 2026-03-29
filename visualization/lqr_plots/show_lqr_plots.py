"""LQR verification plots: Lyapunov, cost, return difference, poles, MC.

Layout: 3x2 grid
    [Lyapunov V(t),        cost breakdown]
    [return difference,    pole map      ]
    [MC robustness hist,   P eigenvalues ]
"""

import numpy as np
import matplotlib.pyplot as plt

from visualization.common.axis_style import apply_style


def show_lqr_plots(sim_result, verification):
    """Generate the 3x2 LQR verification figure."""
    t = sim_result["t"]
    lyap = verification["lyapunov"]
    cost = verification["cost"]
    rd = verification["return_difference"]
    poles = verification["pole_analysis"]
    omega = verification["omega"]

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle("LQR Verification Suite", fontsize=14, fontweight="bold")

    # Lyapunov function
    ax = axes[0, 0]
    ax.semilogy(t, lyap["V"] + 1e-20, "b-", linewidth=0.8)
    ax.set_ylabel("V(t) = z^T P z")
    ax.set_xlabel("Time [s]")
    ax.set_title(f"Lyapunov Function (monotone: {lyap['monotone_decreasing']})")
    apply_style(ax, zero_line=False)

    # Cost breakdown
    ax = axes[0, 1]
    labels = ["State", "Control", "Total"]
    vals = [cost["state_cost"], cost["control_cost"], cost["total_cost"]]
    colors = ["#2196F3", "#F44336", "#333333"]
    bars = ax.bar(labels, vals, color=colors, alpha=0.8)
    ax.set_ylabel("Cost")
    ax.set_title(f"Cost Breakdown (ratio={cost['cost_ratio']:.2f})")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{v:.4f}", ha="center", va="bottom", fontsize=8)
    apply_style(ax, zero_line=False)

    # Return difference (Kalman inequality)
    ax = axes[1, 0]
    ax.semilogx(omega, rd["return_difference"], "b-", linewidth=0.8)
    ax.axhline(1.0, color="r", linewidth=1, linestyle="--", label="|1+L| = 1")
    ax.set_ylabel("|1 + L(jω)|")
    ax.set_xlabel("ω [rad/s]")
    ax.set_title(f"Kalman Inequality (min={rd['min_return_difference']:.4f})")
    ax.legend(fontsize=7)
    apply_style(ax, zero_line=False)

    # Pole map
    ax = axes[1, 1]
    p = poles["poles"]
    ax.scatter(p.real, p.imag, s=80, c="blue", marker="x", linewidths=2, zorder=5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("Re")
    ax.set_ylabel("Im")
    ax.set_title(f"Closed-Loop Poles (min ζ={poles['min_damping']:.3f})")
    ax.grid(True, alpha=0.3)

    # MC robustness — improved: dual histogram (converged vs diverged)
    ax = axes[2, 0]
    if "monte_carlo" in verification:
        mc = verification["monte_carlo"]
        devs = mc["max_deviations"]
        converged = devs[devs < 10.0]
        diverged = devs[devs >= 10.0]

        bins = np.linspace(0, min(np.max(devs) * 1.1, 15.0), 40)
        if len(converged) > 0:
            ax.hist(converged, bins=bins, color="#4CAF50", alpha=0.8,
                    edgecolor="white", linewidth=0.5, label=f"Stable ({len(converged)})")
        if len(diverged) > 0:
            ax.hist(diverged, bins=bins, color="#F44336", alpha=0.8,
                    edgecolor="white", linewidth=0.5, label=f"Unstable ({len(diverged)})")
        ax.axvline(10.0, color="black", linewidth=1.5, linestyle="--",
                   label="Divergence threshold", alpha=0.7)
        ax.set_xlabel("Max State Deviation", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.set_title(f"MC Robustness ({100*mc['success_rate']:.0f}% stable, N={mc['N_trials']})",
                     fontsize=10)
        ax.legend(fontsize=7, loc="upper right")
    else:
        ax.text(0.5, 0.5, "MC not run", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="gray")
    apply_style(ax, zero_line=False)

    # P matrix eigenvalues
    ax = axes[2, 1]
    from analysis.lqr_verification.compute_verification import lqr_verification
    # We don't have P here directly, so show damping ratios instead
    dr = poles["damping_ratios"]
    ax.bar(range(len(dr)), dr, color="#9C27B0", alpha=0.7)
    ax.set_xticks(range(len(dr)))
    ax.set_xticklabels([f"p{i+1}" for i in range(len(dr))])
    ax.set_ylabel("Damping Ratio ζ")
    ax.set_title("Closed-Loop Damping Ratios")
    apply_style(ax, zero_line=False)

    fig.tight_layout()
    return fig
