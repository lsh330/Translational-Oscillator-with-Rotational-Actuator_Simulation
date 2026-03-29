"""Control analysis plots: torque, Bode magnitude/phase, margins.

Layout: 2x2 grid
    [torque time history,  torque spectrum]
    [Bode magnitude,       Bode phase     ]
"""

import numpy as np
import matplotlib.pyplot as plt

from visualization.common.colors import COLORS
from visualization.common.axis_style import apply_style


def show_control_plots(sim_result, freq_result):
    """Generate the 2x2 control analysis figure.

    Returns
    -------
    fig : matplotlib Figure.
    """
    t = sim_result["t"]
    u = sim_result["u"]
    dt = t[1] - t[0]

    omega = freq_result["omega"]
    ol = freq_result["open_loop"]
    margins = freq_result["margins"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("TORA Control Analysis", fontsize=14, fontweight="bold")

    # Torque time history — use fill + envelope for dense signals
    ax = axes[0, 0]
    t_u = t[:len(u)]
    ax.fill_between(t_u, u, alpha=0.3, color=COLORS["control"])
    ax.plot(t_u, u, color=COLORS["control"], linewidth=0.4, alpha=0.7)
    # Add running envelope for readability
    win = min(200, len(u) // 10)
    if win > 1:
        from numpy.lib.stride_tricks import sliding_window_view
        u_abs = np.abs(u)
        if len(u_abs) > win:
            env = np.max(sliding_window_view(u_abs, win), axis=1)
            t_env = t_u[:len(env)]
            ax.plot(t_env, env, color="black", linewidth=1.2, alpha=0.8, label="Envelope")
            ax.plot(t_env, -env, color="black", linewidth=1.2, alpha=0.8)
            ax.legend(fontsize=7)
    ax.set_ylabel("Torque τ [N·m]")
    ax.set_xlabel("Time [s]")
    ax.set_title("Control Torque")
    apply_style(ax)

    # Torque frequency spectrum
    ax = axes[0, 1]
    N_fft = len(u)
    freqs = np.fft.rfftfreq(N_fft, d=dt)
    spectrum = np.abs(np.fft.rfft(u)) / N_fft
    ax.semilogy(freqs[1:], spectrum[1:], color=COLORS["control"], linewidth=0.6)
    ax.set_ylabel("|FFT(τ)|")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_title("Torque Spectrum")
    apply_style(ax, zero_line=False)

    # Bode magnitude
    ax = axes[1, 0]
    ax.semilogx(omega, ol["mag"], color=COLORS["lqr"], linewidth=1.0)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    if not np.isnan(margins["wg"]):
        ax.axvline(margins["wg"], color="r", linewidth=0.8, linestyle=":",
                   label=f"ωg={margins['wg']:.1f} rad/s")
    ax.set_ylabel("Magnitude [dB]")
    ax.set_xlabel("ω [rad/s]")
    ax.set_title(f"Open-Loop Bode (GM={margins['gain_margin_dB']:.1f} dB)")
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(fontsize=7)
    apply_style(ax, zero_line=False)

    # Bode phase
    ax = axes[1, 1]
    ax.semilogx(omega, ol["phase"], color=COLORS["lqr"], linewidth=1.0)
    ax.axhline(-180, color="gray", linewidth=0.5, linestyle="--")
    if not np.isnan(margins["wg"]):
        ax.axvline(margins["wg"], color="r", linewidth=0.8, linestyle=":")
    ax.set_ylabel("Phase [deg]")
    ax.set_xlabel("ω [rad/s]")
    ax.set_title(f"Phase (PM={margins['phase_margin_deg']:.1f}°)")
    apply_style(ax, zero_line=False)

    fig.tight_layout()
    return fig
