"""TORA animation: cart on spring with spinning eccentric rotor.

Elements:
    - Wall (fixed, left side)
    - Spring (zigzag from wall to cart)
    - Cart (rectangle sliding on rail)
    - Rotor (circle centered on cart)
    - Eccentric mass (small circle at distance e from rotor center)
    - Rail (ground line)
    - Time/state display
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch


def _draw_spring(ax, x_start, x_end, y, n_coils=8, amplitude=0.015):
    """Draw a zigzag spring between two x positions."""
    n_pts = 4 * n_coils + 2
    xs = np.linspace(x_start, x_end, n_pts)
    ys = np.zeros(n_pts)
    for i in range(1, n_pts - 1):
        ys[i] = amplitude * ((-1) ** i)
    ys += y
    return xs, ys


def show_animation(sim_result, e=0.0592, interval=20, save_path=None):
    """Create and display the TORA animation.

    Parameters
    ----------
    sim_result : dict  Simulation output.
    e          : float  Rotor eccentricity [m].
    interval   : int  Animation frame interval [ms].
    save_path  : str or None  Path to save as GIF.
    """
    x = sim_result["x"]
    theta = sim_result["theta"]
    t = sim_result["t"]
    u = sim_result["u"]

    # Subsample for smooth animation
    step = max(1, len(t) // 1000)
    x_s = x[::step]
    theta_s = theta[::step]
    t_s = t[::step]
    u_s = np.append(u, u[-1])[::step]

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.set_xlim(-0.25, 0.25)
    ax.set_ylim(-0.08, 0.12)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_title("TORA System Animation")

    # Wall
    wall_x = -0.20
    ax.plot([wall_x, wall_x], [-0.06, 0.10], "k-", linewidth=3)
    for y_h in np.linspace(-0.05, 0.09, 8):
        ax.plot([wall_x - 0.01, wall_x], [y_h + 0.01, y_h], "k-", linewidth=1)

    # Rail
    ax.plot([-0.25, 0.25], [-0.03, -0.03], "k-", linewidth=2)

    # Cart dimensions
    cart_w = 0.06
    cart_h = 0.04
    rotor_r = 0.025

    # Dynamic elements
    spring_line, = ax.plot([], [], "g-", linewidth=1.5)
    cart_patch = Rectangle((0, 0), cart_w, cart_h, fc="#2196F3", ec="black", lw=1.5)
    ax.add_patch(cart_patch)

    rotor_circle = Circle((0, 0), rotor_r, fc="white", ec="black", lw=1.5)
    ax.add_patch(rotor_circle)

    ecc_mass = Circle((0, 0), 0.006, fc="#F44336", ec="black", lw=1)
    ax.add_patch(ecc_mass)

    rotor_line, = ax.plot([], [], "k-", linewidth=1.5)
    trace_line, = ax.plot([], [], "r-", alpha=0.3, linewidth=0.5)

    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, fontsize=9,
                        verticalalignment="top", fontfamily="monospace")

    trace_x = []
    trace_y = []

    def init():
        spring_line.set_data([], [])
        cart_patch.set_xy((-100, -100))
        rotor_circle.center = (-100, -100)
        ecc_mass.center = (-100, -100)
        rotor_line.set_data([], [])
        trace_line.set_data([], [])
        time_text.set_text("")
        return spring_line, cart_patch, rotor_circle, ecc_mass, rotor_line, trace_line, time_text

    def update(frame):
        cx = x_s[frame]
        th = theta_s[frame]
        ti = t_s[frame]
        ui = u_s[frame]

        # Spring
        sp_x, sp_y = _draw_spring(ax, wall_x, cx - cart_w / 2, -0.01)
        spring_line.set_data(sp_x, sp_y)

        # Cart
        cart_patch.set_xy((cx - cart_w / 2, -0.03))

        # Rotor center
        rx = cx
        ry = -0.03 + cart_h + rotor_r
        rotor_circle.center = (rx, ry)

        # Eccentric mass
        ex = rx + e * np.sin(th)
        ey = ry + e * np.cos(th)
        ecc_mass.center = (ex, ey)

        # Line from rotor center to eccentric mass
        rotor_line.set_data([rx, ex], [ry, ey])

        # Trace
        trace_x.append(ex)
        trace_y.append(ey)
        if len(trace_x) > 500:
            trace_x.pop(0)
            trace_y.pop(0)
        trace_line.set_data(trace_x, trace_y)

        time_text.set_text(
            f"t = {ti:.2f} s\n"
            f"x = {cx:.4f} m\n"
            f"θ = {th:.4f} rad\n"
            f"τ = {ui:.5f} N·m"
        )

        return spring_line, cart_patch, rotor_circle, ecc_mass, rotor_line, trace_line, time_text

    anim = FuncAnimation(fig, update, init_func=init,
                         frames=len(x_s), interval=interval, blit=True)

    if save_path:
        anim.save(save_path, writer="pillow", fps=50)

    plt.tight_layout()
    return fig, anim
