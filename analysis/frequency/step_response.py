"""Step response analysis of the closed-loop system."""

import numpy as np
from scipy.linalg import expm


def step_response(A_cl, B, t_end=20.0, dt=0.01):
    """Compute closed-loop step response to unit torque input.

    Returns
    -------
    result : dict  Keys: t, y, overshoot_pct, settling_time.
    """
    N = int(t_end / dt)
    t = np.arange(N + 1) * dt
    n = A_cl.shape[0]

    x = np.zeros(n)
    y = np.zeros(N + 1)
    y[0] = x[0]

    A_dt = expm(A_cl * dt)
    # B_dt via first-order hold approximation
    B_dt = np.linalg.solve(A_cl, (A_dt - np.eye(n)) @ B).flatten()

    for i in range(N):
        x = A_dt @ x + B_dt * 1.0  # unit step input
        y[i + 1] = x[0]  # cart displacement output

    # Steady-state
    y_ss = y[-1]
    if abs(y_ss) < 1e-12:
        y_ss = np.mean(y[-100:]) if N > 100 else y[-1]

    # Overshoot
    if abs(y_ss) > 1e-12:
        overshoot = (np.max(np.abs(y)) - abs(y_ss)) / abs(y_ss) * 100
    else:
        overshoot = 0.0

    # Settling time (2% band)
    threshold = 0.02 * abs(y_ss) if abs(y_ss) > 1e-12 else 0.02
    settled = np.where(np.abs(y - y_ss) > threshold)[0]
    settling_time = t[settled[-1]] if len(settled) > 0 else 0.0

    return {
        "t": t,
        "y": y,
        "overshoot_pct": max(0.0, overshoot),
        "settling_time": settling_time,
        "steady_state": y_ss,
    }
