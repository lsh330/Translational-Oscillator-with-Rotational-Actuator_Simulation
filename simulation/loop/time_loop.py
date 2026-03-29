"""High-level simulation wrapper that dispatches to JIT loops."""

import numpy as np

from simulation.loop.time_loop_fast import (
    _run_loop_lqr,
    _run_loop_energy,
    _run_loop_smc,
)
from simulation.disturbance.generate_disturbance import generate_disturbance
from simulation.initial_conditions.displacement_ic import initial_displacement
from control.energy_based import default_energy_gains
from control.sliding_mode import default_smc_gains
from utils.logger import get_logger

_log = get_logger("tora.sim")


def simulate(cfg, controller_type="lqr", K=None,
             t_end=20.0, dt=0.001, x0=0.1, theta0=0.0,
             x_dot0=0.0, theta_dot0=0.0, tau_max=0.1,
             dist_amplitude=0.01, dist_bandwidth=5.0, seed=42,
             dist_channel="torque"):
    """Run a TORA simulation.

    Note: The main simulation uses RK4 via @njit fast loops for performance.
    Alternative integrators (RK45, Störmer-Verlet) are available via the
    simulation.integrator module for custom analysis code, but are not
    selectable for the main pipeline simulation.

    Parameters
    ----------
    cfg             : SystemConfig  System configuration.
    controller_type : str  One of "lqr", "lqi", "energy", "hybrid", "smc".
    K               : (1,4) ndarray  LQR gain (required for "lqr").
    t_end, dt       : float  Simulation duration and timestep.
    x0              : float  Initial cart displacement [m].
    tau_max         : float  Torque saturation [N*m].
    dist_amplitude  : float  Disturbance RMS [N*m].
    dist_bandwidth  : float  Disturbance cutoff [Hz].
    seed            : int  Random seed.
    dist_channel    : str  Disturbance injection point. Currently "torque" only.
        "torque" — additive torque disturbance on rotor (default, benchmark standard)
        Future: "cart" for external cart force, "both" for combined.

    Returns
    -------
    result : dict  Time histories and metadata.
    """

    p = cfg.pack()
    N = int(t_end / dt)
    t = np.arange(N + 1) * dt

    z0 = np.array([x0, theta0, x_dot0, theta_dot0])

    # Generate disturbance
    dist = generate_disturbance(N, dt, dist_amplitude, dist_bandwidth, seed)

    _log.info("Simulating %s controller: t_end=%.1fs, dt=%.4fs, x0=%.3f, tau_max=%.3f",
              controller_type.upper(), t_end, dt, x0, tau_max)

    if controller_type == "lqr":
        if K is None:
            raise ValueError("LQR simulation requires gain matrix K")
        K_flat = K.flatten().astype(np.float64)
        x_h, th_h, xd_h, td_h, u_h, u_raw, sat = _run_loop_lqr(
            N, dt, z0[0], z0[1], z0[2], z0[3], K_flat, p, dist, tau_max
        )
        result = {
            "t": t, "x": x_h, "theta": th_h, "x_dot": xd_h, "theta_dot": td_h,
            "u": u_h, "u_raw": u_raw, "disturbance": dist,
            "sat_count": sat, "controller": "lqr",
        }

    elif controller_type == "energy":
        gains = default_energy_gains(p)
        x_h, th_h, xd_h, td_h, u_h, sat = _run_loop_energy(
            N, dt, z0[0], z0[1], z0[2], z0[3], p,
            gains["kp"], gains["kd"], gains["kc"], dist, tau_max
        )
        result = {
            "t": t, "x": x_h, "theta": th_h, "x_dot": xd_h, "theta_dot": td_h,
            "u": u_h, "disturbance": dist,
            "sat_count": sat, "controller": "energy",
        }

    elif controller_type == "smc":
        gains = default_smc_gains(p)
        x_h, th_h, xd_h, td_h, u_h, s_h, sat = _run_loop_smc(
            N, dt, z0[0], z0[1], z0[2], z0[3], p,
            gains["c1"], gains["c2"], gains["c3"],
            gains["eta"], gains["phi"], dist, tau_max
        )
        result = {
            "t": t, "x": x_h, "theta": th_h, "x_dot": xd_h, "theta_dot": td_h,
            "u": u_h, "s": s_h, "disturbance": dist,
            "sat_count": sat, "controller": "smc",
        }
    elif controller_type == "lqi":
        from control.lqi import compute_lqi
        lqi_result = compute_lqi(p, tau_max=tau_max)
        K_aug = lqi_result["K_aug"].flatten()

        # Run manually since LQI needs integral state
        x_h = np.empty(N + 1)
        th_h = np.empty(N + 1)
        xd_h = np.empty(N + 1)
        td_h = np.empty(N + 1)
        u_h = np.empty(N)

        x, theta, xd, td = z0
        int_x = 0.0  # integral of x
        x_h[0], th_h[0], xd_h[0], td_h[0] = x, theta, xd, td
        sat = 0

        from simulation.integrator.rk4_step import rk4_step_fast
        for i in range(N):
            # Augmented state feedback
            tau = -(K_aug[0]*x + K_aug[1]*theta + K_aug[2]*xd + K_aug[3]*td + K_aug[4]*int_x)
            tau += dist[i]

            if tau > tau_max:
                tau = tau_max; sat += 1
            elif tau < -tau_max:
                tau = -tau_max; sat += 1
            u_h[i] = tau

            x, theta, xd, td = rk4_step_fast(x, theta, xd, td, tau, p, dt)
            int_x += x * dt  # Euler integration of integral state

            x_h[i+1], th_h[i+1], xd_h[i+1], td_h[i+1] = x, theta, xd, td

        result = {
            "t": t, "x": x_h, "theta": th_h, "x_dot": xd_h, "theta_dot": td_h,
            "u": u_h, "disturbance": dist,
            "sat_count": sat, "controller": "lqi",
        }

    elif controller_type == "hybrid":
        from control.hybrid_energy import hybrid_energy_control
        from control.energy_based import default_energy_gains as _deg_hybrid
        from control.lqr import compute_lqr as _compute_lqr

        eg = _deg_hybrid(p)
        # Need LQR gain for fine regulation phase
        if K is None:
            lqr_res = _compute_lqr(p, tau_max=tau_max)
            K_flat = lqr_res["K"].flatten()
        else:
            K_flat = K.flatten()

        x_h = np.empty(N + 1)
        th_h = np.empty(N + 1)
        xd_h = np.empty(N + 1)
        td_h = np.empty(N + 1)
        u_h = np.empty(N)

        x, theta, xd, td = z0
        x_h[0], th_h[0], xd_h[0], td_h[0] = x, theta, xd, td
        sat = 0

        from simulation.integrator.rk4_step import rk4_step_fast
        for i in range(N):
            tau, mode = hybrid_energy_control(
                x, theta, xd, td, p, K_flat,
                eg["kp"], eg["kd"], eg["kc"])
            tau += dist[i]

            if tau > tau_max:
                tau = tau_max; sat += 1
            elif tau < -tau_max:
                tau = -tau_max; sat += 1
            u_h[i] = tau

            x, theta, xd, td = rk4_step_fast(x, theta, xd, td, tau, p, dt)
            x_h[i+1], th_h[i+1], xd_h[i+1], td_h[i+1] = x, theta, xd, td

        result = {
            "t": t, "x": x_h, "theta": th_h, "x_dot": xd_h, "theta_dot": td_h,
            "u": u_h, "disturbance": dist,
            "sat_count": sat, "controller": "hybrid",
        }

    else:
        raise ValueError(f"Unknown controller type: {controller_type}")

    _log.info("Simulation complete: %d steps, %d saturations (%.1f%%)",
              N, result["sat_count"], 100.0 * result["sat_count"] / N)

    return result
