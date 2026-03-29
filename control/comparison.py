"""Compare all 4 controllers on the same initial condition.

Runs each controller through a time-domain simulation and collects
performance metrics for side-by-side comparison.
"""

import numpy as np

from dynamics.forward_dynamics.forward_dynamics_fast import forward_dynamics_fast
from control.energy_based import default_energy_gains, energy_based_control
from control.sliding_mode import default_smc_gains, sliding_mode_control
from utils.logger import get_logger

_log = get_logger("tora.comparison")


def _simulate_controller(controller_fn, z0, p, dt, N, tau_max):
    """Run a generic simulation loop for comparison.

    Note: This uses a simplified RK4 loop without disturbance injection,
    JIT compilation, or saturation tracking. For exact condition matching
    with the main pipeline, use simulation.loop.time_loop.simulate() instead.
    Results are suitable for relative controller comparison but may differ
    slightly from main pipeline results in absolute values.
    """
    z_hist = np.zeros((N + 1, 4))
    u_hist = np.zeros(N)
    z_hist[0] = z0

    for i in range(N):
        x, theta, xd, td = z_hist[i]
        tau = controller_fn(x, theta, xd, td)
        tau = np.clip(tau, -tau_max, tau_max)
        u_hist[i] = tau

        # RK4 step
        def step(x_, th_, xd_, td_, tau_):
            ddx, ddth = forward_dynamics_fast(x_, th_, xd_, td_, tau_, p)
            return xd_, td_, ddx, ddth

        k1 = step(x, theta, xd, td, tau)
        x2 = x + 0.5 * dt * k1[0]
        th2 = theta + 0.5 * dt * k1[1]
        xd2 = xd + 0.5 * dt * k1[2]
        td2 = td + 0.5 * dt * k1[3]

        k2 = step(x2, th2, xd2, td2, tau)
        x3 = x + 0.5 * dt * k2[0]
        th3 = theta + 0.5 * dt * k2[1]
        xd3 = xd + 0.5 * dt * k2[2]
        td3 = td + 0.5 * dt * k2[3]

        k3 = step(x3, th3, xd3, td3, tau)
        x4 = x + dt * k3[0]
        th4 = theta + dt * k3[1]
        xd4 = xd + dt * k3[2]
        td4 = td + dt * k3[3]

        k4 = step(x4, th4, xd4, td4, tau)

        z_hist[i + 1, 0] = x + (dt / 6) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        z_hist[i + 1, 1] = theta + (dt / 6) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        z_hist[i + 1, 2] = xd + (dt / 6) * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
        z_hist[i + 1, 3] = td + (dt / 6) * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3])

    return z_hist, u_hist


def _compute_metrics(z_hist, u_hist, dt, tau_max=0.1):
    """Compute comprehensive performance metrics."""
    t = np.arange(len(z_hist)) * dt
    x = z_hist[:, 0]
    theta = z_hist[:, 1]
    N = len(u_hist)

    x0 = abs(x[0]) if abs(x[0]) > 1e-10 else 0.1

    # Settling time (2% band)
    threshold = 0.02 * x0
    settled = np.where(np.abs(x) > threshold)[0]
    settling_time = t[settled[-1]] if len(settled) > 0 else 0.0

    # Overshoot
    overshoot = max(0.0, (np.max(np.abs(x)) - x0) / x0 * 100) if x0 > 1e-10 else 0.0

    # Integral costs
    state_integral_x = np.sum(x[:N] ** 2) * dt
    state_integral_theta = np.sum(theta[:N] ** 2) * dt
    control_effort = np.sum(u_hist ** 2) * dt

    # Peak values
    peak_theta = np.max(np.abs(theta))
    peak_theta_dot = np.max(np.abs(z_hist[:, 3]))

    # Saturation fraction (approximate — check if near limits)
    u_max_observed = np.max(np.abs(u_hist))

    # Saturation fraction (count near tau_max)
    sat_count = np.sum(np.abs(u_hist) >= tau_max * 0.99)
    sat_fraction = sat_count / N if N > 0 else 0.0

    # Final state norm
    final_norm = np.linalg.norm(z_hist[-1])

    return {
        "settling_time": settling_time,
        "overshoot_pct": overshoot,
        "control_effort": control_effort,
        "final_state_norm": final_norm,
        "integral_x2": state_integral_x,
        "integral_theta2": state_integral_theta,
        "peak_theta": peak_theta,
        "peak_theta_dot": peak_theta_dot,
        "peak_torque": u_max_observed,
        "sat_fraction": sat_fraction,
    }


def compare_controllers(p, K_lqr, ilqr_result=None,
                        t_end=20.0, dt=0.001, x0=0.1, tau_max=0.1):
    """Run all controllers and collect results.

    Parameters
    ----------
    p           : float64[4]  Packed parameters.
    K_lqr       : (1,4)       LQR gain matrix.
    ilqr_result : dict or None  iLQR result from ilqr().
    t_end, dt   : float  Simulation duration and timestep.
    x0          : float  Initial cart displacement.
    tau_max     : float  Torque saturation.

    Returns
    -------
    results : dict  Per-controller time histories and metrics.
    """
    z0 = np.array([x0, 0.0, 0.0, 0.0])
    N = int(t_end / dt)
    t = np.arange(N + 1) * dt

    results = {"t": t}

    # 1. LQR
    K_flat = K_lqr.flatten()

    def lqr_ctrl(x, theta, xd, td):
        return -(K_flat[0] * x + K_flat[1] * theta + K_flat[2] * xd + K_flat[3] * td)

    z_lqr, u_lqr = _simulate_controller(lqr_ctrl, z0, p, dt, N, tau_max)
    results["lqr"] = {"z": z_lqr, "u": u_lqr, "metrics": _compute_metrics(z_lqr, u_lqr, dt, tau_max)}

    # 2. Energy-based
    eg = default_energy_gains(p)

    def energy_ctrl(x, theta, xd, td):
        return energy_based_control(x, theta, xd, td, p, eg["kp"], eg["kd"], eg["kc"])

    z_eb, u_eb = _simulate_controller(energy_ctrl, z0, p, dt, N, tau_max)
    results["energy"] = {"z": z_eb, "u": u_eb, "metrics": _compute_metrics(z_eb, u_eb, dt, tau_max)}

    # 3. SMC
    sg = default_smc_gains(p)

    def smc_ctrl(x, theta, xd, td):
        return sliding_mode_control(
            x, theta, xd, td, p,
            sg["c1"], sg["c2"], sg["c3"], sg["eta"], sg["phi"]
        )

    z_smc, u_smc = _simulate_controller(smc_ctrl, z0, p, dt, N, tau_max)
    results["smc"] = {"z": z_smc, "u": u_smc, "metrics": _compute_metrics(z_smc, u_smc, dt, tau_max)}

    # 4. Hybrid Energy
    from control.hybrid_energy import hybrid_energy_control
    from control.energy_based import default_energy_gains as _deg
    _eg = _deg(p)
    K_flat_h = K_lqr.flatten()

    def hybrid_ctrl(x, theta, xd, td):
        tau, mode = hybrid_energy_control(x, theta, xd, td, p, K_flat_h,
                                           _eg["kp"], _eg["kd"], _eg["kc"])
        return tau

    z_hyb, u_hyb = _simulate_controller(hybrid_ctrl, z0, p, dt, N, tau_max)
    results["hybrid"] = {"z": z_hyb, "u": u_hyb, "metrics": _compute_metrics(z_hyb, u_hyb, dt, tau_max)}

    # 5. LQI (needs integral state — use specialized loop)
    from control.lqi import compute_lqi
    try:
        lqi_res = compute_lqi(p, tau_max=tau_max)
        K_aug_flat = lqi_res["K_aug"].flatten()

        z_lqi = np.zeros((N + 1, 4))
        u_lqi = np.zeros(N)
        z_lqi[0] = z0
        int_x = 0.0
        from simulation.integrator.rk4_step import rk4_step_fast
        for i in range(N):
            x, theta, xd, td = z_lqi[i]
            tau = -(K_aug_flat[0]*x + K_aug_flat[1]*theta + K_aug_flat[2]*xd + K_aug_flat[3]*td + K_aug_flat[4]*int_x)
            tau = np.clip(tau, -tau_max, tau_max)
            u_lqi[i] = tau
            nx, nth, nxd, ntd = rk4_step_fast(x, theta, xd, td, tau, p, dt)
            z_lqi[i+1] = [nx, nth, nxd, ntd]
            int_x += x * dt
        results["lqi"] = {"z": z_lqi, "u": u_lqi, "metrics": _compute_metrics(z_lqi, u_lqi, dt, tau_max)}
    except Exception as e:
        _log.warning("LQI comparison failed: %s", e)

    # 6. iLQR (if available)
    if ilqr_result is not None:
        results["ilqr"] = {
            "z": ilqr_result["z_traj"],
            "u": ilqr_result["u_traj"],
            "metrics": _compute_metrics(
                ilqr_result["z_traj"], ilqr_result["u_traj"], dt, tau_max
            ),
        }

    # Summary
    _log.info("=== Controller Comparison ===")
    for name in ["lqr", "energy", "smc", "hybrid", "lqi", "ilqr"]:
        if name in results and name != "t":
            m = results[name]["metrics"]
            _log.info("  %-8s  settle=%.2fs  effort=%.4f  peak_θ=%.3f  final=%.2e",
                      name.upper(), m["settling_time"], m["control_effort"],
                      m["peak_theta"], m["final_state_norm"])

    return results
