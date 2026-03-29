"""Batch experiment runner for parameter sweeps and controller comparison."""

import numpy as np
from itertools import product

from parameters.config import SystemConfig
from control.lqr import compute_lqr
from simulation.loop.time_loop import simulate
from utils.logger import get_logger

_log = get_logger("tora.experiment")


def run_batch(cfg_overrides=None, controllers=None, x0_values=None,
              tau_max_values=None, t_end=20.0, dt=0.001, seed=42):
    """Run batch experiments over parameter/controller combinations.

    Parameters
    ----------
    cfg_overrides : list of dict or None  Parameter overrides for each config.
    controllers   : list of str  Controller types to test.
    x0_values     : list of float  Initial displacements to sweep.
    tau_max_values: list of float  Torque limits to sweep.
    t_end, dt     : float  Simulation parameters.
    seed          : int  Random seed.

    Returns
    -------
    results : list of dict  Each entry has 'config', 'controller', 'x0',
                              'tau_max', 'metrics', 'sim_result'.
    """
    if cfg_overrides is None:
        cfg_overrides = [{}]
    if controllers is None:
        controllers = ["lqr"]
    if x0_values is None:
        x0_values = [0.1]
    if tau_max_values is None:
        tau_max_values = [0.1]

    results = []
    total = len(cfg_overrides) * len(controllers) * len(x0_values) * len(tau_max_values)
    count = 0

    for cfg_ov in cfg_overrides:
        cfg_params = {"M": 1.3608, "m": 0.096, "e": 0.0592, "k": 186.3, "I": 0.0002175}
        cfg_params.update(cfg_ov)
        cfg = SystemConfig(**cfg_params)
        p = cfg.pack()

        for tau_max in tau_max_values:
            # Compute LQR for this config
            lqr_result = compute_lqr(p, tau_max=tau_max)
            K = lqr_result["K"]

            for ctrl in controllers:
                for x0 in x0_values:
                    count += 1
                    _log.info("Batch %d/%d: %s, x0=%.3f, tau_max=%.3f",
                              count, total, ctrl, x0, tau_max)

                    sim = simulate(cfg, controller_type=ctrl, K=K,
                                   t_end=t_end, dt=dt, x0=x0,
                                   tau_max=tau_max, dist_amplitude=0.0, seed=seed)

                    # Compute basic metrics
                    x_final = abs(sim["x"][-1])
                    theta_final = abs(sim["theta"][-1])
                    effort = np.sum(sim["u"]**2) * dt

                    results.append({
                        "config": cfg_params,
                        "controller": ctrl,
                        "x0": x0,
                        "tau_max": tau_max,
                        "x_final": x_final,
                        "theta_final": theta_final,
                        "control_effort": effort,
                        "sat_count": sim["sat_count"],
                        "sim_result": sim,
                    })

    _log.info("Batch complete: %d experiments", len(results))
    return results


def parameter_sweep(param_name, param_values, controller="lqr",
                    t_end=20.0, dt=0.001, x0=0.1, tau_max=0.1):
    """Sweep a single parameter and collect results.

    Parameters
    ----------
    param_name   : str  One of "M", "m", "e", "k", "I".
    param_values : array-like  Values to sweep.

    Returns
    -------
    results : dict  Keys: param_values, x_final, theta_final, effort, settling.
    """
    x_finals = []
    efforts = []

    for val in param_values:
        cfg_ov = {param_name: val}
        batch = run_batch(cfg_overrides=[cfg_ov], controllers=[controller],
                          x0_values=[x0], tau_max_values=[tau_max],
                          t_end=t_end, dt=dt)
        x_finals.append(batch[0]["x_final"])
        efforts.append(batch[0]["control_effort"])

    return {
        "param_name": param_name,
        "param_values": np.array(param_values),
        "x_final": np.array(x_finals),
        "control_effort": np.array(efforts),
    }
