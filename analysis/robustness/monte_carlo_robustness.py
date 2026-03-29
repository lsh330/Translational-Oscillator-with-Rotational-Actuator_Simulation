"""Monte Carlo robustness analysis (standalone module)."""

import numpy as np

from simulation.integrator.rk4_step import rk4_step_fast
from utils.logger import get_logger

_log = get_logger("tora.mc")


def monte_carlo_robustness(K, N_trials=300, t_horizon=5.0, dt=0.002,
                           x0=0.1, seed=42):
    """Test LQR robustness under random parameter perturbations.

    Perturbation ranges:
        M:  ±10%
        m:  ±10%
        e:  ±5%
        k:  ±10%
        I:  ±10%

    Returns
    -------
    result : dict  Keys: success_rate, settling_times, max_states.
    """
    rng = np.random.default_rng(seed)
    K_flat = K.flatten()
    N_steps = int(t_horizon / dt)

    successes = 0
    settling_times = []
    max_states = []

    # Memory-efficient: only final state checked, no full trajectory stored.
    # For N=51x51=2601 trials at dt=0.002 for 10s (5000 steps),
    # this uses O(1) memory per trial instead of O(5000*4) per trial.
    for _ in range(N_trials):
        M_p = 1.3608 * (1.0 + 0.1 * rng.uniform(-1, 1))
        m_p = 0.096 * (1.0 + 0.1 * rng.uniform(-1, 1))
        e_p = 0.0592 * (1.0 + 0.05 * rng.uniform(-1, 1))
        k_p = 186.3 * (1.0 + 0.1 * rng.uniform(-1, 1))
        I_p = 0.0002175 * (1.0 + 0.1 * rng.uniform(-1, 1))

        p = np.array([M_p + m_p, m_p * e_p, I_p + m_p * e_p ** 2, k_p])

        x, theta, xd, td = x0, 0.0, 0.0, 0.0
        max_state = 0.0
        settle_step = N_steps
        diverged = False

        for step in range(N_steps):
            tau = -(K_flat[0] * x + K_flat[1] * theta
                    + K_flat[2] * xd + K_flat[3] * td)
            tau = max(-0.5, min(0.5, tau))

            x, theta, xd, td = rk4_step_fast(x, theta, xd, td, tau, p, dt)

            norm = abs(x) + abs(theta) + abs(xd) + abs(td)
            if norm > max_state:
                max_state = norm
            if np.isnan(x) or norm > 10.0:
                diverged = True
                break

            if norm < 0.01 and settle_step == N_steps:
                settle_step = step

        if not diverged and (abs(x) + abs(theta)) < 0.05:
            successes += 1
            settling_times.append(settle_step * dt)
        max_states.append(max_state)

    rate = successes / N_trials
    _log.info("MC robustness: %.1f%% (%d/%d)", 100 * rate, successes, N_trials)

    return {
        "success_rate": rate,
        "N_trials": N_trials,
        "settling_times": np.array(settling_times),
        "max_states": np.array(max_states),
    }
