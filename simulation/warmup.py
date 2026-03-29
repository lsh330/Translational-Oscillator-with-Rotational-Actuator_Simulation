"""Pre-trigger all Numba JIT compilations at startup.

Call warmup() once before the main simulation to avoid JIT overhead
during timed execution.
"""

import numpy as np

from utils.logger import get_logger

_log = get_logger("tora.warmup")


def warmup():
    """Trigger JIT compilation for all hot-path functions."""
    _log.info("JIT warmup starting...")

    p = np.array([1.4568, 0.005683, 0.000554, 186.3])

    # Forward dynamics (array)
    from dynamics.forward_dynamics.forward_dynamics import forward_dynamics
    q = np.zeros(2)
    dq = np.zeros(2)
    forward_dynamics(q, dq, 0.0, p)

    # Forward dynamics (scalar)
    from dynamics.forward_dynamics.forward_dynamics_fast import forward_dynamics_fast
    forward_dynamics_fast(0.0, 0.0, 0.0, 0.0, 0.0, p)

    # RK4 steps
    from simulation.integrator.rk4_step import rk4_step, rk4_step_fast
    from simulation.integrator.state_derivative import state_derivative
    z = np.zeros(4)
    state_derivative(z, 0.0, p)
    rk4_step(z, 0.0, p, 0.001)
    rk4_step_fast(0.0, 0.0, 0.0, 0.0, 0.0, p, 0.001)

    # Control laws
    from simulation.loop.control_law import lqr_control
    K_flat = np.zeros(4)
    lqr_control(0.0, 0.0, 0.0, 0.0, K_flat)

    from control.energy_based import energy_based_control
    energy_based_control(0.0, 0.0, 0.0, 0.0, p, 0.5, 0.1, 0.01)

    from control.sliding_mode import sliding_mode_control
    sliding_mode_control(0.0, 0.0, 0.0, 0.0, p, 1.0, 5.0, 1.5, 0.05, 0.01)

    # Mass matrix, coriolis, spring
    from dynamics.mass_matrix.assembly import mass_matrix
    from dynamics.coriolis.coriolis_vector import coriolis_vector
    from dynamics.spring.spring_force import spring_force
    mass_matrix(0.0, p)
    coriolis_vector(0.0, 0.0, p)
    spring_force(0.0, p)

    # Simulation loops (1-step warmup)
    from simulation.loop.time_loop_fast import (
        _run_loop_lqr, _run_loop_energy, _run_loop_smc
    )
    d = np.zeros(1)
    _run_loop_lqr(1, 0.001, 0.0, 0.0, 0.0, 0.0, K_flat, p, d, 1.0)
    _run_loop_energy(1, 0.001, 0.0, 0.0, 0.0, 0.0, p, 0.5, 0.1, 0.01, d, 1.0)
    _run_loop_smc(1, 0.001, 0.0, 0.0, 0.0, 0.0, p, 1.0, 5.0, 1.5, 0.05, 0.01, d, 1.0)

    _log.info("JIT warmup complete.")
