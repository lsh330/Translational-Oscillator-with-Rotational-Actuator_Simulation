"""Tests for the nonlinear controllers (energy-based, SMC, iLQR)."""

import numpy as np
import pytest

from parameters.config import SystemConfig
from dynamics.forward_dynamics.forward_dynamics_fast import forward_dynamics_fast
from control.energy_based import (
    energy_based_control, energy_lyapunov, default_energy_gains
)
from control.sliding_mode import (
    sliding_mode_control, sliding_surface, default_smc_gains
)
from simulation.integrator.rk4_step import rk4_step_fast


@pytest.fixture
def p():
    return SystemConfig().pack()


def _simulate_short(controller_fn, p, N=5000, dt=0.001, x0=0.1):
    """Helper: run a short simulation and return final state."""
    x, theta, xd, td = x0, 0.0, 0.0, 0.0
    for _ in range(N):
        tau = controller_fn(x, theta, xd, td)
        tau = max(-0.5, min(0.5, tau))
        x, theta, xd, td = rk4_step_fast(x, theta, xd, td, tau, p, dt)
        if np.isnan(x):
            return None
    return np.array([x, theta, xd, td])


class TestEnergyBasedControl:
    def test_converges(self, p):
        """Energy-based controller should reduce amplitude over time."""
        gains = default_energy_gains(p)
        ctrl = lambda x, th, xd, td: energy_based_control(
            x, th, xd, td, p, gains["kp"], gains["kd"], gains["kc"]
        )
        z_final = _simulate_short(ctrl, p, N=50000)
        assert z_final is not None
        # Verify amplitude is decreasing (weaker than exact convergence)
        assert abs(z_final[0]) < 0.1

    def test_energy_decreases_overall(self, p):
        """Total system energy should decrease over the trajectory."""
        gains = default_energy_gains(p)
        x, theta, xd, td = 0.1, 0.0, 0.0, 0.0
        dt = 0.001
        Mt, me, I_eff, k = p[0], p[1], p[2], p[3]

        def total_energy(x, th, xd, td):
            T = 0.5 * (Mt * xd**2 + 2*me*np.cos(th)*xd*td + I_eff*td**2)
            V = 0.5 * k * x**2
            return T + V

        H_initial = total_energy(x, theta, xd, td)
        for _ in range(10000):
            tau = energy_based_control(x, theta, xd, td, p,
                                       gains["kp"], gains["kd"], gains["kc"])
            tau = max(-0.5, min(0.5, tau))
            x, theta, xd, td = rk4_step_fast(x, theta, xd, td, tau, p, dt)
        H_final = total_energy(x, theta, xd, td)

        # Energy should decrease overall (controller dissipates energy)
        assert H_final < H_initial

    def test_no_nan(self, p):
        gains = default_energy_gains(p)
        ctrl = lambda x, th, xd, td: energy_based_control(
            x, th, xd, td, p, gains["kp"], gains["kd"], gains["kc"]
        )
        z_final = _simulate_short(ctrl, p)
        assert z_final is not None


class TestSlidingModeControl:
    def test_converges(self, p):
        """SMC should reduce state amplitude over time."""
        gains = default_smc_gains(p)
        ctrl = lambda x, th, xd, td: sliding_mode_control(
            x, th, xd, td, 0.0, p,
            gains["c1"], gains["c2"], gains["c3"], gains["eta"], gains["phi"]
        )
        z_final = _simulate_short(ctrl, p, N=50000)
        assert z_final is not None
        assert abs(z_final[0]) < 0.1

    def test_sliding_variable_converges(self, p):
        gains = default_smc_gains(p)
        x, theta, xd, td = 0.1, 0.0, 0.0, 0.0
        dt = 0.001

        s_values = []
        for _ in range(10000):
            tau = sliding_mode_control(x, theta, xd, td, 0.0, p,
                                       gains["c1"], gains["c2"], gains["c3"],
                                       gains["eta"], gains["phi"])
            tau = max(-0.5, min(0.5, tau))
            s = sliding_surface(x, theta, xd, td,
                               gains["c1"], gains["c2"], gains["c3"])
            s_values.append(abs(s))
            x, theta, xd, td = rk4_step_fast(x, theta, xd, td, tau, p, dt)

        # s should decrease toward zero
        assert np.mean(s_values[-1000:]) < np.mean(s_values[:1000])

    def test_no_nan(self, p):
        gains = default_smc_gains(p)
        ctrl = lambda x, th, xd, td: sliding_mode_control(
            x, th, xd, td, 0.0, p,
            gains["c1"], gains["c2"], gains["c3"], gains["eta"], gains["phi"]
        )
        z_final = _simulate_short(ctrl, p)
        assert z_final is not None
