"""Tests for the simulation engine."""

import numpy as np
import pytest

from parameters.config import SystemConfig
from control.lqr import compute_lqr
from simulation.loop.time_loop import simulate
from simulation.disturbance.generate_disturbance import generate_disturbance
from simulation.initial_conditions.displacement_ic import initial_displacement


@pytest.fixture
def cfg():
    return SystemConfig()


@pytest.fixture
def p(cfg):
    return cfg.pack()


@pytest.fixture
def K(p):
    return compute_lqr(p)["K"]


class TestSimulationOutput:
    def test_output_shapes(self, cfg, K):
        result = simulate(cfg, "lqr", K=K, t_end=1.0, dt=0.001)
        N = 1000
        assert len(result["t"]) == N + 1
        assert len(result["x"]) == N + 1
        assert len(result["theta"]) == N + 1
        assert len(result["u"]) == N

    def test_initial_condition(self, cfg, K):
        result = simulate(cfg, "lqr", K=K, t_end=0.5, x0=0.1)
        assert result["x"][0] == pytest.approx(0.1)
        assert result["theta"][0] == pytest.approx(0.0)


class TestLQRStability:
    def test_lqr_stabilizes(self, cfg, K):
        result = simulate(cfg, "lqr", K=K, t_end=20.0, dt=0.001,
                         x0=0.1, tau_max=0.5, dist_amplitude=0.0)
        # TORA has weak coupling so convergence is slow
        assert abs(result["x"][-1]) < 0.05
        assert abs(result["theta"][-1]) < 0.5

    def test_no_nan(self, cfg, K):
        result = simulate(cfg, "lqr", K=K, t_end=5.0)
        assert not np.any(np.isnan(result["x"]))
        assert not np.any(np.isnan(result["theta"]))


class TestSaturation:
    def test_torque_bounded(self, cfg, K):
        tau_max = 0.05
        result = simulate(cfg, "lqr", K=K, t_end=2.0, tau_max=tau_max)
        assert np.all(np.abs(result["u"]) <= tau_max + 1e-10)


class TestDisturbance:
    def test_disturbance_shape(self):
        d = generate_disturbance(1000, 0.001, amplitude=0.01)
        assert d.shape == (1000,)

    def test_disturbance_rms(self):
        d = generate_disturbance(10000, 0.001, amplitude=0.01)
        rms = np.sqrt(np.mean(d ** 2))
        assert rms == pytest.approx(0.01, rel=0.1)


class TestInitialConditions:
    def test_default_ic(self):
        z0 = initial_displacement()
        np.testing.assert_allclose(z0, [0.1, 0.0, 0.0, 0.0])

    def test_custom_ic(self):
        z0 = initial_displacement(0.05)
        assert z0[0] == 0.05


class TestEnergyController:
    def test_energy_simulation_runs(self, cfg):
        result = simulate(cfg, "energy", t_end=2.0, dist_amplitude=0.0)
        assert not np.any(np.isnan(result["x"]))


class TestSMCController:
    def test_smc_simulation_runs(self, cfg):
        result = simulate(cfg, "smc", t_end=2.0, dist_amplitude=0.0)
        assert not np.any(np.isnan(result["x"]))
