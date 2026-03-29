"""Tests for v2.0 extension modules: observer, LQI, hybrid, friction, actuator."""

import numpy as np
import pytest

from parameters.config import SystemConfig


@pytest.fixture
def cfg():
    return SystemConfig()


@pytest.fixture
def p(cfg):
    return cfg.pack()


class TestObserver:
    def test_import(self):
        """Observer module must import without error."""
        from control.observer import design_observer, observer_update

    def test_design_observer(self, p):
        """Observer design should produce valid gain matrix."""
        from control.observer import design_observer
        result = design_observer(p)
        assert result["L"].shape == (4, 2)
        # Observer poles should be in LHP (stable)
        assert np.all(result["observer_poles"].real < 0)

    def test_observer_update(self, p):
        """Observer update should produce valid state estimate."""
        from control.observer import design_observer, observer_update
        result = design_observer(p)

        L_flat = result["L"].flatten()
        A_flat = result["A"].flatten()
        B_flat = np.zeros(4)  # Will get from linearization
        from control.linearization.linearize import linearize
        _, B = linearize(p)
        B_flat = B.flatten()
        C_flat = result["C"].flatten()

        z_hat = np.zeros(4)
        y = np.array([0.1, 0.0])  # measured x, theta
        z_new = observer_update(z_hat, y, 0.0, A_flat, B_flat, L_flat, C_flat, 0.001)
        assert z_new.shape == (4,)
        assert not np.any(np.isnan(z_new))


class TestLQI:
    def test_import(self):
        from control.lqi import compute_lqi

    def test_lqi_design(self, p):
        from control.lqi import compute_lqi
        result = compute_lqi(p)
        assert result["K_aug"].shape == (1, 5)
        # All poles in LHP
        assert np.all(result["poles_cl"].real < 0)


class TestHybridEnergy:
    def test_import(self):
        from control.hybrid_energy import hybrid_energy_control

    def test_mode_switching(self, p):
        """Should switch modes based on state magnitude."""
        from control.hybrid_energy import hybrid_energy_control
        K_flat = np.array([0.8, 0.1, 0.05, 0.01])

        # Large state -> mode 0 (energy pumping)
        tau, mode = hybrid_energy_control(0.1, 0.0, 0.0, 0.0, p, K_flat, 0.5, 0.1, 0.01)
        assert mode == 0

        # Near zero -> mode 2 (fine regulation)
        tau, mode = hybrid_energy_control(0.001, 0.001, 0.001, 0.001, p, K_flat, 0.5, 0.1, 0.01)
        assert mode == 2

    def test_no_nan(self, p):
        from control.hybrid_energy import hybrid_energy_control
        K_flat = np.array([0.8, 0.1, 0.05, 0.01])
        tau, mode = hybrid_energy_control(0.1, 0.5, 0.3, -1.0, p, K_flat, 0.5, 0.1, 0.01)
        assert not np.isnan(tau)


class TestCoulombFriction:
    def test_import(self):
        from dynamics.friction.coulomb import coulomb_friction

    def test_zero_velocity(self):
        from dynamics.friction.coulomb import coulomb_friction
        f = coulomb_friction(0.0, 1.0)
        assert abs(f) < 1e-10

    def test_positive_velocity(self):
        from dynamics.friction.coulomb import coulomb_friction
        f = coulomb_friction(1.0, 0.5)
        assert f > 0  # opposes motion

    def test_negative_velocity(self):
        from dynamics.friction.coulomb import coulomb_friction
        f = coulomb_friction(-1.0, 0.5)
        assert f < 0


class TestActuatorLag:
    def test_import(self):
        from simulation.actuator.first_order_lag import actuator_lag_step

    def test_ideal_passthrough(self):
        """T_a = 0 should give instant response."""
        from simulation.actuator.first_order_lag import actuator_lag_step
        tau_next = actuator_lag_step(0.0, 1.0, 0.0, 0.001)
        assert tau_next == 1.0

    def test_lag_convergence(self):
        """With T_a > 0, output should converge to command."""
        from simulation.actuator.first_order_lag import actuator_lag_step
        tau = 0.0
        for _ in range(10000):
            tau = actuator_lag_step(tau, 1.0, 0.01, 0.001)
        assert abs(tau - 1.0) < 0.01


class TestAngleUtils:
    def test_wrap_to_pi(self):
        from utils.angle import wrap_to_pi
        assert abs(wrap_to_pi(0.0)) < 1e-10
        assert abs(wrap_to_pi(2 * np.pi)) < 1e-10
        assert abs(wrap_to_pi(np.pi) - np.pi) < 1e-10 or abs(wrap_to_pi(np.pi) + np.pi) < 1e-10

    def test_angle_error(self):
        from utils.angle import angle_error
        err = angle_error(0.1, 0.0)
        assert abs(err - 0.1) < 1e-10


class TestModelPresets:
    def test_ideal_benchmark(self):
        from parameters.model_presets import ideal_benchmark
        cfg = ideal_benchmark()
        p = cfg.pack()
        assert p[4] == 0.0  # c_x
        assert p[5] == 0.0  # c_theta

    def test_engineering_model(self):
        from parameters.model_presets import engineering_model
        cfg = engineering_model("moderate")
        p = cfg.pack()
        assert p[4] > 0.0  # c_x > 0
        assert p[5] > 0.0  # c_theta > 0
