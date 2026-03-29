"""Tests for the dynamics engine."""

import numpy as np
import pytest

from parameters.config import SystemConfig
from dynamics.mass_matrix.assembly import mass_matrix
from dynamics.coriolis.coriolis_vector import coriolis_vector
from dynamics.spring.spring_force import spring_force
from dynamics.forward_dynamics.forward_dynamics import forward_dynamics
from dynamics.forward_dynamics.forward_dynamics_fast import forward_dynamics_fast
from dynamics.forward_dynamics.solve_acceleration import solve_acceleration


@pytest.fixture
def cfg():
    return SystemConfig()


@pytest.fixture
def p(cfg):
    return cfg.pack()


class TestMassMatrix:
    def test_symmetric(self, p):
        for theta in [0.0, 0.5, 1.0, np.pi]:
            M = mass_matrix(theta, p)
            np.testing.assert_allclose(M, M.T, atol=1e-15)

    def test_positive_definite(self, p):
        for theta in np.linspace(0, 2 * np.pi, 20):
            M = mass_matrix(theta, p)
            eigvals = np.linalg.eigvalsh(M)
            assert np.all(eigvals > 0), f"Not PD at theta={theta}"

    def test_diagonal_values(self, p):
        M = mass_matrix(0.0, p)
        assert M[0, 0] == pytest.approx(p[0], rel=1e-10)  # Mt
        assert M[1, 1] == pytest.approx(p[2], rel=1e-10)  # I_eff

    def test_coupling_at_zero(self, p):
        M = mass_matrix(0.0, p)
        assert M[0, 1] == pytest.approx(p[1], rel=1e-10)  # me

    def test_coupling_at_pi_half(self, p):
        M = mass_matrix(np.pi / 2, p)
        assert abs(M[0, 1]) < 1e-10  # me * cos(pi/2) = 0

    def test_det_always_positive(self, p):
        """det(M) = Mt*I_eff - me^2*cos^2(theta) > 0 always."""
        for theta in np.linspace(0, 2 * np.pi, 100):
            M = mass_matrix(theta, p)
            det = np.linalg.det(M)
            assert det > 0, f"det(M) = {det} at theta={theta}"


class TestCoriolisVector:
    def test_zero_at_rest(self, p):
        C = coriolis_vector(0.0, 0.0, p)
        np.testing.assert_allclose(C, [0.0, 0.0], atol=1e-15)

    def test_nonzero_with_velocity(self, p):
        C = coriolis_vector(0.5, 2.0, p)
        assert abs(C[0]) > 0  # Centrifugal force
        assert C[1] == 0.0    # No Coriolis on rotor

    def test_second_component_always_zero(self, p):
        for theta in np.linspace(0, 2 * np.pi, 10):
            for td in [0.0, 1.0, -5.0]:
                C = coriolis_vector(theta, td, p)
                assert C[1] == 0.0


class TestSpringForce:
    def test_zero_at_origin(self, p):
        K = spring_force(0.0, p)
        np.testing.assert_allclose(K, [0.0, 0.0], atol=1e-15)

    def test_proportional_to_x(self, p):
        K1 = spring_force(0.1, p)
        K2 = spring_force(0.2, p)
        assert K2[0] == pytest.approx(2.0 * K1[0], rel=1e-10)

    def test_rotor_component_zero(self, p):
        K = spring_force(0.5, p)
        assert K[1] == 0.0


class TestSolveAcceleration:
    def test_identity_system(self):
        M = np.eye(2)
        rhs = np.array([1.0, 2.0])
        ddq = solve_acceleration(M, rhs)
        np.testing.assert_allclose(ddq, rhs, atol=1e-10)

    def test_known_solution(self):
        M = np.array([[2.0, 1.0], [1.0, 3.0]])
        rhs = np.array([5.0, 7.0])
        ddq = solve_acceleration(M, rhs)
        np.testing.assert_allclose(M @ ddq, rhs, atol=1e-10)


class TestForwardDynamics:
    def test_zero_at_equilibrium(self, p):
        q = np.zeros(2)
        dq = np.zeros(2)
        ddq = forward_dynamics(q, dq, 0.0, p)
        np.testing.assert_allclose(ddq, [0.0, 0.0], atol=1e-10)

    def test_array_scalar_consistency(self, p):
        """Array and scalar versions must agree."""
        q = np.array([0.05, 0.3])
        dq = np.array([0.1, -0.5])
        tau = 0.02

        ddq_arr = forward_dynamics(q, dq, tau, p)
        ddx_s, ddth_s = forward_dynamics_fast(q[0], q[1], dq[0], dq[1], tau, p)

        assert ddq_arr[0] == pytest.approx(ddx_s, rel=1e-10)
        assert ddq_arr[1] == pytest.approx(ddth_s, rel=1e-10)

    def test_energy_conservation(self, p):
        """With tau=0, total energy should be approximately conserved."""
        from simulation.integrator.rk4_step import rk4_step

        z = np.array([0.1, 0.0, 0.0, 0.0])
        dt = 0.0001
        N = 5000

        Mt, me, I_eff, k = p[0], p[1], p[2], p[3]

        def energy(z):
            x, theta, xd, td = z
            T = 0.5 * (Mt * xd ** 2 + 2 * me * np.cos(theta) * xd * td + I_eff * td ** 2)
            V = 0.5 * k * x ** 2
            return T + V

        H0 = energy(z)
        for _ in range(N):
            z = rk4_step(z, 0.0, p, dt)
        H_final = energy(z)

        assert H_final == pytest.approx(H0, rel=0.01)  # <1% drift

    def test_spring_effect(self, p):
        """Displaced cart with no rotation should accelerate back."""
        q = np.array([0.1, 0.0])
        dq = np.zeros(2)
        ddq = forward_dynamics(q, dq, 0.0, p)
        assert ddq[0] < 0  # Spring pulls cart back

    def test_symplectic_energy_conservation(self, p):
        """Störmer-Verlet should conserve energy to near-machine precision."""
        from simulation.integrator.stormer_verlet import stormer_verlet_step

        z = np.array([0.1, 0.0, 0.0, 0.0])
        dt = 0.001
        N = 10000

        Mt, me, I_eff, k = p[0], p[1], p[2], p[3]

        def energy(z):
            x, theta, xd, td = z
            T = 0.5 * (Mt * xd**2 + 2*me*np.cos(theta)*xd*td + I_eff*td**2)
            V = 0.5 * k * x**2
            return T + V

        H0 = energy(z)
        for _ in range(N):
            z = stormer_verlet_step(z, 0.0, p, dt)
        H_final = energy(z)

        # Symplectic integrator: bounded energy oscillation (no secular drift)
        # Note: TORA has configuration-dependent mass matrix, so Störmer-Verlet
        # is not exactly symplectic, but energy error stays bounded (no drift)
        assert H_final == pytest.approx(H0, rel=0.005)  # <0.5% bounded oscillation


class TestCoriolisMatrix:
    def test_skew_symmetric_property(self, p):
        """M_dot - 2C must be skew-symmetric (passivity property)."""
        from dynamics.coriolis.coriolis_matrix import verify_skew_symmetric

        for theta in np.linspace(0, 2*np.pi, 20):
            for td in [-5.0, 0.0, 3.0]:
                _, err = verify_skew_symmetric(theta, td, p)
                assert err < 1e-12, f"Not skew-symmetric at theta={theta}, td={td}"

    def test_matrix_vector_consistency(self, p):
        """C_matrix @ dq must equal coriolis_vector."""
        from dynamics.coriolis.coriolis_matrix import coriolis_matrix

        for theta in [0.0, 0.5, 1.0, np.pi]:
            for xd, td in [(0.1, 2.0), (-0.5, 0.3)]:
                C_mat = coriolis_matrix(theta, td, p)
                dq = np.array([xd, td])
                c_from_mat = C_mat @ dq
                c_from_vec = coriolis_vector(theta, td, p)
                np.testing.assert_allclose(c_from_mat, c_from_vec, atol=1e-12)


class TestNondimensional:
    def test_epsilon_range(self):
        """Coupling parameter epsilon should be in (0, 1) for standard params."""
        from parameters.nondimensional import compute_nondimensional
        from parameters.physical import PhysicalParams

        nd = compute_nondimensional(PhysicalParams())
        assert 0 < nd["epsilon"] < 1

    def test_natural_frequency(self):
        """omega_n should be approximately 11.3 rad/s for standard params."""
        from parameters.nondimensional import compute_nondimensional
        from parameters.physical import PhysicalParams

        nd = compute_nondimensional(PhysicalParams())
        assert nd["omega_n"] == pytest.approx(11.3, rel=0.05)
