"""Tests for the LQR control design."""

import numpy as np
import pytest

from parameters.config import SystemConfig
from control.lqr import compute_lqr


@pytest.fixture
def p():
    return SystemConfig().pack()


@pytest.fixture
def lqr(p):
    return compute_lqr(p)


class TestRiccatiSolution:
    def test_P_positive_definite(self, lqr):
        P = lqr["P"]
        eigvals = np.linalg.eigvalsh(P)
        assert np.all(eigvals > 0)

    def test_P_symmetric(self, lqr):
        P = lqr["P"]
        np.testing.assert_allclose(P, P.T, atol=1e-10)

    def test_riccati_residual(self, lqr):
        A, B, P, Q, R = lqr["A"], lqr["B"], lqr["P"], lqr["Q"], lqr["R"]
        residual = A.T @ P + P @ A - P @ B @ np.linalg.solve(R, B.T @ P) + Q
        assert np.max(np.abs(residual)) < 1e-6


class TestGainMatrix:
    def test_K_shape(self, lqr):
        K = lqr["K"]
        assert K.shape == (1, 4)

    def test_K_formula(self, lqr):
        B, R, P, K = lqr["B"], lqr["R"], lqr["P"], lqr["K"]
        K_check = np.linalg.solve(R, B.T @ P)
        np.testing.assert_allclose(K, K_check, rtol=1e-8)


class TestClosedLoopStability:
    def test_all_poles_in_lhp(self, lqr):
        poles = lqr["poles_cl"]
        assert np.all(poles.real < 0)

    def test_max_real_pole_negative(self, lqr):
        poles = lqr["poles_cl"]
        assert np.max(poles.real) < -1e-6

    def test_kalman_inequality(self, lqr):
        """Return difference |1 + L(jw)| >= 1 for all omega."""
        A, B, K = lqr["A"], lqr["B"], lqr["K"]
        I4 = np.eye(4)
        omega = np.logspace(-1, 3, 500)

        for w in omega:
            resolvent = np.linalg.solve(1j * w * I4 - A, B)
            L = (K @ resolvent)[0, 0]
            rd = abs(1.0 + L)
            assert rd >= 1.0 - 1e-4, f"|1+L| = {rd} at omega={w}"


class TestAdaptiveQ:
    def test_adaptive_q_stable(self, p):
        result = compute_lqr(p, use_adaptive_q=True)
        assert np.all(result["poles_cl"].real < 0)
