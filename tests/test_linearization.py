"""Tests for the linearization module."""

import numpy as np
import pytest

from parameters.config import SystemConfig
from control.linearization.linearize import linearize
from control.linearization.analytical_jacobian import analytical_A, analytical_B


@pytest.fixture
def p():
    return SystemConfig().pack()


class TestAnalyticalJacobian:
    def test_A_shape(self, p):
        A = analytical_A(p)
        assert A.shape == (4, 4)

    def test_B_shape(self, p):
        B = analytical_B(p)
        assert B.shape == (4, 1)

    def test_top_right_identity(self, p):
        A = analytical_A(p)
        np.testing.assert_allclose(A[0, 2], 1.0)
        np.testing.assert_allclose(A[1, 3], 1.0)

    def test_top_left_zero(self, p):
        A = analytical_A(p)
        np.testing.assert_allclose(A[0, 0], 0.0)
        np.testing.assert_allclose(A[0, 1], 0.0)
        np.testing.assert_allclose(A[1, 0], 0.0)
        np.testing.assert_allclose(A[1, 1], 0.0)

    def test_no_damping(self, p):
        """A_dq block should be zero (no dissipation in TORA)."""
        A = analytical_A(p)
        np.testing.assert_allclose(A[2:4, 2:4], 0.0, atol=1e-15)


class TestNumericalVsAnalytical:
    def test_A_matrices_agree(self, p):
        A_a, B_a = linearize(p, method="analytical")
        A_n, B_n = linearize(p, method="numerical")
        np.testing.assert_allclose(A_a, A_n, rtol=1e-5, atol=1e-8)

    def test_B_matrices_agree(self, p):
        _, B_a = linearize(p, method="analytical")
        _, B_n = linearize(p, method="numerical")
        np.testing.assert_allclose(B_a, B_n, rtol=1e-5, atol=1e-8)


class TestOpenLoopProperties:
    def test_marginally_stable(self, p):
        """Open-loop TORA has purely imaginary poles (not unstable!)."""
        A, _ = linearize(p)
        eigvals = np.linalg.eigvals(A)
        # All eigenvalues should have ~zero real part
        for ev in eigvals:
            assert abs(ev.real) < 1e-6, f"Non-imaginary pole: {ev}"

    def test_imaginary_poles_exist(self, p):
        """Should have at least 2 conjugate pairs on imaginary axis."""
        A, _ = linearize(p)
        eigvals = np.linalg.eigvals(A)
        imag_parts = np.abs(eigvals.imag)
        assert np.sum(imag_parts > 0.1) >= 2

    def test_controllable(self, p):
        """System must be controllable (rank of controllability matrix = 4)."""
        A, B = linearize(p)
        n = 4
        C_mat = B.copy()
        AB = B.copy()
        for _ in range(n - 1):
            AB = A @ AB
            C_mat = np.hstack([C_mat, AB])
        rank = np.linalg.matrix_rank(C_mat)
        assert rank == n
