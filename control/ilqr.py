"""Iterative LQR (iLQR) trajectory optimization for the TORA.

Algorithm:
    1. Forward pass: simulate nominal trajectory with current controls
    2. Backward pass: compute time-varying gains via Riccati recursion
    3. Forward pass: update trajectory with new gains
    4. Repeat until convergence (cost reduction < tol)
"""

import numpy as np
from scipy.linalg import expm

from dynamics.forward_dynamics.forward_dynamics import forward_dynamics
from control.linearization.linearize import linearize
from control.cost_matrices.default_Q import default_Q
from control.cost_matrices.default_R import default_R
from control.riccati.solve_care import solve_care
from utils.logger import get_logger

_log = get_logger("tora.ilqr")


def _rk4_step(q, dq, tau, p, dt):
    """Single RK4 integration step (array interface)."""
    def _deriv(q_, dq_, tau_):
        ddq_ = forward_dynamics(q_, dq_, tau_, p)
        return dq_.copy(), ddq_

    dq1, ddq1 = _deriv(q, dq, tau)
    q2 = q + 0.5 * dt * dq1
    dq2_v = dq + 0.5 * dt * ddq1
    dq2, ddq2 = _deriv(q2, dq2_v, tau)
    q3 = q + 0.5 * dt * dq2
    dq3_v = dq + 0.5 * dt * ddq2
    dq3, ddq3 = _deriv(q3, dq3_v, tau)
    q4 = q + dt * dq3
    dq4_v = dq + dt * ddq3
    dq4, ddq4 = _deriv(q4, dq4_v, tau)

    q_next = q + (dt / 6.0) * (dq1 + 2.0 * dq2 + 2.0 * dq3 + dq4)
    dq_next = dq + (dt / 6.0) * (ddq1 + 2.0 * ddq2 + 2.0 * ddq3 + ddq4)
    return q_next, dq_next


def _state_pack(q, dq):
    return np.array([q[0], q[1], dq[0], dq[1]])


def _state_unpack(z):
    return z[:2].copy(), z[2:].copy()


def _linearize_at(q, dq, tau, p, dt):
    """Linearize and discretize about an operating point.

    Uses ZOH discretization: A_d = expm(A*dt), B_d = integral.
    """
    from control.linearization.jit_jacobians import compute_numerical_state_space

    A_c, B_c = compute_numerical_state_space(q, dq, tau, p)

    # Exact discretization via matrix exponential
    n = 4
    M_aug = np.zeros((n + 1, n + 1))
    M_aug[:n, :n] = A_c * dt
    M_aug[:n, n:] = B_c * dt
    E = expm(M_aug)
    A_d = E[:n, :n]
    B_d = E[:n, n:]
    return A_d, B_d


def ilqr(p, dt, horizon, max_iter=15, tol=1e-4, z0=None):
    """Run iLQR trajectory optimization.

    Parameters
    ----------
    p        : float64[4]  Packed parameters.
    dt       : float       Integration timestep [s].
    horizon  : int         Number of time steps.
    max_iter : int         Maximum iterations.
    tol      : float       Convergence tolerance on relative cost reduction.
    z0       : float64[4]  Initial state (default: [0.1, 0, 0, 0]).

    Returns
    -------
    result : dict  Keys: z_traj (N+1,4), u_traj (N,), K_traj (N,1,4),
                          cost_history, converged.
    """
    if z0 is None:
        z0 = np.array([0.1, 0.0, 0.0, 0.0])

    N = horizon
    Q = default_Q()
    R = default_R()
    R_scalar = R[0, 0]

    # Terminal cost from infinite-horizon LQR
    A_lin, B_lin = linearize(p, method="analytical")
    P_f = solve_care(A_lin, B_lin, Q, R)

    # Initialize controls to zero
    u_traj = np.zeros(N)

    cost_history = []

    for iteration in range(max_iter):
        # Forward pass: simulate
        z_traj = np.zeros((N + 1, 4))
        z_traj[0] = z0
        for i in range(N):
            q, dq = _state_unpack(z_traj[i])
            q_next, dq_next = _rk4_step(q, dq, u_traj[i], p, dt)
            z_traj[i + 1] = _state_pack(q_next, dq_next)

        # Compute cost
        cost = 0.0
        for i in range(N):
            z = z_traj[i]
            u = u_traj[i]
            cost += 0.5 * (z @ Q @ z + R_scalar * u * u) * dt
        cost += 0.5 * z_traj[N] @ P_f @ z_traj[N]
        cost_history.append(cost)

        if iteration > 0:
            reduction = (cost_history[-2] - cost) / (abs(cost_history[-2]) + 1e-12)
            _log.info("iLQR iter %d: cost=%.6e, reduction=%.4e", iteration, cost, reduction)
            if reduction < tol and reduction >= 0:
                _log.info("iLQR converged at iteration %d", iteration)
                break
        else:
            _log.info("iLQR iter 0: cost=%.6e", cost)

        # Backward pass: Riccati recursion
        K_traj = np.zeros((N, 1, 4))
        k_traj = np.zeros((N, 1))
        V_x = P_f @ z_traj[N]
        V_xx = P_f.copy()

        for i in range(N - 1, -1, -1):
            q, dq = _state_unpack(z_traj[i])
            A_d, B_d = _linearize_at(q, dq, u_traj[i], p, dt)

            Q_x = Q @ z_traj[i] * dt + A_d.T @ V_x
            Q_u = R_scalar * u_traj[i] * dt + B_d.T @ V_x
            Q_xx = Q * dt + A_d.T @ V_xx @ A_d
            Q_ux = B_d.T @ V_xx @ A_d
            Q_uu = R_scalar * dt + B_d.T @ V_xx @ B_d

            Q_uu_inv = 1.0 / (Q_uu[0, 0] + 1e-8)

            K_i = Q_uu_inv * Q_ux
            k_i = Q_uu_inv * Q_u

            K_traj[i] = K_i
            k_traj[i] = k_i.reshape(1)

            V_x = Q_x - K_i.T @ (Q_uu @ k_i.reshape(-1, 1)).flatten()
            V_xx = Q_xx - K_i.T @ Q_uu @ K_i

        # Forward pass with updated gains
        z_new = np.zeros((N + 1, 4))
        u_new = np.zeros(N)
        z_new[0] = z0
        alpha = 1.0  # Line search step size

        for i in range(N):
            dz = z_new[i] - z_traj[i]
            du = -k_traj[i, 0] - K_traj[i, 0, :] @ dz
            u_new[i] = u_traj[i] + alpha * du

            q, dq = _state_unpack(z_new[i])
            q_next, dq_next = _rk4_step(q, dq, u_new[i], p, dt)
            z_new[i + 1] = _state_pack(q_next, dq_next)

        u_traj = u_new

    converged = len(cost_history) > 1 and (
        (cost_history[-2] - cost_history[-1]) / (abs(cost_history[-2]) + 1e-12) < tol
    )

    return {
        "z_traj": z_traj,
        "u_traj": u_traj,
        "K_traj": K_traj,
        "cost_history": np.array(cost_history),
        "converged": converged,
    }
