"""LQR verification suite: Lyapunov, Kalman inequality, Nyquist, robustness.

Comprehensive verification of the LQR design quality, following the
same methodology as the Triple-Pendulum-LQR reference project.
"""

import numpy as np

from utils.logger import get_logger

_log = get_logger("tora.verify")


def _lyapunov_verification(sim_result, P):
    """Check that V(t) = z^T P z decreases monotonically."""
    z = np.column_stack([
        sim_result["x"], sim_result["theta"],
        sim_result["x_dot"], sim_result["theta_dot"],
    ])
    N = z.shape[0]
    V = np.array([z[i] @ P @ z[i] for i in range(N)])

    dV = np.diff(V)
    violations = np.sum(dV > 1e-10)

    return {
        "V": V,
        "monotone_decreasing": violations == 0,
        "violations": violations,
        "V_initial": V[0],
        "V_final": V[-1],
    }


def _cost_breakdown(sim_result, Q, R, dt):
    """Compute state and control cost integrals."""
    z = np.column_stack([
        sim_result["x"], sim_result["theta"],
        sim_result["x_dot"], sim_result["theta_dot"],
    ])
    u = sim_result["u"]
    N = len(u)

    state_cost = np.sum([z[i] @ Q @ z[i] for i in range(N)]) * dt
    control_cost = np.sum(u ** 2) * R[0, 0] * dt
    total_cost = state_cost + control_cost

    return {
        "state_cost": state_cost,
        "control_cost": control_cost,
        "total_cost": total_cost,
        "cost_ratio": state_cost / (control_cost + 1e-15),
    }


def _return_difference(A, B, K, R, omega):
    """Compute |1 + L(jw)| for Kalman inequality verification.

    Kalman inequality: |1 + L(jw)| >= 1 for all w
    (guaranteed for LQR with full-state feedback).
    """
    I4 = np.eye(4)
    rd = np.empty(len(omega))

    for i, w in enumerate(omega):
        resolvent = np.linalg.solve(1j * w * I4 - A, B)
        L_jw = (K @ resolvent)[0, 0]
        rd[i] = abs(1.0 + L_jw)

    min_rd = np.min(rd)
    kalman_satisfied = min_rd >= 1.0 - 1e-6

    return {
        "return_difference": rd,
        "min_return_difference": min_rd,
        "kalman_satisfied": kalman_satisfied,
    }


def _pole_analysis(poles_cl):
    """Analyze closed-loop pole properties."""
    damping = []
    for p in poles_cl:
        wn = abs(p)
        zeta = -p.real / wn if wn > 1e-12 else 1.0
        damping.append(zeta)

    return {
        "poles": poles_cl,
        "damping_ratios": np.array(damping),
        "min_damping": min(damping),
        "max_real_part": np.max(poles_cl.real),
    }


def _monte_carlo_robustness(physical_params, K, N_trials=200, seed=42):
    """Monte Carlo robustness test with parameter perturbations.

    Perturbs around the ACTUAL physical parameters (not hardcoded benchmark values).
    Perturbation: M ±10%, m ±10%, e ±5%, k ±10%, I ±10%.
    """
    from simulation.integrator.rk4_step import rk4_step_fast
    rng = np.random.default_rng(seed)

    K_flat = K.flatten()
    stable_count = 0
    max_deviations = []

    # Use actual parameters as perturbation center
    M_nom = physical_params["M"]
    m_nom = physical_params["m"]
    e_nom = physical_params["e"]
    k_nom = physical_params["k"]
    I_nom = physical_params["I"]

    for trial in range(N_trials):
        # Perturb parameters
        M_pert = M_nom * (1.0 + 0.1 * (2.0 * rng.random() - 1.0))
        m_pert = m_nom * (1.0 + 0.1 * (2.0 * rng.random() - 1.0))
        e_pert = e_nom * (1.0 + 0.05 * (2.0 * rng.random() - 1.0))
        k_pert = k_nom * (1.0 + 0.1 * (2.0 * rng.random() - 1.0))
        I_pert = I_nom * (1.0 + 0.1 * (2.0 * rng.random() - 1.0))

        Mt_p = M_pert + m_pert
        me_p = m_pert * e_pert
        Ie_p = I_pert + m_pert * e_pert ** 2
        p_pert = np.array([Mt_p, me_p, Ie_p, k_pert])

        # Simulate 5 seconds
        x, theta, xd, td = 0.1, 0.0, 0.0, 0.0
        dt = 0.002
        diverged = False
        max_dev = 0.0

        for step in range(2500):
            tau = -(K_flat[0] * x + K_flat[1] * theta
                    + K_flat[2] * xd + K_flat[3] * td)
            tau = np.clip(tau, -0.5, 0.5)
            x, theta, xd, td = rk4_step_fast(x, theta, xd, td, tau, p_pert, dt)

            dev = abs(x) + abs(theta)
            if dev > max_dev:
                max_dev = dev
            if np.isnan(x) or dev > 10.0:
                diverged = True
                break

        if not diverged and abs(x) < 0.05 and abs(theta) < 0.5:
            stable_count += 1
        max_deviations.append(max_dev)

    return {
        "success_rate": stable_count / N_trials,
        "N_trials": N_trials,
        "max_deviations": np.array(max_deviations),
    }


def lqr_verification(sim_result, lqr_result, physical_params=None):
    """Run complete LQR verification suite.

    Parameters
    ----------
    sim_result      : dict  From simulate().
    lqr_result      : dict  From compute_lqr().
    physical_params : dict or None  Physical params dict with keys M,m,e,k,I
                      for MC robustness.

    Returns
    -------
    result : dict  All verification results.
    """
    A = lqr_result["A"]
    B = lqr_result["B"]
    K = lqr_result["K"]
    P = lqr_result["P"]
    Q = lqr_result["Q"]
    R = lqr_result["R"]
    poles_cl = lqr_result["poles_cl"]

    dt = sim_result["t"][1] - sim_result["t"][0]
    omega = np.logspace(-1, 3, 1000)

    lyap = _lyapunov_verification(sim_result, P)
    cost = _cost_breakdown(sim_result, Q, R, dt)
    rd = _return_difference(A, B, K, R, omega)
    pole_info = _pole_analysis(poles_cl)

    _log.info("Lyapunov monotone: %s (violations=%d)", lyap["monotone_decreasing"], lyap["violations"])
    _log.info("Kalman inequality: %s (min |1+L|=%.4f)", rd["kalman_satisfied"], rd["min_return_difference"])
    _log.info("Cost: state=%.4f, control=%.4f, total=%.4f", cost["state_cost"], cost["control_cost"], cost["total_cost"])

    result = {
        "lyapunov": lyap,
        "cost": cost,
        "return_difference": rd,
        "pole_analysis": pole_info,
        "omega": omega,
    }

    if physical_params is not None:
        mc = _monte_carlo_robustness(physical_params, K)
        _log.info("MC robustness: %.1f%% stable (%d trials)", 100 * mc["success_rate"], mc["N_trials"])
        result["monte_carlo"] = mc

    return result
