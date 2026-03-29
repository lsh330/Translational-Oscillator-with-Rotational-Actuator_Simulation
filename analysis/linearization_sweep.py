"""Operating-point sweep linearization analysis.

Linearizes the TORA at multiple operating points to assess
where the linear approximation is valid and where it breaks down.
"""

import numpy as np
from dynamics.forward_dynamics.forward_dynamics import forward_dynamics
from control.linearization.jit_jacobians import compute_numerical_state_space


def linearization_sweep(p, x_range=(-0.2, 0.2), theta_range=(-1.0, 1.0),
                        n_x=11, n_theta=21):
    """Sweep linearization across operating points.

    Returns
    -------
    result : dict  Keys: x_grid, theta_grid, A_norms, B_norms,
                          eigenvalue_map, controllability_map.
    """
    x_vals = np.linspace(x_range[0], x_range[1], n_x)
    theta_vals = np.linspace(theta_range[0], theta_range[1], n_theta)

    A_norms = np.zeros((n_x, n_theta))
    eig_real_max = np.zeros((n_x, n_theta))
    controllable = np.zeros((n_x, n_theta))

    for i, x in enumerate(x_vals):
        for j, theta in enumerate(theta_vals):
            q_eq = np.array([x, theta])
            dq_eq = np.zeros(2)
            A, B = compute_numerical_state_space(q_eq, dq_eq, 0.0, p)

            A_norms[i, j] = np.linalg.norm(A)
            eigvals = np.linalg.eigvals(A)
            eig_real_max[i, j] = np.max(eigvals.real)

            # Controllability rank
            C_mat = B.copy()
            AB = B.copy()
            for _ in range(3):
                AB = A @ AB
                C_mat = np.hstack([C_mat, AB])
            controllable[i, j] = float(np.linalg.matrix_rank(C_mat) == 4)

    return {
        "x_grid": x_vals,
        "theta_grid": theta_vals,
        "A_norms": A_norms,
        "eig_real_max": eig_real_max,
        "controllability_map": controllable,
    }
