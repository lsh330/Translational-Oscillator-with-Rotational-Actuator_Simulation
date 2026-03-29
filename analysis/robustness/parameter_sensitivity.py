"""Parameter sensitivity analysis for the TORA LQR design."""

import numpy as np

from control.linearization.linearize import linearize
from control.riccati.solve_care import solve_care
from control.gain_computation.compute_K import compute_K
from control.cost_matrices.default_Q import default_Q
from control.cost_matrices.default_R import default_R


def parameter_sensitivity(p_nominal, param_names=None, delta=0.05):
    """Analyze how closed-loop poles move with ±delta parameter perturbation.

    Parameters
    ----------
    p_nominal : float64[4]  Nominal packed parameters.
    param_names : list or None  Names for display.
    delta : float  Fractional perturbation (default 5%).

    Returns
    -------
    result : dict  Per-parameter pole sensitivity.
    """
    if param_names is None:
        param_names = ["Mt", "me", "I_eff", "k"]

    Q = default_Q()
    R = default_R()

    # Nominal poles
    A_nom, B_nom = linearize(p_nominal)
    P_nom = solve_care(A_nom, B_nom, Q, R)
    K_nom = compute_K(B_nom, R, P_nom)
    poles_nom = np.linalg.eigvals(A_nom - B_nom @ K_nom)

    sensitivities = {}
    for idx, name in enumerate(param_names):
        poles_plus = []
        poles_minus = []

        for sign, label in [(+1, "plus"), (-1, "minus")]:
            p_pert = p_nominal.copy()
            p_pert[idx] *= (1.0 + sign * delta)

            A_p, B_p = linearize(p_pert)
            try:
                P_p = solve_care(A_p, B_p, Q, R)
                K_p = compute_K(B_p, R, P_p)
                poles_p = np.linalg.eigvals(A_p - B_p @ K_p)
            except Exception:
                poles_p = np.full(4, np.nan + 0j)

            if sign == 1:
                poles_plus = poles_p
            else:
                poles_minus = poles_p

        sensitivities[name] = {
            "poles_nominal": poles_nom,
            "poles_plus": np.array(poles_plus),
            "poles_minus": np.array(poles_minus),
        }

    return sensitivities
