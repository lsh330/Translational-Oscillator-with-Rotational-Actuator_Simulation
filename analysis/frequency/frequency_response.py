"""Master frequency analysis combining all sub-modules."""

import numpy as np

from analysis.frequency.poles import compute_poles
from analysis.frequency.open_loop_response import open_loop_response
from analysis.frequency.closed_loop_response import closed_loop_response
from analysis.frequency.sensitivity import sensitivity_functions
from analysis.frequency.stability_margins import stability_margins
from analysis.frequency.step_response import step_response


def frequency_analysis(A, B, K):
    """Run complete frequency-domain analysis.

    Returns
    -------
    result : dict  All frequency analysis results.
    """
    omega = np.logspace(-1, 3, 2000)

    poles = compute_poles(A, B, K)
    ol_mag, ol_phase, L = open_loop_response(A, B, K, omega)
    cl_mag, cl_phase, T = closed_loop_response(L, omega)
    S, T_func, Ms, Mt = sensitivity_functions(L)
    margins = stability_margins(omega, L)

    A_cl = A - B @ K
    step = step_response(A_cl, B)

    return {
        "omega": omega,
        "poles": poles,
        "open_loop": {"mag": ol_mag, "phase": ol_phase, "L": L},
        "closed_loop": {"mag": cl_mag, "phase": cl_phase, "T": T},
        "sensitivity": {"S": S, "T": T_func, "Ms": Ms, "Mt": Mt},
        "margins": margins,
        "step_response": step,
    }
