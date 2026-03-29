"""First-order actuator dynamics: tau_dot = (tau_cmd - tau) / T_a

Models the delay between commanded torque and actual applied torque.
When T_a = 0, the actuator is ideal (instantaneous response).
"""

from numba import njit


@njit(cache=True)
def actuator_lag_step(tau_actual, tau_cmd, T_a, dt):
    """One-step first-order actuator lag integration.

    Parameters
    ----------
    tau_actual : float  Current actual torque.
    tau_cmd    : float  Commanded torque.
    T_a        : float  Actuator time constant [s]. 0 = ideal.
    dt         : float  Time step.

    Returns
    -------
    tau_next : float  Updated actual torque.
    """
    if T_a <= 0.0:
        return tau_cmd
    alpha = dt / T_a
    return tau_actual + alpha * (tau_cmd - tau_actual)
