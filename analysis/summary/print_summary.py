"""Formatted console summary of simulation and analysis results."""

from utils.logger import get_logger

_log = get_logger("tora.summary")


def print_summary(sim_result, lqr_result, freq_result=None, verification=None):
    """Print a formatted summary table to the logger."""
    import numpy as np

    _log.info("=" * 60)
    _log.info("  TORA Simulation Summary")
    _log.info("=" * 60)

    # Simulation info
    t = sim_result["t"]
    _log.info("  Controller : %s", sim_result["controller"].upper())
    _log.info("  Duration   : %.1f s  (%d steps)", t[-1], len(t) - 1)
    _log.info("  Saturations: %d (%.1f%%)",
              sim_result["sat_count"],
              100.0 * sim_result["sat_count"] / (len(t) - 1))

    # Final state
    _log.info("  Final state: x=%.4e  theta=%.4e  xd=%.4e  td=%.4e",
              sim_result["x"][-1], sim_result["theta"][-1],
              sim_result["x_dot"][-1], sim_result["theta_dot"][-1])

    # LQR info
    K = lqr_result["K"]
    _log.info("  LQR gain K : %s", np.array2string(K.flatten(), precision=4))
    poles = lqr_result["poles_cl"]
    _log.info("  CL poles   : %s", np.array2string(poles, precision=4))

    # Frequency info
    if freq_result is not None:
        m = freq_result["margins"]
        _log.info("  Gain margin: %.1f dB", m["gain_margin_dB"])
        _log.info("  Phase margin: %.1f deg", m["phase_margin_deg"])

    # Verification
    if verification is not None:
        lyap = verification["lyapunov"]
        _log.info("  Lyapunov OK: %s (violations=%d)",
                  lyap["monotone_decreasing"], lyap["violations"])
        rd = verification["return_difference"]
        _log.info("  Kalman ineq: %s (min|1+L|=%.4f)",
                  rd["kalman_satisfied"], rd["min_return_difference"])
        if "monte_carlo" in verification:
            mc = verification["monte_carlo"]
            _log.info("  MC robust  : %.1f%% (%d trials)",
                      100 * mc["success_rate"], mc["N_trials"])

    _log.info("=" * 60)
