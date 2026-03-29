"""Closed-loop system properties for the TORA."""

import numpy as np

from utils.logger import get_logger

_log = get_logger("tora.closed_loop")


def closed_loop_analysis(A, B, K):
    """Analyze the closed-loop system A_cl = A - B*K.

    Parameters
    ----------
    A : (4,4)  Open-loop state matrix.
    B : (4,1)  Input matrix.
    K : (1,4)  Gain matrix.

    Returns
    -------
    result : dict  Keys: A_cl, poles, damping_ratios, natural_freqs,
                          is_stable, max_real_pole.
    """
    A_cl = A - B @ K
    poles = np.linalg.eigvals(A_cl)

    # Sort by real part
    idx = np.argsort(poles.real)
    poles = poles[idx]

    # Damping ratios and natural frequencies for complex poles
    damping_ratios = []
    natural_freqs = []
    for p in poles:
        wn = abs(p)
        if wn > 1e-12:
            zeta = -p.real / wn
        else:
            zeta = 1.0
        damping_ratios.append(zeta)
        natural_freqs.append(wn)

    max_real = np.max(poles.real)
    is_stable = max_real < 0

    _log.info("Closed-loop poles: %s", np.array2string(poles, precision=4))
    _log.info("Stable: %s (max Re = %.4e)", is_stable, max_real)

    return {
        "A_cl": A_cl,
        "poles": poles,
        "damping_ratios": np.array(damping_ratios),
        "natural_freqs": np.array(natural_freqs),
        "is_stable": is_stable,
        "max_real_pole": max_real,
    }
