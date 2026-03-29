"""Main pipeline orchestrator for the TORA simulation.

Workflow:
    0. JIT warmup
    1. LQR design (always computed as baseline)
    2. Simulate with selected controller
    3. Analysis: energy, coupling, frequency, LQR verification, phase
    4. Visualization: 7 figure categories + animation
    5. (Optional) Controller comparison
    6. (Optional) iLQR trajectory optimization
    7. (Optional) ROA estimation
    8. Save outputs
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from parameters.config import SystemConfig
from control.lqr import compute_lqr
from control.closed_loop import closed_loop_analysis
from simulation.loop.time_loop import simulate
from simulation.warmup import warmup
from analysis.energy.total_energy import total_energy
from analysis.coupling.coupling_strength import coupling_strength
from analysis.coupling.effective_inertia import effective_inertia
from analysis.frequency.frequency_response import frequency_analysis
from analysis.lqr_verification.compute_verification import lqr_verification
from analysis.state.phase_portrait import phase_portrait_data
from analysis.summary.print_summary import print_summary
from visualization.dynamics_plots.show_dynamics_plots import show_dynamics_plots
from visualization.control_plots.show_control_plots import show_control_plots
from visualization.lqr_plots.show_lqr_plots import show_lqr_plots
from visualization.phase_plots.show_phase_plots import show_phase_plots
from visualization.animation.show_animation import show_animation
from pipeline.save_outputs import save_figure, save_animation
from utils.logger import get_logger

_log = get_logger("tora.runner")


def run(cfg, t_end=20.0, dt=0.001, x0=0.1,
        controller_type="lqr", tau_max=0.1,
        dist_amplitude=0.01, dist_bandwidth=5.0, seed=42,
        use_ilqr=False, ilqr_horizon=1000, ilqr_iterations=15,
        adaptive_q=False, compare_all=False, no_display=False):
    """Execute the full TORA simulation pipeline."""

    p = cfg.pack()

    # 0. JIT warmup
    _log.info("Phase 0: JIT warmup")
    warmup()

    # 1. LQR design
    _log.info("Phase 1: LQR design")
    lqr_result = compute_lqr(p, use_adaptive_q=adaptive_q)
    K = lqr_result["K"]

    cl_info = closed_loop_analysis(lqr_result["A"], lqr_result["B"], K)

    # 2. Simulate
    _log.info("Phase 2: Simulation (%s)", controller_type.upper())
    sim_result = simulate(
        cfg, controller_type=controller_type, K=K,
        t_end=t_end, dt=dt, x0=x0, tau_max=tau_max,
        dist_amplitude=dist_amplitude, dist_bandwidth=dist_bandwidth, seed=seed,
    )

    # 3. Analysis
    _log.info("Phase 3: Analysis")
    freq_result = frequency_analysis(lqr_result["A"], lqr_result["B"], K)
    verification = lqr_verification(sim_result, lqr_result, cfg_class=SystemConfig)

    # Summary
    print_summary(sim_result, lqr_result, freq_result, verification)

    # 4. Visualization
    _log.info("Phase 4: Visualization")
    figs = {}

    figs["dynamics"] = show_dynamics_plots(sim_result, p)
    save_figure(figs["dynamics"], "dynamics")

    figs["control"] = show_control_plots(sim_result, freq_result)
    save_figure(figs["control"], "control")

    figs["lqr"] = show_lqr_plots(sim_result, verification)
    save_figure(figs["lqr"], "lqr_verification")

    figs["phase"] = show_phase_plots(sim_result)
    save_figure(figs["phase"], "phase_portraits")

    try:
        fig_anim, anim = show_animation(sim_result, e=cfg.physical.e)
        save_figure(fig_anim, "animation_frame")
        save_animation(anim, "tora_animation", fps=50)
    except Exception as e:
        _log.warning("Animation failed: %s", e)

    # 5. Controller comparison (optional)
    if compare_all:
        _log.info("Phase 5: Controller comparison")
        from control.comparison import compare_controllers

        ilqr_res = None
        if use_ilqr:
            from control.ilqr import ilqr
            ilqr_res = ilqr(p, dt, ilqr_horizon, max_iter=ilqr_iterations)

        comp = compare_controllers(p, K, ilqr_result=ilqr_res,
                                   t_end=t_end, dt=dt, x0=x0, tau_max=tau_max)

        from visualization.comparison_plots.show_comparison_plots import show_comparison_plots
        figs["comparison"] = show_comparison_plots(comp)
        save_figure(figs["comparison"], "comparison")

    # 6. ROA (optional, slow)
    # Uncomment to enable: takes ~30s
    # from analysis.region_of_attraction import estimate_roa
    # roa = estimate_roa(p, K, nx=31, ntheta=31)
    # from visualization.roa_plots.show_roa_plots import show_roa_plots
    # figs["roa"] = show_roa_plots(roa)
    # save_figure(figs["roa"], "roa")

    if not no_display:
        plt.show()

    plt.close("all")

    _log.info("Pipeline complete.")
    return {
        "sim_result": sim_result,
        "lqr_result": lqr_result,
        "freq_result": freq_result,
        "verification": verification,
        "cl_info": cl_info,
        "figs": figs,
    }
