"""TORA Nonlinear Optimal Control Simulator.

Usage:
    python main.py                            # Default LQR
    python main.py --controller energy        # Passivity-based (Jankovic)
    python main.py --controller smc           # Sliding mode
    python main.py --compare-all              # Compare all 4 controllers
    python main.py --use-ilqr --compare-all   # Include iLQR
    python main.py --x0 0.2 --t-end 30       # Custom IC and duration
    python main.py --config config.yaml       # YAML configuration
    python main.py --help
"""

import argparse
import sys

from parameters.config import SystemConfig
from pipeline.runner import run


def _build_parser():
    p = argparse.ArgumentParser(
        description="TORA Nonlinear Optimal Control Simulator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # System parameters
    p.add_argument("--M", type=float, default=1.3608, help="Cart mass [kg]")
    p.add_argument("--m", type=float, default=0.096, help="Rotor mass [kg]")
    p.add_argument("--e", type=float, default=0.0592, help="Eccentricity [m]")
    p.add_argument("--k", type=float, default=186.3, help="Spring constant [N/m]")
    p.add_argument("--I", type=float, default=0.0002175, help="Rotor inertia [kg*m^2]")

    # Simulation parameters
    p.add_argument("--t-end", type=float, default=20.0, help="Duration [s]")
    p.add_argument("--dt", type=float, default=0.001, help="Time step [s]")
    p.add_argument("--x0", type=float, default=0.1, help="Initial cart displacement [m]")
    p.add_argument("--tau-max", type=float, default=0.1, help="Torque saturation [N*m]")
    p.add_argument("--dist-amplitude", type=float, default=0.01, help="Disturbance RMS [N*m]")
    p.add_argument("--dist-bandwidth", type=float, default=5.0, help="Disturbance bandwidth [Hz]")
    p.add_argument("--seed", type=int, default=42, help="Random seed")

    # Controller selection
    p.add_argument("--controller", choices=["lqr", "energy", "smc"],
                   default="lqr", help="Controller type")
    p.add_argument("--use-ilqr", action="store_true", help="Enable iLQR optimization")
    p.add_argument("--ilqr-horizon", type=int, default=1000, help="iLQR horizon steps")
    p.add_argument("--ilqr-iterations", type=int, default=15, help="iLQR max iterations")
    p.add_argument("--compare-all", action="store_true", help="Compare all controllers")
    p.add_argument("--adaptive-q", action="store_true", help="Use Bryson's rule Q matrix")

    # Config file
    p.add_argument("--config", type=str, default=None, help="YAML config file path")

    # Output
    p.add_argument("--no-display", action="store_true", help="Skip matplotlib display")
    p.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING"],
                   default="INFO", help="Log level")

    return p


def _load_yaml_config(path):
    """Load and validate configuration from YAML file."""
    try:
        import yaml
    except ImportError:
        print("ERROR: pyyaml required for --config. Install: pip install pyyaml")
        sys.exit(1)

    try:
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"ERROR: Config file not found: {path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"ERROR: Invalid YAML in {path}: {e}")
        sys.exit(1)

    return cfg if cfg is not None else {}


def main(argv=None):
    """Entry point for CLI and programmatic use."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    import logging
    logging.getLogger("tora").setLevel(getattr(logging, args.log_level))

    # Validate
    if args.dt <= 0:
        print("ERROR: --dt must be positive"); sys.exit(1)
    if args.t_end <= 0:
        print("ERROR: --t-end must be positive"); sys.exit(1)
    if args.tau_max < 0:
        print("ERROR: --tau-max must be non-negative"); sys.exit(1)

    # YAML config override
    if args.config is not None:
        yaml_cfg = _load_yaml_config(args.config)

        sys_params = yaml_cfg.get("system", {})
        for key in ["M", "m", "e", "k", "I"]:
            if key in sys_params and f"--{key}" not in sys.argv:
                setattr(args, key, sys_params[key])

        sim_params = yaml_cfg.get("simulation", {})
        yaml_to_attr = {
            "t_end": "t_end", "dt": "dt", "x0": "x0",
            "tau_max": "tau_max", "dist_amplitude": "dist_amplitude",
            "dist_bandwidth": "dist_bandwidth", "seed": "seed",
        }
        for ykey, attr in yaml_to_attr.items():
            if ykey in sim_params:
                setattr(args, attr, sim_params[ykey])

        feat_params = yaml_cfg.get("features", {})
        if "controller" in feat_params:
            args.controller = feat_params["controller"]
        if feat_params.get("use_ilqr"):
            args.use_ilqr = True
        if feat_params.get("compare_all"):
            args.compare_all = True
        if feat_params.get("adaptive_q"):
            args.adaptive_q = True

    cfg = SystemConfig(M=args.M, m=args.m, e=args.e, k=args.k, I=args.I)

    run(cfg,
        t_end=args.t_end, dt=args.dt, x0=args.x0,
        controller_type=args.controller, tau_max=args.tau_max,
        dist_amplitude=args.dist_amplitude, dist_bandwidth=args.dist_bandwidth,
        seed=args.seed, use_ilqr=args.use_ilqr,
        ilqr_horizon=args.ilqr_horizon, ilqr_iterations=args.ilqr_iterations,
        adaptive_q=args.adaptive_q, compare_all=args.compare_all,
        no_display=args.no_display)


if __name__ == "__main__":
    main()
