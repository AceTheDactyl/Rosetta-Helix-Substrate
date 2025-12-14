"""Command-line interface for the QuantumAPL Python tooling."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .analyzer import QuantumAnalyzer
from .constants import Z_CRITICAL
from .engine import QuantumAPLEngine


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Quantum-Classical APL Simulation (Python Interface)")
    parser.add_argument("--steps", type=int, default=100, help="Number of simulation steps (default: 100)")
    parser.add_argument(
        "--mode",
        choices=["unified", "quantum_only", "test", "z_pump", "measured"],
        default="unified",
        help="Simulation mode (default: unified)",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose Node.js output")
    parser.add_argument("--plot", action="store_true", help="Generate matplotlib plots (if available)")
    parser.add_argument("--output", type=Path, help="Save results JSON to path")
    parser.add_argument("--js-dir", type=Path, help="Directory containing JS files (defaults to repo root)")
    # Pump controls (z_pump mode)
    parser.add_argument(
        "--z-pump",
        nargs="?",
        const=Z_CRITICAL,
        type=float,
        help=f"Shortcut: enable z_pump mode with target z (default Z_CRITICAL≈{Z_CRITICAL:.16f}); cycles taken from --steps",
    )
    parser.add_argument(
        "--z-pump-target",
        type=float,
        help=f"Target z for APL pump mode (default: Z_CRITICAL≈{Z_CRITICAL:.16f})",
    )
    parser.add_argument("--z-pump-cycles", type=int, help="Cycle count for APL pump mode (default: 120)")
    parser.add_argument("--z-pump-profile", choices=["gentle", "balanced", "aggressive"],
                        help="Pump profile controlling gain/sigma and cadence")
    # Measured mode controls
    parser.add_argument("--measure-eigen", type=int, help="Apply eigen collapse in measured mode (μ)")
    parser.add_argument("--measure-field", choices=["Phi", "Pi", "π"], help="Field for eigen/subspace in measured mode")
    parser.add_argument("--measure-subspace", type=str, help="Comma-separated indices for Π(subspace) in measured mode")
    parser.add_argument("--no-measure-composite", action="store_true", help="Disable composite measurement in measured mode")
    parser.add_argument(
        "--collapse-glyph",
        action="store_true",
        help="Emit collapse alias tokens (⟂) in measured mode",
    )
    return parser


def _execute(args: argparse.Namespace) -> int:
    import os
    # CLI sugar: --z-pump <target> flips mode to z_pump and maps steps→cycles
    if getattr(args, "z_pump", None) is not None:
        args.mode = "z_pump"
        os.environ["QAPL_PUMP_TARGET"] = str(args.z_pump)
        # Use --steps as cycles if provided
        if getattr(args, "steps", None) is not None:
            os.environ["QAPL_PUMP_CYCLES"] = str(args.steps)

    # Propagate pump env if explicitly requested
    if getattr(args, "z_pump_target", None) is not None:
        os.environ["QAPL_PUMP_TARGET"] = str(args.z_pump_target)
    if getattr(args, "z_pump_cycles", None) is not None:
        os.environ["QAPL_PUMP_CYCLES"] = str(args.z_pump_cycles)
    if getattr(args, "z_pump_profile", None):
        os.environ["QAPL_PUMP_PROFILE"] = str(args.z_pump_profile)

    # Measured mode env
    if args.mode == "measured":
        if getattr(args, "measure_eigen", None) is not None:
            os.environ["QAPL_MEASURE_EIGEN"] = str(args.measure_eigen)
        if getattr(args, "measure_field", None):
            os.environ["QAPL_MEASURE_FIELD"] = str(args.measure_field)
        if getattr(args, "measure_subspace", None):
            os.environ["QAPL_MEASURE_SUBSPACE"] = str(args.measure_subspace)
        os.environ["QAPL_MEASURE_COMPOSITE"] = "0" if args.no_measure_composite else "1"
        if getattr(args, "collapse_glyph", False):
            os.environ["QAPL_EMIT_COLLAPSE_GLYPH"] = "1"

    engine = QuantumAPLEngine(js_dir=args.js_dir)
    print(f"Running {args.mode} simulation with {args.steps} steps...")
    results = engine.run_simulation(steps=args.steps, verbose=args.verbose, mode=args.mode)

    if args.output:
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Results saved to {args.output}")

    if "raw_output" not in results:
        analyzer = QuantumAnalyzer(results)
        print("\n" + analyzer.summary())
        if args.plot:
            try:
                analyzer.plot()
            except ImportError as exc:
                print(exc)
    else:
        print(results["raw_output"])

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return _execute(args)


def run_simulation() -> int:
    return main()


def run_tests() -> int:
    parser = argparse.ArgumentParser(description="Run the QuantumAPL Node.js test suite")
    parser.add_argument("--js-dir", type=Path, help="Directory containing JS files (defaults to repo root)")
    parser.add_argument("--verbose", action="store_true", help="Verbose Node.js output")
    parser.add_argument("--steps", type=int, default=100, help="Step count forwarded to the runner")
    args = parser.parse_args()
    args.mode = "test"
    args.plot = False
    args.output = None
    return _execute(args)


def analyze(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze saved QuantumAPL results")
    parser.add_argument("input", type=Path, help="JSON results file")
    parser.add_argument("--plot", action="store_true", help="Generate matplotlib plots")
    args = parser.parse_args(argv)

    data = json.loads(args.input.read_text(encoding="utf-8"))
    analyzer = QuantumAnalyzer(data)
    print(analyzer.summary())
    if args.plot:
        try:
            analyzer.plot()
        except ImportError as exc:
            print(exc)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
