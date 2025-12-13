#!/usr/bin/env python3
"""
Helix Training Engine CLI
=========================

Usage:
    helix train  --config configs/full.yaml
    helix eval   --run runs/<run_id>
    helix resume --run runs/<run_id>
    helix export --run runs/<run_id> --format onnx|torchscript|bundle
    helix nightly --config configs/nightly.yaml
    helix promote --run <run_id> --name rosetta_v1
    helix list [--status completed|failed|running]
    helix show --run <run_id>

Signature: cli|v0.1.0|helix
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional


def train_command(args: argparse.Namespace) -> int:
    """Run training."""
    from helix_engine import run_training

    print(f"Starting training with config: {args.config}")

    kwargs = {}
    if args.steps:
        kwargs["total_steps"] = args.steps
    if args.seed:
        kwargs["seed"] = args.seed
    if args.output:
        kwargs["output_dir"] = args.output

    result = run_training(config_path=args.config, **kwargs)

    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"  Run ID:     {result.run_id}")
    print(f"  Status:     {result.status}")
    print(f"  Exit Code:  {result.exit_code.name} ({result.exit_code})")
    print(f"  Duration:   {result.duration_seconds:.1f}s")
    print(f"  Steps:      {result.total_steps}")
    print(f"  Gates:      {'PASSED' if result.gates_passed else 'FAILED'}")
    print(f"  Artifacts:  {result.artifacts_dir}")
    print(f"{'='*60}")

    return int(result.exit_code)


def eval_command(args: argparse.Namespace) -> int:
    """Evaluate a run."""
    from helix_engine import evaluate_run

    print(f"Evaluating run: {args.run}")

    results = evaluate_run(args.run, runs_dir=args.runs_dir)

    if "error" in results:
        print(f"Error: {results['error']}")
        return 1

    print(f"\n{'='*60}")
    print(f"Evaluation Results")
    print(f"{'='*60}")
    print(f"  Gates Passed: {results['gates_passed']}")
    print(f"\n  Gate Results:")
    for gate in results.get("gate_results", []):
        status = "PASS" if gate["passed"] else "FAIL"
        print(f"    [{status}] {gate['name']}: {gate['message']}")

    print(f"\n  Final Metrics:")
    for key, value in results.get("final_metrics", {}).items():
        print(f"    {key}: {value:.6f}")

    print(f"{'='*60}")

    return 0 if results["gates_passed"] else 1


def resume_command(args: argparse.Namespace) -> int:
    """Resume a run."""
    from helix_engine import resume_run

    print(f"Resuming run: {args.run}")

    checkpoint = args.checkpoint if hasattr(args, "checkpoint") else None
    result = resume_run(args.run, checkpoint_path=checkpoint, runs_dir=args.runs_dir)

    print(f"\n{'='*60}")
    print(f"Resume completed!")
    print(f"  Status:     {result.status}")
    print(f"  Exit Code:  {result.exit_code.name}")
    print(f"{'='*60}")

    return int(result.exit_code)


def export_command(args: argparse.Namespace) -> int:
    """Export a model."""
    from helix_engine import export_model

    print(f"Exporting run: {args.run}")
    print(f"  Format: {args.format}")

    try:
        path = export_model(
            args.run,
            format=args.format,
            output_path=args.output,
            runs_dir=args.runs_dir,
        )
        print(f"  Exported to: {path}")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


def promote_command(args: argparse.Namespace) -> int:
    """Promote a run to named model."""
    from helix_engine import promote_model

    print(f"Promoting run: {args.run}")
    print(f"  Name: {args.name}")

    tags = args.tags.split(",") if args.tags else None

    entry = promote_model(
        run_id=args.run,
        name=args.name,
        version=args.version,
        tags=tags,
        description=args.description or "",
        runs_dir=args.runs_dir,
    )

    print(f"\n{'='*60}")
    print(f"Model promoted!")
    print(f"  Name:    {entry.name}")
    print(f"  Version: {entry.version}")
    print(f"  Path:    {entry.checkpoint_path}")
    print(f"{'='*60}")

    return 0


def nightly_command(args: argparse.Namespace) -> int:
    """Run nightly training."""
    from helix_engine import run_training

    print("Starting nightly training run")

    result = run_training(
        config_path=args.config,
        tags=["nightly"],
    )

    # Generate nightly report
    report = {
        "run_id": result.run_id,
        "status": result.status,
        "exit_code": int(result.exit_code),
        "gates_passed": result.gates_passed,
        "duration_seconds": result.duration_seconds,
        "final_metrics": result.final_metrics,
    }

    if args.report:
        with open(args.report, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to: {args.report}")

    print(f"\n{'='*60}")
    print(f"Nightly run: {'PASSED' if result.gates_passed else 'FAILED'}")
    print(f"{'='*60}")

    return 0 if result.gates_passed else 1


def list_command(args: argparse.Namespace) -> int:
    """List runs."""
    from helix_engine import list_runs

    runs = list_runs(status=args.status, runs_dir=args.runs_dir)

    if not runs:
        print("No runs found")
        return 0

    print(f"\n{'Run ID':<40} {'Status':<12} {'Gates':<8} {'Started'}")
    print("-" * 80)

    for run in runs:
        run_id = run.get("run_id", "?")
        status = run.get("status", "?")
        gates = "PASS" if run.get("gates_passed") else "FAIL" if "gates_passed" in run else "-"
        started = run.get("started_at", "-")[:19] if run.get("started_at") else "-"
        print(f"{run_id:<40} {status:<12} {gates:<8} {started}")

    print(f"\nTotal: {len(runs)} runs")

    return 0


def show_command(args: argparse.Namespace) -> int:
    """Show run details."""
    from helix_engine import get_run

    run = get_run(args.run, runs_dir=args.runs_dir)

    if run is None:
        print(f"Run not found: {args.run}")
        return 1

    if args.json:
        print(json.dumps(run, indent=2))
    else:
        print(f"\n{'='*60}")
        print(f"Run: {run['run_id']}")
        print(f"{'='*60}")
        print(f"  Status:      {run.get('status', '?')}")
        print(f"  Exit Code:   {run.get('exit_code_name', '?')}")
        print(f"  Started:     {run.get('started_at', '?')}")
        print(f"  Completed:   {run.get('completed_at', '?')}")
        print(f"  Duration:    {run.get('duration_seconds', 0):.1f}s")
        print(f"  Steps:       {run.get('total_steps', 0)}")
        print(f"  Gates:       {'PASSED' if run.get('gates_passed') else 'FAILED'}")

        if run.get("final_metrics"):
            print(f"\n  Final Metrics:")
            for key, value in run["final_metrics"].items():
                if isinstance(value, float):
                    print(f"    {key}: {value:.6f}")
                else:
                    print(f"    {key}: {value}")

        if run.get("error_message"):
            print(f"\n  Error: {run['error_message']}")

        print(f"{'='*60}")

    return 0


def cli() -> argparse.ArgumentParser:
    """Create the CLI parser."""
    parser = argparse.ArgumentParser(
        prog="helix",
        description="Helix Training Engine - Productized training for Rosetta-Helix",
    )
    parser.add_argument("--version", action="version", version="helix-engine 0.1.0")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # train
    train_parser = subparsers.add_parser("train", help="Run training")
    train_parser.add_argument("--config", "-c", required=True, help="Path to config YAML")
    train_parser.add_argument("--steps", "-s", type=int, help="Override total steps")
    train_parser.add_argument("--seed", type=int, help="Override random seed")
    train_parser.add_argument("--output", "-o", help="Override output directory")

    # eval
    eval_parser = subparsers.add_parser("eval", help="Evaluate a run")
    eval_parser.add_argument("--run", "-r", required=True, help="Run ID to evaluate")
    eval_parser.add_argument("--runs-dir", default="runs", help="Runs directory")

    # resume
    resume_parser = subparsers.add_parser("resume", help="Resume training")
    resume_parser.add_argument("--run", "-r", required=True, help="Run ID to resume")
    resume_parser.add_argument("--checkpoint", help="Specific checkpoint to resume from")
    resume_parser.add_argument("--runs-dir", default="runs", help="Runs directory")

    # export
    export_parser = subparsers.add_parser("export", help="Export model")
    export_parser.add_argument("--run", "-r", required=True, help="Run ID to export")
    export_parser.add_argument("--format", "-f", default="bundle",
                               choices=["onnx", "torchscript", "pt", "bundle"],
                               help="Export format")
    export_parser.add_argument("--output", "-o", help="Output path")
    export_parser.add_argument("--runs-dir", default="runs", help="Runs directory")

    # promote
    promote_parser = subparsers.add_parser("promote", help="Promote run to named model")
    promote_parser.add_argument("--run", "-r", required=True, help="Run ID to promote")
    promote_parser.add_argument("--name", "-n", required=True, help="Model name")
    promote_parser.add_argument("--version", "-v", help="Version (auto-generated if not set)")
    promote_parser.add_argument("--tags", "-t", help="Comma-separated tags")
    promote_parser.add_argument("--description", "-d", help="Description")
    promote_parser.add_argument("--runs-dir", default="runs", help="Runs directory")

    # nightly
    nightly_parser = subparsers.add_parser("nightly", help="Run nightly training")
    nightly_parser.add_argument("--config", "-c", required=True, help="Path to config YAML")
    nightly_parser.add_argument("--report", help="Output path for nightly report JSON")

    # list
    list_parser = subparsers.add_parser("list", help="List runs")
    list_parser.add_argument("--status", choices=["pending", "running", "completed", "failed"],
                             help="Filter by status")
    list_parser.add_argument("--runs-dir", default="runs", help="Runs directory")

    # show
    show_parser = subparsers.add_parser("show", help="Show run details")
    show_parser.add_argument("--run", "-r", required=True, help="Run ID")
    show_parser.add_argument("--json", action="store_true", help="Output as JSON")
    show_parser.add_argument("--runs-dir", default="runs", help="Runs directory")

    return parser


def main(argv: Optional[list] = None) -> int:
    """Main entry point."""
    parser = cli()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    commands = {
        "train": train_command,
        "eval": eval_command,
        "resume": resume_command,
        "export": export_command,
        "promote": promote_command,
        "nightly": nightly_command,
        "list": list_command,
        "show": show_command,
    }

    handler = commands.get(args.command)
    if handler:
        try:
            return handler(args)
        except KeyboardInterrupt:
            print("\nInterrupted")
            return 130
        except Exception as e:
            print(f"Error: {e}")
            return 1

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
