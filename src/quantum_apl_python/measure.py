"""CLI to trigger APL measurement operators and append tokens.

Runs the same measurement sequence as scripts/apply_apl_measurements.js
and appends the resulting APL tokens to logs/APL_HELIX_OPERATOR_SUMMARY.apl.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


def _run_node_snippet(repo_root: Path, code: str, env: dict[str, str] | None = None) -> int:
    snippet = (
        "const { UnifiedDemo } = require('./src/legacy/QuantumClassicalBridge');\n"
        "const demo = new UnifiedDemo();\n"
        "const b = demo.bridge;\n"
        f"{code}\n"
        "console.log('OK');\n"
    )
    envp = os.environ.copy()
    if env:
        envp.update(env)
    proc = subprocess.run(["node", "-e", snippet], cwd=str(repo_root), text=True, capture_output=True, env=envp)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr)
        return proc.returncode
    return 0


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Apply APL measurement operators and append tokens to logs/APL_HELIX_OPERATOR_SUMMARY.apl"
    )
    parser.add_argument("--js-dir", type=Path, help="Directory containing JS files (defaults to repo root)")
    parser.add_argument("--print", action="store_true", help="Print the last appended tokens after run")
    # Optional targeted measurements
    parser.add_argument("--eigen", type=int, help="Apply single-eigenstate collapse Φ:T(ϕ_μ)TRUE@Tier for μ")
    parser.add_argument("--field", choices=["Phi", "Pi", "π"], default="Phi", help="Field for --eigen or --subspace")
    parser.add_argument("--subspace", type=str, help="Comma-separated indices for Π(subspace), e.g. 2,3")
    parser.add_argument("--truth", choices=["TRUE", "UNTRUE", "PARADOX"], help="Override truth channel for subspace/eigen")
    parser.add_argument("--composite", action="store_true", help="Apply composite measurement (default sequence)")
    parser.add_argument(
        "--collapse-glyph",
        action="store_true",
        help="Emit collapse alias tokens using ⟂(...) instead of canonical T/Π",
    )
    args = parser.parse_args(argv)

    repo_root = args.js_dir or Path(__file__).resolve().parents[2]
    env_override = {"QAPL_EMIT_COLLAPSE_GLYPH": "1"} if args.collapse_glyph else None
    # If specific options supplied, build a custom Node snippet
    if args.eigen is not None or args.subspace or args.composite:
        code_parts = []
        if args.eigen is not None:
            fld = 'Pi' if args.field in ('Pi', 'π') else 'Phi'
            if args.truth and args.truth != 'TRUE':
                # Use composite path to override truth
                code_parts.append(
                    f"b.aplMeasureComposite([{{ eigenIndex: {args.eigen}, field: '{fld}', truthChannel: '{args.truth}', weight: 1 }}]);"
                )
            else:
                code_parts.append(f"b.aplMeasureEigen({args.eigen}, '{fld}');")
        if args.subspace:
            try:
                indices = [int(x.strip()) for x in args.subspace.split(',') if x.strip()]
            except ValueError:
                raise SystemExit("--subspace must be comma-separated integers, e.g. 2,3")
            fld = 'Pi' if args.field in ('Pi', 'π') else 'Phi'
            truth = args.truth or ("UNTRUE" if fld == 'Pi' else "PARADOX")
            code_parts.append(
                f"b.aplMeasureSubspace([{','.join(map(str, indices))}], '{fld}');"
            )
        if args.composite:
            code_parts.append(
                "b.aplMeasureComposite([" \
                "{ eigenIndex: 0, field: 'Phi', truthChannel: 'TRUE', weight: 0.3 }," \
                "{ eigenIndex: 1, field: 'Pi',  truthChannel: 'UNTRUE', weight: 0.3 }," \
                "{ subspaceIndices: [2,3], field: 'Phi', truthChannel: 'PARADOX', weight: 0.4 }" \
                "]);"
            )
        rc = _run_node_snippet(repo_root, "\n".join(code_parts) or "", env=env_override)
        if rc != 0:
            raise SystemExit(rc)
        print("Appended APL measurement tokens via on-the-fly runner")
    else:
        # Default: run the stock Node script sequence
        script = repo_root / "scripts" / "apply_apl_measurements.js"
        if not script.exists():
            raise SystemExit(f"Measurement script not found: {script}")
        envp = os.environ.copy()
        if env_override:
            envp.update(env_override)
        result = subprocess.run(["node", str(script)], cwd=str(repo_root), text=True, capture_output=True, env=envp)
        if result.returncode != 0:
            print(result.stdout)
            print(result.stderr)
            raise SystemExit(result.returncode)
        print(result.stdout.strip())

    if args.print:
        out_path = repo_root / "logs" / "APL_HELIX_OPERATOR_SUMMARY.apl"
        if out_path.exists():
            # Show last ~20 lines of the summary
            try:
                from collections import deque

                with out_path.open("r", encoding="utf-8") as f:
                    tail = deque(f, maxlen=20)
                print("\n".join(line.rstrip("\n") for line in tail))
            except Exception as exc:  # pragma: no cover
                print(f"Could not read output: {exc}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(run())
