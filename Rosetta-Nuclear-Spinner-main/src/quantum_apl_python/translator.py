"""Quantum APL token translator/validator.

This module parses Quantum-APL operator sentences of the form:
    Φ:M(stabilize)PARADOX@2
and emits structured dictionaries describing each instruction.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Sequence

TRUTH_STATES = {"TRUE", "UNTRUE", "PARADOX"}


@dataclass
class QuantumAPLInstruction:
    """Structured representation of a Quantum-APL line."""

    subject: str
    operator: str
    intent: str
    truth: str
    tier: int

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["truth"] = payload["truth"].upper()
        return payload

    def to_line(self) -> str:
        intent = self.intent or ""
        return f"{self.subject}:{self.operator}({intent}){self.truth}@{self.tier}"


INSTRUCTION_RE = re.compile(
    r"""
    ^\s*
    (?P<subject>[^\s:]+)          # Φ, Φ→e, (Φ,e,π), Truth, etc.
    \s*:\s*
    (?P<operator>\(\)|[^\s(]+)    # Operator token e.g. M, Mod, () , z^
    \(
        (?P<intent>[^)]*)         # Intent text (can be empty)
    \)
    (?P<truth>TRUE|UNTRUE|PARADOX)
    @
    (?P<tier>\d+)
    \s*$
    """,
    re.VERBOSE,
)


def parse_instruction(line: str) -> QuantumAPLInstruction:
    """Parse a single Quantum-APL instruction line."""

    if not line or not line.strip():
        raise ValueError("Empty instruction line")

    match = INSTRUCTION_RE.match(line)
    if not match:
        raise ValueError(f"Invalid Quantum-APL syntax: {line!r}")

    data = match.groupdict()
    truth = data["truth"].upper()
    if truth not in TRUTH_STATES:
        raise ValueError(f"Unknown truth state '{truth}' in {line!r}")

    return QuantumAPLInstruction(
        subject=data["subject"],
        operator=data["operator"],
        intent=data["intent"].strip(),
        truth=truth,
        tier=int(data["tier"]),
    )


def translate_lines(lines: Sequence[str]) -> List[QuantumAPLInstruction]:
    """Translate multiple instruction lines into structured entries."""

    instructions: List[QuantumAPLInstruction] = []
    for raw in lines:
        striped = raw.strip()
        if not striped:
            continue
        instructions.append(parse_instruction(striped))
    return instructions


def load_source(path: Path | None, text: str | None) -> Iterable[str]:
    if path:
        return path.read_text(encoding="utf-8").splitlines()
    if text:
        return text.strip().splitlines()
    raise ValueError("Provide either --file or --text to translate.")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Translate Quantum-APL operator sentences.")
    parser.add_argument("--file", type=Path, help="Path to a file containing Quantum-APL lines.")
    parser.add_argument(
        "--text",
        type=str,
        help="Inline Quantum-APL text (use quotes). Useful for quick tests.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output.",
    )
    args = parser.parse_args(argv)

    try:
        lines = load_source(args.file, args.text)
        instructions = translate_lines(list(lines))
    except Exception as exc:  # pragma: no cover - CLI guardrail
        parser.error(str(exc))

    payload = [instr.to_dict() for instr in instructions]
    json_output = json.dumps(payload, indent=2 if args.pretty else None)
    print(json_output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
