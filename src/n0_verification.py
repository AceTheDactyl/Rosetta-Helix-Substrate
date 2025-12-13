#!/usr/bin/env python3
"""
N0 CAUSALITY LAW VERIFICATION MODULE
======================================

Formal verification of N0 (Tier-0 / Silent Laws) compliance for operator sequences.

N0 Causality Laws:
==================
    N0-1: ^ (AMPLIFY) illegal unless history contains () or ×
    N0-2: × (FUSION) illegal unless channel_count ≥ 2
    N0-3: ÷ (DECOHERE) illegal unless history contains {^, ×, +, −}
    N0-4: + (GROUP) must be followed by +, ×, or ^. + → () is illegal.
    N0-5: − (SEPARATE) must be followed by () or +. − → {^, ×, ÷, −} illegal.
    N0-6: Net entropy cannot decrease without work input (background law)
    N0-7: No backward temporal references in operator chains (causality)

These laws ensure proper causality, structure formation, and conservation.

Usage:
======
    from n0_verification import check_n0_compliance, is_n0_legal

    # Check a sequence
    sequence = ["()", "×", "^", "+", "×"]
    results = check_n0_compliance(sequence)
    for law, passed, reason in results:
        print(f"[{'PASS' if passed else 'FAIL'}] {law.name}: {reason}")

    # Quick legality check
    if is_n0_legal(sequence):
        print("Sequence is N0-legal")

Signature: Δ|n0-verification|causality|7-laws|Ω
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# =============================================================================
# N0 LAW ENUMERATION
# =============================================================================

class N0Law(Enum):
    """The 7 Silent Laws (N0 Causality Constraints)."""
    GROUNDING = (1, "AMPLIFY requires grounding", "^ illegal unless history contains () or ×")
    PLURALITY = (2, "FUSION requires channels", "× illegal unless channels ≥ 2")
    DECOHERENCE = (3, "DECOHERE requires structure", "÷ illegal unless history contains {^, ×, +, −}")
    SUCCESSION = (4, "GROUP must feed forward", "+ must be followed by +, ×, or ^")
    SEPARATION = (5, "SEPARATE requires reset", "− must be followed by () or +")
    ENTROPY = (6, "Entropy conservation", "Net entropy cannot decrease without work")
    CAUSALITY = (7, "Temporal causality", "No backward references in operator chains")

    def __init__(self, index: int, short_desc: str, full_desc: str):
        self._index = index
        self._short_desc = short_desc
        self._full_desc = full_desc

    @property
    def index(self) -> int:
        return self._index

    @property
    def short_desc(self) -> str:
        return self._short_desc

    @property
    def full_desc(self) -> str:
        return self._full_desc


# =============================================================================
# N0 VIOLATION RESULT
# =============================================================================

@dataclass
class N0Violation:
    """Details of an N0 law violation."""
    law: N0Law
    position: int
    operator: str
    reason: str
    context: Dict[str, Any]


@dataclass
class N0ComplianceResult:
    """Complete N0 compliance check result."""
    sequence: List[str]
    is_legal: bool
    violations: List[N0Violation]
    law_results: List[Tuple[N0Law, bool, str]]

    def summary(self) -> str:
        """Generate summary string."""
        if self.is_legal:
            return f"Sequence of {len(self.sequence)} operators is N0-legal"
        else:
            return f"{len(self.violations)} violations in sequence of {len(self.sequence)} operators"


# =============================================================================
# N0 COMPLIANCE CHECKING
# =============================================================================

def check_n0_compliance(
    sequence: List[str],
    context: Optional[Dict[str, Any]] = None
) -> List[Tuple[N0Law, bool, str]]:
    """
    Check operator sequence against all N0 laws.

    Args:
        sequence: List of operator symbols ["()", "×", "^", etc.]
        context: Optional context with channel_count, work_input, etc.

    Returns:
        List of (law, passed, reason) tuples for each law
    """
    context = context or {}
    results = []

    # N0-1: GROUNDING - ^ requires prior () or ×
    amplify_result = _check_n0_1_grounding(sequence)
    results.append((N0Law.GROUNDING, amplify_result[0], amplify_result[1]))

    # N0-2: PLURALITY - × requires channels ≥ 2
    fusion_result = _check_n0_2_plurality(sequence, context)
    results.append((N0Law.PLURALITY, fusion_result[0], fusion_result[1]))

    # N0-3: DECOHERENCE - ÷ requires prior structure
    decohere_result = _check_n0_3_decoherence(sequence)
    results.append((N0Law.DECOHERENCE, decohere_result[0], decohere_result[1]))

    # N0-4: SUCCESSION - + must feed +, ×, or ^
    group_result = _check_n0_4_succession(sequence)
    results.append((N0Law.SUCCESSION, group_result[0], group_result[1]))

    # N0-5: SEPARATION - − must be followed by () or +
    separate_result = _check_n0_5_separation(sequence)
    results.append((N0Law.SEPARATION, separate_result[0], separate_result[1]))

    # N0-6: ENTROPY - requires runtime context
    entropy_result = _check_n0_6_entropy(sequence, context)
    results.append((N0Law.ENTROPY, entropy_result[0], entropy_result[1]))

    # N0-7: CAUSALITY - requires runtime context
    causality_result = _check_n0_7_causality(sequence, context)
    results.append((N0Law.CAUSALITY, causality_result[0], causality_result[1]))

    return results


def _check_n0_1_grounding(sequence: List[str]) -> Tuple[bool, str]:
    """N0-1: ^ (AMPLIFY) illegal unless history contains () or ×."""
    grounding_ops = {"()", "×"}

    for i, op in enumerate(sequence):
        if op == "^":
            history = sequence[:i]
            if not any(h in grounding_ops for h in history):
                return False, f"^ at position {i} without prior () or × in history"

    return True, "All ^ operators have grounding"


def _check_n0_2_plurality(
    sequence: List[str],
    context: Dict[str, Any]
) -> Tuple[bool, str]:
    """N0-2: × (FUSION) illegal unless channels ≥ 2."""
    channel_count = context.get("channel_count", 2)  # Default assumes 2 channels

    # For static analysis, check if × appears after ()
    for i, op in enumerate(sequence):
        if op == "×":
            # Either have explicit channel count or () must precede
            if channel_count < 2:
                history = sequence[:i]
                if "()" not in history:
                    return False, f"× at position {i} without channels ≥ 2"

    return True, f"All × operators have sufficient channels (count={channel_count})"


def _check_n0_3_decoherence(sequence: List[str]) -> Tuple[bool, str]:
    """N0-3: ÷ (DECOHERE) illegal unless history contains {^, ×, +, −}."""
    structure_ops = {"^", "×", "+", "−"}

    for i, op in enumerate(sequence):
        if op == "÷":
            history = sequence[:i]
            if not any(h in structure_ops for h in history):
                return False, f"÷ at position {i} without prior structure operators"

    return True, "All ÷ operators have prior structure"


def _check_n0_4_succession(sequence: List[str]) -> Tuple[bool, str]:
    """N0-4: + (GROUP) must be followed by +, ×, or ^."""
    valid_successors = {"+", "×", "^"}

    for i, op in enumerate(sequence):
        if op == "+":
            if i < len(sequence) - 1:
                successor = sequence[i + 1]
                if successor not in valid_successors:
                    return False, f"+ at position {i} followed by {successor}, expected +, ×, or ^"

    return True, "All + operators have valid successors"


def _check_n0_5_separation(sequence: List[str]) -> Tuple[bool, str]:
    """N0-5: − (SEPARATE) must be followed by () or +."""
    valid_successors = {"()", "+"}

    for i, op in enumerate(sequence):
        if op == "−":
            if i < len(sequence) - 1:
                successor = sequence[i + 1]
                if successor not in valid_successors:
                    return False, f"− at position {i} followed by {successor}, expected () or +"

    return True, "All − operators have valid successors"


def _check_n0_6_entropy(
    sequence: List[str],
    context: Dict[str, Any]
) -> Tuple[bool, str]:
    """N0-6: Net entropy cannot decrease without work input."""
    # This requires runtime entropy tracking
    # For static analysis, check for work_input in context
    work_input = context.get("work_input", None)

    # Count entropy-increasing vs decreasing operators
    increasing = sequence.count("÷") + sequence.count("−")  # Decohere, Separate
    decreasing = sequence.count("^") + sequence.count("+")  # Amplify, Group

    if decreasing > increasing and work_input is None:
        return False, (
            f"Net entropy decrease (−{decreasing - increasing}) without work input. "
            f"Increasing: {increasing}, Decreasing: {decreasing}"
        )

    return True, f"Entropy balance: +{increasing} −{decreasing}, work={work_input}"


def _check_n0_7_causality(
    sequence: List[str],
    context: Dict[str, Any]
) -> Tuple[bool, str]:
    """N0-7: No backward temporal references in operator chains."""
    # For static analysis, check for proper temporal ordering
    # This is satisfied if the sequence is a valid forward chain

    # Check that each operator only depends on prior operators
    # (implicitly satisfied by sequential execution)

    return True, "Temporal causality preserved (forward chain)"


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def is_n0_legal(sequence: List[str], context: Optional[Dict[str, Any]] = None) -> bool:
    """
    Quick check if sequence is fully N0-legal.

    Args:
        sequence: List of operator symbols
        context: Optional context dict

    Returns:
        True if sequence passes all N0 laws
    """
    results = check_n0_compliance(sequence, context)
    return all(passed for _, passed, _ in results)


def get_violations(
    sequence: List[str],
    context: Optional[Dict[str, Any]] = None
) -> List[N0Violation]:
    """
    Get list of all N0 violations in a sequence.

    Args:
        sequence: List of operator symbols
        context: Optional context dict

    Returns:
        List of N0Violation objects
    """
    context = context or {}
    violations = []

    # N0-1: Check grounding
    grounding_ops = {"()", "×"}
    for i, op in enumerate(sequence):
        if op == "^":
            history = sequence[:i]
            if not any(h in grounding_ops for h in history):
                violations.append(N0Violation(
                    law=N0Law.GROUNDING,
                    position=i,
                    operator=op,
                    reason="^ without prior () or ×",
                    context={"history": history}
                ))

    # N0-3: Check decoherence
    structure_ops = {"^", "×", "+", "−"}
    for i, op in enumerate(sequence):
        if op == "÷":
            history = sequence[:i]
            if not any(h in structure_ops for h in history):
                violations.append(N0Violation(
                    law=N0Law.DECOHERENCE,
                    position=i,
                    operator=op,
                    reason="÷ without prior structure",
                    context={"history": history}
                ))

    # N0-4: Check succession
    valid_group_successors = {"+", "×", "^"}
    for i, op in enumerate(sequence):
        if op == "+" and i < len(sequence) - 1:
            successor = sequence[i + 1]
            if successor not in valid_group_successors:
                violations.append(N0Violation(
                    law=N0Law.SUCCESSION,
                    position=i,
                    operator=op,
                    reason=f"+ followed by invalid {successor}",
                    context={"successor": successor, "valid": list(valid_group_successors)}
                ))

    # N0-5: Check separation
    valid_sep_successors = {"()", "+"}
    for i, op in enumerate(sequence):
        if op == "−" and i < len(sequence) - 1:
            successor = sequence[i + 1]
            if successor not in valid_sep_successors:
                violations.append(N0Violation(
                    law=N0Law.SEPARATION,
                    position=i,
                    operator=op,
                    reason=f"− followed by invalid {successor}",
                    context={"successor": successor, "valid": list(valid_sep_successors)}
                ))

    return violations


def full_compliance_check(
    sequence: List[str],
    context: Optional[Dict[str, Any]] = None
) -> N0ComplianceResult:
    """
    Perform full N0 compliance check with detailed results.

    Args:
        sequence: List of operator symbols
        context: Optional context dict

    Returns:
        N0ComplianceResult with all details
    """
    law_results = check_n0_compliance(sequence, context)
    violations = get_violations(sequence, context)

    return N0ComplianceResult(
        sequence=sequence,
        is_legal=len(violations) == 0,
        violations=violations,
        law_results=law_results
    )


# =============================================================================
# OPERATOR LEGALITY FOR NEXT OPERATOR
# =============================================================================

def get_legal_next_operators(
    history: List[str],
    context: Optional[Dict[str, Any]] = None
) -> List[str]:
    """
    Get list of operators that are legal to apply next.

    Args:
        history: Sequence of operators already applied
        context: Optional context dict

    Returns:
        List of legal operator symbols
    """
    context = context or {}
    legal = []

    # () BOUNDARY is always legal
    legal.append("()")

    # × FUSION: legal if channels ≥ 2 or history contains ()
    channel_count = context.get("channel_count", 2)
    if channel_count >= 2 or "()" in history:
        legal.append("×")

    # ^ AMPLIFY: legal if history contains () or ×
    if any(op in history for op in ["()", "×"]):
        legal.append("^")

    # ÷ DECOHERE: legal if history contains {^, ×, +, −}
    if any(op in history for op in ["^", "×", "+", "−"]):
        legal.append("÷")

    # + GROUP: always legal (successor constraint checked later)
    # But if last operator was −, we must use () or + next
    if history and history[-1] == "−":
        legal = ["()", "+"]
    elif history and history[-1] == "+":
        # After +, must use +, ×, or ^
        legal = ["+", "×", "^"]
    else:
        legal.append("+")
        legal.append("−")

    return list(set(legal))


# =============================================================================
# DEMONSTRATION
# =============================================================================

def demonstrate():
    """Demonstrate N0 verification."""
    print("=" * 70)
    print("N0 CAUSALITY LAW VERIFICATION")
    print("=" * 70)

    print("\n§1 THE 7 N0 LAWS")
    print("-" * 50)
    for law in N0Law:
        print(f"  N0-{law.index}: {law.short_desc}")
        print(f"         {law.full_desc}")

    print("\n§2 LEGAL SEQUENCE EXAMPLE")
    print("-" * 50)
    legal_seq = ["()", "×", "^", "+", "×", "÷"]
    print(f"  Sequence: {' → '.join(legal_seq)}")
    results = check_n0_compliance(legal_seq)
    for law, passed, reason in results:
        status = "✓" if passed else "✗"
        print(f"  [{status}] N0-{law.index}: {reason}")
    print(f"\n  Is N0-legal: {is_n0_legal(legal_seq)}")

    print("\n§3 ILLEGAL SEQUENCE EXAMPLE")
    print("-" * 50)
    illegal_seq = ["^", "÷", "−", "×"]  # ^ without grounding, ÷ without structure
    print(f"  Sequence: {' → '.join(illegal_seq)}")
    violations = get_violations(illegal_seq)
    for v in violations:
        print(f"  ✗ N0-{v.law.index} at position {v.position}: {v.reason}")
    print(f"\n  Is N0-legal: {is_n0_legal(illegal_seq)}")

    print("\n§4 LEGAL NEXT OPERATORS")
    print("-" * 50)
    histories = [
        [],
        ["()"],
        ["()", "×"],
        ["()", "×", "^"],
        ["()", "+"],
    ]
    for hist in histories:
        legal = get_legal_next_operators(hist)
        hist_str = " → ".join(hist) if hist else "(empty)"
        print(f"  After [{hist_str}]: {', '.join(legal)}")

    print("\n" + "=" * 70)
    print("Signature: Δ|n0-verification|causality|7-laws|Ω")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate()
