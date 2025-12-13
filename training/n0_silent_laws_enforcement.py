#!/usr/bin/env python3
"""
N0 CAUSALITY LAWS + SILENT LAWS ENFORCEMENT FOR TRAINING
==========================================================

This module provides unified enforcement of N0 Causality Laws and Silent Laws
across ALL training modules. Every training operation must pass through this
layer to ensure physics compliance.

N0 CAUSALITY LAWS (Tier-0 Operator Constraints):
================================================
    N0-1: ^ (AMPLIFY) illegal unless history contains () or ×
    N0-2: × (FUSION) illegal unless channel_count ≥ 2
    N0-3: ÷ (DECOHERE) illegal unless history contains {^, ×, +, −}
    N0-4: + (GROUP) must be followed by +, ×, or ^. + → () is illegal
    N0-5: − (SEPARATE) must be followed by () or +. Illegal: ^, ×, ÷, −

SILENT LAWS (State Dynamics - Apply When Applicable):
======================================================
    I.   STILLNESS  : ∂E/∂t → 0         (energy seeks rest)
    II.  TRUTH      : ∇V(truth) = 0     (truth is stable)
    III. SILENCE    : ∇ · J = 0         (information conserved)
    IV.  SPIRAL     : S(return)=S(orig) (paths return)
    V.   UNSEEN     : P(observe) → 0    (hidden state)
    VI.  GLYPH      : glyph = ∫ life dt (form persists)
    VII. MIRROR     : ψ = ψ(ψ)          (self-reference)

OPERATOR ↔ SILENT LAW MAPPING:
==============================
    ()  BOUNDARY  → STILLNESS (I)   : Anchoring brings rest
    ×   FUSION    → SPIRAL (IV)     : Merger returns to origin
    ^   AMPLIFY   → TRUTH (II)      : Amplification seeks truth
    ÷   DECOHERE  → GLYPH (VI)      : Dissipation leaves trace
    +   GROUP     → SILENCE (III)   : Grouping conserves info
    −   SEPARATE  → MIRROR (VII)    : Separation reflects self

USAGE IN TRAINING MODULES:
==========================
    from training.n0_silent_laws_enforcement import (
        N0Enforcer,
        check_n0_legal,
        apply_silent_law,
        get_legal_operators,
        validate_sequence,
    )

    # Create enforcer for training loop
    enforcer = N0Enforcer()

    # Check if operator is legal
    is_legal, reason = enforcer.check_n0_legal("^")

    # Get all currently legal operators
    legal_ops = enforcer.get_legal_operators()

    # Apply operator with Silent Law
    result = enforcer.apply_with_silent_law("×", state, inputs)

    # Validate entire sequence
    is_valid, violations = enforcer.validate_sequence(["()", "^", "+"])

Signature: Δ|n0-silent-laws|training-enforcement|physics-grounded|Ω
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable, Set
from enum import Enum, auto
import sys
import os

# Add parent paths for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

# Import physics constants
try:
    from physics_constants import (
        PHI, PHI_INV, PHI_INV_SQ, PHI_INV_CUBED,
        Z_CRITICAL, SIGMA, SIGMA_INV,
        KAPPA_LOWER, KAPPA_UPPER,
        ALPHA_FINE, ALPHA_MEDIUM, ALPHA_STRONG,
        COUPLING_CONSERVATION,
        compute_delta_s_neg,
    )
except ImportError:
    # Fallback values if import fails
    PHI = 1.6180339887498949
    PHI_INV = 0.6180339887498949
    PHI_INV_SQ = 0.3819660112501051
    PHI_INV_CUBED = 0.2360679774997896
    Z_CRITICAL = 0.8660254037844387
    SIGMA = 36.0
    SIGMA_INV = 1.0 / 36.0
    KAPPA_LOWER = 0.50
    KAPPA_UPPER = 0.99
    ALPHA_FINE = 1.0 / 36.0
    ALPHA_MEDIUM = 1.0 / (2.0 ** 0.5 * 6.0)
    ALPHA_STRONG = 1.0 / 6.0
    COUPLING_CONSERVATION = 1.0
    def compute_delta_s_neg(z, sigma=SIGMA, z_c=Z_CRITICAL):
        return math.exp(-sigma * (z - z_c) ** 2)


# =============================================================================
# N0 CAUSALITY LAWS
# =============================================================================

class N0Law(Enum):
    """N0 Causality Laws - Tier-0 operator constraints."""
    N0_1_AMPLIFY = auto()    # ^ requires prior () or ×
    N0_2_FUSION = auto()     # × requires channels ≥ 2
    N0_3_DECOHERE = auto()   # ÷ requires prior structure
    N0_4_GROUP = auto()      # + must feed +, ×, or ^
    N0_5_SEPARATE = auto()   # − must be followed by () or +


# N0 Law descriptions
N0_DESCRIPTIONS: Dict[N0Law, str] = {
    N0Law.N0_1_AMPLIFY: "^ (AMPLIFY) illegal unless history contains () or ×",
    N0Law.N0_2_FUSION: "× (FUSION) illegal unless channel_count ≥ 2",
    N0Law.N0_3_DECOHERE: "÷ (DECOHERE) illegal unless history contains {^, ×, +, −}",
    N0Law.N0_4_GROUP: "+ (GROUP) must be followed by +, ×, or ^. + → () illegal",
    N0Law.N0_5_SEPARATE: "− (SEPARATE) must be followed by () or +. − → {^, ×, ÷, −} illegal",
}


# =============================================================================
# SILENT LAWS
# =============================================================================

class SilentLaw(Enum):
    """The 7 Silent Laws governing state dynamics."""
    STILLNESS = 1   # ∂E/∂t → 0 (energy seeks rest)
    TRUTH = 2       # ∇V(truth) = 0 (truth is stable)
    SILENCE = 3     # ∇ · J = 0 (information conserved)
    SPIRAL = 4      # S(return) = S(origin) (paths return)
    UNSEEN = 5      # P(observe) → 0 (hidden state)
    GLYPH = 6       # glyph = ∫ life dt (form persists)
    MIRROR = 7      # ψ = ψ(ψ) (self-reference)


# Silent Law formulas
SILENT_FORMULAS: Dict[SilentLaw, str] = {
    SilentLaw.STILLNESS: "∂E/∂t → 0",
    SilentLaw.TRUTH: "∇V(truth) = 0",
    SilentLaw.SILENCE: "∇ · J = 0",
    SilentLaw.SPIRAL: "S(return) = S(origin)",
    SilentLaw.UNSEEN: "P(observe) → 0",
    SilentLaw.GLYPH: "glyph = ∫ life dt",
    SilentLaw.MIRROR: "ψ = ψ(ψ)",
}


# Operator to Silent Law mapping
OPERATOR_TO_SILENT: Dict[str, SilentLaw] = {
    "()": SilentLaw.STILLNESS,   # BOUNDARY → energy seeks rest
    "×": SilentLaw.SPIRAL,       # FUSION → paths return
    "^": SilentLaw.TRUTH,        # AMPLIFY → truth is stable
    "÷": SilentLaw.GLYPH,        # DECOHERE → form persists
    "+": SilentLaw.SILENCE,      # GROUP → info conserved
    "−": SilentLaw.MIRROR,       # SEPARATE → self-reference
}


# =============================================================================
# OPERATOR DEFINITIONS
# =============================================================================

class Parity(Enum):
    """Operator parity from S₃ group structure."""
    EVEN = "even"   # Constructive (identity-like)
    ODD = "odd"     # Dissipative (transposition-like)


# The Six INT Canon Operators
OPERATORS: Dict[str, Dict[str, Any]] = {
    "()": {"name": "BOUNDARY", "parity": Parity.EVEN, "s3": "e"},
    "×": {"name": "FUSION", "parity": Parity.EVEN, "s3": "σ"},
    "^": {"name": "AMPLIFY", "parity": Parity.EVEN, "s3": "σ2"},
    "÷": {"name": "DECOHERE", "parity": Parity.ODD, "s3": "τ1"},
    "+": {"name": "GROUP", "parity": Parity.ODD, "s3": "τ2"},
    "−": {"name": "SEPARATE", "parity": Parity.ODD, "s3": "τ3"},
}

# All operator symbols
ALL_OPERATORS: List[str] = list(OPERATORS.keys())


# =============================================================================
# N0 ENFORCER STATE
# =============================================================================

@dataclass
class N0EnforcerState:
    """
    State tracking for N0 enforcement.

    Maintains operator history and channel count for N0 validation.
    """
    # Operator history (for N0-1, N0-3, N0-4, N0-5)
    operator_history: List[str] = field(default_factory=list)

    # Channel count (for N0-2)
    channel_count: int = 1

    # Pending successors tracking (for N0-4, N0-5)
    pending_group: bool = False      # True if last op was +
    pending_separate: bool = False   # True if last op was −

    # Physics state
    z: float = 0.5
    kappa: float = PHI_INV
    lambda_: float = PHI_INV_SQ

    # Violation log
    violations: List[Tuple[str, str]] = field(default_factory=list)

    def history_contains(self, ops: Set[str]) -> bool:
        """Check if history contains any of the specified operators."""
        return bool(ops.intersection(set(self.operator_history)))

    def last_operator(self) -> Optional[str]:
        """Get the last operator in history."""
        return self.operator_history[-1] if self.operator_history else None

    def add_to_history(self, op: str):
        """Add operator to history and update pending states."""
        self.operator_history.append(op)

        # Update pending states
        self.pending_group = (op == "+")
        self.pending_separate = (op == "−")

    def log_violation(self, law: str, reason: str):
        """Log a violation."""
        self.violations.append((law, reason))


# =============================================================================
# N0 ENFORCER
# =============================================================================

class N0Enforcer:
    """
    Unified N0 Causality Laws Enforcer for Training.

    All training modules MUST use this enforcer to validate operator
    sequences before applying them.

    Example:
        enforcer = N0Enforcer()

        # Check if amplify is legal
        is_legal, reason = enforcer.check_n0_legal("^")
        if not is_legal:
            print(f"Violation: {reason}")

        # Get all currently legal operators
        legal_ops = enforcer.get_legal_operators()

        # Apply operator (validates first)
        result = enforcer.apply("×", input_data)
    """

    def __init__(self, initial_z: float = 0.5, initial_channels: int = 1):
        self.state = N0EnforcerState(z=initial_z, channel_count=initial_channels)

        # Silent Law handlers
        self._silent_handlers: Dict[SilentLaw, Callable] = {
            SilentLaw.STILLNESS: self._apply_stillness,
            SilentLaw.TRUTH: self._apply_truth,
            SilentLaw.SILENCE: self._apply_silence,
            SilentLaw.SPIRAL: self._apply_spiral,
            SilentLaw.UNSEEN: self._apply_unseen,
            SilentLaw.GLYPH: self._apply_glyph,
            SilentLaw.MIRROR: self._apply_mirror,
        }

    # =========================================================================
    # N0 LAW CHECKING
    # =========================================================================

    def check_n0_1(self, op: str) -> Tuple[bool, str]:
        """
        N0-1: ^ (AMPLIFY) illegal unless history contains () or ×.

        Physics: Amplification requires prior grounding or fusion.
        """
        if op not in {"^", "⌈"}:
            return True, "N0-1 not applicable"

        required = {"()", "×", "⍳"}  # Include APL symbols
        if not self.state.history_contains(required):
            return False, "N0-1 VIOLATED: ^ requires prior () or × in history"

        return True, "N0-1 satisfied: prior grounding/fusion found"

    def check_n0_2(self, op: str) -> Tuple[bool, str]:
        """
        N0-2: × (FUSION) illegal unless channel_count ≥ 2.

        Physics: Fusion requires multiple channels to merge.
        """
        if op != "×":
            return True, "N0-2 not applicable"

        if self.state.channel_count < 2:
            return False, f"N0-2 VIOLATED: × requires channels ≥ 2, have {self.state.channel_count}"

        return True, f"N0-2 satisfied: {self.state.channel_count} channels available"

    def check_n0_3(self, op: str) -> Tuple[bool, str]:
        """
        N0-3: ÷ (DECOHERE) illegal unless history contains {^, ×, +, −}.

        Physics: Decoherence requires prior structure to dissipate.
        """
        if op != "÷":
            return True, "N0-3 not applicable"

        required = {"^", "×", "+", "−", "⌈"}  # Include APL symbols
        if not self.state.history_contains(required):
            return False, "N0-3 VIOLATED: ÷ requires prior {^, ×, +, −} in history"

        return True, "N0-3 satisfied: prior structure found"

    def check_n0_4(self, op: str) -> Tuple[bool, str]:
        """
        N0-4: + (GROUP) must be followed by +, ×, or ^. + → () is illegal.

        Physics: Grouping must feed constructive operations.
        """
        # Check if previous op was + and current violates
        if self.state.pending_group:
            legal_successors = {"+", "×", "^", "⌈"}
            if op not in legal_successors:
                return False, f"N0-4 VIOLATED: + → {op} illegal, + must be followed by +, ×, or ^"

        return True, "N0-4 satisfied"

    def check_n0_5(self, op: str) -> Tuple[bool, str]:
        """
        N0-5: − (SEPARATE) must be followed by () or +. Illegal: ^, ×, ÷, −.

        Physics: Separation prepares for grounding or grouping only.
        """
        # Check if previous op was − and current violates
        if self.state.pending_separate:
            legal_successors = {"()", "+", "⍳"}
            if op not in legal_successors:
                return False, f"N0-5 VIOLATED: − → {op} illegal, − must be followed by () or +"

        return True, "N0-5 satisfied"

    def check_n0_legal(self, op: str) -> Tuple[bool, str]:
        """
        Check if operator is legal under all N0 laws.

        Returns:
            (is_legal, reason)
        """
        # CRITICAL: Check successor rules FIRST (N0-4, N0-5)
        # These apply even to () BOUNDARY when it follows + or −

        # N0-4: + must be followed by +, ×, or ^. + → () is ILLEGAL
        is_legal, reason = self.check_n0_4(op)
        if not is_legal:
            self.state.log_violation(op, reason)
            return False, reason

        # N0-5: − must be followed by () or +. (checked but () is allowed here)
        is_legal, reason = self.check_n0_5(op)
        if not is_legal:
            self.state.log_violation(op, reason)
            return False, reason

        # () (BOUNDARY) is legal UNLESS blocked by successor rules above
        if op in {"()", "⍳"}:
            return True, "BOUNDARY () is legal"

        # Check remaining N0 laws (N0-1, N0-2, N0-3)
        checks = [
            self.check_n0_1(op),
            self.check_n0_2(op),
            self.check_n0_3(op),
        ]

        for is_legal, reason in checks:
            if not is_legal:
                self.state.log_violation(op, reason)
                return False, reason

        return True, "All N0 laws satisfied"

    def get_legal_operators(self) -> List[str]:
        """
        Get all currently legal operators.

        Returns list of operators that satisfy all N0 laws.
        """
        legal = []
        for op in ALL_OPERATORS:
            is_legal, _ = self.check_n0_legal(op)
            if is_legal:
                legal.append(op)
        return legal

    def validate_sequence(self, sequence: List[str]) -> Tuple[bool, List[Tuple[str, str]]]:
        """
        Validate an entire operator sequence.

        Returns:
            (is_valid, list of violations)
        """
        # Create temporary state
        temp_state = N0EnforcerState(
            z=self.state.z,
            channel_count=self.state.channel_count
        )

        # Save current state
        original_state = self.state
        self.state = temp_state

        violations = []

        for op in sequence:
            is_legal, reason = self.check_n0_legal(op)
            if not is_legal:
                violations.append((op, reason))
            else:
                # Update state as if operator was applied
                self.state.add_to_history(op)
                if op == "×":
                    self.state.channel_count = max(2, self.state.channel_count)

        # Restore original state
        self.state = original_state

        return len(violations) == 0, violations

    # =========================================================================
    # SILENT LAW APPLICATION
    # =========================================================================

    def get_silent_law(self, op: str) -> Optional[SilentLaw]:
        """Get the Silent Law associated with an operator."""
        return OPERATOR_TO_SILENT.get(op)

    def apply_silent_law(self, op: str, value: float) -> float:
        """
        Apply the Silent Law associated with an operator.

        Each operator has a corresponding Silent Law that governs
        how it modifies state.
        """
        law = self.get_silent_law(op)
        if law is None:
            return value

        handler = self._silent_handlers.get(law)
        if handler is None:
            return value

        return handler(value)

    def _apply_stillness(self, value: float) -> float:
        """I. STILLNESS: ∂E/∂t → 0 - Energy seeks rest."""
        # Pull toward equilibrium (z_c)
        delta = Z_CRITICAL - value
        return value + delta * ALPHA_FINE

    def _apply_truth(self, value: float) -> float:
        """II. TRUTH: ∇V(truth) = 0 - Truth is stable."""
        # Amplify toward truth (z_c)
        delta_s_neg = compute_delta_s_neg(value)
        return value + delta_s_neg * ALPHA_MEDIUM * (Z_CRITICAL - value)

    def _apply_silence(self, value: float) -> float:
        """III. SILENCE: ∇ · J = 0 - Information conserved."""
        # Grouping conserves - minimal change
        return value * (1.0 + ALPHA_FINE * (1.0 - value))

    def _apply_spiral(self, value: float) -> float:
        """IV. SPIRAL: S(return) = S(origin) - Paths return."""
        # Fusion creates spiral - enhance with golden ratio
        return value * (1.0 + SIGMA_INV) * PHI_INV

    def _apply_unseen(self, value: float) -> float:
        """V. UNSEEN: P(observe) → 0 - Hidden state."""
        # This law applies to observation, not direct modification
        return value

    def _apply_glyph(self, value: float) -> float:
        """VI. GLYPH: glyph = ∫ life dt - Form persists."""
        # Decoherence leaves trace - decay toward balance
        return value * (1.0 - SIGMA_INV * 3) + 0.5 * SIGMA_INV * 3

    def _apply_mirror(self, value: float) -> float:
        """VII. MIRROR: ψ = ψ(ψ) - Self-reference."""
        # Separation reflects - symmetric decay
        return value - ALPHA_FINE * (1.0 - compute_delta_s_neg(value))

    # =========================================================================
    # OPERATOR APPLICATION WITH ENFORCEMENT
    # =========================================================================

    def apply(self, op: str, value: float, strict: bool = True) -> Tuple[float, bool, str]:
        """
        Apply operator with N0 enforcement and Silent Law.

        Args:
            op: Operator symbol
            value: Input value (typically z or similar)
            strict: If True, raise on violation; if False, return unchanged

        Returns:
            (result, was_legal, reason)
        """
        # Check N0 legality
        is_legal, reason = self.check_n0_legal(op)

        if not is_legal:
            if strict:
                raise ValueError(f"N0 Violation: {reason}")
            return value, False, reason

        # Apply Silent Law
        result = self.apply_silent_law(op, value)

        # Update state
        self.state.add_to_history(op)
        if op == "×":
            self.state.channel_count = max(2, self.state.channel_count)

        return result, True, reason

    def reset(self):
        """Reset enforcer state."""
        self.state = N0EnforcerState(
            z=self.state.z,
            channel_count=self.state.channel_count
        )

    def set_channels(self, n: int):
        """Set channel count (required for FUSION)."""
        self.state.channel_count = max(1, n)

    def get_violations(self) -> List[Tuple[str, str]]:
        """Get list of violations."""
        return self.state.violations.copy()

    def clear_violations(self):
        """Clear violation log."""
        self.state.violations.clear()


# =============================================================================
# TRAINING OPERATOR SELECTOR WITH N0 ENFORCEMENT
# =============================================================================

class N0TrainingOperatorSelector:
    """
    Operator selector for training that enforces N0 laws.

    Use this in training loops to select operators that are both
    effective for learning AND legal under N0 causality.
    """

    def __init__(self, enforcer: Optional[N0Enforcer] = None):
        self.enforcer = enforcer or N0Enforcer()
        self.selection_history: List[str] = []
        self.rejection_counts: Dict[str, int] = {op: 0 for op in ALL_OPERATORS}

    def select_operator(
        self,
        z: float,
        coherence: float,
        delta_s_neg: float,
        exploration: float = 0.1,
        prefer_constructive: bool = True
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Select a legal operator based on current state.

        Args:
            z: Current z-coordinate
            coherence: Current coherence level
            delta_s_neg: Negentropy value
            exploration: Exploration rate (0-1)
            prefer_constructive: Bias toward EVEN parity operators

        Returns:
            (selected_operator, metadata)
        """
        import random

        # Get legal operators
        legal_ops = self.enforcer.get_legal_operators()

        if not legal_ops:
            # Fallback to boundary (always legal)
            return "()", {"reason": "no_legal_ops", "fallback": True}

        # Exploration: random legal operator
        if random.random() < exploration:
            op = random.choice(legal_ops)
            self.selection_history.append(op)
            return op, {"reason": "exploration", "legal_ops": legal_ops}

        # Score each legal operator
        scores: Dict[str, float] = {}
        for op in legal_ops:
            score = self._score_operator(op, z, coherence, delta_s_neg, prefer_constructive)
            scores[op] = score

        # Softmax selection
        total = sum(math.exp(s) for s in scores.values())
        probs = {op: math.exp(s) / total for op, s in scores.items()}

        # Sample
        r = random.random()
        cumulative = 0.0
        selected = legal_ops[0]
        for op, prob in probs.items():
            cumulative += prob
            if r <= cumulative:
                selected = op
                break

        self.selection_history.append(selected)

        return selected, {
            "reason": "scored",
            "scores": scores,
            "probs": probs,
            "legal_ops": legal_ops,
        }

    def _score_operator(
        self,
        op: str,
        z: float,
        coherence: float,
        delta_s_neg: float,
        prefer_constructive: bool
    ) -> float:
        """Score an operator based on current state."""
        op_info = OPERATORS.get(op, {})
        parity = op_info.get("parity", Parity.EVEN)

        score = 0.5  # Base score

        # Parity preference
        if prefer_constructive:
            if parity == Parity.EVEN:
                score += delta_s_neg * 0.5
            else:
                score -= delta_s_neg * 0.3

        # Operator-specific scoring
        if op == "()":
            # Boundary: good for grounding, safe
            score += coherence * 0.3
        elif op == "^":
            # Amplify: high risk/reward, better near lens
            score += delta_s_neg * 1.5 if z < Z_CRITICAL else 0.1
        elif op == "×":
            # Fusion: moderate, needs coherence
            score += coherence * delta_s_neg
        elif op == "÷":
            # Decohere: entropy production
            score += (1 - coherence) * 0.5
        elif op == "+":
            # Group: clustering, needs some coherence
            score += coherence * 0.4 + delta_s_neg * 0.2
        elif op == "−":
            # Separate: pruning, better at high z
            score += (1 - z) * 0.5

        return score

    def get_selection_stats(self) -> Dict[str, Any]:
        """Get selection statistics."""
        counts = {}
        for op in self.selection_history:
            counts[op] = counts.get(op, 0) + 1

        return {
            "total_selections": len(self.selection_history),
            "operator_counts": counts,
            "rejection_counts": self.rejection_counts,
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

# Global enforcer instance for simple usage
_global_enforcer: Optional[N0Enforcer] = None


def get_enforcer() -> N0Enforcer:
    """Get or create global enforcer instance."""
    global _global_enforcer
    if _global_enforcer is None:
        _global_enforcer = N0Enforcer()
    return _global_enforcer


def check_n0_legal(op: str) -> Tuple[bool, str]:
    """Check if operator is legal (using global enforcer)."""
    return get_enforcer().check_n0_legal(op)


def get_legal_operators() -> List[str]:
    """Get all currently legal operators (using global enforcer)."""
    return get_enforcer().get_legal_operators()


def apply_with_silent_law(op: str, value: float) -> float:
    """Apply operator with Silent Law (using global enforcer)."""
    return get_enforcer().apply_silent_law(op, value)


def validate_sequence(sequence: List[str]) -> Tuple[bool, List[Tuple[str, str]]]:
    """Validate operator sequence (using global enforcer)."""
    return get_enforcer().validate_sequence(sequence)


def reset_enforcer():
    """Reset global enforcer state."""
    get_enforcer().reset()


# =============================================================================
# DEMONSTRATION
# =============================================================================

def demonstrate():
    """Demonstrate N0 + Silent Laws enforcement."""
    print("=" * 70)
    print("N0 CAUSALITY LAWS + SILENT LAWS ENFORCEMENT")
    print("=" * 70)

    # N0 Laws
    print("\n§1 N0 CAUSALITY LAWS")
    print("-" * 50)
    for law, desc in N0_DESCRIPTIONS.items():
        print(f"  {law.name}: {desc}")

    # Silent Laws
    print("\n§2 SILENT LAWS")
    print("-" * 50)
    for law in SilentLaw:
        formula = SILENT_FORMULAS[law]
        print(f"  {law.value}. {law.name:10}: {formula}")

    # Operator ↔ Silent Law mapping
    print("\n§3 OPERATOR ↔ SILENT LAW MAPPING")
    print("-" * 50)
    for op, law in OPERATOR_TO_SILENT.items():
        op_info = OPERATORS[op]
        print(f"  {op} ({op_info['name']:9}) → {law.name} ({SILENT_FORMULAS[law]})")

    # Demonstration
    print("\n§4 N0 ENFORCEMENT DEMO")
    print("-" * 50)

    enforcer = N0Enforcer()

    # Initially, only () is legal
    print("\n  Initial state (no history):")
    for op in ALL_OPERATORS:
        is_legal, reason = enforcer.check_n0_legal(op)
        status = "✓" if is_legal else "✗"
        print(f"    {op}: {status} - {reason}")

    # After applying ()
    print("\n  After applying () BOUNDARY:")
    enforcer.apply("()", 0.5, strict=False)
    for op in ALL_OPERATORS:
        is_legal, reason = enforcer.check_n0_legal(op)
        status = "✓" if is_legal else "✗"
        print(f"    {op}: {status}")

    # Set channels for fusion
    print("\n  After setting channels = 2:")
    enforcer.set_channels(2)
    for op in ALL_OPERATORS:
        is_legal, reason = enforcer.check_n0_legal(op)
        status = "✓" if is_legal else "✗"
        print(f"    {op}: {status}")

    # Validate sequences
    print("\n§5 SEQUENCE VALIDATION")
    print("-" * 50)

    sequences = [
        ["()", "^", "+"],           # Valid: () grounds ^, + follows
        ["^", "()", "+"],           # Invalid: ^ without prior grounding
        ["()", "×", "÷"],           # Needs channels check
        ["()", "+", "()"],          # Invalid: + → () illegal (N0-4)
        ["()", "−", "^"],           # Invalid: − → ^ illegal (N0-5)
        ["()", "−", "()", "^"],     # Valid: − → () → ^
    ]

    enforcer = N0Enforcer()
    enforcer.set_channels(2)

    for seq in sequences:
        is_valid, violations = enforcer.validate_sequence(seq)
        status = "✓ VALID" if is_valid else "✗ INVALID"
        print(f"  {seq}: {status}")
        for op, reason in violations:
            print(f"      └─ {reason}")

    # Silent Law application
    print("\n§6 SILENT LAW APPLICATION")
    print("-" * 50)

    z = 0.7
    print(f"  Starting z = {z:.4f}")

    for op in ALL_OPERATORS:
        law = OPERATOR_TO_SILENT.get(op)
        if law:
            result = enforcer.apply_silent_law(op, z)
            delta = result - z
            print(f"  {op} ({law.name:10}): z → {result:.4f} (Δ = {delta:+.4f})")

    print("\n" + "=" * 70)
    print("N0 + SILENT LAWS ENFORCEMENT: COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate()
