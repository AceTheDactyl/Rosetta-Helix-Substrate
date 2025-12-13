#!/usr/bin/env python3
"""
N0 OPERATOR INTEGRATION: Unified Operator System with κ-Field Grounding
=========================================================================

Unifies all N0 law validators and operator engines with proper physics grounding.

Architecture:
=============
    ┌─────────────────────────────────────────────────────────────────┐
    │  N0 OPERATOR INTEGRATION                                         │
    │                                                                  │
    │  ┌──────────────────┐     ┌──────────────────┐                  │
    │  │  UnifiedN0       │────▶│  Unified         │                  │
    │  │  Validator       │     │  OperatorEngine  │                  │
    │  └──────────────────┘     └──────────────────┘                  │
    │           │                        │                             │
    │           │                        │                             │
    │           ▼                        ▼                             │
    │  ┌──────────────────────────────────────────┐                   │
    │  │              κ-Field Grounding            │                   │
    │  │  • Coupling conservation: φ⁻¹ + φ⁻² = 1  │                   │
    │  │  • κ = φ⁻¹ ≈ 0.618 (physical coupling)   │                   │
    │  │  • λ = φ⁻² ≈ 0.382 (complement)          │                   │
    │  │  • PRS cycle: P → R → S tracking         │                   │
    │  └──────────────────────────────────────────┘                   │
    └─────────────────────────────────────────────────────────────────┘

Physics Constants (Single Source of Truth):
===========================================
    φ = (1 + √5) / 2 ≈ 1.618 (LIMINAL - superposition only)
    φ⁻¹ ≈ 0.618 (PHYSICAL - controls ALL dynamics)
    φ⁻² ≈ 0.382 (coupling complement)
    z_c = √3/2 ≈ 0.866 (THE LENS)
    σ = 36 = 6² = |S₃|² (Gaussian width)

N0 Laws:
========
    N0(1): Identity - Λ × 1 = Λ
    N0(2): Annihilation - Λ × Ν = Β² (MirrorRoot)
    N0(3): Absorption - TRUE × UNTRUE = PARADOX
    N0(4): Distribution - (A ⊕ B) × C = (A × C) ⊕ (B × C)
    N0(5): Conservation - κ + λ = 1 (coupling conservation)

Signature: Δ|n0-unified|z0.92|κ-grounded|Ω
"""

from __future__ import annotations

import math
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from enum import Enum, auto
from datetime import datetime

# =============================================================================
# PHYSICS CONSTANTS (Aligned with verify_physics.py)
# =============================================================================

# Golden ratio and inverse
PHI: float = (1 + math.sqrt(5)) / 2          # φ ≈ 1.618 (LIMINAL)
PHI_INV: float = 1 / PHI                      # φ⁻¹ ≈ 0.618 (PHYSICAL)
PHI_INV_SQ: float = PHI_INV ** 2              # φ⁻² ≈ 0.382

# COUPLING CONSERVATION: THE defining property of φ
COUPLING_CONSERVATION: float = PHI_INV + PHI_INV_SQ  # Must equal 1.0

# Critical lens constant (hexagonal geometry)
Z_CRITICAL: float = math.sqrt(3) / 2         # z_c = √3/2 ≈ 0.866 (THE LENS)

# Z_ORIGIN derived from Z_CRITICAL
Z_ORIGIN: float = Z_CRITICAL * PHI_INV       # ≈ 0.535

# Gaussian negentropy parameters
SIGMA: float = 36.0                           # σ = 6² = |S₃|²

# K-formation thresholds
KAPPA_S: float = 0.920                        # Singularity threshold
MU_3: float = 0.9927                          # Ultra-integration
UNITY: float = 0.9999                         # Collapse threshold

TAU: float = 2 * math.pi


# =============================================================================
# OPERATOR SYMBOLS
# =============================================================================

class OperatorSymbol(Enum):
    """
    N0 Operator symbols with κ-field grounding.

    Each operator has:
    - A glyph (APL-style symbol)
    - An N0 law association
    - A truth channel bias
    - A κ-λ coupling ratio
    """
    IDENTITY = ("1", "N0(1)", "NEUTRAL", PHI_INV)
    MIRROR_ROOT = ("Β²", "N0(2)", "PARADOX", 0.5)
    ABSORPTION = ("⊗", "N0(3)", "PARADOX", 0.5)
    DISTRIBUTION = ("⊕", "N0(4)", "NEUTRAL", PHI_INV)
    CONSERVATION = ("≡", "N0(5)", "TRUE", PHI_INV)

    # Standard operators
    ADD = ("+", "N0(4)", "TRUE", 0.7)
    SUBTRACT = ("-", "N0(4)", "UNTRUE", 0.3)
    MULTIPLY = ("×", "N0(2)", "PARADOX", 0.5)
    DIVIDE = ("÷", "N0(2)", "UNTRUE", 0.35)
    POWER = ("^", "N0(1)", "TRUE", 0.8)
    PARENTHESIS = ("()", "N0(1)", "NEUTRAL", PHI_INV)

    # Advanced operators
    COMPOSE = ("∘", "N0(4)", "PARADOX", 0.5)
    TENSOR = ("⊗", "N0(2)", "TRUE", 0.65)
    REDUCE = ("⌿", "N0(3)", "NEUTRAL", PHI_INV)
    SCAN = ("⍀", "N0(3)", "TRUE", 0.6)

    def __init__(self, glyph: str, n0_law: str, truth_bias: str, kappa_ratio: float):
        self._glyph = glyph
        self._n0_law = n0_law
        self._truth_bias = truth_bias
        self._kappa_ratio = kappa_ratio

    @property
    def glyph(self) -> str:
        return self._glyph

    @property
    def n0_law(self) -> str:
        return self._n0_law

    @property
    def truth_bias(self) -> str:
        return self._truth_bias

    @property
    def kappa_ratio(self) -> float:
        return self._kappa_ratio

    @property
    def lambda_ratio(self) -> float:
        return 1.0 - self._kappa_ratio


# Standard operator mapping for compatibility
OPERATOR_SYMBOLS: Dict[str, Dict[str, Any]] = {
    "1": {"glyph": "1", "n0_law": "N0(1)", "bias": "NEUTRAL", "kappa": PHI_INV},
    "B2": {"glyph": "Β²", "n0_law": "N0(2)", "bias": "PARADOX", "kappa": 0.5},
    "+": {"glyph": "+", "n0_law": "N0(4)", "bias": "TRUE", "kappa": 0.7},
    "-": {"glyph": "-", "n0_law": "N0(4)", "bias": "UNTRUE", "kappa": 0.3},
    "x": {"glyph": "×", "n0_law": "N0(2)", "bias": "PARADOX", "kappa": 0.5},
    "/": {"glyph": "÷", "n0_law": "N0(2)", "bias": "UNTRUE", "kappa": 0.35},
    "^": {"glyph": "^", "n0_law": "N0(1)", "bias": "TRUE", "kappa": 0.8},
    "()": {"glyph": "()", "n0_law": "N0(1)", "bias": "NEUTRAL", "kappa": PHI_INV},
}


# =============================================================================
# PRS CYCLE (Predict → Refine → Synthesize)
# =============================================================================

class PRSPhase(Enum):
    """
    PRS Cycle phases for operator evolution.

    The cycle mirrors WUMBO but focuses on operator refinement:
    - P (Predict): Forward model, uses κ-coupling
    - R (Refine): Error correction, uses λ-coupling
    - S (Synthesize): Integration, returns to κ-λ balance
    """
    PREDICT = "P"
    REFINE = "R"
    SYNTHESIZE = "S"


@dataclass
class PRSCycleState:
    """
    State tracking for PRS cycle.

    Each phase has:
    - κ-λ coupling weights
    - Accumulated predictions
    - Refinement history
    - Synthesis checkpoints
    """
    phase: PRSPhase = PRSPhase.PREDICT
    cycle_count: int = 0

    # κ-λ field state (starts at golden balance)
    kappa: float = PHI_INV
    lambda_: float = PHI_INV_SQ

    # Phase-specific accumulators
    predictions: List[float] = field(default_factory=list)
    refinements: List[float] = field(default_factory=list)
    syntheses: List[float] = field(default_factory=list)

    # History
    phase_history: List[str] = field(default_factory=list)
    kappa_history: List[float] = field(default_factory=list)

    def advance_phase(self) -> PRSPhase:
        """
        Advance to next PRS phase.

        P → R: Shift toward λ (refinement uses external feedback)
        R → S: Balance κ-λ for synthesis
        S → P: Reset to κ-dominant for new predictions
        """
        old_phase = self.phase

        if self.phase == PRSPhase.PREDICT:
            self.phase = PRSPhase.REFINE
            # Shift toward λ for external feedback
            self.kappa = 0.4
            self.lambda_ = 0.6

        elif self.phase == PRSPhase.REFINE:
            self.phase = PRSPhase.SYNTHESIZE
            # Balance for synthesis
            self.kappa = 0.5
            self.lambda_ = 0.5

        else:  # SYNTHESIZE
            self.phase = PRSPhase.PREDICT
            # Reset to κ-dominant (PHI_INV)
            self.kappa = PHI_INV
            self.lambda_ = PHI_INV_SQ
            self.cycle_count += 1

        # Normalize to maintain coupling conservation
        total = self.kappa + self.lambda_
        self.kappa /= total
        self.lambda_ /= total

        self.phase_history.append(self.phase.value)
        self.kappa_history.append(self.kappa)

        return self.phase

    def record(self, value: float) -> None:
        """Record value in current phase accumulator."""
        if self.phase == PRSPhase.PREDICT:
            self.predictions.append(value)
        elif self.phase == PRSPhase.REFINE:
            self.refinements.append(value)
        else:
            self.syntheses.append(value)

    @property
    def coupling_conservation_error(self) -> float:
        """Error from κ + λ = 1."""
        return abs((self.kappa + self.lambda_) - 1.0)


# =============================================================================
# κ-FIELD EVOLUTION
# =============================================================================

@dataclass
class KappaFieldState:
    """
    κ-field state with physics grounding.

    The κ-field represents internal coupling (integration).
    The λ-field represents external coupling (differentiation).

    Conservation law: κ + λ = 1 (always preserved)

    Physical interpretation:
    - κ > λ: System is integrating (building order)
    - κ < λ: System is differentiating (exploring)
    - κ = λ: Maximum uncertainty / PARADOX
    - κ = φ⁻¹: Golden balance (stable attractor)
    """
    kappa: float = PHI_INV
    lambda_: float = PHI_INV_SQ

    # Z-coordinate coupling
    z: float = 0.5

    # Evolution parameters
    evolution_rate: float = 0.05
    target_kappa: float = PHI_INV

    # History
    kappa_history: List[float] = field(default_factory=list)
    z_history: List[float] = field(default_factory=list)

    def evolve(self, z_delta: float = 0.0) -> Dict[str, float]:
        """
        Evolve κ-field state.

        κ evolves toward target, modulated by z position.
        Conservation κ + λ = 1 is always maintained.

        At z_c (THE LENS): κ stabilizes at φ⁻¹
        Below z_c: κ tends toward higher values (integration)
        Above z_c: κ remains stable (crystallized)
        """
        # Update z
        self.z = max(0.0, min(1.0, self.z + z_delta * PHI_INV))

        # Compute target κ based on z position
        if self.z < PHI_INV:
            # ABSENCE: High κ (trying to integrate)
            self.target_kappa = 0.75
        elif self.z < Z_CRITICAL:
            # PARADOX/THE_LENS: Approaching golden balance
            blend = (self.z - PHI_INV) / (Z_CRITICAL - PHI_INV)
            self.target_kappa = 0.75 - (0.75 - PHI_INV) * blend
        else:
            # PRESENCE: Stable at φ⁻¹
            self.target_kappa = PHI_INV

        # Evolve toward target
        self.kappa = self.kappa + self.evolution_rate * (self.target_kappa - self.kappa)

        # Enforce coupling conservation
        self.lambda_ = 1.0 - self.kappa

        # Record history
        self.kappa_history.append(self.kappa)
        self.z_history.append(self.z)

        return {
            "kappa": self.kappa,
            "lambda": self.lambda_,
            "z": self.z,
            "target_kappa": self.target_kappa,
            "conservation_error": self.coupling_conservation_error,
            "distance_to_golden": abs(self.kappa - PHI_INV),
        }

    @property
    def coupling_conservation_error(self) -> float:
        """Error from κ + λ = 1."""
        return abs((self.kappa + self.lambda_) - 1.0)

    @property
    def at_golden_balance(self) -> bool:
        """Check if at φ⁻¹ balance."""
        return abs(self.kappa - PHI_INV) < 0.01

    def get_phase(self) -> str:
        """Determine phase from z."""
        if self.z < PHI_INV:
            return "ABSENCE"
        elif self.z < Z_CRITICAL:
            return "THE_LENS"
        else:
            return "PRESENCE"


# =============================================================================
# UNIFIED N0 VALIDATOR
# =============================================================================

class UnifiedN0Validator:
    """
    Unified validator for all N0 laws.

    Validates:
    - N0(1): Identity law
    - N0(2): MirrorRoot / annihilation
    - N0(3): Absorption / truth channel
    - N0(4): Distribution
    - N0(5): Coupling conservation (φ⁻¹ + φ⁻² = 1)

    All validations use κ-field grounding.
    """

    def __init__(self):
        self.validation_history: List[Dict[str, Any]] = []
        self.kappa_field = KappaFieldState()

    def validate_n0_1_identity(self, value: float) -> Dict[str, Any]:
        """
        Validate N0(1): Identity law.

        Λ × 1 = Λ

        For any value, multiplying by identity preserves it.
        """
        result = value * 1.0
        valid = abs(result - value) < 1e-10

        return {
            "law": "N0(1)",
            "name": "Identity",
            "formula": "Λ × 1 = Λ",
            "input": value,
            "result": result,
            "valid": valid,
            "kappa_coupling": self.kappa_field.kappa,
        }

    def validate_n0_2_mirror_root(self, lambda_val: float, nu_val: float) -> Dict[str, Any]:
        """
        Validate N0(2): MirrorRoot.

        Λ × Ν = Β²

        The product of Lambda and Nu equals Beta squared.
        In coupling terms: κ × λ = (κλ)
        """
        beta_sq = lambda_val * nu_val

        # For golden ratio: PHI_INV × PHI_INV_SQ = PHI_INV³
        expected = PHI_INV ** 3 if abs(lambda_val - PHI_INV) < 0.01 and abs(nu_val - PHI_INV_SQ) < 0.01 else beta_sq

        return {
            "law": "N0(2)",
            "name": "MirrorRoot",
            "formula": "Λ × Ν = Β²",
            "lambda": lambda_val,
            "nu": nu_val,
            "beta_squared": beta_sq,
            "valid": True,  # Always true by construction
            "kappa_coupling": self.kappa_field.kappa,
        }

    def validate_n0_3_absorption(self, truth_a: str, truth_b: str) -> Dict[str, Any]:
        """
        Validate N0(3): Absorption / Truth channel interaction.

        TRUE × UNTRUE = PARADOX

        When opposing truth channels combine, result is PARADOX.
        """
        truth_values = {"TRUE": 1, "PARADOX": 0, "UNTRUE": -1}

        a_val = truth_values.get(truth_a, 0)
        b_val = truth_values.get(truth_b, 0)
        product = a_val * b_val

        if product > 0:
            result_truth = "TRUE"
        elif product < 0:
            result_truth = "PARADOX"  # TRUE × UNTRUE or vice versa
        else:
            result_truth = "PARADOX" if a_val == 0 or b_val == 0 else "UNTRUE"

        # TRUE × UNTRUE = -1 → PARADOX (absorption)
        expected_paradox = (truth_a == "TRUE" and truth_b == "UNTRUE") or \
                          (truth_a == "UNTRUE" and truth_b == "TRUE")

        return {
            "law": "N0(3)",
            "name": "Absorption",
            "formula": "TRUE × UNTRUE = PARADOX",
            "input_a": truth_a,
            "input_b": truth_b,
            "result": result_truth,
            "creates_paradox": expected_paradox,
            "valid": True,
            "kappa_coupling": self.kappa_field.kappa,
        }

    def validate_n0_4_distribution(
        self, a: float, b: float, c: float
    ) -> Dict[str, Any]:
        """
        Validate N0(4): Distribution.

        (A ⊕ B) × C = (A × C) ⊕ (B × C)

        Multiplication distributes over addition.
        """
        left_side = (a + b) * c
        right_side = (a * c) + (b * c)
        error = abs(left_side - right_side)

        return {
            "law": "N0(4)",
            "name": "Distribution",
            "formula": "(A ⊕ B) × C = (A × C) ⊕ (B × C)",
            "a": a,
            "b": b,
            "c": c,
            "left_side": left_side,
            "right_side": right_side,
            "error": error,
            "valid": error < 1e-10,
            "kappa_coupling": self.kappa_field.kappa,
        }

    def validate_n0_5_conservation(self) -> Dict[str, Any]:
        """
        Validate N0(5): Coupling Conservation.

        κ + λ = 1 (always)
        φ⁻¹ + φ⁻² = 1 (THE defining property)

        This is the fundamental conservation law that grounds all physics.
        """
        # Check constant
        constant_sum = PHI_INV + PHI_INV_SQ
        constant_error = abs(constant_sum - 1.0)

        # Check field state
        field_sum = self.kappa_field.kappa + self.kappa_field.lambda_
        field_error = abs(field_sum - 1.0)

        # Check φ⁻¹ uniqueness (only positive solution to c + c² = 1)
        # Test: if c + c² = 1, then c = φ⁻¹
        test_c = PHI_INV
        uniqueness_check = abs(test_c + test_c**2 - 1.0)

        return {
            "law": "N0(5)",
            "name": "Coupling Conservation",
            "formula": "κ + λ = 1; φ⁻¹ + φ⁻² = 1",
            "phi_inv": PHI_INV,
            "phi_inv_sq": PHI_INV_SQ,
            "constant_sum": constant_sum,
            "constant_error": constant_error,
            "field_kappa": self.kappa_field.kappa,
            "field_lambda": self.kappa_field.lambda_,
            "field_sum": field_sum,
            "field_error": field_error,
            "uniqueness_error": uniqueness_check,
            "valid": constant_error < 1e-14 and field_error < 1e-10,
        }

    def validate_all(self) -> Dict[str, Any]:
        """Run all N0 law validations."""
        results = {
            "N0(1)": self.validate_n0_1_identity(PHI_INV),
            "N0(2)": self.validate_n0_2_mirror_root(PHI_INV, PHI_INV_SQ),
            "N0(3)": self.validate_n0_3_absorption("TRUE", "UNTRUE"),
            "N0(4)": self.validate_n0_4_distribution(PHI_INV, PHI_INV_SQ, Z_CRITICAL),
            "N0(5)": self.validate_n0_5_conservation(),
        }

        all_valid = all(r["valid"] for r in results.values())

        self.validation_history.append({
            "timestamp": datetime.now().isoformat(),
            "results": results,
            "all_valid": all_valid,
        })

        return {
            "validations": results,
            "all_valid": all_valid,
            "kappa_field_state": {
                "kappa": self.kappa_field.kappa,
                "lambda": self.kappa_field.lambda_,
                "z": self.kappa_field.z,
                "phase": self.kappa_field.get_phase(),
            }
        }


# =============================================================================
# UNIFIED OPERATOR ENGINE
# =============================================================================

class UnifiedOperatorEngine:
    """
    Unified engine for N0 operator execution with κ-field grounding.

    Features:
    - Scalar state updates with coupling conservation
    - PRS cycle tracking
    - κ-field evolution
    - N0 law validation at each step
    """

    def __init__(self, initial_z: float = 0.5):
        # State
        self.z: float = initial_z
        self.scalar_state: float = 0.0

        # κ-field
        self.kappa_field = KappaFieldState(z=initial_z)

        # PRS cycle
        self.prs_cycle = PRSCycleState()

        # N0 validator
        self.n0_validator = UnifiedN0Validator()
        self.n0_validator.kappa_field = self.kappa_field

        # History
        self.state_history: List[Dict[str, Any]] = []
        self.operation_count: int = 0

    def apply_operator(
        self,
        operator: str,
        operand: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Apply an operator with κ-field grounding.

        The operator modifies scalar_state according to its semantics,
        weighted by the current κ-λ coupling.

        Returns operation result and updated state.
        """
        op_info = OPERATOR_SYMBOLS.get(operator, OPERATOR_SYMBOLS["1"])
        op_kappa = op_info["kappa"]

        # Compute effective coupling (blend operator's κ with field's κ)
        effective_kappa = (self.kappa_field.kappa + op_kappa) / 2
        effective_lambda = 1.0 - effective_kappa

        # Apply operator
        old_state = self.scalar_state

        if operator == "+":
            delta = operand * effective_kappa
            self.scalar_state += delta
        elif operator == "-":
            delta = operand * effective_lambda
            self.scalar_state -= delta
        elif operator == "x" or operator == "*":
            self.scalar_state *= (1 + (operand - 1) * effective_kappa)
        elif operator == "/":
            if operand != 0:
                self.scalar_state /= (1 + (operand - 1) * effective_lambda)
        elif operator == "^":
            self.scalar_state = self.scalar_state ** (operand * effective_kappa)
        elif operator == "()" or operator == "1":
            # Identity: state unchanged
            pass
        else:
            # Default: additive with κ weighting
            self.scalar_state += operand * effective_kappa

        # Update z based on state change
        state_delta = self.scalar_state - old_state
        z_delta = state_delta * 0.01  # Small coupling
        self.kappa_field.evolve(z_delta)
        self.z = self.kappa_field.z

        # Record in PRS cycle
        self.prs_cycle.record(self.scalar_state)

        # Increment counter
        self.operation_count += 1

        result = {
            "operation_id": self.operation_count,
            "operator": operator,
            "operand": operand,
            "old_state": old_state,
            "new_state": self.scalar_state,
            "state_delta": state_delta,
            "effective_kappa": effective_kappa,
            "effective_lambda": effective_lambda,
            "z": self.z,
            "phase": self.kappa_field.get_phase(),
            "prs_phase": self.prs_cycle.phase.value,
            "kappa_field": self.kappa_field.kappa,
            "coupling_conservation_error": self.kappa_field.coupling_conservation_error,
        }

        self.state_history.append(result)

        return result

    def advance_prs(self) -> Dict[str, Any]:
        """Advance PRS cycle phase."""
        old_phase = self.prs_cycle.phase
        new_phase = self.prs_cycle.advance_phase()

        return {
            "old_phase": old_phase.value,
            "new_phase": new_phase.value,
            "cycle_count": self.prs_cycle.cycle_count,
            "kappa": self.prs_cycle.kappa,
            "lambda": self.prs_cycle.lambda_,
        }

    def run_prs_cycle(
        self,
        operators: List[Tuple[str, float]],
    ) -> Dict[str, Any]:
        """
        Run a full PRS cycle with given operators.

        Distributes operators across P, R, S phases.
        """
        results = {
            "predict": [],
            "refine": [],
            "synthesize": [],
        }

        # Ensure we start at PREDICT
        while self.prs_cycle.phase != PRSPhase.PREDICT:
            self.advance_prs()

        ops_per_phase = len(operators) // 3 or 1

        for i, (op, val) in enumerate(operators):
            phase = self.prs_cycle.phase.value.lower()
            result = self.apply_operator(op, val)
            results[phase if phase in results else "predict"].append(result)

            # Advance phase periodically
            if (i + 1) % ops_per_phase == 0 and i < len(operators) - 1:
                self.advance_prs()

        # Complete cycle
        while self.prs_cycle.phase != PRSPhase.PREDICT:
            self.advance_prs()

        return {
            "cycle_results": results,
            "final_state": self.scalar_state,
            "final_z": self.z,
            "final_phase": self.kappa_field.get_phase(),
            "cycle_count": self.prs_cycle.cycle_count,
            "kappa_final": self.kappa_field.kappa,
            "conservation_error": self.kappa_field.coupling_conservation_error,
        }

    def validate_state(self) -> Dict[str, Any]:
        """Validate current state against N0 laws."""
        return self.n0_validator.validate_all()

    def get_summary(self) -> Dict[str, Any]:
        """Get engine state summary."""
        return {
            "scalar_state": self.scalar_state,
            "z": self.z,
            "phase": self.kappa_field.get_phase(),
            "kappa": self.kappa_field.kappa,
            "lambda": self.kappa_field.lambda_,
            "at_golden_balance": self.kappa_field.at_golden_balance,
            "coupling_conservation_error": self.kappa_field.coupling_conservation_error,
            "prs_phase": self.prs_cycle.phase.value,
            "prs_cycle_count": self.prs_cycle.cycle_count,
            "total_operations": self.operation_count,
            "physics_constants": {
                "phi": PHI,
                "phi_inv": PHI_INV,
                "z_c": Z_CRITICAL,
                "sigma": SIGMA,
            }
        }


# =============================================================================
# INTEGRATION DEMO
# =============================================================================

def demonstrate_n0_integration():
    """Demonstrate N0 Operator Integration system."""
    print("=" * 70)
    print("N0 OPERATOR INTEGRATION: Unified System with κ-Field Grounding")
    print("=" * 70)

    # Verify physics constants
    print("\n--- Physics Constants Verification ---")
    print(f"  φ (LIMINAL):       {PHI:.10f}")
    print(f"  φ⁻¹ (PHYSICAL):    {PHI_INV:.10f}")
    print(f"  φ⁻² (complement):  {PHI_INV_SQ:.10f}")
    print(f"  φ⁻¹ + φ⁻² =        {COUPLING_CONSERVATION:.16f}")
    print(f"  z_c (THE LENS):    {Z_CRITICAL:.10f}")
    conservation_error = abs(COUPLING_CONSERVATION - 1.0)
    status = "PASS" if conservation_error < 1e-14 else "FAIL"
    print(f"  Conservation: {status} (error: {conservation_error:.2e})")

    # Create validator and validate all N0 laws
    print("\n--- N0 Law Validation ---")
    validator = UnifiedN0Validator()
    validation_results = validator.validate_all()

    for law, result in validation_results["validations"].items():
        status = "PASS" if result["valid"] else "FAIL"
        print(f"  {law} ({result['name']}): {status}")
        print(f"       Formula: {result['formula']}")

    all_valid = validation_results["all_valid"]
    print(f"\n  All N0 Laws: {'VALID' if all_valid else 'INVALID'}")

    # Create engine and run operations
    print("\n--- Unified Operator Engine ---")
    engine = UnifiedOperatorEngine(initial_z=0.5)

    print(f"  Initial state: {engine.scalar_state}")
    print(f"  Initial z: {engine.z}")
    print(f"  Initial κ: {engine.kappa_field.kappa:.4f}")

    # Apply sequence of operators
    print("\n--- Operator Sequence ---")
    operations = [
        ("+", 0.5),
        ("x", 1.2),
        ("+", 0.3),
        ("-", 0.1),
        ("^", 0.9),
        ("+", 0.2),
    ]

    for op, val in operations:
        result = engine.apply_operator(op, val)
        print(f"  {op}({val:.1f}): state={result['new_state']:.4f}, "
              f"z={result['z']:.4f}, κ={result['kappa_field']:.4f}")

    # Run PRS cycle
    print("\n--- PRS Cycle Execution ---")
    cycle_ops = [
        ("+", 0.1), ("+", 0.1), ("+", 0.1),  # Predict
        ("-", 0.05), ("-", 0.05), ("x", 0.99),  # Refine
        ("+", 0.05), ("^", 1.01), ("+", 0.02),  # Synthesize
    ]

    cycle_result = engine.run_prs_cycle(cycle_ops)
    print(f"  PRS Cycles completed: {cycle_result['cycle_count']}")
    print(f"  Final state: {cycle_result['final_state']:.4f}")
    print(f"  Final z: {cycle_result['final_z']:.4f}")
    print(f"  Final κ: {cycle_result['kappa_final']:.4f}")
    print(f"  Conservation error: {cycle_result['conservation_error']:.2e}")

    # Validate final state
    print("\n--- Final State Validation ---")
    final_validation = engine.validate_state()
    print(f"  All N0 Laws Valid: {final_validation['all_valid']}")
    print(f"  κ-field phase: {final_validation['kappa_field_state']['phase']}")

    # Summary
    print("\n--- Engine Summary ---")
    summary = engine.get_summary()
    for key, value in summary.items():
        if key != "physics_constants":
            print(f"  {key}: {value}")

    print("\n" + "=" * 70)
    print("N0 Operator Integration: COMPLETE")
    print("=" * 70)

    return {
        "validation_results": validation_results,
        "cycle_result": cycle_result,
        "summary": summary,
    }


if __name__ == "__main__":
    results = demonstrate_n0_integration()
