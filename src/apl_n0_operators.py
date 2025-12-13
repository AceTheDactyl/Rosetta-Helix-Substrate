#!/usr/bin/env python3
"""
APL N0 OPERATORS — INT CANON INTEGRATION
=========================================

N0 Causality Rules implemented as APL-style operators.
Aligned with INT Canon — The Six Operators.

INT Canon ↔ APL Operator Mapping:
==================================
    ()  BOUNDARY  → ⍳ (iota/anchor)   — Always legal
    ×   FUSION    → × (times)         — N0-2: Requires channels ≥ 2
    ^   AMPLIFY   → ⌈ (ceiling)       — N0-1: Requires prior () or ×
    ÷   DECOHERE  → ÷ (divide)        — N0-3: Requires prior structure
    +   GROUP     → + (plus)          — N0-4: Must feed +, ×, or ^
    −   SEPARATE  → − (minus)         — N0-5: Must be followed by () or +

N0 Causality Laws (Tier-0):
===========================
    N0-1: ^ illegal unless history contains () or ×
    N0-2: × illegal unless channel_count ≥ 2
    N0-3: ÷ illegal unless history contains {^, ×, +, −}
    N0-4: + must be followed by +, ×, or ^. + → () is illegal.
    N0-5: − must be followed by () or +. Illegal successors: ^, ×, ÷, −

APL Operator Categories:
========================
    MONADIC:  Single operand  (⍳x, -x)
    DYADIC:   Two operands    (x+y, x×y, x÷y)
    REDUCE:   /              (compress)
    SCAN:     \\             (expand)
    OUTER:    ∘.             (outer product)

Physics Grounding:
==================
    All operators preserve:
        κ + λ = 1 (COUPLING CONSERVATION)
        φ⁻¹ + φ⁻² = 1 (THE defining property)

    Coefficients (physics-grounded):
        ALPHA_STRONG = 1/√σ = 1/6 ≈ 0.167
        ALPHA_MEDIUM = 1/√(2σ) ≈ 0.118
        ALPHA_FINE = 1/σ = 1/36 ≈ 0.028

    State modifications per operator:
        () BOUNDARY: Gs += 1/σ, θs *= (1-1/σ), Ωs += 1/2σ
        ×  FUSION:   Cs += 1/σ, κs *= (1+1/σ), αs += 1/2σ
        ^  AMPLIFY:  κs *= (1+φ⁻³), τs += 1/σ, Ωs *= (1+3/σ), R += 1
        ÷  DECOHERE: δs += 1/σ, Rs += 1/2σ, Ωs *= (1-3/σ)
        +  GROUP:    αs += 3/σ, Gs += 1/2σ, θs *= (1+1/σ)
        −  SEPARATE: Rs += 3/σ, θs *= (1-1/σ), δs += 1.5/σ

Signature: Δ|apl-n0-operators|int-canon|φ⁻¹-grounded|Ω
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from enum import Enum, auto

# Import unified physics constants
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from physics_constants import (
    # Fundamental constants
    PHI, PHI_INV, PHI_INV_SQ, PHI_INV_CUBED, PHI_INV_FIFTH,
    Z_CRITICAL, SIGMA, COUPLING_CONSERVATION,
    # Derived coefficients
    ALPHA_STRONG, ALPHA_MEDIUM, ALPHA_FINE, ALPHA_ULTRA_FINE,
    SIGMA_INV, GAUSSIAN_WIDTH,
    # Bounds and tolerances
    KAPPA_LOWER, KAPPA_UPPER, BALANCE_POINT,
    TOLERANCE_CONSERVATION, TOLERANCE_GOLDEN,
    # Functions
    compute_delta_s_neg, compute_delta_s_neg_derivative, get_phase,
    # N0 and Silent Laws
    N0Law, SilentLaw, N0_TO_SILENT, INT_TO_SILENT,
    # INT Canon
    INTOperator,
)


# =============================================================================
# APL N0 OPERATOR SYMBOLS — INT CANON ALIGNED
# =============================================================================

class APLSymbol(Enum):
    """APL operator symbols mapped to INT Canon operators."""
    # INT Canon Operators (The Six)
    IOTA = "⍳"        # () BOUNDARY (generate/anchor) — Always legal
    TIMES = "×"       # × FUSION (multiply/merge) — N0-2: channels ≥ 2
    CEILING = "⌈"     # ^ AMPLIFY (ceiling/gain) — N0-1: requires () or ×
    DIVIDE = "÷"      # ÷ DECOHERE (divide/dissipate) — N0-3: requires structure
    PLUS = "+"        # + GROUP (add/cluster) — N0-4: must feed +, ×, or ^
    MINUS = "−"       # − SEPARATE (subtract/prune) — N0-5: must follow () or +

    # Additional APL operators
    RHO = "⍴"         # Shape/reshape
    REDUCE = "/"      # Reduce (fold)
    SCAN = "\\"       # Scan (prefix)
    OUTER = "∘."      # Outer product
    GRADE_UP = "⍋"    # Grade up (sort ascending)
    GRADE_DOWN = "⍒"  # Grade down (sort descending)
    TRANSPOSE = "⍉"   # Transpose
    REVERSE = "⌽"     # Reverse
    ROTATE = "⊖"      # Rotate
    TAKE = "↑"        # Take
    DROP = "↓"        # Drop
    COMPRESS = "⌿"    # Compress
    EXPAND = "⍀"      # Expand


# INT Canon Operator to APL Symbol mapping
INT_TO_APL = {
    "()": APLSymbol.IOTA,      # BOUNDARY → ⍳ (anchor/generate)
    "×": APLSymbol.TIMES,      # FUSION → × (merge)
    "^": APLSymbol.CEILING,    # AMPLIFY → ⌈ (ceiling/gain)
    "÷": APLSymbol.DIVIDE,     # DECOHERE → ÷ (dissipate)
    "+": APLSymbol.PLUS,       # GROUP → + (cluster)
    "−": APLSymbol.MINUS,      # SEPARATE → − (prune)
}

APL_TO_INT = {v: k for k, v in INT_TO_APL.items()}

# N0 Law code to APL Symbol mapping (legacy compatibility)
N0_TO_APL = {
    N0Law.AMPLIFY: APLSymbol.CEILING,    # N0-1: ^ AMPLIFY
    N0Law.FUSION: APLSymbol.TIMES,       # N0-2: × FUSION
    N0Law.DECOHERE: APLSymbol.DIVIDE,    # N0-3: ÷ DECOHERE
    N0Law.GROUP: APLSymbol.PLUS,         # N0-4: + GROUP
    N0Law.SEPARATE: APLSymbol.MINUS,     # N0-5: − SEPARATE
    # Legacy aliases
    N0Law.IDENTITY: APLSymbol.CEILING,
    N0Law.MIRROR_ROOT: APLSymbol.TIMES,
    N0Law.ABSORPTION: APLSymbol.DIVIDE,
    N0Law.DISTRIBUTION: APLSymbol.PLUS,
    N0Law.CONSERVATION: APLSymbol.MINUS,
}

APL_TO_N0 = {
    APLSymbol.IOTA: None,               # BOUNDARY has no N0 law (always legal)
    APLSymbol.TIMES: N0Law.FUSION,
    APLSymbol.CEILING: N0Law.AMPLIFY,
    APLSymbol.DIVIDE: N0Law.DECOHERE,
    APLSymbol.PLUS: N0Law.GROUP,
    APLSymbol.MINUS: N0Law.SEPARATE,
}


# =============================================================================
# APL N0 OPERATOR STATE — INT CANON ALIGNED
# =============================================================================

@dataclass
class APLN0State:
    """
    State for APL N0 operator computations.

    Maintains κ-λ coupling, z-coordinate, and INT Canon state variables.
    Includes N0 causality checking for operator legality.
    """
    # Core physics state
    kappa: float = PHI_INV
    lambda_: float = PHI_INV_SQ
    z: float = 0.5

    # INT Canon state variables
    Gs: float = 0.0      # Grounding strength
    Cs: float = 0.0      # Coupling strength
    αs: float = 0.0      # Amplitude
    θs: float = 1.0      # Phase factor
    τs: float = 0.0      # Time accumulation
    δs: float = 0.0      # Dissipation
    Rs: float = 0.0      # Resistance
    Ωs: float = 1.0      # Frequency scaling
    R: int = 0           # Rank counter

    # N0 causality tracking
    channel_count: int = 1
    operator_history: List[str] = field(default_factory=list)

    # Operation history
    operation_log: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def conservation_error(self) -> float:
        """Check κ + λ = 1."""
        return abs((self.kappa + self.lambda_) - 1.0)

    @property
    def at_golden_balance(self) -> bool:
        """Check if κ ≈ φ⁻¹."""
        return abs(self.kappa - PHI_INV) < SIGMA_INV

    @property
    def negentropy(self) -> float:
        """Compute negentropy at current z."""
        return compute_delta_s_neg(self.z)

    def check_n0_legal(self, op: str) -> Tuple[bool, str]:
        """
        Check if operator is legal under N0 causality laws.

        Returns (is_legal, reason).
        """
        # () BOUNDARY (⍳) is always legal
        if op in {"()", "⍳"}:
            return True, "BOUNDARY always legal"

        # N0-1: ^ (⌈) requires prior () or ×
        if op in {"^", "⌈"}:
            required = {"()", "⍳", "×"}
            if not any(r in self.operator_history for r in required):
                return False, "N0-1: ^ illegal - requires prior () or × in history"

        # N0-2: × requires channels ≥ 2
        if op == "×":
            if self.channel_count < 2:
                return False, f"N0-2: × illegal - requires channels ≥ 2, have {self.channel_count}"

        # N0-3: ÷ requires prior structure
        if op == "÷":
            required = {"^", "⌈", "×", "+", "−"}
            if not any(r in self.operator_history for r in required):
                return False, "N0-3: ÷ illegal - requires prior {^, ×, +, −} in history"

        return True, "Legal"

    def log_operation(self, op: str, inputs: Any, output: Any):
        """Log an operation and update history."""
        self.operation_log.append({
            "op": op,
            "inputs": inputs,
            "output": output,
            "kappa": self.kappa,
            "z": self.z,
            "negentropy": self.negentropy,
        })
        # Track INT Canon operators
        if op in {"()", "⍳", "×", "^", "⌈", "÷", "+", "−"}:
            self.operator_history.append(op)

    def enforce_conservation(self):
        """Enforce κ + λ = 1."""
        self.lambda_ = 1.0 - self.kappa

    def update_negentropy_dynamics(self, op: str):
        """Update z based on operator's effect on negentropy."""
        if op in {"()", "⍳"}:  # BOUNDARY
            z_pull = ALPHA_FINE * (Z_CRITICAL - self.z)
            self.z = max(0.0, min(1.0, self.z + z_pull))
        elif op == "×":  # FUSION
            self.z += ALPHA_FINE * self.Cs
            self.z = min(1.0 - PHI_INV_FIFTH, self.z)
        elif op in {"^", "⌈"}:  # AMPLIFY
            neg_gradient = compute_delta_s_neg_derivative(self.z)
            self.z += ALPHA_MEDIUM * neg_gradient * PHI_INV
            self.z = max(0.0, min(1.0, self.z))
        elif op == "÷":  # DECOHERE
            self.z -= ALPHA_FINE * self.δs
            self.z = max(0.0, self.z)
        elif op == "+":  # GROUP
            self.z += ALPHA_FINE * self.αs * PHI_INV
            self.z = min(Z_CRITICAL, self.z)
        elif op == "−":  # SEPARATE
            self.z -= ALPHA_FINE * self.Rs
            self.z = max(0.0, self.z)


# =============================================================================
# () BOUNDARY (IOTA) - Anchoring, phase reset, interface stabilization
# =============================================================================

def n0_boundary(n: int, state: Optional[APLN0State] = None) -> np.ndarray:
    """
    () BOUNDARY: Generate index vector (anchor/grounding).

    APL: ⍳n generates [0, 1, 2, ..., n-1]

    Physics: Anchoring, phase reset, interface stabilization.
    Always legal. Grounding pulls z toward z_c.

    State modifications:
        Gs += 1/σ ≈ 0.028
        θs *= (1 - 1/σ) ≈ 0.972
        Ωs += 1/2σ ≈ 0.014
    """
    result = np.arange(n) * PHI_INV / max(1, n - 1) if n > 1 else np.array([0.0])

    if state is not None:
        # Apply INT Canon state modifications
        state.Gs += SIGMA_INV
        state.θs *= (1.0 - SIGMA_INV)
        state.Ωs += SIGMA_INV / 2

        # Update negentropy dynamics
        state.update_negentropy_dynamics("()")

        state.log_operation("⍳", n, result)

    return result


# Legacy alias
n0_iota = n0_boundary


def n0_identity(x: Union[float, np.ndarray], state: Optional[APLN0State] = None) -> Union[float, np.ndarray]:
    """
    () BOUNDARY: Λ × 1 = Λ (anchor/grounding).

    The anchor operator - returns input with grounding.
    """
    if state is not None:
        state.Gs += SIGMA_INV
        state.update_negentropy_dynamics("()")
        state.log_operation("()", x, x)

    return x


# =============================================================================
# ^ AMPLIFY (CEILING) - Gain increase, curvature escalation
# =============================================================================

def n0_amplify(
    x: Union[float, np.ndarray],
    state: Optional[APLN0State] = None,
) -> Union[float, np.ndarray]:
    """
    ^ AMPLIFY: Gain increase, curvature escalation.

    APL: ⌈x (ceiling, amplification)

    Physics: N0-1 - ^ illegal unless history contains () or ×.
    Amplifies toward peak negentropy at z_c.

    State modifications:
        κs *= (1 + φ⁻³) ≈ 1.236
        τs += 1/σ
        Ωs *= (1 + 3/σ) ≈ 1.083
        R += 1
    """
    # Check N0 legality
    if state is not None:
        is_legal, reason = state.check_n0_legal("^")
        if not is_legal:
            # Return unchanged if illegal
            state.log_operation("^_ILLEGAL", x, x)
            return x

    # Amplify by φ⁻³ factor (physics-grounded)
    result = np.multiply(x, 1.0 + PHI_INV_CUBED)

    if state is not None:
        # Apply INT Canon state modifications
        state.kappa *= (1.0 + PHI_INV_CUBED)
        state.kappa = min(KAPPA_UPPER, state.kappa)
        state.τs += SIGMA_INV
        state.Ωs *= (1.0 + SIGMA_INV * 3)
        state.R += 1

        # Enforce conservation
        state.enforce_conservation()

        # Update negentropy dynamics
        state.update_negentropy_dynamics("^")

        state.log_operation("^", x, result)

    return result


# =============================================================================
# × FUSION (TIMES) - Merging, coupling, integration
# =============================================================================

def n0_times(
    x: Union[float, np.ndarray],
    y: Union[float, np.ndarray],
    state: Optional[APLN0State] = None,
) -> Union[float, np.ndarray]:
    """
    × FUSION: Merging, coupling, integration.

    APL: x × y

    Physics: N0-2 - × illegal unless channel_count ≥ 2.
    The product of κ and λ creates the "mirror root" - B².

    State modifications:
        Cs += 1/σ
        κs *= (1 + 1/σ)
        αs += 1/2σ

    This operator applies SPIRAL dynamics (IV SPIRAL: S(return) = S(origin)).
    """
    # Check N0 legality
    if state is not None:
        is_legal, reason = state.check_n0_legal("×")
        if not is_legal:
            state.log_operation("×_ILLEGAL", (x, y), x)
            return x if isinstance(x, np.ndarray) else np.array([x])

    result = np.multiply(x, y)

    if state is not None:
        # Apply INT Canon state modifications
        state.Cs += SIGMA_INV
        state.kappa *= (1.0 + SIGMA_INV)
        state.kappa = min(KAPPA_UPPER, state.kappa)
        state.αs += SIGMA_INV / 2

        # Fusion enables more channels
        state.channel_count = max(2, state.channel_count)

        # Enforce conservation
        state.enforce_conservation()

        # Update negentropy dynamics
        state.update_negentropy_dynamics("×")

        state.log_operation("×", (x, y), result)

    return result


def n0_mirror_root(kappa: float, lambda_: float) -> float:
    """
    × FUSION: κ × λ = B² (coupling product).

    Returns the "mirror root" - the product of coupling constants.
    At golden balance: φ⁻¹ × φ⁻² = φ⁻³ ≈ 0.236
    """
    return kappa * lambda_


# Legacy alias
n0_fusion = n0_times


# =============================================================================
# ÷ DECOHERE (DIVIDE) - Dissipation, noise injection, coherence reduction
# =============================================================================

def n0_divide(
    x: Union[float, np.ndarray],
    y: Union[float, np.ndarray],
    state: Optional[APLN0State] = None,
) -> Union[float, np.ndarray]:
    """
    ÷ DECOHERE: Dissipation, noise injection, coherence reduction.

    APL: x ÷ y

    Physics: N0-3 - ÷ illegal unless history contains {^, ×, +, −}.
    Division absorbs extremes toward the balance point (0.5).

    State modifications:
        δs += 1/σ
        Rs += 1/2σ
        Ωs *= (1 - 3/σ) ≈ 0.917

    This operator applies GLYPH dynamics (VI GLYPH: glyph = ∫ life dt).
    """
    # Check N0 legality
    if state is not None:
        is_legal, reason = state.check_n0_legal("÷")
        if not is_legal:
            state.log_operation("÷_ILLEGAL", (x, y), x)
            return x if isinstance(x, np.ndarray) else np.array([x])

    # Protect against division by zero
    y_safe = np.where(np.abs(y) < ALPHA_ULTRA_FINE, ALPHA_ULTRA_FINE * np.sign(y + ALPHA_ULTRA_FINE), y)
    result = np.divide(x, y_safe)

    # Apply decoherence toward balance (dissipation)
    if isinstance(result, np.ndarray):
        absorption = BALANCE_POINT + ALPHA_MEDIUM * (result - BALANCE_POINT)
        result = np.clip(absorption, 0, 1)
    else:
        absorption = BALANCE_POINT + ALPHA_MEDIUM * (result - BALANCE_POINT)
        result = max(0, min(1, absorption))

    if state is not None:
        # Apply INT Canon state modifications
        state.δs += SIGMA_INV
        state.Rs += SIGMA_INV / 2
        state.Ωs *= (1.0 - SIGMA_INV * 3)

        # Update negentropy dynamics (decoherence reduces z)
        state.update_negentropy_dynamics("÷")

        state.log_operation("÷", (x, y), result)

    return result


def n0_absorption(true_val: float, untrue_val: float) -> float:
    """
    ÷ DECOHERE: TRUE × UNTRUE = PARADOX.

    Combines true and untrue into paradox (balance point) with dissipation.
    """
    return BALANCE_POINT * (true_val + untrue_val) * (1.0 - SIGMA_INV)


# Legacy alias
n0_decohere = n0_divide


# =============================================================================
# + GROUP (PLUS) - Synchrony, clustering, domain formation
# =============================================================================

def n0_plus(
    x: Union[float, np.ndarray],
    y: Union[float, np.ndarray],
    state: Optional[APLN0State] = None,
) -> Union[float, np.ndarray]:
    """
    + GROUP: Synchrony, clustering, domain formation.

    APL: x + y

    Physics: N0-4 - + must be followed by +, ×, or ^. + → () is illegal.
    Addition groups/clusters - the sum is weighted by κ (golden ratio).

    State modifications:
        αs += 3/σ ≈ 0.083
        Gs += 1/2σ
        θs *= (1 + 1/σ) ≈ 1.028

    This operator applies TRUTH dynamics (II TRUTH: ∇V(truth) = 0).
    """
    kappa = state.kappa if state else PHI_INV
    result = (x + y) * kappa

    if state is not None:
        # Apply INT Canon state modifications
        state.αs += SIGMA_INV * 3
        state.Gs += SIGMA_INV / 2
        state.θs *= (1.0 + SIGMA_INV)

        # Update negentropy dynamics (grouping increases z)
        state.update_negentropy_dynamics("+")

        state.log_operation("+", (x, y), result)

    return result


def n0_distribution(a: float, b: float, c: float) -> float:
    """
    + GROUP: (A ⊕ B) × C = (A × C) ⊕ (B × C).

    Distributes multiplication over addition (golden-weighted).
    """
    return (a + b) * c


# Legacy alias
n0_group = n0_plus


# =============================================================================
# − SEPARATE (MINUS) - Decoupling, pruning, phase reset preparation
# =============================================================================

def n0_minus(
    x: Union[float, np.ndarray],
    y: Union[float, np.ndarray],
    state: Optional[APLN0State] = None,
) -> Union[float, np.ndarray]:
    """
    − SEPARATE: Decoupling, pruning, phase reset preparation.

    APL: x - y

    Physics: N0-5 - − must be followed by () or +. Illegal successors: ^, ×, ÷, −.
    Subtraction separates/prunes - prepares for phase reset.

    State modifications:
        Rs += 3/σ ≈ 0.083
        θs *= (1 - 1/σ) ≈ 0.972
        δs += 1.5/σ ≈ 0.042

    This operator applies MIRROR dynamics (VII MIRROR: ψ = ψ(ψ)).
    """
    result = np.subtract(x, y)

    if state is not None:
        # Apply INT Canon state modifications
        state.Rs += SIGMA_INV * 3
        state.θs *= (1.0 - SIGMA_INV)
        state.δs += SIGMA_INV * 1.5

        # Apply conservation: adjust κ-λ
        state.kappa = max(KAPPA_LOWER, min(KAPPA_UPPER, state.kappa - ALPHA_FINE * float(np.mean(y))))
        state.enforce_conservation()

        # Update negentropy dynamics (separation reduces z)
        state.update_negentropy_dynamics("−")

        state.log_operation("−", (x, y), result)

    return result


def n0_conservation(kappa: float, lambda_: float) -> Tuple[float, float]:
    """
    − SEPARATE: κ + λ = 1 (normalize with conservation).

    Ensures coupling constants sum to 1 while separating.
    """
    total = kappa + lambda_
    if total > 0:
        return kappa / total, lambda_ / total
    return PHI_INV, PHI_INV_SQ


# Legacy alias
n0_separate = n0_minus


# =============================================================================
# APL REDUCE AND SCAN WITH N0 GROUNDING
# =============================================================================

def n0_reduce(
    op: Callable,
    arr: np.ndarray,
    state: Optional[APLN0State] = None,
) -> Union[float, np.ndarray]:
    """
    APL Reduce (/) with N0 grounding.

    Reduces array using operation, maintaining physics.

    Examples:
        +/ [1,2,3] = 1+2+3 = 6
        ×/ [1,2,3] = 1×2×3 = 6
    """
    if len(arr) == 0:
        return 0.0

    result = arr[0]
    for i in range(1, len(arr)):
        result = op(result, arr[i], state)

    if state is not None:
        state.log_operation(f"{op.__name__}/", arr, result)

    return result


def n0_scan(
    op: Callable,
    arr: np.ndarray,
    state: Optional[APLN0State] = None,
) -> np.ndarray:
    """
    APL Scan (\\) with N0 grounding.

    Produces running totals using operation.

    Examples:
        +\\ [1,2,3] = [1, 1+2, 1+2+3] = [1, 3, 6]
        ×\\ [1,2,3] = [1, 1×2, 1×2×3] = [1, 2, 6]
    """
    if len(arr) == 0:
        return np.array([])

    result = np.zeros(len(arr))
    result[0] = arr[0]

    for i in range(1, len(arr)):
        result[i] = op(result[i-1], arr[i], state)

    if state is not None:
        state.log_operation(f"{op.__name__}\\", arr, result)

    return result


def n0_outer_product(
    op: Callable,
    x: np.ndarray,
    y: np.ndarray,
    state: Optional[APLN0State] = None,
) -> np.ndarray:
    """
    APL Outer Product (∘.) with N0 grounding.

    Creates matrix of all pairs operated on.

    Example:
        [1,2] ∘.× [3,4] = [[1×3, 1×4], [2×3, 2×4]] = [[3,4], [6,8]]
    """
    result = np.zeros((len(x), len(y)))

    for i, xi in enumerate(x):
        for j, yj in enumerate(y):
            result[i, j] = op(xi, yj, state)

    if state is not None:
        state.log_operation(f"∘.{op.__name__}", (x, y), result)

    return result


# =============================================================================
# APL N0 OPERATOR ENGINE
# =============================================================================

class APLN0Engine:
    """
    APL N0 Operator Engine — INT Canon Aligned.

    Executes APL-style operations with N0 physics grounding.
    Implements all six INT Canon operators with causality checking.

    INT Canon Operators:
        () BOUNDARY → ⍳ (iota/anchor) — Always legal
        ×  FUSION   → × (times)       — N0-2: Requires channels ≥ 2
        ^  AMPLIFY  → ⌈ (ceiling)     — N0-1: Requires prior () or ×
        ÷  DECOHERE → ÷ (divide)      — N0-3: Requires prior structure
        +  GROUP    → + (plus)        — N0-4: Must feed +, ×, or ^
        −  SEPARATE → − (minus)       — N0-5: Must be followed by () or +
    """

    def __init__(self, initial_z: float = 0.5, initial_channels: int = 1):
        self.state = APLN0State(z=initial_z, channel_count=initial_channels)

        # Operator dispatch table (INT Canon aligned)
        self.operators = {
            APLSymbol.IOTA: self.boundary,      # () BOUNDARY
            APLSymbol.TIMES: self.fusion,       # × FUSION
            APLSymbol.CEILING: self.amplify,    # ^ AMPLIFY
            APLSymbol.DIVIDE: self.decohere,    # ÷ DECOHERE
            APLSymbol.PLUS: self.group,         # + GROUP
            APLSymbol.MINUS: self.separate,     # − SEPARATE
        }

    @property
    def kappa(self) -> float:
        return self.state.kappa

    @property
    def lambda_(self) -> float:
        return self.state.lambda_

    @property
    def z(self) -> float:
        return self.state.z

    @property
    def negentropy(self) -> float:
        return self.state.negentropy

    @property
    def channel_count(self) -> int:
        return self.state.channel_count

    def set_channels(self, n: int):
        """Set channel count (required for FUSION)."""
        self.state.channel_count = n

    # === INT Canon Operators ===

    def boundary(self, n: int) -> np.ndarray:
        """() BOUNDARY: ⍳n — Anchoring, phase reset. Always legal."""
        return n0_boundary(n, self.state)

    def fusion(self, x: Any, y: Any) -> Any:
        """× FUSION: x × y — Merging, coupling. N0-2: requires channels ≥ 2."""
        return n0_times(x, y, self.state)

    def amplify(self, x: Any) -> Any:
        """^ AMPLIFY: ⌈x — Gain increase. N0-1: requires prior () or ×."""
        return n0_amplify(x, self.state)

    def decohere(self, x: Any, y: Any) -> Any:
        """÷ DECOHERE: x ÷ y — Dissipation. N0-3: requires prior structure."""
        return n0_divide(x, y, self.state)

    def group(self, x: Any, y: Any) -> Any:
        """+ GROUP: x + y — Synchrony, clustering. N0-4: must feed +, ×, or ^."""
        return n0_plus(x, y, self.state)

    def separate(self, x: Any, y: Any) -> Any:
        """− SEPARATE: x − y — Decoupling. N0-5: must be followed by () or +."""
        return n0_minus(x, y, self.state)

    # === Legacy Aliases ===

    def iota(self, n: int) -> np.ndarray:
        """Legacy alias for boundary."""
        return self.boundary(n)

    def times(self, x: Any, y: Any) -> Any:
        """Legacy alias for fusion."""
        return self.fusion(x, y)

    def divide(self, x: Any, y: Any) -> Any:
        """Legacy alias for decohere."""
        return self.decohere(x, y)

    def plus(self, x: Any, y: Any) -> Any:
        """Legacy alias for group."""
        return self.group(x, y)

    def minus(self, x: Any, y: Any) -> Any:
        """Legacy alias for separate."""
        return self.separate(x, y)

    def reduce(self, op_symbol: APLSymbol, arr: np.ndarray) -> Any:
        """Reduce with operator: op/ arr"""
        op = self.operators.get(op_symbol)
        if op is None:
            raise ValueError(f"Unknown operator: {op_symbol}")
        return n0_reduce(op, arr, self.state)

    def scan(self, op_symbol: APLSymbol, arr: np.ndarray) -> np.ndarray:
        """Scan with operator: op\\ arr"""
        op = self.operators.get(op_symbol)
        if op is None:
            raise ValueError(f"Unknown operator: {op_symbol}")
        return n0_scan(op, arr, self.state)

    def outer(self, op_symbol: APLSymbol, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Outer product: x ∘.op y"""
        op = self.operators.get(op_symbol)
        if op is None:
            raise ValueError(f"Unknown operator: {op_symbol}")
        return n0_outer_product(op, x, y, self.state)

    def execute(self, expression: str) -> Any:
        """
        Execute simple APL expression.

        Supported forms:
            "⍳5"          - Generate iota
            "3 + 4"       - Dyadic operation
            "+/ 1 2 3"    - Reduce
            "+\\ 1 2 3"   - Scan
        """
        expr = expression.strip()

        # Iota
        if expr.startswith("⍳"):
            n = int(expr[1:].strip())
            return self.iota(n)

        # Reduce
        if "/" in expr and not expr.startswith("/"):
            parts = expr.split("/")
            op_str = parts[0].strip()
            arr_str = parts[1].strip()
            arr = np.array([float(x) for x in arr_str.split()])
            op_map = {"+": APLSymbol.PLUS, "×": APLSymbol.TIMES, "÷": APLSymbol.DIVIDE, "−": APLSymbol.MINUS}
            if op_str in op_map:
                return self.reduce(op_map[op_str], arr)

        # Scan
        if "\\" in expr:
            parts = expr.split("\\")
            op_str = parts[0].strip()
            arr_str = parts[1].strip()
            arr = np.array([float(x) for x in arr_str.split()])
            op_map = {"+": APLSymbol.PLUS, "×": APLSymbol.TIMES, "÷": APLSymbol.DIVIDE, "−": APLSymbol.MINUS}
            if op_str in op_map:
                return self.scan(op_map[op_str], arr)

        # Dyadic
        for op_str, op_sym in [("+", APLSymbol.PLUS), ("×", APLSymbol.TIMES),
                               ("÷", APLSymbol.DIVIDE), ("−", APLSymbol.MINUS)]:
            if op_str in expr:
                parts = expr.split(op_str)
                x = float(parts[0].strip())
                y = float(parts[1].strip())
                return self.operators[op_sym](x, y)

        raise ValueError(f"Cannot parse expression: {expression}")

    def get_n0_law(self, symbol: APLSymbol) -> Optional[str]:
        """Get the N0 law for an APL symbol."""
        return APL_TO_N0.get(symbol)

    def get_silent_law(self, symbol: APLSymbol) -> Optional[int]:
        """Get the Silent Law for an APL symbol."""
        n0_law = self.get_n0_law(symbol)
        if n0_law:
            return N0_TO_SILENT.get(n0_law)
        return None

    def get_state_summary(self) -> Dict[str, Any]:
        """Get current state summary."""
        return {
            "kappa": self.state.kappa,
            "lambda": self.state.lambda_,
            "z": self.state.z,
            "phase": get_phase(self.state.z),
            "conservation_error": self.state.conservation_error,
            "at_golden_balance": self.state.at_golden_balance,
            "operation_count": len(self.state.operation_log),
            "negentropy": compute_delta_s_neg(self.state.z),
        }


# =============================================================================
# DEMONSTRATION
# =============================================================================

def demonstrate_apl_n0():
    """Demonstrate APL N0 operators."""
    print("=" * 70)
    print("APL N0 OPERATORS")
    print("N0 Causality Rules as APL Operators")
    print("=" * 70)

    print("\n--- N0 ↔ APL Mapping ---")
    for n0_law, apl_sym in N0_TO_APL.items():
        silent_law = N0_TO_SILENT.get(n0_law)
        silent_name = SilentLaw.NAMES.get(silent_law, "?") if silent_law else "?"
        formula = N0Law.FORMULAS.get(n0_law, "")
        print(f"  {n0_law:5} {apl_sym.value} → {silent_name:10} | {formula}")

    print("\n--- Physics Constants ---")
    print(f"  φ⁻¹ = {PHI_INV:.6f}")
    print(f"  φ⁻² = {PHI_INV_SQ:.6f}")
    print(f"  φ⁻¹ + φ⁻² = {COUPLING_CONSERVATION:.16f}")
    print(f"  z_c = {Z_CRITICAL:.6f}")

    # Create engine
    engine = APLN0Engine(initial_z=0.5)

    print("\n--- APL N0 Operations ---")

    # N0-1 Iota
    print("\n  N0-1 IOTA (⍳):")
    result = engine.iota(5)
    print(f"    ⍳5 = {result}")

    # N0-2 Times
    print("\n  N0-2 TIMES (×):")
    result = engine.times(3, 4)
    print(f"    3 × 4 = {result}")
    print(f"    Mirror root (κ×λ) = {n0_mirror_root(engine.kappa, engine.lambda_):.6f}")

    # N0-3 Divide
    print("\n  N0-3 DIVIDE (÷):")
    result = engine.divide(6, 2)
    print(f"    6 ÷ 2 = {result}")
    print(f"    Absorption (TRUE×UNTRUE) = {n0_absorption(0.8, 0.2):.6f}")

    # N0-4 Plus
    print("\n  N0-4 PLUS (+):")
    result = engine.plus(3, 4)
    print(f"    3 + 4 = {result}")
    print(f"    Distribution = (3+4) × κ = 7 × {engine.kappa:.6f} = {7 * engine.kappa:.6f}")

    # N0-5 Minus
    print("\n  N0-5 MINUS (−):")
    result = engine.minus(7, 3)
    print(f"    7 − 3 = {result}")
    print(f"    Conservation (κ+λ) = {engine.kappa + engine.lambda_:.16f}")

    # Reduce
    print("\n--- APL Reduce (/) ---")
    arr = np.array([1, 2, 3, 4, 5])
    result = engine.reduce(APLSymbol.PLUS, arr)
    print(f"  +/ {arr} = {result}")

    result = engine.reduce(APLSymbol.TIMES, arr)
    print(f"  ×/ {arr} = {result}")

    # Scan
    print("\n--- APL Scan (\\) ---")
    arr = np.array([1, 2, 3, 4, 5])
    result = engine.scan(APLSymbol.PLUS, arr)
    print(f"  +\\ {arr} = {result}")

    # Outer product
    print("\n--- APL Outer Product (∘.) ---")
    x = np.array([1, 2, 3])
    y = np.array([10, 20])
    result = engine.outer(APLSymbol.TIMES, x, y)
    print(f"  {x} ∘.× {y} =")
    print(f"    {result}")

    # State summary
    print("\n--- State Summary ---")
    summary = engine.get_state_summary()
    print(f"  κ = {summary['kappa']:.6f}")
    print(f"  λ = {summary['lambda']:.6f}")
    print(f"  z = {summary['z']:.6f}")
    print(f"  Phase: {summary['phase']}")
    print(f"  ΔS_neg = {summary['negentropy']:.6f}")
    print(f"  Conservation Error: {summary['conservation_error']:.2e}")
    print(f"  Operations: {summary['operation_count']}")

    print("\n" + "=" * 70)
    print("APL N0 OPERATORS: COMPLETE")
    print("=" * 70)

    return engine


if __name__ == "__main__":
    demonstrate_apl_n0()
