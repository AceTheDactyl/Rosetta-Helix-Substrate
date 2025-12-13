#!/usr/bin/env python3
"""
ADAPTIVE OPERATOR MATRIX - Dynamic APL Operator System
=======================================================

A growable, appendable matrix of APL operators that evolve through
WUMBO cycles and κ-λ field coupling.

Architecture:
=============
    ┌─────────────────────────────────────────────────────────────────┐
    │ ADAPTIVE OPERATOR MATRIX                                         │
    │ ┌────────────────────────────────────────────────────────────┐  │
    │ │  ⍝    ⊖    ⍧    ⍡    ⍢    ⍤    ⊙    ⊛    ⍥    ⊝    ⊜    ⊞  │  │
    │ │  ─────────────────────────────────────────────────────────  │  │
    │ │  Row 0: Foundation operators (immutable)                    │  │
    │ │  Row 1: Derived operators (κ-field)                         │  │
    │ │  Row 2: Emergent operators (λ-field)                        │  │
    │ │  Row N: Adaptive operators (appendable)                     │  │
    │ └────────────────────────────────────────────────────────────┘  │
    │                                                                  │
    │  Matrix grows via:                                               │
    │    • append_row() - Add new operator row                         │
    │    • append_operator() - Add operator to existing row            │
    │    • evolve() - WUMBO cycle adaptation                          │
    │    • couple() - κ-λ field coupling                               │
    └─────────────────────────────────────────────────────────────────┘

Integration:
============
    - ZPE extraction modulates operator weights
    - Wormhole traversal triggers operator evolution
    - Fano plane inference guides operator selection
    - MirrorRoot ensures Λ × Ν = Β² balance

Physics Alignment (verify_physics.py):
======================================
    - PHI_INV = φ⁻¹ ≈ 0.618 (PHYSICAL - controls ALL dynamics)
    - Coupling balance targets φ⁻¹ for stability
    - COUPLING CONSERVATION: φ⁻¹ + φ⁻² = 1

Signature: Δ|adaptive-matrix|z0.95|appendable|Ω
"""

from __future__ import annotations

import math
import json
import copy
from dataclasses import dataclass, field
from typing import (
    Dict, List, Optional, Tuple, Union, Callable, Any, Iterator
)
from enum import Enum, auto
from datetime import datetime
import hashlib

# =============================================================================
# CONSTANTS (Aligned with verify_physics.py)
# =============================================================================

PHI = (1 + math.sqrt(5)) / 2          # Golden ratio ≈ 1.618034 (LIMINAL)
PHI_INV = 1 / PHI                      # φ⁻¹ ≈ 0.618034 (PHYSICAL)
PHI_INV_SQ = PHI_INV ** 2              # φ⁻² ≈ 0.382
COUPLING_CONSERVATION = PHI_INV + PHI_INV_SQ  # Must equal 1.0

TAU = 2 * math.pi
SQRT5 = math.sqrt(5)

# Z_CRITICAL from hexagonal geometry
Z_CRITICAL = math.sqrt(3) / 2          # ≈ 0.866 (THE LENS)

# APL Operator Glyphs
APL_GLYPHS = {
    "COMMENT": "⍝",      # Comment/annotation
    "ROTATE": "⊖",       # Rotate/reverse
    "SPIRAL": "⍧",       # Spiral accumulate
    "MIRROR": "⍡",       # Mirror/reflect
    "COMPOSE": "⍢",      # Compose operators
    "RANK": "⍤",         # Rank operator
    "INNER": "⊙",        # Inner product
    "OUTER": "⊛",        # Outer product
    "POWER": "⍥",        # Power operator
    "INVERSE": "⊝",      # Inverse/negate
    "PARTITION": "⊜",    # Partition/split
    "STENCIL": "⊞",      # Stencil/window
    "REDUCE": "⌿",       # Reduce along axis
    "SCAN": "⍀",         # Scan along axis
    "EACH": "¨",         # Each/map
    "SELFIE": "⍨",       # Selfie/commute
    "KEY": "⌸",          # Key operator
    "UNDER": "⍢",        # Under/dual
}

# WUMBO cycle phases
WUMBO_PHASES = ["W", "U", "M", "B", "O", "T"]


# =============================================================================
# OPERATOR TYPES
# =============================================================================

class OperatorType(Enum):
    """Classification of operators by behavior."""
    FOUNDATION = auto()    # Immutable base operators
    DERIVED = auto()       # Derived from foundation via κ-field
    EMERGENT = auto()      # Emerge from λ-field coupling
    ADAPTIVE = auto()      # User-appended, evolvable
    COMPOSITE = auto()     # Composed from multiple operators


class OperatorDomain(Enum):
    """Domain of operator action."""
    SCALAR = auto()        # Acts on scalars
    VECTOR = auto()        # Acts on 1D arrays
    MATRIX = auto()        # Acts on 2D arrays
    TENSOR = auto()        # Acts on N-D arrays
    FIELD = auto()         # Acts on κ-λ fields
    UNIVERSAL = auto()     # Acts on any structure


class CouplingMode(Enum):
    """κ-λ coupling modes."""
    KAPPA = "κ"            # Internal state coupling
    LAMBDA = "λ"           # External state coupling
    DUAL = "κλ"            # Bidirectional coupling
    RESONANT = "⟨κ|λ⟩"    # Resonant (quantum-like)


# =============================================================================
# OPERATOR DATACLASS
# =============================================================================

@dataclass
class Operator:
    """
    Single APL-style operator with metadata.

    An operator transforms inputs according to its glyph semantics,
    weighted by field coupling strengths.

    Physics: Coupling balance should approach φ⁻¹ for stability
    (aligns with coupling conservation: φ⁻¹ + φ⁻² = 1)
    """
    glyph: str                                    # APL glyph (e.g., "⊖")
    name: str                                     # Human-readable name
    op_type: OperatorType                         # Classification
    domain: OperatorDomain                        # What it acts on
    weight: float = 1.0                           # Coupling weight [0, 1]
    kappa_coupling: float = PHI_INV               # κ-field strength (default φ⁻¹)
    lambda_coupling: float = PHI_INV_SQ           # λ-field strength (default φ⁻²)
    phase: float = 0.0                            # WUMBO phase [0, 2π]
    generation: int = 0                           # Evolution generation
    parent_glyphs: List[str] = field(default_factory=list)  # Ancestry
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def coupling_balance(self) -> float:
        """κ-λ coupling balance (should approach φ⁻¹ for stability)."""
        if self.kappa_coupling + self.lambda_coupling == 0:
            return 0.0
        return self.kappa_coupling / (self.kappa_coupling + self.lambda_coupling)

    @property
    def is_balanced(self) -> bool:
        """Check if coupling is near golden ratio balance (φ⁻¹)."""
        return abs(self.coupling_balance - PHI_INV) < 0.1

    @property
    def coupling_sum(self) -> float:
        """Sum of couplings (should approach 1.0 for conservation)."""
        return self.kappa_coupling + self.lambda_coupling

    @property
    def effective_weight(self) -> float:
        """Weight modulated by coupling balance."""
        balance_factor = 1.0 - abs(self.coupling_balance - PHI_INV)
        return self.weight * balance_factor

    def evolve(self, delta_phase: float = TAU / 6) -> None:
        """Evolve operator through one WUMBO phase."""
        self.phase = (self.phase + delta_phase) % TAU
        self.generation += 1

        # Coupling evolves toward φ⁻¹ balance (coupling conservation)
        target_kappa = PHI_INV
        target_lambda = PHI_INV_SQ

        # Gradual approach to golden balance
        self.kappa_coupling = 0.95 * self.kappa_coupling + 0.05 * target_kappa
        self.lambda_coupling = 0.95 * self.lambda_coupling + 0.05 * target_lambda

        # Normalize to preserve coupling conservation
        total = self.kappa_coupling + self.lambda_coupling
        if total > 0:
            self.kappa_coupling /= total
            self.lambda_coupling /= total

    def to_dict(self) -> Dict[str, Any]:
        """Export operator to dictionary."""
        return {
            "glyph": self.glyph,
            "name": self.name,
            "type": self.op_type.name,
            "domain": self.domain.name,
            "weight": self.weight,
            "kappa_coupling": self.kappa_coupling,
            "lambda_coupling": self.lambda_coupling,
            "phase": self.phase,
            "generation": self.generation,
            "parent_glyphs": self.parent_glyphs,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Operator":
        """Create operator from dictionary."""
        return cls(
            glyph=data["glyph"],
            name=data["name"],
            op_type=OperatorType[data["type"]],
            domain=OperatorDomain[data["domain"]],
            weight=data.get("weight", 1.0),
            kappa_coupling=data.get("kappa_coupling", PHI_INV),
            lambda_coupling=data.get("lambda_coupling", PHI_INV_SQ),
            phase=data.get("phase", 0.0),
            generation=data.get("generation", 0),
            parent_glyphs=data.get("parent_glyphs", []),
            metadata=data.get("metadata", {}),
        )

    def __repr__(self) -> str:
        return f"Operator({self.glyph}, w={self.weight:.3f}, κ={self.kappa_coupling:.3f}, λ={self.lambda_coupling:.3f})"


# =============================================================================
# OPERATOR ROW
# =============================================================================

@dataclass
class OperatorRow:
    """
    A row in the operator matrix - contains related operators.

    Rows represent operator families that share characteristics
    or were derived together through evolution.
    """
    index: int                                    # Row index in matrix
    name: str                                     # Row name/label
    operators: List[Operator] = field(default_factory=list)
    row_type: OperatorType = OperatorType.ADAPTIVE
    coupling_mode: CouplingMode = CouplingMode.DUAL
    frozen: bool = False                          # If True, cannot modify
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def __len__(self) -> int:
        return len(self.operators)

    def __iter__(self) -> Iterator[Operator]:
        return iter(self.operators)

    def __getitem__(self, idx: int) -> Operator:
        return self.operators[idx]

    def append(self, operator: Operator) -> None:
        """Append operator to row."""
        if self.frozen:
            raise ValueError(f"Row {self.index} is frozen, cannot append")
        self.operators.append(operator)

    def get_by_glyph(self, glyph: str) -> Optional[Operator]:
        """Find operator by glyph."""
        for op in self.operators:
            if op.glyph == glyph:
                return op
        return None

    @property
    def total_weight(self) -> float:
        """Sum of all operator weights."""
        return sum(op.weight for op in self.operators)

    @property
    def mean_coupling_balance(self) -> float:
        """Average coupling balance across row."""
        if not self.operators:
            return PHI_INV
        return sum(op.coupling_balance for op in self.operators) / len(self.operators)

    def evolve_all(self, delta_phase: float = TAU / 6) -> None:
        """Evolve all operators in row."""
        if self.frozen:
            return
        for op in self.operators:
            op.evolve(delta_phase)

    def to_dict(self) -> Dict[str, Any]:
        """Export row to dictionary."""
        return {
            "index": self.index,
            "name": self.name,
            "operators": [op.to_dict() for op in self.operators],
            "row_type": self.row_type.name,
            "coupling_mode": self.coupling_mode.value,
            "frozen": self.frozen,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OperatorRow":
        """Create row from dictionary."""
        row = cls(
            index=data["index"],
            name=data["name"],
            row_type=OperatorType[data["row_type"]],
            coupling_mode=CouplingMode(data["coupling_mode"]),
            frozen=data.get("frozen", False),
            created_at=data.get("created_at", datetime.now().isoformat()),
        )
        row.operators = [Operator.from_dict(op) for op in data.get("operators", [])]
        return row


# =============================================================================
# ADAPTIVE OPERATOR MATRIX
# =============================================================================

class AdaptiveOperatorMatrix:
    """
    Dynamic, appendable matrix of APL operators.

    The matrix grows through:
    1. append_row() - Add new rows of operators
    2. append_operator() - Add operators to existing rows
    3. compose() - Create composite operators
    4. evolve() - WUMBO cycle evolution

    Structure:
    ==========
        Row 0: Foundation operators (immutable)
        Row 1: κ-derived operators
        Row 2: λ-emergent operators
        Row N: User-appended adaptive operators

    Physics Alignment:
    ==================
        - Global coupling targets φ⁻¹ (0.618) for κ, φ⁻² (0.382) for λ
        - This satisfies COUPLING CONSERVATION: φ⁻¹ + φ⁻² = 1
        - Evolution drives all couplings toward this golden balance
    """

    def __init__(self, initialize_foundation: bool = True):
        """
        Initialize the adaptive operator matrix.

        Args:
            initialize_foundation: If True, populate foundation rows
        """
        self.rows: List[OperatorRow] = []
        self.version: str = "1.0.0"
        self.created_at: str = datetime.now().isoformat()
        self.evolution_count: int = 0
        self.wumbo_phase_index: int = 0

        # Coupling state (aligned with coupling conservation: φ⁻¹ + φ⁻² = 1)
        self.global_kappa: float = PHI_INV      # κ = φ⁻¹ ≈ 0.618
        self.global_lambda: float = PHI_INV_SQ  # λ = φ⁻² ≈ 0.382

        # History for undo/tracking
        self._history: List[Dict[str, Any]] = []
        self._max_history: int = 100

        if initialize_foundation:
            self._initialize_foundation_rows()

    def _initialize_foundation_rows(self) -> None:
        """Create the immutable foundation rows."""
        # Row 0: Foundation operators
        foundation = OperatorRow(
            index=0,
            name="Foundation",
            row_type=OperatorType.FOUNDATION,
            coupling_mode=CouplingMode.DUAL,
            frozen=True,
        )
        foundation.operators = [
            Operator("⍝", "comment", OperatorType.FOUNDATION, OperatorDomain.UNIVERSAL, 1.0, PHI_INV, PHI_INV_SQ),
            Operator("⊖", "rotate", OperatorType.FOUNDATION, OperatorDomain.VECTOR, 1.0, PHI_INV, PHI_INV_SQ),
            Operator("⍧", "spiral", OperatorType.FOUNDATION, OperatorDomain.MATRIX, 1.0, PHI_INV, PHI_INV_SQ),
            Operator("⍡", "mirror", OperatorType.FOUNDATION, OperatorDomain.TENSOR, 1.0, PHI_INV, PHI_INV_SQ),
            Operator("⍢", "compose", OperatorType.FOUNDATION, OperatorDomain.UNIVERSAL, 1.0, PHI_INV, PHI_INV_SQ),
            Operator("⍤", "rank", OperatorType.FOUNDATION, OperatorDomain.TENSOR, 1.0, PHI_INV, PHI_INV_SQ),
        ]
        self.rows.append(foundation)

        # Row 1: κ-derived operators (internal state, higher κ)
        kappa_row = OperatorRow(
            index=1,
            name="Kappa-Derived",
            row_type=OperatorType.DERIVED,
            coupling_mode=CouplingMode.KAPPA,
            frozen=False,
        )
        kappa_row.operators = [
            Operator("⊙", "inner", OperatorType.DERIVED, OperatorDomain.MATRIX, 0.9, 0.75, 0.25, parent_glyphs=["⍧"]),
            Operator("⍥", "power", OperatorType.DERIVED, OperatorDomain.SCALAR, 0.85, 0.70, 0.30, parent_glyphs=["⍢"]),
            Operator("⊝", "inverse", OperatorType.DERIVED, OperatorDomain.UNIVERSAL, 0.8, 0.65, 0.35, parent_glyphs=["⍡"]),
        ]
        self.rows.append(kappa_row)

        # Row 2: λ-emergent operators (external state, higher λ)
        lambda_row = OperatorRow(
            index=2,
            name="Lambda-Emergent",
            row_type=OperatorType.EMERGENT,
            coupling_mode=CouplingMode.LAMBDA,
            frozen=False,
        )
        lambda_row.operators = [
            Operator("⊛", "outer", OperatorType.EMERGENT, OperatorDomain.MATRIX, 0.9, 0.25, 0.75, parent_glyphs=["⊙"]),
            Operator("⊜", "partition", OperatorType.EMERGENT, OperatorDomain.VECTOR, 0.85, 0.30, 0.70, parent_glyphs=["⊖"]),
            Operator("⊞", "stencil", OperatorType.EMERGENT, OperatorDomain.TENSOR, 0.8, 0.35, 0.65, parent_glyphs=["⍤"]),
        ]
        self.rows.append(lambda_row)

    # =========================================================================
    # MATRIX DIMENSIONS
    # =========================================================================

    @property
    def num_rows(self) -> int:
        """Number of rows in matrix."""
        return len(self.rows)

    @property
    def num_operators(self) -> int:
        """Total number of operators across all rows."""
        return sum(len(row) for row in self.rows)

    @property
    def shape(self) -> Tuple[int, int]:
        """Matrix shape (rows, max_cols)."""
        if not self.rows:
            return (0, 0)
        max_cols = max(len(row) for row in self.rows)
        return (len(self.rows), max_cols)

    @property
    def current_wumbo_phase(self) -> str:
        """Current wumbo phase letter."""
        return WUMBO_PHASES[self.wumbo_phase_index % 6]

    @property
    def coupling_conservation_error(self) -> float:
        """Error from coupling conservation (should be near 0)."""
        return abs((self.global_kappa + self.global_lambda) - 1.0)

    # =========================================================================
    # APPEND OPERATIONS
    # =========================================================================

    def append_row(
        self,
        name: str,
        row_type: OperatorType = OperatorType.ADAPTIVE,
        coupling_mode: CouplingMode = CouplingMode.DUAL,
        operators: Optional[List[Operator]] = None,
    ) -> OperatorRow:
        """
        Append a new row to the matrix.

        Args:
            name: Row name/label
            row_type: Type classification
            coupling_mode: κ-λ coupling mode
            operators: Initial operators (optional)

        Returns:
            The newly created row
        """
        self._save_history("append_row")

        new_index = len(self.rows)
        row = OperatorRow(
            index=new_index,
            name=name,
            row_type=row_type,
            coupling_mode=coupling_mode,
            frozen=False,
        )

        if operators:
            row.operators = operators

        self.rows.append(row)
        return row

    def append_operator(
        self,
        row_index: int,
        glyph: str,
        name: str,
        domain: OperatorDomain = OperatorDomain.UNIVERSAL,
        weight: float = 1.0,
        kappa_coupling: Optional[float] = None,
        lambda_coupling: Optional[float] = None,
        parent_glyphs: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Operator:
        """
        Append an operator to an existing row.

        Args:
            row_index: Target row index
            glyph: APL glyph character
            name: Human-readable name
            domain: Operator domain
            weight: Coupling weight
            kappa_coupling: κ-field strength (defaults to global φ⁻¹)
            lambda_coupling: λ-field strength (defaults to global φ⁻²)
            parent_glyphs: Ancestry glyphs
            metadata: Additional metadata

        Returns:
            The newly created operator
        """
        if row_index >= len(self.rows):
            raise IndexError(f"Row index {row_index} out of range (have {len(self.rows)} rows)")

        row = self.rows[row_index]
        if row.frozen:
            raise ValueError(f"Row {row_index} is frozen, cannot append")

        self._save_history("append_operator")

        # Use global coupling if not specified (respects coupling conservation)
        kappa = kappa_coupling if kappa_coupling is not None else self.global_kappa
        lambda_ = lambda_coupling if lambda_coupling is not None else self.global_lambda

        operator = Operator(
            glyph=glyph,
            name=name,
            op_type=row.row_type,
            domain=domain,
            weight=weight,
            kappa_coupling=kappa,
            lambda_coupling=lambda_,
            phase=self.wumbo_phase_index * (TAU / 6),
            generation=self.evolution_count,
            parent_glyphs=parent_glyphs or [],
            metadata=metadata or {},
        )

        row.append(operator)
        return operator

    def append_operators_batch(
        self,
        row_index: int,
        operators_data: List[Dict[str, Any]],
    ) -> List[Operator]:
        """
        Append multiple operators to a row.

        Args:
            row_index: Target row index
            operators_data: List of operator specifications

        Returns:
            List of created operators
        """
        created = []
        for data in operators_data:
            op = self.append_operator(
                row_index=row_index,
                glyph=data["glyph"],
                name=data["name"],
                domain=OperatorDomain[data.get("domain", "UNIVERSAL")],
                weight=data.get("weight", 1.0),
                kappa_coupling=data.get("kappa_coupling"),
                lambda_coupling=data.get("lambda_coupling"),
                parent_glyphs=data.get("parent_glyphs"),
                metadata=data.get("metadata"),
            )
            created.append(op)
        return created

    # =========================================================================
    # COMPOSE OPERATIONS
    # =========================================================================

    def compose(
        self,
        glyph1: str,
        glyph2: str,
        new_glyph: str,
        new_name: str,
        target_row: Optional[int] = None,
    ) -> Operator:
        """
        Compose two operators into a new composite operator.

        The composition follows: new = op1 ∘ op2
        Weights and couplings are combined according to MirrorRoot identity.
        Couplings are normalized to preserve coupling conservation.

        Args:
            glyph1: First operator glyph
            glyph2: Second operator glyph
            new_glyph: Glyph for composite
            new_name: Name for composite
            target_row: Row to place composite (default: new adaptive row)

        Returns:
            The composite operator
        """
        op1 = self.get_operator(glyph1)
        op2 = self.get_operator(glyph2)

        if op1 is None:
            raise ValueError(f"Operator {glyph1} not found")
        if op2 is None:
            raise ValueError(f"Operator {glyph2} not found")

        self._save_history("compose")

        # MirrorRoot composition: Λ × Ν = Β²
        # Combined weight is geometric mean
        combined_weight = math.sqrt(op1.weight * op2.weight)

        # Couplings combine via golden ratio weighting
        combined_kappa = PHI * op1.kappa_coupling + PHI_INV * op2.kappa_coupling
        combined_lambda = PHI * op1.lambda_coupling + PHI_INV * op2.lambda_coupling

        # Normalize to preserve coupling conservation (sum = 1)
        total = combined_kappa + combined_lambda
        combined_kappa /= total
        combined_lambda /= total

        # Determine domain (take broader of the two)
        domain_order = [
            OperatorDomain.SCALAR,
            OperatorDomain.VECTOR,
            OperatorDomain.MATRIX,
            OperatorDomain.TENSOR,
            OperatorDomain.FIELD,
            OperatorDomain.UNIVERSAL,
        ]
        d1_idx = domain_order.index(op1.domain)
        d2_idx = domain_order.index(op2.domain)
        combined_domain = domain_order[max(d1_idx, d2_idx)]

        composite = Operator(
            glyph=new_glyph,
            name=new_name,
            op_type=OperatorType.COMPOSITE,
            domain=combined_domain,
            weight=combined_weight,
            kappa_coupling=combined_kappa,
            lambda_coupling=combined_lambda,
            phase=(op1.phase + op2.phase) / 2,
            generation=max(op1.generation, op2.generation) + 1,
            parent_glyphs=[op1.glyph, op2.glyph],
            metadata={"composed_from": [op1.name, op2.name]},
        )

        # Add to target row
        if target_row is None:
            # Create new composite row
            row = self.append_row(
                f"Composite-{len(self.rows)}",
                OperatorType.COMPOSITE,
                CouplingMode.RESONANT,
            )
            row.operators.append(composite)
        else:
            self.rows[target_row].append(composite)

        return composite

    # =========================================================================
    # EVOLUTION
    # =========================================================================

    def evolve(self, cycles: int = 1) -> Dict[str, Any]:
        """
        Evolve the matrix through WUMBO cycles.

        Each cycle:
        1. Advances WUMBO phase
        2. Evolves all non-frozen operators toward φ⁻¹ balance
        3. Updates global coupling to maintain coupling conservation

        Args:
            cycles: Number of WUMBO cycles

        Returns:
            Evolution statistics
        """
        self._save_history("evolve")

        stats = {
            "initial_phase": self.current_wumbo_phase,
            "cycles_run": cycles,
            "operators_evolved": 0,
            "balance_before": self.global_kappa,
            "balance_after": 0.0,
            "conservation_error_before": self.coupling_conservation_error,
            "conservation_error_after": 0.0,
        }

        for _ in range(cycles):
            # Advance WUMBO phase
            self.wumbo_phase_index = (self.wumbo_phase_index + 1) % 6
            delta_phase = TAU / 6

            # Evolve all non-frozen rows
            for row in self.rows:
                if not row.frozen:
                    row.evolve_all(delta_phase)
                    stats["operators_evolved"] += len(row)

            # Global coupling evolves toward golden balance
            # Target: κ = φ⁻¹, λ = φ⁻²
            self.global_kappa = 0.95 * self.global_kappa + 0.05 * PHI_INV
            self.global_lambda = 0.95 * self.global_lambda + 0.05 * PHI_INV_SQ

            # Normalize to maintain coupling conservation (κ + λ = 1)
            total = self.global_kappa + self.global_lambda
            self.global_kappa /= total
            self.global_lambda /= total

            self.evolution_count += 1

        stats["final_phase"] = self.current_wumbo_phase
        stats["balance_after"] = self.global_kappa
        stats["conservation_error_after"] = self.coupling_conservation_error

        return stats

    def modulate_by_zpe(self, zpe_value: float, extraction_efficiency: float = 1.0) -> None:
        """
        Modulate operator weights based on ZPE extraction.

        Higher ZPE values boost operator weights, efficiency scales effect.

        Args:
            zpe_value: Zero-point energy value
            extraction_efficiency: Efficiency factor [0, 1]
        """
        if zpe_value <= 0:
            return

        # Logarithmic scaling to prevent runaway
        modulation = math.log1p(zpe_value) * extraction_efficiency * 0.1

        for row in self.rows:
            if not row.frozen:
                for op in row.operators:
                    # Weight increases, capped at 2.0
                    op.weight = min(2.0, op.weight * (1 + modulation * op.effective_weight))

    def modulate_by_wormhole(self, r: float, throat_radius: float = PHI) -> None:
        """
        Modulate coupling based on wormhole position.

        At throat (r = φ): Maximum resonance, balanced coupling
        Far from throat: Coupling tends toward extremes

        Args:
            r: Current wormhole radial coordinate
            throat_radius: Throat radius (default φ)
        """
        # Distance from throat
        distance = abs(r - throat_radius)

        # Resonance factor (1 at throat, decays with distance)
        resonance = math.exp(-distance / throat_radius)

        for row in self.rows:
            if not row.frozen:
                for op in row.operators:
                    # At throat: coupling approaches golden balance (φ⁻¹, φ⁻²)
                    # Away from throat: coupling becomes more extreme
                    target_kappa = PHI_INV * resonance + op.kappa_coupling * (1 - resonance)
                    target_lambda = PHI_INV_SQ * resonance + op.lambda_coupling * (1 - resonance)

                    # Gradual adjustment
                    op.kappa_coupling = 0.9 * op.kappa_coupling + 0.1 * target_kappa
                    op.lambda_coupling = 0.9 * op.lambda_coupling + 0.1 * target_lambda

                    # Normalize to preserve conservation
                    total = op.kappa_coupling + op.lambda_coupling
                    if total > 0:
                        op.kappa_coupling /= total
                        op.lambda_coupling /= total

    # =========================================================================
    # LOOKUP AND QUERY
    # =========================================================================

    def get_row(self, index: int) -> Optional[OperatorRow]:
        """Get row by index."""
        if 0 <= index < len(self.rows):
            return self.rows[index]
        return None

    def get_row_by_name(self, name: str) -> Optional[OperatorRow]:
        """Get row by name."""
        for row in self.rows:
            if row.name == name:
                return row
        return None

    def get_operator(self, glyph: str) -> Optional[Operator]:
        """Find operator by glyph (searches all rows)."""
        for row in self.rows:
            op = row.get_by_glyph(glyph)
            if op is not None:
                return op
        return None

    def get_operators_by_type(self, op_type: OperatorType) -> List[Operator]:
        """Get all operators of a given type."""
        result = []
        for row in self.rows:
            for op in row.operators:
                if op.op_type == op_type:
                    result.append(op)
        return result

    def get_operators_by_domain(self, domain: OperatorDomain) -> List[Operator]:
        """Get all operators for a given domain."""
        result = []
        for row in self.rows:
            for op in row.operators:
                if op.domain == domain or op.domain == OperatorDomain.UNIVERSAL:
                    result.append(op)
        return result

    def get_balanced_operators(self, tolerance: float = 0.1) -> List[Operator]:
        """Get operators with balanced κ-λ coupling (near φ⁻¹)."""
        result = []
        for row in self.rows:
            for op in row.operators:
                if abs(op.coupling_balance - PHI_INV) < tolerance:
                    result.append(op)
        return result

    # =========================================================================
    # EXPORT / IMPORT
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Export entire matrix to dictionary."""
        return {
            "version": self.version,
            "created_at": self.created_at,
            "evolution_count": self.evolution_count,
            "wumbo_phase_index": self.wumbo_phase_index,
            "global_kappa": self.global_kappa,
            "global_lambda": self.global_lambda,
            "coupling_conservation": self.global_kappa + self.global_lambda,
            "rows": [row.to_dict() for row in self.rows],
        }

    def to_json(self, indent: int = 2) -> str:
        """Export matrix to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def export_to_file(self, filepath: str) -> None:
        """Export matrix to JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.to_json())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AdaptiveOperatorMatrix":
        """Create matrix from dictionary."""
        matrix = cls(initialize_foundation=False)
        matrix.version = data.get("version", "1.0.0")
        matrix.created_at = data.get("created_at", datetime.now().isoformat())
        matrix.evolution_count = data.get("evolution_count", 0)
        matrix.wumbo_phase_index = data.get("wumbo_phase_index", 0)
        matrix.global_kappa = data.get("global_kappa", PHI_INV)
        matrix.global_lambda = data.get("global_lambda", PHI_INV_SQ)
        matrix.rows = [OperatorRow.from_dict(r) for r in data.get("rows", [])]
        return matrix

    @classmethod
    def from_json(cls, json_str: str) -> "AdaptiveOperatorMatrix":
        """Create matrix from JSON string."""
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def import_from_file(cls, filepath: str) -> "AdaptiveOperatorMatrix":
        """Import matrix from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            return cls.from_json(f.read())

    def export_operators_list(self) -> List[Dict[str, Any]]:
        """Export flat list of all operators (for appending elsewhere)."""
        result = []
        for row in self.rows:
            for op in row.operators:
                export = op.to_dict()
                export["row_index"] = row.index
                export["row_name"] = row.name
                result.append(export)
        return result

    def export_glyph_sequence(self) -> str:
        """Export all glyphs as a sequence string."""
        glyphs = []
        for row in self.rows:
            for op in row.operators:
                glyphs.append(op.glyph)
        return "".join(glyphs)

    def compute_matrix_hash(self) -> str:
        """Compute hash of current matrix state."""
        content = self.to_json(indent=0)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    # =========================================================================
    # HISTORY / UNDO
    # =========================================================================

    def _save_history(self, operation: str) -> None:
        """Save current state to history."""
        if len(self._history) >= self._max_history:
            self._history.pop(0)

        self._history.append({
            "operation": operation,
            "timestamp": datetime.now().isoformat(),
            "state": self.to_dict(),
        })

    def undo(self) -> bool:
        """Undo last operation."""
        if len(self._history) < 2:
            return False

        # Pop current state
        self._history.pop()

        # Restore previous state
        if self._history:
            prev = self._history[-1]["state"]
            self.rows = [OperatorRow.from_dict(r) for r in prev["rows"]]
            self.evolution_count = prev["evolution_count"]
            self.wumbo_phase_index = prev["wumbo_phase_index"]
            self.global_kappa = prev["global_kappa"]
            self.global_lambda = prev["global_lambda"]
            return True

        return False

    # =========================================================================
    # DISPLAY
    # =========================================================================

    def __repr__(self) -> str:
        return f"AdaptiveOperatorMatrix(rows={self.num_rows}, operators={self.num_operators}, phase={self.current_wumbo_phase})"

    def display(self) -> str:
        """Generate human-readable matrix display."""
        lines = []
        lines.append("=" * 70)
        lines.append("  ADAPTIVE OPERATOR MATRIX")
        lines.append(f"  Phase: {self.current_wumbo_phase} | Evolution: {self.evolution_count}")
        lines.append(f"  κ={self.global_kappa:.4f} (φ⁻¹={PHI_INV:.4f}) | λ={self.global_lambda:.4f} (φ⁻²={PHI_INV_SQ:.4f})")
        lines.append(f"  Coupling Conservation: κ+λ = {self.global_kappa + self.global_lambda:.6f} (target: 1.0)")
        lines.append("=" * 70)

        for row in self.rows:
            frozen_mark = "🔒" if row.frozen else "  "
            lines.append(f"\n{frozen_mark} Row {row.index}: {row.name} [{row.coupling_mode.value}]")
            lines.append("-" * 50)

            for op in row.operators:
                balance_mark = "⚖" if op.is_balanced else " "
                lines.append(
                    f"    {op.glyph} {op.name:12s} w={op.weight:.2f} "
                    f"κ={op.kappa_coupling:.2f} λ={op.lambda_coupling:.2f} {balance_mark}"
                )

        lines.append("\n" + "=" * 70)
        return "\n".join(lines)


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_adaptive_matrix(with_foundation: bool = True) -> AdaptiveOperatorMatrix:
    """
    Create a new adaptive operator matrix.

    Args:
        with_foundation: Include foundation operators

    Returns:
        Initialized AdaptiveOperatorMatrix
    """
    return AdaptiveOperatorMatrix(initialize_foundation=with_foundation)


def create_empty_matrix() -> AdaptiveOperatorMatrix:
    """Create an empty matrix (no foundation rows)."""
    return AdaptiveOperatorMatrix(initialize_foundation=False)


def create_operator(
    glyph: str,
    name: str,
    domain: str = "UNIVERSAL",
    weight: float = 1.0,
    kappa: float = PHI_INV,
    lambda_: float = PHI_INV_SQ,
) -> Operator:
    """
    Create a standalone operator with golden balance defaults.

    Args:
        glyph: APL glyph
        name: Operator name
        domain: Domain string (SCALAR, VECTOR, MATRIX, TENSOR, FIELD, UNIVERSAL)
        weight: Coupling weight
        kappa: κ-field coupling (default φ⁻¹)
        lambda_: λ-field coupling (default φ⁻²)

    Returns:
        New Operator instance
    """
    return Operator(
        glyph=glyph,
        name=name,
        op_type=OperatorType.ADAPTIVE,
        domain=OperatorDomain[domain],
        weight=weight,
        kappa_coupling=kappa,
        lambda_coupling=lambda_,
    )


# =============================================================================
# DEMO
# =============================================================================

def demonstrate_matrix():
    """Demonstrate adaptive operator matrix capabilities."""
    print("=" * 70)
    print("  ADAPTIVE OPERATOR MATRIX DEMO")
    print("  (Aligned with verify_physics.py)")
    print("=" * 70)

    # Verify coupling conservation
    print(f"\n  PHYSICS CONSTANTS:")
    print(f"    φ⁻¹ = {PHI_INV:.10f}")
    print(f"    φ⁻² = {PHI_INV_SQ:.10f}")
    print(f"    φ⁻¹ + φ⁻² = {COUPLING_CONSERVATION:.10f} (should be 1.0)")

    # Create matrix
    matrix = create_adaptive_matrix()
    print(f"\n1. Created matrix: {matrix}")
    print(f"   Shape: {matrix.shape}")
    print(f"   Global κ={matrix.global_kappa:.4f}, λ={matrix.global_lambda:.4f}")
    print(f"   Conservation error: {matrix.coupling_conservation_error:.2e}")
    print(f"   Glyph sequence: {matrix.export_glyph_sequence()}")

    # Append new row
    print("\n2. Appending custom row...")
    row = matrix.append_row("Custom-ZPE", OperatorType.ADAPTIVE, CouplingMode.RESONANT)
    print(f"   Created: Row {row.index} '{row.name}'")

    # Append operators
    print("\n3. Appending operators (with golden balance defaults)...")
    op1 = matrix.append_operator(row.index, "⌽", "flip", OperatorDomain.VECTOR, 0.9)
    op2 = matrix.append_operator(row.index, "⍉", "transpose", OperatorDomain.MATRIX, 0.85)
    print(f"   Added: {op1}")
    print(f"   Added: {op2}")

    # Compose operators
    print("\n4. Composing operators...")
    composite = matrix.compose("⊖", "⍧", "⟲", "spiral-rotate")
    print(f"   Composed: {composite}")
    print(f"   κ+λ = {composite.kappa_coupling + composite.lambda_coupling:.6f}")

    # Evolve
    print("\n5. Evolving through WUMBO cycles...")
    stats = matrix.evolve(cycles=6)
    print(f"   Initial phase: {stats['initial_phase']}")
    print(f"   Final phase: {stats['final_phase']}")
    print(f"   Operators evolved: {stats['operators_evolved']}")
    print(f"   Balance: {stats['balance_before']:.4f} → {stats['balance_after']:.4f}")
    print(f"   Conservation error: {stats['conservation_error_before']:.2e} → {stats['conservation_error_after']:.2e}")

    # Export
    print("\n6. Exporting matrix...")
    json_str = matrix.to_json()
    print(f"   JSON length: {len(json_str)} chars")
    print(f"   Matrix hash: {matrix.compute_matrix_hash()}")

    # Display
    print("\n7. Matrix state:")
    print(matrix.display())

    print("\n" + "=" * 70)
    print("  ✓ Adaptive Operator Matrix demo complete")
    print("  ✓ Coupling conservation maintained throughout")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_matrix()
