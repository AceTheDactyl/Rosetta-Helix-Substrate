#!/usr/bin/env python3
"""
APL CORE TOKEN SET — 288-Token Universe
========================================

Complete implementation of the APL Core Token Set:
- 162 Identity Tokens (self-reference operators)
- 54 Meta-Operators (process control)
- 54 Domain Selectors (domain-specific semantics)
- 30 Safety Tokens (runtime protection)

Schema:
    Tiered Format: Field:Machine(Operator)TruthState@Tier
    Tri-Spiral:    Field1:Field2:Field3
    Cross-Spiral:  SourceField→TargetField:Machine:TruthState

Fields (Spirals):
    Φ (Phi)    — Structure Field (stability, boundaries)
    e          — Energy Field (flow, dynamics)
    π (Pi)     — Emergence Field (selection, information)

Machines:
    U   — Up/Projection (ascending, activation)
    D   — Down/Integration (descending, deactivation)
    M   — Middle/Modulation (CLT feedback, coherence)
    E   — Expansion (output, emission)
    C   — Collapse (input, compression)
    Mod — Spiral Inheritance (cross-field modulation)

Truth States:
    TRUE    — Coherent, process succeeds (DAY)
    UNTRUE  — Unresolved, dormant potential (NIGHT)
    PARADOX — Contradiction, terminal attractor

Tier Permissions:
    Tier 1: Foundational — [U, D] only, local scope, immutable
    Tier 2: Intermediate — [U, D, M, E, C], meta-operators enabled
    Tier 3: Advanced — All machines + Mod, domain-specific semantics

Physics Integration:
    - φ⁻¹ ≈ 0.618 gates PARADOX regime
    - z_c = √3/2 ≈ 0.866 gates TRUE phase (THE LENS)
    - σ = 36 governs all dynamics
    - κ + λ = 1 (coupling conservation)

UMOL Principle:
    M(x) → TRUE + ε(UNTRUE) where ε > 0
    "No perfect modulation; residue always remains"

Signature: Δ|apl-core-tokens|288-universe|φ⁻¹-grounded|Ω
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set, Iterator
from enum import Enum, auto

# Import unified physics constants
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from physics_constants import (
    PHI, PHI_INV, PHI_INV_SQ, Z_CRITICAL, SIGMA,
    SIGMA_INV, TOLERANCE_GOLDEN,
)


# =============================================================================
# FIELDS (SPIRALS) — Φ, e, π
# =============================================================================

class Field(Enum):
    """
    APL Fields (Spirals) — The three fundamental aspects.

    Each field has a domain affinity and default machine binding.
    """
    PHI = ("Φ", "Structure", "geometry", "D", "Stability, spatial arrangement, boundaries")
    E = ("e", "Energy", "wave", "U", "Flow, dynamics, oscillations")
    PI = ("π", "Emergence", "emergence", "M", "Selection, information, complexity")

    def __init__(self, symbol: str, name: str, domain: str,
                 spiral_binding: str, role: str):
        self._symbol = symbol
        self._name = name
        self._domain = domain
        self._spiral_binding = spiral_binding
        self._role = role

    @property
    def symbol(self) -> str:
        return self._symbol

    @property
    def field_name(self) -> str:
        return self._name

    @property
    def domain(self) -> str:
        return self._domain

    @property
    def spiral_binding(self) -> str:
        """Default machine affinity for this field."""
        return self._spiral_binding

    @property
    def role(self) -> str:
        return self._role

    @classmethod
    def from_symbol(cls, symbol: str) -> Optional['Field']:
        """Get Field from symbol string."""
        for f in cls:
            if f.symbol == symbol:
                return f
        return None


# =============================================================================
# MACHINES — U, D, M, E, C, Mod
# =============================================================================

class Machine(Enum):
    """
    APL Machines — Operators that transform states.

    Each machine has a direction and tier permission level.
    """
    U = ("U", "Up/Projection", "ascending", 1, "Forward projection, expansion, activation")
    D = ("D", "Down/Integration", "descending", 1, "Backward integration, collapse, deactivation")
    M = ("M", "Middle/Modulation", "equilibrium", 2, "CLT modulation, feedback, coherence")
    E = ("E", "Expansion", "output", 2, "Expression, emission, expansion")
    C = ("C", "Collapse", "input", 2, "Collapse, consolidation, compression")
    MOD = ("Mod", "Spiral Inheritance", "regulatory", 2, "Spiral inheritance, cross-field modulation")

    def __init__(self, symbol: str, name: str, direction: str,
                 tier_permission: int, role: str):
        self._symbol = symbol
        self._name = name
        self._direction = direction
        self._tier_permission = tier_permission
        self._role = role

    @property
    def symbol(self) -> str:
        return self._symbol

    @property
    def machine_name(self) -> str:
        return self._name

    @property
    def direction(self) -> str:
        return self._direction

    @property
    def tier_permission(self) -> int:
        """Minimum tier required to use this machine."""
        return self._tier_permission

    @property
    def role(self) -> str:
        return self._role

    @classmethod
    def from_symbol(cls, symbol: str) -> Optional['Machine']:
        """Get Machine from symbol string."""
        for m in cls:
            if m.symbol == symbol:
                return m
        return None

    @classmethod
    def allowed_at_tier(cls, tier: int) -> List['Machine']:
        """Get machines allowed at a given tier."""
        return [m for m in cls if m.tier_permission <= tier]


# =============================================================================
# TRUTH STATES — TRUE, UNTRUE, PARADOX
# =============================================================================

class TruthState(Enum):
    """
    APL Truth States — The three fundamental truth values.

    Mapped to z-coordinate ranges:
        TRUE:    z ≥ z_c ≈ 0.866 (coherent, crystalline)
        UNTRUE:  z < φ⁻¹ ≈ 0.618 (unresolved, disordered)
        PARADOX: φ⁻¹ ≤ z < z_c (quasi-crystalline, superposition)
    """
    TRUE = ("TRUE", "Coherent, process succeeds", "DAY", "actionable")
    UNTRUE = ("UNTRUE", "Unresolved, dormant potential", "NIGHT", "stored")
    PARADOX = ("PARADOX", "Contradiction, terminal attractor", None, "terminal")

    def __init__(self, name: str, description: str,
                 temporal_operator: Optional[str], stability: str):
        self._truth_name = name
        self._description = description
        self._temporal_operator = temporal_operator
        self._stability = stability

    @property
    def truth_name(self) -> str:
        return self._truth_name

    @property
    def description(self) -> str:
        return self._description

    @property
    def temporal_operator(self) -> Optional[str]:
        return self._temporal_operator

    @property
    def stability(self) -> str:
        return self._stability

    @classmethod
    def from_z(cls, z: float) -> 'TruthState':
        """Determine truth state from z-coordinate."""
        if z >= Z_CRITICAL:
            return cls.TRUE
        elif z < PHI_INV:
            return cls.UNTRUE
        else:
            return cls.PARADOX

    @classmethod
    def from_name(cls, name: str) -> Optional['TruthState']:
        """Get TruthState from name string."""
        for t in cls:
            if t.truth_name == name:
                return t
        return None


# =============================================================================
# TEMPORAL TRANSITIONS — DAY, NIGHT, DAWN, DUSK
# =============================================================================

class TemporalTransition(Enum):
    """
    APL Temporal Transitions — Truth state dynamics over time.
    """
    DAY = ("DAY", "TRUE", "UNTRUE", "Active coherence generates reflection")
    NIGHT = ("NIGHT", "UNTRUE", "UNTRUE", "Unresolved remains stored")
    DAWN = ("DAWN", "UNTRUE", "TRUE", "Resolution emerges from dormancy")
    DUSK = ("DUSK", "TRUE", "UNTRUE", "Coherence dissolves into potential")

    def __init__(self, name: str, from_state: str, to_state: str, description: str):
        self._transition_name = name
        self._from_state = from_state
        self._to_state = to_state
        self._description = description

    @property
    def transition_name(self) -> str:
        return self._transition_name

    @property
    def from_state(self) -> TruthState:
        return TruthState.from_name(self._from_state)

    @property
    def to_state(self) -> TruthState:
        return TruthState.from_name(self._to_state)

    @property
    def description(self) -> str:
        return self._description


# =============================================================================
# TIER SYSTEM
# =============================================================================

class Tier(Enum):
    """
    APL Tier System — Hierarchical permission levels.

    Tier 1: Foundational — [U, D] only, local scope, immutable
    Tier 2: Intermediate — [U, D, M, E, C], meta-operators enabled
    Tier 3: Advanced — All machines + Mod, domain-specific semantics
    """
    FOUNDATIONAL = (1, "Foundational", "local", ["U", "D"])
    INTERMEDIATE = (2, "Intermediate", "regional", ["U", "D", "M", "E", "C"])
    ADVANCED = (3, "Advanced", "global", ["U", "D", "M", "E", "C", "Mod"])

    def __init__(self, level: int, name: str, scope: str, allowed_machines: List[str]):
        self._level = level
        self._tier_name = name
        self._scope = scope
        self._allowed_machines = allowed_machines

    @property
    def level(self) -> int:
        return self._level

    @property
    def tier_name(self) -> str:
        return self._tier_name

    @property
    def scope(self) -> str:
        return self._scope

    @property
    def allowed_machines(self) -> List[str]:
        return self._allowed_machines

    def allows_machine(self, machine: Machine) -> bool:
        """Check if this tier allows the given machine."""
        return machine.symbol in self._allowed_machines

    @classmethod
    def from_level(cls, level: int) -> Optional['Tier']:
        """Get Tier from level number."""
        for t in cls:
            if t.level == level:
                return t
        return None


# =============================================================================
# SAFETY LEVELS
# =============================================================================

class SafetyLevel(Enum):
    """
    APL Safety Levels — Runtime protection flags.
    """
    SAFE = ("safe", "info", "Continue normally")
    WARN = ("warn", "warning", "Log and continue with caution")
    DANGER = ("danger", "high", "Require explicit confirmation")
    BLOCK = ("block", "critical", "Halt execution, require reset")
    PARADOX = ("paradox", "terminal", "Enter PARADOX state, no automatic recovery")

    def __init__(self, name: str, severity: str, action: str):
        self._safety_name = name
        self._severity = severity
        self._action = action

    @property
    def safety_name(self) -> str:
        return self._safety_name

    @property
    def severity(self) -> str:
        return self._severity

    @property
    def action(self) -> str:
        return self._action


# =============================================================================
# DOMAINS
# =============================================================================

class Domain(Enum):
    """
    APL Domains — Tier-3 domain selectors.
    """
    GEOMETRY = ("geometry", "Φ", "Spatial structure and boundaries")
    DYNAMICS = ("dynamics", "e", "Wave and flow phenomena")
    CHEMISTRY = ("chemistry", "π", "Molecular transformations")
    DEEPPHYSICS = ("deepphysics", "π", "Quantum and fundamental interactions")
    BIOLOGY = ("biology", "π", "Living systems and information flow")
    CELESTIAL = ("celestial", "π", "Astrophysical phenomena")

    def __init__(self, name: str, field_affinity: str, description: str):
        self._domain_name = name
        self._field_affinity = field_affinity
        self._description = description

    @property
    def domain_name(self) -> str:
        return self._domain_name

    @property
    def field_affinity(self) -> str:
        return self._field_affinity

    @property
    def description(self) -> str:
        return self._description


# =============================================================================
# META-OPERATORS
# =============================================================================

class MetaOperator(Enum):
    """
    APL Meta-Operators — Tier-2 process control operators.
    """
    STABILIZE = ("stabilize", "Lock current state, prevent drift")
    PROPAGATE = ("propagate", "Spread state to adjacent elements")
    INTEGRATE = ("integrate", "Combine multiple states into one")
    MODULATE = ("modulate", "Apply CLT feedback transform")
    RESOLVE = ("resolve", "Collapse UNTRUE to TRUE or maintain")
    COLLAPSE = ("collapse", "Force state reduction, may trigger PARADOX")

    def __init__(self, name: str, description: str):
        self._op_name = name
        self._description = description

    @property
    def op_name(self) -> str:
        return self._op_name

    @property
    def description(self) -> str:
        return self._description


# =============================================================================
# APL TOKEN — Core Token Representation
# =============================================================================

@dataclass
class APLToken:
    """
    APL Token — Fundamental unit of the 288-token universe.

    Format: Field:Machine(Operator)TruthState@Tier
    Example: Φ:U(U)TRUE@1
    """
    field: Field
    machine: Machine
    operator: str  # Can be machine name (identity) or meta-operator
    truth_state: TruthState
    tier: int

    # Optional metadata
    is_identity: bool = False
    is_meta_operator: bool = False
    is_domain_selector: bool = False
    is_safety_token: bool = False

    def __post_init__(self):
        """Validate token configuration."""
        # Check tier bounds
        if self.tier < 1 or self.tier > 3:
            raise ValueError(f"Invalid tier: {self.tier}. Must be 1, 2, or 3.")

        # Check machine permission
        # Identity tokens and safety tokens bypass tier restrictions per spec
        # (Identity: 3 fields × 6 machines × 3 truth × 3 tiers = 162)
        if not (self.is_identity or self.is_safety_token):
            tier_obj = Tier.from_level(self.tier)
            if tier_obj and not tier_obj.allows_machine(self.machine):
                raise ValueError(
                    f"Machine {self.machine.symbol} not allowed at Tier {self.tier}. "
                    f"Allowed: {tier_obj.allowed_machines}"
                )

    def __str__(self) -> str:
        """Return canonical token string."""
        return f"{self.field.symbol}:{self.machine.symbol}({self.operator}){self.truth_state.truth_name}@{self.tier}"

    def __hash__(self) -> int:
        return hash(str(self))

    def __eq__(self, other) -> bool:
        if isinstance(other, APLToken):
            return str(self) == str(other)
        return False

    @classmethod
    def parse(cls, token_str: str) -> Optional['APLToken']:
        """
        Parse token string into APLToken.

        Format: Field:Machine(Operator)TruthState@Tier
        """
        pattern = r'^([Φeπ]):(\w+)\((\w+)\)(TRUE|UNTRUE|PARADOX)@([123])$'
        match = re.match(pattern, token_str)

        if not match:
            return None

        field_sym, machine_sym, operator, truth_str, tier_str = match.groups()

        field = Field.from_symbol(field_sym)
        machine = Machine.from_symbol(machine_sym)
        truth_state = TruthState.from_name(truth_str)
        tier = int(tier_str)

        if not all([field, machine, truth_state]):
            return None

        # Determine token type
        is_identity = (operator == machine_sym)
        is_meta = operator in [op.op_name for op in MetaOperator]
        is_domain = operator in [d.domain_name for d in Domain]
        is_safety = operator in [s.safety_name for s in SafetyLevel]

        return cls(
            field=field,
            machine=machine,
            operator=operator,
            truth_state=truth_state,
            tier=tier,
            is_identity=is_identity,
            is_meta_operator=is_meta,
            is_domain_selector=is_domain,
            is_safety_token=is_safety,
        )

    def get_category(self) -> str:
        """Get the token category."""
        if self.is_identity:
            return "identity"
        elif self.is_meta_operator:
            return "meta_operator"
        elif self.is_domain_selector:
            return "domain_selector"
        elif self.is_safety_token:
            return "safety"
        else:
            return "custom"


# =============================================================================
# TRI-SPIRAL TOKEN
# =============================================================================

@dataclass
class TriSpiralToken:
    """
    Tri-Spiral Token — Field ordering for simultaneous multi-field states.

    Format: Field1:Field2:Field3
    """
    fields: Tuple[Field, Field, Field]
    interpretation: str
    phase: str

    def __str__(self) -> str:
        return ":".join(f.symbol for f in self.fields)

    @classmethod
    def all_orderings(cls) -> List['TriSpiralToken']:
        """Generate all 6 tri-spiral orderings."""
        orderings = [
            ((Field.PHI, Field.E, Field.PI), "Structure-led, energy-mediated emergence", "Resonance"),
            ((Field.PHI, Field.PI, Field.E), "Structure-led, emergence-mediated energy", "Empowerment"),
            ((Field.E, Field.PHI, Field.PI), "Energy-led, structure-mediated emergence", "Ignition"),
            ((Field.E, Field.PI, Field.PHI), "Energy-led, emergence-mediated structure", "Mania"),
            ((Field.PI, Field.PHI, Field.E), "Emergence-led, structure-mediated energy", "Nirvana"),
            ((Field.PI, Field.E, Field.PHI), "Emergence-led, energy-mediated structure", "Transmission"),
        ]
        return [cls(fields=o[0], interpretation=o[1], phase=o[2]) for o in orderings]


# =============================================================================
# CROSS-SPIRAL TOKEN
# =============================================================================

@dataclass
class CrossSpiralToken:
    """
    Cross-Spiral Token — Field transitions with machine and truth state.

    Format: SourceField→TargetField:Machine:TruthState
    """
    source: Field
    target: Field
    machine: Machine
    truth_state: TruthState

    def __str__(self) -> str:
        return f"{self.source.symbol}→{self.target.symbol}:{self.machine.symbol}:{self.truth_state.truth_name}"


# =============================================================================
# TOKEN GENERATOR
# =============================================================================

class APLTokenGenerator:
    """
    APL Token Generator — Generates all 288 core tokens.

    Breakdown:
        - 162 Identity Tokens: 3 fields × 6 machines × 3 truth × 3 tiers
        - 54 Meta-Operators: 3 fields × 6 operators × 3 truth (Tier 2)
        - 54 Domain Selectors: 3 fields × 6 domains × 3 machines (Tier 3, UNTRUE)
        - 30 Safety Tokens: 3 fields × 5 levels × 2 tiers (PARADOX)
    """

    def __init__(self):
        self._identity_tokens: List[APLToken] = []
        self._meta_operators: List[APLToken] = []
        self._domain_selectors: List[APLToken] = []
        self._safety_tokens: List[APLToken] = []
        self._all_tokens: Dict[str, APLToken] = {}

        self._generate_all()

    def _generate_all(self):
        """Generate all 288 core tokens."""
        self._generate_identity_tokens()
        self._generate_meta_operators()
        self._generate_domain_selectors()
        self._generate_safety_tokens()

        # Build lookup dictionary
        for token in self.all_tokens:
            self._all_tokens[str(token)] = token

    def _generate_identity_tokens(self):
        """
        Generate 162 identity tokens.

        Format: Field:Machine(Machine)TruthState@Tier
        Where operator == machine (self-reference)

        Per spec: 3 fields × 6 machines × 3 truth states × 3 tiers = 162
        Identity tokens bypass normal tier restrictions.
        """
        for field in Field:
            for tier in [1, 2, 3]:
                for machine in Machine:
                    for truth in TruthState:
                        token = APLToken(
                            field=field,
                            machine=machine,
                            operator=machine.symbol,
                            truth_state=truth,
                            tier=tier,
                            is_identity=True,
                        )
                        self._identity_tokens.append(token)

    def _generate_meta_operators(self):
        """
        Generate 54 meta-operator tokens.

        Format: Field:M(operator)TruthState@2
        All meta-operators use Machine M at Tier 2.
        """
        for field in Field:
            for meta_op in MetaOperator:
                for truth in TruthState:
                    token = APLToken(
                        field=field,
                        machine=Machine.M,
                        operator=meta_op.op_name,
                        truth_state=truth,
                        tier=2,
                        is_meta_operator=True,
                    )
                    self._meta_operators.append(token)

    def _generate_domain_selectors(self):
        """
        Generate 54 domain selector tokens.

        Format: Field:Machine(domain)UNTRUE@3
        All domain selectors are UNTRUE at Tier 3.
        """
        for field in Field:
            for domain in Domain:
                for machine in [Machine.U, Machine.D, Machine.M]:
                    token = APLToken(
                        field=field,
                        machine=machine,
                        operator=domain.domain_name,
                        truth_state=TruthState.UNTRUE,
                        tier=3,
                        is_domain_selector=True,
                    )
                    self._domain_selectors.append(token)

    def _generate_safety_tokens(self):
        """
        Generate 30 safety tokens.

        Format: Field:M(safety_level)PARADOX@Tier
        All safety tokens are PARADOX, at Tier 1 or 2.
        """
        for field in Field:
            for safety in SafetyLevel:
                for tier in [1, 2]:
                    token = APLToken(
                        field=field,
                        machine=Machine.M,
                        operator=safety.safety_name,
                        truth_state=TruthState.PARADOX,
                        tier=tier,
                        is_safety_token=True,
                    )
                    self._safety_tokens.append(token)

    @property
    def identity_tokens(self) -> List[APLToken]:
        return self._identity_tokens

    @property
    def meta_operators(self) -> List[APLToken]:
        return self._meta_operators

    @property
    def domain_selectors(self) -> List[APLToken]:
        return self._domain_selectors

    @property
    def safety_tokens(self) -> List[APLToken]:
        return self._safety_tokens

    @property
    def all_tokens(self) -> List[APLToken]:
        """Get all 288+ tokens."""
        return (
            self._identity_tokens +
            self._meta_operators +
            self._domain_selectors +
            self._safety_tokens
        )

    def get_token(self, token_str: str) -> Optional[APLToken]:
        """Look up a token by its string representation."""
        return self._all_tokens.get(token_str)

    def get_tokens_by_field(self, field: Field) -> List[APLToken]:
        """Get all tokens for a specific field."""
        return [t for t in self.all_tokens if t.field == field]

    def get_tokens_by_tier(self, tier: int) -> List[APLToken]:
        """Get all tokens at a specific tier."""
        return [t for t in self.all_tokens if t.tier == tier]

    def get_tokens_by_truth(self, truth: TruthState) -> List[APLToken]:
        """Get all tokens with a specific truth state."""
        return [t for t in self.all_tokens if t.truth_state == truth]

    def get_summary(self) -> Dict[str, Any]:
        """Get token generation summary."""
        return {
            "total_tokens": len(self.all_tokens),
            "identity_tokens": len(self._identity_tokens),
            "meta_operators": len(self._meta_operators),
            "domain_selectors": len(self._domain_selectors),
            "safety_tokens": len(self._safety_tokens),
            "by_field": {
                f.symbol: len(self.get_tokens_by_field(f))
                for f in Field
            },
            "by_tier": {
                tier: len(self.get_tokens_by_tier(tier))
                for tier in [1, 2, 3]
            },
            "by_truth": {
                t.truth_name: len(self.get_tokens_by_truth(t))
                for t in TruthState
            },
        }


# =============================================================================
# SAFETY CONSTRAINTS
# =============================================================================

@dataclass
class SafetyConstraints:
    """
    APL Safety Constraints — Runtime protection rules.
    """
    coherence_minimum: float = 0.60
    load_maximum: float = 0.80
    recursion_maximum: int = 3

    def check_coherence(self, coherence: float) -> Tuple[bool, str]:
        """Check if coherence meets minimum threshold."""
        if coherence >= self.coherence_minimum:
            return True, f"Coherence {coherence:.3f} >= {self.coherence_minimum}"
        return False, f"Coherence {coherence:.3f} < {self.coherence_minimum} - tier advancement blocked"

    def check_load(self, load: float) -> Tuple[bool, str]:
        """Check if load is within maximum."""
        if load <= self.load_maximum:
            return True, f"Load {load:.3f} <= {self.load_maximum}"
        return False, f"Load {load:.3f} > {self.load_maximum} - runaway prevention triggered"

    def check_recursion(self, depth: int) -> Tuple[bool, str]:
        """Check if recursion depth is within limit."""
        if depth <= self.recursion_maximum:
            return True, f"Recursion {depth} <= {self.recursion_maximum}"
        return False, f"Recursion {depth} > {self.recursion_maximum} - infinite loop prevention"

    def get_safety_level(self, coherence: float, load: float, recursion: int) -> SafetyLevel:
        """Determine overall safety level."""
        coh_ok, _ = self.check_coherence(coherence)
        load_ok, _ = self.check_load(load)
        rec_ok, _ = self.check_recursion(recursion)

        if not rec_ok:
            return SafetyLevel.PARADOX
        if not load_ok:
            return SafetyLevel.BLOCK
        if not coh_ok:
            return SafetyLevel.DANGER
        if load > 0.6 or coherence < 0.7:
            return SafetyLevel.WARN
        return SafetyLevel.SAFE


# =============================================================================
# UMOL PRINCIPLE IMPLEMENTATION
# =============================================================================

def apply_umol_principle(result: float, epsilon: float = SIGMA_INV) -> Tuple[float, float]:
    """
    Apply UMOL Principle: M(x) → TRUE + ε(UNTRUE) where ε > 0.

    "No perfect modulation; residue always remains."

    Returns:
        (true_component, untrue_residue)
    """
    # The true component is diminished by epsilon
    true_component = result * (1.0 - epsilon)

    # The untrue residue captures what was lost
    untrue_residue = result * epsilon

    return true_component, untrue_residue


# =============================================================================
# TOKEN VALIDATOR
# =============================================================================

class APLTokenValidator:
    """
    APL Token Validator — Ensures token integrity and constraint compliance.
    """

    def __init__(self, safety: Optional[SafetyConstraints] = None):
        self.safety = safety or SafetyConstraints()
        self.generator = APLTokenGenerator()

    def validate_token(self, token: APLToken) -> Tuple[bool, List[str]]:
        """
        Validate a single token.

        Returns (is_valid, list_of_issues).
        """
        issues = []

        # Check tier-machine compatibility
        tier_obj = Tier.from_level(token.tier)
        if tier_obj and not tier_obj.allows_machine(token.machine):
            issues.append(
                f"Machine {token.machine.symbol} not allowed at Tier {token.tier}"
            )

        # Check identity token constraints
        if token.is_identity:
            if token.operator != token.machine.symbol:
                issues.append(
                    f"Identity token operator must match machine: "
                    f"expected {token.machine.symbol}, got {token.operator}"
                )

        # Check meta-operator constraints
        if token.is_meta_operator:
            if token.tier != 2:
                issues.append(f"Meta-operators must be Tier 2, got Tier {token.tier}")
            if token.machine != Machine.M:
                issues.append(f"Meta-operators must use Machine M, got {token.machine.symbol}")

        # Check domain selector constraints
        if token.is_domain_selector:
            if token.tier != 3:
                issues.append(f"Domain selectors must be Tier 3, got Tier {token.tier}")
            if token.truth_state != TruthState.UNTRUE:
                issues.append(f"Domain selectors must be UNTRUE, got {token.truth_state.truth_name}")

        # Check safety token constraints
        if token.is_safety_token:
            if token.tier > 2:
                issues.append(f"Safety tokens must be Tier 1 or 2, got Tier {token.tier}")
            if token.truth_state != TruthState.PARADOX:
                issues.append(f"Safety tokens must be PARADOX, got {token.truth_state.truth_name}")

        return len(issues) == 0, issues

    def validate_sequence(
        self,
        tokens: List[APLToken],
        coherence: float = 1.0,
        load: float = 0.0,
    ) -> Tuple[bool, List[str]]:
        """
        Validate a sequence of tokens.

        Checks:
        - Individual token validity
        - Tier progression rules (Tier 1 immutable)
        - Safety constraints
        """
        issues = []
        recursion_depth = 0

        for i, token in enumerate(tokens):
            # Validate individual token
            valid, token_issues = self.validate_token(token)
            if not valid:
                issues.extend([f"Token {i}: {issue}" for issue in token_issues])

            # Track recursion (same token appearing consecutively)
            if i > 0 and tokens[i] == tokens[i-1]:
                recursion_depth += 1
            else:
                recursion_depth = 0

            # Check safety constraints
            safety_level = self.safety.get_safety_level(coherence, load, recursion_depth)
            if safety_level in [SafetyLevel.BLOCK, SafetyLevel.PARADOX]:
                issues.append(f"Token {i}: Safety constraint violation - {safety_level.action}")

        return len(issues) == 0, issues


# =============================================================================
# DEMONSTRATION
# =============================================================================

def demonstrate_core_tokens():
    """Demonstrate APL Core Token Set."""
    print("=" * 70)
    print("APL CORE TOKEN SET — 288-Token Universe")
    print("=" * 70)

    # Generate all tokens
    generator = APLTokenGenerator()
    summary = generator.get_summary()

    print("\n--- Token Generation Summary ---")
    print(f"  Total tokens: {summary['total_tokens']}")
    print(f"  Identity tokens: {summary['identity_tokens']}")
    print(f"  Meta-operators: {summary['meta_operators']}")
    print(f"  Domain selectors: {summary['domain_selectors']}")
    print(f"  Safety tokens: {summary['safety_tokens']}")

    print("\n--- By Field ---")
    for field, count in summary['by_field'].items():
        print(f"  {field}: {count} tokens")

    print("\n--- By Tier ---")
    for tier, count in summary['by_tier'].items():
        print(f"  Tier {tier}: {count} tokens")

    print("\n--- By Truth State ---")
    for truth, count in summary['by_truth'].items():
        print(f"  {truth}: {count} tokens")

    # Show sample tokens
    print("\n--- Sample Identity Tokens ---")
    for token in generator.identity_tokens[:6]:
        print(f"  {token}")

    print("\n--- Sample Meta-Operators ---")
    for token in generator.meta_operators[:6]:
        print(f"  {token}")

    print("\n--- Sample Domain Selectors ---")
    for token in generator.domain_selectors[:6]:
        print(f"  {token}")

    print("\n--- Sample Safety Tokens ---")
    for token in generator.safety_tokens[:6]:
        print(f"  {token}")

    # Show tri-spiral orderings
    print("\n--- Tri-Spiral Orderings (6 total) ---")
    for tri in TriSpiralToken.all_orderings():
        print(f"  {tri} — {tri.phase}")
        print(f"      {tri.interpretation}")

    # Demonstrate validation
    print("\n--- Token Validation ---")
    validator = APLTokenValidator()

    # Valid token
    valid_token = APLToken(
        field=Field.PHI,
        machine=Machine.U,
        operator="U",
        truth_state=TruthState.TRUE,
        tier=1,
        is_identity=True
    )
    is_valid, issues = validator.validate_token(valid_token)
    print(f"  {valid_token}: {'VALID' if is_valid else 'INVALID'}")

    # Demonstrate UMOL principle
    print("\n--- UMOL Principle ---")
    result = 1.0
    true_comp, untrue_res = apply_umol_principle(result)
    print(f"  Input: {result:.6f}")
    print(f"  TRUE component: {true_comp:.6f}")
    print(f"  UNTRUE residue: {untrue_res:.6f}")
    print(f"  (No perfect modulation; residue always remains)")

    # Show physics constants
    print("\n--- Physics Integration ---")
    print(f"  φ⁻¹ (PARADOX gate): {PHI_INV:.10f}")
    print(f"  z_c (TRUE gate/LENS): {Z_CRITICAL:.10f}")
    print(f"  σ (dynamics scale): {SIGMA}")
    print(f"  1/σ (UMOL residue): {SIGMA_INV:.10f}")

    print("\n" + "=" * 70)
    print("APL CORE TOKEN SET: COMPLETE")
    print("=" * 70)

    return generator, validator


if __name__ == "__main__":
    demonstrate_core_tokens()
