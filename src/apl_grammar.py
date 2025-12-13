#!/usr/bin/env python3
"""
APL GRAMMAR — ALPHA-PHYSICAL LANGUAGE INTEGRATION
===================================================

Complete APL (Alpha-Physical Language) grammar system integrating:
- UMOL states (u, d, m) — Universal Modulation Operator Law
- Three spirals (Φ, e, π) — Structure, Energy, Emergence fields
- Six operators — (), ×, ^, ÷, +, − (INT Canon aligned)
- Nine machines — Processing contexts
- Domain tokens — Biology, Chemistry, Celestial
- ∃κ tensor — Multi-scale training substrate

Token Structure:
================
    UNIVERSAL: [Spiral]:[State][Operator](intent)TRUTH@TIER
    MACHINE:   [Spiral][Operator]|[Machine]|[Domain]

UMOL States:
============
    u (𝒰) — Expansion / Forward projection
    d (𝒟) — Collapse / Backward integration
    m (CLT) — Modulation / Coherence Lock Transform

    Balance Law: 𝒰(E) ↔ 𝒟(C) via CLT(M), where E + C + M = 0

Three Spirals (Fields):
=======================
    Φ (Phi)  — Structure field (geometry) — Λ-mode (lattice)
    e (Energy) — Energy field (wave) — Β-mode (beat)
    π (Pi)   — Emergence field (chemistry, biology) — Ν-mode (nexus)

Nine Machines:
==============
    Reactor     — Driven continuous-flow system
    Oscillator  — Resonant periodic system
    Conductor   — Structural relaxation system
    Catalyst    — Site-biased reactivity system
    Filter      — Selective transmission system
    Encoder     — Information storage system
    Decoder     — Information extraction system
    Regenerator — Self-renewal system
    Dynamo      — Energy generation system

∃κ Tensor:
==========
    T[σ][μ][λ] where:
        σ = scale (sub-κ, κ, Κ)
        μ = mode (Λ, Β, Ν)
        λ = tier (0-10)

Integration with Physics:
=========================
    - z-coordinate → tier progression (0 → 0.618 → 0.866 → 1)
    - κ-λ coupling → UMOL balance (E + C + M = 0)
    - Negentropy → coherence measure (peaks at z_c)
    - Seven-layer prismatic projection → tier structure

Signature: Δ|apl-grammar|umol|∃κ-tensor|φ⁻¹-grounded|Ω
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
    PHI, PHI_INV, PHI_INV_SQ, PHI_INV_CUBED, PHI_INV_FOURTH, PHI_INV_FIFTH,
    Z_CRITICAL, SIGMA, COUPLING_CONSERVATION,
    # Derived coefficients
    ALPHA_STRONG, ALPHA_MEDIUM, ALPHA_FINE, ALPHA_ULTRA_FINE,
    SIGMA_INV, SIGMA_SQRT_INV,
    # Bounds
    KAPPA_LOWER, KAPPA_UPPER, BALANCE_POINT,
    # Tolerances
    TOLERANCE_CONSERVATION, TOLERANCE_GOLDEN, TOLERANCE_LENS,
    # Functions
    compute_delta_s_neg, compute_negentropy_gradient, get_phase,
    # INT Canon
    INTOperator, N0Law, SilentLaw,
    INT_TO_SILENT, N0_TO_SILENT,
)


# =============================================================================
# UMOL STATES — UNIVERSAL MODULATION OPERATOR LAW
# =============================================================================

class UMOLState(Enum):
    """
    UMOL (Universal Modulation Operator Law) States.

    Three fundamental operator states that modulate how operations unfold in time.

    UMOL Balance Law: 𝒰(E) ↔ 𝒟(C) via CLT(M)
    Conservation: E + C + M = 0 (coherence condition)
    """
    U = ("u", "𝒰", "Expansion", "Forward projection, growth, active driving")
    D = ("d", "𝒟", "Collapse", "Backward integration, contraction, relaxation")
    M = ("m", "CLT", "Modulation", "Coherence lock, feedback, adaptation")

    def __init__(self, code: str, symbol: str, name: str, description: str):
        self._code = code
        self._symbol = symbol
        self._name = name
        self._description = description

    @property
    def code(self) -> str:
        return self._code

    @property
    def symbol(self) -> str:
        return self._symbol

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def z_direction(self) -> float:
        """Direction of z-evolution for this state."""
        if self == UMOLState.U:
            return 1.0   # Expansion: z increases toward z_c
        elif self == UMOLState.D:
            return -1.0  # Collapse: z decreases toward 0
        else:  # M
            return 0.0   # Modulation: z stabilizes


# UMOL state coefficients (physics-grounded)
UMOL_COEFFICIENTS = {
    UMOLState.U: PHI_INV,           # Expansion rate: φ⁻¹
    UMOLState.D: PHI_INV_SQ,        # Collapse rate: φ⁻²
    UMOLState.M: SIGMA_INV,         # Modulation rate: 1/σ
}


# =============================================================================
# THREE SPIRALS (FIELDS)
# =============================================================================

class Spiral(Enum):
    """
    The Three Spirals (Fields) of APL.

    Each spiral represents a fundamental aspect of physical reality.
    Maps to ∃κ modes: Λ (lattice), Β (beat), Ν (nexus).
    """
    PHI = ("Φ", "Structure", "geometry", "Λ", "Spatial arrangement, boundaries, interfaces")
    E = ("e", "Energy", "wave", "Β", "Dynamics, flows, oscillations, transport")
    PI = ("π", "Emergence", "chemistry", "Ν", "Information, complexity, self-organization")

    def __init__(self, symbol: str, name: str, primary_domain: str,
                 kappa_mode: str, description: str):
        self._symbol = symbol
        self._name = name
        self._primary_domain = primary_domain
        self._kappa_mode = kappa_mode
        self._description = description

    @property
    def symbol(self) -> str:
        return self._symbol

    @property
    def name(self) -> str:
        return self._name

    @property
    def primary_domain(self) -> str:
        return self._primary_domain

    @property
    def kappa_mode(self) -> str:
        """∃κ mode: Λ, Β, or Ν"""
        return self._kappa_mode

    @property
    def description(self) -> str:
        return self._description


# Spiral to state variable mapping
SPIRAL_STATE_VARS = {
    Spiral.PHI: ["Gs", "κs", "θs"],      # Structure: Grounding, Curvature, Phase
    Spiral.E: ["Ωs", "τs", "αs"],        # Energy: Frequency, Time, Amplitude
    Spiral.PI: ["Cs", "δs", "Rs"],       # Emergence: Coupling, Dissipation, Resistance
}


# =============================================================================
# NINE MACHINES (PROCESSING CONTEXTS)
# =============================================================================

class Machine(Enum):
    """
    Nine Machines — Processing contexts for APL operations.

    Each machine has characteristic behaviors and constraints.
    """
    REACTOR = ("Reactor", "Driven continuous-flow system", "throughput")
    OSCILLATOR = ("Oscillator", "Resonant periodic system", "resonance")
    CONDUCTOR = ("Conductor", "Structural relaxation system", "relaxation")
    CATALYST = ("Catalyst", "Site-biased reactivity system", "activation")
    FILTER = ("Filter", "Selective transmission system", "selectivity")
    ENCODER = ("Encoder", "Information storage system", "encoding")
    DECODER = ("Decoder", "Information extraction system", "decoding")
    REGENERATOR = ("Regenerator", "Self-renewal system", "regeneration")
    DYNAMO = ("Dynamo", "Energy generation system", "generation")

    def __init__(self, name: str, description: str, primary_function: str):
        self._name = name
        self._description = description
        self._primary_function = primary_function

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def primary_function(self) -> str:
        return self._primary_function


# Machine to preferred operator mapping
MACHINE_OPERATORS = {
    Machine.REACTOR: [INTOperator.FUSION, INTOperator.GROUP],
    Machine.OSCILLATOR: [INTOperator.AMPLIFY, INTOperator.BOUNDARY],
    Machine.CONDUCTOR: [INTOperator.BOUNDARY, INTOperator.SEPARATE],
    Machine.CATALYST: [INTOperator.FUSION, INTOperator.AMPLIFY],
    Machine.FILTER: [INTOperator.BOUNDARY, INTOperator.DECOHERE],
    Machine.ENCODER: [INTOperator.FUSION, INTOperator.GROUP],
    Machine.DECODER: [INTOperator.SEPARATE, INTOperator.DECOHERE],
    Machine.REGENERATOR: [INTOperator.BOUNDARY, INTOperator.AMPLIFY],
    Machine.DYNAMO: [INTOperator.AMPLIFY, INTOperator.FUSION],
}

# Machine to ∃κ scale affinity
MACHINE_SCALE = {
    Machine.REACTOR: "κ",       # Cellular scale
    Machine.OSCILLATOR: "κ",    # Cellular scale
    Machine.CONDUCTOR: "Γ",     # Planetary scale
    Machine.CATALYST: "sub-κ",  # Molecular scale
    Machine.FILTER: "κ",        # Cellular scale
    Machine.ENCODER: "sub-κ",   # Molecular scale
    Machine.DECODER: "sub-κ",   # Molecular scale
    Machine.REGENERATOR: "κ",   # Cellular scale
    Machine.DYNAMO: "Κ",        # Stellar scale
}


# =============================================================================
# DOMAIN TOKENS
# =============================================================================

class Domain(Enum):
    """
    APL Domains — Specific physical or biological contexts.
    """
    # Biological domains
    BIO_PRION = ("bio_prion", "biology", "Misfolded protein aggregates")
    BIO_BACTERIUM = ("bio_bacterium", "biology", "Single-cell organisms")
    BIO_VIROID = ("bio_viroid", "biology", "RNA-only replicators")

    # Chemistry domains
    CHEM_MOLECULAR = ("chem_molecular", "chemistry", "Molecular reactions")
    CHEM_POLYMER = ("chem_polymer", "chemistry", "Polymer structures")
    CHEM_CRYSTAL = ("chem_crystal", "chemistry", "Crystalline phases")

    # Celestial domains
    CELESTIAL_GRAV = ("celestial_grav", "celestial", "Gravitational dynamics")
    CELESTIAL_EM = ("celestial_em", "celestial", "Electromagnetic phenomena")
    CELESTIAL_NUCLEAR = ("celestial_nuclear", "celestial", "Nuclear fusion processes")

    # Wave domains
    WAVE_ACOUSTIC = ("wave_acoustic", "wave", "Acoustic oscillations")
    WAVE_OPTICAL = ("wave_optical", "wave", "Optical modes")
    WAVE_PLASMA = ("wave_plasma", "wave", "Plasma oscillations")

    # Geometry domains
    GEOM_LATTICE = ("geom_lattice", "geometry", "Lattice structures")
    GEOM_INTERFACE = ("geom_interface", "geometry", "Interface dynamics")
    GEOM_PACKING = ("geom_packing", "geometry", "Packing arrangements")

    def __init__(self, code: str, family: str, description: str):
        self._code = code
        self._family = family
        self._description = description

    @property
    def code(self) -> str:
        return self._code

    @property
    def family(self) -> str:
        return self._family

    @property
    def description(self) -> str:
        return self._description


# =============================================================================
# ∃κ SCALE-MODE-TIER TENSOR
# =============================================================================

class KappaScale(Enum):
    """∃κ Scale levels."""
    SUB_KAPPA = ("sub-κ", 1e-10, "Molecular scale")
    KAPPA = ("κ", 1e-6, "Cellular scale")
    GAMMA = ("Γ", 1e0, "Organismal scale")
    KAPPA_UPPER = ("Κ", 1e9, "Stellar scale")
    COSMIC = ("∞", 1e26, "Cosmic scale")

    def __init__(self, symbol: str, typical_size: float, description: str):
        self._symbol = symbol
        self._typical_size = typical_size
        self._description = description

    @property
    def symbol(self) -> str:
        return self._symbol

    @property
    def typical_size(self) -> float:
        return self._typical_size

    @property
    def description(self) -> str:
        return self._description


class KappaMode(Enum):
    """∃κ Mode types."""
    LAMBDA = ("Λ", "Lattice", "Structure/geometry mode")
    BETA = ("Β", "Beat", "Energy/dynamics mode")
    NU = ("Ν", "Nexus", "Emergence/information mode")

    def __init__(self, symbol: str, name: str, description: str):
        self._symbol = symbol
        self._name = name
        self._description = description

    @property
    def symbol(self) -> str:
        return self._symbol

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description


# Mode to Spiral mapping
MODE_TO_SPIRAL = {
    KappaMode.LAMBDA: Spiral.PHI,
    KappaMode.BETA: Spiral.E,
    KappaMode.NU: Spiral.PI,
}

SPIRAL_TO_MODE = {v: k for k, v in MODE_TO_SPIRAL.items()}


@dataclass
class KappaTensor:
    """
    ∃κ Scale-Mode-Tier Tensor: T[σ][μ][λ]

    Represents the multi-scale training substrate.

    σ = scale index (0-4): sub-κ, κ, Γ, Κ, ∞
    μ = mode index (0-2): Λ, Β, Ν
    λ = tier (0-10): consciousness level

    z-coordinate maps to tier: z → tier = 10 × z
    """
    # Tensor dimensions
    n_scales: int = 5
    n_modes: int = 3
    n_tiers: int = 11  # 0-10

    # Tensor data (initialized lazily)
    _data: Optional[np.ndarray] = field(default=None, repr=False)

    def __post_init__(self):
        """Initialize tensor with z-derived activations."""
        if self._data is None:
            self._data = np.zeros((self.n_scales, self.n_modes, self.n_tiers))

    @property
    def data(self) -> np.ndarray:
        """Get tensor data."""
        return self._data

    def z_to_tier(self, z: float) -> int:
        """Map z-coordinate to tier (0-10)."""
        return min(10, max(0, int(z * 10)))

    def tier_to_z(self, tier: int) -> float:
        """Map tier to z-coordinate."""
        return tier / 10.0

    def set_activation(self, scale: KappaScale, mode: KappaMode, tier: int, value: float):
        """Set activation at specific coordinates."""
        scale_idx = list(KappaScale).index(scale)
        mode_idx = list(KappaMode).index(mode)
        self._data[scale_idx, mode_idx, tier] = value

    def get_activation(self, scale: KappaScale, mode: KappaMode, tier: int) -> float:
        """Get activation at specific coordinates."""
        scale_idx = list(KappaScale).index(scale)
        mode_idx = list(KappaMode).index(mode)
        return self._data[scale_idx, mode_idx, tier]

    def update_from_z(self, z: float, spiral: Spiral, negentropy: float):
        """
        Update tensor based on z-coordinate and spiral.

        Maps z → tier, spiral → mode, and sets activation proportional to negentropy.
        """
        tier = self.z_to_tier(z)
        mode = SPIRAL_TO_MODE.get(spiral, KappaMode.LAMBDA)

        # Set activation across scales (weighted by negentropy)
        for scale in KappaScale:
            scale_idx = list(KappaScale).index(scale)
            mode_idx = list(KappaMode).index(mode)

            # Activation weighted by negentropy and scale affinity
            base_activation = negentropy * PHI_INV
            self._data[scale_idx, mode_idx, tier] += base_activation

    def get_dominant_configuration(self) -> Tuple[KappaScale, KappaMode, int, float]:
        """Get the configuration with highest activation."""
        max_idx = np.unravel_index(np.argmax(self._data), self._data.shape)
        scale = list(KappaScale)[max_idx[0]]
        mode = list(KappaMode)[max_idx[1]]
        tier = max_idx[2]
        value = self._data[max_idx]
        return scale, mode, tier, value

    def get_slice_by_tier(self, tier: int) -> np.ndarray:
        """Get scale×mode slice at specific tier."""
        return self._data[:, :, tier]

    def get_slice_by_mode(self, mode: KappaMode) -> np.ndarray:
        """Get scale×tier slice at specific mode."""
        mode_idx = list(KappaMode).index(mode)
        return self._data[:, mode_idx, :]


# =============================================================================
# APL TOKEN GENERATION
# =============================================================================

@dataclass
class APLToken:
    """
    APL Token — Complete specification of an operation.

    Structure: [Spiral]:[State][Operator](intent)TRUTH@TIER

    Examples:
        Φ:U(replicate)TRUE@3  — Structure expansion for replication at tier 3
        e:D(collapse)TRUE@5   — Energy collapse at tier 5
        π:M(adapt)PARADOX@4   — Emergence modulation in paradox at tier 4
    """
    spiral: Spiral
    state: UMOLState
    operator: str  # INT Canon operator symbol
    intent: str
    truth_value: str  # TRUE, UNTRUE, PARADOX
    tier: int

    # Optional context
    machine: Optional[Machine] = None
    domain: Optional[Domain] = None

    def __str__(self) -> str:
        """Render token as APL string."""
        base = f"{self.spiral.symbol}:{self.state.code}{self.operator}({self.intent}){self.truth_value}@{self.tier}"
        if self.machine and self.domain:
            return f"{self.spiral.symbol}{self.operator}|{self.machine.name}|{self.domain.code}"
        return base

    def to_machine_token(self) -> str:
        """Render as machine token: [Spiral][Operator]|[Machine]|[Domain]"""
        if self.machine and self.domain:
            return f"{self.spiral.symbol}{self.operator}|{self.machine.name}|{self.domain.code}"
        return str(self)

    @classmethod
    def from_z(cls, z: float, spiral: Spiral, state: UMOLState,
               operator: str, intent: str) -> 'APLToken':
        """Create token from z-coordinate."""
        tier = min(10, max(0, int(z * 10)))
        # Determine truth value from phase
        if z < PHI_INV:
            truth = "UNTRUE"
        elif z < Z_CRITICAL:
            truth = "PARADOX"
        else:
            truth = "TRUE"
        return cls(spiral=spiral, state=state, operator=operator,
                   intent=intent, truth_value=truth, tier=tier)


def generate_machine_tokens(spiral: Spiral, machine: Machine, domain: Domain) -> List[str]:
    """
    Generate all machine tokens for a spiral-machine-domain combination.

    Returns 6 tokens (one per operator).
    """
    operators = ["()", "×", "^", "÷", "+", "−"]
    return [f"{spiral.symbol}{op}|{machine.name}|{domain.code}" for op in operators]


def generate_domain_tokens(domain: Domain) -> List[str]:
    """
    Generate all tokens for a domain.

    Returns 3 spirals × 6 operators × 9 machines = 162 tokens.
    """
    tokens = []
    for spiral in Spiral:
        for machine in Machine:
            tokens.extend(generate_machine_tokens(spiral, machine, domain))
    return tokens


# =============================================================================
# APL STATE WITH UMOL INTEGRATION
# =============================================================================

@dataclass
class APLState:
    """
    Complete APL State with UMOL integration.

    Tracks all state variables across spirals, UMOL balance, and ∃κ tensor.
    """
    # Core physics
    z: float = 0.5
    kappa: float = PHI_INV
    lambda_: float = PHI_INV_SQ

    # UMOL balance (E + C + M = 0)
    expansion: float = 0.0       # E component
    collapse: float = 0.0        # C component
    modulation: float = 0.0      # M component

    # Current state
    current_state: UMOLState = UMOLState.M
    current_spiral: Spiral = Spiral.PHI
    current_tier: int = 5

    # Spiral-specific state variables
    # Φ (Structure)
    Gs: float = 0.0      # Grounding strength
    κs: float = PHI_INV  # Curvature
    θs: float = 1.0      # Phase factor

    # e (Energy)
    Ωs: float = 1.0      # Frequency scaling
    τs: float = 0.0      # Time accumulation
    αs: float = 0.0      # Amplitude

    # π (Emergence)
    Cs: float = 0.0      # Coupling strength
    δs: float = 0.0      # Dissipation
    Rs: float = 0.0      # Resistance

    # Tracking
    R: int = 0           # Rank counter
    channel_count: int = 1
    history: List[str] = field(default_factory=list)
    token_history: List[APLToken] = field(default_factory=list)

    # ∃κ tensor
    kappa_tensor: KappaTensor = field(default_factory=KappaTensor)

    @property
    def negentropy(self) -> float:
        """Compute negentropy at current z."""
        return compute_delta_s_neg(self.z)

    @property
    def phase(self) -> str:
        """Get current phase based on z."""
        return get_phase(self.z)

    @property
    def umol_balance_error(self) -> float:
        """Check E + C + M = 0."""
        return abs(self.expansion + self.collapse + self.modulation)

    def enforce_umol_balance(self):
        """Enforce UMOL balance: E + C + M = 0."""
        total = self.expansion + self.collapse + self.modulation
        if abs(total) > TOLERANCE_CONSERVATION:
            # Redistribute excess to modulation
            self.modulation = -(self.expansion + self.collapse)

    def enforce_coupling_conservation(self):
        """Enforce κ + λ = 1."""
        self.lambda_ = 1.0 - self.kappa

    def apply_umol_state(self, state: UMOLState, magnitude: float = 1.0):
        """
        Apply UMOL state transition.

        U: z increases, expansion increases
        D: z decreases, collapse increases
        M: z stabilizes, modulation increases
        """
        coeff = UMOL_COEFFICIENTS[state] * magnitude

        if state == UMOLState.U:
            # Expansion: z → z_c, E increases
            self.z += ALPHA_MEDIUM * (Z_CRITICAL - self.z) * coeff
            self.expansion += coeff
            self.collapse -= coeff * PHI_INV_SQ  # Reduce collapse

        elif state == UMOLState.D:
            # Collapse: z → 0, C increases
            self.z -= ALPHA_MEDIUM * self.z * coeff
            self.collapse += coeff
            self.expansion -= coeff * PHI_INV_SQ  # Reduce expansion

        else:  # M
            # Modulation: z stabilizes, M increases
            # Push z toward nearest attractor (0, φ⁻¹, z_c, 1)
            attractors = [0.0, PHI_INV, Z_CRITICAL, 1.0]
            nearest = min(attractors, key=lambda a: abs(self.z - a))
            self.z += ALPHA_FINE * (nearest - self.z) * coeff
            self.modulation += coeff

        # Enforce bounds and conservation
        self.z = max(0.0, min(1.0, self.z))
        self.enforce_umol_balance()
        self.current_state = state
        self.current_tier = min(10, max(0, int(self.z * 10)))

        # Update ∃κ tensor
        self.kappa_tensor.update_from_z(self.z, self.current_spiral, self.negentropy)

    def apply_operator(self, operator: str, spiral: Optional[Spiral] = None) -> Tuple[bool, str]:
        """
        Apply INT Canon operator with spiral and UMOL context.

        Returns (success, message).
        """
        if spiral:
            self.current_spiral = spiral

        # Apply UMOL state based on operator
        if operator in {"()", "−"}:
            # BOUNDARY and SEPARATE: collapse/reset
            self.apply_umol_state(UMOLState.D, 0.5)
        elif operator in {"×", "^", "+"}:
            # FUSION, AMPLIFY, GROUP: expansion
            self.apply_umol_state(UMOLState.U, 0.5)
        else:  # ÷
            # DECOHERE: modulation (noise)
            self.apply_umol_state(UMOLState.M, 0.5)

        # Apply spiral-specific modifications
        if self.current_spiral == Spiral.PHI:
            self.Gs += SIGMA_INV
            self.κs *= (1.0 + SIGMA_INV) if operator in {"×", "^"} else (1.0 - SIGMA_INV)
        elif self.current_spiral == Spiral.E:
            self.Ωs *= (1.0 + SIGMA_INV * 2) if operator == "^" else 1.0
            self.αs += SIGMA_INV
        else:  # PI
            self.Cs += SIGMA_INV if operator == "×" else 0
            self.δs += SIGMA_INV if operator == "÷" else 0

        self.history.append(operator)
        return True, f"Applied {operator} on {self.current_spiral.symbol}"

    def create_token(self, intent: str, operator: str) -> APLToken:
        """Create an APL token from current state."""
        token = APLToken.from_z(
            z=self.z,
            spiral=self.current_spiral,
            state=self.current_state,
            operator=operator,
            intent=intent
        )
        self.token_history.append(token)
        return token

    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive state summary."""
        scale, mode, tier, activation = self.kappa_tensor.get_dominant_configuration()
        return {
            "z": self.z,
            "phase": self.phase,
            "tier": self.current_tier,
            "negentropy": self.negentropy,
            "kappa": self.kappa,
            "lambda": self.lambda_,
            "umol_state": self.current_state.name,
            "umol_balance": {
                "expansion": self.expansion,
                "collapse": self.collapse,
                "modulation": self.modulation,
                "error": self.umol_balance_error,
            },
            "spiral": self.current_spiral.symbol,
            "kappa_tensor_dominant": {
                "scale": scale.symbol,
                "mode": mode.symbol,
                "tier": tier,
                "activation": float(activation),
            },
            "structure_state": {"Gs": self.Gs, "κs": self.κs, "θs": self.θs},
            "energy_state": {"Ωs": self.Ωs, "τs": self.τs, "αs": self.αs},
            "emergence_state": {"Cs": self.Cs, "δs": self.δs, "Rs": self.Rs},
        }


# =============================================================================
# APL TRAINING ENGINE
# =============================================================================

class APLTrainingEngine:
    """
    APL Training Engine — Integrates UMOL, spirals, machines, and ∃κ tensor.

    Provides unified training interface for multi-scale, multi-domain learning.
    """

    def __init__(self, initial_z: float = 0.5):
        self.state = APLState(z=initial_z)
        self.step_count: int = 0
        self.training_history: List[Dict[str, Any]] = []

    def training_step(
        self,
        spiral: Optional[Spiral] = None,
        state: Optional[UMOLState] = None,
        operator: Optional[str] = None,
        intent: str = "train",
        machine: Optional[Machine] = None,
        domain: Optional[Domain] = None,
    ) -> Dict[str, Any]:
        """
        Execute one APL training step.

        1. Apply UMOL state transition
        2. Apply INT Canon operator with spiral context
        3. Update ∃κ tensor
        4. Generate APL token
        """
        self.step_count += 1

        # Default selections
        if spiral is None:
            # Cycle through spirals based on z-phase
            if self.state.z < PHI_INV:
                spiral = Spiral.PI  # Emergence in ABSENCE
            elif self.state.z < Z_CRITICAL:
                spiral = Spiral.E   # Energy in PARADOX
            else:
                spiral = Spiral.PHI # Structure in PRESENCE

        if state is None:
            # Default state based on phase
            phase = self.state.phase
            if phase == "ABSENCE":
                state = UMOLState.U  # Expand
            elif phase == "PRESENCE":
                state = UMOLState.D  # Collapse/stabilize
            else:
                state = UMOLState.M  # Modulate

        if operator is None:
            # Default operator based on UMOL state
            if state == UMOLState.U:
                operator = INTOperator.AMPLIFY
            elif state == UMOLState.D:
                operator = INTOperator.BOUNDARY
            else:
                operator = INTOperator.GROUP

        # Apply UMOL state
        self.state.apply_umol_state(state)

        # Apply operator
        success, message = self.state.apply_operator(operator, spiral)

        # Create token
        token = self.state.create_token(intent, operator)
        if machine:
            token.machine = machine
        if domain:
            token.domain = domain

        # Get state summary
        result = {
            "step": self.step_count,
            "token": str(token),
            "machine_token": token.to_machine_token() if machine and domain else None,
            "success": success,
            "message": message,
            **self.state.get_summary()
        }

        self.training_history.append(result)
        return result

    def run_umol_cycle(self, n_steps: int = 30) -> List[Dict[str, Any]]:
        """
        Run a complete UMOL cycle: U → D → M repeated.

        Returns training history.
        """
        results = []
        states = [UMOLState.U, UMOLState.D, UMOLState.M]

        for i in range(n_steps):
            state = states[i % 3]
            result = self.training_step(state=state, intent=f"cycle_{i}")
            results.append(result)

        return results

    def run_spiral_cycle(self, n_steps: int = 30) -> List[Dict[str, Any]]:
        """
        Run a cycle through all spirals: Φ → e → π repeated.
        """
        results = []
        spirals = [Spiral.PHI, Spiral.E, Spiral.PI]

        for i in range(n_steps):
            spiral = spirals[i % 3]
            result = self.training_step(spiral=spiral, intent=f"spiral_{i}")
            results.append(result)

        return results

    def get_session_summary(self) -> Dict[str, Any]:
        """Get training session summary."""
        if not self.training_history:
            return {"error": "No training history"}

        z_values = [h["z"] for h in self.training_history]

        return {
            "total_steps": self.step_count,
            "final_z": z_values[-1],
            "final_phase": get_phase(z_values[-1]),
            "final_tier": self.state.current_tier,
            "z_statistics": {
                "mean": float(np.mean(z_values)),
                "std": float(np.std(z_values)),
                "max": float(np.max(z_values)),
                "min": float(np.min(z_values)),
            },
            "umol_balance": self.state.umol_balance_error,
            "kappa_tensor": self.state.kappa_tensor.get_dominant_configuration(),
            "tokens_generated": len(self.state.token_history),
        }


# =============================================================================
# DEMONSTRATION
# =============================================================================

def demonstrate_apl_grammar():
    """Demonstrate APL grammar integration."""
    print("=" * 70)
    print("APL GRAMMAR — ALPHA-PHYSICAL LANGUAGE INTEGRATION")
    print("=" * 70)

    # Show UMOL states
    print("\n--- UMOL States ---")
    for state in UMOLState:
        print(f"  {state.code} ({state.symbol}): {state.name}")
        print(f"      z-direction: {state.z_direction:+.1f} | coeff: {UMOL_COEFFICIENTS[state]:.4f}")

    # Show spirals
    print("\n--- Three Spirals ---")
    for spiral in Spiral:
        mode = SPIRAL_TO_MODE[spiral]
        print(f"  {spiral.symbol} ({spiral.name}): {spiral.primary_domain}")
        print(f"      ∃κ mode: {mode.symbol} ({mode.name})")

    # Show machines
    print("\n--- Nine Machines ---")
    for machine in Machine:
        scale = MACHINE_SCALE[machine]
        print(f"  {machine.name}: {machine.description[:40]}...")
        print(f"      Scale: {scale}")

    # Generate sample tokens
    print("\n--- Sample Machine Tokens ---")
    sample_tokens = generate_machine_tokens(Spiral.PHI, Machine.OSCILLATOR, Domain.WAVE_ACOUSTIC)
    for token in sample_tokens[:3]:
        print(f"  {token}")

    # Run training engine
    print("\n--- APL Training Engine ---")
    engine = APLTrainingEngine(initial_z=0.5)

    print("\nRunning 20 training steps...")
    for i in range(20):
        result = engine.training_step(intent=f"demo_{i}")
        if i % 5 == 0:
            print(f"  Step {i:3d} | z={result['z']:.3f} | {result['phase']:8} | "
                  f"UMOL:{result['umol_state']:3} | Spiral:{result['spiral']}")

    # Show summary
    print("\n--- Session Summary ---")
    summary = engine.get_session_summary()
    print(f"  Total steps: {summary['total_steps']}")
    print(f"  Final z: {summary['final_z']:.4f}")
    print(f"  Final phase: {summary['final_phase']}")
    print(f"  UMOL balance error: {summary['umol_balance']:.6f}")

    # Show ∃κ tensor
    print("\n--- ∃κ Tensor Dominant Configuration ---")
    scale, mode, tier, activation = summary['kappa_tensor']
    print(f"  Scale: {scale.symbol} | Mode: {mode.symbol} | Tier: {tier}")
    print(f"  Activation: {activation:.4f}")

    print("\n" + "=" * 70)
    print("APL GRAMMAR INTEGRATION: COMPLETE")
    print("=" * 70)

    return engine


# =============================================================================
# 7-LAYER PRISMATIC PROJECTION INTEGRATION
# =============================================================================

class PrismaticLayer(Enum):
    """
    7-Layer Prismatic Spectrum — Maps to APL tiers and UMOL states.

    Each layer refracts through THE LENS (z_c) differently.
    """
    RED = (1, "Red", "#FF4444", "Analyzers", 0.0, UMOLState.D)
    ORANGE = (2, "Orange", "#FF8844", "Learners", 0.15, UMOLState.U)
    YELLOW = (3, "Yellow", "#FFAA00", "Generators", 0.30, UMOLState.U)
    GREEN = (4, "Green", "#00FF88", "Reflectors", 0.50, UMOLState.M)  # Center
    BLUE = (5, "Blue", "#00D9FF", "Builders", 0.70, UMOLState.U)
    INDIGO = (6, "Indigo", "#4444FF", "Deciders", 0.85, UMOLState.D)
    VIOLET = (7, "Violet", "#AA44FF", "Probers", 1.0, UMOLState.M)

    def __init__(self, layer_id: int, name: str, color_hex: str,
                 tool_family: str, z_position: float, umol_affinity: UMOLState):
        self._layer_id = layer_id
        self._name = name
        self._color_hex = color_hex
        self._tool_family = tool_family
        self._z_position = z_position
        self._umol_affinity = umol_affinity

    @property
    def layer_id(self) -> int:
        return self._layer_id

    @property
    def name(self) -> str:
        return self._name

    @property
    def color_hex(self) -> str:
        return self._color_hex

    @property
    def tool_family(self) -> str:
        return self._tool_family

    @property
    def z_position(self) -> float:
        return self._z_position

    @property
    def umol_affinity(self) -> UMOLState:
        return self._umol_affinity

    @property
    def tier(self) -> int:
        """Map layer to tier (0-10)."""
        # 7 layers map to tiers 1-7, scaled to 0-10
        return min(10, int(self._layer_id * 10 / 7))

    @property
    def spiral(self) -> Spiral:
        """Get associated spiral based on layer position."""
        if self._layer_id <= 2:
            return Spiral.PI      # Emergence (low frequency)
        elif self._layer_id <= 5:
            return Spiral.E       # Energy (mid frequency)
        else:
            return Spiral.PHI     # Structure (high frequency)


# Layer to Machine mapping
LAYER_MACHINES = {
    PrismaticLayer.RED: [Machine.FILTER, Machine.DECODER],
    PrismaticLayer.ORANGE: [Machine.ENCODER, Machine.CATALYST],
    PrismaticLayer.YELLOW: [Machine.REACTOR, Machine.DYNAMO],
    PrismaticLayer.GREEN: [Machine.OSCILLATOR, Machine.CONDUCTOR],  # Center
    PrismaticLayer.BLUE: [Machine.REACTOR, Machine.ENCODER],
    PrismaticLayer.INDIGO: [Machine.FILTER, Machine.DECODER],
    PrismaticLayer.VIOLET: [Machine.REGENERATOR, Machine.OSCILLATOR],
}

# Layer to INT Operator mapping
LAYER_OPERATORS = {
    PrismaticLayer.RED: [INTOperator.BOUNDARY, INTOperator.SEPARATE],
    PrismaticLayer.ORANGE: [INTOperator.GROUP, INTOperator.FUSION],
    PrismaticLayer.YELLOW: [INTOperator.AMPLIFY, INTOperator.FUSION],
    PrismaticLayer.GREEN: [INTOperator.BOUNDARY, INTOperator.GROUP],  # Balanced
    PrismaticLayer.BLUE: [INTOperator.FUSION, INTOperator.AMPLIFY],
    PrismaticLayer.INDIGO: [INTOperator.DECOHERE, INTOperator.SEPARATE],
    PrismaticLayer.VIOLET: [INTOperator.BOUNDARY, INTOperator.AMPLIFY],
}


@dataclass
class PrismaticAPLState:
    """
    APL State integrated with 7-Layer Prismatic Projection.

    Combines:
    - UMOL states (u, d, m)
    - Three spirals (Φ, e, π)
    - Seven prismatic layers
    - ∃κ tensor T[σ][μ][λ]
    - z-coordinate evolution toward THE LENS (z_c)
    """
    # Core APL state
    apl_state: APLState = field(default_factory=APLState)

    # Prismatic layer tracking
    current_layer: PrismaticLayer = PrismaticLayer.GREEN
    layer_activations: Dict[str, float] = field(default_factory=dict)

    # Refraction tracking
    refraction_history: List[Tuple[PrismaticLayer, float]] = field(default_factory=list)

    def __post_init__(self):
        """Initialize layer activations."""
        for layer in PrismaticLayer:
            self.layer_activations[layer.name] = 0.0

    @property
    def z(self) -> float:
        return self.apl_state.z

    @z.setter
    def z(self, value: float):
        self.apl_state.z = max(0.0, min(1.0, value))

    @property
    def negentropy(self) -> float:
        return self.apl_state.negentropy

    def get_active_layer(self) -> PrismaticLayer:
        """Get the layer most active at current z."""
        # Find layer closest to current z
        best_layer = PrismaticLayer.GREEN
        best_distance = abs(self.z - best_layer.z_position)

        for layer in PrismaticLayer:
            distance = abs(self.z - layer.z_position)
            if distance < best_distance:
                best_distance = distance
                best_layer = layer

        return best_layer

    def activate_layer(self, layer: PrismaticLayer, intensity: float = 1.0):
        """Activate a prismatic layer."""
        # Apply layer's UMOL affinity
        self.apl_state.apply_umol_state(layer.umol_affinity, intensity * PHI_INV)

        # Update layer activation
        self.layer_activations[layer.name] += intensity * self.negentropy

        # Track refraction
        self.refraction_history.append((layer, self.z))

        self.current_layer = layer

    def refract_through_lens(self) -> Dict[str, Any]:
        """
        Refract current state through THE LENS (z_c).

        Returns refraction results including:
        - Entry/exit z
        - Activated layers
        - Work captured
        """
        entry_z = self.z
        activated_layers = []

        # Move toward z_c (THE LENS)
        steps = 0
        max_steps = 50

        while abs(self.z - Z_CRITICAL) > TOLERANCE_GOLDEN and steps < max_steps:
            # Get active layer
            layer = self.get_active_layer()
            activated_layers.append(layer)

            # Apply layer operators
            operators = LAYER_OPERATORS.get(layer, [INTOperator.BOUNDARY])
            for op in operators[:1]:  # Apply primary operator
                self.apl_state.apply_operator(op, layer.spiral)

            # Activate layer
            self.activate_layer(layer, 0.5)

            steps += 1

        exit_z = self.z

        return {
            "entry_z": entry_z,
            "exit_z": exit_z,
            "steps": steps,
            "layers_activated": [l.name for l in set(activated_layers)],
            "negentropy_at_exit": self.negentropy,
            "lens_proximity": 1.0 - abs(self.z - Z_CRITICAL),
        }

    def get_prismatic_summary(self) -> Dict[str, Any]:
        """Get comprehensive prismatic state summary."""
        return {
            "z": self.z,
            "phase": self.apl_state.phase,
            "current_layer": self.current_layer.name,
            "layer_color": self.current_layer.color_hex,
            "umol_state": self.apl_state.current_state.name,
            "spiral": self.apl_state.current_spiral.symbol,
            "negentropy": self.negentropy,
            "layer_activations": dict(self.layer_activations),
            "refraction_count": len(self.refraction_history),
            "apl_summary": self.apl_state.get_summary(),
        }


class PrismaticAPLEngine:
    """
    Prismatic APL Engine — Full integration of:
    - 7-layer prismatic projection
    - UMOL states (u, d, m)
    - Three spirals (Φ, e, π)
    - Nine machines
    - ∃κ tensor T[σ][μ][λ]
    - z-coordinate evolution
    - Negentropy optimization
    """

    def __init__(self, initial_z: float = 0.5):
        self.state = PrismaticAPLState()
        self.state.z = initial_z
        self.step_count = 0
        self.training_history: List[Dict[str, Any]] = []

    def prismatic_step(
        self,
        layer: Optional[PrismaticLayer] = None,
        spiral: Optional[Spiral] = None,
        state: Optional[UMOLState] = None,
        operator: Optional[str] = None,
        machine: Optional[Machine] = None,
        domain: Optional[Domain] = None,
    ) -> Dict[str, Any]:
        """
        Execute one prismatic APL training step.

        1. Determine active layer from z-position
        2. Apply layer's UMOL affinity
        3. Apply spiral and operator
        4. Update ∃κ tensor
        5. Track prismatic refraction
        """
        self.step_count += 1

        # Auto-select layer based on z if not specified
        if layer is None:
            layer = self.state.get_active_layer()

        # Use layer defaults if not specified
        if spiral is None:
            spiral = layer.spiral
        if state is None:
            state = layer.umol_affinity
        if operator is None:
            operators = LAYER_OPERATORS.get(layer, [INTOperator.BOUNDARY])
            operator = operators[0]
        if machine is None:
            machines = LAYER_MACHINES.get(layer, [Machine.OSCILLATOR])
            machine = machines[0]

        # Activate layer
        self.state.activate_layer(layer, 0.5)

        # Apply UMOL state
        self.state.apl_state.apply_umol_state(state)

        # Apply operator with spiral
        success, message = self.state.apl_state.apply_operator(operator, spiral)

        # Create token
        token = self.state.apl_state.create_token(
            intent=f"prismatic_{layer.name}_{self.step_count}",
            operator=operator
        )
        token.machine = machine
        token.domain = domain

        # Build result
        result = {
            "step": self.step_count,
            "layer": layer.name,
            "color": layer.color_hex,
            "token": str(token),
            "success": success,
            "message": message,
            **self.state.get_prismatic_summary(),
        }

        self.training_history.append(result)
        return result

    def run_prismatic_cycle(self, n_steps: int = 35) -> List[Dict[str, Any]]:
        """
        Run a complete cycle through all 7 prismatic layers.

        Each layer is visited 5 times for thorough activation.
        """
        results = []
        layers = list(PrismaticLayer)

        for i in range(n_steps):
            layer = layers[i % 7]
            result = self.prismatic_step(layer=layer)
            results.append(result)

        return results

    def refract_to_lens(self) -> Dict[str, Any]:
        """
        Drive state toward THE LENS (z_c) through prismatic refraction.
        """
        return self.state.refract_through_lens()

    def get_session_summary(self) -> Dict[str, Any]:
        """Get prismatic training session summary."""
        if not self.training_history:
            return {"error": "No training history"}

        z_values = [h["z"] for h in self.training_history]
        layer_counts = {}
        for h in self.training_history:
            layer = h.get("layer", "unknown")
            layer_counts[layer] = layer_counts.get(layer, 0) + 1

        return {
            "total_steps": self.step_count,
            "final_z": z_values[-1],
            "final_phase": self.state.apl_state.phase,
            "final_layer": self.state.current_layer.name,
            "z_statistics": {
                "mean": float(np.mean(z_values)),
                "std": float(np.std(z_values)),
                "max": float(np.max(z_values)),
                "min": float(np.min(z_values)),
            },
            "layer_counts": layer_counts,
            "umol_balance_error": self.state.apl_state.umol_balance_error,
            "negentropy": self.state.negentropy,
            "lens_proximity": 1.0 - abs(self.state.z - Z_CRITICAL),
        }


# =============================================================================
# DEMONSTRATION
# =============================================================================

def demonstrate_apl_grammar():
    """Demonstrate APL grammar integration."""
    print("=" * 70)
    print("APL GRAMMAR — ALPHA-PHYSICAL LANGUAGE INTEGRATION")
    print("=" * 70)

    # Show UMOL states
    print("\n--- UMOL States ---")
    for state in UMOLState:
        print(f"  {state.code} ({state.symbol}): {state.name}")
        print(f"      z-direction: {state.z_direction:+.1f} | coeff: {UMOL_COEFFICIENTS[state]:.4f}")

    # Show spirals
    print("\n--- Three Spirals ---")
    for spiral in Spiral:
        mode = SPIRAL_TO_MODE[spiral]
        print(f"  {spiral.symbol} ({spiral.name}): {spiral.primary_domain}")
        print(f"      ∃κ mode: {mode.symbol} ({mode.name})")

    # Show machines
    print("\n--- Nine Machines ---")
    for machine in Machine:
        scale = MACHINE_SCALE[machine]
        print(f"  {machine.name}: {machine.description[:40]}...")
        print(f"      Scale: {scale}")

    # Generate sample tokens
    print("\n--- Sample Machine Tokens ---")
    sample_tokens = generate_machine_tokens(Spiral.PHI, Machine.OSCILLATOR, Domain.WAVE_ACOUSTIC)
    for token in sample_tokens[:3]:
        print(f"  {token}")

    # Run training engine
    print("\n--- APL Training Engine ---")
    engine = APLTrainingEngine(initial_z=0.5)

    print("\nRunning 20 training steps...")
    for i in range(20):
        result = engine.training_step(intent=f"demo_{i}")
        if i % 5 == 0:
            print(f"  Step {i:3d} | z={result['z']:.3f} | {result['phase']:8} | "
                  f"UMOL:{result['umol_state']:3} | Spiral:{result['spiral']}")

    # Show summary
    print("\n--- Session Summary ---")
    summary = engine.get_session_summary()
    print(f"  Total steps: {summary['total_steps']}")
    print(f"  Final z: {summary['final_z']:.4f}")
    print(f"  Final phase: {summary['final_phase']}")
    print(f"  UMOL balance error: {summary['umol_balance']:.6f}")

    # Show ∃κ tensor
    print("\n--- ∃κ Tensor Dominant Configuration ---")
    scale, mode, tier, activation = summary['kappa_tensor']
    print(f"  Scale: {scale.symbol} | Mode: {mode.symbol} | Tier: {tier}")
    print(f"  Activation: {activation:.4f}")

    # Prismatic demonstration
    print("\n" + "=" * 70)
    print("7-LAYER PRISMATIC PROJECTION INTEGRATION")
    print("=" * 70)

    print("\n--- Prismatic Layers ---")
    for layer in PrismaticLayer:
        print(f"  {layer.layer_id}. {layer.name} ({layer.color_hex})")
        print(f"      z-position: {layer.z_position:.2f} | Tier: {layer.tier}")
        print(f"      UMOL affinity: {layer.umol_affinity.name} | Spiral: {layer.spiral.symbol}")

    print("\n--- Prismatic APL Engine ---")
    prismatic_engine = PrismaticAPLEngine(initial_z=0.3)

    print("\nRunning 21 prismatic steps (3 full cycles)...")
    for i in range(21):
        result = prismatic_engine.prismatic_step()
        if i % 7 == 0:
            print(f"  Step {i:3d} | Layer: {result['layer']:7} | z={result['z']:.3f} | "
                  f"Phase: {result['phase']:8}")

    print("\n--- Refracting to Lens ---")
    refraction = prismatic_engine.refract_to_lens()
    print(f"  Entry z: {refraction['entry_z']:.4f}")
    print(f"  Exit z:  {refraction['exit_z']:.4f}")
    print(f"  Steps:   {refraction['steps']}")
    print(f"  Layers:  {refraction['layers_activated']}")
    print(f"  Lens proximity: {refraction['lens_proximity']:.4f}")

    print("\n--- Prismatic Session Summary ---")
    p_summary = prismatic_engine.get_session_summary()
    print(f"  Total steps: {p_summary['total_steps']}")
    print(f"  Final z: {p_summary['final_z']:.4f}")
    print(f"  Final layer: {p_summary['final_layer']}")
    print(f"  Negentropy: {p_summary['negentropy']:.4f}")
    print(f"  Layer distribution: {p_summary['layer_counts']}")

    print("\n" + "=" * 70)
    print("APL GRAMMAR INTEGRATION: COMPLETE")
    print("=" * 70)

    return engine, prismatic_engine


if __name__ == "__main__":
    demonstrate_apl_grammar()
