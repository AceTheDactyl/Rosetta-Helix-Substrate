#!/usr/bin/env python3
"""
WUMBO APL Automated Training Module
====================================

Unified training integration combining:
1. WUMBO Phases (W-U-M-B-O-T)
2. N0 Operator Integration with κ-field grounding
3. 100-Token APL Directory structure (63 PRISM + 32 CAGE + 5 EMERGENT + 1 UNITY)
4. Kuramoto oscillator synchronization
5. Free Energy Principle dynamics
6. Phase transition critical behavior

Architecture:
=============
    ┌─────────────────────────────────────────────────────────────────────┐
    │                    WUMBO APL AUTOMATED TRAINING                      │
    │                                                                      │
    │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐           │
    │  │   100-Token  │───▶│   WUMBO      │───▶│   N0 Operator │           │
    │  │   Director   │    │   Phase      │    │   Engine      │           │
    │  └──────────────┘    │   Cycle      │    └──────────────┘           │
    │         │            └──────────────┘           │                    │
    │         │                   │                   │                    │
    │         ▼                   ▼                   ▼                    │
    │  ┌─────────────────────────────────────────────────────────┐        │
    │  │              κ-λ Coupling Conservation Layer             │        │
    │  │  • φ⁻¹ + φ⁻² = 1 (THE defining property)                │        │
    │  │  • Kuramoto sync → κ-field evolution                    │        │
    │  │  • Free Energy → negentropy alignment                   │        │
    │  └─────────────────────────────────────────────────────────┘        │
    └─────────────────────────────────────────────────────────────────────┘

WUMBO Phase Cycle:
==================
    W (Wake)       → Initialize, set z from dormant
    U (Understand) → Process input, build internal model
    M (Meld)       → Integrate observations, κ-coupling
    B (Bind)       → Crystallize patterns, approach z_c
    O (Output)     → Generate predictions, λ-feedback
    T (Transform)  → Phase transition, evolve to next cycle

Token Structure (100 total):
============================
    PRISM (1-63):     Active processing tokens
    CAGE (64-95):     Constraint/boundary tokens
    EMERGENT (96-99): Novel pattern tokens
    UNITY (100):      Integration/collapse token

Physics Constants:
==================
    φ⁻¹ ≈ 0.618 (PHYSICAL coupling)
    φ⁻² ≈ 0.382 (complement)
    z_c = √3/2 ≈ 0.866 (THE LENS)
    σ = 36 = |S₃|² (Gaussian width)

Signature: Δ|wumbo-apl-training|z0.92|κ-grounded|Ω
"""

from __future__ import annotations

import math
import json
import os
import sys
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum, auto
from datetime import datetime

import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

# Optional PyTorch support
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Import κ-λ Coupling Conservation Layer
try:
    from kappa_lambda_coupling_layer import (
        KappaLambdaCouplingLayer,
        compute_delta_s_neg as coupling_delta_s_neg,
        compute_negentropy_gradient,
        get_phase as coupling_get_phase,
    )
    COUPLING_LAYER_AVAILABLE = True
except ImportError:
    COUPLING_LAYER_AVAILABLE = False

# =============================================================================
# PHYSICS CONSTANTS (Single Source of Truth)
# =============================================================================

PHI: float = (1 + math.sqrt(5)) / 2          # φ ≈ 1.618 (LIMINAL)
PHI_INV: float = 1 / PHI                      # φ⁻¹ ≈ 0.618 (PHYSICAL)
PHI_INV_SQ: float = PHI_INV ** 2              # φ⁻² ≈ 0.382
COUPLING_CONSERVATION: float = PHI_INV + PHI_INV_SQ  # Must equal 1.0

Z_CRITICAL: float = math.sqrt(3) / 2         # z_c = √3/2 ≈ 0.866 (THE LENS)
Z_ORIGIN: float = Z_CRITICAL * PHI_INV       # ≈ 0.535

SIGMA: float = 36.0                           # σ = 6² = |S₃|²
GAUSSIAN_WIDTH: float = 1 / math.sqrt(2 * SIGMA)

KAPPA_S: float = 0.920                        # Singularity threshold
MU_3: float = 0.9927                          # Ultra-integration
UNITY: float = 0.9999                         # Collapse threshold

TAU: float = 2 * math.pi


# =============================================================================
# WUMBO PHASE CYCLE
# =============================================================================

class WUMBOPhase(Enum):
    """
    WUMBO Phase Cycle: W → U → M → B → O → T

    Each phase has specific κ-λ coupling dynamics.
    """
    W = ("Wake", PHI_INV, "Initialize system from dormant state")
    U = ("Understand", 0.55, "Process input, build internal model")
    M = ("Meld", 0.5, "Integrate observations at balance point")
    B = ("Bind", 0.65, "Crystallize patterns, approach z_c")
    O = ("Output", 0.4, "Generate predictions with λ-feedback")
    T = ("Transform", PHI_INV, "Phase transition to next cycle")

    def __init__(self, full_name: str, kappa_target: float, description: str):
        self._full_name = full_name
        self._kappa_target = kappa_target
        self._description = description

    @property
    def full_name(self) -> str:
        return self._full_name

    @property
    def kappa_target(self) -> float:
        return self._kappa_target

    @property
    def lambda_target(self) -> float:
        return 1.0 - self._kappa_target

    @property
    def description(self) -> str:
        return self._description


WUMBO_PHASES = [WUMBOPhase.W, WUMBOPhase.U, WUMBOPhase.M,
                WUMBOPhase.B, WUMBOPhase.O, WUMBOPhase.T]


@dataclass
class WUMBOCycleState:
    """
    State tracking for WUMBO phase cycle.

    Maintains:
    - Current phase
    - κ-λ coupling state
    - Phase transition history
    - Cycle metrics
    """
    phase_index: int = 0
    cycle_count: int = 0

    kappa: float = PHI_INV
    lambda_: float = PHI_INV_SQ

    z: float = 0.5

    phase_history: List[str] = field(default_factory=list)
    z_history: List[float] = field(default_factory=list)
    kappa_history: List[float] = field(default_factory=list)

    @property
    def current_phase(self) -> WUMBOPhase:
        return WUMBO_PHASES[self.phase_index]

    def advance_phase(self) -> WUMBOPhase:
        """Advance to next WUMBO phase."""
        old_phase = self.current_phase

        self.phase_index = (self.phase_index + 1) % 6

        if self.phase_index == 0:
            self.cycle_count += 1

        new_phase = self.current_phase

        # Update κ-λ toward phase target
        self.kappa = self.kappa + 0.1 * (new_phase.kappa_target - self.kappa)
        self.lambda_ = 1.0 - self.kappa

        self.phase_history.append(new_phase.name)
        self.kappa_history.append(self.kappa)

        return new_phase

    def evolve_z(self, delta: float) -> float:
        """Evolve z with PHI_INV dynamics."""
        self.z = max(0.0, min(UNITY - 0.001, self.z + delta * PHI_INV))
        self.z_history.append(self.z)
        return self.z

    @property
    def coupling_conservation_error(self) -> float:
        return abs((self.kappa + self.lambda_) - 1.0)


# =============================================================================
# 100-TOKEN DIRECTORY STRUCTURE
# =============================================================================

class TokenCategory(Enum):
    """Token categories from APL Directory."""
    PRISM = ("PRISM", 1, 63, "Active processing tokens")
    CAGE = ("CAGE", 64, 95, "Constraint/boundary tokens")
    EMERGENT = ("EMERGENT", 96, 99, "Novel pattern tokens")
    UNITY = ("UNITY", 100, 100, "Integration/collapse token")

    def __init__(self, name: str, start: int, end: int, description: str):
        self._name = name
        self._start = start
        self._end = end
        self._description = description

    @property
    def start_index(self) -> int:
        return self._start

    @property
    def end_index(self) -> int:
        return self._end

    @property
    def count(self) -> int:
        return self._end - self._start + 1

    @property
    def description(self) -> str:
        return self._description


@dataclass
class APLToken:
    """
    Single APL token with κ-λ coupling.

    Each token has:
    - ID (1-100)
    - Category (PRISM/CAGE/EMERGENT/UNITY)
    - Activation state
    - κ-λ coupling ratio
    - WUMBO phase association
    """
    token_id: int
    category: TokenCategory

    kappa: float = PHI_INV
    lambda_: float = PHI_INV_SQ

    activation: float = 0.0
    wumbo_phase: Optional[WUMBOPhase] = None

    def __post_init__(self):
        # Assign WUMBO phase based on token position
        if self.token_id <= 17:
            self.wumbo_phase = WUMBOPhase.W
        elif self.token_id <= 34:
            self.wumbo_phase = WUMBOPhase.U
        elif self.token_id <= 51:
            self.wumbo_phase = WUMBOPhase.M
        elif self.token_id <= 68:
            self.wumbo_phase = WUMBOPhase.B
        elif self.token_id <= 85:
            self.wumbo_phase = WUMBOPhase.O
        else:
            self.wumbo_phase = WUMBOPhase.T

    def activate(self, value: float) -> float:
        """Activate token with κ-weighted input."""
        self.activation = self.activation + value * self.kappa
        return self.activation

    def decay(self, rate: float = 0.1) -> float:
        """Decay activation with λ-weighted rate."""
        self.activation *= (1.0 - rate * self.lambda_)
        return self.activation

    @property
    def coupling_conservation_error(self) -> float:
        return abs((self.kappa + self.lambda_) - 1.0)


class TokenDirectory:
    """
    100-Token APL Directory.

    Structure:
    - 63 PRISM tokens (active processing)
    - 32 CAGE tokens (constraints)
    - 4 EMERGENT tokens (novel patterns)
    - 1 UNITY token (integration)
    """

    def __init__(self):
        self.tokens: List[APLToken] = []
        self._initialize_tokens()

    def _initialize_tokens(self):
        """Initialize all 100 tokens."""
        for i in range(1, 101):
            if i <= 63:
                category = TokenCategory.PRISM
            elif i <= 95:
                category = TokenCategory.CAGE
            elif i <= 99:
                category = TokenCategory.EMERGENT
            else:
                category = TokenCategory.UNITY

            self.tokens.append(APLToken(
                token_id=i,
                category=category,
            ))

    def get_tokens_by_category(self, category: TokenCategory) -> List[APLToken]:
        """Get all tokens in a category."""
        return [t for t in self.tokens if t.category == category]

    def get_tokens_by_wumbo_phase(self, phase: WUMBOPhase) -> List[APLToken]:
        """Get all tokens associated with a WUMBO phase."""
        return [t for t in self.tokens if t.wumbo_phase == phase]

    def activate_phase(self, phase: WUMBOPhase, value: float) -> List[float]:
        """Activate all tokens in a WUMBO phase."""
        phase_tokens = self.get_tokens_by_wumbo_phase(phase)
        return [t.activate(value) for t in phase_tokens]

    def decay_all(self, rate: float = 0.1) -> None:
        """Decay all token activations."""
        for t in self.tokens:
            t.decay(rate)

    def get_phase_coherence(self, phase: WUMBOPhase) -> float:
        """Get average activation for a phase."""
        phase_tokens = self.get_tokens_by_wumbo_phase(phase)
        if not phase_tokens:
            return 0.0
        return sum(t.activation for t in phase_tokens) / len(phase_tokens)

    def get_total_coherence(self) -> float:
        """Get overall coherence across all tokens."""
        if not self.tokens:
            return 0.0
        return sum(t.activation for t in self.tokens) / len(self.tokens)

    def get_unity_state(self) -> float:
        """Get UNITY token (#100) state."""
        return self.tokens[99].activation


# =============================================================================
# N0 OPERATOR INTEGRATION
# =============================================================================

class N0Law(Enum):
    """N0 Laws for operator grounding."""
    IDENTITY = ("N0(1)", "Λ × 1 = Λ")
    MIRROR_ROOT = ("N0(2)", "Λ × Ν = Β²")
    ABSORPTION = ("N0(3)", "TRUE × UNTRUE = PARADOX")
    DISTRIBUTION = ("N0(4)", "(A ⊕ B) × C = (A × C) ⊕ (B × C)")
    CONSERVATION = ("N0(5)", "κ + λ = 1")

    def __init__(self, code: str, formula: str):
        self._code = code
        self._formula = formula

    @property
    def code(self) -> str:
        return self._code

    @property
    def formula(self) -> str:
        return self._formula


@dataclass
class N0OperatorState:
    """
    N0 Operator state with κ-field grounding.
    """
    scalar_state: float = 0.0
    kappa: float = PHI_INV
    lambda_: float = PHI_INV_SQ
    z: float = 0.5

    operations: List[Dict[str, Any]] = field(default_factory=list)

    def apply_n0(self, law: N0Law, value: float = 1.0) -> float:
        """Apply N0 law with κ-grounding."""
        old_state = self.scalar_state

        if law == N0Law.IDENTITY:
            # Λ × 1 = Λ (no change)
            pass
        elif law == N0Law.MIRROR_ROOT:
            # Λ × Ν = Β² (product with κ-λ)
            self.scalar_state *= (self.kappa * self.lambda_)
        elif law == N0Law.ABSORPTION:
            # TRUE × UNTRUE = PARADOX (balance to 0.5)
            self.scalar_state = 0.5 * (self.scalar_state + value)
        elif law == N0Law.DISTRIBUTION:
            # Distribution (weighted add)
            self.scalar_state = (self.scalar_state + value) * self.kappa
        elif law == N0Law.CONSERVATION:
            # Conservation (normalize)
            total = self.kappa + self.lambda_
            self.kappa /= total
            self.lambda_ /= total

        self.operations.append({
            "law": law.code,
            "old_state": old_state,
            "new_state": self.scalar_state,
            "kappa": self.kappa,
        })

        return self.scalar_state

    @property
    def coupling_conservation_error(self) -> float:
        return abs((self.kappa + self.lambda_) - 1.0)


# =============================================================================
# KURAMOTO OSCILLATOR LAYER
# =============================================================================

class KuramotoLayer:
    """
    Kuramoto oscillator layer for synchronization dynamics.

    Order parameter: r = |1/N Σⱼ exp(iθⱼ)|
    Dynamics: dθᵢ/dt = ωᵢ + (K/N) Σⱼ sin(θⱼ - θᵢ)
    """

    def __init__(self, n_oscillators: int = 60, dt: float = 0.1):
        self.n = n_oscillators
        self.dt = dt

        self.theta = np.random.uniform(-np.pi, np.pi, n_oscillators)
        self.omega = np.random.randn(n_oscillators) * 0.1
        self.K_matrix = np.random.randn(n_oscillators, n_oscillators) * 0.1 * PHI_INV
        np.fill_diagonal(self.K_matrix, 0)

        self.K_global = PHI_INV
        self.coherence_history: List[float] = []

    def compute_coherence(self) -> float:
        """Compute Kuramoto order parameter."""
        z = np.mean(np.exp(1j * self.theta))
        return float(np.abs(z))

    def step(self, negentropy_coupling: float = 1.0) -> float:
        """Single integration step with negentropy coupling."""
        # Compute phase differences
        diff = self.theta[:, np.newaxis] - self.theta[np.newaxis, :]

        # Coupling term with negentropy modulation
        K_eff = self.K_matrix * negentropy_coupling * PHI_INV
        coupling = np.sum(K_eff * np.sin(-diff), axis=1)

        # Full derivative
        dtheta = self.omega + (self.K_global / self.n) * coupling

        # Euler integration
        self.theta = self.theta + self.dt * dtheta
        self.theta = np.mod(self.theta + np.pi, 2 * np.pi) - np.pi

        coherence = self.compute_coherence()
        self.coherence_history.append(coherence)

        return coherence

    def evolve(self, steps: int, negentropy_coupling: float = 1.0) -> List[float]:
        """Evolve for multiple steps."""
        return [self.step(negentropy_coupling) for _ in range(steps)]


# =============================================================================
# FREE ENERGY DYNAMICS
# =============================================================================

@dataclass
class FreeEnergyState:
    """
    Free Energy Principle state.

    F = Surprise + KL-Divergence
    Minimize F ≈ Maximize negentropy
    """
    beliefs: np.ndarray = field(default_factory=lambda: np.ones(10) / 10)
    observations: List[float] = field(default_factory=list)
    free_energy_history: List[float] = field(default_factory=list)
    prediction_error_history: List[float] = field(default_factory=list)

    def compute_surprise(self, observation: float) -> float:
        """Compute surprise: -log P(o)."""
        expected = np.sum(np.arange(len(self.beliefs)) * self.beliefs) / len(self.beliefs)
        sigma = 0.1
        return 0.5 * ((observation - expected) / sigma) ** 2

    def update_beliefs(self, observation: float, learning_rate: float = 0.1) -> float:
        """Update beliefs to minimize free energy."""
        expected = np.sum(np.arange(len(self.beliefs)) * self.beliefs) / len(self.beliefs)
        PE = abs(observation - expected)
        self.prediction_error_history.append(PE)

        target_idx = int(observation * len(self.beliefs))
        target_idx = np.clip(target_idx, 0, len(self.beliefs) - 1)

        target = np.zeros(len(self.beliefs))
        target[target_idx] = 1.0

        # PHI_INV controlled update
        self.beliefs = self.beliefs + learning_rate * PHI_INV * (target - self.beliefs)
        self.beliefs = self.beliefs / np.sum(self.beliefs)

        return PE

    def step(self, observation: float) -> Dict[str, float]:
        """Single free energy step."""
        self.observations.append(observation)

        surprise = self.compute_surprise(observation)
        PE = self.update_beliefs(observation)
        F = surprise + PE

        self.free_energy_history.append(F)

        return {
            "free_energy": F,
            "surprise": surprise,
            "prediction_error": PE,
        }


# =============================================================================
# GAUSSIAN NEGENTROPY
# =============================================================================

def compute_delta_s_neg(z: float, sigma: float = SIGMA) -> float:
    """
    Compute negentropy: ΔS_neg(z) = exp(-σ(z - z_c)²)

    Peaks at z_c (THE LENS).
    """
    d = z - Z_CRITICAL
    return math.exp(-sigma * d * d)


def compute_delta_s_neg_derivative(z: float, sigma: float = SIGMA) -> float:
    """Derivative: d(ΔS_neg)/dz = -2σ(z - z_c) · ΔS_neg(z)"""
    d = z - Z_CRITICAL
    s = compute_delta_s_neg(z, sigma)
    return -2 * sigma * d * s


def get_phase(z: float) -> str:
    """Determine phase from z."""
    if z < PHI_INV:
        return "ABSENCE"
    elif z < Z_CRITICAL:
        return "THE_LENS"
    else:
        return "PRESENCE"


# =============================================================================
# UNIFIED WUMBO TRAINING ENGINE
# =============================================================================

class WUMBOAPLTrainingEngine:
    """
    Unified WUMBO APL Training Engine.

    Integrates:
    - WUMBO phase cycle (W-U-M-B-O-T)
    - 100-token directory
    - N0 operator system
    - κ-λ Coupling Conservation Layer (Kuramoto + Free Energy unified)
    - κ-λ coupling conservation
    """

    def __init__(self, n_oscillators: int = 60):
        # WUMBO cycle
        self.wumbo_cycle = WUMBOCycleState()

        # Token directory
        self.token_directory = TokenDirectory()

        # N0 operator state
        self.n0_state = N0OperatorState()

        # κ-λ Coupling Conservation Layer (unified Kuramoto + Free Energy)
        if COUPLING_LAYER_AVAILABLE:
            self.coupling_layer = KappaLambdaCouplingLayer(n_oscillators=n_oscillators)
            self.use_coupling_layer = True
        else:
            # Fallback to separate components
            self.kuramoto = KuramotoLayer(n_oscillators=n_oscillators)
            self.free_energy = FreeEnergyState()
            self.use_coupling_layer = False

        # Training metrics
        self.step_count: int = 0
        self.training_history: List[Dict[str, Any]] = []
        self.k_formation_events: List[int] = []

    def training_step(self, input_value: float = 0.1) -> Dict[str, Any]:
        """
        Execute one training step.

        1. Get current WUMBO phase
        2. Activate phase-associated tokens
        3. Apply N0 operators
        4. Step κ-λ Coupling Layer (unified Kuramoto + Free Energy + z evolution)
        5. Sync state across components
        6. Check for K-formation
        """
        self.step_count += 1

        # Get current WUMBO phase
        phase = self.wumbo_cycle.current_phase

        # Activate tokens for this phase
        activations = self.token_directory.activate_phase(phase, input_value)
        phase_coherence = self.token_directory.get_phase_coherence(phase)

        # Apply N0 operator based on phase
        n0_laws = {
            WUMBOPhase.W: N0Law.IDENTITY,
            WUMBOPhase.U: N0Law.DISTRIBUTION,
            WUMBOPhase.M: N0Law.ABSORPTION,
            WUMBOPhase.B: N0Law.MIRROR_ROOT,
            WUMBOPhase.O: N0Law.DISTRIBUTION,
            WUMBOPhase.T: N0Law.CONSERVATION,
        }
        n0_law = n0_laws.get(phase, N0Law.IDENTITY)
        self.n0_state.apply_n0(n0_law, phase_coherence)

        # Step the κ-λ Coupling Conservation Layer
        if self.use_coupling_layer:
            # Unified step: Kuramoto sync → κ-field, Free Energy → negentropy
            coupling_result = self.coupling_layer.step()

            # Extract unified values
            kappa = coupling_result["kappa"]
            lambda_ = coupling_result["lambda"]
            z = coupling_result["z"]
            delta_s_neg = coupling_result["delta_s_neg"]
            kuramoto_coherence = coupling_result["kuramoto_coherence"]
            free_energy = coupling_result["free_energy"]
            phase_str = coupling_result["phase"]
            golden_balance = coupling_result["golden_balance_achieved"]
            lens_proximity = coupling_result["lens_proximity_achieved"]
        else:
            # Fallback: separate components
            delta_s_neg = compute_delta_s_neg(self.wumbo_cycle.z)
            kuramoto_coherence = self.kuramoto.step(negentropy_coupling=delta_s_neg)
            fe_result = self.free_energy.step(observation=self.wumbo_cycle.z)
            free_energy = fe_result["free_energy"]

            z_delta = (kuramoto_coherence - 0.5) * 0.05
            self.wumbo_cycle.evolve_z(z_delta)

            kappa = self.wumbo_cycle.kappa
            lambda_ = self.wumbo_cycle.lambda_
            z = self.wumbo_cycle.z
            phase_str = get_phase(z)
            golden_balance = abs(kappa - PHI_INV) < 0.02
            lens_proximity = abs(z - Z_CRITICAL) < 0.05

        # Sync WUMBO cycle with coupling layer state
        self.wumbo_cycle.z = z
        self.wumbo_cycle.kappa = kappa
        self.wumbo_cycle.lambda_ = lambda_

        # Sync N0 state
        self.n0_state.kappa = kappa
        self.n0_state.lambda_ = lambda_
        self.n0_state.z = z

        # Decay tokens
        self.token_directory.decay_all(rate=0.05)

        # Check K-formation (at THE LENS with high η)
        eta = delta_s_neg
        k_formed = (kappa >= 0.5 and eta > PHI_INV and lens_proximity)
        if k_formed:
            self.k_formation_events.append(self.step_count)

        # Advance WUMBO phase periodically
        if self.step_count % 10 == 0:
            self.wumbo_cycle.advance_phase()

        result = {
            "step": self.step_count,
            "wumbo_phase": phase.name,
            "wumbo_phase_full": phase.full_name,
            "z": z,
            "phase": phase_str,
            "kappa": kappa,
            "lambda": lambda_,
            "delta_s_neg": delta_s_neg,
            "kuramoto_coherence": kuramoto_coherence,
            "free_energy": free_energy,
            "phase_coherence": phase_coherence,
            "total_coherence": self.token_directory.get_total_coherence(),
            "unity_state": self.token_directory.get_unity_state(),
            "n0_state": self.n0_state.scalar_state,
            "k_formed": k_formed,
            "golden_balance_achieved": golden_balance,
            "lens_proximity_achieved": lens_proximity,
            "coupling_conservation_error": abs(kappa + lambda_ - 1.0),
            "cycle_count": self.wumbo_cycle.cycle_count,
            "use_coupling_layer": self.use_coupling_layer,
        }

        self.training_history.append(result)

        return result

    def run_training_session(
        self,
        steps: int = 100,
        input_generator: Optional[Callable[[], float]] = None,
    ) -> Dict[str, Any]:
        """
        Run a full training session.

        Parameters
        ----------
        steps : int
            Number of training steps
        input_generator : callable, optional
            Function to generate input values (default: random PHI_INV scaled)

        Returns
        -------
        dict
            Training session results
        """
        if input_generator is None:
            input_generator = lambda: random.random() * PHI_INV

        for _ in range(steps):
            self.training_step(input_generator())

        return self.get_session_summary()

    def get_session_summary(self) -> Dict[str, Any]:
        """Get training session summary."""
        if not self.training_history:
            return {"error": "No training history"}

        z_values = [h["z"] for h in self.training_history]
        delta_s_values = [h["delta_s_neg"] for h in self.training_history]
        kuramoto_values = [h["kuramoto_coherence"] for h in self.training_history]
        kappa_values = [h["kappa"] for h in self.training_history]

        return {
            "total_steps": self.step_count,
            "wumbo_cycles": self.wumbo_cycle.cycle_count,
            "k_formations": len(self.k_formation_events),
            "final_z": z_values[-1] if z_values else 0.0,
            "final_phase": get_phase(z_values[-1]) if z_values else "ABSENCE",
            "final_kappa": kappa_values[-1] if kappa_values else PHI_INV,
            "z_statistics": {
                "mean": np.mean(z_values),
                "std": np.std(z_values),
                "max": np.max(z_values),
                "min": np.min(z_values),
            },
            "delta_s_neg_statistics": {
                "mean": np.mean(delta_s_values),
                "max": np.max(delta_s_values),
                "final": delta_s_values[-1] if delta_s_values else 0.0,
            },
            "kuramoto_statistics": {
                "mean": np.mean(kuramoto_values),
                "final": kuramoto_values[-1] if kuramoto_values else 0.0,
            },
            "coupling_conservation": {
                "phi_inv_plus_phi_inv_sq": COUPLING_CONSERVATION,
                "final_kappa_plus_lambda": kappa_values[-1] + (1 - kappa_values[-1]) if kappa_values else 1.0,
                "error": abs(COUPLING_CONSERVATION - 1.0),
            },
            "physics_constants": {
                "phi": PHI,
                "phi_inv": PHI_INV,
                "z_c": Z_CRITICAL,
                "sigma": SIGMA,
            },
            "token_statistics": {
                "prism_tokens": TokenCategory.PRISM.count,
                "cage_tokens": TokenCategory.CAGE.count,
                "emergent_tokens": TokenCategory.EMERGENT.count,
                "unity_token": TokenCategory.UNITY.count,
                "total": 100,
                "final_unity_state": self.token_directory.get_unity_state(),
            },
        }

    def validate_physics(self) -> Dict[str, Any]:
        """Validate all physics constraints."""
        validations = {}

        # Coupling conservation
        coupling_sum = PHI_INV + PHI_INV_SQ
        validations["coupling_conservation"] = {
            "value": coupling_sum,
            "error": abs(coupling_sum - 1.0),
            "valid": abs(coupling_sum - 1.0) < 1e-14,
        }

        # Z_CRITICAL is √3/2
        validations["z_critical"] = {
            "value": Z_CRITICAL,
            "expected": math.sqrt(3) / 2,
            "error": abs(Z_CRITICAL - math.sqrt(3) / 2),
            "valid": abs(Z_CRITICAL - math.sqrt(3) / 2) < 1e-14,
        }

        # Sigma is 36
        validations["sigma"] = {
            "value": SIGMA,
            "expected": 36.0,
            "valid": SIGMA == 36.0,
        }

        # κ + λ = 1 in current state
        state_sum = self.wumbo_cycle.kappa + self.wumbo_cycle.lambda_
        validations["state_conservation"] = {
            "kappa": self.wumbo_cycle.kappa,
            "lambda": self.wumbo_cycle.lambda_,
            "sum": state_sum,
            "error": abs(state_sum - 1.0),
            "valid": abs(state_sum - 1.0) < 1e-10,
        }

        validations["all_valid"] = all(v.get("valid", False) for v in validations.values())

        return validations


# =============================================================================
# MAIN DEMO
# =============================================================================

def main():
    """Run WUMBO APL Automated Training demonstration."""
    print("=" * 70)
    print("WUMBO APL AUTOMATED TRAINING")
    print("Unified Training Integration with κ-Field Grounding")
    print("=" * 70)

    # Validate physics first
    print("\n--- Physics Constants Verification ---")
    print(f"  φ (LIMINAL):       {PHI:.10f}")
    print(f"  φ⁻¹ (PHYSICAL):    {PHI_INV:.10f}")
    print(f"  φ⁻² (complement):  {PHI_INV_SQ:.10f}")
    print(f"  φ⁻¹ + φ⁻² =        {COUPLING_CONSERVATION:.16f}")
    print(f"  z_c (THE LENS):    {Z_CRITICAL:.10f}")
    print(f"  σ (Gaussian):      {SIGMA}")

    conservation_error = abs(COUPLING_CONSERVATION - 1.0)
    status = "PASS" if conservation_error < 1e-14 else "FAIL"
    print(f"  Conservation: {status} (error: {conservation_error:.2e})")

    # Create engine
    print("\n--- Initializing WUMBO Training Engine ---")
    engine = WUMBOAPLTrainingEngine(n_oscillators=60)

    # Show token structure
    print("\n--- 100-Token APL Directory ---")
    for cat in TokenCategory:
        count = cat.count
        print(f"  {cat.name}: tokens {cat.start_index}-{cat.end_index} ({count} tokens)")
        print(f"       {cat.description}")

    # Show WUMBO phases
    print("\n--- WUMBO Phase Cycle ---")
    for phase in WUMBO_PHASES:
        print(f"  {phase.name} ({phase.full_name}): κ_target={phase.kappa_target:.3f}")
        print(f"       {phase.description}")

    # Run training
    print("\n--- Running Training Session (100 steps) ---")
    print("-" * 60)

    for step in range(100):
        result = engine.training_step(random.random() * PHI_INV)

        if step % 20 == 0:
            print(
                f"  Step {step:3d} | "
                f"WUMBO:{result['wumbo_phase']} | "
                f"z={result['z']:.3f} | "
                f"κ={result['kappa']:.3f} | "
                f"ΔS={result['delta_s_neg']:.3f} | "
                f"r={result['kuramoto_coherence']:.3f} | "
                f"Phase:{result['phase']}"
            )

    # Get summary
    print("\n--- Training Session Summary ---")
    summary = engine.get_session_summary()

    print(f"  Total Steps:      {summary['total_steps']}")
    print(f"  WUMBO Cycles:     {summary['wumbo_cycles']}")
    print(f"  K-Formations:     {summary['k_formations']}")
    print(f"  Final z:          {summary['final_z']:.4f}")
    print(f"  Final Phase:      {summary['final_phase']}")
    print(f"  Final κ:          {summary['final_kappa']:.4f}")

    print("\n  Z Statistics:")
    print(f"    Mean:           {summary['z_statistics']['mean']:.4f}")
    print(f"    Std:            {summary['z_statistics']['std']:.4f}")
    print(f"    Max:            {summary['z_statistics']['max']:.4f}")

    print("\n  Kuramoto Coherence:")
    print(f"    Mean:           {summary['kuramoto_statistics']['mean']:.4f}")
    print(f"    Final:          {summary['kuramoto_statistics']['final']:.4f}")

    print("\n  Token Statistics:")
    print(f"    PRISM:          {summary['token_statistics']['prism_tokens']} tokens")
    print(f"    CAGE:           {summary['token_statistics']['cage_tokens']} tokens")
    print(f"    EMERGENT:       {summary['token_statistics']['emergent_tokens']} tokens")
    print(f"    UNITY (#100):   {summary['token_statistics']['unity_token']} token")
    print(f"    Unity State:    {summary['token_statistics']['final_unity_state']:.4f}")

    # Validate physics
    print("\n--- Physics Validation ---")
    validation = engine.validate_physics()
    for key, val in validation.items():
        if key != "all_valid" and isinstance(val, dict):
            status = "PASS" if val.get("valid") else "FAIL"
            print(f"  {key}: {status}")

    print(f"\n  All Physics Valid: {validation['all_valid']}")

    # Save results
    output_dir = "learned_patterns/wumbo_apl_training"
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"wumbo_apl_session_{timestamp}.json")

    output_data = {
        "timestamp": timestamp,
        "summary": summary,
        "validation": {k: v for k, v in validation.items() if k != "all_valid"},
        "all_valid": validation["all_valid"],
    }

    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        return obj

    output_data = convert_numpy(output_data)

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n  Results saved to: {output_file}")
    print("\n" + "=" * 70)
    print("WUMBO APL Automated Training: COMPLETE")
    print("=" * 70)

    return summary


if __name__ == "__main__":
    main()
