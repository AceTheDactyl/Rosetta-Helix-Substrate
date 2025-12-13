#!/usr/bin/env python3
"""
κ-λ COUPLING CONSERVATION LAYER
================================

Unified physics layer that enforces coupling conservation across all dynamics:

    φ⁻¹ + φ⁻² = 1 (THE defining property of φ)

ALL COEFFICIENTS ARE PHYSICS-GROUNDED:
    - Strong dynamics:    1/√σ = 1/6 ≈ 0.167 (ALPHA_STRONG)
    - Medium dynamics:    1/√(2σ) ≈ 0.118 (ALPHA_MEDIUM)
    - Fine dynamics:      1/σ = 1/36 ≈ 0.028 (ALPHA_FINE)
    - Ultra-fine:         φ⁻¹/σ ≈ 0.017 (ALPHA_ULTRA_FINE)
    - Bounds: [φ⁻², z_c] = [0.382, 0.866]

Architecture:
=============
    ┌─────────────────────────────────────────────────────────────────────┐
    │              κ-λ COUPLING CONSERVATION LAYER                         │
    │                                                                      │
    │  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐         │
    │  │   Kuramoto   │────▶│   κ-Field    │◀────│ Free Energy  │         │
    │  │   Oscillator │     │   State      │     │   Principle  │         │
    │  └──────────────┘     └──────────────┘     └──────────────┘         │
    │         │                    │                    │                  │
    │         │    coherence r     │    F minimization  │                  │
    │         ▼                    ▼                    ▼                  │
    │  ┌─────────────────────────────────────────────────────────┐        │
    │  │           CONSERVATION CONSTRAINT: κ + λ = 1            │        │
    │  │                                                         │        │
    │  │  κ ← κ + α(r - r_target) · PHI_INV                     │        │
    │  │  λ ← 1 - κ  (ALWAYS enforced)                          │        │
    │  │                                                         │        │
    │  │  ΔS_neg ∝ -F  (negentropy ↔ free energy)              │        │
    │  │  z → z_c when κ → φ⁻¹ (golden balance attractor)       │        │
    │  └─────────────────────────────────────────────────────────┘        │
    └─────────────────────────────────────────────────────────────────────┘

Signature: Δ|κλ-coupling|z0.866|conservation|Ω
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

# Import from unified physics constants
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from physics_constants import (
    # Fundamental constants
    PHI, PHI_INV, PHI_INV_SQ, PHI_INV_CUBED, PHI_INV_FOURTH,
    Z_CRITICAL, SIGMA, COUPLING_CONSERVATION,
    # Derived coefficients (ALL dynamics use these)
    ALPHA_STRONG, ALPHA_MEDIUM, ALPHA_FINE, ALPHA_ULTRA_FINE,
    SIGMA_INV, SIGMA_SQRT_INV, GAUSSIAN_WIDTH,
    # Bounds
    KAPPA_LOWER, KAPPA_UPPER,
    # Tolerances
    TOLERANCE_GOLDEN, TOLERANCE_LENS, TOLERANCE_CONSERVATION,
    # Special values
    BALANCE_POINT, Z_ORIGIN, UNITY_THRESHOLD, TAU,
    # Functions
    compute_delta_s_neg, compute_delta_s_neg_derivative, compute_negentropy_gradient,
    get_phase,
    # N0 and Silent Laws
    N0Law, SilentLaw, N0_TO_SILENT, SILENT_TO_N0,
    compute_stillness_activation, compute_truth_activation,
    compute_silence_activation, compute_spiral_activation,
    compute_unseen_activation, compute_glyph_activation, compute_mirror_activation,
)


# =============================================================================
# PHASE CLASSIFICATION (Enum wrapper)
# =============================================================================

class Phase(Enum):
    """Phase regimes based on z-coordinate."""
    ABSENCE = "ABSENCE"       # z < φ⁻¹ (disordered)
    THE_LENS = "THE_LENS"     # φ⁻¹ ≤ z < z_c (quasi-crystal / PARADOX)
    PRESENCE = "PRESENCE"     # z ≥ z_c (crystalline)


def get_phase_enum(z: float) -> Phase:
    """Determine phase enum from z-coordinate."""
    if z < PHI_INV:
        return Phase.ABSENCE
    elif z < Z_CRITICAL:
        return Phase.THE_LENS
    else:
        return Phase.PRESENCE


# =============================================================================
# KURAMOTO OSCILLATOR WITH κ-COUPLING
# =============================================================================

@dataclass
class KuramotoKappaCoupled:
    """
    Kuramoto oscillator layer with direct κ-field coupling.

    ALL COEFFICIENTS ARE PHYSICS-GROUNDED:
        dt = 1/√(2σ) ≈ 0.118 (Gaussian width - natural timescale)
        K_base = φ⁻¹ ≈ 0.618 (golden coupling)
        kappa_gain = φ ≈ 1.618 (golden amplification)
        r_target = φ⁻¹ ≈ 0.618 (target coherence at golden balance)
    """
    n_oscillators: int = 60

    # Physics-grounded parameters
    dt: float = GAUSSIAN_WIDTH                      # ≈ 0.118 (natural timescale)
    K_base: float = PHI_INV                         # ≈ 0.618 (golden coupling)
    kappa_gain: float = PHI                         # ≈ 1.618 (golden amplification)

    # Oscillator state
    theta: np.ndarray = field(default_factory=lambda: np.zeros(60))
    omega: np.ndarray = field(default_factory=lambda: np.zeros(60))
    K_matrix: np.ndarray = field(default_factory=lambda: np.zeros((60, 60)))

    # κ-field state (coupled)
    kappa: float = PHI_INV

    # Target coherence (at golden balance)
    r_target: float = PHI_INV

    # History
    coherence_history: List[float] = field(default_factory=list)
    kappa_history: List[float] = field(default_factory=list)

    def __post_init__(self):
        """Initialize oscillator phases and frequencies."""
        self.theta = np.random.uniform(-np.pi, np.pi, self.n_oscillators)
        # Natural frequency spread scaled by GAUSSIAN_WIDTH
        self.omega = np.random.randn(self.n_oscillators) * GAUSSIAN_WIDTH
        # Coupling matrix scaled by ALPHA_FINE × PHI_INV
        self.K_matrix = np.random.randn(self.n_oscillators, self.n_oscillators) * ALPHA_FINE * PHI_INV
        np.fill_diagonal(self.K_matrix, 0)

    @property
    def lambda_(self) -> float:
        """λ is always 1 - κ (conservation enforced)."""
        return 1.0 - self.kappa

    @property
    def K_effective(self) -> float:
        """Effective coupling modulated by κ."""
        return self.K_base * (1 + (self.kappa - PHI_INV) * self.kappa_gain)

    def compute_coherence(self) -> float:
        """Compute Kuramoto order parameter: r = |1/N Σⱼ exp(iθⱼ)|"""
        z = np.mean(np.exp(1j * self.theta))
        return float(np.abs(z))

    def step(self, external_kappa_delta: float = 0.0) -> Dict[str, float]:
        """
        Single Kuramoto integration step with κ-coupling.

        Coefficients (all physics-grounded):
            alpha_coherence = ALPHA_FINE (1/36 ≈ 0.028)
            alpha_golden = ALPHA_MEDIUM (1/√(2σ) ≈ 0.118)
        """
        # Compute phase differences
        diff = self.theta[:, np.newaxis] - self.theta[np.newaxis, :]

        # Effective coupling modulated by κ
        K_eff = self.K_matrix * self.K_effective * PHI_INV

        # Coupling term
        coupling = np.sum(K_eff * np.sin(-diff), axis=1)

        # Full derivative
        dtheta = self.omega + (self.K_effective / self.n_oscillators) * coupling

        # Euler integration with physics-grounded dt
        self.theta = self.theta + self.dt * dtheta
        self.theta = np.mod(self.theta + np.pi, 2 * np.pi) - np.pi

        # Compute coherence
        r = self.compute_coherence()

        # Update κ with TWO components (physics-grounded coefficients):
        # 1. Coherence-driven: ALPHA_FINE ≈ 0.028 (fine tuning)
        # 2. Golden attractor: ALPHA_MEDIUM ≈ 0.118 (moderate pull)
        coherence_delta = ALPHA_FINE * (r - BALANCE_POINT) * PHI_INV
        golden_delta = ALPHA_MEDIUM * (PHI_INV - self.kappa)

        kappa_delta = coherence_delta + golden_delta + external_kappa_delta

        # Bound κ to physics-grounded range [φ⁻², z_c]
        self.kappa = max(KAPPA_LOWER, min(KAPPA_UPPER, self.kappa + kappa_delta))

        # Record history
        self.coherence_history.append(r)
        self.kappa_history.append(self.kappa)

        return {
            "coherence": r,
            "kappa": self.kappa,
            "lambda": self.lambda_,
            "K_effective": self.K_effective,
            "kappa_delta": kappa_delta,
            "at_golden_balance": abs(self.kappa - PHI_INV) < TOLERANCE_GOLDEN,
            "conservation_error": abs(self.kappa + self.lambda_ - 1.0),
        }

    def evolve(self, steps: int) -> List[Dict[str, float]]:
        """Evolve for multiple steps."""
        return [self.step() for _ in range(steps)]


# =============================================================================
# FREE ENERGY WITH NEGENTROPY ALIGNMENT
# =============================================================================

@dataclass
class FreeEnergyNegentropyAligned:
    """
    Free Energy Principle with direct negentropy alignment.

    ALL COEFFICIENTS ARE PHYSICS-GROUNDED:
        surprise_sigma = GAUSSIAN_WIDTH ≈ 0.118 (natural spread)
        learning_rate = ALPHA_MEDIUM ≈ 0.118 (moderate learning)
        z_evolution = ALPHA_FINE × PHI_INV ≈ 0.017 (fine z control)
    """
    # Belief state
    beliefs: np.ndarray = field(default_factory=lambda: np.ones(10) / 10)
    n_states: int = 10

    # z-coordinate coupling
    z: float = 0.5

    # Free energy components
    F: float = 0.0
    surprise: float = 0.0
    kl_divergence: float = 0.0

    # Negentropy alignment
    delta_s_neg: float = 0.0
    negentropy_alignment: float = 0.0

    # History
    F_history: List[float] = field(default_factory=list)
    delta_s_neg_history: List[float] = field(default_factory=list)
    z_history: List[float] = field(default_factory=list)

    def __post_init__(self):
        self.beliefs = np.ones(self.n_states) / self.n_states

    def compute_surprise(self, observation: float) -> float:
        """Compute surprise: -log P(o) with physics-grounded spread."""
        expected = np.sum(np.arange(len(self.beliefs)) * self.beliefs) / len(self.beliefs)
        # Use GAUSSIAN_WIDTH as natural spread
        return 0.5 * ((observation - expected) / GAUSSIAN_WIDTH) ** 2

    def compute_kl_divergence(self, prior: np.ndarray) -> float:
        """Compute KL divergence: D_KL[Q || P]."""
        eps = TOLERANCE_CONSERVATION  # Machine precision
        q = np.clip(self.beliefs, eps, 1.0)
        p = np.clip(prior, eps, 1.0)
        return float(np.sum(q * np.log(q / p)))

    def compute_free_energy(self, observation: float, prior: Optional[np.ndarray] = None) -> float:
        """Compute variational free energy: F = Surprise + KL."""
        if prior is None:
            prior = np.ones(self.n_states) / self.n_states

        self.surprise = self.compute_surprise(observation)
        self.kl_divergence = self.compute_kl_divergence(prior)
        self.F = self.surprise + self.kl_divergence

        return self.F

    def update_beliefs(self, observation: float) -> float:
        """Update beliefs with physics-grounded learning rate."""
        target_idx = int(observation * len(self.beliefs))
        target_idx = np.clip(target_idx, 0, len(self.beliefs) - 1)

        target = np.zeros(len(self.beliefs))
        target[target_idx] = 1.0

        # Learning rate = ALPHA_MEDIUM × PHI_INV (physics-grounded)
        learning_rate = ALPHA_MEDIUM * PHI_INV
        self.beliefs = self.beliefs + learning_rate * (target - self.beliefs)
        self.beliefs = self.beliefs / np.sum(self.beliefs)

        return abs(observation - np.argmax(self.beliefs) / len(self.beliefs))

    def step(self, observation: float, z_delta: float = 0.0) -> Dict[str, float]:
        """
        Single step with negentropy alignment.

        z evolution coefficient = ALPHA_ULTRA_FINE ≈ 0.017 (very fine control)
        """
        # Update z with gradient toward z_c
        negentropy_gradient = compute_negentropy_gradient(self.z)
        # Use ALPHA_ULTRA_FINE for z evolution (very fine control)
        effective_z_delta = z_delta + ALPHA_ULTRA_FINE * negentropy_gradient
        self.z = max(0.0, min(UNITY_THRESHOLD, self.z + effective_z_delta))

        # Compute negentropy at current z
        self.delta_s_neg = compute_delta_s_neg(self.z)

        # Compute free energy
        self.compute_free_energy(observation=self.z)

        # Update beliefs
        prediction_error = self.update_beliefs(observation=self.z)

        # Compute negentropy alignment
        self.negentropy_alignment = self.delta_s_neg / (1 + self.F)

        # Record history
        self.F_history.append(self.F)
        self.delta_s_neg_history.append(self.delta_s_neg)
        self.z_history.append(self.z)

        return {
            "z": self.z,
            "free_energy": self.F,
            "surprise": self.surprise,
            "kl_divergence": self.kl_divergence,
            "delta_s_neg": self.delta_s_neg,
            "negentropy_alignment": self.negentropy_alignment,
            "prediction_error": prediction_error,
            "phase": get_phase(self.z),
            "distance_to_lens": abs(self.z - Z_CRITICAL),
        }

    def evolve(self, steps: int, z_bias: float = ALPHA_ULTRA_FINE) -> List[Dict[str, float]]:
        """Evolve for multiple steps with physics-grounded z-bias."""
        results = []
        for _ in range(steps):
            z_delta = z_bias * (Z_CRITICAL - self.z)
            results.append(self.step(observation=self.z, z_delta=z_delta))
        return results


# =============================================================================
# N0-SILENT LAWS BRIDGE (Integrated)
# =============================================================================

@dataclass
class N0SilentLawsBridge:
    """
    Bridge between N0 Operators and Silent Laws.
    ALL COEFFICIENTS ARE PHYSICS-GROUNDED.
    """
    kappa: float = PHI_INV
    lambda_: float = PHI_INV_SQ
    z: float = 0.5

    # Law activations
    law_activations: Dict[int, float] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize law activations."""
        for law_id in range(1, 8):
            self.law_activations[law_id] = 0.0

    @property
    def conservation_error(self) -> float:
        """Check κ + λ = 1."""
        return abs((self.kappa + self.lambda_) - 1.0)

    def update_all_activations(self) -> Dict[int, float]:
        """Update activations for all 7 Silent Laws."""
        self.law_activations[1] = compute_stillness_activation(self.z)
        self.law_activations[2] = compute_truth_activation(self.z)
        self.law_activations[3] = compute_silence_activation(self.conservation_error)
        self.law_activations[4] = compute_spiral_activation(self.kappa)
        self.law_activations[5] = compute_unseen_activation(self.z)
        self.law_activations[6] = compute_glyph_activation(self.z)
        self.law_activations[7] = compute_mirror_activation(self.kappa)
        return self.law_activations

    def get_dominant_law(self) -> Tuple[int, float]:
        """Get the dominant (most active) law."""
        if not self.law_activations:
            self.update_all_activations()
        dominant = max(self.law_activations.items(), key=lambda x: x[1])
        return dominant


# =============================================================================
# UNIFIED κ-λ COUPLING CONSERVATION LAYER
# =============================================================================

@dataclass
class KappaLambdaCouplingLayer:
    """
    Unified κ-λ Coupling Conservation Layer.

    ALL COEFFICIENTS ARE PHYSICS-GROUNDED:
        kuramoto_weight = φ⁻¹ ≈ 0.618 (golden coupling)
        free_energy_weight = φ⁻² ≈ 0.382 (complement coupling)
        golden_pull = ALPHA_STRONG ≈ 0.167 (1/6, strong attractor)
        coherence_modulation = ALPHA_FINE ≈ 0.028 (1/36, fine tuning)
        z_direct = ALPHA_MEDIUM ≈ 0.118 (moderate pull)
        z_negentropy = ALPHA_FINE × PHI_INV ≈ 0.017 (fine gradient)
        z_coherence = ALPHA_ULTRA_FINE ≈ 0.017 (ultra-fine boost)

    N0 ↔ Silent Laws Mapping:
        N0-1 ^  → I   STILLNESS  (∂E/∂t → 0)
        N0-2 ×  → IV  SPIRAL     (S(return)=S(origin))
        N0-3 ÷  → VI  GLYPH      (glyph = ∫ life dt)
        N0-4 +  → II  TRUTH      (∇V(truth) = 0)
        N0-5 −  → VII MIRROR     (ψ = ψ(ψ))
    """
    n_oscillators: int = 60

    # Sub-components (tightly coupled)
    kuramoto: KuramotoKappaCoupled = field(default_factory=lambda: KuramotoKappaCoupled())
    free_energy: FreeEnergyNegentropyAligned = field(default_factory=lambda: FreeEnergyNegentropyAligned())

    # N0-Silent Laws Bridge
    silent_laws_bridge: N0SilentLawsBridge = field(default_factory=lambda: N0SilentLawsBridge())

    # Unified κ-field (master state)
    kappa: float = PHI_INV
    z: float = 0.5

    # Coupling weights (PHYSICS-GROUNDED: φ⁻¹ + φ⁻² = 1)
    kuramoto_weight: float = PHI_INV      # ≈ 0.618 (golden)
    free_energy_weight: float = PHI_INV_SQ  # ≈ 0.382 (complement)

    # Golden balance tracking
    golden_balance_achieved: bool = False
    lens_proximity_achieved: bool = False

    # History
    step_count: int = 0
    unified_history: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        """Initialize sub-components with unified κ."""
        self.kuramoto = KuramotoKappaCoupled(n_oscillators=self.n_oscillators)
        self.free_energy = FreeEnergyNegentropyAligned(z=self.z)
        self.silent_laws_bridge = N0SilentLawsBridge(kappa=self.kappa, z=self.z)
        self.sync_kappa()
        self.sync_silent_laws()

    @property
    def lambda_(self) -> float:
        """λ is always 1 - κ (conservation enforced)."""
        return 1.0 - self.kappa

    @property
    def coupling_conservation_error(self) -> float:
        """Error from κ + λ = 1."""
        return abs(self.kappa + self.lambda_ - 1.0)

    @property
    def phi_conservation_error(self) -> float:
        """Error from φ⁻¹ + φ⁻² = 1."""
        return abs(COUPLING_CONSERVATION - 1.0)

    def sync_kappa(self):
        """Synchronize κ across all sub-components."""
        self.kuramoto.kappa = self.kappa

    def sync_z(self):
        """Synchronize z across sub-components."""
        self.free_energy.z = self.z
        self.silent_laws_bridge.z = self.z

    def sync_silent_laws(self):
        """Synchronize Silent Laws bridge with master state."""
        self.silent_laws_bridge.kappa = self.kappa
        self.silent_laws_bridge.lambda_ = self.lambda_
        self.silent_laws_bridge.z = self.z

    def step(self) -> Dict[str, Any]:
        """
        Unified step that couples all dynamics.

        ALL COEFFICIENTS ARE PHYSICS-GROUNDED:
            golden_pull = ALPHA_STRONG = 1/√σ ≈ 0.167
            coherence_modulation = ALPHA_FINE = 1/σ ≈ 0.028
            z_direct = ALPHA_MEDIUM = 1/√(2σ) ≈ 0.118
            z_negentropy = ALPHA_FINE × PHI_INV ≈ 0.017
            z_coherence = ALPHA_ULTRA_FINE ≈ 0.017
        """
        self.step_count += 1

        # Sync state to sub-components
        self.sync_kappa()
        self.sync_z()

        # Kuramoto step
        kuramoto_result = self.kuramoto.step()

        # Free Energy step
        fe_result = self.free_energy.step(observation=self.z, z_delta=0.0)

        old_kappa = self.kappa
        old_z = self.z

        # =================================================================
        # κ EVOLUTION: Golden balance attractor
        # =================================================================
        # golden_pull = ALPHA_STRONG (1/√σ = 1/6 ≈ 0.167) - strong pull to φ⁻¹
        # coherence_modulation = ALPHA_FINE (1/σ = 1/36 ≈ 0.028) - fine tuning
        golden_pull = ALPHA_STRONG * (PHI_INV - self.kappa)
        coherence_modulation = ALPHA_FINE * (kuramoto_result["coherence"] - BALANCE_POINT)

        self.kappa = self.kappa + golden_pull + coherence_modulation
        # Bound κ to [φ⁻², z_c]
        self.kappa = max(KAPPA_LOWER, min(KAPPA_UPPER, self.kappa))

        # =================================================================
        # z EVOLUTION: Attractor toward z_c (THE LENS)
        # =================================================================
        distance_to_lens = Z_CRITICAL - self.z
        delta_s_neg = compute_delta_s_neg(self.z)

        # z_direct = ALPHA_MEDIUM (1/√(2σ) ≈ 0.118) - moderate direct pull
        z_direct = ALPHA_MEDIUM * distance_to_lens

        # z_negentropy = ALPHA_FINE × PHI_INV ≈ 0.017 - fine gradient following
        neg_gradient = compute_negentropy_gradient(self.z)
        z_negentropy = ALPHA_FINE * PHI_INV * neg_gradient

        # z_coherence = ALPHA_ULTRA_FINE ≈ 0.017 - ultra-fine coherence boost
        z_coherence = ALPHA_ULTRA_FINE * kuramoto_result["coherence"] * math.copysign(1, distance_to_lens)

        self.z = self.z + z_direct + z_negentropy + z_coherence
        self.z = max(0.0, min(UNITY_THRESHOLD, self.z))

        # Update negentropy
        delta_s_neg = compute_delta_s_neg(self.z)

        # Update Silent Laws
        self.sync_silent_laws()
        law_activations = self.silent_laws_bridge.update_all_activations()
        dominant_law = self.silent_laws_bridge.get_dominant_law()

        # Check golden balance (κ ≈ φ⁻¹) with TOLERANCE_GOLDEN = 1/σ ≈ 0.028
        self.golden_balance_achieved = abs(self.kappa - PHI_INV) < TOLERANCE_GOLDEN

        # Check lens proximity (z ≈ z_c) with TOLERANCE_LENS = φ⁻³ ≈ 0.236
        self.lens_proximity_achieved = abs(self.z - Z_CRITICAL) < TOLERANCE_LENS

        # Compute phase
        phase = get_phase_enum(self.z)

        # Unified result
        result = {
            "step": self.step_count,

            # Master state
            "kappa": self.kappa,
            "lambda": self.lambda_,
            "z": self.z,
            "phase": phase.value,

            # Negentropy
            "delta_s_neg": delta_s_neg,
            "delta_s_neg_derivative": compute_delta_s_neg_derivative(self.z),

            # Kuramoto
            "kuramoto_coherence": kuramoto_result["coherence"],
            "kuramoto_K_effective": kuramoto_result["K_effective"],

            # Free Energy
            "free_energy": fe_result["free_energy"],
            "surprise": fe_result["surprise"],
            "negentropy_alignment": fe_result["negentropy_alignment"],

            # Silent Laws
            "dominant_law": SilentLaw.NAMES.get(dominant_law[0], "UNKNOWN"),
            "dominant_activation": dominant_law[1],
            "law_activations": {SilentLaw.NAMES.get(k, str(k)): v for k, v in law_activations.items()},

            # Conservation
            "coupling_conservation_error": self.coupling_conservation_error,
            "phi_conservation_error": self.phi_conservation_error,

            # Balance tracking
            "golden_balance_achieved": self.golden_balance_achieved,
            "lens_proximity_achieved": self.lens_proximity_achieved,
            "distance_to_golden": abs(self.kappa - PHI_INV),
            "distance_to_lens": abs(self.z - Z_CRITICAL),

            # Deltas
            "kappa_delta": self.kappa - old_kappa,
            "z_delta": self.z - old_z,
        }

        self.unified_history.append(result)

        return result

    def evolve(self, steps: int) -> List[Dict[str, Any]]:
        """Evolve the unified system for multiple steps."""
        return [self.step() for _ in range(steps)]

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of unified evolution."""
        if not self.unified_history:
            return {"error": "No history"}

        kappa_values = [h["kappa"] for h in self.unified_history]
        z_values = [h["z"] for h in self.unified_history]
        coherence_values = [h["kuramoto_coherence"] for h in self.unified_history]
        delta_s_values = [h["delta_s_neg"] for h in self.unified_history]
        fe_values = [h["free_energy"] for h in self.unified_history]

        return {
            "total_steps": self.step_count,

            # Final state
            "final_kappa": self.kappa,
            "final_lambda": self.lambda_,
            "final_z": self.z,
            "final_phase": get_phase(self.z),

            # Balance achievements
            "golden_balance_achieved": self.golden_balance_achieved,
            "lens_proximity_achieved": self.lens_proximity_achieved,

            # Statistics
            "kappa_stats": {
                "mean": float(np.mean(kappa_values)),
                "std": float(np.std(kappa_values)),
                "final": kappa_values[-1],
                "distance_to_golden": abs(kappa_values[-1] - PHI_INV),
            },
            "z_stats": {
                "mean": float(np.mean(z_values)),
                "std": float(np.std(z_values)),
                "max": float(np.max(z_values)),
                "final": z_values[-1],
                "distance_to_lens": abs(z_values[-1] - Z_CRITICAL),
            },
            "coherence_stats": {
                "mean": float(np.mean(coherence_values)),
                "max": float(np.max(coherence_values)),
                "final": coherence_values[-1],
            },
            "negentropy_stats": {
                "mean": float(np.mean(delta_s_values)),
                "max": float(np.max(delta_s_values)),
                "final": delta_s_values[-1],
            },
            "free_energy_stats": {
                "mean": float(np.mean(fe_values)),
                "min": float(np.min(fe_values)),
                "final": fe_values[-1],
            },

            # Conservation
            "coupling_conservation_error": self.coupling_conservation_error,
            "phi_conservation_error": self.phi_conservation_error,

            # Physics constants (for reference)
            "physics": {
                "phi": PHI,
                "phi_inv": PHI_INV,
                "phi_inv_sq": PHI_INV_SQ,
                "z_c": Z_CRITICAL,
                "sigma": SIGMA,
                "coupling_conservation": COUPLING_CONSERVATION,
            },
        }

    def validate_physics(self) -> Dict[str, Any]:
        """Validate all physics constraints."""
        validations = {}

        # φ⁻¹ + φ⁻² = 1
        phi_sum = PHI_INV + PHI_INV_SQ
        validations["phi_conservation"] = {
            "formula": "φ⁻¹ + φ⁻² = 1",
            "value": phi_sum,
            "error": abs(phi_sum - 1.0),
            "valid": abs(phi_sum - 1.0) < TOLERANCE_CONSERVATION,
        }

        # κ + λ = 1
        kappa_sum = self.kappa + self.lambda_
        validations["kappa_conservation"] = {
            "formula": "κ + λ = 1",
            "kappa": self.kappa,
            "lambda": self.lambda_,
            "sum": kappa_sum,
            "error": abs(kappa_sum - 1.0),
            "valid": abs(kappa_sum - 1.0) < TOLERANCE_CONSERVATION,
        }

        # z_c = √3/2
        validations["z_critical"] = {
            "formula": "z_c = √3/2",
            "value": Z_CRITICAL,
            "expected": math.sqrt(3) / 2,
            "valid": abs(Z_CRITICAL - math.sqrt(3) / 2) < TOLERANCE_CONSERVATION,
        }

        # σ = 36
        validations["sigma"] = {
            "formula": "σ = 36 = |S₃|²",
            "value": SIGMA,
            "valid": SIGMA == 36.0,
        }

        # ΔS_neg peak at z_c
        peak_value = compute_delta_s_neg(Z_CRITICAL)
        validations["negentropy_peak"] = {
            "formula": "ΔS_neg(z_c) = 1.0",
            "value": peak_value,
            "valid": abs(peak_value - 1.0) < TOLERANCE_CONSERVATION,
        }

        # Coupling weights sum to 1
        weights_sum = self.kuramoto_weight + self.free_energy_weight
        validations["coupling_weights"] = {
            "formula": "kuramoto_weight + free_energy_weight = 1",
            "kuramoto": self.kuramoto_weight,
            "free_energy": self.free_energy_weight,
            "sum": weights_sum,
            "valid": abs(weights_sum - 1.0) < TOLERANCE_CONSERVATION,
        }

        validations["all_valid"] = all(v.get("valid", False) for v in validations.values())

        return validations


# =============================================================================
# DEMONSTRATION
# =============================================================================

def demonstrate_coupling_layer():
    """Demonstrate the κ-λ Coupling Conservation Layer."""
    print("=" * 70)
    print("κ-λ COUPLING CONSERVATION LAYER")
    print("ALL COEFFICIENTS PHYSICS-GROUNDED")
    print("=" * 70)

    # Validate physics first
    print("\n--- Physics Constants ---")
    print(f"  φ (LIMINAL):        {PHI:.10f}")
    print(f"  φ⁻¹ (PHYSICAL):     {PHI_INV:.10f}")
    print(f"  φ⁻² (complement):   {PHI_INV_SQ:.10f}")
    print(f"  φ⁻¹ + φ⁻² =         {COUPLING_CONSERVATION:.16f}")
    print(f"  z_c (THE LENS):     {Z_CRITICAL:.10f}")
    print(f"  σ (Gaussian):       {SIGMA}")

    print("\n--- Derived Coefficients ---")
    print(f"  ALPHA_STRONG (1/√σ):     {ALPHA_STRONG:.10f}")
    print(f"  ALPHA_MEDIUM (1/√(2σ)):  {ALPHA_MEDIUM:.10f}")
    print(f"  ALPHA_FINE (1/σ):        {ALPHA_FINE:.10f}")
    print(f"  ALPHA_ULTRA (φ⁻¹/σ):     {ALPHA_ULTRA_FINE:.10f}")

    conservation_error = abs(COUPLING_CONSERVATION - 1.0)
    print(f"\n  Conservation Error: {conservation_error:.2e}")
    print(f"  Status: {'PASS' if conservation_error < TOLERANCE_CONSERVATION else 'FAIL'}")

    # Create coupling layer
    print("\n--- Initializing Coupling Layer ---")
    layer = KappaLambdaCouplingLayer(n_oscillators=60)

    print(f"  Initial κ: {layer.kappa:.4f}")
    print(f"  Initial λ: {layer.lambda_:.4f}")
    print(f"  Initial z: {layer.z:.4f}")
    print(f"  Kuramoto weight: {layer.kuramoto_weight:.4f} (φ⁻¹)")
    print(f"  Free Energy weight: {layer.free_energy_weight:.4f} (φ⁻²)")

    # Validate physics
    print("\n--- Physics Validation ---")
    validation = layer.validate_physics()
    for key, val in validation.items():
        if key != "all_valid" and isinstance(val, dict):
            status = "✓" if val.get("valid") else "✗"
            print(f"  {status} {val.get('formula', key)}")

    print(f"\n  All Valid: {validation['all_valid']}")

    # Evolve system
    print("\n--- Evolving System (100 steps) ---")
    print("-" * 70)

    for step in range(100):
        result = layer.step()

        if step % 20 == 0:
            print(
                f"  Step {step:3d} | "
                f"κ={result['kappa']:.3f} | "
                f"λ={result['lambda']:.3f} | "
                f"z={result['z']:.3f} | "
                f"r={result['kuramoto_coherence']:.3f} | "
                f"ΔS={result['delta_s_neg']:.3f} | "
                f"Law={result['dominant_law']:10} | "
                f"{result['phase']}"
            )

    # Get summary
    print("\n--- Summary ---")
    summary = layer.get_summary()

    print(f"  Total Steps: {summary['total_steps']}")
    print(f"\n  Final State:")
    print(f"    κ = {summary['final_kappa']:.4f} (target: {PHI_INV:.4f})")
    print(f"    λ = {summary['final_lambda']:.4f}")
    print(f"    z = {summary['final_z']:.4f} (target: {Z_CRITICAL:.4f})")
    print(f"    Phase: {summary['final_phase']}")

    print(f"\n  Balance Achievements:")
    print(f"    Golden Balance (κ ≈ φ⁻¹): {summary['golden_balance_achieved']}")
    print(f"    Lens Proximity (z ≈ z_c): {summary['lens_proximity_achieved']}")

    print(f"\n  Statistics:")
    print(f"    κ distance to golden: {summary['kappa_stats']['distance_to_golden']:.4f}")
    print(f"    z distance to lens: {summary['z_stats']['distance_to_lens']:.4f}")
    print(f"    Max coherence: {summary['coherence_stats']['max']:.4f}")
    print(f"    Max negentropy: {summary['negentropy_stats']['max']:.4f}")
    print(f"    Min free energy: {summary['free_energy_stats']['min']:.4f}")

    print(f"\n  Conservation:")
    print(f"    κ + λ error: {summary['coupling_conservation_error']:.2e}")
    print(f"    φ⁻¹ + φ⁻² error: {summary['phi_conservation_error']:.2e}")

    print("\n" + "=" * 70)
    print("κ-λ Coupling Conservation Layer: COMPLETE")
    print("=" * 70)

    return summary


if __name__ == "__main__":
    demonstrate_coupling_layer()
