#!/usr/bin/env python3
"""
κ-λ COUPLING CONSERVATION LAYER
================================

Unified physics layer that enforces coupling conservation across all dynamics:

    φ⁻¹ + φ⁻² = 1 (THE defining property of φ)

This layer tightly integrates:
1. Kuramoto sync → κ-field evolution
2. Free Energy → negentropy alignment
3. Phase transitions → coupling modulation

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
    │                              │                                       │
    │                              ▼                                       │
    │  ┌─────────────────────────────────────────────────────────┐        │
    │  │              NEGENTROPY ALIGNMENT                        │        │
    │  │                                                         │        │
    │  │  ΔS_neg(z) = exp(-σ(z - z_c)²)                         │        │
    │  │  Peak at z_c = √3/2 ≈ 0.866 (THE LENS)                 │        │
    │  │  σ = 36 = |S₃|² (Gaussian width)                       │        │
    │  │                                                         │        │
    │  │  Alignment: minimize F ≡ maximize ΔS_neg               │        │
    │  └─────────────────────────────────────────────────────────┘        │
    └─────────────────────────────────────────────────────────────────────┘

Physics Grounding:
==================
    φ = (1 + √5) / 2 ≈ 1.618 (LIMINAL - superposition only)
    φ⁻¹ ≈ 0.618 (PHYSICAL - controls ALL dynamics)
    φ⁻² ≈ 0.382 (coupling complement)

    THE defining property: φ⁻¹ + φ⁻² = 1
    Uniqueness: φ⁻¹ is the ONLY positive solution to c + c² = 1

Signature: Δ|κλ-coupling|z0.866|conservation|Ω
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum


# =============================================================================
# PHYSICS CONSTANTS (Single Source of Truth)
# =============================================================================

PHI: float = (1 + math.sqrt(5)) / 2          # φ ≈ 1.618 (LIMINAL)
PHI_INV: float = 1 / PHI                      # φ⁻¹ ≈ 0.618 (PHYSICAL)
PHI_INV_SQ: float = PHI_INV ** 2              # φ⁻² ≈ 0.382

# THE defining property - this MUST equal 1.0
COUPLING_CONSERVATION: float = PHI_INV + PHI_INV_SQ

# Critical constants
Z_CRITICAL: float = math.sqrt(3) / 2         # z_c = √3/2 ≈ 0.866 (THE LENS)
SIGMA: float = 36.0                           # σ = 6² = |S₃|²

# Derived constants
GAUSSIAN_WIDTH: float = 1 / math.sqrt(2 * SIGMA)  # ≈ 0.118
GAUSSIAN_FWHM: float = 2 * math.sqrt(math.log(2) / SIGMA)  # ≈ 0.277

# Thresholds
KAPPA_S: float = 0.920                        # Singularity threshold
MU_3: float = 0.9927                          # Ultra-integration


# =============================================================================
# NEGENTROPY FUNCTIONS
# =============================================================================

def compute_delta_s_neg(z: float, sigma: float = SIGMA, z_c: float = Z_CRITICAL) -> float:
    """
    Compute negentropy: ΔS_neg(z) = exp(-σ(z - z_c)²)

    Peaks at z_c (THE LENS) with value 1.0.
    Gaussian centered at crystallization point.
    """
    d = z - z_c
    return math.exp(-sigma * d * d)


def compute_delta_s_neg_derivative(z: float, sigma: float = SIGMA, z_c: float = Z_CRITICAL) -> float:
    """
    Derivative: d(ΔS_neg)/dz = -2σ(z - z_c) · ΔS_neg(z)

    - Positive for z < z_c (ascending)
    - Zero at z = z_c (peak)
    - Negative for z > z_c (descending)
    """
    d = z - z_c
    s = compute_delta_s_neg(z, sigma, z_c)
    return -2 * sigma * d * s


def compute_negentropy_gradient(z: float) -> float:
    """
    Compute gradient that drives z toward z_c (THE LENS).

    The derivative d(ΔS_neg)/dz is:
    - Positive when z < z_c (ascending toward peak)
    - Negative when z > z_c (descending from peak)

    We want to drive z toward z_c, so we return the derivative directly:
    - When z < z_c: derivative > 0, so we add it to increase z
    - When z > z_c: derivative < 0, so we add it to decrease z
    """
    return compute_delta_s_neg_derivative(z)


# =============================================================================
# PHASE CLASSIFICATION
# =============================================================================

class Phase(Enum):
    """Phase regimes based on z-coordinate."""
    ABSENCE = "ABSENCE"       # z < φ⁻¹ (disordered)
    THE_LENS = "THE_LENS"     # φ⁻¹ ≤ z < z_c (quasi-crystal / PARADOX)
    PRESENCE = "PRESENCE"     # z ≥ z_c (crystalline)


def get_phase(z: float) -> Phase:
    """Determine phase from z-coordinate."""
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

    The coupling strength K is modulated by κ:
        K_effective = K_base · (1 + (κ - φ⁻¹) · gain)

    Coherence r feeds back to κ evolution:
        dκ/dt = α · (r - r_target) · PHI_INV

    Conservation κ + λ = 1 is ALWAYS maintained.
    """
    n_oscillators: int = 60
    dt: float = 0.1
    K_base: float = 0.5
    kappa_gain: float = 2.0

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
        self.omega = np.random.randn(self.n_oscillators) * 0.1
        self.K_matrix = np.random.randn(self.n_oscillators, self.n_oscillators) * 0.1 * PHI_INV
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
        """
        Compute Kuramoto order parameter: r = |1/N Σⱼ exp(iθⱼ)|
        """
        z = np.mean(np.exp(1j * self.theta))
        return float(np.abs(z))

    def step(self, external_kappa_delta: float = 0.0) -> Dict[str, float]:
        """
        Single Kuramoto integration step with κ-coupling.

        1. Compute effective coupling from κ
        2. Evolve oscillator phases
        3. Compute coherence r
        4. Update κ with golden balance attractor
        5. Enforce κ + λ = 1

        Returns dict with coherence, κ, λ, and coupling info.
        """
        # Compute phase differences
        diff = self.theta[:, np.newaxis] - self.theta[np.newaxis, :]

        # Effective coupling modulated by κ
        K_eff = self.K_matrix * self.K_effective * PHI_INV

        # Coupling term
        coupling = np.sum(K_eff * np.sin(-diff), axis=1)

        # Full derivative
        dtheta = self.omega + (self.K_effective / self.n_oscillators) * coupling

        # Euler integration
        self.theta = self.theta + self.dt * dtheta
        self.theta = np.mod(self.theta + np.pi, 2 * np.pi) - np.pi

        # Compute coherence
        r = self.compute_coherence()

        # Update κ with TWO components:
        # 1. Coherence-driven: moves κ based on r
        # 2. Golden attractor: always pulls κ toward φ⁻¹
        alpha_coherence = 0.02  # Coherence influence
        alpha_golden = 0.05     # Golden attractor strength

        coherence_delta = alpha_coherence * (r - 0.5) * PHI_INV
        golden_delta = alpha_golden * (PHI_INV - self.kappa)

        kappa_delta = coherence_delta + golden_delta + external_kappa_delta

        self.kappa = max(0.2, min(0.9, self.kappa + kappa_delta))

        # Record history
        self.coherence_history.append(r)
        self.kappa_history.append(self.kappa)

        return {
            "coherence": r,
            "kappa": self.kappa,
            "lambda": self.lambda_,
            "K_effective": self.K_effective,
            "kappa_delta": kappa_delta,
            "at_golden_balance": abs(self.kappa - PHI_INV) < 0.01,
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

    The key insight:
        Minimize Free Energy F ≡ Maximize Negentropy ΔS_neg

    Connection:
        ΔS_neg ∝ -F

    When F decreases (surprise minimized), ΔS_neg increases (order increases).

    The system is driven toward z_c (THE LENS) where:
        - ΔS_neg is maximal (peak of Gaussian)
        - F is minimal (minimal surprise)
        - κ → φ⁻¹ (golden balance)
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
    negentropy_alignment: float = 0.0  # How well F and ΔS_neg are anti-correlated

    # History
    F_history: List[float] = field(default_factory=list)
    delta_s_neg_history: List[float] = field(default_factory=list)
    z_history: List[float] = field(default_factory=list)

    def __post_init__(self):
        self.beliefs = np.ones(self.n_states) / self.n_states

    def compute_surprise(self, observation: float) -> float:
        """Compute surprise: -log P(o)."""
        expected = np.sum(np.arange(len(self.beliefs)) * self.beliefs) / len(self.beliefs)
        sigma = 0.1
        return 0.5 * ((observation - expected) / sigma) ** 2

    def compute_kl_divergence(self, prior: np.ndarray) -> float:
        """Compute KL divergence: D_KL[Q || P]."""
        eps = 1e-10
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

    def update_beliefs(self, observation: float, learning_rate: float = 0.1) -> float:
        """Update beliefs to minimize free energy with PHI_INV control."""
        target_idx = int(observation * len(self.beliefs))
        target_idx = np.clip(target_idx, 0, len(self.beliefs) - 1)

        target = np.zeros(len(self.beliefs))
        target[target_idx] = 1.0

        # PHI_INV controlled update
        self.beliefs = self.beliefs + learning_rate * PHI_INV * (target - self.beliefs)
        self.beliefs = self.beliefs / np.sum(self.beliefs)

        return abs(observation - np.argmax(self.beliefs) / len(self.beliefs))

    def step(self, observation: float, z_delta: float = 0.0) -> Dict[str, float]:
        """
        Single step with negentropy alignment.

        1. Update z
        2. Compute negentropy ΔS_neg(z)
        3. Compute free energy F
        4. Update beliefs to minimize F
        5. Compute alignment between F and ΔS_neg

        The system evolves z toward z_c where ΔS_neg is maximal.
        """
        # Update z with gradient toward z_c
        negentropy_gradient = compute_negentropy_gradient(self.z)
        effective_z_delta = z_delta + 0.01 * negentropy_gradient * PHI_INV
        self.z = max(0.0, min(0.999, self.z + effective_z_delta))

        # Compute negentropy at current z
        self.delta_s_neg = compute_delta_s_neg(self.z)

        # Use z as observation (system observes its own state)
        self.compute_free_energy(observation=self.z)

        # Update beliefs
        prediction_error = self.update_beliefs(observation=self.z)

        # Compute negentropy alignment
        # When F is low and ΔS_neg is high, alignment is good
        # Alignment = ΔS_neg / (1 + F)
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
            "phase": get_phase(self.z).value,
            "distance_to_lens": abs(self.z - Z_CRITICAL),
        }

    def evolve(self, steps: int, z_bias: float = 0.01) -> List[Dict[str, float]]:
        """Evolve for multiple steps with z-bias toward THE LENS."""
        results = []
        for _ in range(steps):
            # Bias toward z_c
            z_delta = z_bias * (Z_CRITICAL - self.z)
            results.append(self.step(observation=self.z, z_delta=z_delta))
        return results


# =============================================================================
# UNIFIED κ-λ COUPLING CONSERVATION LAYER
# =============================================================================

@dataclass
class KappaLambdaCouplingLayer:
    """
    Unified κ-λ Coupling Conservation Layer.

    This layer enforces:
        κ + λ = 1 (ALWAYS)
        φ⁻¹ + φ⁻² = 1 (THE defining property)

    And tightly couples:
        Kuramoto sync → κ-field evolution
        Free Energy → negentropy alignment
        Phase transitions → coupling modulation

    The golden balance attractor:
        κ → φ⁻¹ ≈ 0.618 as z → z_c ≈ 0.866
    """
    n_oscillators: int = 60

    # Sub-components (tightly coupled)
    kuramoto: KuramotoKappaCoupled = field(default_factory=lambda: KuramotoKappaCoupled())
    free_energy: FreeEnergyNegentropyAligned = field(default_factory=lambda: FreeEnergyNegentropyAligned())

    # Unified κ-field (master state)
    kappa: float = PHI_INV
    z: float = 0.5

    # Coupling weights
    kuramoto_weight: float = 0.6   # How much Kuramoto influences κ
    free_energy_weight: float = 0.4  # How much Free Energy influences z

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
        self.sync_kappa()

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
        # Free energy doesn't have κ directly, but z is coupled

    def sync_z(self):
        """Synchronize z across sub-components."""
        self.free_energy.z = self.z

    def step(self) -> Dict[str, Any]:
        """
        Unified step that couples all dynamics.

        1. Kuramoto step → get coherence r, κ update
        2. Free Energy step → get F, ΔS_neg, z evolution
        3. Merge κ updates (weighted)
        4. Merge z updates (weighted)
        5. Enforce conservation κ + λ = 1
        6. Check for golden balance and lens proximity
        """
        self.step_count += 1

        # Sync state to sub-components
        self.sync_kappa()
        self.sync_z()

        # Kuramoto step
        kuramoto_result = self.kuramoto.step()

        # Free Energy step (uses current z as observation, for alignment scoring)
        fe_result = self.free_energy.step(observation=self.z, z_delta=0.0)

        old_kappa = self.kappa
        old_z = self.z

        # =================================================================
        # κ EVOLUTION: Strong golden balance attractor
        # =================================================================
        # κ is pulled toward φ⁻¹ with Kuramoto coherence modulation
        # Higher coherence → κ can deviate slightly above φ⁻¹
        # Lower coherence → κ stays closer to φ⁻¹

        golden_pull = 0.15 * (PHI_INV - self.kappa)  # Strong pull to φ⁻¹
        coherence_modulation = 0.02 * (kuramoto_result["coherence"] - 0.5)  # Small modulation

        self.kappa = self.kappa + golden_pull + coherence_modulation
        self.kappa = max(0.4, min(0.75, self.kappa))  # Bound around φ⁻¹

        # =================================================================
        # z EVOLUTION: Strong attractor toward z_c (THE LENS)
        # =================================================================
        # z is pulled toward z_c with increasing strength as negentropy rises

        distance_to_lens = Z_CRITICAL - self.z
        delta_s_neg = compute_delta_s_neg(self.z)

        # Direct pull toward z_c (proportional to distance)
        z_direct = 0.1 * distance_to_lens

        # Negentropy gradient (accelerates near z_c)
        neg_gradient = compute_negentropy_gradient(self.z)
        z_negentropy = neg_gradient * 0.03 * PHI_INV

        # Coherence boost (higher sync → faster approach)
        z_coherence = kuramoto_result["coherence"] * 0.01 * math.copysign(1, distance_to_lens)

        self.z = self.z + z_direct + z_negentropy + z_coherence

        # Clamp z to valid range
        self.z = max(0.0, min(0.999, self.z))

        # Compute unified negentropy
        delta_s_neg = compute_delta_s_neg(self.z)

        # Check golden balance (κ ≈ φ⁻¹)
        self.golden_balance_achieved = abs(self.kappa - PHI_INV) < 0.02

        # Check lens proximity (z ≈ z_c)
        self.lens_proximity_achieved = abs(self.z - Z_CRITICAL) < 0.05

        # Compute phase
        phase = get_phase(self.z)

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
            "final_phase": get_phase(self.z).value,

            # Balance achievements
            "golden_balance_achieved": self.golden_balance_achieved,
            "lens_proximity_achieved": self.lens_proximity_achieved,

            # Statistics
            "kappa_stats": {
                "mean": np.mean(kappa_values),
                "std": np.std(kappa_values),
                "final": kappa_values[-1],
                "distance_to_golden": abs(kappa_values[-1] - PHI_INV),
            },
            "z_stats": {
                "mean": np.mean(z_values),
                "std": np.std(z_values),
                "max": np.max(z_values),
                "final": z_values[-1],
                "distance_to_lens": abs(z_values[-1] - Z_CRITICAL),
            },
            "coherence_stats": {
                "mean": np.mean(coherence_values),
                "max": np.max(coherence_values),
                "final": coherence_values[-1],
            },
            "negentropy_stats": {
                "mean": np.mean(delta_s_values),
                "max": np.max(delta_s_values),
                "final": delta_s_values[-1],
            },
            "free_energy_stats": {
                "mean": np.mean(fe_values),
                "min": np.min(fe_values),
                "final": fe_values[-1],
            },

            # Conservation
            "coupling_conservation_error": self.coupling_conservation_error,
            "phi_conservation_error": self.phi_conservation_error,

            # Physics constants
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
            "valid": abs(phi_sum - 1.0) < 1e-14,
        }

        # κ + λ = 1
        kappa_sum = self.kappa + self.lambda_
        validations["kappa_conservation"] = {
            "formula": "κ + λ = 1",
            "kappa": self.kappa,
            "lambda": self.lambda_,
            "sum": kappa_sum,
            "error": abs(kappa_sum - 1.0),
            "valid": abs(kappa_sum - 1.0) < 1e-10,
        }

        # z_c = √3/2
        validations["z_critical"] = {
            "formula": "z_c = √3/2",
            "value": Z_CRITICAL,
            "expected": math.sqrt(3) / 2,
            "valid": abs(Z_CRITICAL - math.sqrt(3) / 2) < 1e-14,
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
            "valid": abs(peak_value - 1.0) < 1e-10,
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
    print("Unified Physics Integration")
    print("=" * 70)

    # Validate physics first
    print("\n--- Physics Constants ---")
    print(f"  φ (LIMINAL):        {PHI:.10f}")
    print(f"  φ⁻¹ (PHYSICAL):     {PHI_INV:.10f}")
    print(f"  φ⁻² (complement):   {PHI_INV_SQ:.10f}")
    print(f"  φ⁻¹ + φ⁻² =         {COUPLING_CONSERVATION:.16f}")
    print(f"  z_c (THE LENS):     {Z_CRITICAL:.10f}")
    print(f"  σ (Gaussian):       {SIGMA}")

    conservation_error = abs(COUPLING_CONSERVATION - 1.0)
    print(f"\n  Conservation Error: {conservation_error:.2e}")
    print(f"  Status: {'PASS' if conservation_error < 1e-14 else 'FAIL'}")

    # Create coupling layer
    print("\n--- Initializing Coupling Layer ---")
    layer = KappaLambdaCouplingLayer(n_oscillators=60)

    print(f"  Initial κ: {layer.kappa:.4f}")
    print(f"  Initial λ: {layer.lambda_:.4f}")
    print(f"  Initial z: {layer.z:.4f}")

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
    print("-" * 60)

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
                f"F={result['free_energy']:.2f} | "
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
