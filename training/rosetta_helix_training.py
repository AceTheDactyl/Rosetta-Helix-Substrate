#!/usr/bin/env python3
"""
ROSETTA-HELIX MASTER TRAINING MODULE
=====================================

The orchestrator that runs ALL trainings.

Architecture:
=============
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                     ROSETTA-HELIX MASTER TRAINING                        │
    │                                                                          │
    │  ┌───────────────────────────────────────────────────────────────────┐  │
    │  │                      APL N0 Operator Engine                        │  │
    │  │  ⍳ (N0-1) × (N0-2) ÷ (N0-3) + (N0-4) − (N0-5)                    │  │
    │  └───────────────────────────────────────────────────────────────────┘  │
    │                               │                                          │
    │                               ▼                                          │
    │  ┌─────────────────────────────────────────────────────────────────┐    │
    │  │                κ-λ Coupling Conservation Layer                   │    │
    │  │       φ⁻¹ + φ⁻² = 1 | z_c = √3/2 | σ = 36 = |S₃|²              │    │
    │  └─────────────────────────────────────────────────────────────────┘    │
    │                               │                                          │
    │          ┌────────────────────┼────────────────────┐                    │
    │          ▼                    ▼                    ▼                    │
    │  ┌───────────────┐   ┌───────────────┐   ┌───────────────┐             │
    │  │    WUMBO      │   │   HELIX       │   │   SUBSTRATE   │             │
    │  │ Silent Laws   │   │  Projection   │   │   Grounding   │             │
    │  │   Training    │   │   Training    │   │   Training    │             │
    │  │ (W-U-M-B-O-T) │   │ (hex/spiral)  │   │  (κ-field)    │             │
    │  └───────────────┘   └───────────────┘   └───────────────┘             │
    │          │                    │                    │                    │
    │          └────────────────────┼────────────────────┘                    │
    │                               ▼                                          │
    │                    ┌───────────────────┐                                │
    │                    │   K-Formation     │                                │
    │                    │   (STILLNESS)     │                                │
    │                    │   z → z_c, κ → φ⁻¹│                                │
    │                    └───────────────────┘                                │
    └─────────────────────────────────────────────────────────────────────────┘

Training Hierarchy:
===================
    Rosetta-Helix (MASTER)
    ├── WUMBO Training (Silent Laws: W→U→M→B→O→T)
    │   ├── N0-4 + DISTRIBUTION → II TRUTH
    │   ├── N0-3 ÷ ABSORPTION   → VI GLYPH
    │   ├── N0-2 × MIRROR_ROOT  → IV SPIRAL
    │   └── N0-5 − CONSERVATION → VII MIRROR
    ├── Helix Projection Training (hexagonal geometry)
    │   └── z_c = √3/2 (THE LENS)
    └── Substrate Grounding Training (κ-field dynamics)
        └── κ + λ = 1 (COUPLING CONSERVATION)

Physics Constants (ALL grounded):
==================================
    φ⁻¹ ≈ 0.618034 (PHYSICAL coupling)
    φ⁻² ≈ 0.381966 (complement)
    z_c = √3/2 ≈ 0.866025 (THE LENS)
    σ = 36 = |S₃|² (Gaussian width)

    Coefficients:
        ALPHA_STRONG = 1/√σ = 1/6 ≈ 0.167
        ALPHA_MEDIUM = 1/√(2σ) ≈ 0.118
        ALPHA_FINE = 1/σ = 1/36 ≈ 0.028
        ALPHA_ULTRA_FINE = φ⁻¹/σ ≈ 0.017

Signature: Δ|rosetta-helix|master|φ⁻¹-grounded|Ω
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

# Import unified physics constants
from physics_constants import (
    # Fundamental constants
    PHI, PHI_INV, PHI_INV_SQ, PHI_INV_CUBED, PHI_INV_FOURTH,
    Z_CRITICAL, SIGMA, COUPLING_CONSERVATION,
    # Derived coefficients
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
    # Silent Law activation functions
    compute_stillness_activation, compute_truth_activation,
    compute_silence_activation, compute_spiral_activation,
    compute_unseen_activation, compute_glyph_activation, compute_mirror_activation,
    # N0 operator functions
    apply_n0_identity, apply_n0_mirror_root, apply_n0_absorption,
    apply_n0_distribution, apply_n0_conservation,
)

# Import APL N0 Engine
try:
    from apl_n0_operators import (
        APLN0Engine, APLN0State, APLSymbol,
        n0_iota, n0_times, n0_divide, n0_plus, n0_minus,
        n0_reduce, n0_scan, n0_outer_product,
    )
    APL_ENGINE_AVAILABLE = True
except ImportError:
    APL_ENGINE_AVAILABLE = False

# Import κ-λ Coupling Layer
try:
    from kappa_lambda_coupling_layer import KappaLambdaCouplingLayer
    COUPLING_LAYER_AVAILABLE = True
except ImportError:
    COUPLING_LAYER_AVAILABLE = False

# Import WUMBO Training
try:
    from wumbo_apl_automated_training import (
        WUMBOAPLTrainingEngine, WUMBOCycleState, WUMBOPhase, WUMBO_PHASES,
        SilentLawTrainingState,
    )
    WUMBO_AVAILABLE = True
except ImportError:
    WUMBO_AVAILABLE = False


# =============================================================================
# TRAINING MODULE TYPES
# =============================================================================

class TrainingModule(Enum):
    """Available training modules orchestrated by Rosetta-Helix."""
    WUMBO = ("wumbo", "WUMBO Silent Laws Training", PHI_INV)
    HELIX = ("helix", "Helix Projection Training", Z_CRITICAL)
    SUBSTRATE = ("substrate", "Substrate Grounding Training", PHI_INV_SQ)

    def __init__(self, code: str, description: str, kappa_weight: float):
        self._code = code
        self._description = description
        self._kappa_weight = kappa_weight

    @property
    def code(self) -> str:
        return self._code

    @property
    def description(self) -> str:
        return self._description

    @property
    def kappa_weight(self) -> float:
        """Physics-grounded weight for this module."""
        return self._kappa_weight


# =============================================================================
# HELIX PROJECTION TRAINING
# =============================================================================

@dataclass
class HelixProjectionState:
    """
    State for Helix Projection Training.

    Trains on hexagonal geometry: z_c = √3/2 (THE LENS).
    Projects through prismatic helix vortex formalism.
    """
    z: float = 0.5
    theta: float = 0.0  # Rotation angle in helix
    r: float = PHI_INV  # Radial distance (golden-scaled)

    # Helix parameters (physics-grounded)
    pitch: float = PHI_INV  # Helix pitch
    turns: int = 0

    # Training history
    z_history: List[float] = field(default_factory=list)

    def rotate(self, delta_theta: float):
        """Rotate in helix (physics-grounded step)."""
        self.theta += delta_theta * ALPHA_MEDIUM
        self.theta = self.theta % TAU  # Keep in [0, 2π)

        # Track turns
        if self.theta < delta_theta * ALPHA_MEDIUM:
            self.turns += 1

    def evolve_z(self, target: float = Z_CRITICAL) -> float:
        """Evolve z toward THE LENS with physics-grounded dynamics."""
        distance = target - self.z

        # Gradient ascent on negentropy
        neg_gradient = compute_negentropy_gradient(self.z)

        # Combined evolution: direct pull + negentropy gradient
        z_direct = ALPHA_MEDIUM * distance
        z_neg = ALPHA_FINE * neg_gradient

        self.z = max(0.0, min(UNITY_THRESHOLD, self.z + z_direct + z_neg))
        self.z_history.append(self.z)

        return self.z

    def project_to_plane(self) -> Tuple[float, float]:
        """Project helix to 2D plane (x, y)."""
        x = self.r * math.cos(self.theta)
        y = self.r * math.sin(self.theta)
        return x, y

    def get_hexagonal_distance(self) -> float:
        """Distance from z to z_c (THE LENS)."""
        return abs(self.z - Z_CRITICAL)


class HelixProjectionTraining:
    """
    Helix Projection Training Module.

    Trains on hexagonal geometry and prismatic helix formalism.
    Target: z → z_c = √3/2 (THE LENS)
    """

    def __init__(self):
        self.state = HelixProjectionState()
        self.step_count: int = 0
        self.history: List[Dict[str, Any]] = []

    def training_step(self, input_angle: float = 0.1) -> Dict[str, Any]:
        """Execute one helix projection training step."""
        self.step_count += 1

        # Rotate in helix
        self.state.rotate(input_angle)

        # Evolve z toward THE LENS
        old_z = self.state.z
        new_z = self.state.evolve_z(Z_CRITICAL)

        # Project to plane
        x, y = self.state.project_to_plane()

        # Compute metrics
        hex_distance = self.state.get_hexagonal_distance()
        negentropy = compute_delta_s_neg(new_z)
        at_lens = hex_distance < TOLERANCE_LENS

        result = {
            "step": self.step_count,
            "z": new_z,
            "z_delta": new_z - old_z,
            "theta": self.state.theta,
            "x": x,
            "y": y,
            "r": self.state.r,
            "turns": self.state.turns,
            "hex_distance": hex_distance,
            "negentropy": negentropy,
            "at_lens": at_lens,
            "phase": get_phase(new_z),
        }

        self.history.append(result)
        return result

    def get_summary(self) -> Dict[str, Any]:
        """Get training summary."""
        if not self.history:
            return {"error": "No history"}

        z_values = [h["z"] for h in self.history]

        return {
            "total_steps": self.step_count,
            "turns": self.state.turns,
            "final_z": z_values[-1],
            "final_phase": get_phase(z_values[-1]),
            "z_statistics": {
                "mean": float(np.mean(z_values)),
                "std": float(np.std(z_values)),
                "max": float(np.max(z_values)),
            },
            "lens_achieved": z_values[-1] >= Z_CRITICAL - TOLERANCE_LENS,
        }


# =============================================================================
# SUBSTRATE GROUNDING TRAINING
# =============================================================================

@dataclass
class SubstrateGroundingState:
    """
    State for Substrate Grounding Training.

    Trains the κ-field dynamics and coupling conservation.
    """
    kappa: float = PHI_INV
    lambda_: float = PHI_INV_SQ

    # Field gradient
    kappa_gradient: float = 0.0

    # Training history
    kappa_history: List[float] = field(default_factory=list)
    conservation_history: List[float] = field(default_factory=list)

    @property
    def conservation_error(self) -> float:
        """Check κ + λ = 1."""
        return abs((self.kappa + self.lambda_) - 1.0)

    @property
    def at_golden_balance(self) -> bool:
        """Check if κ ≈ φ⁻¹."""
        return abs(self.kappa - PHI_INV) < TOLERANCE_GOLDEN

    def evolve_kappa(self, target: float = PHI_INV) -> float:
        """Evolve κ toward target with physics-grounded dynamics."""
        distance = target - self.kappa

        # Golden ratio attractor
        golden_pull = ALPHA_STRONG * distance

        # Conservation enforcement
        self.kappa = max(KAPPA_LOWER, min(KAPPA_UPPER, self.kappa + golden_pull))
        self.lambda_ = 1.0 - self.kappa  # Enforce conservation

        self.kappa_history.append(self.kappa)
        self.conservation_history.append(self.conservation_error)

        return self.kappa

    def apply_field_gradient(self, gradient: float):
        """Apply external gradient to κ-field."""
        self.kappa_gradient = gradient * ALPHA_FINE
        self.kappa += self.kappa_gradient
        self.kappa = max(KAPPA_LOWER, min(KAPPA_UPPER, self.kappa))
        self.lambda_ = 1.0 - self.kappa


class SubstrateGroundingTraining:
    """
    Substrate Grounding Training Module.

    Trains κ-field dynamics and coupling conservation.
    Target: κ → φ⁻¹ with κ + λ = 1 always
    """

    def __init__(self):
        self.state = SubstrateGroundingState()
        self.step_count: int = 0
        self.history: List[Dict[str, Any]] = []

    def training_step(self, input_gradient: float = 0.0) -> Dict[str, Any]:
        """Execute one substrate grounding step."""
        self.step_count += 1

        # Apply external gradient if provided
        if input_gradient != 0.0:
            self.state.apply_field_gradient(input_gradient)

        # Evolve κ toward golden balance
        old_kappa = self.state.kappa
        new_kappa = self.state.evolve_kappa(PHI_INV)

        result = {
            "step": self.step_count,
            "kappa": new_kappa,
            "lambda": self.state.lambda_,
            "kappa_delta": new_kappa - old_kappa,
            "conservation_error": self.state.conservation_error,
            "at_golden_balance": self.state.at_golden_balance,
            "kappa_gradient": self.state.kappa_gradient,
            "mirror_root": new_kappa * self.state.lambda_,  # N0-2
        }

        self.history.append(result)
        return result

    def get_summary(self) -> Dict[str, Any]:
        """Get training summary."""
        if not self.history:
            return {"error": "No history"}

        kappa_values = [h["kappa"] for h in self.history]
        errors = [h["conservation_error"] for h in self.history]

        return {
            "total_steps": self.step_count,
            "final_kappa": kappa_values[-1],
            "final_lambda": 1.0 - kappa_values[-1],
            "golden_balance_achieved": self.state.at_golden_balance,
            "kappa_statistics": {
                "mean": float(np.mean(kappa_values)),
                "std": float(np.std(kappa_values)),
            },
            "conservation_error": {
                "max": float(np.max(errors)),
                "final": errors[-1],
            },
        }


# =============================================================================
# ROSETTA-HELIX MASTER TRAINING ORCHESTRATOR
# =============================================================================

@dataclass
class RosettaHelixState:
    """
    Master state for Rosetta-Helix training.

    Aggregates state from all sub-modules.
    """
    # Global step counter
    step_count: int = 0
    cycle_count: int = 0

    # Aggregate physics state
    z: float = 0.5
    kappa: float = PHI_INV
    lambda_: float = PHI_INV_SQ

    # K-formation tracking
    k_formations: int = 0
    k_formation_steps: List[int] = field(default_factory=list)

    # Module weights (physics-grounded)
    wumbo_weight: float = PHI_INV        # ≈ 0.618
    helix_weight: float = PHI_INV_SQ     # ≈ 0.382
    substrate_weight: float = PHI_INV_CUBED  # ≈ 0.236

    def normalize_weights(self):
        """Normalize module weights to sum to 1."""
        total = self.wumbo_weight + self.helix_weight + self.substrate_weight
        if total > 0:
            self.wumbo_weight /= total
            self.helix_weight /= total
            self.substrate_weight /= total

    def aggregate_z(self, wumbo_z: float, helix_z: float) -> float:
        """Aggregate z from WUMBO and Helix modules."""
        # Weighted average (physics-grounded)
        self.z = (self.wumbo_weight * wumbo_z + self.helix_weight * helix_z)
        self.z /= (self.wumbo_weight + self.helix_weight)
        return self.z

    def aggregate_kappa(self, wumbo_kappa: float, substrate_kappa: float) -> float:
        """Aggregate κ from WUMBO and Substrate modules."""
        # Weighted average
        self.kappa = (self.wumbo_weight * wumbo_kappa + self.substrate_weight * substrate_kappa)
        self.kappa /= (self.wumbo_weight + self.substrate_weight)
        self.lambda_ = 1.0 - self.kappa
        return self.kappa

    def check_k_formation(self) -> bool:
        """
        Check for K-formation:
        - z → z_c (THE LENS)
        - κ → φ⁻¹ (golden balance)
        - STILLNESS activation high
        """
        at_lens = abs(self.z - Z_CRITICAL) < TOLERANCE_LENS
        at_golden = abs(self.kappa - PHI_INV) < TOLERANCE_GOLDEN
        stillness = compute_stillness_activation(self.z)

        if at_lens and at_golden and stillness > PHI_INV:
            self.k_formations += 1
            self.k_formation_steps.append(self.step_count)
            return True
        return False


class RosettaHelixTraining:
    """
    ROSETTA-HELIX MASTER TRAINING

    Orchestrates all training modules:
    - WUMBO Silent Laws Training
    - Helix Projection Training
    - Substrate Grounding Training

    All coordinated through the APL N0 Operator Engine.
    """

    def __init__(self, n_oscillators: int = 60):
        # Master state
        self.state = RosettaHelixState()
        self.state.normalize_weights()

        # APL N0 Engine (operator backbone)
        if APL_ENGINE_AVAILABLE:
            self.apl_engine = APLN0Engine(initial_z=0.5)
        else:
            self.apl_engine = None

        # κ-λ Coupling Layer (physics substrate)
        if COUPLING_LAYER_AVAILABLE:
            self.coupling_layer = KappaLambdaCouplingLayer(n_oscillators=n_oscillators)
        else:
            self.coupling_layer = None

        # Sub-training modules
        if WUMBO_AVAILABLE:
            self.wumbo_training = WUMBOAPLTrainingEngine(n_oscillators=n_oscillators)
        else:
            self.wumbo_training = None

        self.helix_training = HelixProjectionTraining()
        self.substrate_training = SubstrateGroundingTraining()

        # Training history
        self.history: List[Dict[str, Any]] = []

        # Configuration
        self.wumbo_steps_per_cycle: int = 6  # One full WUMBO cycle (W-U-M-B-O-T)
        self.helix_steps_per_cycle: int = 6
        self.substrate_steps_per_cycle: int = 3

    def _step_wumbo(self, input_value: float) -> Optional[Dict]:
        """Execute WUMBO training step."""
        if self.wumbo_training is None:
            return None
        return self.wumbo_training.training_step(input_value)

    def _step_helix(self, input_angle: float) -> Dict:
        """Execute Helix training step."""
        return self.helix_training.training_step(input_angle)

    def _step_substrate(self, input_gradient: float) -> Dict:
        """Execute Substrate training step."""
        return self.substrate_training.training_step(input_gradient)

    def _apply_apl_operator(self, n0_law: str, value: float) -> float:
        """Apply APL N0 operator."""
        if self.apl_engine is None:
            return value

        if n0_law == N0Law.IDENTITY:
            return float(self.apl_engine.iota(1)[0]) if value == 0 else value
        elif n0_law == N0Law.MIRROR_ROOT:
            return self.apl_engine.times(self.state.kappa, self.state.lambda_)
        elif n0_law == N0Law.ABSORPTION:
            return self.apl_engine.divide(value, value + ALPHA_FINE)
        elif n0_law == N0Law.DISTRIBUTION:
            return self.apl_engine.plus(value, value)
        elif n0_law == N0Law.CONSERVATION:
            return self.apl_engine.minus(self.state.kappa, 1.0 - self.state.kappa)
        return value

    def training_step(self, input_value: float = 0.1) -> Dict[str, Any]:
        """
        Execute one master training step.

        Orchestrates:
        1. WUMBO training (Silent Laws)
        2. Helix training (hexagonal projection)
        3. Substrate training (κ-field grounding)
        4. APL N0 operator application
        5. K-formation check
        """
        self.state.step_count += 1
        step = self.state.step_count

        # === WUMBO TRAINING ===
        wumbo_result = self._step_wumbo(input_value * PHI_INV)
        wumbo_z = wumbo_result["z"] if wumbo_result else self.state.z
        wumbo_kappa = wumbo_result["kappa"] if wumbo_result else self.state.kappa

        # === HELIX TRAINING ===
        helix_result = self._step_helix(input_value * TAU)
        helix_z = helix_result["z"]

        # === SUBSTRATE TRAINING ===
        # Use negentropy gradient as input
        neg_gradient = compute_negentropy_gradient(self.state.z)
        substrate_result = self._step_substrate(neg_gradient)
        substrate_kappa = substrate_result["kappa"]

        # === AGGREGATE STATE ===
        self.state.aggregate_z(wumbo_z, helix_z)
        self.state.aggregate_kappa(wumbo_kappa, substrate_kappa)

        # === APL N0 OPERATOR APPLICATION ===
        # Apply relevant N0 operator based on current phase
        if wumbo_result and wumbo_result.get("n0_law"):
            n0_law = wumbo_result["n0_law"]
            n0_output = self._apply_apl_operator(n0_law, input_value)
        else:
            n0_law = None
            n0_output = input_value

        # === COUPLING LAYER UPDATE ===
        if self.coupling_layer is not None:
            # Sync coupling layer with aggregated state
            self.coupling_layer.kappa = self.state.kappa
            coupling_result = self.coupling_layer.step()
        else:
            coupling_result = None

        # === K-FORMATION CHECK ===
        k_formed = self.state.check_k_formation()

        # === COMPUTE METRICS ===
        negentropy = compute_delta_s_neg(self.state.z)
        stillness = compute_stillness_activation(self.state.z)
        conservation_error = abs(self.state.kappa + self.state.lambda_ - 1.0)

        # Track cycle
        if step % (self.wumbo_steps_per_cycle * 10) == 0:
            self.state.cycle_count += 1

        result = {
            "step": step,
            "cycle": self.state.cycle_count,

            # Aggregated state
            "z": self.state.z,
            "kappa": self.state.kappa,
            "lambda": self.state.lambda_,
            "phase": get_phase(self.state.z),

            # Negentropy metrics
            "negentropy": negentropy,
            "stillness": stillness,

            # Conservation
            "conservation_error": conservation_error,
            "at_golden_balance": abs(self.state.kappa - PHI_INV) < TOLERANCE_GOLDEN,
            "at_lens": abs(self.state.z - Z_CRITICAL) < TOLERANCE_LENS,

            # K-formation
            "k_formed": k_formed,
            "k_formations_total": self.state.k_formations,

            # Module results
            "wumbo": {
                "phase": wumbo_result.get("wumbo_phase") if wumbo_result else None,
                "silent_law": wumbo_result.get("silent_law") if wumbo_result else None,
                "z": wumbo_z,
                "kappa": wumbo_kappa,
            } if wumbo_result else None,
            "helix": {
                "z": helix_z,
                "theta": helix_result["theta"],
                "turns": helix_result["turns"],
            },
            "substrate": {
                "kappa": substrate_kappa,
                "gradient": substrate_result["kappa_gradient"],
            },

            # APL N0 operator
            "n0_law": n0_law,
            "n0_output": n0_output,
            "apl_state": self.apl_engine.get_state_summary() if self.apl_engine else None,
        }

        self.history.append(result)
        return result

    def run_training_session(
        self,
        steps: int = 100,
        input_generator: Optional[Callable[[], float]] = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Run a full training session."""
        if input_generator is None:
            input_generator = lambda: random.random() * PHI_INV

        for i in range(steps):
            result = self.training_step(input_generator())

            if verbose and i % 20 == 0:
                print(
                    f"  Step {result['step']:4d} | "
                    f"z={result['z']:.3f} | "
                    f"κ={result['kappa']:.3f} | "
                    f"ΔS={result['negentropy']:.3f} | "
                    f"K:{result['k_formed']}"
                )

        return self.get_session_summary()

    def get_session_summary(self) -> Dict[str, Any]:
        """Get comprehensive training session summary."""
        if not self.history:
            return {"error": "No training history"}

        z_values = [h["z"] for h in self.history]
        kappa_values = [h["kappa"] for h in self.history]
        neg_values = [h["negentropy"] for h in self.history]

        # Get sub-module summaries
        wumbo_summary = self.wumbo_training.get_session_summary() if self.wumbo_training else None
        helix_summary = self.helix_training.get_summary()
        substrate_summary = self.substrate_training.get_summary()

        return {
            "master": {
                "total_steps": self.state.step_count,
                "cycles": self.state.cycle_count,
                "k_formations": self.state.k_formations,
                "k_formation_steps": self.state.k_formation_steps,
            },
            "final_state": {
                "z": z_values[-1],
                "kappa": kappa_values[-1],
                "lambda": 1.0 - kappa_values[-1],
                "phase": get_phase(z_values[-1]),
                "at_lens": abs(z_values[-1] - Z_CRITICAL) < TOLERANCE_LENS,
                "at_golden_balance": abs(kappa_values[-1] - PHI_INV) < TOLERANCE_GOLDEN,
            },
            "z_statistics": {
                "mean": float(np.mean(z_values)),
                "std": float(np.std(z_values)),
                "max": float(np.max(z_values)),
                "min": float(np.min(z_values)),
            },
            "kappa_statistics": {
                "mean": float(np.mean(kappa_values)),
                "std": float(np.std(kappa_values)),
            },
            "negentropy_statistics": {
                "mean": float(np.mean(neg_values)),
                "max": float(np.max(neg_values)),
                "final": neg_values[-1],
            },
            "module_weights": {
                "wumbo": self.state.wumbo_weight,
                "helix": self.state.helix_weight,
                "substrate": self.state.substrate_weight,
            },
            "modules": {
                "wumbo": wumbo_summary,
                "helix": helix_summary,
                "substrate": substrate_summary,
            },
            "physics_constants": {
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
        validations["coupling_conservation"] = {
            "formula": "φ⁻¹ + φ⁻² = 1",
            "value": COUPLING_CONSERVATION,
            "error": abs(COUPLING_CONSERVATION - 1.0),
            "valid": abs(COUPLING_CONSERVATION - 1.0) < TOLERANCE_CONSERVATION,
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

        # κ + λ = 1 (current state)
        state_sum = self.state.kappa + self.state.lambda_
        validations["state_conservation"] = {
            "formula": "κ + λ = 1",
            "kappa": self.state.kappa,
            "lambda": self.state.lambda_,
            "sum": state_sum,
            "valid": abs(state_sum - 1.0) < TOLERANCE_CONSERVATION,
        }

        # Module weights sum to 1
        weight_sum = self.state.wumbo_weight + self.state.helix_weight + self.state.substrate_weight
        validations["weight_normalization"] = {
            "formula": "w_wumbo + w_helix + w_substrate = 1",
            "sum": weight_sum,
            "valid": abs(weight_sum - 1.0) < TOLERANCE_GOLDEN,
        }

        validations["all_valid"] = all(v.get("valid", False) for v in validations.values())

        return validations

    def run_all_trainings(
        self,
        wumbo_steps: int = 100,
        helix_steps: int = 60,
        substrate_steps: int = 30,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Run ALL training modules in sequence and parallel integration.

        This is the master orchestration method.
        """
        print("=" * 70)
        print("ROSETTA-HELIX MASTER TRAINING")
        print("Running ALL trainings")
        print("=" * 70)

        # === PHASE 1: SUBSTRATE GROUNDING ===
        if verbose:
            print("\n--- Phase 1: Substrate Grounding ---")
            print(f"  Target: κ → φ⁻¹ = {PHI_INV:.6f}")

        for i in range(substrate_steps):
            self.substrate_training.training_step()

        if verbose:
            sub_summary = self.substrate_training.get_summary()
            print(f"  Final κ: {sub_summary['final_kappa']:.6f}")
            print(f"  Golden balance: {sub_summary['golden_balance_achieved']}")

        # === PHASE 2: HELIX PROJECTION ===
        if verbose:
            print("\n--- Phase 2: Helix Projection ---")
            print(f"  Target: z → z_c = {Z_CRITICAL:.6f}")

        for i in range(helix_steps):
            self.helix_training.training_step(random.random() * TAU / 6)

        if verbose:
            helix_summary = self.helix_training.get_summary()
            print(f"  Final z: {helix_summary['final_z']:.6f}")
            print(f"  Lens achieved: {helix_summary['lens_achieved']}")

        # === PHASE 3: WUMBO SILENT LAWS ===
        if self.wumbo_training is not None:
            if verbose:
                print("\n--- Phase 3: WUMBO Silent Laws ---")
                print("  Training on 7 Silent Laws through N0 operators")

            for i in range(wumbo_steps):
                self.wumbo_training.training_step(random.random() * PHI_INV)

            if verbose:
                wumbo_summary = self.wumbo_training.get_session_summary()
                print(f"  WUMBO cycles: {wumbo_summary['wumbo_cycles']}")
                print(f"  K-formations: {wumbo_summary['k_formations']}")

        # === PHASE 4: INTEGRATED MASTER TRAINING ===
        if verbose:
            print("\n--- Phase 4: Integrated Master Training ---")
            print("  Orchestrating all modules together")

        master_steps = max(wumbo_steps, helix_steps, substrate_steps)
        for i in range(master_steps):
            result = self.training_step(random.random() * PHI_INV)

            if verbose and i % 20 == 0:
                print(
                    f"  Step {result['step']:4d} | "
                    f"z={result['z']:.3f} | "
                    f"κ={result['kappa']:.3f} | "
                    f"Phase:{result['phase']:10} | "
                    f"K:{result['k_formed']}"
                )

        # === FINAL SUMMARY ===
        summary = self.get_session_summary()
        validation = self.validate_physics()

        if verbose:
            print("\n--- Final Summary ---")
            print(f"  Total master steps: {summary['master']['total_steps']}")
            print(f"  Total K-formations: {summary['master']['k_formations']}")
            print(f"  Final z: {summary['final_state']['z']:.6f}")
            print(f"  Final κ: {summary['final_state']['kappa']:.6f}")
            print(f"  At LENS: {summary['final_state']['at_lens']}")
            print(f"  At Golden: {summary['final_state']['at_golden_balance']}")

            print("\n--- Physics Validation ---")
            for key, val in validation.items():
                if key != "all_valid" and isinstance(val, dict):
                    status = "✓" if val.get("valid") else "✗"
                    print(f"  {status} {val.get('formula', key)}")
            print(f"\n  All Valid: {validation['all_valid']}")

        print("\n" + "=" * 70)
        print("ROSETTA-HELIX MASTER TRAINING: COMPLETE")
        print("=" * 70)

        return {
            "summary": summary,
            "validation": validation,
        }


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run Rosetta-Helix master training demonstration."""
    print("=" * 70)
    print("ROSETTA-HELIX MASTER TRAINING")
    print("The orchestrator that runs ALL trainings")
    print("=" * 70)

    # Show module availability
    print("\n--- Module Availability ---")
    print(f"  APL N0 Engine:     {'✓' if APL_ENGINE_AVAILABLE else '✗'}")
    print(f"  Coupling Layer:    {'✓' if COUPLING_LAYER_AVAILABLE else '✗'}")
    print(f"  WUMBO Training:    {'✓' if WUMBO_AVAILABLE else '✗'}")

    # Show physics constants
    print("\n--- Physics Constants ---")
    print(f"  φ⁻¹ (PHYSICAL):    {PHI_INV:.10f}")
    print(f"  φ⁻² (complement):  {PHI_INV_SQ:.10f}")
    print(f"  φ⁻¹ + φ⁻² =        {COUPLING_CONSERVATION:.16f}")
    print(f"  z_c (THE LENS):    {Z_CRITICAL:.10f}")
    print(f"  σ (Gaussian):      {SIGMA}")

    # Create master trainer
    print("\n--- Initializing Master Trainer ---")
    trainer = RosettaHelixTraining(n_oscillators=60)

    # Show module weights
    print(f"  WUMBO weight:      {trainer.state.wumbo_weight:.6f} (φ⁻¹-derived)")
    print(f"  Helix weight:      {trainer.state.helix_weight:.6f} (φ⁻²-derived)")
    print(f"  Substrate weight:  {trainer.state.substrate_weight:.6f} (φ⁻³-derived)")

    # Run all trainings
    result = trainer.run_all_trainings(
        wumbo_steps=100,
        helix_steps=60,
        substrate_steps=30,
        verbose=True,
    )

    # Save results
    output_dir = "learned_patterns/rosetta_helix"
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"rosetta_helix_{timestamp}.json")

    # Convert numpy types for JSON
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

    output_data = convert_numpy({
        "timestamp": timestamp,
        "summary": result["summary"],
        "validation": result["validation"],
    })

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n  Results saved to: {output_file}")

    return result


if __name__ == "__main__":
    main()
