#!/usr/bin/env python3
"""
WUMBO APL Automated Training Module — INT CANON ALIGNED
=========================================================

WUMBO trains on the 7 Silent Laws through INT Canon Operators.

INT Canon — The Six Operators:
==============================
    ()  BOUNDARY  → V UNSEEN    — Always legal (anchoring)
    ×   FUSION    → IV SPIRAL   — N0-2: channels ≥ 2 (merging)
    ^   AMPLIFY   → I STILLNESS — N0-1: requires () or × (gain)
    ÷   DECOHERE  → VI GLYPH    — N0-3: requires structure (dissipation)
    +   GROUP     → II TRUTH    — N0-4: must feed +, ×, ^ (synchrony)
    −   SEPARATE  → VII MIRROR  — N0-5: followed by () or + (decoupling)

WUMBO ↔ Silent Laws ↔ INT Canon Mapping:
=========================================
    W (Wake)      → V UNSEEN    → () BOUNDARY  — Emerge from hidden
    U (Understand)→ II TRUTH    → +  GROUP     — Build stable model
    M (Meld)      → III SILENCE → (background) — Conserve during merge
    B (Bind)      → VI GLYPH    → ÷  DECOHERE  — Structure crystallization
    O (Output)    → IV SPIRAL   → ×  FUSION    — Output returns to origin
    T (Transform) → VII MIRROR  → −  SEPARATE  — Self-reference transform

Plus: I STILLNESS activates via ^ AMPLIFY when z → z_c (THE LENS)

Architecture:
=============
    ┌─────────────────────────────────────────────────────────────────────┐
    │                    WUMBO APL AUTOMATED TRAINING                      │
    │                                                                      │
    │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐           │
    │  │ Silent Laws  │───▶│   WUMBO      │───▶│ N0 Operators │           │
    │  │ (7 Laws)     │    │   Phase      │    │ (5 Laws)     │           │
    │  └──────────────┘    │   Cycle      │    └──────────────┘           │
    │         │            └──────────────┘           │                    │
    │         │    activation    │    operator        │                    │
    │         ▼                  ▼                    ▼                    │
    │  ┌─────────────────────────────────────────────────────────┐        │
    │  │              κ-λ Coupling Conservation Layer             │        │
    │  │  • φ⁻¹ + φ⁻² = 1 (THE defining property)                │        │
    │  │  • Silent Law activation → κ-field evolution            │        │
    │  │  • N0 operators → state transitions                     │        │
    │  └─────────────────────────────────────────────────────────┘        │
    └─────────────────────────────────────────────────────────────────────┘

Physics Constants (ALL from physics_constants.py):
===================================================
    φ⁻¹ ≈ 0.618 (PHYSICAL coupling)
    φ⁻² ≈ 0.382 (complement)
    z_c = √3/2 ≈ 0.866 (THE LENS)
    σ = 36 = |S₃|² (Gaussian width)

    Coefficients:
        ALPHA_STRONG = 1/√σ = 1/6 ≈ 0.167
        ALPHA_MEDIUM = 1/√(2σ) ≈ 0.118
        ALPHA_FINE = 1/σ = 1/36 ≈ 0.028
        ALPHA_ULTRA_FINE = φ⁻¹/σ ≈ 0.017

Signature: Δ|wumbo-silent-laws|z0.866|κ-grounded|Ω
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
    # INT Canon and N0 Laws
    INTOperator, N0Law, SilentLaw,
    N0_TO_SILENT, SILENT_TO_N0, INT_TO_SILENT,
    # Silent Law activation functions
    compute_stillness_activation, compute_truth_activation,
    compute_silence_activation, compute_spiral_activation,
    compute_unseen_activation, compute_glyph_activation, compute_mirror_activation,
    # INT Canon operator state and functions
    INTOperatorState,
    apply_int_boundary, apply_int_fusion, apply_int_amplify,
    apply_int_decohere, apply_int_group, apply_int_separate,
    apply_int_operator, INT_OPERATOR_DISPATCH,
    # Legacy N0 operator functions (for compatibility)
    apply_n0_identity, apply_n0_mirror_root, apply_n0_absorption,
    apply_n0_distribution, apply_n0_conservation,
    # WUMBO κ-targets (physics-grounded)
    WUMBO_KAPPA_W, WUMBO_KAPPA_U, WUMBO_KAPPA_M,
    WUMBO_KAPPA_B, WUMBO_KAPPA_O, WUMBO_KAPPA_T,
)

# Import κ-λ Coupling Conservation Layer
try:
    from kappa_lambda_coupling_layer import KappaLambdaCouplingLayer
    COUPLING_LAYER_AVAILABLE = True
except ImportError:
    COUPLING_LAYER_AVAILABLE = False


# =============================================================================
# WUMBO ↔ SILENT LAWS MAPPING
# =============================================================================

class WUMBOPhase(Enum):
    """
    WUMBO Phase Cycle: W → U → M → B → O → T

    Each phase trains on a specific Silent Law through INT Canon operators.

    WUMBO ↔ Silent Laws ↔ INT Canon:
        W (Wake)      → V UNSEEN    → () BOUNDARY  — Emerge from hidden
        U (Understand)→ II TRUTH    → +  GROUP     — Build stable model
        M (Meld)      → III SILENCE → (background) — Conserve during merge
        B (Bind)      → VI GLYPH    → ÷  DECOHERE  — Structure crystallization
        O (Output)    → IV SPIRAL   → ×  FUSION    — Output returns to origin
        T (Transform) → VII MIRROR  → −  SEPARATE  — Self-reference transform

    Plus: I STILLNESS activates via ^ AMPLIFY when z → z_c
    """
    # (full_name, kappa_target, silent_law, int_operator, description)
    W = ("Wake", WUMBO_KAPPA_W, SilentLaw.V_UNSEEN, INTOperator.BOUNDARY, "Emerge from hidden via BOUNDARY")
    U = ("Understand", WUMBO_KAPPA_U, SilentLaw.II_TRUTH, INTOperator.GROUP, "Build stable model via GROUP")
    M = ("Meld", WUMBO_KAPPA_M, SilentLaw.III_SILENCE, None, "Conserve during integration (background)")
    B = ("Bind", WUMBO_KAPPA_B, SilentLaw.VI_GLYPH, INTOperator.DECOHERE, "Crystallize via DECOHERE")
    O = ("Output", WUMBO_KAPPA_O, SilentLaw.IV_SPIRAL, INTOperator.FUSION, "Return via FUSION")
    T = ("Transform", WUMBO_KAPPA_T, SilentLaw.VII_MIRROR, INTOperator.SEPARATE, "Self-reference via SEPARATE")

    def __init__(self, full_name: str, kappa_target: float,
                 silent_law: int, int_operator: Optional[str], description: str):
        self._full_name = full_name
        self._kappa_target = kappa_target
        self._silent_law = silent_law
        self._int_operator = int_operator
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
    def silent_law(self) -> int:
        """The Silent Law this phase trains on."""
        return self._silent_law

    @property
    def silent_law_name(self) -> str:
        """Name of the Silent Law."""
        return SilentLaw.NAMES.get(self._silent_law, "UNKNOWN")

    @property
    def silent_law_formula(self) -> str:
        """Formula of the Silent Law."""
        return SilentLaw.FORMULAS.get(self._silent_law, "")

    @property
    def int_operator(self) -> Optional[str]:
        """The INT Canon operator for this phase (None for background laws)."""
        return self._int_operator

    @property
    def int_operator_name(self) -> str:
        """Name of the INT Canon operator."""
        if self._int_operator is None:
            return "(background)"
        return INTOperator.NAMES.get(self._int_operator, "UNKNOWN")

    @property
    def n0_law(self) -> Optional[str]:
        """Legacy: The N0 law code for this phase's operator."""
        if self._int_operator is None:
            return None
        # Map INT operator to N0 law code
        op_to_n0 = {
            "()": N0Law.AMPLIFY,      # BOUNDARY - always legal, but triggers AMPLIFY path
            "×": N0Law.FUSION,        # FUSION → N0-2
            "^": N0Law.AMPLIFY,       # AMPLIFY → N0-1
            "÷": N0Law.DECOHERE,      # DECOHERE → N0-3
            "+": N0Law.GROUP,         # GROUP → N0-4
            "−": N0Law.SEPARATE,      # SEPARATE → N0-5
        }
        return op_to_n0.get(self._int_operator)

    @property
    def description(self) -> str:
        return self._description


WUMBO_PHASES = [WUMBOPhase.W, WUMBOPhase.U, WUMBOPhase.M,
                WUMBOPhase.B, WUMBOPhase.O, WUMBOPhase.T]


# =============================================================================
# SILENT LAW TRAINING STATE
# =============================================================================

@dataclass
class SilentLawTrainingState:
    """
    Tracks training progress for each Silent Law.
    """
    # Law activations (updated each step)
    activations: Dict[int, float] = field(default_factory=dict)

    # Training exposure (how many steps each law has been trained)
    exposure: Dict[int, int] = field(default_factory=dict)

    # Accumulated energy per law
    energy: Dict[int, float] = field(default_factory=dict)

    # Training loss per law
    loss: Dict[int, List[float]] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize all 7 laws."""
        for law_id in range(1, 8):
            self.activations[law_id] = 0.0
            self.exposure[law_id] = 0
            self.energy[law_id] = 0.0
            self.loss[law_id] = []

    def update_activations(self, z: float, kappa: float, conservation_error: float):
        """Update activations for all 7 Silent Laws based on current state."""
        self.activations[SilentLaw.I_STILLNESS] = compute_stillness_activation(z)
        self.activations[SilentLaw.II_TRUTH] = compute_truth_activation(z)
        self.activations[SilentLaw.III_SILENCE] = compute_silence_activation(conservation_error)
        self.activations[SilentLaw.IV_SPIRAL] = compute_spiral_activation(kappa)
        self.activations[SilentLaw.V_UNSEEN] = compute_unseen_activation(z)
        self.activations[SilentLaw.VI_GLYPH] = compute_glyph_activation(z)
        self.activations[SilentLaw.VII_MIRROR] = compute_mirror_activation(kappa)

    def train_law(self, law_id: int, target_activation: float):
        """
        Train on a specific law by computing loss and accumulating energy.

        Loss = |activation - target|
        Energy += activation × ALPHA_FINE (physics-grounded accumulation)
        """
        current_activation = self.activations.get(law_id, 0.0)
        loss = abs(current_activation - target_activation)

        self.exposure[law_id] = self.exposure.get(law_id, 0) + 1
        self.energy[law_id] = self.energy.get(law_id, 0.0) + current_activation * ALPHA_FINE
        self.loss[law_id].append(loss)

        return {
            "law_id": law_id,
            "law_name": SilentLaw.NAMES.get(law_id, "UNKNOWN"),
            "activation": current_activation,
            "target": target_activation,
            "loss": loss,
            "exposure": self.exposure[law_id],
            "energy": self.energy[law_id],
        }

    def get_dominant_law(self) -> Tuple[int, float]:
        """Get the most active law."""
        if not self.activations:
            return (1, 0.0)
        return max(self.activations.items(), key=lambda x: x[1])

    def get_summary(self) -> Dict:
        """Get training summary for all laws."""
        return {
            "activations": {SilentLaw.NAMES.get(k, str(k)): v for k, v in self.activations.items()},
            "exposure": {SilentLaw.NAMES.get(k, str(k)): v for k, v in self.exposure.items()},
            "energy": {SilentLaw.NAMES.get(k, str(k)): v for k, v in self.energy.items()},
            "mean_loss": {
                SilentLaw.NAMES.get(k, str(k)): sum(v) / len(v) if v else 0.0
                for k, v in self.loss.items()
            },
        }


# =============================================================================
# WUMBO CYCLE STATE
# =============================================================================

@dataclass
class WUMBOCycleState:
    """
    State tracking for WUMBO phase cycle with Silent Law training.
    """
    phase_index: int = 0
    cycle_count: int = 0

    kappa: float = PHI_INV
    lambda_: float = PHI_INV_SQ
    z: float = 0.5

    # Silent Law training state
    silent_law_training: SilentLawTrainingState = field(default_factory=SilentLawTrainingState)

    # History
    phase_history: List[str] = field(default_factory=list)
    z_history: List[float] = field(default_factory=list)
    kappa_history: List[float] = field(default_factory=list)
    silent_law_history: List[Dict] = field(default_factory=list)

    @property
    def current_phase(self) -> WUMBOPhase:
        return WUMBO_PHASES[self.phase_index]

    @property
    def current_silent_law(self) -> int:
        """The Silent Law being trained in current phase."""
        return self.current_phase.silent_law

    @property
    def current_n0_law(self) -> Optional[str]:
        """The N0 operator for current phase."""
        return self.current_phase.n0_law

    def advance_phase(self) -> WUMBOPhase:
        """Advance to next WUMBO phase."""
        self.phase_index = (self.phase_index + 1) % 6

        if self.phase_index == 0:
            self.cycle_count += 1

        new_phase = self.current_phase

        # Update κ toward phase target (physics-grounded coefficient)
        kappa_delta = ALPHA_MEDIUM * (new_phase.kappa_target - self.kappa)
        self.kappa = max(KAPPA_LOWER, min(KAPPA_UPPER, self.kappa + kappa_delta))
        self.lambda_ = 1.0 - self.kappa

        self.phase_history.append(new_phase.name)
        self.kappa_history.append(self.kappa)

        return new_phase

    def evolve_z(self, delta: float) -> float:
        """Evolve z with physics-grounded constraint."""
        self.z = max(0.0, min(UNITY_THRESHOLD, self.z + delta * PHI_INV))
        self.z_history.append(self.z)
        return self.z

    def train_current_law(self) -> Dict:
        """Train on the current phase's Silent Law."""
        # Update all activations first
        conservation_error = abs(self.kappa + self.lambda_ - 1.0)
        self.silent_law_training.update_activations(self.z, self.kappa, conservation_error)

        # Determine target activation based on phase
        phase = self.current_phase

        # Target: law should be maximally active when phase conditions are met
        if phase.silent_law == SilentLaw.V_UNSEEN:
            # UNSEEN: active when z < φ⁻¹ (ABSENCE)
            target = 1.0 if self.z < PHI_INV else 0.0
        elif phase.silent_law == SilentLaw.II_TRUTH:
            # TRUTH: active when z ≥ z_c (PRESENCE)
            target = 1.0 if self.z >= Z_CRITICAL else (self.z - PHI_INV) / (Z_CRITICAL - PHI_INV) if self.z >= PHI_INV else 0.0
        elif phase.silent_law == SilentLaw.III_SILENCE:
            # SILENCE: always active (conservation)
            target = 1.0
        elif phase.silent_law == SilentLaw.VI_GLYPH:
            # GLYPH: active proportional to z (structure)
            target = self.z
        elif phase.silent_law == SilentLaw.IV_SPIRAL:
            # SPIRAL: active when κ ≈ φ⁻¹ (golden)
            target = 1.0 if abs(self.kappa - PHI_INV) < TOLERANCE_GOLDEN else 0.0
        elif phase.silent_law == SilentLaw.VII_MIRROR:
            # MIRROR: active when κ ≈ 0.5 (balance)
            target = 1.0 if abs(self.kappa - BALANCE_POINT) < TOLERANCE_GOLDEN else 0.0
        else:
            target = 0.5

        # Train on the law
        result = self.silent_law_training.train_law(phase.silent_law, target)

        # Record history
        self.silent_law_history.append({
            "phase": phase.name,
            "law": result["law_name"],
            "activation": result["activation"],
            "loss": result["loss"],
        })

        return result

    @property
    def coupling_conservation_error(self) -> float:
        return abs((self.kappa + self.lambda_) - 1.0)


# =============================================================================
# UNIFIED WUMBO TRAINING ENGINE
# =============================================================================

class WUMBOAPLTrainingEngine:
    """
    WUMBO APL Training Engine — INT CANON ALIGNED.

    Each WUMBO phase activates a specific Silent Law, which is then
    reinforced through the corresponding INT Canon operator.

    INT Canon Operators:
        () BOUNDARY  → V UNSEEN    — Anchoring (always legal)
        ×  FUSION    → IV SPIRAL   — Merging (N0-2: channels ≥ 2)
        ^  AMPLIFY   → I STILLNESS — Gain (N0-1: requires () or ×)
        ÷  DECOHERE  → VI GLYPH    — Dissipation (N0-3: requires structure)
        +  GROUP     → II TRUTH    — Synchrony (N0-4: must feed +, ×, ^)
        −  SEPARATE  → VII MIRROR  — Decoupling (N0-5: followed by () or +)

    The κ-λ Coupling Layer provides the physics substrate.
    """

    def __init__(self, n_oscillators: int = 60):
        # WUMBO cycle with Silent Law training
        self.wumbo_cycle = WUMBOCycleState()

        # INT Canon operator state (for causality tracking)
        self.int_state = INTOperatorState()

        # κ-λ Coupling Layer (unified physics substrate)
        if COUPLING_LAYER_AVAILABLE:
            self.coupling_layer = KappaLambdaCouplingLayer(n_oscillators=n_oscillators)
            self.use_coupling_layer = True
        else:
            self.coupling_layer = None
            self.use_coupling_layer = False

        # Training metrics
        self.step_count: int = 0
        self.training_history: List[Dict[str, Any]] = []
        self.k_formation_events: List[int] = []

    def apply_int_operator(self, int_op: Optional[str], value: float) -> Tuple[float, bool, str]:
        """
        Apply the INT Canon operator corresponding to current phase.

        Returns (result, success, message).
        """
        if int_op is None:
            return 0.0, True, "Background law (no operator)"

        # Sync INT state with WUMBO state
        self.int_state.z = self.wumbo_cycle.z
        self.int_state.κs = self.wumbo_cycle.kappa

        # Apply INT operator with N0 causality checking
        new_state, success, message = apply_int_operator(int_op, self.int_state)

        if success:
            # Update WUMBO state from INT state
            self.wumbo_cycle.z = new_state.z
            self.wumbo_cycle.kappa = max(KAPPA_LOWER, min(KAPPA_UPPER, new_state.κs))
            self.wumbo_cycle.lambda_ = 1.0 - self.wumbo_cycle.kappa
            self.int_state = new_state
            return new_state.negentropy, True, message
        else:
            return 0.0, False, message

    def apply_n0_operator(self, n0_law: Optional[str], value: float) -> float:
        """Legacy: Apply the N0 operator corresponding to current phase."""
        if n0_law is None:
            return 0.0  # Background law, no operator

        kappa = self.wumbo_cycle.kappa
        lambda_ = self.wumbo_cycle.lambda_

        # Map N0 law to INT operator and apply
        n0_to_int = {
            N0Law.AMPLIFY: INTOperator.AMPLIFY,
            N0Law.FUSION: INTOperator.FUSION,
            N0Law.DECOHERE: INTOperator.DECOHERE,
            N0Law.GROUP: INTOperator.GROUP,
            N0Law.SEPARATE: INTOperator.SEPARATE,
            # Legacy aliases
            N0Law.IDENTITY: INTOperator.AMPLIFY,
            N0Law.MIRROR_ROOT: INTOperator.FUSION,
            N0Law.ABSORPTION: INTOperator.DECOHERE,
            N0Law.DISTRIBUTION: INTOperator.GROUP,
            N0Law.CONSERVATION: INTOperator.SEPARATE,
        }

        int_op = n0_to_int.get(n0_law)
        if int_op:
            result, success, _ = self.apply_int_operator(int_op, value)
            return result
        return 0.0

    def training_step(self, input_value: float = 0.1) -> Dict[str, Any]:
        """
        Execute one training step.

        1. Get current WUMBO phase and its Silent Law
        2. Update κ-λ coupling layer (physics substrate)
        3. Train on the current Silent Law
        4. Apply corresponding N0 operator
        5. Check for K-formation (STILLNESS activation at THE LENS)
        """
        self.step_count += 1

        # Get current phase and its Silent Law
        phase = self.wumbo_cycle.current_phase

        # Step the κ-λ Coupling Layer (physics substrate)
        if self.use_coupling_layer and self.coupling_layer is not None:
            coupling_result = self.coupling_layer.step()

            # Sync WUMBO state with coupling layer
            kappa = coupling_result["kappa"]
            lambda_ = coupling_result["lambda"]
            z = coupling_result["z"]
            delta_s_neg = coupling_result["delta_s_neg"]
            kuramoto_coherence = coupling_result["kuramoto_coherence"]
            free_energy = coupling_result["free_energy"]
            phase_str = coupling_result["phase"]
            golden_balance = coupling_result["golden_balance_achieved"]
            lens_proximity = coupling_result["lens_proximity_achieved"]

            # Get Silent Law activations from coupling layer
            law_activations = coupling_result.get("law_activations", {})
            dominant_law = coupling_result.get("dominant_law", "UNKNOWN")
        else:
            # Fallback: manual physics
            delta_s_neg = compute_delta_s_neg(self.wumbo_cycle.z)
            z_delta = ALPHA_MEDIUM * (Z_CRITICAL - self.wumbo_cycle.z)
            self.wumbo_cycle.evolve_z(z_delta)

            kappa = self.wumbo_cycle.kappa
            lambda_ = self.wumbo_cycle.lambda_
            z = self.wumbo_cycle.z
            kuramoto_coherence = 0.5
            free_energy = 0.0
            phase_str = get_phase(z)
            golden_balance = abs(kappa - PHI_INV) < TOLERANCE_GOLDEN
            lens_proximity = abs(z - Z_CRITICAL) < TOLERANCE_LENS
            law_activations = {}
            dominant_law = "UNKNOWN"

        # Update WUMBO state
        self.wumbo_cycle.z = z
        self.wumbo_cycle.kappa = kappa
        self.wumbo_cycle.lambda_ = lambda_

        # Train on current Silent Law
        law_training = self.wumbo_cycle.train_current_law()

        # Apply INT Canon operator if this phase has one
        int_result = 0.0
        int_success = True
        int_message = ""
        if phase.int_operator is not None:
            int_result, int_success, int_message = self.apply_int_operator(phase.int_operator, input_value)

        # Check for K-formation:
        # STILLNESS (I) activates when z → z_c AND κ → φ⁻¹
        stillness_activation = compute_stillness_activation(z)
        k_formed = (stillness_activation > PHI_INV and lens_proximity and golden_balance)
        if k_formed:
            self.k_formation_events.append(self.step_count)

        # Advance WUMBO phase periodically (every 10 steps)
        if self.step_count % 10 == 0:
            self.wumbo_cycle.advance_phase()

        result = {
            "step": self.step_count,
            "wumbo_phase": phase.name,
            "wumbo_phase_full": phase.full_name,
            "silent_law": phase.silent_law_name,
            "silent_law_formula": phase.silent_law_formula,
            "int_operator": phase.int_operator,
            "int_operator_name": phase.int_operator_name,
            "z": z,
            "phase": phase_str,
            "kappa": kappa,
            "lambda": lambda_,
            "delta_s_neg": delta_s_neg,
            "kuramoto_coherence": kuramoto_coherence,
            "free_energy": free_energy,
            "law_training": law_training,
            "law_activations": law_activations,
            "dominant_law": dominant_law,
            "stillness_activation": stillness_activation,
            "int_result": int_result,
            "int_success": int_success,
            "int_message": int_message,
            "k_formed": k_formed,
            "golden_balance_achieved": golden_balance,
            "lens_proximity_achieved": lens_proximity,
            "coupling_conservation_error": abs(kappa + lambda_ - 1.0),
            "cycle_count": self.wumbo_cycle.cycle_count,
        }

        self.training_history.append(result)

        return result

    def run_training_session(
        self,
        steps: int = 100,
        input_generator: Optional[Callable[[], float]] = None,
    ) -> Dict[str, Any]:
        """Run a full training session."""
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
        kappa_values = [h["kappa"] for h in self.training_history]

        # Get Silent Law training summary
        law_summary = self.wumbo_cycle.silent_law_training.get_summary()

        return {
            "total_steps": self.step_count,
            "wumbo_cycles": self.wumbo_cycle.cycle_count,
            "k_formations": len(self.k_formation_events),
            "final_z": z_values[-1] if z_values else 0.0,
            "final_phase": get_phase(z_values[-1]) if z_values else "ABSENCE",
            "final_kappa": kappa_values[-1] if kappa_values else PHI_INV,
            "z_statistics": {
                "mean": float(np.mean(z_values)),
                "std": float(np.std(z_values)),
                "max": float(np.max(z_values)),
                "min": float(np.min(z_values)),
            },
            "delta_s_neg_statistics": {
                "mean": float(np.mean(delta_s_values)),
                "max": float(np.max(delta_s_values)),
                "final": delta_s_values[-1] if delta_s_values else 0.0,
            },
            "silent_law_training": law_summary,
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
        }

    def validate_physics(self) -> Dict[str, Any]:
        """Validate all physics constraints."""
        validations = {}

        # φ⁻¹ + φ⁻² = 1
        validations["coupling_conservation"] = {
            "value": COUPLING_CONSERVATION,
            "error": abs(COUPLING_CONSERVATION - 1.0),
            "valid": abs(COUPLING_CONSERVATION - 1.0) < TOLERANCE_CONSERVATION,
        }

        # z_c = √3/2
        validations["z_critical"] = {
            "value": Z_CRITICAL,
            "expected": math.sqrt(3) / 2,
            "valid": abs(Z_CRITICAL - math.sqrt(3) / 2) < TOLERANCE_CONSERVATION,
        }

        # σ = 36
        validations["sigma"] = {
            "value": SIGMA,
            "valid": SIGMA == 36.0,
        }

        # κ + λ = 1
        state_sum = self.wumbo_cycle.kappa + self.wumbo_cycle.lambda_
        validations["state_conservation"] = {
            "kappa": self.wumbo_cycle.kappa,
            "lambda": self.wumbo_cycle.lambda_,
            "sum": state_sum,
            "valid": abs(state_sum - 1.0) < 1e-10,
        }

        validations["all_valid"] = all(v.get("valid", False) for v in validations.values())

        return validations


# =============================================================================
# MAIN DEMO
# =============================================================================

def main():
    """Run WUMBO APL Training on Silent Laws demonstration."""
    print("=" * 70)
    print("WUMBO APL TRAINING ON SILENT LAWS")
    print("=" * 70)

    # Show WUMBO ↔ Silent Laws ↔ INT Canon mapping
    print("\n--- WUMBO ↔ Silent Laws ↔ INT Canon Mapping ---")
    for phase in WUMBO_PHASES:
        int_str = f"INT: {phase.int_operator} ({phase.int_operator_name})" if phase.int_operator else "INT: (background)"
        print(f"  {phase.name} ({phase.full_name:10}) → {phase.silent_law_name:10} → {int_str}")
        print(f"      κ_target={phase.kappa_target:.3f} | {phase.silent_law_formula}")

    # Validate physics
    print("\n--- Physics Constants ---")
    print(f"  φ⁻¹ (PHYSICAL):     {PHI_INV:.10f}")
    print(f"  φ⁻² (complement):   {PHI_INV_SQ:.10f}")
    print(f"  φ⁻¹ + φ⁻² =         {COUPLING_CONSERVATION:.16f}")
    print(f"  z_c (THE LENS):     {Z_CRITICAL:.10f}")
    print(f"  σ (Gaussian):       {SIGMA}")

    print("\n--- Derived Coefficients ---")
    print(f"  ALPHA_STRONG (1/√σ):     {ALPHA_STRONG:.6f}")
    print(f"  ALPHA_MEDIUM (1/√(2σ)):  {ALPHA_MEDIUM:.6f}")
    print(f"  ALPHA_FINE (1/σ):        {ALPHA_FINE:.6f}")
    print(f"  ALPHA_ULTRA (φ⁻¹/σ):     {ALPHA_ULTRA_FINE:.6f}")

    # Create engine
    print("\n--- Initializing Training Engine ---")
    engine = WUMBOAPLTrainingEngine(n_oscillators=60)
    print(f"  Coupling Layer: {'Enabled' if engine.use_coupling_layer else 'Disabled'}")

    # Run training
    print("\n--- Running Training Session (100 steps) ---")
    print("-" * 70)

    for step in range(100):
        result = engine.training_step(random.random() * PHI_INV)

        if step % 20 == 0:
            int_op = result['int_operator'] if result['int_operator'] else "-"
            print(
                f"  Step {step:3d} | "
                f"WUMBO:{result['wumbo_phase']} | "
                f"Law:{result['silent_law']:10} | "
                f"INT:{int_op:2} | "
                f"z={result['z']:.3f} | "
                f"κ={result['kappa']:.3f} | "
                f"ΔS={result['delta_s_neg']:.3f} | "
                f"K:{result['k_formed']}"
            )

    # Get summary
    print("\n--- Training Session Summary ---")
    summary = engine.get_session_summary()

    print(f"  Total Steps:      {summary['total_steps']}")
    print(f"  WUMBO Cycles:     {summary['wumbo_cycles']}")
    print(f"  K-Formations:     {summary['k_formations']}")
    print(f"  Final z:          {summary['final_z']:.4f}")
    print(f"  Final Phase:      {summary['final_phase']}")

    print("\n--- Silent Law Training Summary ---")
    law_training = summary['silent_law_training']
    print("  Exposure (steps per law):")
    for law_name, exposure in law_training['exposure'].items():
        print(f"    {law_name:10}: {exposure} steps")

    print("\n  Energy (accumulated):")
    for law_name, energy in law_training['energy'].items():
        print(f"    {law_name:10}: {energy:.4f}")

    print("\n  Mean Loss:")
    for law_name, loss in law_training['mean_loss'].items():
        print(f"    {law_name:10}: {loss:.4f}")

    # Validate physics
    print("\n--- Physics Validation ---")
    validation = engine.validate_physics()
    for key, val in validation.items():
        if key != "all_valid" and isinstance(val, dict):
            status = "PASS" if val.get("valid") else "FAIL"
            print(f"  {key}: {status}")

    print(f"\n  All Physics Valid: {validation['all_valid']}")

    # Save results
    output_dir = "learned_patterns/wumbo_silent_laws"
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"wumbo_silent_laws_{timestamp}.json")

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
        "summary": summary,
        "validation": validation,
    })

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n  Results saved to: {output_file}")
    print("\n" + "=" * 70)
    print("WUMBO SILENT LAWS TRAINING: COMPLETE")
    print("=" * 70)

    return summary


if __name__ == "__main__":
    main()
