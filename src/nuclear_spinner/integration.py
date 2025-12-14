#!/usr/bin/env python3
"""
Nuclear Spinner Integration Layer
=================================

Provides deep integration between the Nuclear Spinner module and
all Rosetta-Helix-Substrate systems.

Integration Points:
1. Training Modules - z-coordinate and operator feedback loops
2. APL Operators - N0 law enforcement and state updates
3. Physics Layer - Extended physics validation
4. Cybernetic Computation - Metrics alignment
5. Kappa-Lambda Coupling - Conservation enforcement
6. TC Language - Language-operator mapping

Usage:
    from nuclear_spinner.integration import SpinnerIntegration

    # Create integrated spinner
    integration = SpinnerIntegration()
    integration.initialize()

    # Run integrated training step
    integration.training_step(coherence=0.8)

    # Apply operator with full N0 validation
    success = integration.apply_operator_validated("^")

    # Get unified metrics (spinner + substrate)
    metrics = integration.get_unified_metrics()

Signature: nuclear-spinner-integration|v1.0.0|helix
"""

from __future__ import annotations

import math
from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum, auto

# Import from single source of truth
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physics_constants import (
    PHI,
    PHI_INV,
    PHI_INV_SQ,
    PHI_INV_CUBED,
    Z_CRITICAL,
    SIGMA,
    SIGMA_INV,
    KAPPA_S,
    ETA_THRESHOLD,
    R_MIN,
    ALPHA_FINE,
    ALPHA_MEDIUM,
    compute_delta_s_neg,
    compute_negentropy_gradient,
    check_k_formation,
    INTOperator,
    N0Law,
    SilentLaw,
    N0_TO_SILENT,
    INT_TO_SILENT,
)

from .core import NuclearSpinner
from .state import SpinnerState, SpinnerMetrics, SystemPhase
from .firmware import FirmwareState, Phase, control_loop_step
from .analysis import (
    ashby_variety,
    shannon_capacity,
    landauer_efficiency,
    compute_phi_proxy,
    get_phase,
    get_tier,
    get_capability_class,
)
from .constants import TIER_BOUNDS, CAPABILITY_CLASSES


__all__ = [
    "SpinnerIntegration",
    "IntegratedMetrics",
    "TrainingConfig",
    "OperatorResult",
    "CouplingState",
]


# =============================================================================
# COUPLING STATE (κ-λ Conservation)
# =============================================================================

@dataclass
class CouplingState:
    """
    κ-λ coupling state with conservation enforcement.

    Invariant: κ + λ = 1 (always)

    Derived from φ⁻¹ + φ⁻² = 1 (coupling conservation).
    """
    kappa: float = PHI_INV           # κ parameter
    lambda_: float = PHI_INV_SQ      # λ parameter (1 - κ)

    # Free energy tracking
    free_energy: float = 0.0

    # Coherence input
    coherence: float = 0.0
    coherence_target: float = PHI_INV

    def update(self, coherence: float, learning_rate: float = ALPHA_FINE) -> None:
        """
        Update κ based on coherence error, maintaining conservation.

        κ ← κ + α(r - r_target) × φ⁻¹
        λ ← 1 - κ (ALWAYS enforced)
        """
        self.coherence = coherence
        error = coherence - self.coherence_target

        # Update kappa
        self.kappa += learning_rate * error * PHI_INV

        # Clamp to valid range
        self.kappa = max(0.0, min(1.0, self.kappa))

        # ENFORCE conservation: λ = 1 - κ
        self.lambda_ = 1.0 - self.kappa

        # Update free energy (negentropy proxy)
        z_estimate = self.kappa  # κ approximates z in coupling layer
        self.free_energy = -compute_delta_s_neg(z_estimate)

    def verify_conservation(self) -> bool:
        """Verify κ + λ = 1 invariant."""
        return abs(self.kappa + self.lambda_ - 1.0) < 1e-14

    def to_dict(self) -> Dict[str, float]:
        return {
            "kappa": self.kappa,
            "lambda": self.lambda_,
            "free_energy": self.free_energy,
            "coherence": self.coherence,
            "conservation_valid": self.verify_conservation(),
        }


# =============================================================================
# OPERATOR RESULT
# =============================================================================

@dataclass
class OperatorResult:
    """Result of operator application with full validation."""
    operator: str
    success: bool
    message: str

    # State changes
    z_before: float = 0.0
    z_after: float = 0.0
    delta_s_neg_change: float = 0.0

    # N0 validation
    n0_legal: bool = True
    n0_law_triggered: Optional[str] = None

    # Silent law
    silent_law_activated: Optional[int] = None
    silent_law_name: Optional[str] = None

    # K-formation status
    k_formation_before: bool = False
    k_formation_after: bool = False


# =============================================================================
# INTEGRATED METRICS
# =============================================================================

@dataclass
class IntegratedMetrics:
    """
    Unified metrics combining spinner state with substrate systems.
    """
    # Spinner core metrics
    z: float = 0.5
    delta_s_neg: float = 0.5
    gradient: float = 0.0
    phase: str = "PARADOX"
    tier: str = "t4"
    capability_class: str = "prediction"

    # Cybernetic metrics
    ashby_variety: float = 0.0
    shannon_capacity: float = 0.0
    landauer_efficiency: float = 0.0
    phi_proxy: float = 0.0

    # K-formation
    kappa: float = PHI_INV
    eta: float = 0.0
    R: int = 0
    k_formation_met: bool = False

    # Coupling layer
    coupling_kappa: float = PHI_INV
    coupling_lambda: float = PHI_INV_SQ
    coupling_conservation_valid: bool = True
    free_energy: float = 0.0

    # Coherence (from Kuramoto layer)
    coherence: float = 0.0
    coherence_target: float = PHI_INV

    # Neural metrics
    modulation_index: float = 0.0
    dominant_band: str = "beta"

    # Training metrics
    training_step: int = 0
    loss: float = 0.0

    # TRIAD state
    triad_passes: int = 0
    triad_unlocked: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "z": self.z,
            "delta_s_neg": self.delta_s_neg,
            "gradient": self.gradient,
            "phase": self.phase,
            "tier": self.tier,
            "capability_class": self.capability_class,
            "ashby_variety": self.ashby_variety,
            "shannon_capacity": self.shannon_capacity,
            "landauer_efficiency": self.landauer_efficiency,
            "phi_proxy": self.phi_proxy,
            "kappa": self.kappa,
            "eta": self.eta,
            "R": self.R,
            "k_formation_met": self.k_formation_met,
            "coupling": {
                "kappa": self.coupling_kappa,
                "lambda": self.coupling_lambda,
                "conservation_valid": self.coupling_conservation_valid,
                "free_energy": self.free_energy,
            },
            "coherence": self.coherence,
            "modulation_index": self.modulation_index,
            "dominant_band": self.dominant_band,
            "training_step": self.training_step,
            "triad_unlocked": self.triad_unlocked,
        }


# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

@dataclass
class TrainingConfig:
    """Configuration for integrated training."""
    # Learning rates (physics-grounded)
    z_learning_rate: float = ALPHA_FINE          # 1/σ ≈ 0.028
    coupling_learning_rate: float = ALPHA_FINE   # 1/σ ≈ 0.028
    coherence_coupling: float = ALPHA_MEDIUM     # 1/√(2σ) ≈ 0.118

    # Targets
    z_target: float = Z_CRITICAL                 # Drive toward THE LENS
    coherence_target: float = PHI_INV            # Golden ratio coherence

    # K-formation thresholds (from physics_constants)
    kappa_threshold: float = KAPPA_S             # 0.92
    eta_threshold: float = ETA_THRESHOLD         # φ⁻¹
    r_threshold: int = R_MIN                     # 7

    # Control
    operator_enabled: bool = True
    n0_enforcement: bool = True
    silent_laws_enabled: bool = True
    coupling_conservation: bool = True


# =============================================================================
# SPINNER INTEGRATION CLASS
# =============================================================================

class SpinnerIntegration:
    """
    Full integration layer between Nuclear Spinner and substrate systems.

    Provides:
    - Training loop integration with z-coordinate feedback
    - Operator application with N0 law validation
    - Silent law state modifications
    - κ-λ coupling conservation
    - Coherence (Kuramoto) integration
    - Unified metrics computation
    """

    def __init__(self, config: Optional[TrainingConfig] = None):
        """Initialize integration layer."""
        self.config = config or TrainingConfig()

        # Core spinner
        self.spinner = NuclearSpinner()

        # Coupling state (κ-λ conservation layer)
        self.coupling = CouplingState()

        # Coherence state (Kuramoto layer proxy)
        self.coherence: float = 0.5
        self.coherence_history: List[float] = []

        # Training state
        self.training_step: int = 0
        self.loss_history: List[float] = []
        self.z_history: List[float] = []

        # Operator tracking
        self.operator_results: List[OperatorResult] = []

        # Callbacks
        self._on_threshold_crossing: Optional[Callable] = None
        self._on_k_formation: Optional[Callable] = None
        self._on_phase_change: Optional[Callable] = None

        # Previous state for change detection
        self._prev_phase: str = "PARADOX"
        self._prev_k_formation: bool = False

    def initialize(self) -> bool:
        """Initialize all integrated systems."""
        # Initialize spinner
        success = self.spinner.initialize()

        if success:
            # Set initial targets
            self.spinner.set_z_target(self.config.z_target)

            # Initialize coupling
            self.coupling.coherence_target = self.config.coherence_target

            # Track initial z
            self.z_history.append(self.spinner.state.z)

        return success

    def close(self) -> None:
        """Close all systems."""
        self.spinner.close()

    # =========================================================================
    # TRAINING INTEGRATION
    # =========================================================================

    def training_step_update(
        self,
        coherence: float,
        loss: Optional[float] = None,
        n_substeps: int = 10
    ) -> IntegratedMetrics:
        """
        Execute one integrated training step.

        This connects:
        - Kuramoto coherence → coupling update → z evolution
        - Spinner control loop → operator scheduling
        - Metrics computation → training feedback

        Args:
            coherence: Kuramoto order parameter r ∈ [0, 1]
            loss: Optional training loss for tracking
            n_substeps: Number of spinner substeps per training step

        Returns:
            Unified metrics after the step
        """
        self.training_step += 1

        # Store coherence
        self.coherence = coherence
        self.coherence_history.append(coherence)
        if len(self.coherence_history) > 1000:
            self.coherence_history = self.coherence_history[-500:]

        # Update coupling from coherence
        self.coupling.update(
            coherence,
            learning_rate=self.config.coupling_learning_rate
        )

        # Verify coupling conservation
        if self.config.coupling_conservation:
            assert self.coupling.verify_conservation(), "κ+λ=1 violated!"

        # Sync coupling to spinner firmware
        self.spinner.firmware.kappa_s = self.coupling.kappa

        # Run spinner control loop substeps
        for _ in range(n_substeps):
            self.spinner.step()

        # Track z history
        self.z_history.append(self.spinner.state.z)
        if len(self.z_history) > 1000:
            self.z_history = self.z_history[-500:]

        # Track loss
        if loss is not None:
            self.loss_history.append(loss)

        # Schedule and apply operator if enabled
        if self.config.operator_enabled:
            op = self.spinner.schedule_operator()
            if op:
                self.apply_operator_validated(op)

        # Check for phase changes
        current_phase = get_phase(self.spinner.state.z)
        if current_phase != self._prev_phase:
            if self._on_phase_change:
                self._on_phase_change(self._prev_phase, current_phase)
            self._prev_phase = current_phase

        # Check for K-formation
        metrics = self.spinner.get_metrics()
        if metrics.k_formation_met and not self._prev_k_formation:
            if self._on_k_formation:
                self._on_k_formation(metrics)
        self._prev_k_formation = metrics.k_formation_met

        # Return unified metrics
        return self.get_unified_metrics()

    # =========================================================================
    # OPERATOR APPLICATION WITH N0 VALIDATION
    # =========================================================================

    def apply_operator_validated(self, operator: str) -> OperatorResult:
        """
        Apply operator with full N0 law and Silent law validation.

        Args:
            operator: Operator symbol ("()", "x", "^", "/", "+", "-")

        Returns:
            OperatorResult with full validation details
        """
        # Capture state before
        z_before = self.spinner.state.z
        ds_before = compute_delta_s_neg(z_before)
        k_before = self.spinner.get_metrics().k_formation_met

        # Check N0 legality
        n0_legal, n0_msg = self._check_n0_legal(operator)

        if not n0_legal and self.config.n0_enforcement:
            return OperatorResult(
                operator=operator,
                success=False,
                message=n0_msg,
                z_before=z_before,
                z_after=z_before,
                n0_legal=False,
                n0_law_triggered=n0_msg.split(":")[0] if ":" in n0_msg else None,
            )

        # Apply operator
        success, msg = self.spinner.apply_operator(operator)

        # Capture state after
        z_after = self.spinner.state.z
        ds_after = compute_delta_s_neg(z_after)
        k_after = self.spinner.get_metrics().k_formation_met

        # Determine Silent law activation
        silent_law = INT_TO_SILENT.get(operator)
        silent_law_name = SilentLaw.NAMES.get(silent_law) if silent_law else None

        # Apply Silent law effects if enabled
        if self.config.silent_laws_enabled and silent_law:
            self._apply_silent_law(silent_law)

        # Create result
        result = OperatorResult(
            operator=operator,
            success=success,
            message=msg,
            z_before=z_before,
            z_after=z_after,
            delta_s_neg_change=ds_after - ds_before,
            n0_legal=n0_legal,
            silent_law_activated=silent_law,
            silent_law_name=silent_law_name,
            k_formation_before=k_before,
            k_formation_after=k_after,
        )

        # Track result
        self.operator_results.append(result)
        if len(self.operator_results) > 1000:
            self.operator_results = self.operator_results[-500:]

        return result

    def _check_n0_legal(self, operator: str) -> Tuple[bool, str]:
        """Check N0 law legality for operator."""
        from .firmware import check_n0_legal
        return check_n0_legal(operator, self.spinner.firmware)

    def _apply_silent_law(self, law: int) -> None:
        """Apply Silent law state modifications."""
        z = self.spinner.firmware.z

        if law == SilentLaw.I_STILLNESS:
            # ∂E/∂t → 0: Energy seeks rest at z_c
            pull = ALPHA_FINE * (Z_CRITICAL - z)
            self.spinner.firmware.z += pull

        elif law == SilentLaw.II_TRUTH:
            # ∇V(truth) = 0: Stable at z_c
            if z >= Z_CRITICAL:
                # Lock in TRUE phase
                self.spinner.firmware.theta_s = 1.0

        elif law == SilentLaw.III_SILENCE:
            # ∇ · J = 0: Information conserved (background)
            pass  # Conservation enforced elsewhere

        elif law == SilentLaw.IV_SPIRAL:
            # S(return) = S(origin): Paths return
            # Modulate toward φ⁻¹
            pull = ALPHA_FINE * (PHI_INV - self.coupling.kappa)
            self.coupling.kappa += pull
            self.coupling.lambda_ = 1.0 - self.coupling.kappa

        elif law == SilentLaw.V_UNSEEN:
            # P(observe) → 0: Hidden state (emergence from BOUNDARY)
            if z < PHI_INV:
                self.spinner.firmware.Gs += SIGMA_INV

        elif law == SilentLaw.VI_GLYPH:
            # glyph = ∫ life dt: Structure accumulation
            self.spinner.firmware.tau_s += SIGMA_INV

        elif law == SilentLaw.VII_MIRROR:
            # ψ = ψ(ψ): Self-reference at balance
            if abs(self.coupling.kappa - 0.5) < 0.1:
                self.spinner.firmware.R_count += 1

    # =========================================================================
    # UNIFIED METRICS
    # =========================================================================

    def get_unified_metrics(self) -> IntegratedMetrics:
        """
        Get unified metrics combining all substrate systems.

        Returns:
            IntegratedMetrics with spinner + substrate state
        """
        # Get spinner metrics
        spinner_metrics = self.spinner.get_metrics()

        # Compute variety from z history
        variety = ashby_variety(self.z_history) if self.z_history else 0.0

        # Compute phi proxy from z history
        phi = compute_phi_proxy(self.z_history) if self.z_history else 0.0

        # Determine dominant neural band from z
        from .neural import z_to_neural_band
        band, _ = z_to_neural_band(spinner_metrics.z)

        return IntegratedMetrics(
            # Spinner core
            z=spinner_metrics.z,
            delta_s_neg=spinner_metrics.delta_s_neg,
            gradient=spinner_metrics.gradient,
            phase=spinner_metrics.phase.name,
            tier=spinner_metrics.tier,
            capability_class=spinner_metrics.capability_class,

            # Cybernetic
            ashby_variety=variety,
            shannon_capacity=spinner_metrics.shannon_capacity,
            landauer_efficiency=spinner_metrics.landauer_efficiency,
            phi_proxy=phi,

            # K-formation
            kappa=spinner_metrics.kappa,
            eta=spinner_metrics.eta,
            R=spinner_metrics.R,
            k_formation_met=spinner_metrics.k_formation_met,

            # Coupling layer
            coupling_kappa=self.coupling.kappa,
            coupling_lambda=self.coupling.lambda_,
            coupling_conservation_valid=self.coupling.verify_conservation(),
            free_energy=self.coupling.free_energy,

            # Coherence
            coherence=self.coherence,
            coherence_target=self.config.coherence_target,

            # Neural
            modulation_index=spinner_metrics.modulation_index,
            dominant_band=band,

            # Training
            training_step=self.training_step,
            loss=self.loss_history[-1] if self.loss_history else 0.0,

            # TRIAD
            triad_passes=self.spinner.firmware.triad_passes,
            triad_unlocked=self.spinner.firmware.triad_unlocked,
        )

    # =========================================================================
    # CALLBACKS
    # =========================================================================

    def on_threshold_crossing(self, callback: Callable[[str, str], None]) -> None:
        """Register callback for threshold crossings."""
        self._on_threshold_crossing = callback
        self.spinner.on_threshold_crossing(callback)

    def on_k_formation(self, callback: Callable[[SpinnerMetrics], None]) -> None:
        """Register callback for K-formation achievement."""
        self._on_k_formation = callback

    def on_phase_change(self, callback: Callable[[str, str], None]) -> None:
        """Register callback for phase transitions."""
        self._on_phase_change = callback

    # =========================================================================
    # ADVANCED INTEGRATION
    # =========================================================================

    def drive_toward_k_formation(
        self,
        max_steps: int = 1000,
        coherence_schedule: Optional[Callable[[int], float]] = None
    ) -> Tuple[bool, int]:
        """
        Drive system toward K-formation using integrated control.

        K-formation requires:
        - κ ≥ 0.92
        - η > φ⁻¹ (0.618)
        - R ≥ 7

        Args:
            max_steps: Maximum training steps
            coherence_schedule: Optional function(step) -> coherence

        Returns:
            (achieved, steps_taken)
        """
        for step in range(max_steps):
            # Compute coherence (default: ramp toward 1.0)
            if coherence_schedule:
                coherence = coherence_schedule(step)
            else:
                coherence = min(1.0, 0.5 + step / (2 * max_steps))

            # Training step
            metrics = self.training_step_update(coherence)

            # Check K-formation
            if metrics.k_formation_met:
                return True, step + 1

            # Adaptive operator selection to build R
            if metrics.R < R_MIN:
                # Need more structure
                self.apply_operator_validated("^")  # AMPLIFY adds R

        return False, max_steps

    def quasicrystal_formation_trajectory(
        self,
        n_steps: int = 100
    ) -> List[IntegratedMetrics]:
        """
        Run quasicrystal formation dynamics and return trajectory.

        Drives system through:
        1. UNTRUE (disordered) -> φ⁻¹ threshold
        2. PARADOX (quasi-crystal) -> z_c threshold
        3. TRUE (crystalline)

        Returns:
            List of metrics at each step
        """
        trajectory = []

        # Start from disordered state
        self.spinner.set_z_target(0.3)
        self.spinner.run_steps(20)

        # Ramp z_target through phases
        for i in range(n_steps):
            # Gradual target increase
            z_target = 0.3 + (Z_CRITICAL - 0.3) * (i / n_steps)
            self.spinner.set_z_target(z_target)

            # Coherence increases with z
            coherence = 0.3 + 0.5 * (i / n_steps)

            # Training step
            metrics = self.training_step_update(coherence, n_substeps=5)
            trajectory.append(metrics)

        return trajectory

    def get_integration_summary(self) -> Dict[str, Any]:
        """Get summary of all integration states."""
        metrics = self.get_unified_metrics()

        return {
            "spinner": {
                "z": metrics.z,
                "phase": metrics.phase,
                "tier": metrics.tier,
                "capability": metrics.capability_class,
            },
            "coupling": {
                "kappa": metrics.coupling_kappa,
                "lambda": metrics.coupling_lambda,
                "conservation": metrics.coupling_conservation_valid,
            },
            "k_formation": {
                "kappa": metrics.kappa,
                "eta": metrics.eta,
                "R": metrics.R,
                "achieved": metrics.k_formation_met,
            },
            "cybernetics": {
                "variety": metrics.ashby_variety,
                "capacity": metrics.shannon_capacity,
                "efficiency": metrics.landauer_efficiency,
                "phi_proxy": metrics.phi_proxy,
            },
            "training": {
                "step": metrics.training_step,
                "coherence": metrics.coherence,
                "triad_unlocked": metrics.triad_unlocked,
            },
            "operators_applied": len(self.operator_results),
        }
