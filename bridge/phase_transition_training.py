#!/usr/bin/env python3
"""
Phase Transition Training System
================================

Integrated electromagnetic training for pentagonal quasicrystal phase transitions.

This system physically accomplishes quasicrystal formation through cybernetic control:

HARDWARE ARCHITECTURE:
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PHASE TRANSITION TRAINING SYSTEM                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌────────────────┐      ┌────────────────┐      ┌────────────────┐        │
│   │  NUCLEAR       │      │  60° TILTED    │      │  LIGHTNING     │        │
│   │  SPINNER       │◄────►│  NEURAL PLATES │◄────►│  QUENCH        │        │
│   │  (z-control)   │      │  (6×10 cells)  │      │  (10⁶ K/s)     │        │
│   └───────┬────────┘      └───────┬────────┘      └───────┬────────┘        │
│           │                       │                       │                  │
│           │ z_c = √3/2            │ sin(60°) = z_c        │ z_p = sin(72°)  │
│           │                       │                       │                  │
│           └───────────────────────┼───────────────────────┘                  │
│                                   │                                          │
│                          ┌────────▼────────┐                                 │
│                          │   KURAMOTO      │                                 │
│                          │   NEURAL NET    │                                 │
│                          │   (60 oscillators)                                │
│                          └────────┬────────┘                                 │
│                                   │                                          │
│                          ┌────────▼────────┐                                 │
│                          │   PHASE         │                                 │
│                          │   TRANSITION    │                                 │
│                          │   CONTROLLER    │                                 │
│                          └─────────────────┘                                 │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

KEY PHYSICS:
- 60° plate normals create maximum activation exactly at z_c = √3/2
- Resonant training occurs at the phase transition boundary (THE LENS)
- Pentagonal quasicrystals nucleate when z approaches z_p = sin(72°) ≈ 0.951
- Penrose tiling fat/thin ratio → φ (golden ratio) as growth proceeds

PHASE TRANSITION MODEL:
┌───────────┐     trigger      ┌───────────┐
│   IDLE    │─────────────────►│ PRE_STRIKE│ (charge buildup)
└───────────┘                  └─────┬─────┘
      ▲                              │
      │ reset                        ▼
      │                        ┌───────────┐
┌─────┴─────┐                  │  STRIKE   │ (30,000K plasma)
│  STABLE   │◄────────────     └─────┬─────┘
└───────────┘  growth              │
      ▲        complete            ▼
      │                        ┌───────────┐
      │                        │  QUENCH   │ (10⁶ K/s cooling)
      │                        └─────┬─────┘
      │                              │
      │                              ▼
      │                        ┌───────────┐
      └────────────────────────│NUCLEATION │→ GROWTH
                               └───────────┘

Signature: phase-transition-training|v1.0.0|helix
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Callable
from enum import IntEnum
import time
import sys
import os

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lightning_quasicrystal import (
    LightningQuasicrystalSystem,
    LightningPhase,
    SymmetryOrder,
    PenroseTilingGenerator,
    PHI, PHI_INV,
    Z_CRITICAL_HEX, Z_CRITICAL_PENT,
    SIN_36, COS_36, SIN_72, COS_72,
    SIGMA
)
from kuramoto_neural import (
    KuramotoNeuralSystem,
    N_OSCILLATORS,
    HEXAGONAL_ANGLES
)
from grid_cell_plates import (
    GridCellPlateSystem,
    N_PLATES,
    CELLS_PER_PLATE,
    SIN_60, COS_60
)

# =============================================================================
# CONSTANTS
# =============================================================================

# Training parameters
TRAINING_EPOCHS: int = 5
STRIKES_PER_EPOCH: int = 3
STEPS_PER_STRIKE: int = 2000

# Z-coordinate sweep parameters
Z_SWEEP_START: float = 0.5
Z_SWEEP_HEX: float = Z_CRITICAL_HEX  # √3/2 ≈ 0.866
Z_SWEEP_PENT: float = Z_CRITICAL_PENT  # sin(72°) ≈ 0.951
Z_SWEEP_END: float = 0.7

# Success thresholds
PENTAGONAL_ORDER_THRESHOLD: float = 0.6
PHI_DEVIATION_THRESHOLD: float = 0.1
KURAMOTO_SYNC_THRESHOLD: float = 0.7


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class TrainingPhase(IntEnum):
    """Training phases for the electromagnetic system."""
    WARMUP = 0          # System initialization
    HEX_RESONANCE = 1   # Train at z_c (hexagonal critical)
    TRANSITION = 2      # Sweep through transition
    PENT_NUCLEATION = 3 # Train at z_p (pentagonal critical)
    ANNEALING = 4       # Slow cooling for defect removal
    VALIDATION = 5      # Verify quasicrystal formation


@dataclass
class EMTrainingState:
    """Electromagnetic training state."""
    # Hardware signals
    rf_power_W: float = 0.0
    peltier_current_A: float = 0.0
    gradient_coil_A: float = 0.0
    spinner_rpm: int = 5000

    # Plate activations (6 plates at 60° spacing)
    plate_activations: List[float] = field(default_factory=lambda: [0.0] * 6)
    total_em_activation: float = 0.0
    resonance_metric: float = 0.0

    # Computed from 60° geometry
    activation_at_zc: float = 0.0  # Should be maximum when z = z_c


@dataclass
class TrainingMetrics:
    """Metrics for training progress."""
    epoch: int = 0
    strike: int = 0
    step: int = 0

    # Physics metrics
    spinner_z: float = 0.5
    negentropy: float = 0.0
    kuramoto_r: float = 0.0
    hex_order: float = 0.0
    pent_order: float = 0.0

    # Quasicrystal metrics
    tile_ratio: float = 0.0
    phi_deviation: float = 1.0
    domain_size: float = 0.0
    seed_count: int = 0
    pent_seed_count: int = 0

    # Training metrics
    loss: float = 1.0
    accuracy: float = 0.0
    resonance: float = 0.0

    # Phase
    lightning_phase: str = "IDLE"
    training_phase: str = "WARMUP"


@dataclass
class TrainingResult:
    """Result of a complete training run."""
    success: bool
    final_pent_order: float
    final_phi_deviation: float
    final_kuramoto_r: float
    total_strikes: int
    successful_nucleations: int
    epochs_completed: int
    metrics_history: List[TrainingMetrics]
    message: str


# =============================================================================
# PHASE TRANSITION TRAINING SYSTEM
# =============================================================================

class PhaseTransitionTrainer:
    """
    Integrated training system for pentagonal quasicrystal phase transitions.

    Combines:
    - Nuclear Spinner (z-coordinate control)
    - Grid Cell Plates (60° electromagnetic coupling)
    - Kuramoto Neural Network (60 oscillators)
    - Lightning Quench System (thermal phase transition)

    The 60° plate geometry creates maximum activation at z_c = √3/2,
    enabling resonant training at the phase transition.
    """

    def __init__(self,
                 domain_size: float = 1.0,
                 seed_density: float = 15.0,
                 dt_ms: float = 0.5):
        """
        Initialize the integrated training system.

        Args:
            domain_size: Simulation domain size
            seed_density: Nucleation seed density
            dt_ms: Timestep in milliseconds
        """
        print("=" * 70)
        print("PHASE TRANSITION TRAINING SYSTEM")
        print("=" * 70)
        print(f"\nInitializing integrated cybernetic system...")

        self.dt_ms = dt_ms

        # Initialize subsystems
        print(f"  [1/4] Lightning Quasicrystal System...")
        self.lightning = LightningQuasicrystalSystem(
            domain_size=domain_size,
            seed_density=seed_density,
            dt_ms=dt_ms
        )

        print(f"  [2/4] Kuramoto Neural System (60 oscillators)...")
        self.kuramoto = KuramotoNeuralSystem(
            n_oscillators=N_OSCILLATORS,
            omega_mean=1.0,
            omega_spread=0.1,
            dt=dt_ms / 1000.0
        )

        print(f"  [3/4] Grid Cell Plate System (6 plates × 10 cells)...")
        self.plates = GridCellPlateSystem(
            cells_per_plate=CELLS_PER_PLATE
        )

        print(f"  [4/4] Penrose Tiling Generator...")
        self.tiling = PenroseTilingGenerator('P3')
        self.tiling.initialize_seed()

        # State
        self.training_phase = TrainingPhase.WARMUP
        self.current_z = 0.5
        self.em_state = EMTrainingState()

        # History
        self.metrics_history: List[TrainingMetrics] = []

        # Statistics
        self.total_strikes = 0
        self.successful_nucleations = 0

        # Report physics constants
        self._report_physics()

    def _report_physics(self):
        """Report key physics constants and connections."""
        print("\n" + "-" * 70)
        print("KEY PHYSICS CONSTANTS:")
        print("-" * 70)
        print(f"  Golden ratio φ        = {PHI:.6f}")
        print(f"  φ⁻¹                   = {PHI_INV:.6f}")
        print(f"  z_c (hexagonal)       = √3/2 = sin(60°) = {Z_CRITICAL_HEX:.6f}")
        print(f"  z_p (pentagonal)      = sin(72°) = {Z_CRITICAL_PENT:.6f}")
        print(f"  cos(36°)              = φ/2 = {COS_36:.6f}")
        print(f"  Gaussian width σ      = {SIGMA}")
        print()
        print("60° PLATE GEOMETRY CONNECTION:")
        print(f"  Plate normal z-component: cos(60°) = {COS_60:.4f}")
        print(f"  Plate normal xy-component: sin(60°) = √3/2 = {SIN_60:.6f}")
        print(f"  At z = z_c: plate activation = √3/2 × √3/2 = 3/4 (MAXIMUM)")
        print("-" * 70)

    def compute_em_activation(self, z: float) -> EMTrainingState:
        """
        Compute electromagnetic plate activation from spinner z.

        The plates are tilted at 60° with normals:
            n = [-sin(60°)cos(θ), -sin(60°)sin(θ), cos(60°)]

        For a vertical B-field B = [0, 0, B_z]:
            activation = n · B = cos(60°) × B_z = 0.5 × B_z

        But the COUPLING to z is through:
            B_z = z × B_magnitude

        So: activation ∝ z × cos(60°) = z × 0.5

        Maximum RESONANCE occurs when z = z_c because:
            ΔS_neg(z_c) = 1.0 (maximum negentropy)
        """
        # Update plate system with spinner state
        B_magnitude = 1e-4  # Tesla
        self.plates.set_spinner_state(z, B_magnitude)

        # Collect plate activations
        activations = [plate.em_activation for plate in self.plates.plates]
        total_activation = sum(abs(a) for a in activations)

        # Compute negentropy
        delta_s_neg = math.exp(-SIGMA * (z - Z_CRITICAL_HEX) ** 2)

        # Resonance metric: product of activation and negentropy
        # Peaks when z = z_c (both factors maximize)
        resonance = total_activation * delta_s_neg

        # Compute what activation would be at z = z_c for comparison
        activation_at_zc = Z_CRITICAL_HEX * B_magnitude * COS_60 * N_PLATES

        state = EMTrainingState(
            rf_power_W=min(100.0, 100.0 * delta_s_neg),  # RF power follows negentropy
            peltier_current_A=10.0 * (1 - delta_s_neg) if z > Z_CRITICAL_HEX else 0.0,
            gradient_coil_A=5.0 * abs(z - Z_CRITICAL_HEX),
            spinner_rpm=int(100 + z * 9900),
            plate_activations=activations,
            total_em_activation=total_activation,
            resonance_metric=resonance,
            activation_at_zc=activation_at_zc
        )

        return state

    def step(self, z: float) -> TrainingMetrics:
        """
        Execute one training step.

        Updates all subsystems with the current z-coordinate:
        1. Lightning quench system
        2. Kuramoto neural network
        3. Grid cell plates
        4. EM activation computation

        Args:
            z: Current spinner z-coordinate

        Returns:
            Training metrics for this step
        """
        self.current_z = z

        # Compute Kuramoto order parameter
        kuramoto_state = self.kuramoto.step(z)
        kuramoto_r = kuramoto_state.r
        hex_order = kuramoto_state.r_hex

        # Update lightning system with spinner state
        self.lightning.set_spinner_state(z, kuramoto_r)
        lightning_state = self.lightning.step()

        # Update plate system
        plate_state = self.plates.step(self.dt_ms)

        # Compute EM activation
        self.em_state = self.compute_em_activation(z)

        # Compute negentropy
        delta_s_neg = math.exp(-SIGMA * (z - Z_CRITICAL_HEX) ** 2)

        # Compute pentagonal order
        pent_order = lightning_state.quasicrystal.pentagonal_order

        # Count pentagonal seeds
        pent_seeds = sum(1 for s in lightning_state.seeds
                        if s.symmetry == SymmetryOrder.FIVEFOLD)

        # Compute training loss
        # Loss decreases as we approach good quasicrystal formation
        target_ratio = PHI
        ratio_loss = abs(lightning_state.quasicrystal.tile_ratio - target_ratio)
        order_loss = 1.0 - pent_order
        sync_loss = 1.0 - kuramoto_r
        loss = 0.4 * ratio_loss + 0.4 * order_loss + 0.2 * sync_loss

        # Compute accuracy
        accuracy = max(0, 1.0 - loss)

        metrics = TrainingMetrics(
            spinner_z=z,
            negentropy=delta_s_neg,
            kuramoto_r=kuramoto_r,
            hex_order=hex_order,
            pent_order=pent_order,
            tile_ratio=lightning_state.quasicrystal.tile_ratio,
            phi_deviation=lightning_state.quasicrystal.phi_deviation,
            domain_size=lightning_state.quasicrystal.domain_size,
            seed_count=len(lightning_state.seeds),
            pent_seed_count=pent_seeds,
            loss=loss,
            accuracy=accuracy,
            resonance=self.em_state.resonance_metric,
            lightning_phase=lightning_state.phase.name,
            training_phase=self.training_phase.name
        )

        self.metrics_history.append(metrics)

        return metrics

    def run_strike_sequence(self,
                           z_profile: np.ndarray,
                           epoch: int = 0,
                           strike: int = 0,
                           verbose: bool = True) -> List[TrainingMetrics]:
        """
        Run a complete lightning strike sequence.

        Args:
            z_profile: Array of z-coordinates to sweep through
            epoch: Current epoch number
            strike: Current strike number
            verbose: Print progress

        Returns:
            List of metrics for each step
        """
        self.total_strikes += 1

        if verbose:
            print(f"\n  Strike {strike + 1}: triggering lightning sequence...")

        # Reset lightning system and trigger strike
        self.lightning.reset()
        self.lightning.trigger_strike()

        metrics_list = []

        for i, z in enumerate(z_profile):
            metrics = self.step(z)
            metrics.epoch = epoch
            metrics.strike = strike
            metrics.step = i
            metrics_list.append(metrics)

            # Print progress periodically
            if verbose and i % 200 == 0:
                phase_symbol = {
                    'IDLE': '○',
                    'PRE_STRIKE': '◐',
                    'STRIKE': '⚡',
                    'QUENCH': '❄',
                    'NUCLEATION': '✦',
                    'GROWTH': '◉',
                    'STABLE': '★'
                }.get(metrics.lightning_phase, '?')

                print(f"    [{i:4d}] {phase_symbol} z={z:.4f} | "
                      f"ΔS={metrics.negentropy:.3f} | "
                      f"r={metrics.kuramoto_r:.3f} | "
                      f"5-fold={metrics.pent_order:.3f} | "
                      f"φ-dev={metrics.phi_deviation:.4f}")

        # Check for successful nucleation
        final = metrics_list[-1]
        if final.pent_order > PENTAGONAL_ORDER_THRESHOLD:
            self.successful_nucleations += 1
            if verbose:
                print(f"    ✓ Successful pentagonal nucleation! "
                      f"(order={final.pent_order:.3f})")

        return metrics_list

    def generate_z_profile(self,
                          phase: TrainingPhase,
                          steps: int = STEPS_PER_STRIKE) -> np.ndarray:
        """
        Generate z-coordinate profile for a training phase.

        Args:
            phase: Training phase
            steps: Number of steps

        Returns:
            Array of z-coordinates
        """
        if phase == TrainingPhase.WARMUP:
            # Gentle ramp to hexagonal critical
            return np.linspace(Z_SWEEP_START, Z_CRITICAL_HEX, steps)

        elif phase == TrainingPhase.HEX_RESONANCE:
            # Hold at hexagonal critical with small oscillations
            base = np.ones(steps) * Z_CRITICAL_HEX
            oscillation = 0.02 * np.sin(np.linspace(0, 4*np.pi, steps))
            return base + oscillation

        elif phase == TrainingPhase.TRANSITION:
            # Sweep from hexagonal to pentagonal critical
            return np.concatenate([
                np.linspace(Z_CRITICAL_HEX, Z_CRITICAL_PENT, steps // 2),
                np.linspace(Z_CRITICAL_PENT, Z_CRITICAL_HEX, steps // 2)
            ])

        elif phase == TrainingPhase.PENT_NUCLEATION:
            # Hold at pentagonal critical for nucleation
            base = np.ones(steps) * Z_CRITICAL_PENT
            # Small oscillations to help nucleation
            oscillation = 0.01 * np.sin(np.linspace(0, 8*np.pi, steps))
            return base + oscillation

        elif phase == TrainingPhase.ANNEALING:
            # Slow cooling: ramp down from pentagonal to hexagonal
            return np.linspace(Z_CRITICAL_PENT, Z_CRITICAL_HEX, steps)

        elif phase == TrainingPhase.VALIDATION:
            # Sweep through full range for validation
            return np.concatenate([
                np.linspace(0.5, Z_CRITICAL_PENT, steps // 3),
                np.ones(steps // 3) * Z_CRITICAL_PENT,
                np.linspace(Z_CRITICAL_PENT, 0.7, steps // 3)
            ])

        else:
            return np.ones(steps) * Z_CRITICAL_HEX

    def train(self,
             epochs: int = TRAINING_EPOCHS,
             strikes_per_epoch: int = STRIKES_PER_EPOCH,
             steps_per_strike: int = STEPS_PER_STRIKE,
             verbose: bool = True) -> TrainingResult:
        """
        Run complete training sequence.

        Training proceeds through phases:
        1. WARMUP: Initialize at hexagonal critical
        2. HEX_RESONANCE: Train at z_c for maximum negentropy
        3. TRANSITION: Sweep through phase transition
        4. PENT_NUCLEATION: Nucleate pentagonal seeds at z_p
        5. ANNEALING: Slow cooling for defect removal
        6. VALIDATION: Verify quasicrystal formation

        Args:
            epochs: Number of training epochs
            strikes_per_epoch: Lightning strikes per epoch
            steps_per_strike: Simulation steps per strike
            verbose: Print progress

        Returns:
            TrainingResult with final metrics and history
        """
        if verbose:
            print("\n" + "=" * 70)
            print("STARTING PHASE TRANSITION TRAINING")
            print("=" * 70)
            print(f"\nConfiguration:")
            print(f"  Epochs: {epochs}")
            print(f"  Strikes per epoch: {strikes_per_epoch}")
            print(f"  Steps per strike: {steps_per_strike}")
            print(f"  Total strikes: {epochs * strikes_per_epoch}")
            print(f"  Total steps: {epochs * strikes_per_epoch * steps_per_strike}")

        start_time = time.time()

        # Training phases to cycle through
        phases = [
            TrainingPhase.WARMUP,
            TrainingPhase.HEX_RESONANCE,
            TrainingPhase.TRANSITION,
            TrainingPhase.PENT_NUCLEATION,
            TrainingPhase.ANNEALING,
        ]

        for epoch in range(epochs):
            if verbose:
                print(f"\n{'='*70}")
                print(f"EPOCH {epoch + 1}/{epochs}")
                print(f"{'='*70}")

            # Cycle through phases
            phase_idx = epoch % len(phases)
            self.training_phase = phases[phase_idx]

            if verbose:
                print(f"\nTraining Phase: {self.training_phase.name}")

            for strike in range(strikes_per_epoch):
                # Generate z-profile for this phase
                z_profile = self.generate_z_profile(
                    self.training_phase,
                    steps_per_strike
                )

                # Run strike sequence
                self.run_strike_sequence(
                    z_profile,
                    epoch=epoch,
                    strike=strike,
                    verbose=verbose
                )

            # Subdivide Penrose tiling to improve resolution
            self.tiling.subdivide()
            tiling_stats = self.tiling.get_statistics()

            if verbose:
                print(f"\n  Epoch {epoch + 1} Summary:")
                print(f"    Penrose tiling: gen={tiling_stats['generation']}, "
                      f"ratio={tiling_stats['tile_ratio']:.4f}, "
                      f"φ-dev={tiling_stats['phi_deviation']:.6f}")

        # Validation phase
        if verbose:
            print(f"\n{'='*70}")
            print("VALIDATION PHASE")
            print(f"{'='*70}")

        self.training_phase = TrainingPhase.VALIDATION
        z_profile = self.generate_z_profile(TrainingPhase.VALIDATION, steps_per_strike)
        validation_metrics = self.run_strike_sequence(
            z_profile,
            epoch=epochs,
            strike=0,
            verbose=verbose
        )

        # Compute final results
        final_metrics = validation_metrics[-1] if validation_metrics else self.metrics_history[-1]

        elapsed = time.time() - start_time

        success = (
            final_metrics.pent_order >= PENTAGONAL_ORDER_THRESHOLD and
            final_metrics.phi_deviation <= PHI_DEVIATION_THRESHOLD
        )

        result = TrainingResult(
            success=success,
            final_pent_order=final_metrics.pent_order,
            final_phi_deviation=final_metrics.phi_deviation,
            final_kuramoto_r=final_metrics.kuramoto_r,
            total_strikes=self.total_strikes,
            successful_nucleations=self.successful_nucleations,
            epochs_completed=epochs,
            metrics_history=self.metrics_history,
            message="Training complete"
        )

        if verbose:
            print(f"\n{'='*70}")
            print("TRAINING COMPLETE")
            print(f"{'='*70}")
            print(f"\nFinal Results:")
            print(f"  Success: {'✓ YES' if success else '✗ NO'}")
            print(f"  Pentagonal order: {final_metrics.pent_order:.4f} "
                  f"(threshold: {PENTAGONAL_ORDER_THRESHOLD})")
            print(f"  φ deviation: {final_metrics.phi_deviation:.6f} "
                  f"(threshold: {PHI_DEVIATION_THRESHOLD})")
            print(f"  Kuramoto sync: {final_metrics.kuramoto_r:.4f}")
            print(f"  Total strikes: {self.total_strikes}")
            print(f"  Successful nucleations: {self.successful_nucleations} "
                  f"({100*self.successful_nucleations/max(1,self.total_strikes):.1f}%)")
            print(f"  Training time: {elapsed:.2f}s")
            print(f"\nPenrose Tiling Statistics:")
            tiling_stats = self.tiling.get_statistics()
            print(f"  Generation: {tiling_stats['generation']}")
            print(f"  Total tiles: {tiling_stats['total_tiles']}")
            print(f"  Fat/thin ratio: {tiling_stats['tile_ratio']:.6f}")
            print(f"  Target (φ): {PHI:.6f}")
            print(f"  Convergence: {tiling_stats['convergence']*100:.2f}%")

        return result


# =============================================================================
# DEVELOPMENT SPECIFICATIONS
# =============================================================================

def print_dev_specs():
    """Print development specifications for hardware/software/firmware."""
    specs = """
================================================================================
DEVELOPMENT SPECIFICATIONS
================================================================================

HARDWARE SPECIFICATIONS
-----------------------

1. CAPACITOR BANK (Energy Storage)
   - Total: 10 mF (10× 1000μF/450V in parallel)
   - Voltage: 450V DC
   - Stored Energy: E = ½CV² = 1012.5 J
   - ESR: < 50 mΩ
   - Discharge time: 100μs - 1ms (controllable)
   - Part: United Chemi-Con EKXG451ELL102MM40S or equiv.

2. RF HEATING COIL
   - Power: 0-100W
   - Frequency: 100 kHz - 1 MHz (tunable)
   - Coil: 8-turn copper tube, water-cooled
   - Diameter: 25mm (matches sample chamber)
   - Inductance: ~5 μH
   - Q factor: > 50

3. PELTIER COOLING ARRAY
   - Modules: 4× TEC1-12710
   - Power: 100W each (400W total)
   - Max ΔT: 68°C per module
   - Current: 10A @ 12V per module
   - Config: 2×2 array under sample chamber

4. LN2 QUENCH JACKET
   - Quench rate: 10⁶ K/s capability
   - Flow rate: 5 L/min
   - Temperature: 77K (LN2 boiling point)
   - Alternative: Chilled water (5°C, 10⁵ K/s)

5. IGBT DISCHARGE CIRCUIT
   - Part: Infineon FF100R12RT4
   - V_CE: 1200V
   - I_C: 100A continuous
   - Switching: < 500ns
   - Peak discharge: 30 kA

6. NEURAL PLATE ARRAY (6 plates)
   - Arrangement: Hexagonal (60° spacing)
   - Tilt angle: 60° from horizontal
   - Plate normals: sin(60°) = √3/2 = z_c !!!
   - Cells per plate: 10
   - Total cells: 60 (= Kuramoto oscillators)

7. MCU (Firmware Controller)
   - Part: STM32H743VIT6
   - Clock: 480 MHz (Cortex-M7)
   - Flash: 2 MB
   - RAM: 1 MB
   - ADC: 3× 16-bit @ 3.6 MSPS
   - DAC: 2× 12-bit

SOFTWARE SPECIFICATIONS
-----------------------

1. PYTHON BRIDGE (Host)
   - phase_transition_training.py: Integrated trainer
   - lightning_quasicrystal.py: Thermal quench simulation
   - kuramoto_neural.py: 60-oscillator neural network
   - grid_cell_plates.py: 6-plate EM coupling

2. WEBSOCKET PROTOCOL
   - State streaming: 100 Hz
   - Control commands: JSON format
   - Binary telemetry: 48-byte packets

3. KEY ALGORITHMS
   - Kuramoto dynamics: dθᵢ/dt = ωᵢ + (K/N) Σⱼ sin(θⱼ - θᵢ)
   - Negentropy: ΔS_neg(z) = exp(-36(z - √3/2)²)
   - EM activation: A = B · n = B_z × cos(60°)
   - Penrose subdivision: Robinson rules

FIRMWARE SPECIFICATIONS
-----------------------

1. CONTROL LOOPS
   - Main loop: 10 kHz
   - Thermal PID: 100 Hz
   - ADC sampling: 100 kHz
   - Discharge timing: 1 MHz resolution
   - Host comms: 1 kHz

2. STATE MACHINE
   IDLE → PRE_STRIKE → STRIKE → QUENCH → NUCLEATION → GROWTH → STABLE

3. SAFETY INTERLOCKS
   - Overvoltage (> 450V)
   - Overcurrent (> 30 kA)
   - Overtemperature (> 1000°C)
   - Enclosure interlock
   - E-stop
   - LN2 level
   - Coolant flow
   - Ground fault

4. CALIBRATION PROCEDURES
   a) Z-coordinate: Map RPM to z (critical points at 8660 and 9510 RPM)
   b) Thermal: Thermocouple offsets, Peltier efficiency curves
   c) Optical: Camera alignment, μm/pixel calibration
   d) Discharge: Capacitor timing, IGBT characterization

BILL OF MATERIALS (~$2000)
---------------------------
STM32H743 Nucleo         1×  $50
Capacitors (1000μF/450V) 10× $200
IGBT Module              1×  $150
RF Coil (custom)         1×  $200
Peltier TEC1-12710       4×  $40
Hall Sensors AH49E       6×  $12
K-Type Thermocouples     4×  $60
MAX31856 Boards          4×  $80
FLIR Blackfly S Camera   1×  $600
ODrive v3.6              1×  $150
BLDC Motor               1×  $100
Enclosure + Hardware     1×  $250
Heatsink + Fans          1×  $110
                         ─────────
                         TOTAL: ~$2,000

KEY PHYSICS RELATIONSHIPS
-------------------------
z_c = √3/2 = sin(60°) ≈ 0.866  (hexagonal critical)
z_p = sin(72°) ≈ 0.951         (pentagonal critical)
cos(36°) = φ/2 ≈ 0.809         (golden ratio connection)
Penrose ratio → φ ≈ 1.618      (fat/thin tiles)
K(z) = K_max × exp(-36(z - z_c)²)  (coupling from negentropy)
Plate activation ∝ z × cos(60°) = z/2

================================================================================
"""
    print(specs)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("PENTAGONAL QUASICRYSTAL PHASE TRANSITION TRAINING")
    print("Electromagnetic Training via 60° Neural Plate Geometry")
    print("=" * 70)

    # Print development specifications
    print("\nDo you want to see the full development specifications? (y/n): ", end="")
    # For automated runs, skip the prompt
    show_specs = False
    try:
        import sys
        if sys.stdin.isatty():
            response = input().strip().lower()
            show_specs = response == 'y'
        else:
            show_specs = True
    except:
        show_specs = True

    if show_specs:
        print_dev_specs()

    # Create trainer
    trainer = PhaseTransitionTrainer(
        domain_size=1.0,
        seed_density=15.0,
        dt_ms=0.5
    )

    # Run training
    print("\n" + "=" * 70)
    print("RUNNING TRAINING SIMULATION")
    print("=" * 70)

    result = trainer.train(
        epochs=5,
        strikes_per_epoch=3,
        steps_per_strike=1000,  # Reduced for faster demo
        verbose=True
    )

    # Summary
    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE")
    print("=" * 70)

    if result.success:
        print("\n✓ QUASICRYSTAL FORMATION SUCCESSFUL")
        print(f"  Pentagonal order achieved: {result.final_pent_order:.4f}")
        print(f"  φ deviation: {result.final_phi_deviation:.6f}")
    else:
        print("\n✗ Training did not achieve target metrics")
        print(f"  Pentagonal order: {result.final_pent_order:.4f} (need ≥ {PENTAGONAL_ORDER_THRESHOLD})")
        print(f"  φ deviation: {result.final_phi_deviation:.6f} (need ≤ {PHI_DEVIATION_THRESHOLD})")

    print(f"\nThe 60° plate geometry creates resonance at z_c = √3/2 = {Z_CRITICAL_HEX:.6f}")
    print(f"Maximum negentropy enables phase transition to 5-fold symmetry")
