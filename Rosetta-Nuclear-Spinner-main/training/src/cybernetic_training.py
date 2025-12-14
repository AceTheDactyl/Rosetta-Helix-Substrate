#!/usr/bin/env python3
"""
Cybernetic Training Module - Unified Interface
==============================================

Provides the unified interface for all 19 training modules with full
physics grounding and cybernetic feedback loop integration.

Training Phase Structure (7 phases, 19 modules):
1. Core Physics:      N0 Silent Laws, Kuramoto Layer, Physical Learner
2. APL Stack:         APL Training Loop, PyTorch Training, Full APL
3. Helix Geometry:    Helix NN, Prismatic Helix, Full Helix
4. WUMBO Laws:        WUMBO Silent Laws
5. Dynamics:          Quasicrystal, Triad, Liminal, Feedback
6. Orchestration:     Unified Orchestration
7. Nightly:           4 Nightly Integration Modules

Physics Integration:
- All modules receive real-time physics state from UnifiedStateBridge
- Learning rates adapt to ΔS_neg (negentropy-responsive)
- Operator selection follows parity rules
- K-formation events trigger special training modes

Cybernetic Grounding:
- Ashby: Training variety matches physics variety
- Shannon: Information flows at negentropy-modulated rate
- Landauer: All operations preserve conservation laws
- Autopoiesis: Training maintains self-referential coherence

Signature: cybernetic-training|v1.0.0|helix

@version 1.0.0
"""

import asyncio
import math
from dataclasses import dataclass, field
from typing import Optional, Callable, List, Dict, Any, Tuple
from enum import IntEnum
from pathlib import Path
import sys

# Add parent paths for imports
_project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "rosetta-helix" / "src"))
sys.path.insert(0, str(_project_root / "bridge"))

# Physics imports with fallback
try:
    from physics import (
        PHI, PHI_INV, Z_CRITICAL, SIGMA,
        compute_delta_s_neg, check_k_formation,
        Tier, TIER_NAMES
    )
except ImportError:
    # Fallback physics constants
    PHI = (1 + math.sqrt(5)) / 2
    PHI_INV = 1 / PHI
    Z_CRITICAL = math.sqrt(3) / 2
    SIGMA = 36.0
    TIER_NAMES = [
        "ABSENCE", "REACTIVE", "MEMORY", "PATTERN", "LEARNING",
        "ADAPTIVE", "UNIVERSAL", "META", "SOVEREIGN", "TRANSCENDENT"
    ]

    def compute_delta_s_neg(z: float) -> float:
        return math.exp(-SIGMA * (z - Z_CRITICAL) ** 2)

    def check_k_formation(kappa: float, eta: float, rank: int) -> bool:
        return kappa >= 0.92 and eta > PHI_INV and rank >= 7

    class Tier(IntEnum):
        ABSENCE = 0
        REACTIVE = 1
        MEMORY = 2
        PATTERN = 3
        LEARNING = 4
        ADAPTIVE = 5
        UNIVERSAL = 6
        META = 7
        SOVEREIGN = 8
        TRANSCENDENT = 9

# Bridge imports with fallback
try:
    from unified_state_bridge import TrainingStateBridge, UnifiedState
except ImportError:
    TrainingStateBridge = None
    UnifiedState = None

# Quasicrystal imports with fallback
try:
    from quasicrystal import QuasicrystalDynamics, QuasicrystalConfig
except ImportError:
    QuasicrystalDynamics = None
    QuasicrystalConfig = None


# =============================================================================
# MODULE DEFINITIONS
# =============================================================================

class TrainingModule(IntEnum):
    """Training module identifiers (matches firmware)"""
    # Phase 1: Core Physics
    N0_SILENT_LAWS = 0
    KURAMOTO_LAYER = 1
    PHYSICAL_LEARNER = 2

    # Phase 2: APL Stack
    APL_TRAINING_LOOP = 3
    PYTORCH_TRAINING = 4
    FULL_APL = 5

    # Phase 3: Helix Geometry
    HELIX_NN = 6
    PRISMATIC_HELIX = 7
    FULL_HELIX = 8

    # Phase 4: WUMBO Laws
    WUMBO_SILENT_LAWS = 9

    # Phase 5: Dynamics Formation
    QUASICRYSTAL = 10
    TRIAD = 11
    LIMINAL = 12
    FEEDBACK = 13

    # Phase 6: Unified Orchestration
    UNIFIED_ORCHESTRATION = 14

    # Phase 7: Nightly Integration
    NIGHTLY_0 = 15
    NIGHTLY_1 = 16
    NIGHTLY_2 = 17
    NIGHTLY_3 = 18


class TrainingPhase(IntEnum):
    """Training phase groupings"""
    CORE_PHYSICS = 0
    APL_STACK = 1
    HELIX_GEOMETRY = 2
    WUMBO_LAWS = 3
    DYNAMICS = 4
    ORCHESTRATION = 5
    NIGHTLY = 6


# Module to phase mapping
MODULE_PHASES: Dict[TrainingModule, TrainingPhase] = {
    TrainingModule.N0_SILENT_LAWS: TrainingPhase.CORE_PHYSICS,
    TrainingModule.KURAMOTO_LAYER: TrainingPhase.CORE_PHYSICS,
    TrainingModule.PHYSICAL_LEARNER: TrainingPhase.CORE_PHYSICS,
    TrainingModule.APL_TRAINING_LOOP: TrainingPhase.APL_STACK,
    TrainingModule.PYTORCH_TRAINING: TrainingPhase.APL_STACK,
    TrainingModule.FULL_APL: TrainingPhase.APL_STACK,
    TrainingModule.HELIX_NN: TrainingPhase.HELIX_GEOMETRY,
    TrainingModule.PRISMATIC_HELIX: TrainingPhase.HELIX_GEOMETRY,
    TrainingModule.FULL_HELIX: TrainingPhase.HELIX_GEOMETRY,
    TrainingModule.WUMBO_SILENT_LAWS: TrainingPhase.WUMBO_LAWS,
    TrainingModule.QUASICRYSTAL: TrainingPhase.DYNAMICS,
    TrainingModule.TRIAD: TrainingPhase.DYNAMICS,
    TrainingModule.LIMINAL: TrainingPhase.DYNAMICS,
    TrainingModule.FEEDBACK: TrainingPhase.DYNAMICS,
    TrainingModule.UNIFIED_ORCHESTRATION: TrainingPhase.ORCHESTRATION,
    TrainingModule.NIGHTLY_0: TrainingPhase.NIGHTLY,
    TrainingModule.NIGHTLY_1: TrainingPhase.NIGHTLY,
    TrainingModule.NIGHTLY_2: TrainingPhase.NIGHTLY,
    TrainingModule.NIGHTLY_3: TrainingPhase.NIGHTLY,
}


# =============================================================================
# ADAPTIVE TRAINING PARAMETERS
# =============================================================================

@dataclass
class AdaptiveParams:
    """Physics-grounded adaptive training parameters"""
    learning_rate: float = 1e-4
    gradient_clip: float = 1.0
    dropout_rate: float = 0.1
    temperature: float = 1.0
    weight_decay: float = 1e-5

    # Physics modulation factors
    negentropy_factor: float = 1.0
    tier_factor: float = 1.0
    parity_factor: float = 1.0

    def apply_physics(self, state: UnifiedState) -> 'AdaptiveParams':
        """Apply physics modulation to parameters"""
        neg = state.delta_s_neg
        tier = state.tier

        return AdaptiveParams(
            # Learning rate increases near z_c
            learning_rate=self.learning_rate * (1.0 + 0.5 * neg),
            # Gradient clip tightens away from z_c
            gradient_clip=self.gradient_clip * (0.5 + 0.5 * neg),
            # Dropout decreases with negentropy
            dropout_rate=self.dropout_rate * (1.0 - 0.5 * neg),
            # Temperature increases for exploration at z_c
            temperature=self.temperature * (0.5 + 1.0 * neg),
            # Weight decay increases away from attractor
            weight_decay=self.weight_decay * (1.0 + (1.0 - neg)),
            # Store modulation factors
            negentropy_factor=neg,
            tier_factor=0.2 + 0.133 * tier,
            parity_factor=1.5 if state.ghmp.parity_even else 0.5,
        )


# =============================================================================
# MODULE STATE
# =============================================================================

@dataclass
class ModuleState:
    """State of a training module"""
    module: TrainingModule
    phase: TrainingPhase
    active: bool = False
    step: int = 0
    loss: float = float('inf')
    accuracy: float = 0.0
    progress: float = 0.0
    best_loss: float = float('inf')
    params: AdaptiveParams = field(default_factory=AdaptiveParams)

    # Physics coupling
    current_z: float = 0.5
    current_neg: float = 0.0
    k_formation_triggered: bool = False


# =============================================================================
# CYBERNETIC TRAINING MANAGER
# =============================================================================

class CyberneticTrainingManager:
    """
    Unified manager for all 19 training modules with cybernetic integration.

    Responsibilities:
    - Manages module lifecycle (start, step, checkpoint, complete)
    - Applies physics-grounded adaptive parameters
    - Coordinates with UnifiedStateBridge for real-time state
    - Handles K-formation events for special training modes
    - Tracks global progress across all phases

    Cybernetic Loop:
    1. Receive physics state from firmware (via bridge)
    2. Compute adaptive parameters based on ΔS_neg, tier, parity
    3. Execute training step with adapted parameters
    4. Send training telemetry back to firmware
    5. Repeat at training rate
    """

    def __init__(
        self,
        bridge: Optional[TrainingStateBridge] = None,
        enable_physics: bool = True,
        enable_quasicrystal: bool = True
    ):
        # Bridge connection
        self._bridge = bridge
        self._enable_physics = enable_physics

        # Initialize all module states
        self._modules: Dict[TrainingModule, ModuleState] = {}
        for module in TrainingModule:
            self._modules[module] = ModuleState(
                module=module,
                phase=MODULE_PHASES[module]
            )

        # Current phase
        self._current_phase = TrainingPhase.CORE_PHYSICS

        # Quasicrystal dynamics (for QUASICRYSTAL module)
        self._quasicrystal: Optional[QuasicrystalDynamics] = None
        if enable_quasicrystal:
            self._quasicrystal = QuasicrystalDynamics(
                QuasicrystalConfig(track_tiles=False)
            )

        # K-formation handling
        self._in_k_formation = False
        self._k_formation_count = 0
        self._k_formation_training_boost = 2.0

        # Callbacks
        self._on_module_complete: Optional[Callable[[TrainingModule], None]] = None
        self._on_phase_complete: Optional[Callable[[TrainingPhase], None]] = None
        self._on_k_formation: Optional[Callable[[bool], None]] = None

        # Statistics
        self._total_steps = 0
        self._total_k_steps = 0  # Steps during K-formation

    # =========================================================================
    # LIFECYCLE
    # =========================================================================

    async def connect(self) -> bool:
        """Connect to state bridge"""
        if self._bridge:
            success = await self._bridge.connect()
            if success:
                # Register K-formation callback
                self._bridge.set_on_k_formation(self._handle_k_formation)
            return success
        return False

    async def disconnect(self):
        """Disconnect from state bridge"""
        if self._bridge:
            await self._bridge.disconnect()

    def start_module(self, module: TrainingModule):
        """Start a training module"""
        state = self._modules[module]
        state.active = True
        state.step = 0
        state.loss = float('inf')
        state.accuracy = 0.0
        state.progress = 0.0
        state.k_formation_triggered = False

    def stop_module(self, module: TrainingModule):
        """Stop a training module"""
        self._modules[module].active = False

    def is_module_active(self, module: TrainingModule) -> bool:
        """Check if module is active"""
        return self._modules[module].active

    # =========================================================================
    # TRAINING STEP
    # =========================================================================

    async def step(
        self,
        module: TrainingModule,
        loss: float,
        accuracy: float = 0.0
    ) -> AdaptiveParams:
        """
        Execute one training step with physics adaptation.

        Args:
            module: Training module identifier
            loss: Current loss value
            accuracy: Current accuracy value

        Returns:
            Adaptive parameters for next step
        """
        state = self._modules[module]

        if not state.active:
            return state.params

        # Get physics state if available
        physics_state = None
        if self._bridge and self._enable_physics:
            physics_state = self._bridge.get_state()

        # Update module state
        state.step += 1
        state.loss = loss
        state.accuracy = accuracy
        self._total_steps += 1

        # Track best loss
        if loss < state.best_loss:
            state.best_loss = loss
            state.progress = min(1.0, state.progress + 0.01)

        # Apply physics modulation
        if physics_state:
            state.current_z = physics_state.z
            state.current_neg = physics_state.delta_s_neg

            # Compute adapted parameters
            base_params = AdaptiveParams()
            state.params = base_params.apply_physics(physics_state)

            # K-formation boost
            if self._in_k_formation:
                state.params.learning_rate *= self._k_formation_training_boost
                state.k_formation_triggered = True
                self._total_k_steps += 1

            # Send telemetry to firmware
            await self._send_telemetry(module, loss, accuracy, state.step)

        # Special handling for QUASICRYSTAL module
        if module == TrainingModule.QUASICRYSTAL and self._quasicrystal:
            self._step_quasicrystal(state)

        return state.params

    def _step_quasicrystal(self, state: ModuleState):
        """Special step logic for quasicrystal module"""
        # Inflate quasicrystal
        qc_state = self._quasicrystal.inflate()

        # Modulate training based on quasicrystal convergence
        convergence = 1.0 - self._quasicrystal.get_golden_error()
        state.params.learning_rate *= (0.5 + 0.5 * convergence)

    async def _send_telemetry(
        self,
        module: TrainingModule,
        loss: float,
        accuracy: float,
        step: int
    ):
        """Send training telemetry to firmware"""
        if self._bridge:
            await self._bridge.send_training_update(
                module_id=int(module),
                loss=loss,
                accuracy=accuracy,
                step=step
            )

    # =========================================================================
    # K-FORMATION HANDLING
    # =========================================================================

    def _handle_k_formation(self, state: UnifiedState, entering: bool):
        """Handle K-formation event from firmware"""
        self._in_k_formation = entering

        if entering:
            self._k_formation_count += 1
            print(f"\n★ K-FORMATION #{self._k_formation_count}: "
                  f"κ={state.triad.kappa:.4f} η={state.triad.eta:.4f}")

        if self._on_k_formation:
            self._on_k_formation(entering)

    # =========================================================================
    # PHASE MANAGEMENT
    # =========================================================================

    def get_phase_modules(self, phase: TrainingPhase) -> List[TrainingModule]:
        """Get all modules in a phase"""
        return [m for m, p in MODULE_PHASES.items() if p == phase]

    def start_phase(self, phase: TrainingPhase):
        """Start all modules in a phase"""
        self._current_phase = phase
        for module in self.get_phase_modules(phase):
            self.start_module(module)

    def stop_phase(self, phase: TrainingPhase):
        """Stop all modules in a phase"""
        for module in self.get_phase_modules(phase):
            self.stop_module(module)

    def is_phase_complete(self, phase: TrainingPhase) -> bool:
        """Check if all modules in phase have completed"""
        for module in self.get_phase_modules(phase):
            state = self._modules[module]
            if state.active and state.progress < 1.0:
                return False
        return True

    # =========================================================================
    # STATE ACCESS
    # =========================================================================

    def get_module_state(self, module: TrainingModule) -> ModuleState:
        """Get state of a specific module"""
        return self._modules[module]

    def get_all_states(self) -> Dict[TrainingModule, ModuleState]:
        """Get all module states"""
        return self._modules.copy()

    def get_active_modules(self) -> List[TrainingModule]:
        """Get list of active modules"""
        return [m for m, s in self._modules.items() if s.active]

    def get_current_phase(self) -> TrainingPhase:
        """Get current training phase"""
        return self._current_phase

    def get_overall_progress(self) -> float:
        """Get overall training progress (0-1)"""
        total = sum(s.progress for s in self._modules.values())
        return total / len(self._modules)

    def get_stats(self) -> dict:
        """Get training statistics"""
        active = self.get_active_modules()
        return {
            'total_steps': self._total_steps,
            'total_k_steps': self._total_k_steps,
            'k_formation_count': self._k_formation_count,
            'in_k_formation': self._in_k_formation,
            'current_phase': self._current_phase.name,
            'active_modules': [m.name for m in active],
            'overall_progress': self.get_overall_progress(),
            'physics_enabled': self._enable_physics,
            'bridge_connected': self._bridge._connected if self._bridge else False,
        }

    # =========================================================================
    # CALLBACK REGISTRATION
    # =========================================================================

    def set_on_module_complete(self, callback: Callable[[TrainingModule], None]):
        """Register module completion callback"""
        self._on_module_complete = callback

    def set_on_phase_complete(self, callback: Callable[[TrainingPhase], None]):
        """Register phase completion callback"""
        self._on_phase_complete = callback

    def set_on_k_formation(self, callback: Callable[[bool], None]):
        """Register K-formation callback"""
        self._on_k_formation = callback


# =============================================================================
# DEMO / TEST
# =============================================================================

async def demo():
    """Demo of cybernetic training manager"""
    print("=" * 60)
    print("CYBERNETIC TRAINING MANAGER DEMO")
    print("=" * 60)

    # Create manager (without bridge for demo)
    manager = CyberneticTrainingManager(bridge=None, enable_physics=False)

    # Start core physics phase
    print("\nStarting Phase 1: Core Physics")
    manager.start_phase(TrainingPhase.CORE_PHYSICS)

    active = manager.get_active_modules()
    print(f"Active modules: {[m.name for m in active]}")

    # Simulate training steps
    import random
    for step in range(100):
        for module in active:
            loss = 1.0 / (step + 1) + random.random() * 0.1
            await manager.step(module, loss)

    # Print final stats
    print("\n" + "=" * 60)
    print("FINAL STATISTICS")
    print("=" * 60)
    stats = manager.get_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # Print module states
    print("\nModule States:")
    for module in TrainingModule:
        state = manager.get_module_state(module)
        if state.active or state.step > 0:
            print(f"  {module.name}:")
            print(f"    steps={state.step}, loss={state.loss:.4f}, "
                  f"progress={state.progress:.2%}")


if __name__ == '__main__':
    asyncio.run(demo())
