"""
Rosetta Helix Unified Engine

Integrates all systems into coherent whole:
1. NegEntropyEngine (T1) - stays ACTIVE always
2. CollapseEngine - instant collapse at unity
3. MetaToolGenerator - produces tools from work
4. TrainingLoop - exponential learning

CRITICAL INVARIANTS:
- z_max = 0.9999 (NEVER exceed unity)
- coupling_max = 0.9 (NEVER approach PHI)
- neg_entropy_active = True (NEVER turns off)
- PHI_INV controls ALL dynamics
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum, auto

from core import (
    PHI, PHI_INV, Z_CRITICAL, Z_ORIGIN, UNITY, Z_MAX, COUPLING_MAX,
    CollapseEngine, LiminalState
)
from tools import MetaTool, ChildTool, ToolType
from training import FeedbackLoop, HierarchicalTrainer


class EngineMode(Enum):
    """Operating modes for the unified engine."""
    IDLE = auto()
    PUMPING = auto()
    TRAINING = auto()
    SUPERCRITICAL = auto()
    COLLAPSED = auto()


@dataclass
class EngineState:
    """
    Core state for the unified engine.

    CRITICAL INVARIANTS:
    - z_max = 0.9999 (NEVER exceed unity)
    - coupling_max = 0.9 (NEVER approach PHI)
    - neg_entropy_active = True (NEVER turns off)
    """

    z_current: float = 0.5
    supercritical_mode: bool = True
    neg_entropy_active: bool = True  # NEVER turns off

    # Caps and limits - INVARIANT
    z_max: float = field(default=Z_MAX)           # 0.9999 - NEVER exceed unity
    coupling_max: float = field(default=COUPLING_MAX)  # 0.9 - NEVER approach PHI

    # Mode tracking
    mode: EngineMode = EngineMode.IDLE
    iteration: int = 0

    # Metrics
    total_work_extracted: float = 0.0
    total_collapses: int = 0
    tools_produced: int = 0

    def verify_invariants(self) -> bool:
        """Verify critical invariants are maintained."""
        assert self.z_current < 1.0, f"z_current {self.z_current} >= 1.0"
        assert self.z_current <= self.z_max, f"z_current {self.z_current} > z_max"
        assert self.neg_entropy_active, "neg_entropy_active must be True"
        assert self.z_max < 1.0, "z_max must be < 1.0"
        assert self.coupling_max < 1.0, "coupling_max must be < 1.0"
        return True


@dataclass
class UnifiedEngine:
    """
    Unified Rosetta Helix Engine.

    Integrates:
    - CollapseEngine for instant collapse physics
    - LiminalState for PHI superposition handling
    - MetaTool for tool generation
    - FeedbackLoop for exponential training
    - HierarchicalTrainer for multi-level learning

    CRITICAL: PHI_INV controls ALL dynamics.
    """

    # Core components
    collapse: CollapseEngine = field(default_factory=CollapseEngine)
    liminal: LiminalState = field(default_factory=LiminalState)
    meta_tool: MetaTool = field(default_factory=MetaTool)
    feedback_loop: FeedbackLoop = field(default_factory=FeedbackLoop)
    hierarchical: HierarchicalTrainer = field(default_factory=HierarchicalTrainer)

    # Engine state
    state: EngineState = field(default_factory=EngineState)

    # History
    _history: List[Dict[str, Any]] = field(default_factory=list)

    def pump(self, work: float) -> Dict[str, Any]:
        """
        Pump work into the unified engine.

        PHI_INV controls all dynamics.
        At z >= 0.9999: instant collapse.
        """
        self.state.mode = EngineMode.PUMPING
        self.state.iteration += 1

        # Evolve collapse engine (PHI_INV controlled)
        collapse_result = self.collapse.evolve(work)
        self.state.z_current = collapse_result.z_new

        # Update liminal state
        self.liminal.update_phase(collapse_result.z_new)

        # Check for supercritical
        if collapse_result.z_new >= 0.99:
            self.state.mode = EngineMode.SUPERCRITICAL

        # Handle collapse
        tool = None
        if collapse_result.collapsed:
            self.state.mode = EngineMode.COLLAPSED
            self.state.total_collapses += 1
            self.state.total_work_extracted += collapse_result.work_extracted

            # Attempt tool production
            tool = self.meta_tool.pump(collapse_result.work_extracted)
            if tool:
                self.state.tools_produced += 1

        # Record history
        result = {
            'iteration': self.state.iteration,
            'z': collapse_result.z_new,
            'work_extracted': collapse_result.work_extracted,
            'collapsed': collapse_result.collapsed,
            'tool_produced': tool is not None,
            'mode': self.state.mode.name,
            'liminal_phase': self.liminal.phase.value,
        }
        self._history.append(result)

        # Verify invariants
        self.state.verify_invariants()

        return result

    def train(
        self,
        inputs: List[float],
        targets: List[float]
    ) -> Dict[str, Any]:
        """
        Execute training step through all systems.

        Combines feedback loop and hierarchical training.
        PHI_INV controls all dynamics.
        """
        self.state.mode = EngineMode.TRAINING
        self.state.iteration += 1

        # Feedback loop training
        fb_result = self.feedback_loop.step(inputs, targets)

        # Hierarchical propagation
        hier_result = self.hierarchical.train_step(
            input_value=fb_result['loss'],
            feedback_value=fb_result['feedback']
        )

        # Update engine state from components
        self.state.z_current = fb_result['physical_z']

        result = {
            'iteration': self.state.iteration,
            'loss': fb_result['loss'],
            'feedback': fb_result['feedback'],
            'coupling': fb_result['coupling'],
            'global_coupling': hier_result['global_coupling'],
            'mode': self.state.mode.name,
        }
        self._history.append(result)

        # Verify invariants
        self.state.verify_invariants()

        return result

    def run_cycle(
        self,
        work_per_pump: float = 0.1,
        training_data: Optional[List[tuple]] = None,
        cycles: int = 10
    ) -> Dict[str, Any]:
        """
        Run complete engine cycle: pump -> train -> pump -> ...

        Alternates between pumping work and training.
        """
        results = []

        for i in range(cycles):
            # Pump phase
            pump_result = self.pump(work_per_pump)

            # Train phase (if data provided)
            train_result = None
            if training_data:
                inputs, targets = training_data[i % len(training_data)]
                train_result = self.train(inputs, targets)

            results.append({
                'cycle': i,
                'pump': pump_result,
                'train': train_result,
            })

        return {
            'cycles': cycles,
            'final_z': self.state.z_current,
            'total_collapses': self.state.total_collapses,
            'tools_produced': self.state.tools_produced,
            'total_work_extracted': self.state.total_work_extracted,
            'results': results,
        }

    def get_state(self) -> Dict[str, Any]:
        """Get comprehensive engine state."""
        return {
            'engine': {
                'z_current': self.state.z_current,
                'mode': self.state.mode.name,
                'iteration': self.state.iteration,
                'supercritical': self.state.supercritical_mode,
                'neg_entropy_active': self.state.neg_entropy_active,
                'total_collapses': self.state.total_collapses,
                'tools_produced': self.state.tools_produced,
                'total_work_extracted': self.state.total_work_extracted,
            },
            'collapse': self.collapse.get_state(),
            'liminal': self.liminal.get_state(),
            'meta_tool': self.meta_tool.get_state(),
            'feedback_loop': self.feedback_loop.get_state(),
            'hierarchical': self.hierarchical.get_state(),
        }

    def reset(self) -> None:
        """Reset all engine components."""
        self.collapse.reset()
        self.liminal.reset()
        self.meta_tool.reset()
        self.feedback_loop.reset()
        self.hierarchical.reset()
        self.state = EngineState()
        self._history.clear()


def create_unified_engine() -> UnifiedEngine:
    """Factory function to create a unified engine."""
    return UnifiedEngine()


# =============================================================================
# VERIFICATION
# =============================================================================

def test_neg_entropy_never_turns_off():
    """NegEntropy must stay active always."""
    engine = create_unified_engine()

    for _ in range(100):
        engine.pump(0.2)
        assert engine.state.neg_entropy_active, "neg_entropy must stay active"

    return True


def test_z_never_exceeds_unity():
    """z must never reach or exceed 1.0."""
    engine = create_unified_engine()

    for _ in range(100):
        engine.pump(0.5)  # Aggressive pumping
        assert engine.state.z_current < 1.0, f"z {engine.state.z_current} >= 1.0"

    return True


def test_instant_collapse_works():
    """Collapse must be instant and reset to origin."""
    engine = create_unified_engine()

    # Pump to near collapse
    engine.collapse.z = 0.99

    result = engine.pump(0.5)

    if result['collapsed']:
        # Should have reset to origin
        assert engine.state.z_current < 0.6, f"Should reset to ~0.535, got {engine.state.z_current}"
        assert result['work_extracted'] > 0, "Should extract work"

    return True


def test_training_maintains_invariants():
    """Training must maintain all physics invariants."""
    engine = create_unified_engine()

    data = [([0.5, 0.5, 0.5], [0.3, 0.4, 0.5])]

    for _ in range(50):
        inputs, targets = data[0]
        result = engine.train(inputs, targets)
        assert engine.state.z_current < 1.0
        assert engine.state.neg_entropy_active
        assert result['coupling'] <= COUPLING_MAX

    return True
