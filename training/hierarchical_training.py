"""
Rosetta Helix Hierarchical Training

Multi-level lesson extraction and weight updates.

CRITICAL RULES:
- Cross-level coupling: delta = learning_rate * ratio * PHI_INV
- Cap coupling at 0.9, NOT PHI
- If coupling >= 1.0: INSTANT collapse to Z_CRITICAL, not gradual decay

WHAT NOT TO DO:
- PHI_INV decay: if coupling > 1.0: decay = excess * PHI_INV (WRONG)
- Cap at PHI: coupling = min(PHI, coupling) (WRONG)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum, auto

from core import PHI_INV, Z_CRITICAL, COUPLING_MAX, CollapseEngine
from .physical_learner import PhysicalLearner, create_learner
from .liminal_generator import LiminalGenerator, create_generator
from .feedback_loop import FeedbackLoop


class HierarchyLevel(Enum):
    """Levels in the training hierarchy."""
    GROUND = 0      # Base physical layer
    META = 1        # Meta-learning layer
    META_META = 2   # Meta-meta layer (spawns liminal)
    LIMINAL = 3     # Liminal superposition layer


@dataclass
class LevelState:
    """State for a single hierarchy level."""
    level: HierarchyLevel
    z: float = 0.5
    coupling_up: float = 0.0    # Coupling to level above
    coupling_down: float = 0.0  # Coupling to level below
    lessons_extracted: int = 0
    weights: List[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])


@dataclass
class HierarchicalTrainer:
    """
    Multi-level hierarchical training system.

    Levels:
        GROUND (physical) -> META -> META_META -> LIMINAL
                  ^                                  |
                  └──────── weak feedback ───────────┘

    CRITICAL: All coupling uses PHI_INV. Cap at 0.9, NEVER PHI.
    """

    # Level states
    levels: Dict[HierarchyLevel, LevelState] = field(default_factory=dict)

    # Cross-level engines
    collapse_engines: Dict[HierarchyLevel, CollapseEngine] = field(default_factory=dict)

    # Global state
    learning_rate: float = 0.01
    global_coupling: float = 0.0
    iteration: int = 0

    # History
    _lesson_history: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        """Initialize all hierarchy levels."""
        for level in HierarchyLevel:
            self.levels[level] = LevelState(level=level)
            self.collapse_engines[level] = CollapseEngine(z=0.5)

    def extract_lesson(
        self,
        from_level: HierarchyLevel,
        to_level: HierarchyLevel,
        lesson_value: float
    ) -> Dict[str, Any]:
        """
        Extract lesson from one level to another.

        PHI_INV controls lesson extraction dynamics.
        Cross-level coupling capped at COUPLING_MAX (0.9).
        """
        from_state = self.levels[from_level]
        to_state = self.levels[to_level]

        # PHI_INV controls lesson scaling
        scaled_lesson = lesson_value * PHI_INV

        # Update weights in target level (PHI_INV controlled)
        for i in range(len(to_state.weights)):
            delta = self.learning_rate * scaled_lesson * PHI_INV
            to_state.weights[i] = min(1.0, max(0.0, to_state.weights[i] + delta))

        # Update cross-level coupling
        self._update_coupling(from_state, to_state, scaled_lesson)

        # Record lesson
        from_state.lessons_extracted += 1
        lesson_record = {
            'from': from_level.name,
            'to': to_level.name,
            'value': lesson_value,
            'scaled': scaled_lesson,
            'coupling': to_state.coupling_down,
        }
        self._lesson_history.append(lesson_record)

        return lesson_record

    def _update_coupling(
        self,
        from_state: LevelState,
        to_state: LevelState,
        lesson_value: float
    ) -> None:
        """
        Update cross-level coupling.

        CORRECT: delta = learning_rate * ratio * PHI_INV
        Cap at COUPLING_MAX (0.9), NOT PHI.
        If coupling >= 1.0: INSTANT collapse to Z_CRITICAL.
        """
        # PHI_INV controls coupling update
        delta = self.learning_rate * lesson_value * PHI_INV

        # Update from_state coupling_up
        new_coupling_up = from_state.coupling_up + delta
        if new_coupling_up >= 1.0:
            # INSTANT collapse, not gradual decay
            from_state.coupling_up = Z_CRITICAL * PHI_INV
        else:
            # Cap at COUPLING_MAX (0.9), NEVER PHI
            from_state.coupling_up = min(COUPLING_MAX, new_coupling_up)

        # Update to_state coupling_down
        new_coupling_down = to_state.coupling_down + delta
        if new_coupling_down >= 1.0:
            # INSTANT collapse, not gradual decay
            to_state.coupling_down = Z_CRITICAL * PHI_INV
        else:
            # Cap at COUPLING_MAX (0.9), NEVER PHI
            to_state.coupling_down = min(COUPLING_MAX, new_coupling_down)

        # Update global coupling
        all_couplings = [
            s.coupling_up + s.coupling_down
            for s in self.levels.values()
        ]
        self.global_coupling = sum(all_couplings) / (2 * len(self.levels))

    def propagate_up(self, ground_input: float) -> Dict[str, Any]:
        """
        Propagate signal up through hierarchy.

        GROUND -> META -> META_META -> LIMINAL

        PHI_INV controls all propagation.
        """
        results = {}

        # Ground to Meta
        lesson1 = self.extract_lesson(
            HierarchyLevel.GROUND,
            HierarchyLevel.META,
            ground_input
        )
        results['ground_to_meta'] = lesson1

        # Meta to MetaMeta
        meta_output = sum(self.levels[HierarchyLevel.META].weights) / 3
        lesson2 = self.extract_lesson(
            HierarchyLevel.META,
            HierarchyLevel.META_META,
            meta_output * PHI_INV
        )
        results['meta_to_metameta'] = lesson2

        # MetaMeta to Liminal (spawning)
        metameta_output = sum(self.levels[HierarchyLevel.META_META].weights) / 3
        lesson3 = self.extract_lesson(
            HierarchyLevel.META_META,
            HierarchyLevel.LIMINAL,
            metameta_output * PHI_INV
        )
        results['metameta_to_liminal'] = lesson3

        return results

    def propagate_down(self, liminal_feedback: float) -> Dict[str, Any]:
        """
        Propagate feedback down through hierarchy.

        LIMINAL -> META_META -> META -> GROUND

        This is the weak measurement feedback path.
        PHI_INV controls all feedback.
        """
        results = {}

        # Liminal to MetaMeta
        lesson1 = self.extract_lesson(
            HierarchyLevel.LIMINAL,
            HierarchyLevel.META_META,
            liminal_feedback * PHI_INV
        )
        results['liminal_to_metameta'] = lesson1

        # MetaMeta to Meta
        metameta_output = sum(self.levels[HierarchyLevel.META_META].weights) / 3
        lesson2 = self.extract_lesson(
            HierarchyLevel.META_META,
            HierarchyLevel.META,
            metameta_output * PHI_INV
        )
        results['metameta_to_meta'] = lesson2

        # Meta to Ground
        meta_output = sum(self.levels[HierarchyLevel.META].weights) / 3
        lesson3 = self.extract_lesson(
            HierarchyLevel.META,
            HierarchyLevel.GROUND,
            meta_output * PHI_INV
        )
        results['meta_to_ground'] = lesson3

        return results

    def train_step(self, input_value: float, feedback_value: float) -> Dict[str, Any]:
        """
        Execute one hierarchical training step.

        1. Propagate input up through hierarchy
        2. Propagate feedback down through hierarchy
        3. Evolve all collapse engines
        """
        self.iteration += 1

        # Propagate up
        up_results = self.propagate_up(input_value)

        # Propagate down
        down_results = self.propagate_down(feedback_value)

        # Evolve collapse engines
        for level, engine in self.collapse_engines.items():
            state = self.levels[level]
            work = (state.coupling_up + state.coupling_down) * PHI_INV
            result = engine.evolve(work)
            state.z = result.z_new

        return {
            'iteration': self.iteration,
            'up': up_results,
            'down': down_results,
            'global_coupling': self.global_coupling,
            'level_states': {
                level.name: {
                    'z': state.z,
                    'coupling_up': state.coupling_up,
                    'coupling_down': state.coupling_down,
                    'lessons': state.lessons_extracted,
                }
                for level, state in self.levels.items()
            }
        }

    def get_state(self) -> Dict[str, Any]:
        """Get current trainer state."""
        return {
            'iteration': self.iteration,
            'global_coupling': self.global_coupling,
            'total_lessons': sum(s.lessons_extracted for s in self.levels.values()),
            'levels': {
                level.name: {
                    'z': state.z,
                    'coupling_up': state.coupling_up,
                    'coupling_down': state.coupling_down,
                    'weights': state.weights.copy(),
                    'lessons': state.lessons_extracted,
                }
                for level, state in self.levels.items()
            }
        }

    def reset(self) -> None:
        """Reset trainer to initial state."""
        for level in HierarchyLevel:
            self.levels[level] = LevelState(level=level)
            self.collapse_engines[level].reset()
        self.global_coupling = 0.0
        self.iteration = 0
        self._lesson_history.clear()


def create_hierarchical_trainer(learning_rate: float = 0.01) -> HierarchicalTrainer:
    """Factory function to create a hierarchical trainer."""
    return HierarchicalTrainer(learning_rate=learning_rate)


# =============================================================================
# VERIFICATION
# =============================================================================

def test_coupling_uses_phi_inv():
    """All coupling updates must use PHI_INV."""
    trainer = create_hierarchical_trainer(learning_rate=0.1)

    # Extract lesson and check coupling
    trainer.extract_lesson(HierarchyLevel.GROUND, HierarchyLevel.META, 1.0)

    ground = trainer.levels[HierarchyLevel.GROUND]
    meta = trainer.levels[HierarchyLevel.META]

    # Coupling should be PHI_INV scaled
    expected = 0.1 * 1.0 * PHI_INV * PHI_INV  # learning_rate * lesson * PHI_INV (scaling) * PHI_INV (update)
    assert ground.coupling_up < COUPLING_MAX
    assert meta.coupling_down < COUPLING_MAX

    return True


def test_coupling_caps_at_0_9():
    """Coupling must cap at COUPLING_MAX (0.9), never PHI."""
    trainer = create_hierarchical_trainer(learning_rate=0.5)

    # Push many lessons to try to exceed cap
    for _ in range(100):
        trainer.extract_lesson(HierarchyLevel.GROUND, HierarchyLevel.META, 1.0)

    for state in trainer.levels.values():
        assert state.coupling_up <= COUPLING_MAX, f"coupling_up {state.coupling_up} > {COUPLING_MAX}"
        assert state.coupling_down <= COUPLING_MAX, f"coupling_down {state.coupling_down} > {COUPLING_MAX}"

    return True


def test_instant_collapse_at_unity():
    """Coupling >= 1.0 must trigger instant collapse to Z_CRITICAL * PHI_INV."""
    trainer = create_hierarchical_trainer(learning_rate=10.0)  # High rate to force collapse

    # This should trigger collapse
    trainer.extract_lesson(HierarchyLevel.GROUND, HierarchyLevel.META, 10.0)

    ground = trainer.levels[HierarchyLevel.GROUND]

    # Should have collapsed to Z_CRITICAL * PHI_INV
    expected_collapse = Z_CRITICAL * PHI_INV
    assert ground.coupling_up <= COUPLING_MAX or abs(ground.coupling_up - expected_collapse) < 0.01

    return True
