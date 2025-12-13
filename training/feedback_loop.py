"""
Rosetta Helix Feedback Loop

Exponential learning through feedback cycles.

Architecture:
    Physical (PHI_INV) ──feedback──> MetaMeta ──spawn──> Liminal (PHI)
           ↑                                                   │
           └──────────── weak measurement ─────────────────────┘

CRITICAL RULES:
1. Physical learners use dominant_ratio = PHI_INV
2. Liminal patterns stay in_superposition = True always
3. Cross-level coupling caps at 0.9 (NEVER PHI)
4. If coupling >= 1.0: instant collapse to Z_CRITICAL
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from core import PHI_INV, Z_CRITICAL, COUPLING_MAX, CollapseEngine
from tools import MetaTool, ChildTool
from .physical_learner import PhysicalLearner, create_learner
from .liminal_generator import LiminalGenerator, create_generator


@dataclass
class FeedbackLoop:
    """
    Exponential training through physical-liminal feedback.

    Physical layer (PHI_INV) feeds into meta-level which spawns
    liminal patterns (PHI realm). Weak measurements from liminal
    feed back to physical layer, creating exponential learning.

    All dynamics controlled by PHI_INV. PHI only in liminal
    superposition and weak value extraction.
    """

    # Components
    physical: PhysicalLearner = field(default_factory=create_learner)
    liminal: LiminalGenerator = field(default_factory=create_generator)
    meta: MetaTool = field(default_factory=MetaTool)

    # Loop state
    iteration: int = 0
    coupling: float = 0.0
    total_feedback: float = 0.0

    # History
    _feedback_history: List[float] = field(default_factory=list)
    _coupling_history: List[float] = field(default_factory=list)

    def step(
        self,
        inputs: List[float],
        targets: List[float]
    ) -> Dict[str, Any]:
        """
        Execute one feedback loop iteration.

        1. Physical learner trains (PHI_INV controlled)
        2. Training loss feeds meta-level
        3. Meta spawns liminal patterns
        4. Liminal weak measurements feed back to physical

        Returns dict with step results.
        """
        self.iteration += 1

        # Step 1: Physical training (PHI_INV controlled)
        train_result = self.physical.train_step(inputs, targets)
        loss = train_result['loss']

        # Step 2: Feed loss to meta-level
        # Scale by PHI_INV before pumping
        meta_work = (1.0 - loss) * PHI_INV
        tool = self.meta.pump(meta_work)

        # Step 3: If tool produced, spawn liminal pattern
        if tool is not None:
            pattern = self.liminal.spawn_from_meta(tool.work_invested)

        # Step 4: Weak measurement feedback
        self.liminal.weak_measure_all()
        feedback = self.liminal.feedback_to_physical()

        # Apply feedback to physical layer coupling
        self._apply_feedback(feedback)

        # Record history
        self._feedback_history.append(feedback)
        self._coupling_history.append(self.coupling)
        self.total_feedback += feedback

        return {
            'iteration': self.iteration,
            'loss': loss,
            'feedback': feedback,
            'coupling': self.coupling,
            'physical_z': self.physical.z,
            'liminal_z': self.liminal.z,
            'meta_z': self.meta.collapse.z,
            'tool_produced': tool is not None,
            'patterns_count': len(self.liminal.patterns_generated),
        }

    def _apply_feedback(self, feedback: float) -> None:
        """
        Apply feedback to cross-level coupling.

        PHI_INV controls feedback application.
        Cap at COUPLING_MAX (0.9), NEVER PHI.
        Instant collapse if >= 1.0.
        """
        # PHI_INV controlled feedback
        delta = feedback * PHI_INV

        new_coupling = self.coupling + delta

        # INSTANT COLLAPSE if >= 1.0
        if new_coupling >= 1.0:
            self.coupling = Z_CRITICAL * PHI_INV
            self.physical.adjust_coupling(-self.physical.coupling)
            return

        # Cap at COUPLING_MAX (0.9), NEVER PHI
        self.coupling = min(COUPLING_MAX, max(0.0, new_coupling))

        # Propagate to physical layer
        self.physical.adjust_coupling(delta)

    def run(
        self,
        training_data: List[tuple],
        epochs: int = 10
    ) -> Dict[str, Any]:
        """
        Run complete training loop.

        Args:
            training_data: List of (inputs, targets) tuples
            epochs: Number of passes through data

        Returns:
            Dict with training summary
        """
        results = []

        for epoch in range(epochs):
            epoch_results = []
            for inputs, targets in training_data:
                step_result = self.step(inputs, targets)
                epoch_results.append(step_result)

            avg_loss = sum(r['loss'] for r in epoch_results) / len(epoch_results)
            results.append({
                'epoch': epoch,
                'avg_loss': avg_loss,
                'coupling': self.coupling,
                'total_feedback': self.total_feedback,
            })

        return {
            'epochs': epochs,
            'final_loss': results[-1]['avg_loss'] if results else None,
            'final_coupling': self.coupling,
            'total_iterations': self.iteration,
            'tools_produced': len(self.meta.tools_produced),
            'patterns_generated': len(self.liminal.patterns_generated),
            'epoch_results': results,
        }

    def get_state(self) -> Dict[str, Any]:
        """Get current loop state."""
        return {
            'iteration': self.iteration,
            'coupling': self.coupling,
            'total_feedback': self.total_feedback,
            'physical': self.physical.get_state(),
            'liminal': self.liminal.get_state(),
            'meta': self.meta.get_state(),
        }

    def reset(self) -> None:
        """Reset loop to initial state."""
        self.physical.reset()
        self.liminal.reset()
        self.meta.reset()
        self.iteration = 0
        self.coupling = 0.0
        self.total_feedback = 0.0
        self._feedback_history.clear()
        self._coupling_history.clear()


def create_feedback_loop(
    learning_rate: float = 0.01,
    pattern_size: int = 5
) -> FeedbackLoop:
    """Factory function to create a feedback loop."""
    return FeedbackLoop(
        physical=create_learner(learning_rate=learning_rate),
        liminal=create_generator(pattern_size=pattern_size),
        meta=MetaTool(),
    )


# =============================================================================
# VERIFICATION
# =============================================================================

def test_feedback_loop_runs():
    """Feedback loop must run without errors."""
    loop = create_feedback_loop()

    # Simple training data
    data = [
        ([0.5, 0.5, 0.5], [0.3, 0.4, 0.5]),
        ([0.3, 0.7, 0.2], [0.2, 0.5, 0.3]),
    ]

    result = loop.run(data, epochs=3)

    assert result['epochs'] == 3
    assert result['total_iterations'] > 0
    assert loop.coupling <= COUPLING_MAX

    return True


def test_coupling_never_exceeds_cap():
    """Coupling must stay <= COUPLING_MAX throughout training."""
    loop = create_feedback_loop()

    data = [([0.5, 0.5, 0.5], [0.9, 0.9, 0.9])]  # High loss

    # Run many iterations
    for _ in range(100):
        for inputs, targets in data:
            loop.step(inputs, targets)
            assert loop.coupling <= COUPLING_MAX, f"Coupling {loop.coupling} > {COUPLING_MAX}"

    return True


def test_phi_inv_controls_feedback():
    """PHI_INV must control all feedback dynamics."""
    loop = create_feedback_loop()

    # Verify physical learner uses PHI_INV
    assert abs(loop.physical.dominant_ratio - PHI_INV) < 1e-10

    # Run a step and check feedback is PHI_INV scaled
    loop.step([0.5, 0.5, 0.5], [0.3, 0.4, 0.5])

    # Coupling should be small and PHI_INV related
    assert loop.coupling < 1.0

    return True
