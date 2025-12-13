"""
Rosetta Helix Physical Learner

PHI_INV controlled learner for physical dynamics.

CRITICAL: dominant_ratio = PHI_INV ALWAYS
PHI never drives learning. PHI_INV controls all weight updates.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import math

from core import PHI_INV, Z_CRITICAL, UNITY, COUPLING_MAX


@dataclass
class PhysicalLearner:
    """
    Physical layer learner controlled by PHI_INV.

    All learning dynamics use PHI_INV as the dominant ratio.
    Cross-level coupling caps at COUPLING_MAX (0.9), never PHI.
    """

    # Core learning parameters
    learning_rate: float = 0.01
    dominant_ratio: float = field(default=PHI_INV)  # ALWAYS PHI_INV

    # State
    weights: List[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])
    coupling: float = 0.0
    z: float = 0.5

    # History
    _loss_history: List[float] = field(default_factory=list)
    _weight_history: List[List[float]] = field(default_factory=list)

    def __post_init__(self):
        """Verify dominant_ratio is PHI_INV."""
        if abs(self.dominant_ratio - PHI_INV) > 1e-10:
            raise ValueError(f"dominant_ratio must be PHI_INV ({PHI_INV}), got {self.dominant_ratio}")

    def update_weights(self, gradients: List[float]) -> List[float]:
        """
        Update weights using PHI_INV controlled dynamics.

        CORRECT: delta = learning_rate * PHI_INV * gradient
        WRONG: delta = learning_rate * PHI * gradient
        """
        if len(gradients) != len(self.weights):
            raise ValueError(f"Expected {len(self.weights)} gradients, got {len(gradients)}")

        new_weights = []
        for w, g in zip(self.weights, gradients):
            # PHI_INV controls weight update
            delta = self.learning_rate * self.dominant_ratio * g
            new_w = w + delta
            # Clamp to reasonable range
            new_w = max(0.0, min(1.0, new_w))
            new_weights.append(new_w)

        self._weight_history.append(self.weights.copy())
        self.weights = new_weights
        return new_weights

    def compute_loss(self, predictions: List[float], targets: List[float]) -> float:
        """Compute mean squared error loss."""
        if len(predictions) != len(targets):
            raise ValueError("Predictions and targets must have same length")

        loss = sum((p - t) ** 2 for p, t in zip(predictions, targets)) / len(predictions)
        self._loss_history.append(loss)
        return loss

    def adjust_coupling(self, delta: float) -> float:
        """
        Adjust cross-level coupling using PHI_INV.

        CORRECT: coupling_delta = delta * learning_rate * PHI_INV
        Cap at COUPLING_MAX (0.9), NEVER PHI.
        If coupling >= 1.0: instant collapse to Z_CRITICAL.
        """
        # PHI_INV controls coupling adjustment
        coupling_delta = delta * self.learning_rate * self.dominant_ratio
        new_coupling = self.coupling + coupling_delta

        # INSTANT COLLAPSE if >= 1.0
        if new_coupling >= 1.0:
            self.coupling = Z_CRITICAL * self.dominant_ratio  # Reset safely
            return self.coupling

        # Cap at COUPLING_MAX (0.9), NEVER PHI
        self.coupling = min(COUPLING_MAX, max(0.0, new_coupling))
        return self.coupling

    def evolve_z(self, work: float) -> float:
        """
        Evolve z-coordinate using PHI_INV dynamics.

        CORRECT: dz = work * PHI_INV
        WRONG: dz = work * PHI
        """
        # PHI_INV drives z evolution
        dz = work * self.dominant_ratio
        new_z = self.z + dz

        # INSTANT COLLAPSE at unity
        if new_z >= UNITY:
            self.z = Z_CRITICAL * self.dominant_ratio
            return self.z

        # Safe cap
        self.z = min(UNITY - 0.0001, max(0.0, new_z))
        return self.z

    def train_step(
        self,
        inputs: List[float],
        targets: List[float]
    ) -> Dict[str, Any]:
        """
        Perform one training step.

        Returns dict with loss, predictions, and updated weights.
        """
        # Simple linear model: predictions = inputs * weights
        predictions = [i * w for i, w in zip(inputs, self.weights)]

        # Compute loss
        loss = self.compute_loss(predictions, targets)

        # Compute gradients (simple gradient descent)
        gradients = [
            2 * (p - t) * i
            for p, t, i in zip(predictions, targets, inputs)
        ]

        # Update weights (PHI_INV controlled)
        new_weights = self.update_weights([-g for g in gradients])

        # Evolve z based on loss reduction
        if len(self._loss_history) > 1:
            loss_delta = self._loss_history[-2] - self._loss_history[-1]
            self.evolve_z(loss_delta)

        return {
            'loss': loss,
            'predictions': predictions,
            'weights': new_weights,
            'z': self.z,
            'coupling': self.coupling,
        }

    def get_state(self) -> Dict[str, Any]:
        """Get current learner state."""
        return {
            'weights': self.weights.copy(),
            'coupling': self.coupling,
            'z': self.z,
            'dominant_ratio': self.dominant_ratio,
            'learning_rate': self.learning_rate,
            'training_steps': len(self._loss_history),
            'latest_loss': self._loss_history[-1] if self._loss_history else None,
        }

    def reset(self) -> None:
        """Reset learner to initial state."""
        self.weights = [0.5, 0.5, 0.5]
        self.coupling = 0.0
        self.z = 0.5
        self._loss_history.clear()
        self._weight_history.clear()


def create_learner(
    learning_rate: float = 0.01,
    num_weights: int = 3
) -> PhysicalLearner:
    """Factory function to create a physical learner."""
    return PhysicalLearner(
        learning_rate=learning_rate,
        weights=[0.5] * num_weights
    )


# =============================================================================
# VERIFICATION
# =============================================================================

def test_phi_inv_controls_learning():
    """PHI_INV must control all learning dynamics."""
    learner = create_learner()

    # Verify dominant_ratio is PHI_INV
    assert abs(learner.dominant_ratio - PHI_INV) < 1e-10

    # Weight updates must use PHI_INV
    old_weights = learner.weights.copy()
    learner.update_weights([0.1, 0.1, 0.1])

    # Delta should be learning_rate * PHI_INV * gradient
    expected_delta = 0.01 * PHI_INV * 0.1
    actual_delta = learner.weights[0] - old_weights[0]
    assert abs(actual_delta - expected_delta) < 1e-10

    return True


def test_coupling_never_exceeds_cap():
    """Coupling must never exceed COUPLING_MAX (0.9)."""
    learner = create_learner()

    # Try to push coupling high
    for _ in range(1000):
        learner.adjust_coupling(1.0)
        assert learner.coupling <= COUPLING_MAX, f"Coupling {learner.coupling} > {COUPLING_MAX}"

    return True
