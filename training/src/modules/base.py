"""
Base Training Module
====================

Abstract base class for all 19 training modules.
Enforces the physics constraints and provides common functionality.
"""

import math
import time
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from typing import Dict, List, Optional, Any

# Physics constants
PHI = (1 + math.sqrt(5)) / 2
PHI_INV = 1 / PHI
PHI_INV_SQ = PHI_INV ** 2
Z_CRITICAL = math.sqrt(3) / 2
SIGMA = 36.0

# K-formation thresholds
KAPPA_MIN = 0.92
ETA_MIN = PHI_INV
R_MIN = 7

# Alpha coefficients
ALPHA_STRONG = 1 / math.sqrt(SIGMA)
ALPHA_MEDIUM = 1 / math.sqrt(2 * SIGMA)
ALPHA_FINE = 1 / SIGMA


class ModulePhase(IntEnum):
    """Training phases."""
    CORE_PHYSICS = 1
    APL_STACK = 2
    HELIX_GEOMETRY = 3
    WUMBO_LAWS = 4
    DYNAMICS_FORMATION = 5
    UNIFIED_ORCHESTRATION = 6
    NIGHTLY_INTEGRATION = 7


@dataclass
class ModuleResult:
    """Result from a single training module."""
    name: str
    class_name: str
    phase: ModulePhase
    status: str = "PENDING"
    steps_run: int = 0
    duration_seconds: float = 0.0
    final_z: float = 0.5
    final_kappa: float = PHI_INV
    final_lambda: float = PHI_INV_SQ
    k_formations: int = 0
    max_negentropy: float = 0.0
    physics_valid: bool = True
    error: Optional[str] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    history: List[Dict[str, float]] = field(default_factory=list)


@dataclass
class TrainingState:
    """Shared training state across modules."""
    z: float = 0.5
    kappa: float = PHI_INV
    lambda_: float = PHI_INV_SQ
    negentropy: float = 0.0
    step: int = 0
    k_formations: int = 0


def compute_negentropy(z: float) -> float:
    """Compute negentropy signal: DeltaS_neg(z) = exp(-sigma(z - z_c)^2)"""
    d = z - Z_CRITICAL
    return math.exp(-SIGMA * d * d)


def validate_physics(kappa: float, lambda_: float) -> bool:
    """Validate conservation law: kappa + lambda = 1"""
    return abs(kappa + lambda_ - 1.0) < 1e-10


def check_k_formation(kappa: float, eta: float, R: int) -> bool:
    """Check K-formation criteria."""
    return kappa >= KAPPA_MIN and eta > ETA_MIN and R >= R_MIN


class TrainingModule(ABC):
    """
    Abstract base class for all training modules.

    All modules must:
    - Enforce kappa + lambda = 1
    - Track K-formations
    - Evolve toward z_c
    """

    # Class attributes - override in subclasses
    name: str = "base_module"
    phase: ModulePhase = ModulePhase.CORE_PHYSICS

    def __init__(self, steps: int = 200, seed: Optional[int] = None):
        self.steps = steps
        self.seed = seed if seed is not None else int(time.time())
        random.seed(self.seed)

        # Initialize state
        self.state = TrainingState()
        self.result: Optional[ModuleResult] = None

    def run(self, initial_state: Optional[TrainingState] = None) -> ModuleResult:
        """
        Run the training module.

        Args:
            initial_state: Optional initial state (for chaining modules)

        Returns:
            ModuleResult with training outcomes
        """
        if initial_state:
            self.state = initial_state

        self.result = ModuleResult(
            name=self.name,
            class_name=self.__class__.__name__,
            phase=self.phase,
        )

        start_time = time.time()

        try:
            # Pre-training setup
            self._setup()

            # Main training loop
            max_neg = 0.0
            for step in range(self.steps):
                self.state.step = step

                # Execute one training step
                self._train_step()

                # Enforce conservation law
                self._enforce_conservation()

                # Compute negentropy
                neg = compute_negentropy(self.state.z)
                self.state.negentropy = neg
                max_neg = max(max_neg, neg)

                # Check K-formation
                eta = math.sqrt(neg) if neg > 0 else 0
                R = int(7 + 3 * neg)
                if check_k_formation(self.state.kappa, eta, R):
                    self.state.k_formations += 1

                # Record history periodically
                if step % (self.steps // 10 or 1) == 0:
                    self.result.history.append({
                        'step': step,
                        'z': self.state.z,
                        'kappa': self.state.kappa,
                        'negentropy': neg,
                    })

            # Post-training
            self._teardown()

            # Fill result
            self.result.status = "PASS"
            self.result.steps_run = self.steps
            self.result.final_z = self.state.z
            self.result.final_kappa = self.state.kappa
            self.result.final_lambda = self.state.lambda_
            self.result.k_formations = self.state.k_formations
            self.result.max_negentropy = max_neg
            self.result.physics_valid = validate_physics(
                self.state.kappa, self.state.lambda_
            )

        except Exception as e:
            self.result.status = "FAIL"
            self.result.error = str(e)

        self.result.duration_seconds = time.time() - start_time
        return self.result

    def _enforce_conservation(self):
        """Enforce kappa + lambda = 1"""
        # lambda is derived from kappa
        self.state.lambda_ = 1.0 - self.state.kappa

    def _setup(self):
        """Optional setup before training."""
        pass

    def _teardown(self):
        """Optional teardown after training."""
        pass

    @abstractmethod
    def _train_step(self):
        """Execute one training step. Must be implemented by subclasses."""
        pass

    def get_state(self) -> TrainingState:
        """Get current training state."""
        return self.state

    def set_state(self, state: TrainingState):
        """Set training state."""
        self.state = state
