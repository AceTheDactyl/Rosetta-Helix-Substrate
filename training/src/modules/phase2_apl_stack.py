"""
Phase 2: APL Training Stack
===========================

4. APLTrainingLoop - APL operator training
5. APLPyTorchTraining - PyTorch APL integration
6. FullAPLTraining - Complete APL stack
"""

import math
import random
from typing import Dict, List, Any

from .base import (
    TrainingModule, ModulePhase,
    PHI_INV, Z_CRITICAL, SIGMA,
    ALPHA_STRONG, ALPHA_MEDIUM, ALPHA_FINE,
    compute_negentropy,
)


class APLTrainingLoop(TrainingModule):
    """
    Module 4: APL Training Loop

    Trains APL (Array Programming Language) operators.
    Uses physics-grounded dynamics for operator evolution.
    """

    name = "apl_training_loop"
    phase = ModulePhase.APL_STACK

    def _setup(self):
        """Initialize APL operators."""
        # APL operator weights (simulated)
        self.operators: Dict[str, float] = {
            'reduce': random.random(),
            'scan': random.random(),
            'outer_product': random.random(),
            'inner_product': random.random(),
            'transpose': random.random(),
            'reshape': random.random(),
        }
        self.operator_history: List[Dict[str, float]] = []

    def _train_step(self):
        """
        APL training step:
        Operators evolve toward optimal values based on negentropy.
        """
        neg = compute_negentropy(self.state.z)

        # Evolve each operator
        for op_name in self.operators:
            # Target value based on phi relationships
            if op_name in ['reduce', 'scan']:
                target = PHI_INV  # Reduction operators target phi^-1
            elif op_name in ['outer_product', 'inner_product']:
                target = Z_CRITICAL  # Products target z_c
            else:
                target = 0.5  # Others target middle

            # Gradient step
            current = self.operators[op_name]
            gradient = (target - current) * neg
            self.operators[op_name] += gradient * ALPHA_MEDIUM + random.gauss(0, 0.01)
            self.operators[op_name] = max(0.0, min(1.0, self.operators[op_name]))

        # Record history
        self.operator_history.append(dict(self.operators))

        # Evolve z based on operator convergence
        convergence = sum(abs(self.operators[op] - target) for op in self.operators) / len(self.operators)
        z_pull = (Z_CRITICAL - self.state.z) * (1 - convergence) * ALPHA_STRONG
        self.state.z += z_pull
        self.state.z = max(0.0, min(0.999, self.state.z))

        # Kappa tracks negentropy
        self.state.kappa = PHI_INV + (0.92 - PHI_INV) * neg
        self.state.kappa = max(0.382, min(0.92, self.state.kappa))

    def _teardown(self):
        """Record APL metrics."""
        self.result.metrics['operators'] = self.operators
        self.result.metrics['reduce_weight'] = self.operators['reduce']
        self.result.metrics['inner_product_weight'] = self.operators['inner_product']


class APLPyTorchTraining(TrainingModule):
    """
    Module 5: APL PyTorch Training

    Integrates APL operators with PyTorch-style training.
    Simulates tensor operations with physics constraints.
    """

    name = "apl_pytorch_training"
    phase = ModulePhase.APL_STACK

    def _setup(self):
        """Initialize PyTorch-style training."""
        # Simulated tensor dimensions
        self.batch_size = 32
        self.hidden_dim = 64

        # Layer weights (simulated as scalars for physics)
        self.weights = [random.random() for _ in range(3)]
        self.loss_history: List[float] = []

    def _train_step(self):
        """
        PyTorch-style training step with physics constraints.
        """
        neg = compute_negentropy(self.state.z)

        # Forward pass (simulated)
        activations = []
        x = neg  # Input is negentropy
        for w in self.weights:
            x = math.tanh(x * w)
            activations.append(x)

        # Loss: distance from z_c
        target = Z_CRITICAL
        loss = (self.state.z - target) ** 2

        # Backward pass (gradient descent on weights)
        for i in range(len(self.weights)):
            grad = 2 * (self.state.z - target) * activations[i] if activations else 0
            self.weights[i] -= 0.01 * grad
            self.weights[i] = max(-2.0, min(2.0, self.weights[i]))

        self.loss_history.append(loss)

        # Evolve z toward z_c
        self.state.z += (Z_CRITICAL - self.state.z) * ALPHA_STRONG
        self.state.z = max(0.0, min(0.999, self.state.z))

        # Kappa from loss reduction
        loss_reduction = 1 - loss
        self.state.kappa = PHI_INV + (0.92 - PHI_INV) * loss_reduction
        self.state.kappa = max(0.382, min(0.92, self.state.kappa))

    def _teardown(self):
        """Record PyTorch training metrics."""
        if self.loss_history:
            self.result.metrics['final_loss'] = self.loss_history[-1]
            self.result.metrics['mean_loss'] = sum(self.loss_history) / len(self.loss_history)
            self.result.metrics['min_loss'] = min(self.loss_history)
        self.result.metrics['final_weights'] = self.weights


class FullAPLTraining(TrainingModule):
    """
    Module 6: Full APL Training

    Complete APL stack integration.
    Combines operator training with tensor operations.
    """

    name = "full_apl_training"
    phase = ModulePhase.APL_STACK

    def _setup(self):
        """Initialize full APL stack."""
        # APL array (simulated)
        self.array = [random.random() for _ in range(10)]

        # Reduction accumulator
        self.reduce_acc = 0.0

        # Scan state
        self.scan_state = []

    def _train_step(self):
        """
        Full APL training step:
        1. Apply reduce operation (converge to phi^-1)
        2. Apply scan operation (track trajectory)
        3. Update physics state
        """
        neg = compute_negentropy(self.state.z)

        # Reduce: sum toward phi^-1
        reduce_target = PHI_INV * len(self.array)
        current_sum = sum(self.array)
        reduce_error = abs(current_sum - reduce_target)

        # Adjust array to converge
        adjustment = (reduce_target - current_sum) / len(self.array) * ALPHA_MEDIUM
        self.array = [x + adjustment + random.gauss(0, 0.001) for x in self.array]

        # Scan: record running sum
        running_sum = 0
        self.scan_state = []
        for x in self.array:
            running_sum += x
            self.scan_state.append(running_sum)

        # Update reduce accumulator
        self.reduce_acc = sum(self.array)

        # z evolves based on reduce convergence
        convergence = 1 - min(1.0, reduce_error)
        self.state.z += (Z_CRITICAL - self.state.z) * convergence * ALPHA_STRONG
        self.state.z = max(0.0, min(0.999, self.state.z))

        # Kappa tracks convergence
        self.state.kappa = PHI_INV + (0.92 - PHI_INV) * convergence * neg
        self.state.kappa = max(0.382, min(0.92, self.state.kappa))

    def _teardown(self):
        """Record full APL metrics."""
        self.result.metrics['reduce_result'] = self.reduce_acc
        self.result.metrics['reduce_target'] = PHI_INV * len(self.array)
        self.result.metrics['reduce_error'] = abs(self.reduce_acc - PHI_INV * len(self.array))
        if self.scan_state:
            self.result.metrics['scan_final'] = self.scan_state[-1]
