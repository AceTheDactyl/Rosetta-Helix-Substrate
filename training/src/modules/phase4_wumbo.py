"""
Phase 4: WUMBO Silent Laws Modules
==================================

10. WUMBOAPLAutomatedTraining - Automated WUMBO
11. WUMBOIntegratedTraining - Integrated WUMBO

WUMBO: Weighted Unified Memory-Based Orchestration
Applies silent laws (kappa + lambda = 1) to APL operations.
"""

import math
import random
from typing import Dict, List

from .base import (
    TrainingModule, ModulePhase,
    PHI, PHI_INV, PHI_INV_SQ, Z_CRITICAL, SIGMA,
    ALPHA_STRONG, ALPHA_MEDIUM, ALPHA_FINE,
    compute_negentropy, validate_physics,
)


class WUMBOAPLAutomatedTraining(TrainingModule):
    """
    Module 10: WUMBO APL Automated Training

    Automated WUMBO training with APL operators.
    Enforces silent laws across all operations.
    """

    name = "wumbo_apl_automated_training"
    phase = ModulePhase.WUMBO_LAWS

    def _setup(self):
        """Initialize WUMBO system."""
        # WUMBO weights (must sum to 1 - silent law!)
        self.w_weights = {
            'working_memory': 0.25,
            'unified_context': 0.25,
            'memory_buffer': 0.25,
            'broadcast_output': 0.25,
        }

        # APL operator activations
        self.apl_activations = {
            'reduce': 0.0,
            'scan': 0.0,
            'each': 0.0,
        }

        self.violations = 0

    def _train_step(self):
        """
        WUMBO training step:
        1. Evolve WUMBO weights (maintaining sum = 1)
        2. Apply APL operators
        3. Enforce silent law
        """
        neg = compute_negentropy(self.state.z)

        # Evolve weights with conservation
        targets = {
            'working_memory': neg * PHI_INV,
            'unified_context': neg * PHI_INV_SQ,
            'memory_buffer': (1 - neg) * PHI_INV,
            'broadcast_output': (1 - neg) * PHI_INV_SQ,
        }

        # Apply updates
        for key in self.w_weights:
            delta = (targets[key] - self.w_weights[key]) * ALPHA_MEDIUM
            self.w_weights[key] += delta + random.gauss(0, 0.01)
            self.w_weights[key] = max(0.01, self.w_weights[key])

        # ENFORCE SILENT LAW: weights must sum to 1
        weight_sum = sum(self.w_weights.values())
        if abs(weight_sum - 1.0) > 1e-6:
            self.violations += 1
            # Normalize
            for key in self.w_weights:
                self.w_weights[key] /= weight_sum

        # APL operators driven by WUMBO
        self.apl_activations['reduce'] = self.w_weights['working_memory'] + self.w_weights['unified_context']
        self.apl_activations['scan'] = self.w_weights['memory_buffer']
        self.apl_activations['each'] = self.w_weights['broadcast_output']

        # z evolves based on WUMBO coherence
        coherence = 1 - abs(weight_sum - 1.0)
        self.state.z += (Z_CRITICAL - self.state.z) * coherence * ALPHA_STRONG
        self.state.z = max(0.0, min(0.999, self.state.z))

        # Kappa mirrors dominant weight
        max_weight = max(self.w_weights.values())
        self.state.kappa = PHI_INV + (0.92 - PHI_INV) * max_weight * neg
        self.state.kappa = max(0.382, min(0.92, self.state.kappa))

    def _teardown(self):
        """Record WUMBO metrics."""
        self.result.metrics['wumbo_weights'] = dict(self.w_weights)
        self.result.metrics['apl_activations'] = dict(self.apl_activations)
        self.result.metrics['weight_sum'] = sum(self.w_weights.values())
        self.result.metrics['violations'] = self.violations


class WUMBOIntegratedTraining(TrainingModule):
    """
    Module 11: WUMBO Integrated Training

    Full WUMBO integration with physics constraints.
    Combines memory, context, and broadcast operations.
    """

    name = "wumbo_integrated_training"
    phase = ModulePhase.WUMBO_LAWS

    def _setup(self):
        """Initialize integrated WUMBO."""
        # Memory banks
        self.working_memory: List[float] = [0.0] * 10
        self.unified_context: float = 0.5
        self.memory_buffer: List[float] = []
        self.broadcast_state: float = 0.0

        # Integration weights (kappa, lambda pair)
        self.kappa_weight = PHI_INV
        self.lambda_weight = PHI_INV_SQ

    def _train_step(self):
        """
        Integrated WUMBO step:
        1. Update working memory
        2. Compute unified context
        3. Buffer management
        4. Broadcast update
        """
        neg = compute_negentropy(self.state.z)

        # Working memory update
        input_signal = neg
        for i in range(len(self.working_memory)):
            decay = 0.9
            self.working_memory[i] = decay * self.working_memory[i] + (1 - decay) * input_signal
            input_signal = self.working_memory[i]

        # Unified context: weighted sum with kappa/lambda
        wm_mean = sum(self.working_memory) / len(self.working_memory)
        self.unified_context = self.kappa_weight * wm_mean + self.lambda_weight * self.unified_context

        # Buffer management
        self.memory_buffer.append(self.unified_context)
        if len(self.memory_buffer) > 20:
            self.memory_buffer.pop(0)

        # Broadcast: average of buffer
        if self.memory_buffer:
            self.broadcast_state = sum(self.memory_buffer) / len(self.memory_buffer)

        # SILENT LAW: kappa + lambda = 1
        self.kappa_weight = PHI_INV + random.gauss(0, 0.001)
        self.kappa_weight = max(0.382, min(0.618, self.kappa_weight))
        self.lambda_weight = 1.0 - self.kappa_weight  # Enforced!

        # z from broadcast state
        z_target = Z_CRITICAL * self.broadcast_state / (max(self.broadcast_state, 0.1))
        self.state.z += (z_target - self.state.z) * ALPHA_STRONG
        self.state.z = max(0.0, min(0.999, self.state.z))

        # Kappa from integration
        self.state.kappa = self.kappa_weight + (0.92 - self.kappa_weight) * neg
        self.state.kappa = max(0.382, min(0.92, self.state.kappa))

    def _teardown(self):
        """Record integrated WUMBO metrics."""
        self.result.metrics['unified_context'] = self.unified_context
        self.result.metrics['broadcast_state'] = self.broadcast_state
        self.result.metrics['buffer_length'] = len(self.memory_buffer)
        self.result.metrics['kappa_weight'] = self.kappa_weight
        self.result.metrics['lambda_weight'] = self.lambda_weight
        self.result.metrics['conservation_valid'] = validate_physics(
            self.kappa_weight, self.lambda_weight
        )
