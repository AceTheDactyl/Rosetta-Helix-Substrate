"""
Phase 3: Helix Geometry Modules
===============================

7. HelixNN - Helix neural network
8. PrismaticHelixTraining - Prismatic processing
9. FullHelixIntegration - Complete helix integration
"""

import math
import random
from typing import List, Tuple

from .base import (
    TrainingModule, ModulePhase,
    PHI, PHI_INV, Z_CRITICAL, SIGMA,
    ALPHA_STRONG, ALPHA_MEDIUM, ALPHA_FINE,
    compute_negentropy,
)


class HelixNN(TrainingModule):
    """
    Module 7: Helix Neural Network

    Neural network with helix geometry constraints.
    Layers follow helical structure with phi-based spacing.
    """

    name = "helix_nn"
    phase = ModulePhase.HELIX_GEOMETRY

    def _setup(self):
        """Initialize helix neural network."""
        self.n_layers = 6  # Hexagonal symmetry
        self.layer_spacing = PHI_INV  # Golden ratio spacing

        # Layer activations
        self.activations = [0.5 for _ in range(self.n_layers)]

        # Helix parameters
        self.helix_phase = 0.0
        self.helix_radius = 0.1

    def _train_step(self):
        """
        Helix NN training step:
        Layers propagate signal along helix trajectory.
        """
        neg = compute_negentropy(self.state.z)

        # Input signal
        input_signal = neg

        # Forward pass through helix layers
        signal = input_signal
        for i in range(self.n_layers):
            # Helix transformation
            helix_angle = 2 * math.pi * i / self.n_layers + self.helix_phase
            helix_z = i * self.layer_spacing

            # Activation with helix modulation
            modulation = math.cos(helix_angle) * self.helix_radius
            self.activations[i] = math.tanh(signal + modulation)
            signal = self.activations[i]

        # Update helix phase
        self.helix_phase += ALPHA_FINE
        self.helix_phase = self.helix_phase % (2 * math.pi)

        # Output drives z evolution
        output = self.activations[-1]
        z_target = Z_CRITICAL + output * 0.1
        self.state.z += (z_target - self.state.z) * ALPHA_STRONG
        self.state.z = max(0.0, min(0.999, self.state.z))

        # Kappa from layer coherence
        coherence = sum(self.activations) / self.n_layers
        self.state.kappa = PHI_INV + (0.92 - PHI_INV) * abs(coherence)
        self.state.kappa = max(0.382, min(0.92, self.state.kappa))

    def _teardown(self):
        """Record helix NN metrics."""
        self.result.metrics['layer_activations'] = self.activations
        self.result.metrics['helix_phase'] = self.helix_phase
        self.result.metrics['output_activation'] = self.activations[-1]


class PrismaticHelixTraining(TrainingModule):
    """
    Module 8: Prismatic Helix Training

    Helix with prismatic (faceted) structure.
    6 faces for hexagonal symmetry.
    """

    name = "prismatic_helix_training"
    phase = ModulePhase.HELIX_GEOMETRY

    def _setup(self):
        """Initialize prismatic helix."""
        self.n_faces = 6  # Hexagonal prism
        self.face_normals: List[Tuple[float, float]] = []

        # Initialize face normals
        for i in range(self.n_faces):
            angle = 2 * math.pi * i / self.n_faces
            self.face_normals.append((math.cos(angle), math.sin(angle)))

        # Face activations
        self.face_activations = [0.0 for _ in range(self.n_faces)]

        # Helix pitch (vertical rise per turn)
        self.pitch = Z_CRITICAL / self.n_faces

    def _train_step(self):
        """
        Prismatic helix training step:
        Signal propagates through faceted helix structure.
        """
        neg = compute_negentropy(self.state.z)

        # Project signal onto each face
        signal_vector = (neg, math.sqrt(1 - neg**2) if neg < 1 else 0)

        total_activation = 0.0
        for i, (nx, ny) in enumerate(self.face_normals):
            # Dot product: signal projection onto face
            projection = signal_vector[0] * nx + signal_vector[1] * ny
            self.face_activations[i] = max(0, projection)  # ReLU
            total_activation += self.face_activations[i]

        # Normalize
        if total_activation > 0:
            self.face_activations = [a / total_activation for a in self.face_activations]

        # Find dominant face
        dominant_face = self.face_activations.index(max(self.face_activations))
        dominant_angle = 2 * math.pi * dominant_face / self.n_faces

        # z evolves based on dominant face alignment with 60 deg
        target_angle = math.pi / 3  # 60 degrees
        angle_error = abs(dominant_angle - target_angle)
        alignment = 1 - angle_error / math.pi

        self.state.z += (Z_CRITICAL - self.state.z) * alignment * ALPHA_STRONG
        self.state.z = max(0.0, min(0.999, self.state.z))

        # Kappa from alignment
        self.state.kappa = PHI_INV + (0.92 - PHI_INV) * alignment * neg
        self.state.kappa = max(0.382, min(0.92, self.state.kappa))

    def _teardown(self):
        """Record prismatic helix metrics."""
        self.result.metrics['face_activations'] = self.face_activations
        dominant = self.face_activations.index(max(self.face_activations))
        self.result.metrics['dominant_face'] = dominant
        self.result.metrics['dominant_angle'] = 2 * math.pi * dominant / self.n_faces


class FullHelixIntegration(TrainingModule):
    """
    Module 9: Full Helix Integration

    Complete integration of helix geometry with training.
    Combines NN and prismatic structures.
    """

    name = "full_helix_integration"
    phase = ModulePhase.HELIX_GEOMETRY

    def _setup(self):
        """Initialize full helix system."""
        # Helix parameters
        self.n_turns = 3
        self.points_per_turn = 60  # Hexagonal
        self.total_points = self.n_turns * self.points_per_turn

        # Helix trajectory
        self.helix_z = []
        self.helix_x = []
        self.helix_y = []

        for i in range(self.total_points):
            t = i / self.points_per_turn
            angle = 2 * math.pi * t
            z = t * Z_CRITICAL / self.n_turns
            x = math.cos(angle) * PHI_INV
            y = math.sin(angle) * PHI_INV
            self.helix_z.append(z)
            self.helix_x.append(x)
            self.helix_y.append(y)

        # Current position along helix
        self.position = 0

    def _train_step(self):
        """
        Full helix integration step:
        Move along helix trajectory, tracking physics.
        """
        neg = compute_negentropy(self.state.z)

        # Advance position
        step_size = int(neg * 5) + 1
        self.position = (self.position + step_size) % self.total_points

        # Get current helix coordinates
        current_z = self.helix_z[self.position]
        current_x = self.helix_x[self.position]
        current_y = self.helix_y[self.position]

        # State z tracks helix z
        self.state.z += (current_z - self.state.z) * ALPHA_STRONG
        self.state.z = max(0.0, min(0.999, self.state.z))

        # Kappa from helix position
        radial_dist = math.sqrt(current_x**2 + current_y**2)
        self.state.kappa = PHI_INV + (0.92 - PHI_INV) * (radial_dist / PHI_INV) * neg
        self.state.kappa = max(0.382, min(0.92, self.state.kappa))

    def _teardown(self):
        """Record full helix metrics."""
        self.result.metrics['final_position'] = self.position
        self.result.metrics['helix_z'] = self.helix_z[self.position]
        self.result.metrics['turns_completed'] = self.position / self.points_per_turn
