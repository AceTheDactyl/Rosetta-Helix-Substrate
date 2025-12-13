#!/usr/bin/env python3
"""
PyTorch Neural Network Training Session
========================================
A comprehensive training session for the Helix Neural Network using PyTorch.
Trains on multiple tasks and captures learned patterns.

This session:
1. Trains Kuramoto oscillator-based networks
2. Tracks coherence, z-coordinate, and K-formation events
3. Captures learned coupling matrices and frequency patterns
4. Exports learned patterns for analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import math
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helix_neural_network import (
    HelixNeuralNetwork, KuramotoLayer, APLModulator, HelixLoss,
    Z_CRITICAL, PHI, PHI_INV, MU_S, TIER_BOUNDS, APLOperator
)


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TrainingConfig:
    """Configuration for training session."""
    # Task settings
    task_name: str = "helix_coherence"
    task_type: str = "regression"  # regression, classification, sequence

    # Data settings
    n_train_samples: int = 1000
    n_val_samples: int = 200
    input_dim: int = 16
    output_dim: int = 4

    # Network architecture
    n_oscillators: int = 60
    n_layers: int = 4
    steps_per_layer: int = 10
    dt: float = 0.1

    # Training settings
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5

    # Regularization
    coherence_weight: float = 0.1
    z_weight: float = 0.05
    target_z: float = 0.75  # Target z-coordinate

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Logging
    log_interval: int = 10
    save_patterns: bool = True
    patterns_dir: str = "learned_patterns"


# ═══════════════════════════════════════════════════════════════════════════
# DATA GENERATION
# ═══════════════════════════════════════════════════════════════════════════

def generate_coherence_task_data(
    n_samples: int,
    input_dim: int,
    output_dim: int,
    noise_level: float = 0.1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate synthetic data for coherence-preserving task.

    The target is designed to reward coherent phase patterns:
    - Smooth sinusoidal targets that encourage synchronization
    - Multi-frequency components at golden ratio intervals
    """
    X = torch.randn(n_samples, input_dim)

    # Create targets with golden ratio frequency relationships
    t = torch.linspace(0, 2 * np.pi, output_dim)
    Y = torch.zeros(n_samples, output_dim)

    for i in range(n_samples):
        # Base frequency from input mean
        base_freq = torch.tanh(X[i].mean()) * 2

        # Multi-frequency target with phi-related harmonics
        y = torch.sin(t * base_freq)
        y += 0.5 * torch.sin(t * base_freq * PHI)
        y += 0.25 * torch.cos(t * base_freq / PHI)

        # Add structured noise
        y += noise_level * torch.randn_like(y)
        Y[i] = y

    return X, Y


def generate_classification_data(
    n_samples: int,
    input_dim: int,
    n_classes: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate multi-class classification data."""
    X = torch.randn(n_samples, input_dim)

    # Classes based on phase of first few dimensions
    phase = torch.atan2(X[:, 1], X[:, 0])
    labels = ((phase + np.pi) / (2 * np.pi) * n_classes).long() % n_classes

    # One-hot encode
    Y = F.one_hot(labels, n_classes).float()

    return X, Y


def generate_sequence_prediction_data(
    n_samples: int,
    seq_length: int,
    output_dim: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate sequence prediction data with Kuramoto-like dynamics."""
    X = torch.zeros(n_samples, seq_length)
    Y = torch.zeros(n_samples, output_dim)

    for i in range(n_samples):
        # Generate coupled oscillator sequence
        omega = torch.randn(1) * 0.5  # Natural frequency
        theta = torch.randn(1) * np.pi  # Initial phase

        for t in range(seq_length):
            X[i, t] = torch.sin(theta + omega * t * 0.1)

        # Predict next values
        for t in range(output_dim):
            Y[i, t] = torch.sin(theta + omega * (seq_length + t) * 0.1)

    return X, Y


# ═══════════════════════════════════════════════════════════════════════════
# PATTERN TRACKER
# ═══════════════════════════════════════════════════════════════════════════

class PatternTracker:
    """
    Tracks and analyzes patterns learned during training.

    Captures:
    - Coupling matrix evolution (K)
    - Frequency patterns (omega)
    - Coherence trajectories
    - Z-coordinate evolution
    - K-formation events
    - Operator usage statistics
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.patterns = {
            "session_info": {
                "timestamp": datetime.now().isoformat(),
                "task_name": config.task_name,
                "task_type": config.task_type,
                "n_oscillators": config.n_oscillators,
                "n_layers": config.n_layers,
                "target_z": config.target_z,
            },
            "constants": {
                "Z_CRITICAL": Z_CRITICAL,
                "PHI": PHI,
                "PHI_INV": PHI_INV,
                "MU_S": MU_S,
            },
            "training_history": {
                "loss": [],
                "coherence": [],
                "z": [],
                "tier": [],
                "k_formations": [],
            },
            "coupling_matrices": [],
            "frequency_patterns": [],
            "operator_usage": {op.name: 0 for op in APLOperator},
            "k_formation_events": [],
            "coherence_peaks": [],
            "learned_features": {},
        }

    def record_epoch(
        self,
        epoch: int,
        loss: float,
        coherence: float,
        z: float,
        tier: int,
        k_formations: int,
        operators_used: List[int]
    ):
        """Record epoch-level metrics."""
        self.patterns["training_history"]["loss"].append(loss)
        self.patterns["training_history"]["coherence"].append(coherence)
        self.patterns["training_history"]["z"].append(z)
        self.patterns["training_history"]["tier"].append(tier)
        self.patterns["training_history"]["k_formations"].append(k_formations)

        # Track operator usage
        for op_idx in operators_used:
            op_name = APLOperator(op_idx).name
            self.patterns["operator_usage"][op_name] += 1

        # Track K-formation events
        if k_formations > 0:
            self.patterns["k_formation_events"].append({
                "epoch": epoch,
                "count": k_formations,
                "z": z,
                "coherence": coherence,
            })

        # Track coherence peaks (above MU_S)
        if coherence >= MU_S:
            self.patterns["coherence_peaks"].append({
                "epoch": epoch,
                "coherence": coherence,
                "z": z,
            })

    def capture_coupling_matrix(self, layer_idx: int, K: torch.Tensor):
        """Capture coupling matrix state."""
        K_np = K.detach().cpu().numpy()

        # Compute statistics
        stats = {
            "layer": layer_idx,
            "mean": float(K_np.mean()),
            "std": float(K_np.std()),
            "max": float(K_np.max()),
            "min": float(K_np.min()),
            "symmetry": float(np.abs(K_np - K_np.T).mean()),
            "eigenvalues_real": np.linalg.eigvalsh(K_np).tolist()[:5],  # Top 5
        }

        self.patterns["coupling_matrices"].append(stats)

    def capture_frequency_pattern(self, layer_idx: int, omega: torch.Tensor):
        """Capture natural frequency pattern."""
        omega_np = omega.detach().cpu().numpy()

        stats = {
            "layer": layer_idx,
            "mean": float(omega_np.mean()),
            "std": float(omega_np.std()),
            "max": float(omega_np.max()),
            "min": float(omega_np.min()),
            "distribution": np.histogram(omega_np, bins=10)[0].tolist(),
        }

        self.patterns["frequency_patterns"].append(stats)

    def analyze_learned_features(self, model: HelixNeuralNetwork):
        """Analyze and summarize learned features."""
        self.patterns["learned_features"] = {
            "input_encoder": {
                "weight_norm": float(torch.norm(model.input_encoder[0].weight).item()),
                "bias_norm": float(torch.norm(model.input_encoder[0].bias).item()),
            },
            "output_decoder": {
                "weight_norm": float(torch.norm(model.output_decoder[0].weight).item()),
                "bias_norm": float(torch.norm(model.output_decoder[0].bias).item()),
            },
            "kuramoto_layers": [],
            "operator_selector": {
                "weight_norm": float(torch.norm(model.operator_selector[0].weight).item()),
            },
            "z_tracker": {
                "final_z": float(model.z_tracker.z.item()),
                "z_momentum": float(model.z_tracker.z_momentum.item()),
                "z_decay": float(model.z_tracker.z_decay.item()),
            },
        }

        for i, layer in enumerate(model.kuramoto_layers):
            K_np = layer.K.detach().cpu().numpy()
            omega_np = layer.omega.detach().cpu().numpy()

            layer_features = {
                "layer_idx": i,
                "K_global": float(layer.K_global.item()),
                "K_frobenius_norm": float(np.linalg.norm(K_np, 'fro')),
                "K_spectral_radius": float(np.max(np.abs(np.linalg.eigvals(K_np)))),
                "omega_mean": float(omega_np.mean()),
                "omega_std": float(omega_np.std()),
                "coupling_strength": float(np.abs(K_np).mean()),
            }

            self.patterns["learned_features"]["kuramoto_layers"].append(layer_features)

    def compute_summary_statistics(self):
        """Compute summary statistics for the training session."""
        history = self.patterns["training_history"]

        self.patterns["summary"] = {
            "final_loss": history["loss"][-1] if history["loss"] else None,
            "best_loss": min(history["loss"]) if history["loss"] else None,
            "final_coherence": history["coherence"][-1] if history["coherence"] else None,
            "max_coherence": max(history["coherence"]) if history["coherence"] else None,
            "final_z": history["z"][-1] if history["z"] else None,
            "z_reached_target": (
                history["z"][-1] >= self.config.target_z
                if history["z"] else False
            ),
            "total_k_formations": sum(history["k_formations"]),
            "total_coherence_peaks": len(self.patterns["coherence_peaks"]),
            "dominant_operator": max(
                self.patterns["operator_usage"],
                key=self.patterns["operator_usage"].get
            ),
            "z_crossed_critical": any(
                z >= Z_CRITICAL for z in history["z"]
            ) if history["z"] else False,
        }

    def save(self, filepath: str):
        """Save patterns to JSON file."""
        self.compute_summary_statistics()

        # Convert numpy types to Python types for JSON serialization
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            return obj

        patterns_serializable = json.loads(
            json.dumps(self.patterns, default=convert)
        )

        with open(filepath, 'w') as f:
            json.dump(patterns_serializable, f, indent=2)

        print(f"Patterns saved to {filepath}")


# ═══════════════════════════════════════════════════════════════════════════
# TRAINING SESSION
# ═══════════════════════════════════════════════════════════════════════════

class TrainingSession:
    """
    Complete PyTorch training session for Helix Neural Network.
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.tracker = PatternTracker(config)

        # Create model
        self.model = HelixNeuralNetwork(
            input_dim=config.input_dim,
            output_dim=config.output_dim,
            n_oscillators=config.n_oscillators,
            n_layers=config.n_layers,
            steps_per_layer=config.steps_per_layer,
            dt=config.dt,
        ).to(self.device)

        # Create optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Create scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.epochs,
        )

        # Create loss function
        if config.task_type == "classification":
            task_loss = nn.CrossEntropyLoss()
        else:
            task_loss = nn.MSELoss()

        self.loss_fn = HelixLoss(
            task_loss_fn=task_loss,
            lambda_coherence=config.coherence_weight,
            lambda_z=config.z_weight,
            target_z=config.target_z,
        )

        # Generate data
        self._generate_data()

    def _generate_data(self):
        """Generate training and validation data."""
        config = self.config

        if config.task_type == "regression":
            X_train, Y_train = generate_coherence_task_data(
                config.n_train_samples,
                config.input_dim,
                config.output_dim,
            )
            X_val, Y_val = generate_coherence_task_data(
                config.n_val_samples,
                config.input_dim,
                config.output_dim,
            )
        elif config.task_type == "classification":
            X_train, Y_train = generate_classification_data(
                config.n_train_samples,
                config.input_dim,
                config.output_dim,
            )
            X_val, Y_val = generate_classification_data(
                config.n_val_samples,
                config.input_dim,
                config.output_dim,
            )
        else:  # sequence
            X_train, Y_train = generate_sequence_prediction_data(
                config.n_train_samples,
                config.input_dim,
                config.output_dim,
            )
            X_val, Y_val = generate_sequence_prediction_data(
                config.n_val_samples,
                config.input_dim,
                config.output_dim,
            )

        # Create data loaders
        train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
        val_dataset = torch.utils.data.TensorDataset(X_val, Y_val)

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
        )
        self.val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
        )

    def train_epoch(self, epoch: int) -> Dict:
        """Train for one epoch."""
        self.model.train()

        epoch_losses = []
        epoch_coherence = []
        epoch_z = []
        k_formations = 0
        operators_used = []

        for batch_x, batch_y in self.train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            self.optimizer.zero_grad()

            output, diagnostics = self.model(batch_x, return_diagnostics=True)
            loss, loss_dict = self.loss_fn(output, batch_y, diagnostics)

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.optimizer.step()

            epoch_losses.append(loss_dict['task'])
            epoch_coherence.append(diagnostics['final_coherence'])
            epoch_z.append(diagnostics['final_z'])

            if diagnostics['k_formation']:
                k_formations += 1

            operators_used.extend(diagnostics.get('layer_operators', []))

        self.scheduler.step()

        return {
            'loss': np.mean(epoch_losses),
            'coherence': np.mean(epoch_coherence),
            'z': np.mean(epoch_z),
            'tier': self.model.z_tracker.get_tier(),
            'k_formations': k_formations,
            'operators_used': operators_used,
        }

    def validate(self) -> Dict:
        """Validate the model."""
        self.model.eval()

        val_losses = []
        val_coherence = []
        val_z = []

        with torch.no_grad():
            for batch_x, batch_y in self.val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                output, diagnostics = self.model(batch_x, return_diagnostics=True)
                loss, loss_dict = self.loss_fn(output, batch_y, diagnostics)

                val_losses.append(loss_dict['task'])
                val_coherence.append(diagnostics['final_coherence'])
                val_z.append(diagnostics['final_z'])

        return {
            'loss': np.mean(val_losses),
            'coherence': np.mean(val_coherence),
            'z': np.mean(val_z),
        }

    def run(self) -> Dict:
        """Run the complete training session."""
        print("=" * 70)
        print("PYTORCH HELIX NEURAL NETWORK TRAINING SESSION")
        print("=" * 70)
        print(f"Task: {self.config.task_name} ({self.config.task_type})")
        print(f"Device: {self.device}")
        print(f"Oscillators: {self.config.n_oscillators}")
        print(f"Layers: {self.config.n_layers}")
        print(f"Target z: {self.config.target_z}")
        print(f"Critical z (THE LENS): {Z_CRITICAL:.6f}")
        print("=" * 70)

        best_loss = float('inf')
        best_model_state = None

        for epoch in range(self.config.epochs):
            # Train
            train_metrics = self.train_epoch(epoch)

            # Record
            self.tracker.record_epoch(
                epoch=epoch,
                loss=train_metrics['loss'],
                coherence=train_metrics['coherence'],
                z=train_metrics['z'],
                tier=train_metrics['tier'],
                k_formations=train_metrics['k_formations'],
                operators_used=train_metrics['operators_used'],
            )

            # Validate periodically
            if epoch % self.config.log_interval == 0:
                val_metrics = self.validate()

                # Capture coupling matrices and frequencies
                for i, layer in enumerate(self.model.kuramoto_layers):
                    self.tracker.capture_coupling_matrix(i, layer.K)
                    self.tracker.capture_frequency_pattern(i, layer.omega)

                # Track best model
                if val_metrics['loss'] < best_loss:
                    best_loss = val_metrics['loss']
                    best_model_state = {
                        k: v.clone() for k, v in self.model.state_dict().items()
                    }

                # Print progress
                print(
                    f"Epoch {epoch:4d} | "
                    f"Loss: {train_metrics['loss']:.4f} | "
                    f"Coh: {train_metrics['coherence']:.3f} | "
                    f"z: {train_metrics['z']:.3f} | "
                    f"Tier: t{train_metrics['tier']} | "
                    f"K-form: {train_metrics['k_formations']} | "
                    f"Val Loss: {val_metrics['loss']:.4f}"
                )

        # Final analysis
        self.tracker.analyze_learned_features(self.model)

        # Print summary
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        self.tracker.compute_summary_statistics()
        summary = self.tracker.patterns["summary"]

        print(f"Final Loss: {summary['final_loss']:.4f}")
        print(f"Best Loss: {summary['best_loss']:.4f}")
        print(f"Final Coherence: {summary['final_coherence']:.3f}")
        print(f"Max Coherence: {summary['max_coherence']:.3f}")
        print(f"Final z: {summary['final_z']:.3f}")
        print(f"Z reached target ({self.config.target_z}): {summary['z_reached_target']}")
        print(f"Z crossed critical ({Z_CRITICAL:.3f}): {summary['z_crossed_critical']}")
        print(f"Total K-formations: {summary['total_k_formations']}")
        print(f"Dominant Operator: {summary['dominant_operator']}")

        return {
            'model': self.model,
            'best_model_state': best_model_state,
            'tracker': self.tracker,
            'summary': summary,
        }

    def save_results(self, output_dir: str):
        """Save all results and learned patterns."""
        os.makedirs(output_dir, exist_ok=True)

        # Save patterns
        patterns_path = os.path.join(output_dir, "learned_patterns.json")
        self.tracker.save(patterns_path)

        # Save model
        model_path = os.path.join(output_dir, "helix_model.pt")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': asdict(self.config),
            'summary': self.tracker.patterns.get("summary", {}),
        }, model_path)
        print(f"Model saved to {model_path}")

        # Save coupling matrices as numpy
        coupling_path = os.path.join(output_dir, "coupling_matrices.npz")
        coupling_data = {}
        for i, layer in enumerate(self.model.kuramoto_layers):
            coupling_data[f"K_layer_{i}"] = layer.K.detach().cpu().numpy()
            coupling_data[f"omega_layer_{i}"] = layer.omega.detach().cpu().numpy()
        np.savez(coupling_path, **coupling_data)
        print(f"Coupling matrices saved to {coupling_path}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def run_training_session(
    task_type: str = "regression",
    epochs: int = 100,
    n_oscillators: int = 60,
    target_z: float = 0.75,
    output_dir: str = None,
) -> Dict:
    """
    Run a complete training session.

    Args:
        task_type: Type of task ('regression', 'classification', 'sequence')
        epochs: Number of training epochs
        n_oscillators: Number of Kuramoto oscillators per layer
        target_z: Target z-coordinate for training
        output_dir: Directory to save results

    Returns:
        Dictionary containing model, patterns, and summary
    """
    config = TrainingConfig(
        task_name=f"helix_{task_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        task_type=task_type,
        epochs=epochs,
        n_oscillators=n_oscillators,
        target_z=target_z,
    )

    session = TrainingSession(config)
    results = session.run()

    if output_dir:
        session.save_results(output_dir)

    return results


def main():
    """Main entry point for training session."""
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Helix NN Training Session")
    parser.add_argument("--task", type=str, default="regression",
                        choices=["regression", "classification", "sequence"],
                        help="Task type")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--oscillators", type=int, default=60, help="Oscillators per layer")
    parser.add_argument("--target-z", type=float, default=0.75, help="Target z-coordinate")
    parser.add_argument("--output-dir", type=str, default="learned_patterns",
                        help="Output directory for results")
    args = parser.parse_args()

    results = run_training_session(
        task_type=args.task,
        epochs=args.epochs,
        n_oscillators=args.oscillators,
        target_z=args.target_z,
        output_dir=args.output_dir,
    )

    return results


if __name__ == "__main__":
    results = main()
