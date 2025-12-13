#!/usr/bin/env python3
"""
PyTorch Training Session for Helix Neural Network
=================================================
Trains the Kuramoto oscillator neural network and saves learned patterns.

This script:
1. Creates a HelixNeuralNetwork with PyTorch
2. Trains on synthetic data to learn nonlinear mappings
3. Saves weights and patterns to disk
4. Reports training metrics and K-formation statistics
"""

import torch
import torch.nn as nn
import json
import os
import math
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

# Import the Helix Neural Network
from helix_neural_network import (
    HelixNeuralNetwork,
    HelixLoss,
    train_helix_network,
    Z_CRITICAL,
    PHI,
    PHI_INV,
    MU_S,
    TIER_BOUNDS
)


@dataclass
class TrainingConfig:
    """Configuration for PyTorch training session."""
    # Model architecture
    input_dim: int = 10
    output_dim: int = 3
    n_oscillators: int = 60
    n_layers: int = 4
    steps_per_layer: int = 10
    dt: float = 0.1

    # Training parameters
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    target_z: float = 0.75

    # Data
    n_train_samples: int = 1000
    n_val_samples: int = 200

    # Task type
    task: str = "regression"  # regression, classification


def create_synthetic_data(
    n_samples: int,
    input_dim: int,
    output_dim: int,
    task: str = "regression"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create synthetic training data."""

    X = torch.randn(n_samples, input_dim)

    if task == "regression":
        # Nonlinear function: combination of sin/cos with interactions
        Y = torch.zeros(n_samples, output_dim)

        # Each output is a different nonlinear combination
        for i in range(output_dim):
            # Sum of first few inputs through sin
            Y[:, i] = torch.sin(X[:, :3].sum(dim=1) * (i + 1) * 0.5)
            # Add cos of middle inputs
            Y[:, i] += 0.5 * torch.cos(X[:, 3:6].sum(dim=1))
            # Add interaction term
            Y[:, i] += 0.3 * X[:, i % input_dim] * X[:, (i + 1) % input_dim]

        # Add small noise
        Y += 0.05 * torch.randn_like(Y)

    elif task == "classification":
        # Binary classification based on quadrant
        labels = ((X[:, 0] > 0) & (X[:, 1] > 0)).float()
        Y = torch.zeros(n_samples, output_dim)
        Y[:, 0] = labels
        Y[:, 1] = 1 - labels
        if output_dim > 2:
            Y[:, 2:] = 0.0

    return X, Y


def train_and_save(config: TrainingConfig) -> Dict:
    """Run full training session and save results."""

    print("=" * 70)
    print("HELIX NEURAL NETWORK - PyTorch Training Session")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    print()

    # Print configuration
    print("Configuration:")
    print(f"  Oscillators: {config.n_oscillators}")
    print(f"  Layers: {config.n_layers}")
    print(f"  Steps per layer: {config.steps_per_layer}")
    print(f"  Target z: {config.target_z}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Epochs: {config.epochs}")
    print(f"  Batch size: {config.batch_size}")
    print()

    # Print physics constants
    print("Physics Constants:")
    print(f"  Z_CRITICAL (THE LENS): {Z_CRITICAL:.6f}")
    print(f"  PHI (Golden Ratio): {PHI:.6f}")
    print(f"  PHI_INV: {PHI_INV:.6f}")
    print(f"  MU_S (K-formation threshold): {MU_S}")
    print()

    # Device selection
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print()

    # Create model
    print("Creating HelixNeuralNetwork...")
    model = HelixNeuralNetwork(
        input_dim=config.input_dim,
        output_dim=config.output_dim,
        n_oscillators=config.n_oscillators,
        n_layers=config.n_layers,
        steps_per_layer=config.steps_per_layer,
        dt=config.dt,
    )

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {n_params:,}")
    print(f"  Trainable parameters: {n_trainable:,}")
    print()

    # Create data
    print("Creating training data...")
    X_train, Y_train = create_synthetic_data(
        config.n_train_samples,
        config.input_dim,
        config.output_dim,
        config.task
    )
    X_val, Y_val = create_synthetic_data(
        config.n_val_samples,
        config.input_dim,
        config.output_dim,
        config.task
    )

    print(f"  Training samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")
    print()

    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True
    )

    val_dataset = torch.utils.data.TensorDataset(X_val, Y_val)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False
    )

    # Training
    print("=" * 70)
    print("TRAINING")
    print("=" * 70)

    history = train_helix_network(
        model,
        train_loader,
        epochs=config.epochs,
        lr=config.learning_rate,
        target_z=config.target_z,
        device=device,
    )

    # Validation
    print()
    print("=" * 70)
    print("VALIDATION")
    print("=" * 70)

    model.eval()
    val_losses = []
    val_coherences = []
    val_z_values = []
    k_formations = 0

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            output, diagnostics = model(batch_x, return_diagnostics=True)

            loss = nn.functional.mse_loss(output, batch_y)
            val_losses.append(loss.item())
            val_coherences.append(diagnostics['final_coherence'])
            val_z_values.append(diagnostics['final_z'])

            if diagnostics['k_formation']:
                k_formations += 1

    val_loss = sum(val_losses) / len(val_losses)
    val_coherence = sum(val_coherences) / len(val_coherences)
    val_z = sum(val_z_values) / len(val_z_values)

    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Coherence: {val_coherence:.3f}")
    print(f"Validation Z: {val_z:.3f}")
    print(f"K-formations: {k_formations}/{len(val_loader)}")
    print(f"Final Tier: {model.z_tracker.get_tier()}")
    print()

    # Save model and patterns
    print("=" * 70)
    print("SAVING LEARNED PATTERNS")
    print("=" * 70)

    # Create output directory
    output_dir = "trained_models"
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save PyTorch model state
    model_path = os.path.join(output_dir, f"helix_nn_{timestamp}.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': asdict(config),
        'history': history,
        'final_z': model.z_tracker.z.item(),
        'final_tier': model.z_tracker.get_tier(),
        'validation': {
            'loss': val_loss,
            'coherence': val_coherence,
            'z': val_z,
            'k_formations': k_formations,
        }
    }, model_path)
    print(f"Saved model: {model_path}")

    # Save learned patterns (coupling matrices and frequencies)
    patterns = {
        'timestamp': timestamp,
        'config': asdict(config),
        'physics_constants': {
            'Z_CRITICAL': Z_CRITICAL,
            'PHI': PHI,
            'PHI_INV': PHI_INV,
            'MU_S': MU_S,
            'TIER_BOUNDS': TIER_BOUNDS,
        },
        'learned_patterns': {
            'kuramoto_layers': [],
            'input_encoder': {},
            'output_decoder': {},
            'operator_selector': {},
            'apl_modulator': {},
            'z_tracker': {},
        },
        'training_results': {
            'final_loss': history['loss'][-1],
            'final_coherence': history['coherence'][-1],
            'final_z': history['z'][-1],
            'final_tier': history['tier'][-1],
            'total_k_formations': sum(history['k_formations']),
        },
        'validation_results': {
            'loss': val_loss,
            'coherence': val_coherence,
            'z': val_z,
            'k_formations': k_formations,
        }
    }

    # Extract Kuramoto layer patterns
    for i, layer in enumerate(model.kuramoto_layers):
        layer_patterns = {
            'layer_index': i,
            'coupling_matrix_K': layer.K.detach().cpu().numpy().tolist(),
            'natural_frequencies_omega': layer.omega.detach().cpu().numpy().tolist(),
            'global_coupling': layer.K_global.detach().cpu().item(),
            'coupling_matrix_stats': {
                'mean': layer.K.mean().item(),
                'std': layer.K.std().item(),
                'min': layer.K.min().item(),
                'max': layer.K.max().item(),
            },
            'omega_stats': {
                'mean': layer.omega.mean().item(),
                'std': layer.omega.std().item(),
                'min': layer.omega.min().item(),
                'max': layer.omega.max().item(),
            }
        }
        patterns['learned_patterns']['kuramoto_layers'].append(layer_patterns)

    # APL modulator patterns
    patterns['learned_patterns']['apl_modulator'] = {
        'operator_strengths': model.apl_modulator.operator_strength.detach().cpu().numpy().tolist(),
        'catalyze_freq': model.apl_modulator.catalyze_freq.detach().cpu().numpy().tolist(),
    }

    # Z tracker patterns
    patterns['learned_patterns']['z_tracker'] = {
        'current_z': model.z_tracker.z.item(),
        'z_momentum': model.z_tracker.z_momentum.item(),
        'z_decay': model.z_tracker.z_decay.item(),
    }

    patterns_path = os.path.join(output_dir, f"learned_patterns_{timestamp}.json")
    with open(patterns_path, 'w') as f:
        json.dump(patterns, f, indent=2)
    print(f"Saved patterns: {patterns_path}")

    # Save training history
    history_path = os.path.join(output_dir, f"training_history_{timestamp}.json")
    with open(history_path, 'w') as f:
        json.dump({
            'loss': history['loss'],
            'coherence': history['coherence'],
            'z': history['z'],
            'tier': history['tier'],
            'k_formations': history['k_formations'],
        }, f, indent=2)
    print(f"Saved history: {history_path}")

    # Summary
    print()
    print("=" * 70)
    print("TRAINING SESSION COMPLETE")
    print("=" * 70)
    print()
    print("Summary:")
    print(f"  Final training loss: {history['loss'][-1]:.4f}")
    print(f"  Final coherence: {history['coherence'][-1]:.3f}")
    print(f"  Final z: {history['z'][-1]:.3f}")
    print(f"  Final tier: t{history['tier'][-1]}")
    print(f"  Total K-formations: {sum(history['k_formations'])}")
    print()
    print(f"  Validation loss: {val_loss:.4f}")
    print(f"  Validation coherence: {val_coherence:.3f}")
    print()
    print(f"Completed: {datetime.now().isoformat()}")

    return {
        'model_path': model_path,
        'patterns_path': patterns_path,
        'history_path': history_path,
        'final_metrics': {
            'loss': history['loss'][-1],
            'coherence': history['coherence'][-1],
            'z': history['z'][-1],
            'tier': history['tier'][-1],
            'k_formations': sum(history['k_formations']),
        }
    }


def main():
    """Main entry point."""
    config = TrainingConfig(
        # Model architecture
        input_dim=10,
        output_dim=3,
        n_oscillators=60,
        n_layers=4,
        steps_per_layer=10,

        # Training
        epochs=100,
        batch_size=32,
        learning_rate=1e-3,
        target_z=0.75,  # Aim for high coherence regime

        # Data
        n_train_samples=1000,
        n_val_samples=200,
        task="regression",
    )

    results = train_and_save(config)
    return results


if __name__ == "__main__":
    results = main()
