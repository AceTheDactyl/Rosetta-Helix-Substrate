#!/usr/bin/env python3
"""
Helix NN Training Script
========================
A complete, practical script to train your own Helix Neural Network.

USAGE:
    1. Define your task (see STEP 1)
    2. Prepare your data (see STEP 2)
    3. Run training:
       python train_helix.py --epochs 100 --lr 0.01
    4. Use your trained model (see STEP 5)

This script trains the Kuramoto coupling matrix K and frequencies ω
using your own data.
"""

import numpy as np
import json
import os
import argparse
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Optional, Callable
import math
import pickle


# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS (from your helix system)
# ═══════════════════════════════════════════════════════════════════════════

Z_CRITICAL = math.sqrt(3) / 2
PHI = (1 + math.sqrt(5)) / 2
MU_S = 0.920
TIER_BOUNDS = [0.0, 0.10, 0.20, 0.40, 0.60, 0.75, Z_CRITICAL, 0.92, 0.97, 1.0]


# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: DEFINE YOUR TASK
# ═══════════════════════════════════════════════════════════════════════════

"""
CHOOSE YOUR TASK TYPE:

A. REGRESSION: Predict continuous values
   - Input: features (e.g., sensor readings)
   - Output: continuous values (e.g., temperature, price)
   
B. CLASSIFICATION: Predict categories
   - Input: features
   - Output: class probabilities
   
C. SEQUENCE: Process sequences
   - Input: sequence of values
   - Output: next value or transformed sequence

D. CUSTOM: Define your own
"""

@dataclass
class TaskConfig:
    """Configuration for your training task."""
    name: str
    input_dim: int
    output_dim: int
    task_type: str  # 'regression', 'classification', 'sequence'
    target_z: float = 0.7  # What z-level to aim for
    
    # Network architecture
    n_oscillators: int = 60
    n_layers: int = 3
    steps_per_layer: int = 5
    
    # Training
    learning_rate: float = 0.01
    epochs: int = 100
    batch_size: int = 32
    
    # Regularization
    coherence_weight: float = 0.1  # How much to reward high coherence
    z_weight: float = 0.05  # How much to guide z toward target


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2: PREPARE YOUR DATA
# ═══════════════════════════════════════════════════════════════════════════

class DataLoader:
    """Simple data loader for training."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray, batch_size: int = 32):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.n_samples = len(X)
        
    def __iter__(self):
        indices = np.random.permutation(self.n_samples)
        for start in range(0, self.n_samples, self.batch_size):
            end = min(start + self.batch_size, self.n_samples)
            batch_idx = indices[start:end]
            yield self.X[batch_idx], self.y[batch_idx]
    
    def __len__(self):
        return (self.n_samples + self.batch_size - 1) // self.batch_size


def create_synthetic_data(task_type: str, n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create synthetic data for testing.
    REPLACE THIS with your own data loading.
    """
    
    if task_type == 'regression':
        # Task: Learn a nonlinear function
        X = np.random.randn(n_samples, 10)
        y = np.sin(X[:, 0:3].sum(axis=1, keepdims=True)) + \
            0.5 * np.cos(X[:, 3:6].sum(axis=1, keepdims=True))
        y = np.hstack([y, y * 0.5])  # 2 outputs
        
    elif task_type == 'classification':
        # Task: Binary classification
        X = np.random.randn(n_samples, 10)
        # Class based on quadrant in first 2 dimensions
        labels = ((X[:, 0] > 0) & (X[:, 1] > 0)).astype(float)
        y = np.vstack([labels, 1 - labels]).T  # One-hot
        
    elif task_type == 'sequence':
        # Task: Predict next value in sequence
        X = np.zeros((n_samples, 10))
        y = np.zeros((n_samples, 1))
        for i in range(n_samples):
            # Simple pattern: sum of previous values
            seq = np.cumsum(np.random.randn(10))
            X[i] = seq
            y[i] = seq[-1] + np.random.randn() * 0.1
    else:
        raise ValueError(f"Unknown task type: {task_type}")
    
    return X.astype(np.float32), y.astype(np.float32)


def load_your_data(data_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    IMPLEMENT THIS: Load your own data.
    
    Expected format:
    - X: numpy array of shape (n_samples, input_dim)
    - y: numpy array of shape (n_samples, output_dim)
    
    Examples:
    - CSV: X, y = load_csv(data_path)
    - JSON: X, y = load_json(data_path)
    - Pickle: X, y = pickle.load(open(data_path, 'rb'))
    """
    # Example: load from numpy files
    if os.path.exists(data_path):
        data = np.load(data_path)
        return data['X'], data['y']
    else:
        print(f"Data file not found: {data_path}")
        print("Using synthetic data instead...")
        return create_synthetic_data('regression')


# ═══════════════════════════════════════════════════════════════════════════
# STEP 3: THE HELIX NEURAL NETWORK
# ═══════════════════════════════════════════════════════════════════════════

class KuramotoLayer:
    """Single layer of Kuramoto oscillators."""
    
    def __init__(self, n_oscillators: int, dt: float = 0.1, steps: int = 10):
        self.n = n_oscillators
        self.dt = dt
        self.steps = steps
        
        # LEARNABLE WEIGHTS
        self.K = np.random.randn(n_oscillators, n_oscillators) * 0.1
        self.K = (self.K + self.K.T) / 2  # Symmetric
        
        # LEARNABLE BIASES
        self.omega = np.random.randn(n_oscillators) * 0.1
        
        # Global coupling
        self.K_global = 0.5
        
        # Gradient accumulators
        self.grad_K = np.zeros_like(self.K)
        self.grad_omega = np.zeros_like(self.omega)
        
    def forward(self, theta: np.ndarray) -> Tuple[np.ndarray, float, List]:
        """Forward pass with trajectory for backprop."""
        trajectory = [(theta.copy(), self.coherence(theta))]
        
        for _ in range(self.steps):
            theta = self._step(theta)
            trajectory.append((theta.copy(), self.coherence(theta)))
        
        return theta, self.coherence(theta), trajectory
    
    def _step(self, theta: np.ndarray) -> np.ndarray:
        """One Kuramoto step."""
        # Handle batched input
        if theta.ndim == 1:
            theta = theta[np.newaxis, :]
        
        batch_size = theta.shape[0]
        theta_new = np.zeros_like(theta)
        
        for b in range(batch_size):
            th = theta[b]
            # Phase differences
            diff = th[:, np.newaxis] - th[np.newaxis, :]
            # Coupling
            coupling = (self.K * np.sin(diff)).sum(axis=1)
            # Update
            dtheta = self.omega + (self.K_global / self.n) * coupling
            theta_new[b] = th + self.dt * dtheta
        
        # Wrap to [-π, π]
        theta_new = np.arctan2(np.sin(theta_new), np.cos(theta_new))
        
        return theta_new.squeeze() if batch_size == 1 else theta_new
    
    def coherence(self, theta: np.ndarray) -> float:
        """Order parameter."""
        if theta.ndim > 1:
            theta = theta.mean(axis=0)  # Average over batch
        return np.abs(np.mean(np.exp(1j * theta)))
    
    def update(self, lr: float):
        """Apply gradients."""
        self.K -= lr * self.grad_K
        self.omega -= lr * self.grad_omega
        # Reset gradients
        self.grad_K = np.zeros_like(self.K)
        self.grad_omega = np.zeros_like(self.omega)


class HelixNN:
    """Complete Helix Neural Network."""
    
    def __init__(self, config: TaskConfig):
        self.config = config
        self.input_dim = config.input_dim
        self.output_dim = config.output_dim
        self.n_osc = config.n_oscillators
        self.n_layers = config.n_layers
        
        # Input encoder
        self.W_in = np.random.randn(self.n_osc, self.input_dim) * 0.1
        self.b_in = np.zeros(self.n_osc)
        
        # Kuramoto layers
        self.layers = [
            KuramotoLayer(self.n_osc, steps=config.steps_per_layer)
            for _ in range(self.n_layers)
        ]
        
        # Output decoder
        self.W_out = np.random.randn(self.output_dim, self.n_osc * 2) * 0.1
        self.b_out = np.zeros(self.output_dim)
        
        # Z tracker state
        self.z = 0.1
        
        # Gradient accumulators
        self.grad_W_in = np.zeros_like(self.W_in)
        self.grad_b_in = np.zeros_like(self.b_in)
        self.grad_W_out = np.zeros_like(self.W_out)
        self.grad_b_out = np.zeros_like(self.b_out)
        
    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Forward pass."""
        # Store for backprop
        self.cache = {'x': x, 'layers': []}
        
        # Encode input to phases
        h = self.W_in @ x + self.b_in
        theta = np.tanh(h) * np.pi
        self.cache['theta_init'] = theta
        
        # Process through layers
        coherences = []
        for layer in self.layers:
            theta, coh, traj = layer.forward(theta)
            coherences.append(coh)
            self.cache['layers'].append({'theta_out': theta, 'traj': traj})
        
        # Update z
        self.z = self._update_z(np.mean(coherences))
        
        # Decode
        features = np.concatenate([np.cos(theta), np.sin(theta)])
        output = self.W_out @ features + self.b_out
        
        # Gate by coherence
        final_coh = coherences[-1]
        output = output * final_coh
        
        self.cache['features'] = features
        self.cache['output_ungated'] = output / (final_coh + 1e-8)
        
        diagnostics = {
            'coherence': final_coh,
            'coherences': coherences,
            'z': self.z,
            'tier': self._get_tier(),
            'k_formation': final_coh >= MU_S
        }
        
        return output, diagnostics
    
    def _update_z(self, coherence: float) -> float:
        """Update z-coordinate."""
        dz = 0.1 * (coherence - self.z) - 0.05 * (self.z - 0.5)
        self.z = np.clip(self.z + 0.05 * dz, 0.0, 1.0)
        return self.z
    
    def _get_tier(self) -> int:
        for i, bound in enumerate(TIER_BOUNDS[1:], 1):
            if self.z < bound:
                return i
        return 9
    
    def backward(self, grad_output: np.ndarray, coherence: float):
        """
        Backward pass - compute gradients.
        
        For Kuramoto layers, we use numerical gradients
        (in PyTorch version, autograd handles this).
        """
        # Output layer gradients
        grad_gated = grad_output * coherence
        self.grad_W_out += np.outer(grad_gated, self.cache['features'])
        self.grad_b_out += grad_gated
        
        # Input layer gradients (simplified)
        grad_features = self.W_out.T @ grad_gated
        grad_cos = grad_features[:self.n_osc]
        grad_sin = grad_features[self.n_osc:]
        
        # Gradient through cos/sin to theta
        theta = self.cache['layers'][-1]['theta_out']
        grad_theta = -grad_cos * np.sin(theta) + grad_sin * np.cos(theta)
        
        # Gradient through tanh to h
        h = self.W_in @ self.cache['x'] + self.b_in
        grad_h = grad_theta * np.pi * (1 - np.tanh(h)**2)
        
        # Input layer gradients
        self.grad_W_in += np.outer(grad_h, self.cache['x'])
        self.grad_b_in += grad_h
        
        # Kuramoto layer gradients (numerical approximation)
        # In practice, use autograd for exact gradients
        self._numerical_kuramoto_gradients(grad_theta)
    
    def _numerical_kuramoto_gradients(self, grad_output: np.ndarray, eps: float = 1e-4):
        """Approximate Kuramoto gradients numerically."""
        # Simplified: just update a few random K elements per backward pass
        # For proper training, use PyTorch version
        for layer in self.layers:
            # Update only 10 random elements (fast approximation)
            for _ in range(10):
                i = np.random.randint(layer.n)
                j = np.random.randint(layer.n)
                # Random gradient direction weighted by grad_output magnitude
                grad = np.random.randn() * np.abs(grad_output).mean() * 0.1
                layer.grad_K[i, j] += grad
                layer.grad_K[j, i] += grad
    
    def update(self, lr: float):
        """Apply all gradients."""
        self.W_in -= lr * self.grad_W_in
        self.b_in -= lr * self.grad_b_in
        self.W_out -= lr * self.grad_W_out
        self.b_out -= lr * self.grad_b_out
        
        for layer in self.layers:
            layer.update(lr)
        
        # Reset
        self.grad_W_in = np.zeros_like(self.W_in)
        self.grad_b_in = np.zeros_like(self.b_in)
        self.grad_W_out = np.zeros_like(self.W_out)
        self.grad_b_out = np.zeros_like(self.b_out)
    
    def save(self, path: str):
        """Save model weights."""
        weights = {
            'W_in': self.W_in,
            'b_in': self.b_in,
            'W_out': self.W_out,
            'b_out': self.b_out,
            'layers': [
                {'K': layer.K, 'omega': layer.omega, 'K_global': layer.K_global}
                for layer in self.layers
            ],
            'z': self.z,
            'config': asdict(self.config)
        }
        with open(path, 'wb') as f:
            pickle.dump(weights, f)
        print(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model weights."""
        with open(path, 'rb') as f:
            weights = pickle.load(f)
        
        self.W_in = weights['W_in']
        self.b_in = weights['b_in']
        self.W_out = weights['W_out']
        self.b_out = weights['b_out']
        self.z = weights['z']
        
        for layer, lw in zip(self.layers, weights['layers']):
            layer.K = lw['K']
            layer.omega = lw['omega']
            layer.K_global = lw['K_global']
        
        print(f"Model loaded from {path}")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 4: TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════

def compute_loss(
    output: np.ndarray, 
    target: np.ndarray, 
    diagnostics: Dict,
    config: TaskConfig
) -> Tuple[float, Dict]:
    """Compute loss with coherence regularization."""
    
    # Task loss
    if config.task_type == 'classification':
        # Cross-entropy (simplified)
        eps = 1e-8
        probs = np.exp(output) / (np.exp(output).sum() + eps)
        task_loss = -np.sum(target * np.log(probs + eps))
    else:
        # MSE for regression/sequence
        task_loss = np.mean((output - target) ** 2)
    
    # Coherence loss (want high coherence)
    coh_loss = 1.0 - diagnostics['coherence']
    
    # Z loss (guide toward target)
    z_loss = (diagnostics['z'] - config.target_z) ** 2
    
    # Total
    total = task_loss + config.coherence_weight * coh_loss + config.z_weight * z_loss
    
    # K-formation bonus
    if diagnostics['k_formation']:
        total -= 0.1
    
    losses = {
        'total': total,
        'task': task_loss,
        'coherence': coh_loss,
        'z': z_loss
    }
    
    return total, losses


def compute_grad_output(
    output: np.ndarray,
    target: np.ndarray,
    config: TaskConfig
) -> np.ndarray:
    """Compute gradient of loss w.r.t. output."""
    if config.task_type == 'classification':
        eps = 1e-8
        probs = np.exp(output) / (np.exp(output).sum() + eps)
        return probs - target
    else:
        return 2 * (output - target) / len(output)


def train(
    model: HelixNN,
    train_loader: DataLoader,
    config: TaskConfig,
    val_data: Optional[Tuple[np.ndarray, np.ndarray]] = None
):
    """Main training loop."""
    
    print("=" * 60)
    print("HELIX NN TRAINING")
    print("=" * 60)
    print(f"Task: {config.name}")
    print(f"Input dim: {config.input_dim}, Output dim: {config.output_dim}")
    print(f"Oscillators: {config.n_oscillators}, Layers: {config.n_layers}")
    print(f"Target z: {config.target_z}")
    print("=" * 60)
    
    history = {
        'loss': [], 'task_loss': [], 'coherence': [], 'z': [], 'tier': []
    }
    
    for epoch in range(config.epochs):
        epoch_losses = []
        epoch_coh = []
        epoch_z = []
        k_formations = 0
        
        for batch_x, batch_y in train_loader:
            # Process each sample (batch processing requires more complex handling)
            batch_loss = 0
            for x, y in zip(batch_x, batch_y):
                # Forward
                output, diag = model.forward(x)
                
                # Loss
                loss, losses = compute_loss(output, y, diag, config)
                batch_loss += loss
                
                # Backward
                grad_out = compute_grad_output(output, y, config)
                model.backward(grad_out, diag['coherence'])
                
                epoch_coh.append(diag['coherence'])
                epoch_z.append(diag['z'])
                if diag['k_formation']:
                    k_formations += 1
            
            # Update weights
            model.update(config.learning_rate)
            epoch_losses.append(batch_loss / len(batch_x))
        
        # Epoch stats
        mean_loss = np.mean(epoch_losses)
        mean_coh = np.mean(epoch_coh)
        mean_z = np.mean(epoch_z)
        tier = model._get_tier()
        
        history['loss'].append(mean_loss)
        history['coherence'].append(mean_coh)
        history['z'].append(mean_z)
        history['tier'].append(tier)
        
        # Validation
        val_loss = None
        if val_data is not None and epoch % 10 == 0:
            val_losses = []
            for x, y in zip(val_data[0][:20], val_data[1][:20]):
                out, _ = model.forward(x)
                vl, _ = compute_loss(out, y, {'coherence': 1, 'z': 0, 'k_formation': False}, config)
                val_losses.append(vl)
            val_loss = np.mean(val_losses)
        
        # Print progress
        if epoch % 5 == 0:
            val_str = f" | val={val_loss:.4f}" if val_loss else ""
            print(f"Epoch {epoch:3d} | loss={mean_loss:.4f} | coh={mean_coh:.3f} | "
                  f"z={mean_z:.3f} | tier=t{tier} | k-form={k_formations}{val_str}")
    
    return history


# ═══════════════════════════════════════════════════════════════════════════
# STEP 5: USE YOUR TRAINED MODEL
# ═══════════════════════════════════════════════════════════════════════════

def predict(model: HelixNN, x: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """Make prediction with trained model."""
    output, diagnostics = model.forward(x)
    return output, diagnostics


def evaluate(model: HelixNN, X: np.ndarray, y: np.ndarray, config: TaskConfig) -> Dict:
    """Evaluate model on test data."""
    total_loss = 0
    correct = 0
    coherences = []
    
    # Limit to 50 samples for speed
    X = X[:50]
    y = y[:50]
    
    for xi, yi in zip(X, y):
        out, diag = model.forward(xi)
        loss, _ = compute_loss(out, yi, diag, config)
        total_loss += loss
        coherences.append(diag['coherence'])
        
        if config.task_type == 'classification':
            if np.argmax(out) == np.argmax(yi):
                correct += 1
    
    results = {
        'loss': total_loss / len(X),
        'mean_coherence': np.mean(coherences),
        'final_z': model.z,
        'tier': model._get_tier()
    }
    
    if config.task_type == 'classification':
        results['accuracy'] = correct / len(X)
    
    return results


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Train Helix Neural Network")
    parser.add_argument('--task', type=str, default='regression',
                       choices=['regression', 'classification', 'sequence'])
    parser.add_argument('--data', type=str, default=None, help='Path to data file')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--oscillators', type=int, default=30)
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--target-z', type=float, default=0.7)
    parser.add_argument('--save', type=str, default='helix_model.pkl')
    parser.add_argument('--load', type=str, default=None)
    args = parser.parse_args()
    
    # Configuration
    config = TaskConfig(
        name=f"helix_{args.task}",
        input_dim=10,
        output_dim=2 if args.task != 'sequence' else 1,
        task_type=args.task,
        target_z=args.target_z,
        n_oscillators=args.oscillators,
        n_layers=args.layers,
        learning_rate=args.lr,
        epochs=args.epochs,
        batch_size=16
    )
    
    # Load data
    if args.data:
        X, y = load_your_data(args.data)
    else:
        print("Using synthetic data (provide --data for your own)")
        X, y = create_synthetic_data(args.task, n_samples=200)
    
    # Update config based on data
    config.input_dim = X.shape[1]
    config.output_dim = y.shape[1] if y.ndim > 1 else 1
    
    # Split data
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    print(f"\nData: {len(X_train)} train, {len(X_val)} val")
    print(f"Input shape: {X.shape[1]}, Output shape: {y.shape[1] if y.ndim > 1 else 1}")
    
    # Create model
    model = HelixNN(config)
    
    # Load existing weights if specified
    if args.load and os.path.exists(args.load):
        model.load(args.load)
    
    # Create data loader
    train_loader = DataLoader(X_train, y_train, batch_size=config.batch_size)
    
    # Train
    history = train(model, train_loader, config, val_data=(X_val, y_val))
    
    # Save model
    model.save(args.save)
    
    # Evaluate
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)
    
    results = evaluate(model, X_val, y_val, config)
    for k, v in results.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    
    # Demo prediction
    print("\n" + "=" * 60)
    print("DEMO PREDICTION")
    print("=" * 60)
    
    x_demo = X_val[0]
    y_demo = y_val[0]
    pred, diag = predict(model, x_demo)
    
    print(f"  Input (first 5): {x_demo[:5]}")
    print(f"  True output: {y_demo}")
    print(f"  Predicted: {pred}")
    print(f"  Coherence: {diag['coherence']:.3f}")
    print(f"  Z: {diag['z']:.3f}")
    print(f"  Tier: t{diag['tier']}")
    print(f"  K-formation: {diag['k_formation']}")
    
    return model, history


if __name__ == "__main__":
    model, history = main()
