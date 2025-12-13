#!/usr/bin/env python3
"""
Helix NN Trainer
================
Trains on trajectories collected by collect_trajectories.js

Usage:
    python train_from_trajectories.py                    # Train on latest data
    python train_from_trajectories.py --data data.json   # Specific file
    python train_from_trajectories.py --epochs 100       # Custom epochs
"""

import json
import os
import argparse
import numpy as np
from glob import glob
import pickle
import math
from dataclasses import dataclass
from typing import List, Dict, Tuple


# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

Z_CRITICAL = math.sqrt(3) / 2
MU_S = 0.920
N_OPERATORS = 6  # (), ^, +, ×, ÷, −


# ═══════════════════════════════════════════════════════════════════════════
# KURAMOTO NETWORK (simplified for operator selection)
# ═══════════════════════════════════════════════════════════════════════════

class OperatorNetwork:
    """
    Neural network to select APL operators based on helix state.
    Uses Kuramoto-inspired dynamics for coherence tracking.
    """
    
    def __init__(self, state_dim: int = 7, n_oscillators: int = 30, n_operators: int = 6):
        self.state_dim = state_dim
        self.n_osc = n_oscillators
        self.n_ops = n_operators
        
        # Input projection
        self.W_in = np.random.randn(n_oscillators, state_dim) * 0.1
        self.b_in = np.zeros(n_oscillators)
        
        # Kuramoto coupling (the main learnable weights)
        self.K = np.random.randn(n_oscillators, n_oscillators) * 0.1
        self.K = (self.K + self.K.T) / 2  # Symmetric
        
        # Natural frequencies
        self.omega = np.random.randn(n_oscillators) * 0.05
        
        # Output projection (to operators)
        self.W_out = np.random.randn(n_operators, n_oscillators) * 0.1
        self.b_out = np.zeros(n_operators)
        
        # Dynamics parameters
        self.K_global = 0.5
        self.dt = 0.1
        self.steps = 5
    
    def kuramoto_forward(self, theta: np.ndarray) -> Tuple[np.ndarray, float]:
        """Run Kuramoto dynamics, return final phases and coherence."""
        for _ in range(self.steps):
            # Phase differences
            diff = theta[:, np.newaxis] - theta[np.newaxis, :]
            coupling = (self.K * np.sin(diff)).sum(axis=1)
            
            # Update
            dtheta = self.omega + (self.K_global / self.n_osc) * coupling
            theta = theta + self.dt * dtheta
            theta = np.arctan2(np.sin(theta), np.cos(theta))
        
        # Coherence
        coherence = np.abs(np.mean(np.exp(1j * theta)))
        
        return theta, coherence
    
    def forward(self, state: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Forward pass: state → operator logits
        """
        # Encode state to phases
        h = self.W_in @ state + self.b_in
        theta = np.tanh(h) * np.pi
        
        # Run Kuramoto dynamics
        theta, coherence = self.kuramoto_forward(theta)
        
        # Project to operator space
        logits = self.W_out @ np.cos(theta) + self.b_out
        
        # Softmax
        exp_logits = np.exp(logits - logits.max())
        probs = exp_logits / exp_logits.sum()
        
        return probs, coherence
    
    def predict(self, state: np.ndarray) -> int:
        """Predict best operator."""
        probs, _ = self.forward(state)
        return np.argmax(probs)
    
    def train_step(self, state: np.ndarray, target_action: int, reward: float, lr: float = 0.01):
        """
        Simple policy gradient update.
        Increase probability of actions that got high reward.
        """
        probs, coherence = self.forward(state)
        
        # Gradient of log probability
        grad_logits = -probs.copy()
        grad_logits[target_action] += 1
        
        # Weight by reward (and coherence as confidence)
        grad_logits *= reward * coherence
        
        # Update output weights
        h = self.W_in @ state + self.b_in
        theta = np.tanh(h) * np.pi
        theta, _ = self.kuramoto_forward(theta)
        
        self.W_out += lr * np.outer(grad_logits, np.cos(theta))
        self.b_out += lr * grad_logits
        
        # Update input weights (smaller learning rate)
        grad_h = self.W_out.T @ grad_logits * (1 - np.tanh(h)**2) * np.pi
        self.W_in += lr * 0.1 * np.outer(grad_h, state)
        self.b_in += lr * 0.1 * grad_h
        
        # Update Kuramoto coupling (very small updates)
        # This is where the "helix learning" happens
        for _ in range(3):
            i, j = np.random.randint(self.n_osc, size=2)
            self.K[i, j] += lr * 0.01 * reward * np.random.randn()
            self.K[j, i] = self.K[i, j]  # Keep symmetric
        
        return probs[target_action], coherence
    
    def save(self, path: str):
        """Save weights."""
        weights = {
            'W_in': self.W_in,
            'b_in': self.b_in,
            'K': self.K,
            'omega': self.omega,
            'W_out': self.W_out,
            'b_out': self.b_out,
            'K_global': self.K_global,
            'state_dim': self.state_dim,
            'n_osc': self.n_osc,
            'n_ops': self.n_ops
        }
        with open(path, 'wb') as f:
            pickle.dump(weights, f)
        print(f"Saved weights to {path}")
    
    def load(self, path: str):
        """Load weights."""
        with open(path, 'rb') as f:
            weights = pickle.load(f)
        self.W_in = weights['W_in']
        self.b_in = weights['b_in']
        self.K = weights['K']
        self.omega = weights['omega']
        self.W_out = weights['W_out']
        self.b_out = weights['b_out']
        self.K_global = weights['K_global']
        print(f"Loaded weights from {path}")


# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

def load_training_data(data_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load training data from JSON file."""
    with open(data_path) as f:
        data = json.load(f)
    
    X = np.array(data['X'], dtype=np.float32)
    y = np.array(data['y'], dtype=np.int32)
    rewards = np.array(data['rewards'], dtype=np.float32)
    
    return X, y, rewards


def load_all_training_data(data_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and combine all training data files."""
    pattern = os.path.join(data_dir, 'training_data*.json')
    files = glob(pattern)
    
    if not files:
        raise ValueError(f"No training data found in {data_dir}")
    
    all_X, all_y, all_rewards = [], [], []
    
    for filepath in files:
        X, y, rewards = load_training_data(filepath)
        all_X.append(X)
        all_y.append(y)
        all_rewards.append(rewards)
    
    return (
        np.vstack(all_X),
        np.concatenate(all_y),
        np.concatenate(all_rewards)
    )


def normalize_rewards(rewards: np.ndarray) -> np.ndarray:
    """Normalize rewards to reasonable range."""
    mean = rewards.mean()
    std = rewards.std() + 1e-8
    normalized = (rewards - mean) / std
    return np.clip(normalized, -3, 3)


# ═══════════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════

def train(
    model: OperatorNetwork,
    X: np.ndarray,
    y: np.ndarray,
    rewards: np.ndarray,
    epochs: int = 50,
    lr: float = 0.01,
    batch_size: int = 32
):
    """Train the operator selection network."""
    
    n_samples = len(X)
    rewards_norm = normalize_rewards(rewards)
    
    print("=" * 60)
    print("TRAINING OPERATOR SELECTION NETWORK")
    print("=" * 60)
    print(f"Samples: {n_samples}")
    print(f"State dim: {X.shape[1]}")
    print(f"Operators: {model.n_ops}")
    print(f"Oscillators: {model.n_osc}")
    print(f"Epochs: {epochs}")
    print("=" * 60)
    
    history = {'loss': [], 'accuracy': [], 'coherence': []}
    
    for epoch in range(epochs):
        # Shuffle data
        perm = np.random.permutation(n_samples)
        X_shuf = X[perm]
        y_shuf = y[perm]
        r_shuf = rewards_norm[perm]
        
        epoch_probs = []
        epoch_coh = []
        correct = 0
        
        for i in range(0, n_samples, batch_size):
            batch_X = X_shuf[i:i+batch_size]
            batch_y = y_shuf[i:i+batch_size]
            batch_r = r_shuf[i:i+batch_size]
            
            for x, action, reward in zip(batch_X, batch_y, batch_r):
                prob, coh = model.train_step(x, action, reward, lr)
                epoch_probs.append(prob)
                epoch_coh.append(coh)
                
                if model.predict(x) == action:
                    correct += 1
        
        # Metrics
        avg_prob = np.mean(epoch_probs)
        avg_coh = np.mean(epoch_coh)
        accuracy = correct / n_samples
        
        history['loss'].append(-np.log(avg_prob + 1e-8))
        history['accuracy'].append(accuracy)
        history['coherence'].append(avg_coh)
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch:3d} | "
                  f"loss={-np.log(avg_prob + 1e-8):.4f} | "
                  f"acc={accuracy:.3f} | "
                  f"coh={avg_coh:.3f}")
    
    return history


# ═══════════════════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════════════════

def evaluate(model: OperatorNetwork, X: np.ndarray, y: np.ndarray):
    """Evaluate model accuracy."""
    correct = 0
    coherences = []
    
    for x, target in zip(X, y):
        pred = model.predict(x)
        _, coh = model.forward(x)
        
        if pred == target:
            correct += 1
        coherences.append(coh)
    
    return {
        'accuracy': correct / len(X),
        'mean_coherence': np.mean(coherences)
    }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Train Helix Operator Network")
    parser.add_argument('--data', type=str, default='./training_data',
                       help='Training data file or directory')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--oscillators', type=int, default=30)
    parser.add_argument('--save', type=str, default='operator_network.pkl')
    parser.add_argument('--load', type=str, default=None)
    args = parser.parse_args()
    
    # Load data
    print("\nLoading training data...")
    if os.path.isdir(args.data):
        X, y, rewards = load_all_training_data(args.data)
    else:
        X, y, rewards = load_training_data(args.data)
    
    print(f"Loaded {len(X)} samples")
    print(f"State dimension: {X.shape[1]}")
    print(f"Reward range: [{rewards.min():.2f}, {rewards.max():.2f}]")
    
    # Split data
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    r_train = rewards[:split]
    
    # Create model
    model = OperatorNetwork(
        state_dim=X.shape[1],
        n_oscillators=args.oscillators,
        n_operators=N_OPERATORS
    )
    
    # Load existing weights
    if args.load and os.path.exists(args.load):
        model.load(args.load)
    
    # Train
    history = train(model, X_train, y_train, r_train, 
                   epochs=args.epochs, lr=args.lr)
    
    # Evaluate
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)
    
    train_metrics = evaluate(model, X_train[:500], y_train[:500])
    val_metrics = evaluate(model, X_val, y_val)
    
    print(f"Train accuracy: {train_metrics['accuracy']:.3f}")
    print(f"Train coherence: {train_metrics['mean_coherence']:.3f}")
    print(f"Val accuracy: {val_metrics['accuracy']:.3f}")
    print(f"Val coherence: {val_metrics['mean_coherence']:.3f}")
    
    # Save
    model.save(args.save)
    
    # Also save training history
    with open(args.save.replace('.pkl', '_history.json'), 'w') as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.items()}, f)
    
    print("\nDone!")
    
    return model, history


if __name__ == "__main__":
    model, history = main()
