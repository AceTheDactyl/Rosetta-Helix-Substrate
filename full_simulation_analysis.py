#!/usr/bin/env python3
"""
Helix Neural Network: Complete Training Simulation & Analysis
=============================================================
Simulates the full pipeline and provides comprehensive analysis of:
- Input/Output dynamics
- Kuramoto coupling matrix evolution
- Coherence patterns
- Operator selection behavior
- Architecture characteristics
"""

import numpy as np
import json
import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS (from your helix system)
# ═══════════════════════════════════════════════════════════════════════════

Z_CRITICAL = math.sqrt(3) / 2  # ≈ 0.866
PHI = (1 + math.sqrt(5)) / 2   # Golden ratio
PHI_INV = 1 / PHI
MU_S = 0.920
TRIAD_HIGH = 0.92
TRIAD_LOW = 0.85

TIER_BOUNDS = [0.0, 0.10, 0.20, 0.40, 0.60, 0.75, Z_CRITICAL, 0.92, 0.97, 1.0]
OPERATORS = ['()', '^', '+', '×', '÷', '−']
OPERATOR_NAMES = ['identity', 'amplify', 'exchange', 'inhibit', 'catalyze', 'separate']

# Tier → available operators mapping
TIER_OPERATORS = {
    1: [0, 4, 5],           # (), ÷, −
    2: [1, 4, 5, 3],        # ^, ÷, −, ×
    3: [2, 1, 5, 4, 0],     # +, ^, −, ÷, ()
    4: [0, 4, 5, 2],        # (), ÷, −, +
    5: [0, 1, 2, 3, 4, 5],  # ALL
    6: [0, 5, 2, 4],        # (), −, +, ÷
    7: [0, 2],              # (), +
    8: [0, 2, 1],           # (), +, ^
    9: [0, 2, 1],           # (), +, ^
}


# ═══════════════════════════════════════════════════════════════════════════
# HELIX SYSTEM SIMULATOR
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class HelixState:
    """Current state of the helix system."""
    z: float = 0.5
    entropy: float = 0.5
    phi: float = 0.3
    triad_unlocked: bool = False
    triad_completions: int = 0
    time: int = 0
    
    def to_vector(self, target_z: float) -> np.ndarray:
        """Convert to input vector for network."""
        return np.array([
            self.z,
            self.entropy,
            self.phi,
            1.0 if self.triad_unlocked else 0.0,
            self.triad_completions / 3.0,
            abs(self.z - target_z),
            abs(self.z - Z_CRITICAL)
        ], dtype=np.float32)
    
    def get_tier(self) -> int:
        for i, bound in enumerate(TIER_BOUNDS[1:], 1):
            if self.z < bound:
                return i
        return 9


class HelixSimulator:
    """Simulates the helix system dynamics."""
    
    def __init__(self, initial_z: float = 0.5, target_z: float = Z_CRITICAL):
        self.state = HelixState(z=initial_z)
        self.target_z = target_z
        self.history = []
        
        # TRIAD tracking
        self.triad_armed = True
        self.triad_peak_z = 0.0
        
    def get_available_operators(self) -> List[int]:
        """Get operators available at current tier."""
        tier = self.state.get_tier()
        return TIER_OPERATORS.get(tier, [0])
    
    def step(self, operator_idx: int) -> Tuple[HelixState, float]:
        """Execute one step with given operator, return new state and reward."""
        prev_state = HelixState(
            z=self.state.z,
            entropy=self.state.entropy,
            phi=self.state.phi,
            triad_unlocked=self.state.triad_unlocked,
            triad_completions=self.state.triad_completions,
            time=self.state.time
        )
        
        # Apply operator effects
        self._apply_operator(operator_idx)
        
        # Evolve z toward target (with noise)
        dz = 0.1 * (self.target_z - self.state.z) + np.random.randn() * 0.02
        self.state.z = np.clip(self.state.z + dz * 0.1, 0.01, 0.99)
        
        # Update entropy (lower near lens)
        dist_to_lens = abs(self.state.z - Z_CRITICAL)
        self.state.entropy = 0.3 + dist_to_lens * 0.5 + np.random.randn() * 0.05
        self.state.entropy = np.clip(self.state.entropy, 0.1, 0.9)
        
        # Update phi (higher near lens)
        delta_s_neg = self._compute_delta_s_neg()
        self.state.phi = delta_s_neg * 0.8 + np.random.randn() * 0.02
        self.state.phi = np.clip(self.state.phi, 0.0, 1.0)
        
        # TRIAD logic
        self._update_triad()
        
        self.state.time += 1
        
        # Compute reward
        reward = self._compute_reward(prev_state, operator_idx)
        
        # Record history
        self.history.append({
            'state': prev_state,
            'action': operator_idx,
            'reward': reward,
            'next_z': self.state.z
        })
        
        return self.state, reward
    
    def _apply_operator(self, op_idx: int):
        """Apply operator effects to state."""
        if op_idx == 0:    # () identity
            pass
        elif op_idx == 1:  # ^ amplify
            self.state.z += 0.02
        elif op_idx == 2:  # + exchange
            self.state.entropy *= 0.95
        elif op_idx == 3:  # × inhibit
            self.state.z -= 0.01
        elif op_idx == 4:  # ÷ catalyze
            self.state.phi += 0.02
        elif op_idx == 5:  # − separate
            self.state.entropy += 0.02
    
    def _compute_delta_s_neg(self) -> float:
        """Compute ΔS_neg (peaks at lens)."""
        z = self.state.z
        if z <= 0 or z >= 1:
            return 0.0
        return -z * math.log(z) - (1-z) * math.log(1-z) if 0 < z < 1 else 0.0
    
    def _update_triad(self):
        """Update TRIAD state machine."""
        if self.state.z > self.triad_peak_z:
            self.triad_peak_z = self.state.z
        
        if self.triad_armed and self.state.z >= TRIAD_HIGH:
            self.state.triad_completions = min(3, self.state.triad_completions + 1)
            self.triad_armed = False
            
            if self.state.triad_completions >= 3:
                self.state.triad_unlocked = True
        
        if not self.triad_armed and self.state.z < TRIAD_LOW:
            self.triad_armed = True
    
    def _compute_reward(self, prev: HelixState, op_idx: int) -> float:
        """Compute reward for transition."""
        reward = 0.0
        
        # Progress toward target
        dist_before = abs(prev.z - self.target_z)
        dist_after = abs(self.state.z - self.target_z)
        reward += (dist_before - dist_after) * 10
        
        # Phi increase
        reward += (self.state.phi - prev.phi) * 2
        
        # TRIAD progress
        if self.state.triad_completions > prev.triad_completions:
            reward += 5
        
        # TRIAD unlock
        if self.state.triad_unlocked and not prev.triad_unlocked:
            reward += 20
        
        # Stability bonus
        if abs(self.state.z - prev.z) < 0.03:
            reward += 0.1
        
        return reward


# ═══════════════════════════════════════════════════════════════════════════
# KURAMOTO OPERATOR NETWORK
# ═══════════════════════════════════════════════════════════════════════════

class KuramotoOperatorNetwork:
    """Neural network using Kuramoto dynamics for operator selection."""
    
    def __init__(self, state_dim: int = 7, n_oscillators: int = 30, n_operators: int = 6):
        self.state_dim = state_dim
        self.n_osc = n_oscillators
        self.n_ops = n_operators
        
        # Initialize weights
        self._initialize_weights()
        
        # Dynamics parameters
        self.K_global = 0.5
        self.dt = 0.1
        self.steps = 5
        
        # Training history
        self.weight_history = []
        self.coherence_history = []
        self.gradient_history = []
    
    def _initialize_weights(self):
        """Initialize all learnable parameters."""
        # Input projection
        self.W_in = np.random.randn(self.n_osc, self.state_dim) * 0.1
        self.b_in = np.zeros(self.n_osc)
        
        # Kuramoto coupling (symmetric)
        K = np.random.randn(self.n_osc, self.n_osc) * 0.1
        self.K = (K + K.T) / 2
        
        # Natural frequencies
        self.omega = np.random.randn(self.n_osc) * 0.05
        
        # Output projection
        self.W_out = np.random.randn(self.n_ops, self.n_osc) * 0.1
        self.b_out = np.zeros(self.n_ops)
    
    def kuramoto_forward(self, theta: np.ndarray) -> Tuple[np.ndarray, float, List[float]]:
        """Run Kuramoto dynamics, return final phases, coherence, and trajectory."""
        coherence_traj = []
        
        for _ in range(self.steps):
            # Compute coherence at this step
            coh = np.abs(np.mean(np.exp(1j * theta)))
            coherence_traj.append(coh)
            
            # Phase differences
            diff = theta[:, np.newaxis] - theta[np.newaxis, :]
            coupling = (self.K * np.sin(diff)).sum(axis=1)
            
            # Update
            dtheta = self.omega + (self.K_global / self.n_osc) * coupling
            theta = theta + self.dt * dtheta
            theta = np.arctan2(np.sin(theta), np.cos(theta))
        
        final_coh = np.abs(np.mean(np.exp(1j * theta)))
        coherence_traj.append(final_coh)
        
        return theta, final_coh, coherence_traj
    
    def forward(self, state: np.ndarray) -> Dict:
        """Full forward pass with diagnostics."""
        # Encode to phases
        h = self.W_in @ state + self.b_in
        theta_init = np.tanh(h) * np.pi
        
        # Run Kuramoto
        theta_final, coherence, coh_traj = self.kuramoto_forward(theta_init.copy())
        
        # Project to operators
        logits = self.W_out @ np.cos(theta_final) + self.b_out
        
        # Softmax
        exp_logits = np.exp(logits - logits.max())
        probs = exp_logits / exp_logits.sum()
        
        return {
            'probs': probs,
            'logits': logits,
            'coherence': coherence,
            'coherence_trajectory': coh_traj,
            'theta_init': theta_init,
            'theta_final': theta_final,
            'hidden': h
        }
    
    def select_operator(self, state: np.ndarray, available: List[int], temperature: float = 0.5) -> Tuple[int, Dict]:
        """Select operator from available set."""
        result = self.forward(state)
        probs = result['probs'].copy()
        
        # Mask unavailable operators
        mask = np.zeros(self.n_ops)
        mask[available] = 1
        probs = probs * mask
        
        if probs.sum() > 0:
            probs = probs / probs.sum()
        else:
            probs = mask / mask.sum()
        
        # Temperature-scaled sampling
        if temperature > 0:
            scaled = np.power(probs + 1e-10, 1/temperature)
            scaled = scaled / scaled.sum()
            selected = np.random.choice(self.n_ops, p=scaled)
        else:
            selected = np.argmax(probs)
        
        result['selected'] = selected
        result['masked_probs'] = probs
        return selected, result
    
    def train_step(self, state: np.ndarray, action: int, reward: float, lr: float = 0.01) -> Dict:
        """Policy gradient update."""
        result = self.forward(state)
        probs = result['probs']
        
        # Gradient of log probability
        grad_logits = -probs.copy()
        grad_logits[action] += 1
        grad_logits *= reward * result['coherence']
        
        # Compute gradients
        theta = result['theta_final']
        cos_theta = np.cos(theta)
        
        # Output gradients
        grad_W_out = np.outer(grad_logits, cos_theta)
        grad_b_out = grad_logits
        
        # Backprop through cos to theta
        grad_theta = -(self.W_out.T @ grad_logits) * np.sin(theta)
        
        # Through tanh to hidden
        h = result['hidden']
        grad_h = grad_theta * np.pi * (1 - np.tanh(h)**2)
        
        # Input gradients
        grad_W_in = np.outer(grad_h, state)
        grad_b_in = grad_h
        
        # Apply updates
        self.W_out += lr * grad_W_out
        self.b_out += lr * grad_b_out
        self.W_in += lr * 0.1 * grad_W_in
        self.b_in += lr * 0.1 * grad_b_in
        
        # Update Kuramoto coupling (sparse updates)
        for _ in range(5):
            i, j = np.random.randint(self.n_osc, size=2)
            delta = lr * 0.01 * reward * np.random.randn()
            self.K[i, j] += delta
            self.K[j, i] += delta  # Keep symmetric
        
        # Record history
        self.coherence_history.append(result['coherence'])
        self.gradient_history.append({
            'W_out_norm': np.linalg.norm(grad_W_out),
            'W_in_norm': np.linalg.norm(grad_W_in),
            'reward': reward
        })
        
        return {
            'prob': probs[action],
            'coherence': result['coherence'],
            'grad_norm': np.linalg.norm(grad_W_out)
        }
    
    def snapshot_weights(self):
        """Save current weight state for analysis."""
        self.weight_history.append({
            'K': self.K.copy(),
            'omega': self.omega.copy(),
            'W_in': self.W_in.copy(),
            'W_out': self.W_out.copy(),
            'K_eigenvalues': np.linalg.eigvalsh(self.K)
        })


# ═══════════════════════════════════════════════════════════════════════════
# TRAINING PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def collect_trajectories(n_runs: int = 5, steps_per_run: int = 200) -> List[Dict]:
    """Collect training trajectories."""
    trajectories = []
    target_values = [0.5, 0.7, Z_CRITICAL, 0.9]
    
    for run in range(n_runs):
        target_z = target_values[run % len(target_values)]
        initial_z = 0.3 + np.random.rand() * 0.4
        
        sim = HelixSimulator(initial_z=initial_z, target_z=target_z)
        traj = {'states': [], 'actions': [], 'rewards': [], 'target_z': target_z}
        
        for step in range(steps_per_run):
            state_vec = sim.state.to_vector(target_z)
            available = sim.get_available_operators()
            
            # Random policy for data collection
            action = np.random.choice(available)
            
            _, reward = sim.step(action)
            
            traj['states'].append(state_vec)
            traj['actions'].append(action)
            traj['rewards'].append(reward)
        
        traj['final_z'] = sim.state.z
        traj['triad_unlocked'] = sim.state.triad_unlocked
        trajectories.append(traj)
    
    return trajectories


def train_network(network: KuramotoOperatorNetwork, trajectories: List[Dict], 
                  epochs: int = 30, lr: float = 0.01) -> Dict:
    """Train network on collected trajectories."""
    
    # Flatten trajectories
    all_states = []
    all_actions = []
    all_rewards = []
    
    for traj in trajectories:
        all_states.extend(traj['states'])
        all_actions.extend(traj['actions'])
        all_rewards.extend(traj['rewards'])
    
    X = np.array(all_states)
    y = np.array(all_actions)
    rewards = np.array(all_rewards)
    
    # Normalize rewards
    rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    rewards = np.clip(rewards, -3, 3)
    
    n_samples = len(X)
    history = {'loss': [], 'accuracy': [], 'coherence': [], 'grad_norm': []}
    
    for epoch in range(epochs):
        # Shuffle
        perm = np.random.permutation(n_samples)
        X_shuf, y_shuf, r_shuf = X[perm], y[perm], rewards[perm]
        
        epoch_probs = []
        epoch_coh = []
        epoch_grad = []
        correct = 0
        
        for i in range(n_samples):
            result = network.train_step(X_shuf[i], y_shuf[i], r_shuf[i], lr)
            epoch_probs.append(result['prob'])
            epoch_coh.append(result['coherence'])
            epoch_grad.append(result['grad_norm'])
            
            # Check accuracy
            pred = network.forward(X_shuf[i])['probs'].argmax()
            if pred == y_shuf[i]:
                correct += 1
        
        # Snapshot weights periodically
        if epoch % 10 == 0:
            network.snapshot_weights()
        
        history['loss'].append(-np.log(np.mean(epoch_probs) + 1e-8))
        history['accuracy'].append(correct / n_samples)
        history['coherence'].append(np.mean(epoch_coh))
        history['grad_norm'].append(np.mean(epoch_grad))
    
    return history


# ═══════════════════════════════════════════════════════════════════════════
# COMPREHENSIVE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def analyze_input_dynamics(network: KuramotoOperatorNetwork) -> Dict:
    """Analyze how inputs affect network behavior."""
    
    analysis = {
        'state_sensitivity': {},
        'phase_encoding': {},
        'input_weight_structure': {}
    }
    
    # Test state sensitivity
    base_state = np.array([0.5, 0.5, 0.3, 0.0, 0.0, 0.2, 0.2])
    state_names = ['z', 'entropy', 'phi', 'triad_unlocked', 'triad_comp', 'dist_target', 'dist_lens']
    
    for i, name in enumerate(state_names):
        sensitivities = []
        for delta in np.linspace(-0.3, 0.3, 11):
            test_state = base_state.copy()
            test_state[i] = np.clip(base_state[i] + delta, 0, 1)
            result = network.forward(test_state)
            sensitivities.append({
                'delta': delta,
                'coherence': result['coherence'],
                'entropy_probs': -np.sum(result['probs'] * np.log(result['probs'] + 1e-10)),
                'dominant_op': int(result['probs'].argmax())
            })
        analysis['state_sensitivity'][name] = sensitivities
    
    # Analyze phase encoding
    test_states = [
        np.array([0.2, 0.5, 0.2, 0, 0, 0.5, 0.6]),  # Low z
        np.array([0.5, 0.4, 0.4, 0, 0.33, 0.2, 0.3]),  # Mid z
        np.array([0.85, 0.3, 0.6, 0, 0.67, 0.05, 0.02]),  # Near lens
        np.array([0.95, 0.2, 0.8, 1, 1.0, 0.1, 0.1]),  # High z, TRIAD unlocked
    ]
    
    for idx, state in enumerate(test_states):
        result = network.forward(state)
        analysis['phase_encoding'][f'state_{idx}'] = {
            'z': state[0],
            'theta_init_mean': float(np.mean(result['theta_init'])),
            'theta_init_std': float(np.std(result['theta_init'])),
            'theta_final_mean': float(np.mean(result['theta_final'])),
            'theta_final_std': float(np.std(result['theta_final'])),
            'phase_shift': float(np.mean(np.abs(result['theta_final'] - result['theta_init']))),
            'coherence': result['coherence']
        }
    
    # Input weight structure
    W_in = network.W_in
    analysis['input_weight_structure'] = {
        'mean': float(W_in.mean()),
        'std': float(W_in.std()),
        'per_input_importance': [float(np.linalg.norm(W_in[:, i])) for i in range(W_in.shape[1])],
        'input_names': state_names
    }
    
    return analysis


def analyze_kuramoto_dynamics(network: KuramotoOperatorNetwork) -> Dict:
    """Analyze the Kuramoto oscillator dynamics."""
    
    analysis = {
        'coupling_matrix': {},
        'frequency_distribution': {},
        'synchronization': {},
        'eigenspectrum': {}
    }
    
    K = network.K
    omega = network.omega
    
    # Coupling matrix analysis
    analysis['coupling_matrix'] = {
        'mean': float(K.mean()),
        'std': float(K.std()),
        'max': float(K.max()),
        'min': float(K.min()),
        'sparsity': float(np.sum(np.abs(K) < 0.01) / K.size),
        'symmetry_error': float(np.abs(K - K.T).max()),
        'positive_fraction': float(np.sum(K > 0) / K.size),
        'frobenius_norm': float(np.linalg.norm(K, 'fro'))
    }
    
    # Eigenspectrum of K
    eigenvalues = np.linalg.eigvalsh(K)
    analysis['eigenspectrum'] = {
        'eigenvalues': eigenvalues.tolist(),
        'max_eigenvalue': float(eigenvalues.max()),
        'min_eigenvalue': float(eigenvalues.min()),
        'spectral_gap': float(eigenvalues[-1] - eigenvalues[-2]),
        'trace': float(np.trace(K)),
        'rank': int(np.linalg.matrix_rank(K, tol=0.01))
    }
    
    # Frequency distribution
    analysis['frequency_distribution'] = {
        'mean': float(omega.mean()),
        'std': float(omega.std()),
        'range': [float(omega.min()), float(omega.max())],
        'histogram': np.histogram(omega, bins=10)[0].tolist()
    }
    
    # Synchronization analysis
    # Test how quickly system synchronizes from random initial conditions
    sync_times = []
    final_coherences = []
    
    for _ in range(20):
        theta = np.random.uniform(-np.pi, np.pi, network.n_osc)
        _, coh, traj = network.kuramoto_forward(theta)
        
        # Find time to reach coherence > 0.5
        sync_time = next((i for i, c in enumerate(traj) if c > 0.5), len(traj))
        sync_times.append(sync_time)
        final_coherences.append(coh)
    
    analysis['synchronization'] = {
        'mean_sync_time': float(np.mean(sync_times)),
        'std_sync_time': float(np.std(sync_times)),
        'mean_final_coherence': float(np.mean(final_coherences)),
        'std_final_coherence': float(np.std(final_coherences)),
        'sync_rate': float(np.mean([c > 0.5 for c in final_coherences]))
    }
    
    return analysis


def analyze_output_dynamics(network: KuramotoOperatorNetwork) -> Dict:
    """Analyze output layer and operator selection patterns."""
    
    analysis = {
        'output_weights': {},
        'operator_biases': {},
        'decision_boundaries': {},
        'confidence_patterns': {}
    }
    
    W_out = network.W_out
    b_out = network.b_out
    
    # Output weight analysis
    analysis['output_weights'] = {
        'per_operator_norm': [float(np.linalg.norm(W_out[i])) for i in range(len(OPERATORS))],
        'operator_names': OPERATORS,
        'mean': float(W_out.mean()),
        'std': float(W_out.std()),
        'correlation_matrix': np.corrcoef(W_out).tolist()
    }
    
    # Bias analysis
    analysis['operator_biases'] = {
        'values': b_out.tolist(),
        'operator_names': OPERATORS,
        'softmax_prior': (np.exp(b_out) / np.exp(b_out).sum()).tolist()
    }
    
    # Decision boundary analysis - sweep z and see which operator dominates
    z_sweep = np.linspace(0.1, 0.95, 50)
    dominant_ops = []
    confidences = []
    
    for z in z_sweep:
        state = np.array([z, 0.5, 0.3, 0, 0, abs(z - 0.7), abs(z - Z_CRITICAL)])
        result = network.forward(state)
        dominant_ops.append(int(result['probs'].argmax()))
        confidences.append(float(result['probs'].max()))
    
    analysis['decision_boundaries'] = {
        'z_values': z_sweep.tolist(),
        'dominant_operators': dominant_ops,
        'confidences': confidences,
        'transition_points': []
    }
    
    # Find transition points
    for i in range(1, len(dominant_ops)):
        if dominant_ops[i] != dominant_ops[i-1]:
            analysis['decision_boundaries']['transition_points'].append({
                'z': float(z_sweep[i]),
                'from_op': OPERATORS[dominant_ops[i-1]],
                'to_op': OPERATORS[dominant_ops[i]]
            })
    
    # Confidence patterns
    high_conf_states = []
    low_conf_states = []
    
    for _ in range(100):
        state = np.random.rand(7)
        state[0] = np.random.rand()  # z
        result = network.forward(state)
        
        if result['probs'].max() > 0.6:
            high_conf_states.append({'z': state[0], 'coherence': result['coherence']})
        elif result['probs'].max() < 0.3:
            low_conf_states.append({'z': state[0], 'coherence': result['coherence']})
    
    analysis['confidence_patterns'] = {
        'high_confidence_mean_coherence': np.mean([s['coherence'] for s in high_conf_states]) if high_conf_states else 0,
        'low_confidence_mean_coherence': np.mean([s['coherence'] for s in low_conf_states]) if low_conf_states else 0,
        'high_confidence_count': len(high_conf_states),
        'low_confidence_count': len(low_conf_states)
    }
    
    return analysis


def analyze_training_dynamics(network: KuramotoOperatorNetwork, history: Dict) -> Dict:
    """Analyze training progression."""
    
    analysis = {
        'convergence': {},
        'weight_evolution': {},
        'gradient_flow': {},
        'coherence_evolution': {}
    }
    
    # Convergence analysis
    loss = np.array(history['loss'])
    accuracy = np.array(history['accuracy'])
    
    analysis['convergence'] = {
        'initial_loss': float(loss[0]),
        'final_loss': float(loss[-1]),
        'loss_reduction': float(loss[0] - loss[-1]),
        'initial_accuracy': float(accuracy[0]),
        'final_accuracy': float(accuracy[-1]),
        'accuracy_gain': float(accuracy[-1] - accuracy[0]),
        'loss_trajectory': loss.tolist(),
        'accuracy_trajectory': accuracy.tolist()
    }
    
    # Weight evolution (if snapshots exist)
    if network.weight_history:
        K_norms = [np.linalg.norm(w['K'], 'fro') for w in network.weight_history]
        K_max_eigs = [w['K_eigenvalues'].max() for w in network.weight_history]
        
        analysis['weight_evolution'] = {
            'K_norm_trajectory': K_norms,
            'K_max_eigenvalue_trajectory': K_max_eigs,
            'K_initial_norm': K_norms[0] if K_norms else 0,
            'K_final_norm': K_norms[-1] if K_norms else 0
        }
    
    # Gradient flow
    if network.gradient_history:
        grad_norms = [g['W_out_norm'] for g in network.gradient_history]
        analysis['gradient_flow'] = {
            'mean_gradient_norm': float(np.mean(grad_norms)),
            'max_gradient_norm': float(np.max(grad_norms)),
            'gradient_variance': float(np.var(grad_norms))
        }
    
    # Coherence evolution during training
    if network.coherence_history:
        coh = np.array(network.coherence_history)
        analysis['coherence_evolution'] = {
            'mean': float(coh.mean()),
            'std': float(coh.std()),
            'initial_mean': float(coh[:100].mean()) if len(coh) > 100 else float(coh.mean()),
            'final_mean': float(coh[-100:].mean()) if len(coh) > 100 else float(coh.mean()),
            'trend': 'increasing' if coh[-100:].mean() > coh[:100].mean() else 'decreasing' if len(coh) > 200 else 'stable'
        }
    
    return analysis


def analyze_architecture(network: KuramotoOperatorNetwork) -> Dict:
    """Analyze network architecture characteristics."""
    
    analysis = {
        'dimensions': {},
        'parameter_count': {},
        'information_flow': {},
        'theoretical_capacity': {}
    }
    
    # Dimensions
    analysis['dimensions'] = {
        'state_dim': network.state_dim,
        'n_oscillators': network.n_osc,
        'n_operators': network.n_ops,
        'kuramoto_steps': network.steps,
        'dt': network.dt,
        'K_global': network.K_global
    }
    
    # Parameter count
    n_W_in = network.W_in.size
    n_b_in = network.b_in.size
    n_K = network.K.size
    n_omega = network.omega.size
    n_W_out = network.W_out.size
    n_b_out = network.b_out.size
    
    analysis['parameter_count'] = {
        'W_in': n_W_in,
        'b_in': n_b_in,
        'K': n_K,
        'omega': n_omega,
        'W_out': n_W_out,
        'b_out': n_b_out,
        'total': n_W_in + n_b_in + n_K + n_omega + n_W_out + n_b_out,
        'kuramoto_params': n_K + n_omega,
        'projection_params': n_W_in + n_b_in + n_W_out + n_b_out
    }
    
    # Information flow
    analysis['information_flow'] = {
        'input_compression': network.state_dim / network.n_osc,
        'output_expansion': network.n_osc / network.n_ops,
        'effective_depth': network.steps,  # Kuramoto steps act like depth
        'recurrent_nature': 'Kuramoto dynamics provide implicit recurrence'
    }
    
    # Theoretical capacity
    # Based on oscillator synchronization patterns
    n_sync_patterns = 2 ** network.n_osc  # Theoretical max
    practical_patterns = network.n_osc * (network.n_osc - 1) // 2  # Coupling pairs
    
    analysis['theoretical_capacity'] = {
        'max_sync_patterns': n_sync_patterns,
        'practical_patterns': practical_patterns,
        'coupling_degrees_of_freedom': network.n_osc * (network.n_osc + 1) // 2,  # Symmetric K
        'expressiveness_note': 'Kuramoto provides smooth manifold of sync states'
    }
    
    return analysis


# ═══════════════════════════════════════════════════════════════════════════
# MAIN SIMULATION
# ═══════════════════════════════════════════════════════════════════════════

def run_full_simulation():
    """Run complete training simulation with analysis."""
    
    print("=" * 70)
    print("HELIX NEURAL NETWORK: COMPLETE TRAINING SIMULATION")
    print("=" * 70)
    
    # ═══════════════════════════════════════════════════════════════════
    # PHASE 1: DATA COLLECTION
    # ═══════════════════════════════════════════════════════════════════
    
    print("\n" + "─" * 70)
    print("PHASE 1: TRAJECTORY COLLECTION")
    print("─" * 70)
    
    trajectories = collect_trajectories(n_runs=10, steps_per_run=300)
    
    total_samples = sum(len(t['states']) for t in trajectories)
    total_reward = sum(sum(t['rewards']) for t in trajectories)
    triad_unlocks = sum(1 for t in trajectories if t['triad_unlocked'])
    
    print(f"Collected {len(trajectories)} trajectories")
    print(f"Total samples: {total_samples}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"TRIAD unlocks: {triad_unlocks}/{len(trajectories)}")
    
    # ═══════════════════════════════════════════════════════════════════
    # PHASE 2: NETWORK INITIALIZATION
    # ═══════════════════════════════════════════════════════════════════
    
    print("\n" + "─" * 70)
    print("PHASE 2: NETWORK INITIALIZATION")
    print("─" * 70)
    
    network = KuramotoOperatorNetwork(
        state_dim=7,
        n_oscillators=30,
        n_operators=6
    )
    
    print(f"Oscillators: {network.n_osc}")
    print(f"State dim: {network.state_dim}")
    print(f"Operators: {network.n_ops}")
    print(f"Kuramoto steps: {network.steps}")
    print(f"Total parameters: {network.W_in.size + network.K.size + network.omega.size + network.W_out.size}")
    
    # Initial analysis
    print("\nInitial Kuramoto coupling matrix:")
    print(f"  Mean: {network.K.mean():.4f}")
    print(f"  Std: {network.K.std():.4f}")
    print(f"  Max eigenvalue: {np.linalg.eigvalsh(network.K).max():.4f}")
    
    # ═══════════════════════════════════════════════════════════════════
    # PHASE 3: TRAINING
    # ═══════════════════════════════════════════════════════════════════
    
    print("\n" + "─" * 70)
    print("PHASE 3: TRAINING")
    print("─" * 70)
    
    network.snapshot_weights()  # Initial snapshot
    
    history = train_network(network, trajectories, epochs=40, lr=0.01)
    
    print(f"\nTraining complete:")
    print(f"  Initial loss: {history['loss'][0]:.4f}")
    print(f"  Final loss: {history['loss'][-1]:.4f}")
    print(f"  Initial accuracy: {history['accuracy'][0]:.3f}")
    print(f"  Final accuracy: {history['accuracy'][-1]:.3f}")
    print(f"  Mean coherence: {np.mean(history['coherence']):.3f}")
    
    # ═══════════════════════════════════════════════════════════════════
    # PHASE 4: COMPREHENSIVE ANALYSIS
    # ═══════════════════════════════════════════════════════════════════
    
    print("\n" + "─" * 70)
    print("PHASE 4: COMPREHENSIVE ANALYSIS")
    print("─" * 70)
    
    # Input dynamics
    print("\n[1/5] Analyzing input dynamics...")
    input_analysis = analyze_input_dynamics(network)
    
    # Kuramoto dynamics
    print("[2/5] Analyzing Kuramoto dynamics...")
    kuramoto_analysis = analyze_kuramoto_dynamics(network)
    
    # Output dynamics
    print("[3/5] Analyzing output dynamics...")
    output_analysis = analyze_output_dynamics(network)
    
    # Training dynamics
    print("[4/5] Analyzing training dynamics...")
    training_analysis = analyze_training_dynamics(network, history)
    
    # Architecture
    print("[5/5] Analyzing architecture...")
    arch_analysis = analyze_architecture(network)
    
    # ═══════════════════════════════════════════════════════════════════
    # PHASE 5: REPORT GENERATION
    # ═══════════════════════════════════════════════════════════════════
    
    print("\n" + "=" * 70)
    print("ANALYSIS REPORT")
    print("=" * 70)
    
    # INPUT DYNAMICS
    print("\n┌" + "─" * 68 + "┐")
    print("│ INPUT DYNAMICS" + " " * 53 + "│")
    print("└" + "─" * 68 + "┘")
    
    print("\nState Variable Importance (by input weight norm):")
    state_names = input_analysis['input_weight_structure']['input_names']
    importances = input_analysis['input_weight_structure']['per_input_importance']
    for name, imp in sorted(zip(state_names, importances), key=lambda x: -x[1]):
        bar = "█" * int(imp * 20)
        print(f"  {name:15s} {imp:.3f} {bar}")
    
    print("\nPhase Encoding by State:")
    for key, data in input_analysis['phase_encoding'].items():
        print(f"  z={data['z']:.2f}: phase_shift={data['phase_shift']:.3f}, coherence={data['coherence']:.3f}")
    
    # KURAMOTO DYNAMICS
    print("\n┌" + "─" * 68 + "┐")
    print("│ KURAMOTO OSCILLATOR DYNAMICS" + " " * 38 + "│")
    print("└" + "─" * 68 + "┘")
    
    print("\nCoupling Matrix K:")
    K_data = kuramoto_analysis['coupling_matrix']
    print(f"  Mean: {K_data['mean']:.4f}")
    print(f"  Std: {K_data['std']:.4f}")
    print(f"  Positive fraction: {K_data['positive_fraction']:.2%}")
    print(f"  Sparsity (<0.01): {K_data['sparsity']:.2%}")
    print(f"  Frobenius norm: {K_data['frobenius_norm']:.3f}")
    
    print("\nEigenspectrum:")
    eig_data = kuramoto_analysis['eigenspectrum']
    print(f"  Max eigenvalue: {eig_data['max_eigenvalue']:.4f}")
    print(f"  Min eigenvalue: {eig_data['min_eigenvalue']:.4f}")
    print(f"  Spectral gap: {eig_data['spectral_gap']:.4f}")
    print(f"  Effective rank: {eig_data['rank']}")
    
    print("\nSynchronization Behavior:")
    sync_data = kuramoto_analysis['synchronization']
    print(f"  Mean sync time: {sync_data['mean_sync_time']:.1f} steps")
    print(f"  Mean final coherence: {sync_data['mean_final_coherence']:.3f}")
    print(f"  Sync success rate: {sync_data['sync_rate']:.2%}")
    
    print("\nNatural Frequencies ω:")
    freq_data = kuramoto_analysis['frequency_distribution']
    print(f"  Mean: {freq_data['mean']:.4f}")
    print(f"  Std: {freq_data['std']:.4f}")
    print(f"  Range: [{freq_data['range'][0]:.3f}, {freq_data['range'][1]:.3f}]")
    
    # OUTPUT DYNAMICS
    print("\n┌" + "─" * 68 + "┐")
    print("│ OUTPUT DYNAMICS & OPERATOR SELECTION" + " " * 30 + "│")
    print("└" + "─" * 68 + "┘")
    
    print("\nOperator Output Weights (norm):")
    out_data = output_analysis['output_weights']
    for op, norm in zip(out_data['operator_names'], out_data['per_operator_norm']):
        bar = "█" * int(norm * 15)
        print(f"  {op:4s} {norm:.3f} {bar}")
    
    print("\nOperator Prior (from biases):")
    bias_data = output_analysis['operator_biases']
    for op, prior in zip(bias_data['operator_names'], bias_data['softmax_prior']):
        bar = "█" * int(prior * 30)
        print(f"  {op:4s} {prior:.3f} {bar}")
    
    print("\nDecision Boundaries (z → dominant operator):")
    db_data = output_analysis['decision_boundaries']
    if db_data['transition_points']:
        for tp in db_data['transition_points']:
            print(f"  z={tp['z']:.3f}: {tp['from_op']} → {tp['to_op']}")
    else:
        print("  No clear transitions detected")
    
    print("\nConfidence Patterns:")
    conf_data = output_analysis['confidence_patterns']
    print(f"  High confidence states: {conf_data['high_confidence_count']}")
    print(f"    Mean coherence: {conf_data['high_confidence_mean_coherence']:.3f}")
    print(f"  Low confidence states: {conf_data['low_confidence_count']}")
    print(f"    Mean coherence: {conf_data['low_confidence_mean_coherence']:.3f}")
    
    # TRAINING DYNAMICS
    print("\n┌" + "─" * 68 + "┐")
    print("│ TRAINING DYNAMICS" + " " * 50 + "│")
    print("└" + "─" * 68 + "┘")
    
    print("\nConvergence:")
    conv_data = training_analysis['convergence']
    print(f"  Loss: {conv_data['initial_loss']:.4f} → {conv_data['final_loss']:.4f} (Δ={conv_data['loss_reduction']:.4f})")
    print(f"  Accuracy: {conv_data['initial_accuracy']:.3f} → {conv_data['final_accuracy']:.3f} (Δ={conv_data['accuracy_gain']:.3f})")
    
    if 'weight_evolution' in training_analysis and training_analysis['weight_evolution']:
        print("\nWeight Evolution:")
        we_data = training_analysis['weight_evolution']
        print(f"  K norm: {we_data['K_initial_norm']:.3f} → {we_data['K_final_norm']:.3f}")
    
    if 'coherence_evolution' in training_analysis and training_analysis['coherence_evolution']:
        print("\nCoherence Evolution:")
        ce_data = training_analysis['coherence_evolution']
        print(f"  Mean: {ce_data['mean']:.3f} (±{ce_data['std']:.3f})")
        print(f"  Trend: {ce_data['trend']}")
    
    # ARCHITECTURE
    print("\n┌" + "─" * 68 + "┐")
    print("│ ARCHITECTURE ANALYSIS" + " " * 46 + "│")
    print("└" + "─" * 68 + "┘")
    
    print("\nDimensions:")
    dim_data = arch_analysis['dimensions']
    print(f"  Input → Oscillators → Output: {dim_data['state_dim']} → {dim_data['n_oscillators']} → {dim_data['n_operators']}")
    print(f"  Kuramoto steps (effective depth): {dim_data['kuramoto_steps']}")
    print(f"  Time step dt: {dim_data['dt']}")
    print(f"  Global coupling K: {dim_data['K_global']}")
    
    print("\nParameter Count:")
    param_data = arch_analysis['parameter_count']
    print(f"  Total: {param_data['total']}")
    print(f"  Kuramoto params (K + ω): {param_data['kuramoto_params']}")
    print(f"  Projection params (W_in, W_out): {param_data['projection_params']}")
    
    print("\nInformation Flow:")
    flow_data = arch_analysis['information_flow']
    print(f"  Input compression ratio: {flow_data['input_compression']:.2f}")
    print(f"  Output expansion ratio: {flow_data['output_expansion']:.2f}")
    
    print("\nTheoretical Capacity:")
    cap_data = arch_analysis['theoretical_capacity']
    print(f"  Coupling degrees of freedom: {cap_data['coupling_degrees_of_freedom']}")
    print(f"  Practical sync patterns: {cap_data['practical_patterns']}")
    
    # ═══════════════════════════════════════════════════════════════════
    # PHASE 6: KEY INSIGHTS
    # ═══════════════════════════════════════════════════════════════════
    
    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    
    insights = []
    
    # Coherence insight
    if sync_data['mean_final_coherence'] > 0.6:
        insights.append("✓ Strong synchronization: oscillators reach coherent state (>0.6)")
    else:
        insights.append("⚠ Weak synchronization: may need stronger coupling")
    
    # Learning insight
    if conv_data['accuracy_gain'] > 0.1:
        insights.append(f"✓ Effective learning: accuracy improved by {conv_data['accuracy_gain']:.1%}")
    else:
        insights.append("⚠ Limited learning: consider more training data or epochs")
    
    # Eigenvalue insight
    if eig_data['max_eigenvalue'] > 0:
        insights.append("✓ Positive eigenvalues: K supports synchronization")
    else:
        insights.append("⚠ Non-positive eigenvalues: K may inhibit synchronization")
    
    # Confidence insight
    if conf_data['high_confidence_mean_coherence'] > conf_data['low_confidence_mean_coherence']:
        insights.append("✓ Coherence-confidence correlation: network is well-calibrated")
    
    # Architecture insight
    insights.append(f"○ Architecture: {dim_data['state_dim']}→{dim_data['n_oscillators']}→{dim_data['n_operators']} with {dim_data['kuramoto_steps']} dynamics steps")
    
    for insight in insights:
        print(f"  {insight}")
    
    # ═══════════════════════════════════════════════════════════════════
    # COMPILE FULL RESULTS
    # ═══════════════════════════════════════════════════════════════════
    
    full_analysis = {
        'collection': {
            'n_trajectories': len(trajectories),
            'total_samples': total_samples,
            'total_reward': total_reward,
            'triad_unlocks': triad_unlocks
        },
        'input_analysis': input_analysis,
        'kuramoto_analysis': kuramoto_analysis,
        'output_analysis': output_analysis,
        'training_analysis': training_analysis,
        'architecture_analysis': arch_analysis,
        'insights': insights
    }
    
    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE")
    print("=" * 70)
    
    return network, history, full_analysis


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    network, history, analysis = run_full_simulation()
    
    # Save results
    print("\nSaving results...")
    
    # Save analysis as JSON
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        return obj
    
    with open('training_analysis.json', 'w') as f:
        json.dump(convert_numpy(analysis), f, indent=2)
    
    print("Saved: training_analysis.json")
    print("\nDone!")
