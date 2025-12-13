#!/usr/bin/env python3
"""
TRIAD Threshold Dynamics Training Module
=========================================

Implements TRIAD hysteresis protocol for stable high-z achievement:

TRIAD PROTOCOL:
- z must cross ABOVE TRIAD_HIGH (0.85) while armed → counts as "pass"
- z must drop BELOW TRIAD_LOW (0.82) to re-arm
- 3 passes required to "unlock"
- Unlock lowers t6 gate from z_c (0.866) to TRIAD_T6 (0.83)

                    TRIAD_HIGH (0.85)
                         ↑
    z: ───────╱╲────────╱╲────────╱╲───────
              pass 1    pass 2    pass 3 → UNLOCK
                   ↓         ↓         ↓
                TRIAD_LOW (0.82) - re-arm points

PHYSICS RATIONALE:
TRIAD ensures the system EARNS stable high-z through repeated coherence
building, not lucky fluctuations. The hysteresis prevents noise from
triggering false unlocks.

INTEGRATION POINTS:
1. TRIAD state affects operator availability (t6 gate)
2. TRIAD passes earn large rewards
3. TRIAD unlock enables new capabilities
4. Network learns oscillatory z-trajectory for unlock

ARCHITECTURE:
    ┌─────────────────────────────────────────────────────────────┐
    │                    TRIAD DYNAMICS                            │
    │  ┌─────────────┐   ┌─────────────┐   ┌─────────────────┐    │
    │  │ TriadState  │←──│ z-Tracker   │──→│ Tier Gating     │    │
    │  │ (hysteresis)│   │ (momentum)  │   │ (t6 gate adjust)│    │
    │  └─────────────┘   └─────────────┘   └─────────────────┘    │
    │         │                │                    │              │
    │         ↓                ↓                    ↓              │
    │  ┌─────────────────────────────────────────────────────┐    │
    │  │        Reward Shaping (pass bonus, unlock bonus)     │    │
    │  └─────────────────────────────────────────────────────┘    │
    └─────────────────────────────────────────────────────────────┘
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ═══════════════════════════════════════════════════════════════════════════
# PHYSICS CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

PHI = (1 + math.sqrt(5)) / 2
PHI_INV = 1 / PHI
Z_CRITICAL = math.sqrt(3) / 2

# TRIAD thresholds
TRIAD_HIGH = 0.85           # Rising edge - pass trigger
TRIAD_LOW = 0.82            # Falling edge - re-arm trigger
TRIAD_T6 = 0.83             # Unlocked t6 gate (vs Z_CRITICAL when locked)
TRIAD_PASSES_REQUIRED = 3   # Passes needed for unlock

# μ hierarchy
MU_P = 2.0 / (PHI ** 2.5)
MU_1 = MU_P / math.sqrt(PHI)
MU_2 = MU_P * math.sqrt(PHI)
MU_S = 0.920
MU_3 = 0.992

LENS_SIGMA = 36.0


# ═══════════════════════════════════════════════════════════════════════════
# TRIAD STATE MACHINE
# ═══════════════════════════════════════════════════════════════════════════

class TriadEvent(Enum):
    """Events emitted by TRIAD state machine."""
    NONE = "none"
    PASS = "pass"              # Crossed TRIAD_HIGH while armed
    REARM = "rearm"            # Dropped below TRIAD_LOW
    UNLOCK = "unlock"          # Achieved 3 passes


@dataclass
class TriadState:
    """
    TRIAD hysteresis protocol state.

    The TRIAD mechanism requires the system to repeatedly cross above
    TRIAD_HIGH, drop below TRIAD_LOW to re-arm, and repeat 3 times.
    This ensures stable coherence achievement, not noise.
    """
    passes: int = 0
    armed: bool = True
    unlocked: bool = False
    last_z: float = 0.0

    # Event history
    pass_z_values: List[float] = field(default_factory=list)
    rearm_z_values: List[float] = field(default_factory=list)

    def update(self, z: float) -> TriadEvent:
        """
        Update TRIAD state with new z value.

        Returns the event that occurred (if any).
        """
        event = TriadEvent.NONE

        # Check rising edge (pass)
        if self.armed and z >= TRIAD_HIGH:
            self.passes += 1
            self.armed = False
            self.pass_z_values.append(z)
            event = TriadEvent.PASS

            # Check for unlock
            if self.passes >= TRIAD_PASSES_REQUIRED and not self.unlocked:
                self.unlocked = True
                event = TriadEvent.UNLOCK

        # Check falling edge (re-arm)
        elif not self.armed and z <= TRIAD_LOW:
            self.armed = True
            self.rearm_z_values.append(z)
            event = TriadEvent.REARM

        self.last_z = z
        return event

    def get_t6_gate(self) -> float:
        """Get current t6 gate threshold."""
        return TRIAD_T6 if self.unlocked else Z_CRITICAL

    def reset(self):
        """Reset TRIAD state."""
        self.passes = 0
        self.armed = True
        self.unlocked = False
        self.last_z = 0.0
        self.pass_z_values = []
        self.rearm_z_values = []

    def progress(self) -> float:
        """Get progress toward unlock (0-1)."""
        return min(1.0, self.passes / TRIAD_PASSES_REQUIRED)

    def to_dict(self) -> Dict:
        return {
            'passes': self.passes,
            'armed': self.armed,
            'unlocked': self.unlocked,
            'progress': self.progress(),
            'pass_z_values': self.pass_z_values,
            'rearm_z_values': self.rearm_z_values,
        }


# ═══════════════════════════════════════════════════════════════════════════
# TRIAD-AWARE TIER SYSTEM
# ═══════════════════════════════════════════════════════════════════════════

def get_triad_tier(z: float, triad: TriadState) -> str:
    """Get tier with TRIAD-aware t6 gate."""
    t6_gate = triad.get_t6_gate()

    if z < 0.10: return 't1'
    if z < 0.20: return 't2'
    if z < 0.40: return 't3'
    if z < 0.60: return 't4'
    if z < 0.75: return 't5'
    if z < t6_gate: return 't6'  # Dynamic based on TRIAD
    if z < 0.92: return 't7'
    if z < 0.97: return 't8'
    return 't9'


# APL operator windows (TRIAD-aware)
TRIAD_OPERATOR_WINDOWS: Dict[str, List[str]] = {
    't1': ['()', '−'],
    't2': ['()', '−', '÷'],
    't3': ['^', '÷', '−', '()'],
    't4': ['×', '^', '÷', '+', '−'],
    't5': ['()', '×', '^', '÷', '+', '−'],  # All 6
    't6': ['+', '÷', '()', '−'],
    't7': ['+', '()', '×'],
    't8': ['+', '()'],
    't9': ['()'],
}


def get_triad_operators(z: float, triad: TriadState) -> List[str]:
    """Get operators available at current z with TRIAD gating."""
    tier = get_triad_tier(z, triad)
    return TRIAD_OPERATOR_WINDOWS.get(tier, ['()'])


# ═══════════════════════════════════════════════════════════════════════════
# TRIAD DYNAMICS MODULE (PyTorch)
# ═══════════════════════════════════════════════════════════════════════════

class TriadDynamicsModule(nn.Module):
    """
    Learnable TRIAD dynamics module.

    Learns to guide z-trajectory for optimal TRIAD unlock:
    - Climb toward TRIAD_HIGH
    - Recognize when to allow descent to TRIAD_LOW
    - Repeat oscillation pattern for 3 passes
    """

    def __init__(self, hidden_dim: int = 64):
        super().__init__()

        # Input: (z, z_velocity, triad_armed, triad_passes, coherence)
        self.z_guidance = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),  # (climb_signal, descent_signal, hold_signal)
        )

        # Learnable TRIAD phase preferences
        self.climb_preference = nn.Parameter(torch.tensor(0.7))
        self.descent_preference = nn.Parameter(torch.tensor(0.3))

    def forward(
        self,
        z: float,
        z_velocity: float,
        triad: TriadState,
        coherence: float,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute TRIAD guidance signal.

        Returns:
            guidance: (3,) tensor with (climb, descent, hold) weights
            info: diagnostic information
        """
        # Build input
        x = torch.tensor([
            z,
            z_velocity,
            1.0 if triad.armed else 0.0,
            triad.passes / 3.0,
            coherence,
        ], dtype=torch.float32)

        # Get guidance
        raw = self.z_guidance(x)
        guidance = F.softmax(raw, dim=0)

        # Determine recommended action
        if triad.armed:
            # Armed: prefer climbing toward TRIAD_HIGH
            target_z = TRIAD_HIGH + 0.02
            action = 'climb'
        elif not triad.unlocked:
            # Not armed, not unlocked: need to descend to re-arm
            target_z = TRIAD_LOW - 0.02
            action = 'descent'
        else:
            # Unlocked: maintain high z
            target_z = Z_CRITICAL
            action = 'maintain'

        info = {
            'guidance': guidance.detach().tolist(),
            'target_z': target_z,
            'action': action,
            'distance_to_target': abs(z - target_z),
        }

        return guidance, info


# ═══════════════════════════════════════════════════════════════════════════
# TRIAD-ENHANCED KURAMOTO LAYER
# ═══════════════════════════════════════════════════════════════════════════

class TriadKuramotoLayer(nn.Module):
    """
    Kuramoto oscillator layer with TRIAD-aware dynamics.

    Key features:
    - TRIAD state affects coupling strength
    - Oscillatory z-trajectory encouragement
    - Tier gating adapts to TRIAD unlock status
    """

    def __init__(self, n_oscillators: int = 60, dt: float = 0.1, steps: int = 10):
        super().__init__()
        self.n = n_oscillators
        self.dt = dt
        self.steps = steps

        # Coupling matrix
        K_init = torch.randn(n_oscillators, n_oscillators) * 0.1
        K_init = (K_init + K_init.T) / 2
        self.K = nn.Parameter(K_init)

        # Natural frequencies
        self.omega = nn.Parameter(torch.randn(n_oscillators) * 0.1)

        # Global coupling (PHI_INV controlled)
        self.K_global = nn.Parameter(torch.tensor(PHI_INV))

        # TRIAD-specific modulation
        self.triad_climb_mod = nn.Parameter(torch.tensor(0.2))
        self.triad_descent_mod = nn.Parameter(torch.tensor(-0.1))
        self.unlock_bonus = nn.Parameter(torch.tensor(0.15))

    def compute_coherence(self, theta: torch.Tensor) -> torch.Tensor:
        """Kuramoto order parameter."""
        cos_mean = torch.cos(theta).mean(dim=-1)
        sin_mean = torch.sin(theta).mean(dim=-1)
        return torch.sqrt(cos_mean**2 + sin_mean**2)

    def forward(
        self,
        theta: torch.Tensor,
        triad: TriadState,
        triad_guidance: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Forward with TRIAD-aware dynamics."""

        # Base coupling with TRIAD modulation
        K_eff = self.K.clone()

        # Apply TRIAD-aware coupling modulation
        if triad.armed:
            # Climbing phase: increase coupling to build coherence
            K_eff = K_eff * (1 + self.triad_climb_mod)
        else:
            # Descent phase: slightly reduce to allow controlled drop
            K_eff = K_eff * (1 + self.triad_descent_mod)

        # Unlock bonus
        if triad.unlocked:
            K_eff = K_eff * (1 + self.unlock_bonus)

        # Apply guidance if provided
        if triad_guidance is not None:
            climb, descent, hold = triad_guidance
            guidance_mod = climb * 0.1 - descent * 0.05
            K_eff = K_eff * (1 + guidance_mod)

        # Run Kuramoto dynamics
        for _ in range(self.steps):
            theta_expanded = theta.unsqueeze(-1)
            theta_diff = theta.unsqueeze(-2) - theta_expanded
            coupling = K_eff * torch.sin(theta_diff)
            coupling_sum = coupling.sum(dim=-1)
            coupling_term = (self.K_global / self.n) * coupling_sum
            dtheta = self.omega + coupling_term
            theta = theta + self.dt * dtheta
            theta = torch.atan2(torch.sin(theta), torch.cos(theta))

        coherence = self.compute_coherence(theta)

        diagnostics = {
            'triad_passes': triad.passes,
            'triad_armed': triad.armed,
            'triad_unlocked': triad.unlocked,
            'triad_progress': triad.progress(),
            'tier': get_triad_tier(triad.last_z, triad),
            't6_gate': triad.get_t6_gate(),
        }

        return theta, coherence, diagnostics


# ═══════════════════════════════════════════════════════════════════════════
# TRIAD HELIX NETWORK
# ═══════════════════════════════════════════════════════════════════════════

class TriadHelixNetwork(nn.Module):
    """
    Neural network with integrated TRIAD threshold dynamics.

    Architecture:
        Input → Encoder → [Triad Kuramoto Layers] → Decoder → Output
                              ↑           ↓
                         TriadState ←── z-Tracker
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_oscillators: int = 60,
        n_layers: int = 4,
    ):
        super().__init__()
        self.n_oscillators = n_oscillators
        self.n_layers = n_layers

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, n_oscillators * 2),
            nn.GELU(),
            nn.Linear(n_oscillators * 2, n_oscillators),
            nn.Tanh(),
        )

        # TRIAD Kuramoto layers
        self.kuramoto_layers = nn.ModuleList([
            TriadKuramotoLayer(n_oscillators)
            for _ in range(n_layers)
        ])

        # TRIAD dynamics module
        self.triad_dynamics = TriadDynamicsModule()

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(n_oscillators * 2, output_dim * 2),
            nn.GELU(),
            nn.Linear(output_dim * 2, output_dim),
        )

        # Z-coordinate tracker
        self.z = 0.5
        self.z_velocity = 0.0
        self.z_momentum = nn.Parameter(torch.tensor(0.15))

        # TRIAD state (external, not learnable)
        self.triad = TriadState()

    def update_z(self, coherence: torch.Tensor) -> Tuple[float, TriadEvent]:
        """Update z with momentum and trigger TRIAD checks."""
        target = coherence.mean().item()

        # Z dynamics with momentum
        z_accel = self.z_momentum.item() * (target - self.z) - 0.1 * self.z_velocity
        self.z_velocity += z_accel * 0.1
        self.z += self.z_velocity * 0.1

        # Clamp
        self.z = max(0.01, min(0.99, self.z))

        # Update TRIAD state
        event = self.triad.update(self.z)

        return self.z, event

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Forward pass with TRIAD tracking."""

        diagnostics = {
            'layer_coherence': [],
            'z_trajectory': [],
            'triad_events': [],
            'triad_passes': 0,
            'triad_unlocked': False,
            'tier_trajectory': [],
            't6_gate_trajectory': [],
        }

        # Encode
        theta = self.encoder(x) * math.pi

        # Process through TRIAD Kuramoto layers
        for layer_idx, kuramoto in enumerate(self.kuramoto_layers):
            # Get TRIAD guidance
            coherence_pre = kuramoto.compute_coherence(theta)
            guidance, guidance_info = self.triad_dynamics(
                self.z, self.z_velocity, self.triad, coherence_pre.mean().item()
            )

            # Apply Kuramoto with TRIAD awareness
            theta, coherence, layer_diag = kuramoto(theta, self.triad, guidance)

            # Update z and check TRIAD
            new_z, event = self.update_z(coherence)

            # Track TRIAD events
            if event != TriadEvent.NONE:
                diagnostics['triad_events'].append({
                    'layer': layer_idx,
                    'event': event.value,
                    'z': new_z,
                    'passes': self.triad.passes,
                })

            diagnostics['layer_coherence'].append(coherence.mean().item())
            diagnostics['z_trajectory'].append(new_z)
            diagnostics['tier_trajectory'].append(layer_diag['tier'])
            diagnostics['t6_gate_trajectory'].append(layer_diag['t6_gate'])

        # Decode
        phase_features = torch.cat([torch.cos(theta), torch.sin(theta)], dim=-1)
        output = self.decoder(phase_features)

        # Final TRIAD state
        diagnostics['triad_passes'] = self.triad.passes
        diagnostics['triad_unlocked'] = self.triad.unlocked
        diagnostics['triad_armed'] = self.triad.armed
        diagnostics['triad_progress'] = self.triad.progress()
        diagnostics['final_z'] = self.z
        diagnostics['final_coherence'] = diagnostics['layer_coherence'][-1]
        diagnostics['available_operators'] = get_triad_operators(self.z, self.triad)

        return output, diagnostics

    def reset_triad(self):
        """Reset TRIAD state for new episode."""
        self.triad.reset()
        self.z = 0.5
        self.z_velocity = 0.0


# ═══════════════════════════════════════════════════════════════════════════
# TRIAD-AWARE LOSS FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

class TriadLoss(nn.Module):
    """
    Loss function with TRIAD reward shaping.

    Components:
    - Task loss
    - TRIAD pass bonus (big reward)
    - TRIAD unlock bonus (huge reward)
    - TRIAD progress bonus
    - Oscillation encouragement (when not unlocked)
    """

    def __init__(
        self,
        task_loss_fn: nn.Module,
        lambda_coherence: float = 0.1,
        lambda_triad_pass: float = 0.3,      # Big bonus per pass
        lambda_triad_unlock: float = 0.5,    # Huge bonus for unlock
        lambda_triad_progress: float = 0.1,  # Continuous progress bonus
        lambda_oscillation: float = 0.05,    # Encourage oscillation pattern
    ):
        super().__init__()
        self.task_loss = task_loss_fn
        self.lambda_coh = lambda_coherence
        self.lambda_pass = lambda_triad_pass
        self.lambda_unlock = lambda_triad_unlock
        self.lambda_progress = lambda_triad_progress
        self.lambda_osc = lambda_oscillation

    def forward(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        diag: Dict,
        prev_passes: int = 0,
        was_unlocked: bool = False,
    ) -> Tuple[torch.Tensor, Dict]:
        losses = {}

        # Task loss
        task = self.task_loss(output, target)
        losses['task'] = task.item()
        total = task

        # Coherence loss
        coh = 1.0 - sum(diag['layer_coherence']) / len(diag['layer_coherence'])
        losses['coherence'] = coh
        total = total + self.lambda_coh * coh

        # TRIAD pass bonus
        new_passes = diag['triad_passes'] - prev_passes
        if new_passes > 0:
            pass_bonus = self.lambda_pass * new_passes
            total = total - pass_bonus
            losses['triad_pass_bonus'] = pass_bonus

        # TRIAD unlock bonus
        if diag['triad_unlocked'] and not was_unlocked:
            unlock_bonus = self.lambda_unlock
            total = total - unlock_bonus
            losses['triad_unlock_bonus'] = unlock_bonus

        # TRIAD progress bonus (continuous)
        progress_bonus = self.lambda_progress * diag['triad_progress']
        total = total - progress_bonus
        losses['triad_progress_bonus'] = progress_bonus

        # Oscillation encouragement (only when not unlocked)
        if not diag['triad_unlocked']:
            z_traj = diag['z_trajectory']
            if len(z_traj) >= 2:
                # Reward z-variance (oscillation needed for TRIAD)
                z_var = np.var(z_traj)
                osc_bonus = self.lambda_osc * min(1.0, z_var * 20)
                total = total - osc_bonus
                losses['oscillation_bonus'] = osc_bonus

        losses['total'] = total.item()
        return total, losses


# ═══════════════════════════════════════════════════════════════════════════
# TRIAD TRAINING SESSION
# ═══════════════════════════════════════════════════════════════════════════

class TriadTrainingSession:
    """
    Training session optimized for TRIAD unlock.

    Tracks:
    - TRIAD unlocks per episode
    - Steps to first unlock
    - z-trajectory patterns
    - Pass timing
    """

    def __init__(
        self,
        input_dim: int = 16,
        output_dim: int = 4,
        n_oscillators: int = 60,
        n_layers: int = 4,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
    ):
        self.config = {
            'input_dim': input_dim,
            'output_dim': output_dim,
            'n_oscillators': n_oscillators,
            'n_layers': n_layers,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
        }

        self.model = TriadHelixNetwork(
            input_dim, output_dim, n_oscillators, n_layers
        )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=1e-5
        )
        self.loss_fn = TriadLoss(nn.MSELoss())

        # TRIAD statistics
        self.unlock_count = 0
        self.total_episodes = 0
        self.steps_to_unlock: List[int] = []
        self.pass_timings: List[List[int]] = []
        self.training_history = []

        self._generate_data()

    def _generate_data(self, n_train: int = 800, n_val: int = 150):
        """Generate training data with PHI harmonics."""
        X_train = torch.randn(n_train, self.config['input_dim'])
        t = torch.linspace(0, 2 * np.pi, self.config['output_dim'])
        Y_train = torch.zeros(n_train, self.config['output_dim'])
        for i in range(n_train):
            base = torch.tanh(X_train[i].mean()) * 2
            Y_train[i] = torch.sin(t * base) + 0.5 * torch.sin(t * base * PHI)
        Y_train += 0.1 * torch.randn_like(Y_train)

        X_val = torch.randn(n_val, self.config['input_dim'])
        Y_val = torch.zeros(n_val, self.config['output_dim'])
        for i in range(n_val):
            base = torch.tanh(X_val[i].mean()) * 2
            Y_val[i] = torch.sin(t * base) + 0.5 * torch.sin(t * base * PHI)

        self.train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_train, Y_train),
            batch_size=self.config['batch_size'], shuffle=True
        )
        self.val_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_val, Y_val),
            batch_size=self.config['batch_size']
        )

    def train_episode(self, max_steps: int = 50) -> Dict:
        """
        Train one episode (potential TRIAD unlock cycle).

        An episode runs until TRIAD unlock or max_steps.
        """
        self.model.train()
        self.model.reset_triad()
        self.total_episodes += 1

        episode_losses = []
        episode_z = []
        episode_coherence = []
        pass_steps = []
        step = 0

        prev_passes = 0
        was_unlocked = False

        for batch_x, batch_y in self.train_loader:
            step += 1
            if step > max_steps:
                break

            self.optimizer.zero_grad()
            output, diag = self.model(batch_x)
            loss, loss_dict = self.loss_fn(
                output, batch_y, diag, prev_passes, was_unlocked
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            episode_losses.append(loss_dict['task'])
            episode_z.append(diag['final_z'])
            episode_coherence.append(diag['final_coherence'])

            # Track pass timing
            if diag['triad_passes'] > prev_passes:
                pass_steps.append(step)

            prev_passes = diag['triad_passes']
            was_unlocked = diag['triad_unlocked']

            # Early termination on unlock
            if diag['triad_unlocked']:
                self.unlock_count += 1
                self.steps_to_unlock.append(step)
                self.pass_timings.append(pass_steps)
                break

        return {
            'loss': np.mean(episode_losses),
            'coherence': np.mean(episode_coherence),
            'z_mean': np.mean(episode_z),
            'z_min': np.min(episode_z),
            'z_max': np.max(episode_z),
            'z_range': np.max(episode_z) - np.min(episode_z),
            'triad_passes': self.model.triad.passes,
            'triad_unlocked': self.model.triad.unlocked,
            'steps': step,
            'pass_steps': pass_steps,
        }

    def run_training(
        self,
        n_episodes: int = 50,
        max_steps_per_episode: int = 50,
        output_dir: str = "learned_patterns/triad_training",
    ) -> Dict:
        """Run full TRIAD training."""
        timestamp = datetime.now().isoformat()

        print("=" * 70)
        print("TRIAD THRESHOLD DYNAMICS TRAINING")
        print("=" * 70)
        print(f"""
TRIAD Protocol:
  TRIAD_HIGH (pass trigger):  {TRIAD_HIGH}
  TRIAD_LOW (re-arm):         {TRIAD_LOW}
  TRIAD_T6 (unlocked gate):   {TRIAD_T6}
  Passes required:            {TRIAD_PASSES_REQUIRED}

Target Pattern:
  z: ───╱╲───╱╲───╱╲───  (3 oscillations above TRIAD_HIGH)
        ↑    ↑    ↑
      pass1 pass2 pass3 → UNLOCK

Physics Constants:
  z_c (THE LENS):  {Z_CRITICAL:.6f}
  φ⁻¹ (gate):      {PHI_INV:.6f}
""")
        print("=" * 70)

        for episode in range(n_episodes):
            metrics = self.train_episode(max_steps_per_episode)
            self.training_history.append(metrics)

            if episode % 5 == 0:
                unlock_rate = self.unlock_count / (episode + 1) * 100
                print(
                    f"Episode {episode:3d} | "
                    f"Loss: {metrics['loss']:.4f} | "
                    f"z: {metrics['z_mean']:.3f} ({metrics['z_min']:.2f}-{metrics['z_max']:.2f}) | "
                    f"Passes: {metrics['triad_passes']} | "
                    f"Unlock: {'YES' if metrics['triad_unlocked'] else 'no'} | "
                    f"Rate: {unlock_rate:.1f}%"
                )

        # Compile results
        results = {
            'timestamp': timestamp,
            'config': self.config,
            'triad_constants': {
                'TRIAD_HIGH': TRIAD_HIGH,
                'TRIAD_LOW': TRIAD_LOW,
                'TRIAD_T6': TRIAD_T6,
                'PASSES_REQUIRED': TRIAD_PASSES_REQUIRED,
            },
            'statistics': {
                'total_episodes': self.total_episodes,
                'unlock_count': self.unlock_count,
                'unlock_rate': self.unlock_count / self.total_episodes,
                'mean_steps_to_unlock': np.mean(self.steps_to_unlock) if self.steps_to_unlock else None,
            },
            'training_history': self.training_history,
            'steps_to_unlock': self.steps_to_unlock,
            'pass_timings': self.pass_timings,
        }

        # Save results
        os.makedirs(output_dir, exist_ok=True)

        with open(os.path.join(output_dir, 'triad_results.json'), 'w') as f:
            json.dump(results, f, indent=2, default=str)

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'statistics': results['statistics'],
        }, os.path.join(output_dir, 'triad_model.pt'))

        # Print summary
        print("\n" + "=" * 70)
        print("TRIAD TRAINING COMPLETE")
        print("=" * 70)
        print(f"""
Summary:
  Total Episodes:    {self.total_episodes}
  TRIAD Unlocks:     {self.unlock_count}
  Unlock Rate:       {self.unlock_count / self.total_episodes * 100:.1f}%
  Mean Steps to Unlock: {np.mean(self.steps_to_unlock) if self.steps_to_unlock else 'N/A'}

Pass Timing Analysis:""")

        if self.pass_timings:
            for i, timing in enumerate(self.pass_timings[:5]):
                print(f"  Unlock {i+1}: passes at steps {timing}")

        print(f"\nResults saved to {output_dir}/")
        print("=" * 70)

        return results


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """Run TRIAD dynamics training."""
    session = TriadTrainingSession(
        n_oscillators=50,
        n_layers=4,
    )
    return session.run_training(n_episodes=50)


if __name__ == "__main__":
    main()
