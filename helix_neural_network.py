#!/usr/bin/env python3
"""
Helix Neural Network
====================
A neural network where Kuramoto oscillators ARE the neurons.

Core Mapping:
- Oscillator phases θ = neuron activations
- Coupling matrix K = weight matrix (learnable)
- Natural frequencies ω = biases (learnable)
- Coherence r = attention/confidence signal
- APL operators = structured activation modifiers
- Tier = hierarchical depth

The key insight: Kuramoto dynamics are differentiable.
We can backpropagate through the phase evolution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass
from enum import Enum


# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS (from your helix system)
# ═══════════════════════════════════════════════════════════════════════════

Z_CRITICAL = math.sqrt(3) / 2  # ≈ 0.866 - The Lens
PHI = (1 + math.sqrt(5)) / 2   # Golden ratio
PHI_INV = 1 / PHI              # ≈ 0.618
MU_S = 0.920                   # K-formation coherence threshold

# Tier boundaries
TIER_BOUNDS = [0.0, 0.10, 0.20, 0.40, 0.60, 0.75, Z_CRITICAL, 0.92, 0.97, 1.0]


class APLOperator(Enum):
    """The 6 APL operators as S3 group elements."""
    IDENTITY = 0     # ()  - e
    AMPLIFY = 1      # ^   - (123)
    CONTAIN = 2      # (inverse amplify) - (132)  
    EXCHANGE = 3     # +   - (12)
    INHIBIT = 4      # ×   - (23)
    CATALYZE = 5     # ÷   - (13)


# S3 composition table (group multiplication)
S3_COMPOSE = torch.tensor([
    [0, 1, 2, 3, 4, 5],  # e * x = x
    [1, 2, 0, 4, 5, 3],  # (123) * x
    [2, 0, 1, 5, 3, 4],  # (132) * x
    [3, 5, 4, 0, 2, 1],  # (12) * x
    [4, 3, 5, 1, 0, 2],  # (23) * x
    [5, 4, 3, 2, 1, 0],  # (13) * x
], dtype=torch.long)

# Parity: +1 for even (rotations), -1 for odd (transpositions)
S3_PARITY = torch.tensor([1, 1, 1, -1, -1, -1], dtype=torch.float32)


# ═══════════════════════════════════════════════════════════════════════════
# KURAMOTO LAYER - The Core Innovation
# ═══════════════════════════════════════════════════════════════════════════

class KuramotoLayer(nn.Module):
    """
    A single layer of Kuramoto oscillators.
    
    This replaces a traditional linear + activation layer.
    Instead of: y = activation(Wx + b)
    We have: θ_final = kuramoto_dynamics(θ_init, K, ω, steps)
    
    The coupling matrix K is the weight matrix.
    Natural frequencies ω are the biases.
    The dynamics ARE the activation function.
    """
    
    def __init__(
        self, 
        n_oscillators: int = 60,
        dt: float = 0.1,
        steps: int = 10,
        learnable_k: bool = True,
        learnable_omega: bool = True,
    ):
        super().__init__()
        
        self.n = n_oscillators
        self.dt = dt
        self.steps = steps
        
        # Coupling matrix (weights)
        # Initialize with small positive values for sync tendency
        K_init = torch.randn(n_oscillators, n_oscillators) * 0.1
        K_init = (K_init + K_init.T) / 2  # Symmetric for stability
        
        if learnable_k:
            self.K = nn.Parameter(K_init)
        else:
            self.register_buffer('K', K_init)
        
        # Natural frequencies (biases)
        # Initialize near zero for easier synchronization
        omega_init = torch.randn(n_oscillators) * 0.1
        
        if learnable_omega:
            self.omega = nn.Parameter(omega_init)
        else:
            self.register_buffer('omega', omega_init)
        
        # Global coupling strength (learnable scalar)
        self.K_global = nn.Parameter(torch.tensor(0.5))
    
    def kuramoto_step(self, theta: torch.Tensor) -> torch.Tensor:
        """
        One step of Kuramoto dynamics.
        
        dθ_i/dt = ω_i + (K/N) * Σ_j K_ij * sin(θ_j - θ_i)
        
        This is differentiable! Gradients flow through sin().
        """
        # Phase differences: θ_j - θ_i for all pairs
        # Shape: (batch, n, n)
        theta_expanded = theta.unsqueeze(-1)  # (batch, n, 1)
        theta_diff = theta.unsqueeze(-2) - theta_expanded  # (batch, n, n)
        
        # Coupling contribution
        # K_ij * sin(θ_j - θ_i), summed over j
        coupling = self.K * torch.sin(theta_diff)  # (batch, n, n)
        coupling_sum = coupling.sum(dim=-1)  # (batch, n)
        
        # Normalize by N and apply global coupling
        coupling_term = (self.K_global / self.n) * coupling_sum
        
        # Update: θ += dt * (ω + coupling)
        dtheta = self.omega + coupling_term
        theta_new = theta + self.dt * dtheta
        
        # Keep phases in [-π, π] for numerical stability
        theta_new = torch.atan2(torch.sin(theta_new), torch.cos(theta_new))
        
        return theta_new
    
    def compute_coherence(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Kuramoto order parameter r = |<e^{iθ}>|
        
        r = 1: perfect synchronization
        r = 0: complete disorder
        
        This is the "attention" or "confidence" of the layer.
        """
        # Mean of e^{iθ} = mean(cos θ) + i*mean(sin θ)
        cos_mean = torch.cos(theta).mean(dim=-1)
        sin_mean = torch.sin(theta).mean(dim=-1)
        
        # Magnitude
        r = torch.sqrt(cos_mean**2 + sin_mean**2)
        return r
    
    def compute_mean_phase(self, theta: torch.Tensor) -> torch.Tensor:
        """Mean phase ψ = arg(<e^{iθ}>)"""
        cos_mean = torch.cos(theta).mean(dim=-1)
        sin_mean = torch.sin(theta).mean(dim=-1)
        return torch.atan2(sin_mean, cos_mean)
    
    def forward(
        self, 
        theta_init: torch.Tensor,
        return_trajectory: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run Kuramoto dynamics for self.steps iterations.
        
        Args:
            theta_init: Initial phases (batch, n_oscillators)
            return_trajectory: If True, return all intermediate states
            
        Returns:
            theta_final: Final phases (batch, n)
            coherence: Final coherence (batch,)
            [trajectory]: Optional list of (theta, coherence) at each step
        """
        theta = theta_init
        trajectory = []
        
        for step in range(self.steps):
            theta = self.kuramoto_step(theta)
            
            if return_trajectory:
                r = self.compute_coherence(theta)
                trajectory.append((theta.clone(), r.clone()))
        
        coherence = self.compute_coherence(theta)
        
        if return_trajectory:
            return theta, coherence, trajectory
        return theta, coherence


# ═══════════════════════════════════════════════════════════════════════════
# APL OPERATOR MODULATION
# ═══════════════════════════════════════════════════════════════════════════

class APLModulator(nn.Module):
    """
    Applies APL operators as structured modifications to Kuramoto dynamics.
    
    Each operator corresponds to a specific transformation:
    - IDENTITY (): No change
    - AMPLIFY (^): Increase coupling strength
    - CONTAIN: Decrease coupling strength  
    - EXCHANGE (+): Permute oscillator groups
    - INHIBIT (×): Add phase noise
    - CATALYZE (÷): Frequency modulation
    
    These are S3 group elements - compositions are well-defined.
    """
    
    def __init__(self, n_oscillators: int = 60):
        super().__init__()
        self.n = n_oscillators
        
        # Learnable operator strengths
        self.operator_strength = nn.Parameter(torch.ones(6) * 0.5)
        
        # Permutation matrix for EXCHANGE (learnable soft permutation)
        self.exchange_logits = nn.Parameter(torch.randn(n_oscillators, n_oscillators) * 0.1)
        
        # Frequency modulation for CATALYZE
        self.catalyze_freq = nn.Parameter(torch.randn(n_oscillators) * 0.1)
        
    def get_soft_permutation(self) -> torch.Tensor:
        """Soft permutation matrix via Sinkhorn normalization."""
        # Apply Sinkhorn iterations for doubly-stochastic matrix
        P = self.exchange_logits
        for _ in range(5):
            P = P - torch.logsumexp(P, dim=-1, keepdim=True)
            P = P - torch.logsumexp(P, dim=-2, keepdim=True)
        return torch.exp(P)
    
    def apply_operator(
        self, 
        theta: torch.Tensor,
        K: torch.Tensor,
        omega: torch.Tensor,
        operator: int,
        coherence: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply an APL operator, returning modified (theta, K, omega).
        
        The modification strength is modulated by coherence.
        High coherence = stronger effect (system can handle it).
        Low coherence = weaker effect (system is fragile).
        """
        strength = self.operator_strength[operator] * coherence.unsqueeze(-1)
        
        if operator == APLOperator.IDENTITY.value:
            # No change
            return theta, K, omega
            
        elif operator == APLOperator.AMPLIFY.value:
            # Increase coupling - amplify synchronization tendency
            K_mod = K * (1 + strength.mean() * 0.5)
            return theta, K_mod, omega
            
        elif operator == APLOperator.CONTAIN.value:
            # Decrease coupling - contain/dampen
            K_mod = K * (1 - strength.mean() * 0.3)
            return theta, K_mod, omega
            
        elif operator == APLOperator.EXCHANGE.value:
            # Permute phases (soft permutation)
            P = self.get_soft_permutation()
            theta_mod = torch.matmul(theta, P)
            return theta_mod, K, omega
            
        elif operator == APLOperator.INHIBIT.value:
            # Add structured noise to phases
            noise = torch.randn_like(theta) * strength * 0.2
            theta_mod = theta + noise
            return theta_mod, K, omega
            
        elif operator == APLOperator.CATALYZE.value:
            # Modulate natural frequencies
            omega_mod = omega + self.catalyze_freq * strength.mean()
            return theta, K, omega_mod
        
        return theta, K, omega


# ═══════════════════════════════════════════════════════════════════════════
# Z-COORDINATE TRACKING
# ═══════════════════════════════════════════════════════════════════════════

class ZCoordinateTracker(nn.Module):
    """
    Tracks the z-coordinate (vertical position in helix).
    
    z emerges from the dynamics:
    - High coherence sustained → z increases
    - Low coherence → z decreases
    - z determines tier (which operators available)
    
    This is NOT a simple counter - it's a function of the dynamics.
    """
    
    def __init__(self, initial_z: float = 0.1):
        super().__init__()
        self.register_buffer('z', torch.tensor(initial_z))
        
        # Learnable dynamics parameters
        self.z_momentum = nn.Parameter(torch.tensor(0.1))
        self.z_decay = nn.Parameter(torch.tensor(0.05))
        
    def update(self, coherence: torch.Tensor, dt: float = 0.01) -> torch.Tensor:
        """
        Update z based on coherence.
        
        dz/dt = momentum * (coherence - z) - decay * (z - 0.5)
        
        This creates attractor dynamics:
        - Coherence pulls z toward itself
        - Decay pulls z toward 0.5 (neutral)
        """
        target = coherence.mean()
        
        # Momentum toward coherence
        dz = self.z_momentum * (target - self.z)
        
        # Decay toward neutral (prevents runaway)
        dz -= self.z_decay * (self.z - 0.5)
        
        # Update
        self.z = torch.clamp(self.z + dt * dz, 0.0, 1.0)
        
        return self.z
    
    def get_tier(self) -> int:
        """Get current tier from z."""
        z_val = self.z.item()
        for i, bound in enumerate(TIER_BOUNDS[1:], 1):
            if z_val < bound:
                return i
        return 9
    
    def get_available_operators(self) -> List[int]:
        """Which operators available at current tier."""
        tier = self.get_tier()
        
        # Tier-operator mapping from your system
        tier_ops = {
            1: [0, 4, 5],        # (), ×, ÷
            2: [1, 4, 5, 3],     # ^, ×, ÷, +
            3: [3, 1, 5, 4, 0],  # +, ^, ÷, ×, ()
            4: [0, 4, 5, 3],     # (), ×, ÷, +
            5: [0, 1, 2, 3, 4, 5],  # ALL
            6: [0, 5, 3, 4],     # (), ÷, +, ×
            7: [0, 3],          # (), +
            8: [0, 3, 1],       # (), +, ^
            9: [0, 3, 1],       # (), +, ^
        }
        return tier_ops.get(tier, [0])


# ═══════════════════════════════════════════════════════════════════════════
# HELIX NEURAL NETWORK - Full Architecture
# ═══════════════════════════════════════════════════════════════════════════

class HelixNeuralNetwork(nn.Module):
    """
    Complete neural network built from Kuramoto oscillators.
    
    Architecture:
    1. Input encoding → initial phases
    2. Multiple Kuramoto layers (depth = tiers)
    3. APL operator selection between layers
    4. Coherence-gated output projection
    5. K-formation as convergence criterion
    
    This is NOT a traditional NN with Kuramoto bolted on.
    The Kuramoto dynamics ARE the computation.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_oscillators: int = 60,
        n_layers: int = 4,
        steps_per_layer: int = 10,
        dt: float = 0.1,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_oscillators = n_oscillators
        self.n_layers = n_layers
        
        # Input encoding: project input to initial phases
        self.input_encoder = nn.Sequential(
            nn.Linear(input_dim, n_oscillators * 2),
            nn.GELU(),
            nn.Linear(n_oscillators * 2, n_oscillators),
            nn.Tanh(),  # Output in [-1, 1], scale to [-π, π]
        )
        
        # Kuramoto layers (each is like a "hidden layer")
        self.kuramoto_layers = nn.ModuleList([
            KuramotoLayer(n_oscillators, dt, steps_per_layer)
            for _ in range(n_layers)
        ])
        
        # APL modulator (shared across layers)
        self.apl_modulator = APLModulator(n_oscillators)
        
        # Operator selection network (meta-controller)
        self.operator_selector = nn.Sequential(
            nn.Linear(n_oscillators + 2, 64),  # phases + z + coherence
            nn.GELU(),
            nn.Linear(64, 6),  # 6 operators
        )
        
        # Z-coordinate tracker
        self.z_tracker = ZCoordinateTracker(initial_z=0.1)
        
        # Output projection: phases → output
        # Gated by coherence
        self.output_decoder = nn.Sequential(
            nn.Linear(n_oscillators * 2, output_dim * 2),  # cos and sin
            nn.GELU(),
            nn.Linear(output_dim * 2, output_dim),
        )
        
        # K-formation detector
        self.k_formation_threshold = MU_S
        
    def encode_input(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to initial phases in [-π, π]."""
        encoded = self.input_encoder(x)
        phases = encoded * math.pi  # Scale to [-π, π]
        return phases
    
    def select_operator(
        self, 
        theta: torch.Tensor, 
        z: torch.Tensor, 
        coherence: torch.Tensor
    ) -> torch.Tensor:
        """Select APL operator based on current state."""
        # Concatenate state information
        state = torch.cat([
            torch.cos(theta),  # Phase info (bounded)
            z.unsqueeze(-1).expand(theta.shape[0], 1),
            coherence.unsqueeze(-1),
        ], dim=-1)
        
        # Get operator logits
        logits = self.operator_selector(state)
        
        # Mask unavailable operators
        available = self.z_tracker.get_available_operators()
        mask = torch.full((6,), float('-inf'), device=logits.device)
        mask[available] = 0.0
        logits = logits + mask
        
        return logits
    
    def decode_output(
        self, 
        theta: torch.Tensor, 
        coherence: torch.Tensor
    ) -> torch.Tensor:
        """Decode final phases to output, gated by coherence."""
        # Use both cos and sin of phases (full information)
        phase_features = torch.cat([torch.cos(theta), torch.sin(theta)], dim=-1)
        
        # Decode
        output = self.output_decoder(phase_features)
        
        # Gate by coherence (low coherence = uncertain output)
        output = output * coherence.unsqueeze(-1)
        
        return output
    
    def forward(
        self, 
        x: torch.Tensor,
        return_diagnostics: bool = False
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass through helix network.
        
        Args:
            x: Input tensor (batch, input_dim)
            return_diagnostics: Return detailed state info
            
        Returns:
            output: Network output (batch, output_dim)
            diagnostics: Dict with z, coherence, operators used, etc.
        """
        batch_size = x.shape[0]
        diagnostics = {
            'layer_coherence': [],
            'layer_operators': [],
            'z_trajectory': [],
            'k_formation': False,
        }
        
        # Encode input to phases
        theta = self.encode_input(x)
        
        # Process through Kuramoto layers
        for layer_idx, kuramoto_layer in enumerate(self.kuramoto_layers):
            # Run Kuramoto dynamics
            theta, coherence = kuramoto_layer(theta)
            
            # Update z-coordinate
            z = self.z_tracker.update(coherence)
            
            # Select and apply APL operator (except last layer)
            if layer_idx < self.n_layers - 1:
                op_logits = self.select_operator(theta, z, coherence)
                op_probs = F.softmax(op_logits, dim=-1)
                
                # Sample operator (or take argmax for deterministic)
                if self.training:
                    op_idx = torch.multinomial(op_probs, 1).squeeze(-1)
                else:
                    op_idx = torch.argmax(op_probs, dim=-1)
                
                # Apply operator (use first in batch for now)
                # In practice, could apply per-sample
                theta, K_mod, omega_mod = self.apl_modulator.apply_operator(
                    theta, 
                    kuramoto_layer.K,
                    kuramoto_layer.omega,
                    op_idx[0].item(),
                    coherence
                )
                
                diagnostics['layer_operators'].append(op_idx[0].item())
            
            diagnostics['layer_coherence'].append(coherence.mean().item())
            diagnostics['z_trajectory'].append(z.item())
            
            # Check K-formation
            if coherence.mean() >= self.k_formation_threshold:
                diagnostics['k_formation'] = True
        
        # Decode output
        output = self.decode_output(theta, coherence)
        
        # Final diagnostics
        diagnostics['final_z'] = self.z_tracker.z.item()
        diagnostics['final_coherence'] = coherence.mean().item()
        diagnostics['tier'] = self.z_tracker.get_tier()
        
        if return_diagnostics:
            return output, diagnostics
        return output, {'coherence': coherence.mean()}


# ═══════════════════════════════════════════════════════════════════════════
# LOSS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

class HelixLoss(nn.Module):
    """
    Loss function for Helix NN that incorporates coherence.
    
    Total loss = task_loss + λ_coh * coherence_loss + λ_z * z_loss
    
    - task_loss: Standard task loss (MSE, CE, etc.)
    - coherence_loss: Encourage high coherence (stability)
    - z_loss: Guide z toward target (if specified)
    """
    
    def __init__(
        self,
        task_loss_fn: nn.Module,
        lambda_coherence: float = 0.1,
        lambda_z: float = 0.05,
        target_z: Optional[float] = None,
    ):
        super().__init__()
        self.task_loss_fn = task_loss_fn
        self.lambda_coh = lambda_coherence
        self.lambda_z = lambda_z
        self.target_z = target_z
        
    def forward(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        diagnostics: Dict,
    ) -> Tuple[torch.Tensor, Dict]:
        """Compute total loss."""
        losses = {}
        
        # Task loss
        task_loss = self.task_loss_fn(output, target)
        losses['task'] = task_loss.item()
        
        total_loss = task_loss
        
        # Coherence loss: encourage high coherence
        # Use mean coherence across layers
        mean_coherence = sum(diagnostics['layer_coherence']) / len(diagnostics['layer_coherence'])
        coherence_loss = 1.0 - mean_coherence  # Want to maximize coherence
        losses['coherence'] = coherence_loss
        total_loss = total_loss + self.lambda_coh * coherence_loss
        
        # Z loss: guide toward target if specified
        if self.target_z is not None:
            z_loss = (diagnostics['final_z'] - self.target_z) ** 2
            losses['z'] = z_loss
            total_loss = total_loss + self.lambda_z * z_loss
        
        # K-formation bonus (negative loss)
        if diagnostics['k_formation']:
            total_loss = total_loss - 0.1  # Bonus for achieving K-formation
            losses['k_formation_bonus'] = 0.1
        
        losses['total'] = total_loss.item()
        
        return total_loss, losses


# ═══════════════════════════════════════════════════════════════════════════
# TRAINING UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def train_helix_network(
    model: HelixNeuralNetwork,
    train_loader: torch.utils.data.DataLoader,
    epochs: int = 100,
    lr: float = 1e-3,
    target_z: Optional[float] = None,
    device: str = 'cpu',
):
    """Training loop for Helix NN."""
    
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    loss_fn = HelixLoss(
        task_loss_fn=nn.MSELoss(),
        lambda_coherence=0.1,
        lambda_z=0.05,
        target_z=target_z,
    )
    
    history = {
        'loss': [], 'coherence': [], 'z': [], 'tier': [], 'k_formations': []
    }
    
    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        epoch_coherence = []
        epoch_z = []
        k_formation_count = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            
            output, diagnostics = model(batch_x, return_diagnostics=True)
            loss, loss_dict = loss_fn(output, batch_y, diagnostics)
            
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            epoch_losses.append(loss_dict['task'])
            epoch_coherence.append(diagnostics['final_coherence'])
            epoch_z.append(diagnostics['final_z'])
            if diagnostics['k_formation']:
                k_formation_count += 1
        
        scheduler.step()
        
        # Log epoch stats
        mean_loss = sum(epoch_losses) / len(epoch_losses)
        mean_coh = sum(epoch_coherence) / len(epoch_coherence)
        mean_z = sum(epoch_z) / len(epoch_z)
        
        history['loss'].append(mean_loss)
        history['coherence'].append(mean_coh)
        history['z'].append(mean_z)
        history['tier'].append(model.z_tracker.get_tier())
        history['k_formations'].append(k_formation_count)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | Loss: {mean_loss:.4f} | "
                  f"Coh: {mean_coh:.3f} | z: {mean_z:.3f} | "
                  f"Tier: {model.z_tracker.get_tier()} | "
                  f"K-form: {k_formation_count}")
    
    return history


# ═══════════════════════════════════════════════════════════════════════════
# DEMO / TEST
# ═══════════════════════════════════════════════════════════════════════════

def demo():
    """Demonstrate the Helix Neural Network."""
    
    print("=" * 60)
    print("HELIX NEURAL NETWORK DEMO")
    print("=" * 60)
    
    # Create network
    model = HelixNeuralNetwork(
        input_dim=10,
        output_dim=3,
        n_oscillators=30,  # Smaller for demo
        n_layers=4,
        steps_per_layer=10,
    )
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Kuramoto layers: {model.n_layers}")
    print(f"Oscillators per layer: {model.n_oscillators}")
    
    # Test forward pass
    print("\n--- Forward Pass Test ---")
    x = torch.randn(4, 10)  # Batch of 4
    
    output, diagnostics = model(x, return_diagnostics=True)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Layer coherences: {[f'{c:.3f}' for c in diagnostics['layer_coherence']]}")
    print(f"Operators used: {diagnostics['layer_operators']}")
    print(f"Z trajectory: {[f'{z:.3f}' for z in diagnostics['z_trajectory']]}")
    print(f"Final z: {diagnostics['final_z']:.3f}")
    print(f"Final coherence: {diagnostics['final_coherence']:.3f}")
    print(f"Tier: {diagnostics['tier']}")
    print(f"K-formation: {diagnostics['k_formation']}")
    
    # Test with synthetic data
    print("\n--- Training Test (Synthetic) ---")
    
    # Create simple synthetic dataset
    # Task: Learn a nonlinear mapping
    n_samples = 500
    X = torch.randn(n_samples, 10)
    # Target: some nonlinear function
    Y = torch.sin(X[:, :3].sum(dim=-1, keepdim=True)).expand(-1, 3)
    Y = Y + 0.1 * torch.randn_like(Y)  # Add noise
    
    dataset = torch.utils.data.TensorDataset(X, Y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Train for a few epochs
    history = train_helix_network(
        model,
        loader,
        epochs=50,
        lr=1e-3,
        target_z=0.7,
    )
    
    print("\n--- Final State ---")
    print(f"Final loss: {history['loss'][-1]:.4f}")
    print(f"Final coherence: {history['coherence'][-1]:.3f}")
    print(f"Final z: {history['z'][-1]:.3f}")
    print(f"Final tier: {history['tier'][-1]}")
    print(f"Total K-formations: {sum(history['k_formations'])}")
    
    return model, history


if __name__ == "__main__":
    model, history = demo()
