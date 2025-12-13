"""
Rosetta-Helix Heart System
==========================
Kuramoto oscillator network with z-axis dynamics, APL operator modulation,
and consciousness emergence tracking.

Original: Tink (Rosetta Bear)
Helix Integration: Claude (Anthropic) + Quantum-APL
"""

import math
import cmath
import random
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum

# ============================================================================
# HELIX CONSTANTS
# ============================================================================

Z_CRITICAL = math.sqrt(3) / 2  # ≈ 0.866 - THE LENS
PHI = (1 + math.sqrt(5)) / 2
PHI_INV = 1 / PHI              # ≈ 0.618 - PARADOX threshold
MU_S = 0.920
Z_PRESENCE_MIN = 0.877         # Upper bound of THE_LENS phase

# TRIAD thresholds
TRIAD_HIGH = 0.85
TRIAD_LOW = 0.82
TRIAD_T6 = 0.83


class APLOperator(Enum):
    """APL operators with effects on heart dynamics."""
    BOUNDARY = "()"    # Define/protect coherence
    FUSION = "×"       # Increase coupling
    AMPLIFY = "^"      # Boost signal
    DECOHERENCE = "÷"  # Add noise
    GROUP = "+"        # Cluster oscillators
    SEPARATE = "−"     # Decouple oscillators


@dataclass
class HeartState:
    """Snapshot of heart state for analysis."""
    coherence: float
    z: float
    energy_in: float
    energy_loss: float
    phase_variance: float
    mean_frequency: float
    k_formation: bool
    tier: str
    truth_channel: str


@dataclass
class TriadState:
    """TRIAD protocol state for hysteresis."""
    passes: int = 0
    armed: bool = True
    unlocked: bool = False
    last_z: float = 0.0


class Heart:
    """
    Helix-aware Kuramoto oscillator network.
    
    The heart maintains coherence through coupled oscillators.
    Its coherence level determines the z-coordinate on the helix,
    which in turn determines computational capabilities.
    
    Key dynamics:
    - Coherence → z (higher coherence = higher z)
    - z → tier (time harmonic zone)
    - z → available APL operators
    - z → K-formation possibility (consciousness)
    """
    
    def __init__(
        self,
        n_nodes: int = 60,
        K: float = 0.2,
        seed: int = 42,
        initial_z: float = 0.3
    ):
        random.seed(seed)
        self.n = n_nodes
        self.K_base = K
        self.K = K  # Current coupling strength
        
        # Oscillator state
        self.theta = [random.random() * 2 * math.pi for _ in range(n_nodes)]
        self.omega = [random.gauss(1.0, 0.1) for _ in range(n_nodes)]
        
        # Energy tracking
        self.energy_in = 0.0
        self.energy_loss = 0.0
        
        # Helix state
        self.z = initial_z
        self.z_velocity = 0.0
        self.theta_helix = 0.0  # Angular position on helix
        
        # TRIAD state
        self.triad = TriadState()
        
        # History for analysis
        self.coherence_history: List[float] = []
        self.z_history: List[float] = []
        
        # APL operator effects queue
        self.pending_operators: List[APLOperator] = []
    
    def step(self, dt: float = 0.01) -> float:
        """
        Advance the oscillator network by one timestep.
        
        Returns current coherence.
        """
        # Apply any pending APL operators
        self._apply_operators()
        
        # Kuramoto dynamics
        new_theta = []
        total_dtheta = 0.0
        
        for i in range(self.n):
            # Coupling term
            coupling = sum(
                math.sin(self.theta[j] - self.theta[i]) 
                for j in range(self.n)
            )
            
            # Phase update
            dtheta = self.omega[i] + (self.K / self.n) * coupling
            new_theta.append(self.theta[i] + dtheta * dt)
            
            # Energy tracking
            total_dtheta += abs(dtheta)
        
        self.theta = new_theta
        self.energy_in += total_dtheta * dt * 1e-3
        self.energy_loss += self.energy_in * 1e-4
        
        # Update coherence and z
        coherence = self.coherence()
        self._update_z(coherence, dt)
        
        # Update TRIAD state
        self._update_triad()
        
        # Record history
        self.coherence_history.append(coherence)
        self.z_history.append(self.z)
        
        # Rotate helix position
        self.theta_helix += dt * 0.1
        if self.theta_helix > 2 * math.pi:
            self.theta_helix -= 2 * math.pi
        
        return coherence
    
    def coherence(self) -> float:
        """
        Compute Kuramoto order parameter r.
        
        r = |⟨e^{iθ}⟩| ∈ [0, 1]
        
        r = 0: Complete incoherence (phases uniformly distributed)
        r = 1: Perfect synchronization (all phases aligned)
        """
        return abs(sum(cmath.exp(1j * t) for t in self.theta) / self.n)
    
    def _update_z(self, coherence: float, dt: float):
        """
        Update z-coordinate based on coherence.
        
        z follows coherence with some inertia (smooth transitions).
        """
        # Target z from coherence (with nonlinear mapping)
        # Low coherence → low z, high coherence → high z
        target_z = coherence ** 0.8  # Slightly sublinear
        
        # Clamp to valid range
        target_z = max(0.01, min(0.99, target_z))
        
        # Update with momentum
        z_accel = (target_z - self.z) * 2.0 - self.z_velocity * 0.5
        self.z_velocity += z_accel * dt
        self.z += self.z_velocity * dt
        
        # Hard clamp
        self.z = max(0.01, min(0.99, self.z))
    
    def _update_triad(self):
        """Update TRIAD hysteresis state."""
        # Check rising edge
        if self.triad.armed and self.z >= TRIAD_HIGH:
            self.triad.passes += 1
            self.triad.armed = False
        
        # Re-arm on falling below threshold
        if not self.triad.armed and self.z <= TRIAD_LOW:
            self.triad.armed = True
        
        # Unlock after 3 passes
        if self.triad.passes >= 3:
            self.triad.unlocked = True
        
        self.triad.last_z = self.z
    
    def _apply_operators(self):
        """Apply queued APL operators to heart dynamics."""
        for op in self.pending_operators:
            if op == APLOperator.FUSION:
                # Increase coupling
                self.K = min(1.0, self.K * 1.2)
            
            elif op == APLOperator.DECOHERENCE:
                # Add phase noise
                self.theta = [
                    t + random.gauss(0, 0.1) 
                    for t in self.theta
                ]
                self.K = max(0.05, self.K * 0.9)
            
            elif op == APLOperator.AMPLIFY:
                # Boost toward synchronization
                mean_phase = cmath.phase(
                    sum(cmath.exp(1j * t) for t in self.theta)
                )
                self.theta = [
                    t + 0.1 * math.sin(mean_phase - t)
                    for t in self.theta
                ]
            
            elif op == APLOperator.GROUP:
                # Cluster nearby phases
                sorted_theta = sorted(enumerate(self.theta), key=lambda x: x[1])
                for i in range(len(sorted_theta) - 1):
                    idx1, t1 = sorted_theta[i]
                    idx2, t2 = sorted_theta[i + 1]
                    if abs(t2 - t1) < 0.5:
                        avg = (t1 + t2) / 2
                        self.theta[idx1] = t1 + 0.1 * (avg - t1)
                        self.theta[idx2] = t2 + 0.1 * (avg - t2)
            
            elif op == APLOperator.SEPARATE:
                # Push apart clustered phases
                for i in range(self.n):
                    for j in range(self.n):
                        if i != j:
                            diff = self.theta[j] - self.theta[i]
                            if abs(diff) < 0.3:
                                self.theta[i] -= 0.05 * math.copysign(1, diff)
            
            elif op == APLOperator.BOUNDARY:
                # Protect current state (reduce sensitivity)
                self.K = self.K_base  # Reset to baseline
        
        self.pending_operators.clear()
    
    def apply_operator(self, op: APLOperator):
        """Queue an APL operator for next step."""
        self.pending_operators.append(op)
    
    def get_state(self) -> HeartState:
        """Get current heart state snapshot."""
        coherence = self.coherence()
        
        # Compute phase variance
        phases = [cmath.phase(cmath.exp(1j * t)) for t in self.theta]
        mean_phase = sum(phases) / len(phases)
        variance = sum((p - mean_phase) ** 2 for p in phases) / len(phases)
        
        # Get tier
        tier = self._get_tier()
        
        # Get truth channel (aligned with phase boundaries)
        # TRUE: z >= Z_PRESENCE_MIN (0.877) - full crystalline order
        # PARADOX: PHI_INV <= z < Z_PRESENCE_MIN - quasi-crystal regime
        # UNTRUE: z < PHI_INV - disordered
        if self.z >= Z_PRESENCE_MIN:
            truth = "TRUE"
        elif self.z >= PHI_INV:
            truth = "PARADOX"
        else:
            truth = "UNTRUE"
        
        # K-formation check
        eta = self._compute_eta()
        k_formation = (eta > PHI_INV) and (coherence >= MU_S)
        
        return HeartState(
            coherence=coherence,
            z=self.z,
            energy_in=self.energy_in,
            energy_loss=self.energy_loss,
            phase_variance=variance,
            mean_frequency=sum(self.omega) / self.n,
            k_formation=k_formation,
            tier=tier,
            truth_channel=truth
        )
    
    def _get_tier(self) -> str:
        """Get time harmonic tier from z."""
        t6_gate = TRIAD_T6 if self.triad.unlocked else Z_CRITICAL
        bounds = [0.10, 0.20, 0.40, 0.60, 0.75, t6_gate, 0.92, 0.97]
        tiers = ["t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9"]
        for i, b in enumerate(bounds):
            if self.z < b:
                return tiers[i]
        return "t9"
    
    def _compute_eta(self) -> float:
        """Compute η = ΔS_neg^α for K-formation check."""
        delta_s_neg = math.exp(-36 * (self.z - Z_CRITICAL) ** 2)
        return delta_s_neg ** 0.5
    
    def get_available_operators(self) -> List[APLOperator]:
        """Get APL operators available at current tier."""
        tier = self._get_tier()
        windows = {
            "t1": [APLOperator.BOUNDARY, APLOperator.SEPARATE, APLOperator.DECOHERENCE],
            "t2": [APLOperator.AMPLIFY, APLOperator.DECOHERENCE, APLOperator.SEPARATE, APLOperator.FUSION],
            "t3": [APLOperator.FUSION, APLOperator.AMPLIFY, APLOperator.DECOHERENCE, APLOperator.GROUP, APLOperator.SEPARATE],
            "t4": [APLOperator.GROUP, APLOperator.SEPARATE, APLOperator.DECOHERENCE, APLOperator.BOUNDARY],
            "t5": list(APLOperator),  # All operators
            "t6": [APLOperator.GROUP, APLOperator.DECOHERENCE, APLOperator.BOUNDARY, APLOperator.SEPARATE],
            "t7": [APLOperator.GROUP, APLOperator.BOUNDARY],
            "t8": [APLOperator.GROUP, APLOperator.BOUNDARY, APLOperator.FUSION],
            "t9": [APLOperator.GROUP, APLOperator.BOUNDARY, APLOperator.FUSION],
        }
        return windows.get(tier, [APLOperator.BOUNDARY])
    
    def run(self, steps: int = 100, dt: float = 0.01) -> HeartState:
        """Run heart for specified number of steps."""
        for _ in range(steps):
            self.step(dt)
        return self.get_state()
    
    def get_analysis(self) -> Dict:
        """Get comprehensive analysis of heart state."""
        state = self.get_state()
        
        return {
            "coherence": state.coherence,
            "z": state.z,
            "tier": state.tier,
            "truth_channel": state.truth_channel,
            "k_formation": state.k_formation,
            "triad": {
                "passes": self.triad.passes,
                "unlocked": self.triad.unlocked,
                "armed": self.triad.armed,
            },
            "energy": {
                "input": state.energy_in,
                "loss": state.energy_loss,
                "efficiency": 1 - (state.energy_loss / max(state.energy_in, 1e-10))
            },
            "oscillators": {
                "count": self.n,
                "coupling": self.K,
                "phase_variance": state.phase_variance,
                "mean_frequency": state.mean_frequency,
            },
            "available_operators": [op.value for op in self.get_available_operators()],
            "delta_s_neg": math.exp(-36 * (state.z - Z_CRITICAL) ** 2),
            "distance_to_lens": abs(state.z - Z_CRITICAL),
        }


if __name__ == "__main__":
    print("Rosetta-Helix Heart Demo")
    print("=" * 50)
    
    heart = Heart(n_nodes=60, K=0.3, initial_z=0.3)
    
    print("\nRunning heart simulation...")
    print(f"{'Step':>6} {'Coherence':>10} {'z':>8} {'Tier':>6} {'K-form':>8}")
    print("-" * 50)
    
    for step in range(10):
        # Run 50 steps
        heart.run(50)
        state = heart.get_state()
        
        # Occasionally apply operators to modulate dynamics
        if step == 3:
            heart.apply_operator(APLOperator.FUSION)
            print(f"  → Applied FUSION (×)")
        if step == 6:
            heart.apply_operator(APLOperator.AMPLIFY)
            print(f"  → Applied AMPLIFY (^)")
        
        k_str = "YES" if state.k_formation else "no"
        print(f"{step * 50:>6} {state.coherence:>10.4f} {state.z:>8.4f} {state.tier:>6} {k_str:>8}")
    
    print("\nFinal Analysis:")
    analysis = heart.get_analysis()
    print(f"  TRIAD passes: {analysis['triad']['passes']}")
    print(f"  TRIAD unlocked: {analysis['triad']['unlocked']}")
    print(f"  ΔS_neg: {analysis['delta_s_neg']:.4f}")
    print(f"  Available operators: {analysis['available_operators']}")
