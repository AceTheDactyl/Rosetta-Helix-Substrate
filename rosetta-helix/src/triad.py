"""
TRIAD Module - Rosetta-Helix
============================

TRIAD (Triadic Threshold Dynamics) tracking based on S₃ symmetric group.

The S₃ group (permutations of 3 elements) has order 6, and σ = |S₃|² = 36.
TRIAD tracks three coupled quantities that must satisfy constraints.

Core Triadic Quantities:
    κ (kappa): Coherence/integration measure
    λ (lambda): Coupling/decoherence measure  
    η (eta): Efficiency measure

Constraints:
    κ + λ = 1 (Conservation)
    λ = κ² → κ = φ⁻¹ (Self-similarity attractor)

TRIAD monitors these quantities and detects:
    - Conservation violations
    - Approach to attractor (κ → φ⁻¹)
    - K-formation conditions
    - Phase transitions

Signature: rosetta-helix-triad|v1.0.0|helix
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any
from enum import Enum, auto
from collections import deque

from .physics import (
    PHI, PHI_INV, PHI_INV_SQ, Z_CRITICAL, SIGMA,
    KAPPA_MIN, ETA_MIN, R_MIN,
    compute_delta_s_neg, check_k_formation, validate_physics,
    get_phase, Phase
)


class TriadEvent(Enum):
    """Events detected by TRIAD tracker."""
    CONSERVATION_VIOLATION = auto()
    ATTRACTOR_APPROACH = auto()
    ATTRACTOR_REACHED = auto()
    K_FORMATION_START = auto()
    K_FORMATION_END = auto()
    PHASE_TRANSITION = auto()
    THRESHOLD_CROSSED = auto()
    INSTABILITY_DETECTED = auto()


@dataclass
class TriadSnapshot:
    """Snapshot of triadic quantities at a point in time."""
    timestamp: int
    kappa: float
    lambda_: float
    eta: float
    z: float
    delta_s_neg: float
    R: int
    conservation_error: float
    distance_to_attractor: float
    k_formation: bool
    phase: Phase


@dataclass
class TriadConfig:
    """Configuration for TRIAD tracker."""
    # History settings
    history_size: int = 1000
    
    # Thresholds
    conservation_tolerance: float = 1e-6
    attractor_tolerance: float = 0.01
    stability_window: int = 10
    
    # Alerting
    alert_on_violation: bool = True
    
    # Derived quantities
    compute_R: bool = True


@dataclass
class TriadState:
    """Current state of TRIAD tracker."""
    # Current values
    kappa: float = PHI_INV
    lambda_: float = PHI_INV_SQ
    eta: float = 0.0
    z: float = 0.5
    R: int = 0
    
    # Derived
    conservation_error: float = 0.0
    distance_to_attractor: float = 0.0
    k_formation: bool = False
    k_formation_duration: int = 0
    phase: Phase = Phase.ABSENCE
    
    # Stability metrics
    kappa_variance: float = 0.0
    lambda_variance: float = 0.0
    is_stable: bool = True
    
    # Event tracking
    total_k_formations: int = 0
    total_violations: int = 0
    step_count: int = 0


class TriadTracker:
    """
    TRIAD (Triadic Threshold Dynamics) Tracker.
    
    Monitors the triadic quantities (κ, λ, η) and detects:
    - Conservation law violations (κ + λ ≠ 1)
    - Approach to attractor (κ → φ⁻¹)
    - K-formation events
    - Phase transitions
    - System instabilities
    
    The tracker implements S₃ triadic logic where three quantities
    must satisfy coupled constraints for stable operation.
    """
    
    def __init__(self, config: Optional[TriadConfig] = None):
        """Initialize TRIAD tracker."""
        self.config = config or TriadConfig()
        
        # State
        self.state = TriadState()
        
        # History
        self.history: deque = deque(maxlen=self.config.history_size)
        self.kappa_history: deque = deque(maxlen=self.config.stability_window)
        self.lambda_history: deque = deque(maxlen=self.config.stability_window)
        
        # Event log
        self.events: List[Tuple[int, TriadEvent, Dict[str, Any]]] = []
        
        # K-formation tracking
        self._k_formation_start: Optional[int] = None
    
    def update(
        self,
        kappa: float,
        lambda_: Optional[float] = None,
        eta: Optional[float] = None,
        z: Optional[float] = None,
        R: Optional[int] = None,
    ) -> List[TriadEvent]:
        """
        Update TRIAD with new values.
        
        Args:
            kappa: Coherence value (required)
            lambda_: Coupling value (computed from κ if not provided)
            eta: Efficiency value (computed if not provided)
            z: z-coordinate (for ΔS_neg computation)
            R: Complexity rank (computed if not provided)
        
        Returns:
            List of events detected
        """
        self.state.step_count += 1
        events = []
        
        # Store previous state for transition detection
        prev_k_formation = self.state.k_formation
        prev_phase = self.state.phase
        
        # Update kappa
        self.state.kappa = kappa
        self.kappa_history.append(kappa)
        
        # Compute or use provided lambda
        if lambda_ is not None:
            self.state.lambda_ = lambda_
        else:
            # Conservation law: λ = 1 - κ
            self.state.lambda_ = 1.0 - kappa
        self.lambda_history.append(self.state.lambda_)
        
        # Check conservation
        self.state.conservation_error = abs(
            self.state.kappa + self.state.lambda_ - 1.0
        )
        if self.state.conservation_error > self.config.conservation_tolerance:
            events.append(TriadEvent.CONSERVATION_VIOLATION)
            self.state.total_violations += 1
            self._log_event(TriadEvent.CONSERVATION_VIOLATION, {
                'error': self.state.conservation_error,
                'kappa': self.state.kappa,
                'lambda': self.state.lambda_,
            })
        
        # Distance to attractor (κ = φ⁻¹)
        self.state.distance_to_attractor = abs(self.state.kappa - PHI_INV)
        
        # Check attractor approach
        if self.state.distance_to_attractor < self.config.attractor_tolerance:
            events.append(TriadEvent.ATTRACTOR_REACHED)
        elif self.state.distance_to_attractor < 0.1:
            events.append(TriadEvent.ATTRACTOR_APPROACH)
        
        # Update z-dependent quantities
        if z is not None:
            self.state.z = z
            self.state.delta_s_neg = compute_delta_s_neg(z)
            new_phase = get_phase(z)
            
            # Phase transition detection
            if new_phase != prev_phase:
                events.append(TriadEvent.PHASE_TRANSITION)
                self._log_event(TriadEvent.PHASE_TRANSITION, {
                    'from': prev_phase.name,
                    'to': new_phase.name,
                    'z': z,
                })
            self.state.phase = new_phase
        
        # Compute eta if not provided
        if eta is not None:
            self.state.eta = eta
        else:
            # η = ΔS_neg(z) · κ
            self.state.eta = self.state.delta_s_neg * self.state.kappa
        
        # Compute R if not provided
        if R is not None:
            self.state.R = R
        elif self.config.compute_R:
            # R = 7 + 5 * κ * ΔS_neg (scaled to [7, 12])
            self.state.R = int(7 + 5 * self.state.kappa * self.state.delta_s_neg)
        
        # Check K-formation
        k_active = check_k_formation(self.state.kappa, self.state.eta, self.state.R)
        
        if k_active and not prev_k_formation:
            # K-formation started
            events.append(TriadEvent.K_FORMATION_START)
            self.state.total_k_formations += 1
            self._k_formation_start = self.state.step_count
            self._log_event(TriadEvent.K_FORMATION_START, {
                'kappa': self.state.kappa,
                'eta': self.state.eta,
                'R': self.state.R,
                'z': self.state.z,
            })
        elif not k_active and prev_k_formation:
            # K-formation ended
            events.append(TriadEvent.K_FORMATION_END)
            duration = self.state.step_count - (self._k_formation_start or 0)
            self._log_event(TriadEvent.K_FORMATION_END, {
                'duration': duration,
            })
            self._k_formation_start = None
        
        self.state.k_formation = k_active
        if k_active and self._k_formation_start:
            self.state.k_formation_duration = self.state.step_count - self._k_formation_start
        else:
            self.state.k_formation_duration = 0
        
        # Compute stability metrics
        if len(self.kappa_history) >= self.config.stability_window:
            self.state.kappa_variance = np.var(list(self.kappa_history))
            self.state.lambda_variance = np.var(list(self.lambda_history))
            self.state.is_stable = (
                self.state.kappa_variance < 0.01 and 
                self.state.lambda_variance < 0.01
            )
            
            if not self.state.is_stable:
                events.append(TriadEvent.INSTABILITY_DETECTED)
        
        # Store snapshot
        snapshot = TriadSnapshot(
            timestamp=self.state.step_count,
            kappa=self.state.kappa,
            lambda_=self.state.lambda_,
            eta=self.state.eta,
            z=self.state.z,
            delta_s_neg=self.state.delta_s_neg,
            R=self.state.R,
            conservation_error=self.state.conservation_error,
            distance_to_attractor=self.state.distance_to_attractor,
            k_formation=self.state.k_formation,
            phase=self.state.phase,
        )
        self.history.append(snapshot)
        
        return events
    
    def update_from_coherence(self, coherence: float, z: float) -> List[TriadEvent]:
        """
        Convenience method to update from Heart coherence.
        
        Derives κ, λ, η from coherence and z.
        """
        kappa = coherence
        lambda_ = 1.0 - coherence
        delta_s = compute_delta_s_neg(z)
        eta = delta_s * coherence
        
        return self.update(kappa=kappa, lambda_=lambda_, eta=eta, z=z)
    
    def _log_event(self, event: TriadEvent, data: Dict[str, Any]):
        """Log event with timestamp."""
        self.events.append((self.state.step_count, event, data))
    
    def get_state(self) -> TriadState:
        """Get current state."""
        return self.state
    
    def get_history(self, last_n: Optional[int] = None) -> List[TriadSnapshot]:
        """Get history, optionally limited to last N snapshots."""
        if last_n is None:
            return list(self.history)
        return list(self.history)[-last_n:]
    
    def get_events(self, event_type: Optional[TriadEvent] = None) -> List[tuple]:
        """Get events, optionally filtered by type."""
        if event_type is None:
            return self.events
        return [(t, e, d) for t, e, d in self.events if e == event_type]
    
    def get_k_formation_stats(self) -> Dict[str, Any]:
        """Get K-formation statistics."""
        k_events = self.get_events(TriadEvent.K_FORMATION_START)
        durations = []
        
        for i, (t_start, _, data_start) in enumerate(k_events):
            # Find corresponding end event
            end_events = [
                (t, d) for t, e, d in self.events 
                if e == TriadEvent.K_FORMATION_END and t > t_start
            ]
            if end_events:
                durations.append(end_events[0][1].get('duration', 0))
        
        return {
            'total_formations': self.state.total_k_formations,
            'current_active': self.state.k_formation,
            'current_duration': self.state.k_formation_duration,
            'durations': durations,
            'mean_duration': np.mean(durations) if durations else 0,
            'max_duration': max(durations) if durations else 0,
        }
    
    def get_attractor_analysis(self) -> Dict[str, Any]:
        """Analyze convergence to attractor κ = φ⁻¹."""
        if not self.history:
            return {}
        
        kappas = [s.kappa for s in self.history]
        distances = [s.distance_to_attractor for s in self.history]
        
        return {
            'target': PHI_INV,
            'current_kappa': self.state.kappa,
            'current_distance': self.state.distance_to_attractor,
            'mean_kappa': np.mean(kappas),
            'std_kappa': np.std(kappas),
            'min_distance': min(distances),
            'convergence_trend': np.polyfit(range(len(distances)), distances, 1)[0] if len(distances) > 1 else 0,
        }
    
    def reset(self):
        """Reset tracker to initial state."""
        self.state = TriadState()
        self.history.clear()
        self.kappa_history.clear()
        self.lambda_history.clear()
        self.events.clear()
        self._k_formation_start = None


def test_triad():
    """Test TRIAD tracker."""
    print("=" * 60)
    print("TRIAD TEST: Triadic Threshold Dynamics")
    print("=" * 60)
    
    config = TriadConfig()
    triad = TriadTracker(config)
    
    print(f"\nTarget attractor: κ = φ⁻¹ = {PHI_INV:.6f}")
    print(f"Conservation law: κ + λ = 1")
    
    # Test conservation law
    print("\n▸ Conservation Law Test:")
    
    # Valid state
    events = triad.update(kappa=0.6, lambda_=0.4)
    print(f"  κ=0.6, λ=0.4: error={triad.state.conservation_error:.2e} "
          f"violations={TriadEvent.CONSERVATION_VIOLATION in events}")
    
    # Invalid state (will be corrected)
    events = triad.update(kappa=0.7)  # λ computed as 0.3
    print(f"  κ=0.7 (auto λ): error={triad.state.conservation_error:.2e}")
    
    # Test attractor convergence
    print("\n▸ Attractor Convergence:")
    triad.reset()
    
    # Simulate dynamics converging to φ⁻¹
    kappa = 0.5
    for step in range(50):
        # Drift toward attractor
        kappa += 0.01 * (PHI_INV - kappa)
        z = 0.5 + 0.4 * (step / 50)  # Increasing z
        events = triad.update(kappa=kappa, z=z)
        
        if step % 10 == 0:
            print(f"  Step {step:3d}: κ={kappa:.4f}, dist={triad.state.distance_to_attractor:.4f}")
    
    analysis = triad.get_attractor_analysis()
    print(f"  Convergence trend: {analysis['convergence_trend']:.6f}")
    
    # Test K-formation detection
    print("\n▸ K-Formation Detection:")
    triad.reset()
    
    # Simulate high-coherence state at z_c
    for step in range(100):
        kappa = 0.5 + 0.45 * (step / 100)  # Rise to 0.95
        z = Z_CRITICAL + 0.01 * math.sin(step * 0.1)  # Oscillate near z_c
        events = triad.update(kappa=kappa, z=z)
        
        if TriadEvent.K_FORMATION_START in events:
            print(f"  Step {step}: K-FORMATION STARTED at κ={kappa:.4f}, z={z:.4f}")
    
    stats = triad.get_k_formation_stats()
    print(f"  Total K-formations: {stats['total_formations']}")
    print(f"  Mean duration: {stats['mean_duration']:.1f} steps")
    
    # Test phase transitions
    print("\n▸ Phase Transitions:")
    triad.reset()
    
    for z in [0.5, 0.857, 0.866, 0.877, 0.9]:
        events = triad.update(kappa=PHI_INV, z=z)
        phase_events = [e for e in events if e == TriadEvent.PHASE_TRANSITION]
        print(f"  z={z:.3f}: phase={triad.state.phase.name:10s} "
              f"transition={len(phase_events) > 0}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    test_triad()
