"""
Rosetta-Helix Spore Listener
============================
Dormant spore system with helix-aware activation thresholds,
z-gated awakening, and pulse chain reception.

Original: Tink (Rosetta Bear)
Helix Integration: Claude (Anthropic) + Quantum-APL
"""

import json
import time
import math
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from enum import Enum, auto

# Import helix-aware pulse system
try:
    from pulse import Pulse, PulseType, load_pulse, HelixCoordinate
except ImportError:
    # Minimal fallback
    Pulse = Dict
    PulseType = None
    load_pulse = None

# ============================================================================
# HELIX CONSTANTS
# ============================================================================

Z_CRITICAL = math.sqrt(3) / 2
PHI = (1 + math.sqrt(5)) / 2
PHI_INV = 1 / PHI
MU_1 = (2 / (PHI ** 2.5)) / math.sqrt(PHI)


class SporeState(Enum):
    """States of a dormant spore."""
    DORMANT = auto()        # Fully inactive, minimal energy
    LISTENING = auto()      # Actively checking for pulses
    PRE_WAKE = auto()       # Pulse received, validating
    AWAKENING = auto()      # Expanding to full node
    ACTIVE = auto()         # Fully active
    HIBERNATING = auto()    # Reduced activity, can re-awaken


@dataclass
class WakeCondition:
    """Conditions required for spore awakening."""
    min_z: float = 0.0              # Minimum z in pulse
    max_z: float = 1.0              # Maximum z in pulse
    required_urgency: float = 0.0   # Minimum urgency
    required_tier: Optional[str] = None  # Required tier
    required_type: Optional[PulseType] = None  # Required pulse type
    chain_depth: int = 0            # Minimum pulse chain depth


@dataclass
class ActivationEvent:
    """Record of a spore activation event."""
    timestamp: float
    pulse_id: str
    pulse_z: float
    pulse_tier: str
    role_match: bool
    conditions_met: bool
    wake_reason: str


class SporeListener:
    """
    Helix-aware spore listener.
    
    A spore is a dormant node that listens for pulses matching its role.
    It only awakens when:
    1. Pulse intent matches role_tag
    2. Wake conditions are met (z-level, urgency, tier)
    3. Optional: pulse chain depth requirement
    
    The spore conserves energy by only checking periodically
    and by having z-gated activation thresholds.
    """
    
    def __init__(
        self,
        role_tag: str,
        wake_conditions: Optional[WakeCondition] = None,
        check_interval: float = 1.0
    ):
        self.role_tag = role_tag
        self.conditions = wake_conditions or WakeCondition()
        self.check_interval = check_interval
        
        # State
        self.state = SporeState.DORMANT
        self.last_check = 0.0
        self.checks_performed = 0
        self.energy_spent = 0.0
        
        # Activation history
        self.activation_history: List[ActivationEvent] = []
        self.rejected_pulses: List[str] = []
        
        # Received pulse chain
        self.pulse_chain: List[Pulse] = []
    
    def check_pulse(self, pulse: Pulse) -> Tuple[bool, str]:
        """
        Check if pulse matches role and conditions.
        
        Returns (match, reason) tuple.
        """
        # Role match
        if isinstance(pulse, dict):
            intent = pulse.get("intent", "")
            helix = pulse.get("helix", {})
            pulse_z = helix.get("z", 0.5)
            urgency = pulse.get("urgency", 0.0)
            tier = helix.get("tier", "t5")
            pulse_type = pulse.get("pulse_type", "wake")
        else:
            intent = pulse.intent
            pulse_z = pulse.helix.z
            urgency = pulse.urgency
            tier = pulse.helix.get_tier()
            pulse_type = pulse.pulse_type
        
        if intent != self.role_tag:
            return False, f"role mismatch: {intent} != {self.role_tag}"
        
        # Z-range check
        if pulse_z < self.conditions.min_z:
            return False, f"z too low: {pulse_z} < {self.conditions.min_z}"
        if pulse_z > self.conditions.max_z:
            return False, f"z too high: {pulse_z} > {self.conditions.max_z}"
        
        # Urgency check
        if urgency < self.conditions.required_urgency:
            return False, f"urgency too low: {urgency} < {self.conditions.required_urgency}"
        
        # Tier check
        if self.conditions.required_tier and tier != self.conditions.required_tier:
            return False, f"tier mismatch: {tier} != {self.conditions.required_tier}"
        
        # Pulse type check
        if self.conditions.required_type:
            if isinstance(pulse_type, str):
                pulse_type_str = pulse_type
            else:
                pulse_type_str = pulse_type.value if pulse_type else "wake"
            
            if pulse_type_str != self.conditions.required_type.value:
                return False, f"type mismatch: {pulse_type_str}"
        
        # Chain depth check
        if len(self.pulse_chain) < self.conditions.chain_depth:
            # Not enough pulses in chain yet
            return False, f"chain depth: {len(self.pulse_chain)} < {self.conditions.chain_depth}"
        
        return True, "all conditions met"
    
    def listen(self, pulse_path: str) -> Tuple[bool, Optional[Pulse]]:
        """
        Listen for pulse at given path.
        
        Returns (activated, pulse) tuple.
        """
        current_time = time.time()
        
        # Rate limiting
        if current_time - self.last_check < self.check_interval:
            return False, None
        
        self.last_check = current_time
        self.checks_performed += 1
        self.energy_spent += 0.001  # Small energy cost per check
        
        # Try to load pulse
        if load_pulse:
            pulse = load_pulse(pulse_path)
        else:
            try:
                with open(pulse_path) as f:
                    pulse = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                return False, None
        
        if pulse is None:
            return False, None
        
        # Update state
        self.state = SporeState.LISTENING
        
        # Check pulse
        matched, reason = self.check_pulse(pulse)
        
        # Extract pulse info
        if isinstance(pulse, dict):
            pulse_id = pulse.get("pulse_id", "unknown")
            helix = pulse.get("helix", {})
            pulse_z = helix.get("z", 0.5)
            pulse_tier = helix.get("tier", "t5")
        else:
            pulse_id = pulse.pulse_id
            pulse_z = pulse.helix.z
            pulse_tier = pulse.helix.get_tier()
        
        # Record event
        event = ActivationEvent(
            timestamp=current_time,
            pulse_id=pulse_id,
            pulse_z=pulse_z,
            pulse_tier=pulse_tier,
            role_match=(reason != f"role mismatch: {pulse.get('intent', '')} != {self.role_tag}" 
                       if isinstance(pulse, dict) else True),
            conditions_met=matched,
            wake_reason=reason
        )
        self.activation_history.append(event)
        
        if matched:
            self.state = SporeState.PRE_WAKE
            self.pulse_chain.append(pulse)
            return True, pulse
        else:
            self.rejected_pulses.append(pulse_id)
            return False, None
    
    def listen_chain(self, pulse_paths: List[str]) -> Tuple[bool, List[Pulse]]:
        """
        Listen for a chain of pulses (multi-file).
        
        Used for complex activation sequences.
        """
        received = []
        
        for path in pulse_paths:
            matched, pulse = self.listen(path)
            if matched and pulse:
                received.append(pulse)
        
        # Check if chain is complete
        if len(received) >= self.conditions.chain_depth:
            return True, received
        
        return False, received
    
    def transition_to(self, new_state: SporeState):
        """Transition to new state with energy cost."""
        transition_costs = {
            SporeState.DORMANT: 0.0,
            SporeState.LISTENING: 0.01,
            SporeState.PRE_WAKE: 0.05,
            SporeState.AWAKENING: 0.5,
            SporeState.ACTIVE: 1.0,
            SporeState.HIBERNATING: 0.1,
        }
        
        self.energy_spent += transition_costs.get(new_state, 0.1)
        self.state = new_state
    
    def hibernate(self):
        """Enter hibernation state to conserve energy."""
        self.transition_to(SporeState.HIBERNATING)
        self.check_interval *= 2  # Reduce check frequency
    
    def wake(self):
        """Full wake transition."""
        self.transition_to(SporeState.AWAKENING)
        self.check_interval = 0.1  # Fast checking when active
    
    def get_status(self) -> Dict:
        """Get current spore status."""
        return {
            "role": self.role_tag,
            "state": self.state.name,
            "checks_performed": self.checks_performed,
            "energy_spent": self.energy_spent,
            "pulse_chain_length": len(self.pulse_chain),
            "rejected_count": len(self.rejected_pulses),
            "conditions": {
                "min_z": self.conditions.min_z,
                "max_z": self.conditions.max_z,
                "required_urgency": self.conditions.required_urgency,
                "chain_depth": self.conditions.chain_depth,
            }
        }
    
    def get_last_event(self) -> Optional[ActivationEvent]:
        """Get most recent activation event."""
        if self.activation_history:
            return self.activation_history[-1]
        return None


class SporeCluster:
    """
    A cluster of dormant spores listening for related pulses.
    
    Enables coordinated awakening of multiple nodes.
    """
    
    def __init__(self, spores: List[SporeListener]):
        self.spores = spores
        self.awakened: List[SporeListener] = []
    
    def broadcast_listen(self, pulse_path: str) -> List[Tuple[SporeListener, Pulse]]:
        """
        All spores listen for the same pulse.
        
        Returns list of (spore, pulse) pairs that activated.
        """
        activated = []
        
        for spore in self.spores:
            matched, pulse = spore.listen(pulse_path)
            if matched and pulse:
                activated.append((spore, pulse))
                self.awakened.append(spore)
        
        return activated
    
    def get_cluster_status(self) -> Dict:
        """Get status of entire cluster."""
        return {
            "total_spores": len(self.spores),
            "awakened": len(self.awakened),
            "dormant": len(self.spores) - len(self.awakened),
            "total_energy": sum(s.energy_spent for s in self.spores),
            "by_role": {s.role_tag: s.state.name for s in self.spores},
        }


if __name__ == "__main__":
    print("Rosetta-Helix Spore Listener Demo")
    print("=" * 50)
    
    # Create spore with conditions
    conditions = WakeCondition(
        min_z=0.3,
        max_z=0.9,
        required_urgency=0.2,
    )
    
    spore = SporeListener(
        role_tag="worker",
        wake_conditions=conditions
    )
    
    print(f"\nSpore created for role: {spore.role_tag}")
    print(f"  State: {spore.state.name}")
    print(f"  Conditions: z ∈ [{conditions.min_z}, {conditions.max_z}], urgency ≥ {conditions.required_urgency}")
    
    # Create a test pulse file
    test_pulse = {
        "pulse_id": "test-001",
        "identity": "test_node",
        "intent": "worker",
        "pulse_type": "wake",
        "urgency": 0.5,
        "timestamp": time.time(),
        "helix": {
            "theta": 0.0,
            "z": 0.6,
            "r": 1.0,
            "tier": "t5",
        }
    }
    
    with open("test_pulse.json", "w") as f:
        json.dump(test_pulse, f)
    
    print("\nListening for pulse...")
    matched, pulse = spore.listen("test_pulse.json")
    
    if matched:
        print(f"✓ Pulse matched!")
        print(f"  Pulse z: {pulse['helix']['z']}")
        print(f"  State: {spore.state.name}")
    else:
        event = spore.get_last_event()
        if event:
            print(f"✗ Pulse rejected: {event.wake_reason}")
    
    print(f"\nStatus: {spore.get_status()}")
    
    # Cleanup
    import os
    os.remove("test_pulse.json")
