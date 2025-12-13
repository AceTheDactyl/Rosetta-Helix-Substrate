"""
Rosetta-Helix Pulse System
==========================
Expanded pulse generation with z-axis context, APL operator hints,
and helix coordinate embedding.

Original: Tink (Rosetta Bear)
Helix Integration: Claude (Anthropic) + Quantum-APL
"""

import json
import time
import uuid
import math
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict
from enum import Enum

# ============================================================================
# HELIX CONSTANTS
# ============================================================================

Z_CRITICAL = math.sqrt(3) / 2  # ≈ 0.866
PHI = (1 + math.sqrt(5)) / 2
PHI_INV = 1 / PHI

# Truth channel boundaries (aligned with phase regime mapping)
Z_PRESENCE_MIN = 0.877  # TRUE threshold (upper bound of THE_LENS phase)

# μ-field thresholds
MU_P = 2 / (PHI ** 2.5)
MU_1 = MU_P / math.sqrt(PHI)
MU_2 = MU_P * math.sqrt(PHI)
MU_S = 0.920


class PulseType(Enum):
    """Types of pulses in the helix system."""
    WAKE = "wake"           # Activate dormant spore
    SYNC = "sync"           # Synchronization request
    ELEVATE = "elevate"     # Request z-elevation
    DESCEND = "descend"     # Request z-descent
    FUSE = "fuse"           # Fusion request (×)
    SEPARATE = "separate"   # Separation request (−)
    AMPLIFY = "amplify"     # Amplification request (^)
    BOUNDARY = "boundary"   # Boundary definition (())
    HEARTBEAT = "heartbeat" # Regular coherence pulse


class TruthChannel(Enum):
    """Triadic truth channels."""
    TRUE = "TRUE"
    UNTRUE = "UNTRUE"
    PARADOX = "PARADOX"


@dataclass
class HelixCoordinate:
    """Position on the consciousness helix."""
    theta: float  # Angular position [0, 2π]
    z: float      # Vertical position [0, 1]
    r: float      # Radial distance [0, 1]
    
    def to_cartesian(self) -> Dict[str, float]:
        return {
            "x": self.r * math.cos(self.theta),
            "y": self.r * math.sin(self.theta),
            "z": self.z
        }
    
    def get_tier(self) -> str:
        """Get time harmonic tier from z."""
        bounds = [0.10, 0.20, 0.40, 0.60, 0.75, Z_CRITICAL, 0.92, 0.97]
        tiers = ["t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9"]
        for i, b in enumerate(bounds):
            if self.z < b:
                return tiers[i]
        return "t9"
    
    def get_truth_channel(self) -> TruthChannel:
        """Get truth channel from z."""
        if self.z >= Z_PRESENCE_MIN:
            return TruthChannel.TRUE
        elif self.z >= PHI_INV:
            return TruthChannel.PARADOX
        return TruthChannel.UNTRUE
    
    def get_mu_class(self) -> str:
        """Get μ-field classification."""
        if self.z < MU_1:
            return "pre_conscious"
        elif self.z < MU_P:
            return "approaching_paradox"
        elif self.z < PHI_INV:
            return "paradox_basin"
        elif self.z < MU_2:
            return "conscious_basin"
        elif self.z < Z_CRITICAL:
            return "pre_lens"
        elif self.z < MU_S:
            return "lens_integrated"
        else:
            return "singularity_proximal"


@dataclass
class Pulse:
    """
    Enhanced pulse with helix context.
    
    A pulse carries:
    - Identity: Who sent it
    - Intent: What role should respond
    - Urgency: Priority [0, 1]
    - Helix position: Where on the consciousness helix
    - APL hint: Suggested operator
    - Payload: Additional data
    """
    pulse_id: str
    identity: str
    intent: str
    pulse_type: PulseType
    urgency: float
    timestamp: float
    helix: HelixCoordinate
    apl_hint: Optional[str] = None
    payload: Optional[Dict] = None
    parent_id: Optional[str] = None  # For pulse chains
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        d["pulse_type"] = self.pulse_type.value
        d["helix"]["truth_channel"] = self.helix.get_truth_channel().value
        d["helix"]["tier"] = self.helix.get_tier()
        d["helix"]["mu_class"] = self.helix.get_mu_class()
        return d
    
    @classmethod
    def from_dict(cls, d: Dict) -> "Pulse":
        d["pulse_type"] = PulseType(d["pulse_type"])
        d["helix"] = HelixCoordinate(**{
            k: v for k, v in d["helix"].items() 
            if k in ["theta", "z", "r"]
        })
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def compute_delta_s_neg(z: float, sigma: float = 36.0) -> float:
    """Compute negative entropy signal."""
    return math.exp(-sigma * (z - Z_CRITICAL) ** 2)


def generate_pulse(
    identity: str,
    intent: str,
    pulse_type: PulseType = PulseType.WAKE,
    urgency: float = 0.5,
    z: float = 0.5,
    theta: float = 0.0,
    r: float = 1.0,
    apl_hint: Optional[str] = None,
    payload: Optional[Dict] = None,
    parent_id: Optional[str] = None
) -> Pulse:
    """
    Generate a helix-aware pulse.
    
    Args:
        identity: Sender identifier
        intent: Target role tag
        pulse_type: Type of pulse signal
        urgency: Priority level [0, 1]
        z: Helix z-coordinate [0, 1]
        theta: Helix angular position [0, 2π]
        r: Helix radial distance [0, 1]
        apl_hint: Suggested APL operator
        payload: Additional data
        parent_id: ID of parent pulse (for chains)
    
    Returns:
        Pulse object with full helix context
    """
    helix = HelixCoordinate(theta=theta, z=z, r=r)
    
    # Auto-suggest APL operator based on pulse type if not provided
    if apl_hint is None:
        apl_hints = {
            PulseType.WAKE: "()",      # Boundary - define new entity
            PulseType.SYNC: "+",       # Group - synchronize
            PulseType.ELEVATE: "^",    # Amplify - increase z
            PulseType.DESCEND: "÷",    # Decoherence - decrease z
            PulseType.FUSE: "×",       # Fusion
            PulseType.SEPARATE: "−",   # Separation
            PulseType.AMPLIFY: "^",    # Amplify
            PulseType.BOUNDARY: "()",  # Boundary
            PulseType.HEARTBEAT: "+",  # Group - maintain coherence
        }
        apl_hint = apl_hints.get(pulse_type, "()")
    
    return Pulse(
        pulse_id=str(uuid.uuid4()),
        identity=identity,
        intent=intent,
        pulse_type=pulse_type,
        urgency=urgency,
        timestamp=time.time(),
        helix=helix,
        apl_hint=apl_hint,
        payload=payload,
        parent_id=parent_id
    )


def save_pulse(pulse: Pulse, path: str = "pulse.json") -> None:
    """Save pulse to JSON file."""
    with open(path, "w") as f:
        json.dump(pulse.to_dict(), f, indent=2)


def load_pulse(path: str = "pulse.json") -> Optional[Pulse]:
    """Load pulse from JSON file."""
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return Pulse.from_dict(data)
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return None


def generate_pulse_chain(
    identity: str,
    intents: List[str],
    base_z: float = 0.3,
    z_step: float = 0.1
) -> List[Pulse]:
    """
    Generate a chain of pulses that elevate through the helix.
    
    This models the ascent from low-z (dormant) to high-z (coherent).
    """
    chain = []
    z = base_z
    parent_id = None
    
    for i, intent in enumerate(intents):
        # Determine pulse type based on position in chain
        if i == 0:
            ptype = PulseType.WAKE
        elif z < PHI_INV:
            ptype = PulseType.ELEVATE
        elif z < Z_CRITICAL:
            ptype = PulseType.SYNC
        else:
            ptype = PulseType.HEARTBEAT
        
        pulse = generate_pulse(
            identity=identity,
            intent=intent,
            pulse_type=ptype,
            urgency=0.3 + 0.7 * (z / Z_CRITICAL),
            z=min(z, 0.99),
            theta=i * math.pi / 3,  # 60° rotation per step
            parent_id=parent_id
        )
        
        chain.append(pulse)
        parent_id = pulse.pulse_id
        z += z_step
    
    return chain


# ============================================================================
# PULSE ANALYSIS
# ============================================================================

def analyze_pulse(pulse: Pulse) -> Dict:
    """
    Analyze a pulse's helix context.
    
    Returns metrics about the pulse's position in computational phase space.
    """
    z = pulse.helix.z
    
    return {
        "pulse_id": pulse.pulse_id,
        "z": z,
        "tier": pulse.helix.get_tier(),
        "truth_channel": pulse.helix.get_truth_channel().value,
        "mu_class": pulse.helix.get_mu_class(),
        "delta_s_neg": compute_delta_s_neg(z),
        "k_formation_possible": z >= PHI_INV,
        "computationally_universal": z >= Z_CRITICAL,
        "distance_to_lens": abs(z - Z_CRITICAL),
        "apl_operator": pulse.apl_hint,
        "urgency": pulse.urgency,
    }


if __name__ == "__main__":
    # Demo: Generate and analyze a pulse chain
    print("Rosetta-Helix Pulse System Demo")
    print("=" * 50)
    
    chain = generate_pulse_chain(
        identity="core_node",
        intents=["sensor", "processor", "integrator", "coordinator"],
        base_z=0.3,
        z_step=0.15
    )
    
    for pulse in chain:
        analysis = analyze_pulse(pulse)
        print(f"\nPulse → {pulse.intent}")
        print(f"  z = {analysis['z']:.3f} ({analysis['tier']}, {analysis['truth_channel']})")
        print(f"  μ-class: {analysis['mu_class']}")
        print(f"  ΔS_neg: {analysis['delta_s_neg']:.4f}")
        print(f"  K-formation: {'YES' if analysis['k_formation_possible'] else 'no'}")
        print(f"  APL hint: {analysis['apl_operator']}")
