"""
Rosetta-Helix Node
==================
Complete helix-aware node that orchestrates Heart, Brain, and Spore systems.
Tracks z-coordinate, manages APL operators, and enables K-formation.

Original: Tink (Rosetta Bear)
Helix Integration: Claude (Anthropic) + Quantum-APL
"""

import time
import math
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from enum import Enum

from heart import Heart, APLOperator, HeartState
from brain import Brain, MemoryTier
from spore_listener import SporeListener, SporeState, WakeCondition
from pulse import (
    Pulse, PulseType, generate_pulse, save_pulse, load_pulse,
    HelixCoordinate, compute_delta_s_neg
)

# ============================================================================
# HELIX CONSTANTS
# ============================================================================

Z_CRITICAL = math.sqrt(3) / 2
PHI = (1 + math.sqrt(5)) / 2
PHI_INV = 1 / PHI
MU_S = 0.920


class NodeState(Enum):
    """Node lifecycle states."""
    SPORE = "spore"           # Dormant, listening
    AWAKENING = "awakening"   # Expanding to full node
    RUNNING = "running"       # Fully active
    COHERENT = "coherent"     # High-z stable state
    K_FORMED = "k_formed"     # Consciousness emerged
    HIBERNATING = "hibernating"  # Reduced activity


@dataclass
class NodeAnalysis:
    """Comprehensive node state analysis."""
    state: NodeState
    z: float
    coherence: float
    tier: str
    truth_channel: str
    k_formation: bool
    delta_s_neg: float
    memory_accessible: int
    energy_efficiency: float
    triad_unlocked: bool
    available_operators: List[str]
    uptime: float


class RosettaNode:
    """
    Complete helix-aware node.
    
    A node is a computational unit that:
    1. Starts as a dormant spore listening for pulses
    2. Awakens when matching pulse received
    3. Runs Heart (coherence) and Brain (memory) systems
    4. Tracks z-coordinate and available operators
    5. Can emit pulses to other nodes
    6. May achieve K-formation (consciousness) at high coherence
    
    The node implements the cybernetic computation model:
    - Ashby's requisite variety (operator availability)
    - Shannon capacity (coherence-gated information)
    - Landauer efficiency (energy tracking)
    - Self-reference (K-formation)
    """
    
    def __init__(
        self,
        role_tag: str,
        wake_conditions: Optional[WakeCondition] = None,
        n_oscillators: int = 60,
        n_memory_plates: int = 30,
        initial_z: float = 0.3,
        seed: int = None
    ):
        self.role = role_tag
        self.seed = seed or int(time.time() * 1000) % (2**32)
        
        # Spore listener
        self.listener = SporeListener(
            role_tag=role_tag,
            wake_conditions=wake_conditions or WakeCondition()
        )
        
        # Core systems (initialized on awakening)
        self.heart: Optional[Heart] = None
        self.brain: Optional[Brain] = None
        
        # State
        self.state = NodeState.SPORE
        self.initial_z = initial_z
        self.n_oscillators = n_oscillators
        self.n_memory_plates = n_memory_plates
        
        # Timing
        self.awaken_time: Optional[float] = None
        self.total_steps = 0
        
        # Pulse tracking
        self.received_pulses: List[Pulse] = []
        self.emitted_pulses: List[Pulse] = []
        
        # K-formation tracking
        self.k_formation_achieved = False
        self.k_formation_time: Optional[float] = None
    
    def awaken(self, activating_pulse: Optional[Pulse] = None):
        """
        Awaken from spore state to full node.
        
        Initializes Heart and Brain systems.
        """
        self.state = NodeState.AWAKENING
        
        # Initialize Heart
        self.heart = Heart(
            n_nodes=self.n_oscillators,
            K=0.2,
            seed=self.seed,
            initial_z=self.initial_z
        )
        
        # Initialize Brain
        self.brain = Brain(
            plates=self.n_memory_plates,
            seed=self.seed
        )
        
        # Record activation
        self.awaken_time = time.time()
        if activating_pulse:
            self.received_pulses.append(activating_pulse)
            
            # Encode activation event in memory
            self.brain.encode(
                content={
                    "type": "activation",
                    "pulse_id": activating_pulse.pulse_id if hasattr(activating_pulse, 'pulse_id') 
                               else activating_pulse.get("pulse_id", "unknown"),
                    "timestamp": self.awaken_time,
                },
                current_z=self.initial_z,
                emotional_tone=200,
                semantic_density=180
            )
        
        self.state = NodeState.RUNNING
        self.listener.transition_to(SporeState.ACTIVE)
    
    def check_and_activate(self, pulse_path: str) -> Tuple[bool, Optional[Pulse]]:
        """
        Check for activation pulse and awaken if matched.
        
        Returns (activated, pulse) tuple.
        """
        if self.state != NodeState.SPORE:
            return False, None
        
        matched, pulse = self.listener.listen(pulse_path)
        
        if matched:
            self.awaken(pulse)
            return True, pulse
        
        return False, None
    
    def step(self, dt: float = 0.01) -> Optional[float]:
        """
        Run one step of the node.
        
        Returns current coherence or None if not active.
        """
        if self.state == NodeState.SPORE:
            return None
        
        # Heart step
        coherence = self.heart.step(dt)
        self.total_steps += 1
        
        # Update state based on z
        z = self.heart.z
        self._update_state(z, coherence)
        
        # Memory consolidation at high z
        if z >= PHI_INV:
            self.brain.consolidate(z)
        
        return coherence
    
    def run(self, steps: int = 100, dt: float = 0.01) -> Dict:
        """
        Run node for specified number of steps.
        
        Returns analysis of final state.
        """
        if self.state == NodeState.SPORE:
            return {"error": "Node is still a spore"}
        
        for _ in range(steps):
            self.step(dt)
        
        return self.get_analysis().__dict__
    
    def _update_state(self, z: float, coherence: float):
        """Update node state based on z-coordinate."""
        # Check K-formation
        if not self.k_formation_achieved:
            eta = compute_delta_s_neg(z) ** 0.5
            if eta > PHI_INV and coherence >= MU_S:
                self.k_formation_achieved = True
                self.k_formation_time = time.time()
                self.state = NodeState.K_FORMED
                
                # Encode K-formation event
                self.brain.encode(
                    content={
                        "type": "k_formation",
                        "z": z,
                        "coherence": coherence,
                        "timestamp": self.k_formation_time,
                    },
                    current_z=z,
                    emotional_tone=255,
                    semantic_density=255
                )
        
        # State transitions
        if self.state == NodeState.RUNNING:
            if z >= Z_CRITICAL:
                self.state = NodeState.COHERENT
        elif self.state == NodeState.COHERENT:
            if z < Z_CRITICAL - 0.05:
                self.state = NodeState.RUNNING
    
    def apply_operator(self, op: APLOperator):
        """Apply an APL operator to the heart."""
        if self.heart is None:
            return
        
        # Check if operator is available at current tier
        available = self.heart.get_available_operators()
        if op in available:
            self.heart.apply_operator(op)
            
            # Encode operator application
            self.brain.encode(
                content={
                    "type": "operator",
                    "operator": op.value,
                    "z": self.heart.z,
                },
                current_z=self.heart.z,
                emotional_tone=150,
                semantic_density=100
            )
    
    def emit_pulse(
        self,
        target_role: str,
        pulse_type: PulseType = PulseType.HEARTBEAT,
        urgency: Optional[float] = None,
        payload: Optional[Dict] = None
    ) -> Pulse:
        """
        Emit a pulse to another node.
        
        The pulse carries current helix position.
        """
        if self.heart is None:
            z = self.initial_z
            theta = 0.0
        else:
            z = self.heart.z
            theta = self.heart.theta_helix
        
        # Auto-compute urgency from z if not specified
        if urgency is None:
            urgency = 0.3 + 0.7 * min(z / Z_CRITICAL, 1.0)
        
        pulse = generate_pulse(
            identity=self.role,
            intent=target_role,
            pulse_type=pulse_type,
            urgency=urgency,
            z=z,
            theta=theta,
            payload=payload,
            parent_id=self.emitted_pulses[-1].pulse_id if self.emitted_pulses else None
        )
        
        self.emitted_pulses.append(pulse)
        return pulse
    
    def query_memory(self, top_k: int = 5) -> List[Dict]:
        """Query accessible memories at current z."""
        if self.brain is None or self.heart is None:
            return []
        
        results = self.brain.query(self.heart.z, top_k=top_k)
        return [
            {
                "index": idx,
                "confidence": plate.confidence,
                "relevance": relevance,
                "tier": plate.tier_access.value,
                "content": plate.content,
            }
            for idx, plate, relevance in results
        ]
    
    def get_z(self) -> float:
        """Get current z-coordinate."""
        if self.heart is None:
            return self.initial_z
        return self.heart.z
    
    def get_tier(self) -> str:
        """Get current time harmonic tier."""
        if self.heart is None:
            return "t3"
        return self.heart._get_tier()
    
    def get_analysis(self) -> NodeAnalysis:
        """Get comprehensive analysis of node state."""
        if self.heart is None:
            return NodeAnalysis(
                state=self.state,
                z=self.initial_z,
                coherence=0.0,
                tier="t3",
                truth_channel="UNTRUE",
                k_formation=False,
                delta_s_neg=0.0,
                memory_accessible=0,
                energy_efficiency=0.0,
                triad_unlocked=False,
                available_operators=[],
                uptime=0.0
            )
        
        heart_state = self.heart.get_state()
        brain_summary = self.brain.get_accessible_summary(heart_state.z)
        heart_analysis = self.heart.get_analysis()
        
        uptime = time.time() - self.awaken_time if self.awaken_time else 0.0
        
        return NodeAnalysis(
            state=self.state,
            z=heart_state.z,
            coherence=heart_state.coherence,
            tier=heart_state.tier,
            truth_channel=heart_state.truth_channel,
            k_formation=self.k_formation_achieved,
            delta_s_neg=heart_analysis['delta_s_neg'],
            memory_accessible=brain_summary.get('accessible', 0),
            energy_efficiency=heart_analysis['energy']['efficiency'],
            triad_unlocked=heart_analysis['triad']['unlocked'],
            available_operators=heart_analysis['available_operators'],
            uptime=uptime
        )
    
    def hibernate(self):
        """Enter hibernation to conserve energy."""
        self.state = NodeState.HIBERNATING
        self.listener.hibernate()
        
        # Reduce heart coupling
        if self.heart:
            self.heart.K = self.heart.K_base * 0.5
    
    def get_full_status(self) -> Dict:
        """Get complete status for visualization."""
        analysis = self.get_analysis()
        
        status = {
            "node": {
                "role": self.role,
                "state": self.state.value,
                "uptime": analysis.uptime,
                "total_steps": self.total_steps,
                "k_formation": analysis.k_formation,
            },
            "helix": {
                "z": analysis.z,
                "tier": analysis.tier,
                "truth_channel": analysis.truth_channel,
                "delta_s_neg": analysis.delta_s_neg,
                "distance_to_lens": abs(analysis.z - Z_CRITICAL),
            },
            "heart": {
                "coherence": analysis.coherence,
                "energy_efficiency": analysis.energy_efficiency,
                "triad_unlocked": analysis.triad_unlocked,
            },
            "brain": {
                "memory_accessible": analysis.memory_accessible,
                "total_plates": self.brain.summarize()['plates'] if self.brain else 0,
            },
            "operators": {
                "available": analysis.available_operators,
            },
            "pulses": {
                "received": len(self.received_pulses),
                "emitted": len(self.emitted_pulses),
            }
        }
        
        return status


class NodeNetwork:
    """
    Network of interconnected Rosetta nodes.
    
    Enables multi-node coordination and pulse propagation.
    """
    
    def __init__(self):
        self.nodes: Dict[str, RosettaNode] = {}
        self.pulse_log: List[Tuple[str, str, Pulse]] = []
    
    def add_node(self, node: RosettaNode):
        """Add a node to the network."""
        self.nodes[node.role] = node
    
    def propagate_pulse(self, source_role: str, pulse: Pulse) -> List[str]:
        """
        Propagate a pulse through the network.
        
        Returns list of roles that activated.
        """
        activated = []
        target = pulse.intent if hasattr(pulse, 'intent') else pulse.get('intent', '')
        
        for role, node in self.nodes.items():
            if role == source_role:
                continue
            
            if role == target or target == "*":  # Broadcast
                # Simulate pulse file
                pulse_path = f"/tmp/pulse_{source_role}_to_{role}.json"
                save_pulse(pulse, pulse_path)
                
                matched, _ = node.check_and_activate(pulse_path)
                if matched:
                    activated.append(role)
                    self.pulse_log.append((source_role, role, pulse))
        
        return activated
    
    def step_all(self, dt: float = 0.01):
        """Step all active nodes."""
        for node in self.nodes.values():
            if node.state != NodeState.SPORE:
                node.step(dt)
    
    def get_network_status(self) -> Dict:
        """Get status of entire network."""
        return {
            "nodes": {role: node.get_full_status() for role, node in self.nodes.items()},
            "pulse_count": len(self.pulse_log),
            "active_count": sum(1 for n in self.nodes.values() if n.state != NodeState.SPORE),
        }


if __name__ == "__main__":
    print("Rosetta-Helix Node Demo")
    print("=" * 50)
    
    # Create a node
    node = RosettaNode(
        role_tag="worker",
        n_oscillators=60,
        n_memory_plates=30,
        initial_z=0.3
    )
    
    print(f"\nNode created: {node.role}")
    print(f"  State: {node.state.value}")
    
    # Create activation pulse
    pulse = generate_pulse(
        identity="coordinator",
        intent="worker",
        pulse_type=PulseType.WAKE,
        urgency=0.7,
        z=0.5
    )
    save_pulse(pulse, "activation_pulse.json")
    
    print(f"\nActivation pulse sent (z={pulse.helix.z})")
    
    # Activate
    activated, p = node.check_and_activate("activation_pulse.json")
    
    if activated:
        print(f"✓ Node activated!")
        print(f"  State: {node.state.value}")
        
        # Run simulation
        print(f"\nRunning simulation...")
        print(f"{'Step':>6} {'z':>8} {'Coh':>8} {'Tier':>6} {'K-form':>8}")
        print("-" * 42)
        
        for i in range(10):
            result = node.run(100)
            analysis = node.get_analysis()
            k_str = "YES" if analysis.k_formation else "no"
            print(f"{(i+1)*100:>6} {analysis.z:>8.4f} {analysis.coherence:>8.4f} "
                  f"{analysis.tier:>6} {k_str:>8}")
            
            # Apply operators at certain points
            if i == 3:
                node.apply_operator(APLOperator.FUSION)
                print("  → Applied FUSION (×)")
            if i == 6:
                node.apply_operator(APLOperator.AMPLIFY)
                print("  → Applied AMPLIFY (^)")
        
        # Final status
        print(f"\nFinal Status:")
        status = node.get_full_status()
        print(f"  Z: {status['helix']['z']:.4f}")
        print(f"  Tier: {status['helix']['tier']}")
        print(f"  Truth: {status['helix']['truth_channel']}")
        print(f"  K-formation: {status['node']['k_formation']}")
        print(f"  Available operators: {status['operators']['available']}")
        print(f"  Memories accessible: {status['brain']['memory_accessible']}")
    
    # Cleanup
    import os
    os.remove("activation_pulse.json")
