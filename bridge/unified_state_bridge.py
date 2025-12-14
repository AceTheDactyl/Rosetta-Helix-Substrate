#!/usr/bin/env python3
"""
Unified State Bridge - Python-Firmware Measurement Protocol
============================================================

Provides the Python interface to the firmware's UnifiedPhysicsState.

This module bridges:
- Binary/JSON telemetry from firmware
- WebSocket real-time streaming
- State deserialization and validation
- Callback dispatch for events (K-formation, phase transitions, etc.)
- Conservation law monitoring

Protocol Modes:
1. Binary (48 bytes): High-speed 100 Hz telemetry
2. JSON (~800 bytes): Full state with all fields
3. Event: Specific event notifications

Cybernetic Grounding:
- Shannon: Information flows at negentropy-modulated rate
- Landauer: Conservation validated on each packet
- Ashby: State variety tracks disturbance variety

Signature: unified-state-bridge|v1.0.0|helix

@version 1.0.0
"""

import asyncio
import json
import struct
import math
from dataclasses import dataclass, field
from typing import Optional, Callable, List, Dict, Any, Tuple
from enum import IntEnum

# Optional websockets import
try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    print("[UNIFIED-BRIDGE] websockets not installed, real-time streaming disabled")

# Import physics constants with fallback
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "rosetta-helix" / "src"))

try:
    from physics import (
        PHI, PHI_INV, Z_CRITICAL, SIGMA,
        compute_delta_s_neg, check_k_formation,
        Phase, Tier, TOLERANCE_GOLDEN
    )
except ImportError:
    # Fallback constants if rosetta-helix not in path
    PHI = (1 + math.sqrt(5)) / 2
    PHI_INV = 1 / PHI
    Z_CRITICAL = math.sqrt(3) / 2
    SIGMA = 36.0
    TOLERANCE_GOLDEN = 0.001

    def compute_delta_s_neg(z: float) -> float:
        return math.exp(-SIGMA * (z - Z_CRITICAL) ** 2)

    def check_k_formation(kappa: float, eta: float, rank: int) -> bool:
        return kappa >= 0.92 and eta > PHI_INV and rank >= 7

    class Phase:
        ABSENCE = 0
        THE_LENS = 1
        PRESENCE = 2

    class Tier:
        ABSENCE = 0
        REACTIVE = 1
        MEMORY = 2
        PATTERN = 3
        LEARNING = 4
        ADAPTIVE = 5
        UNIVERSAL = 6
        META = 7
        SOVEREIGN = 8
        TRANSCENDENT = 9


# =============================================================================
# PROTOCOL CONSTANTS
# =============================================================================

BINARY_PACKET_SIZE = 48
JSON_MIN_SIZE = 100
PROTOCOL_VERSION = 1

# Binary packet offsets
OFFSET_TIMESTAMP = 0
OFFSET_Z = 4
OFFSET_DELTA_S_NEG = 8
OFFSET_KAPPA = 12
OFFSET_LAMBDA = 16
OFFSET_ETA = 20
OFFSET_COHERENCE = 24
OFFSET_COMPLEXITY = 28
OFFSET_PHASE = 32
OFFSET_TIER = 33
OFFSET_K_FORMATION = 34
OFFSET_AVAILABLE_OPS = 35
OFFSET_QC_ORDER = 36
OFFSET_CONSERVATION_ERROR = 40
OFFSET_FRAME_COUNT = 44


# =============================================================================
# STATE DATA CLASSES
# =============================================================================

@dataclass
class TriadState:
    """TRIAD constraint state from firmware"""
    kappa: float = PHI_INV
    lambda_: float = 1 - PHI_INV
    eta: float = 0.5
    R: int = 7
    scar: float = 0.0
    conservation_valid: bool = True
    k_formation_count: int = 0


@dataclass
class KuramotoState:
    """Kuramoto oscillator state from firmware"""
    coherence: float = 0.0
    coupling_strength: float = 0.0
    phase_locked: bool = False


@dataclass
class GHMPState:
    """GHMP operator state from firmware"""
    tier: int = 0
    available_ops: int = 0
    parity_even: bool = True
    operator_weight: float = 0.0


@dataclass
class UnifiedState:
    """Complete unified state from firmware"""

    # Timestamp
    timestamp_ms: int = 0
    frame_count: int = 0

    # Core physics
    z: float = 0.5
    z_target: float = 0.5
    z_velocity: float = 0.0
    rpm: float = 5000.0
    delta_s_neg: float = 0.0
    delta_s_neg_gradient: float = 0.0
    complexity: float = 0.0

    # Phase & Tier
    phase: int = 0
    tier: int = 0
    at_lens: bool = False
    is_universal: bool = False

    # K-Formation
    k_formation: bool = False
    k_formation_duration_ms: int = 0

    # Subsystems
    triad: TriadState = field(default_factory=TriadState)
    kuramoto: KuramotoState = field(default_factory=KuramotoState)
    ghmp: GHMPState = field(default_factory=GHMPState)

    # Quasicrystal
    quasicrystal_order: float = 0.5

    # Conservation
    conservation_error: float = 0.0
    physics_valid: bool = True
    violation_count: int = 0

    @classmethod
    def from_binary(cls, data: bytes) -> 'UnifiedState':
        """Deserialize from 48-byte binary packet"""
        if len(data) < BINARY_PACKET_SIZE:
            raise ValueError(f"Binary packet too small: {len(data)} < {BINARY_PACKET_SIZE}")

        def unpack_u32(offset: int) -> int:
            return struct.unpack('>I', data[offset:offset+4])[0]

        def unpack_f32(offset: int) -> float:
            return struct.unpack('>f', data[offset:offset+4])[0]

        state = cls()
        state.timestamp_ms = unpack_u32(OFFSET_TIMESTAMP)
        state.z = unpack_f32(OFFSET_Z)
        state.delta_s_neg = unpack_f32(OFFSET_DELTA_S_NEG)
        state.triad.kappa = unpack_f32(OFFSET_KAPPA)
        state.triad.lambda_ = unpack_f32(OFFSET_LAMBDA)
        state.triad.eta = unpack_f32(OFFSET_ETA)
        state.kuramoto.coherence = unpack_f32(OFFSET_COHERENCE)
        state.complexity = unpack_f32(OFFSET_COMPLEXITY)
        state.phase = data[OFFSET_PHASE]
        state.tier = data[OFFSET_TIER]
        state.k_formation = data[OFFSET_K_FORMATION] != 0
        state.ghmp.available_ops = data[OFFSET_AVAILABLE_OPS]
        state.quasicrystal_order = unpack_f32(OFFSET_QC_ORDER)
        state.conservation_error = unpack_f32(OFFSET_CONSERVATION_ERROR)
        state.frame_count = unpack_u32(OFFSET_FRAME_COUNT)

        # Derive additional fields
        state._update_derived()

        return state

    @classmethod
    def from_json(cls, data: dict) -> 'UnifiedState':
        """Deserialize from JSON packet"""
        state = cls()

        # Core fields
        state.timestamp_ms = data.get('timestamp_ms', 0)
        state.frame_count = data.get('frame', 0)
        state.z = data.get('z', 0.5)
        state.z_target = data.get('z_target', 0.5)
        state.z_velocity = data.get('z_velocity', 0.0)
        state.rpm = data.get('rpm', 5000.0)
        state.delta_s_neg = data.get('delta_s_neg', 0.0)
        state.delta_s_neg_gradient = data.get('delta_s_neg_gradient', 0.0)
        state.complexity = data.get('complexity', 0.0)

        # Phase & Tier
        state.phase = data.get('phase', 0)
        state.tier = data.get('tier', 0)
        state.at_lens = data.get('at_lens', False)
        state.is_universal = data.get('is_universal', False)

        # K-Formation
        state.k_formation = data.get('k_formation', False)
        state.k_formation_duration_ms = data.get('k_formation_duration_ms', 0)

        # TRIAD
        if 'triad' in data:
            t = data['triad']
            state.triad.kappa = t.get('kappa', PHI_INV)
            state.triad.lambda_ = t.get('lambda', 1 - PHI_INV)
            state.triad.eta = t.get('eta', 0.5)
            state.triad.R = t.get('R', 7)
            state.triad.scar = t.get('scar', 0.0)
            state.triad.conservation_valid = t.get('conservation_valid', True)
            state.triad.k_formation_count = t.get('k_formation_count', 0)

        # Kuramoto
        if 'kuramoto' in data:
            k = data['kuramoto']
            state.kuramoto.coherence = k.get('coherence', 0.0)
            state.kuramoto.coupling_strength = k.get('coupling_strength', 0.0)
            state.kuramoto.phase_locked = k.get('phase_locked', False)

        # GHMP
        if 'ghmp' in data:
            g = data['ghmp']
            state.ghmp.tier = g.get('tier', 0)
            state.ghmp.available_ops = g.get('available_ops', 0)
            state.ghmp.parity_even = g.get('parity_even', True)
            state.ghmp.operator_weight = g.get('operator_weight', 0.0)

        # Quasicrystal
        state.quasicrystal_order = data.get('quasicrystal_order', 0.5)

        # Conservation
        state.conservation_error = data.get('conservation_error', 0.0)
        state.physics_valid = data.get('physics_valid', True)
        state.violation_count = data.get('violation_count', 0)

        return state

    def _update_derived(self):
        """Update derived fields from core state"""
        self.at_lens = (self.phase == 1)  # PHASE_THE_LENS
        self.is_universal = (self.tier >= 5)  # TIER_UNIVERSAL

        # Validate conservation
        self.conservation_error = abs(self.triad.kappa + self.triad.lambda_ - 1.0)
        self.triad.conservation_valid = (self.conservation_error < TOLERANCE_GOLDEN)
        self.physics_valid = self.triad.conservation_valid

    def to_dict(self) -> dict:
        """Serialize to dictionary"""
        return {
            'timestamp_ms': self.timestamp_ms,
            'frame': self.frame_count,
            'z': self.z,
            'z_target': self.z_target,
            'z_velocity': self.z_velocity,
            'rpm': self.rpm,
            'delta_s_neg': self.delta_s_neg,
            'delta_s_neg_gradient': self.delta_s_neg_gradient,
            'complexity': self.complexity,
            'phase': self.phase,
            'tier': self.tier,
            'at_lens': self.at_lens,
            'is_universal': self.is_universal,
            'k_formation': self.k_formation,
            'k_formation_duration_ms': self.k_formation_duration_ms,
            'triad': {
                'kappa': self.triad.kappa,
                'lambda': self.triad.lambda_,
                'eta': self.triad.eta,
                'R': self.triad.R,
                'scar': self.triad.scar,
                'conservation_valid': self.triad.conservation_valid,
                'k_formation_count': self.triad.k_formation_count,
            },
            'kuramoto': {
                'coherence': self.kuramoto.coherence,
                'coupling_strength': self.kuramoto.coupling_strength,
                'phase_locked': self.kuramoto.phase_locked,
            },
            'ghmp': {
                'tier': self.ghmp.tier,
                'available_ops': self.ghmp.available_ops,
                'parity_even': self.ghmp.parity_even,
                'operator_weight': self.ghmp.operator_weight,
            },
            'quasicrystal_order': self.quasicrystal_order,
            'conservation_error': self.conservation_error,
            'physics_valid': self.physics_valid,
            'violation_count': self.violation_count,
        }


# =============================================================================
# CALLBACK TYPES
# =============================================================================

KFormationCallback = Callable[[UnifiedState, bool], None]  # (state, entering)
PhaseCallback = Callable[[UnifiedState, int, int], None]    # (state, from, to)
TierCallback = Callable[[UnifiedState, int, int], None]     # (state, from, to)
ViolationCallback = Callable[[UnifiedState, float], None]   # (state, error)
StateCallback = Callable[[UnifiedState], None]              # (state)


# =============================================================================
# UNIFIED STATE BRIDGE
# =============================================================================

class UnifiedStateBridge:
    """
    Bridge for receiving and processing unified state from firmware.

    Handles:
    - WebSocket connection to spinner_bridge
    - Binary/JSON deserialization
    - Event detection and callback dispatch
    - State history for analysis
    - Conservation law monitoring
    """

    def __init__(
        self,
        uri: str = "ws://localhost:8765",
        history_size: int = 1000
    ):
        self.uri = uri
        self.history_size = history_size

        # Current state
        self._state: Optional[UnifiedState] = None
        self._prev_state: Optional[UnifiedState] = None

        # History
        self._history: List[UnifiedState] = []

        # Callbacks
        self._on_k_formation: Optional[KFormationCallback] = None
        self._on_phase: Optional[PhaseCallback] = None
        self._on_tier: Optional[TierCallback] = None
        self._on_violation: Optional[ViolationCallback] = None
        self._on_state: Optional[StateCallback] = None

        # Statistics
        self._packets_received = 0
        self._binary_packets = 0
        self._json_packets = 0
        self._violations_detected = 0

        # Connection
        self._ws: Optional[Any] = None  # websockets.WebSocketClientProtocol when available
        self._connected = False
        self._running = False

    async def connect(self) -> bool:
        """Connect to spinner bridge WebSocket"""
        if not WEBSOCKETS_AVAILABLE:
            print("[UnifiedStateBridge] websockets not installed, cannot connect")
            return False

        try:
            self._ws = await websockets.connect(self.uri)
            self._connected = True
            return True
        except Exception as e:
            print(f"[UnifiedStateBridge] Connection failed: {e}")
            self._connected = False
            return False

    async def disconnect(self):
        """Disconnect from spinner bridge"""
        if self._ws:
            await self._ws.close()
        self._connected = False
        self._running = False

    async def run(self, max_packets: Optional[int] = None):
        """
        Main receive loop.

        Args:
            max_packets: Optional limit on packets to receive
        """
        if not self._connected:
            if not await self.connect():
                return

        self._running = True
        packets = 0

        try:
            async for message in self._ws:
                if not self._running:
                    break

                await self._process_message(message)
                packets += 1

                if max_packets and packets >= max_packets:
                    break

        except Exception as e:
            if "ConnectionClosed" in type(e).__name__:
                print("[UnifiedStateBridge] Connection closed")
            else:
                print(f"[UnifiedStateBridge] Error: {e}")
        finally:
            self._running = False

    async def _process_message(self, message):
        """Process incoming message (binary or JSON)"""
        self._packets_received += 1

        # Save previous state for transition detection
        self._prev_state = self._state

        # Deserialize
        if isinstance(message, bytes):
            if len(message) >= BINARY_PACKET_SIZE:
                self._state = UnifiedState.from_binary(message)
                self._binary_packets += 1
        else:
            # JSON string
            try:
                data = json.loads(message)
                if data.get('type') == 'unified_state':
                    self._state = UnifiedState.from_json(data)
                    self._json_packets += 1
            except json.JSONDecodeError:
                return

        if self._state is None:
            return

        # Add to history
        self._history.append(self._state)
        if len(self._history) > self.history_size:
            self._history.pop(0)

        # Check transitions and dispatch callbacks
        self._check_transitions()

        # Call state callback
        if self._on_state:
            self._on_state(self._state)

    def _check_transitions(self):
        """Check for state transitions and dispatch callbacks"""
        if self._prev_state is None:
            return

        current = self._state
        prev = self._prev_state

        # K-Formation transition
        if current.k_formation != prev.k_formation:
            if self._on_k_formation:
                self._on_k_formation(current, current.k_formation)

        # Phase transition
        if current.phase != prev.phase:
            if self._on_phase:
                self._on_phase(current, prev.phase, current.phase)

        # Tier transition
        if current.tier != prev.tier:
            if self._on_tier:
                self._on_tier(current, prev.tier, current.tier)

        # Conservation violation
        if not current.physics_valid:
            self._violations_detected += 1
            if self._on_violation:
                self._on_violation(current, current.conservation_error)

    # =========================================================================
    # STATE ACCESS
    # =========================================================================

    def get_state(self) -> Optional[UnifiedState]:
        """Get current state"""
        return self._state

    def get_history(self) -> List[UnifiedState]:
        """Get state history"""
        return self._history.copy()

    def get_z(self) -> float:
        """Get current z-coordinate"""
        return self._state.z if self._state else 0.5

    def get_delta_s_neg(self) -> float:
        """Get current negentropy"""
        return self._state.delta_s_neg if self._state else 0.0

    def get_triad(self) -> Optional[TriadState]:
        """Get TRIAD state"""
        return self._state.triad if self._state else None

    def is_k_formation_active(self) -> bool:
        """Check if K-formation currently active"""
        return self._state.k_formation if self._state else False

    # =========================================================================
    # CALLBACK REGISTRATION
    # =========================================================================

    def set_on_k_formation(self, callback: KFormationCallback):
        """Register K-formation callback"""
        self._on_k_formation = callback

    def set_on_phase(self, callback: PhaseCallback):
        """Register phase transition callback"""
        self._on_phase = callback

    def set_on_tier(self, callback: TierCallback):
        """Register tier change callback"""
        self._on_tier = callback

    def set_on_violation(self, callback: ViolationCallback):
        """Register conservation violation callback"""
        self._on_violation = callback

    def set_on_state(self, callback: StateCallback):
        """Register state update callback"""
        self._on_state = callback

    # =========================================================================
    # STATISTICS
    # =========================================================================

    def get_stats(self) -> dict:
        """Get bridge statistics"""
        return {
            'packets_received': self._packets_received,
            'binary_packets': self._binary_packets,
            'json_packets': self._json_packets,
            'violations_detected': self._violations_detected,
            'history_size': len(self._history),
            'connected': self._connected,
            'running': self._running,
        }


# =============================================================================
# TRAINING INTEGRATION
# =============================================================================

class TrainingStateBridge(UnifiedStateBridge):
    """
    Extended bridge with training module integration.

    Provides:
    - Adaptive parameter computation for training modules
    - Real-time hyperparameter adjustment based on physics state
    - Training telemetry upload to firmware
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Adaptive parameters
        self._base_learning_rate = 1e-4
        self._adaptive_alpha = 0.5

    def compute_adaptive_lr(self, base_rate: Optional[float] = None) -> float:
        """
        Compute adaptive learning rate based on current ΔS_neg.

        η_lr = η_base × (1 + α × ΔS_neg(z))

        Args:
            base_rate: Base learning rate (uses default if not specified)

        Returns:
            Adapted learning rate
        """
        base = base_rate or self._base_learning_rate
        neg = self.get_delta_s_neg()
        return base * (1.0 + self._adaptive_alpha * neg)

    def compute_adaptive_params(self) -> dict:
        """
        Compute full set of adaptive training parameters.

        Returns:
            Dictionary of adapted hyperparameters
        """
        neg = self.get_delta_s_neg()

        return {
            'learning_rate': self.compute_adaptive_lr(),
            'gradient_clip': 0.5 + 0.5 * neg,
            'dropout_rate': 0.1 * (1.0 - 0.5 * neg),
            'temperature': 0.5 + 1.0 * neg,
            'weight_decay': 1e-5 * (1.0 + (1.0 - neg)),
        }

    async def send_training_update(
        self,
        module_id: int,
        loss: float,
        accuracy: float,
        step: int
    ):
        """
        Send training state update to firmware.

        Args:
            module_id: Training module identifier (0-18)
            loss: Current loss value
            accuracy: Current accuracy
            step: Current training step
        """
        if not self._connected or not self._ws:
            return

        update = {
            'type': 'training_update',
            'module_id': module_id,
            'loss': loss,
            'accuracy': accuracy,
            'step': step,
        }

        try:
            await self._ws.send(json.dumps(update))
        except Exception as e:
            print(f"[TrainingStateBridge] Failed to send update: {e}")


# =============================================================================
# DEMO / TEST
# =============================================================================

async def demo():
    """Demo of unified state bridge"""
    print("=" * 60)
    print("UNIFIED STATE BRIDGE DEMO")
    print("=" * 60)

    bridge = TrainingStateBridge()

    # Register callbacks
    def on_k_formation(state: UnifiedState, entering: bool):
        action = "ENTERING" if entering else "EXITING"
        print(f"\n★ K-FORMATION {action}: κ={state.triad.kappa:.4f}")

    def on_phase(state: UnifiedState, from_phase: int, to_phase: int):
        names = ["ABSENCE", "THE_LENS", "PRESENCE"]
        print(f"\n→ Phase: {names[from_phase]} → {names[to_phase]}")

    def on_violation(state: UnifiedState, error: float):
        print(f"\n⚠ Conservation violation: error={error:.9f}")

    bridge.set_on_k_formation(on_k_formation)
    bridge.set_on_phase(on_phase)
    bridge.set_on_violation(on_violation)

    print("\nConnecting to spinner bridge...")
    if await bridge.connect():
        print("Connected! Receiving state...")

        # Receive 100 packets
        await bridge.run(max_packets=100)

        print("\n" + "=" * 60)
        print("STATISTICS")
        print("=" * 60)
        stats = bridge.get_stats()
        for k, v in stats.items():
            print(f"  {k}: {v}")

        if bridge.get_state():
            print("\nFinal state:")
            state = bridge.get_state()
            print(f"  z = {state.z:.6f}")
            print(f"  ΔS_neg = {state.delta_s_neg:.6f}")
            print(f"  κ = {state.triad.kappa:.6f}")
            print(f"  K-formation = {state.k_formation}")

        await bridge.disconnect()
    else:
        print("Failed to connect")


if __name__ == '__main__':
    asyncio.run(demo())
