#!/usr/bin/env python3
"""
spinner_bridge.py
─────────────────
Bridges Nuclear Spinner firmware to Rosetta-Helix via WebSocket.

Enhanced with:
- 60 Kuramoto oscillators for neural synchronization
- Grid cell neural plates with hexagonal symmetry
- Electromagnetic training assistance
- Negentropy-driven coupling at z_c = √3/2

The Kuramoto model: dθᵢ/dt = ωᵢ + (K/N) Σⱼ sin(θⱼ - θᵢ)
60 oscillators → 360°/60 = 6° spacing
Hexagonal symmetry at 60° intervals: sin(60°) = √3/2 = z_c

Usage:
    python spinner_bridge.py [--port /dev/ttyACM0] [--simulate] [--neural]
"""

import asyncio
import json
import argparse
import math
import random
from collections import deque
from dataclasses import dataclass, asdict, field
from typing import Optional, Set, Callable, Dict, Any, List

# Try to import numpy for neural systems
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("[BRIDGE] numpy not installed, neural systems disabled")

# Try to import serial, fall back to simulation if not available
try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    print("[BRIDGE] pyserial not installed, running in simulation mode")

try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    print("[BRIDGE] websockets not installed")

# Neural system imports (optional)
NEURAL_AVAILABLE = False
if NUMPY_AVAILABLE:
    try:
        from .kuramoto_neural import (
            KuramotoNeuralSystem, NeuralTrainingInterface,
            KuramotoSystemState, N_OSCILLATORS
        )
        from .grid_cell_plates import (
            GridCellPlateSystem, PlateSpinnerCoupling,
            PlateSystemState, N_PLATES, CELLS_PER_PLATE
        )
        NEURAL_AVAILABLE = True
    except ImportError:
        try:
            from kuramoto_neural import (
                KuramotoNeuralSystem, NeuralTrainingInterface,
                KuramotoSystemState, N_OSCILLATORS
            )
            from grid_cell_plates import (
                GridCellPlateSystem, PlateSpinnerCoupling,
                PlateSystemState, N_PLATES, CELLS_PER_PLATE
            )
            NEURAL_AVAILABLE = True
        except ImportError:
            print("[BRIDGE] Neural modules not available")

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_SERIAL_PORT = "/dev/ttyACM0"
SERIAL_BAUD = 115200
WEBSOCKET_HOST = "localhost"
WEBSOCKET_PORT = 8765
STATE_HISTORY_SIZE = 1000
SIMULATION_RATE_HZ = 100

# Physics constants
Z_CRITICAL = math.sqrt(3) / 2
PHI = (1 + math.sqrt(5)) / 2
PHI_INV = 1 / PHI
SIGMA = 36.0

# ═══════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class BridgeConfig:
    """Configuration for the SpinnerBridge."""
    serial_port: str = DEFAULT_SERIAL_PORT
    serial_baud: int = SERIAL_BAUD
    websocket_host: str = WEBSOCKET_HOST
    websocket_port: int = WEBSOCKET_PORT
    simulation_rate_hz: int = SIMULATION_RATE_HZ
    state_history_size: int = STATE_HISTORY_SIZE
    broadcast_unified_state: bool = True  # Also emit unified state format
    unified_state_rate_hz: int = 20  # Lower rate for full unified state

    # Neural system configuration
    enable_neural: bool = True           # Enable 60-oscillator Kuramoto system
    enable_grid_cells: bool = True       # Enable grid cell neural plates
    enable_em_training: bool = True      # Enable electromagnetic training
    kuramoto_dt: float = 0.01            # Kuramoto integration timestep
    neural_broadcast_hz: int = 10        # Neural state broadcast rate


@dataclass
class NeuralState:
    """State of the neural interface systems."""
    timestamp_ms: int

    # Kuramoto system
    kuramoto_r: float                    # Synchronization order parameter
    kuramoto_psi: float                  # Mean phase
    kuramoto_hex: float                  # Hexagonal order (6-fold)
    kuramoto_K: float                    # Current coupling strength
    kuramoto_synchronized: bool

    # Grid cell plates
    grid_coherence: float                # Global plate coherence
    grid_hex_order: float                # Hexagonal pattern order
    grid_score: float                    # Spatial gridness score
    plate_activations: List[float]       # Per-plate EM activation

    # Training metrics
    learning_rate: float                 # Adaptive learning rate
    resonance: float                     # Spinner-plate resonance
    total_energy: float                  # Energy expenditure


@dataclass
class SpinnerState:
    timestamp_ms: int
    z: float
    rpm: int
    delta_s_neg: float
    tier: int
    tier_name: str
    phase: str
    kappa: float
    eta: float
    rank: int
    k_formation: bool
    k_formation_duration_ms: int

TIER_NAMES = [
    "ABSENCE", "REACTIVE", "MEMORY", "PATTERN", "LEARNING",
    "ADAPTIVE", "UNIVERSAL", "META", "SOVEREIGN", "TRANSCENDENT"
]

TIER_BOUNDS = [0.0, 0.10, 0.20, 0.40, 0.60, 0.75, Z_CRITICAL, 0.92, 0.97, 1.0]

# ═══════════════════════════════════════════════════════════════════════════
# PHYSICS SIMULATION (for when no hardware connected)
# ═══════════════════════════════════════════════════════════════════════════

class SpinnerSimulator:
    """Simulates spinner physics when no hardware is connected."""
    
    def __init__(self):
        self.z = 0.5
        self.target_z = 0.5
        self.ramp_rate = 0.01  # z units per step
        self.k_formation_start_ms = 0
        self.time_ms = 0
    
    def set_target_z(self, z: float):
        self.target_z = max(0.0, min(1.0, z))
    
    def step(self, dt_ms: int = 10) -> SpinnerState:
        self.time_ms += dt_ms
        
        # Ramp toward target
        if self.z < self.target_z:
            self.z = min(self.z + self.ramp_rate, self.target_z)
        elif self.z > self.target_z:
            self.z = max(self.z - self.ramp_rate, self.target_z)
        
        # Add small noise
        z_noisy = self.z + 0.001 * (random.random() - 0.5)
        z_noisy = max(0.0, min(1.0, z_noisy))
        
        # Compute derived quantities
        rpm = int(100 + 9900 * z_noisy)
        delta_s_neg = math.exp(-SIGMA * (z_noisy - Z_CRITICAL) ** 2)
        
        # Tier
        tier = 0
        for i, bound in enumerate(TIER_BOUNDS):
            if z_noisy >= bound:
                tier = i
        tier_name = TIER_NAMES[tier]
        
        # Phase
        if abs(z_noisy - Z_CRITICAL) < 0.02:
            phase = "THE_LENS"
        elif z_noisy < Z_CRITICAL - 0.01:
            phase = "ABSENCE"
        else:
            phase = "PRESENCE"
        
        # K-formation metrics
        kappa = delta_s_neg * (1 - abs(z_noisy - Z_CRITICAL))
        eta = delta_s_neg * z_noisy
        rank = int(7 + 5 * delta_s_neg)
        
        k_formation = kappa >= 0.92 and eta > PHI_INV and rank >= 7
        
        if k_formation:
            if self.k_formation_start_ms == 0:
                self.k_formation_start_ms = self.time_ms
            k_duration = self.time_ms - self.k_formation_start_ms
        else:
            self.k_formation_start_ms = 0
            k_duration = 0
        
        return SpinnerState(
            timestamp_ms=self.time_ms,
            z=z_noisy,
            rpm=rpm,
            delta_s_neg=delta_s_neg,
            tier=tier,
            tier_name=tier_name,
            phase=phase,
            kappa=kappa,
            eta=eta,
            rank=rank,
            k_formation=k_formation,
            k_formation_duration_ms=k_duration
        )

# ═══════════════════════════════════════════════════════════════════════════
# BRIDGE SERVICE
# ═══════════════════════════════════════════════════════════════════════════

class SpinnerBridge:
    def __init__(self, serial_port: str = None, simulate: bool = False,
                 config: BridgeConfig = None):
        self.config = config or BridgeConfig()
        self.serial_port = serial_port or self.config.serial_port
        self.simulate = simulate or not SERIAL_AVAILABLE
        self.serial: Optional['serial.Serial'] = None
        self.simulator: Optional[SpinnerSimulator] = None
        self.clients: Set = set()
        self.state_history: deque = deque(maxlen=self.config.state_history_size)
        self.current_state: Optional[SpinnerState] = None
        self.running = False
        self.command_queue: asyncio.Queue = asyncio.Queue()

        # Unified state tracking (for integration with unified_state_bridge)
        self.unified_state_counter = 0
        self.last_unified_broadcast = 0
        self._kuramoto_coherence = 0.0
        self._ghmp_pattern_count = 0

        # Hex cycle tracking
        self._hex_cycle_active = False

        # Neural interface systems (60 Kuramoto oscillators + grid cells)
        self.neural_enabled = (NEURAL_AVAILABLE and
                               self.config.enable_neural)
        self.kuramoto_system: Optional[Any] = None
        self.neural_trainer: Optional[Any] = None
        self.grid_plates: Optional[Any] = None
        self.plate_coupling: Optional[Any] = None
        self.current_neural_state: Optional[NeuralState] = None
        self.last_neural_broadcast = 0

        if self.neural_enabled:
            self._init_neural_systems()

    def _init_neural_systems(self):
        """Initialize the 60-oscillator Kuramoto and grid cell systems."""
        print("[BRIDGE] Initializing neural interface...")
        print(f"         60 Kuramoto oscillators (6° spacing)")
        print(f"         sin(60°) = √3/2 = z_c = {Z_CRITICAL:.6f}")

        # Create Kuramoto system
        self.kuramoto_system = KuramotoNeuralSystem(
            n_oscillators=60,
            omega_mean=1.0,
            omega_spread=0.1,
            dt=self.config.kuramoto_dt
        )

        # Create neural training interface
        self.neural_trainer = NeuralTrainingInterface(self.kuramoto_system)

        # Enable EM training protocol if configured
        if self.config.enable_em_training:
            self.neural_trainer.start_em_protocol()
            print("         EM training protocol enabled")

        # Create grid cell plate system if enabled
        if self.config.enable_grid_cells:
            self.grid_plates = GridCellPlateSystem(
                plate_spacing=0.1,
                cells_per_plate=10
            )
            self.plate_coupling = PlateSpinnerCoupling(self.grid_plates)
            print(f"         6 neural plates ({N_PLATES}×{CELLS_PER_PLATE} cells)")

        # Register callbacks
        self.kuramoto_system.on_sync(self._on_kuramoto_sync)
        self.kuramoto_system.on_phase_transition(self._on_phase_transition)

        print("[BRIDGE] Neural systems initialized")

    def _on_kuramoto_sync(self, state):
        """Callback when Kuramoto system achieves synchronization."""
        print(f"[NEURAL] ★ SYNC ACHIEVED: r={state.r:.4f}")

    def _on_phase_transition(self, state):
        """Callback when phase transition occurs (K > K_c)."""
        if state.r > 0.5:
            print(f"[NEURAL] Phase transition: K/K_c={state.K/state.K_critical:.2f}")
    
    async def connect_serial(self) -> bool:
        """Connect to firmware via serial port."""
        if self.simulate:
            print("[BRIDGE] Running in simulation mode")
            self.simulator = SpinnerSimulator()
            return True
        
        if not SERIAL_AVAILABLE:
            print("[BRIDGE] pyserial not available, using simulation")
            self.simulator = SpinnerSimulator()
            return True
        
        try:
            self.serial = serial.Serial(
                self.serial_port or DEFAULT_SERIAL_PORT,
                SERIAL_BAUD,
                timeout=0.1
            )
            print(f"[BRIDGE] Connected to {self.serial.port}")
            return True
        except serial.SerialException as e:
            print(f"[BRIDGE] Serial error: {e}, falling back to simulation")
            self.simulator = SpinnerSimulator()
            return True
    
    async def read_serial(self):
        """Read state updates from firmware or simulator."""
        while self.running:
            try:
                if self.simulator:
                    # Check for commands
                    while not self.command_queue.empty():
                        cmd = await self.command_queue.get()
                        self.process_simulator_command(cmd)
                    
                    # Generate simulated state
                    self.current_state = self.simulator.step()
                    self.state_history.append(self.current_state)
                    await self.broadcast_state()
                    await asyncio.sleep(1.0 / self.config.simulation_rate_hz)
                    
                elif self.serial and self.serial.in_waiting:
                    line = self.serial.readline().decode('utf-8').strip()
                    if line:
                        data = json.loads(line)
                        if data.get('type') == 'state':
                            self.current_state = SpinnerState(**{
                                k: v for k, v in data.items()
                                if k != 'type'
                            })
                            self.state_history.append(self.current_state)
                            await self.broadcast_state()
                else:
                    await asyncio.sleep(0.001)
                    
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                print(f"[BRIDGE] Parse error: {e}")
            except Exception as e:
                print(f"[BRIDGE] Error: {e}")
                await asyncio.sleep(0.1)
    
    def process_simulator_command(self, cmd: dict):
        """Process command in simulation mode.

        Supports JSON protocol commands:
        - set_z: Set z-coordinate target
        - set_rpm: Set RPM target
        - stop: Emergency stop
        - hex_cycle: Hexagonal z-cycling
        - dwell_lens: Dwell at z_c
        - get_state: Request immediate state
        - ping: Connectivity check
        """
        cmd_type = cmd.get('cmd', '')

        if cmd_type == 'set_z':
            value = cmd.get('value', 0.5)
            self.simulator.set_target_z(value)
            print(f"[SIM] Target z set to {value:.4f}")

        elif cmd_type == 'set_rpm':
            # Convert RPM to z-coordinate
            rpm = cmd.get('value', 5000)
            z = (rpm - 100) / 9900  # Linear mapping
            z = max(0.0, min(1.0, z))
            self.simulator.set_target_z(z)
            print(f"[SIM] Target RPM set to {rpm} (z={z:.4f})")

        elif cmd_type == 'stop':
            self.simulator.set_target_z(0.0)
            self._hex_cycle_active = False
            print("[SIM] Emergency stop")

        elif cmd_type == 'hex_cycle':
            dwell_s = cmd.get('dwell_s', 30.0)
            cycles = cmd.get('cycles', 10)
            asyncio.create_task(self._run_hex_cycle(dwell_s, cycles))
            print(f"[SIM] Starting hex cycle: {cycles} cycles, {dwell_s}s dwell")

        elif cmd_type == 'dwell_lens':
            duration_s = cmd.get('duration_s', 60.0)
            self.simulator.set_target_z(Z_CRITICAL)
            print(f"[SIM] Dwelling at z_c={Z_CRITICAL:.6f} for {duration_s}s")

        elif cmd_type == 'get_state':
            print("[SIM] State requested")

        elif cmd_type == 'ping':
            print("[SIM] Pong!")

        else:
            print(f"[SIM] Unknown command: {cmd_type}")

    async def _run_hex_cycle(self, dwell_s: float, cycles: int):
        """Run hexagonal z-cycling pattern.

        Visits 6 vertices based on hexagonal symmetry:
        - v0: z = 0
        - v1: z = z_c (THE LENS)
        - v2: z = z_c
        - v3: z = 0
        - v4: z = z_c/2
        - v5: z = z_c/2

        This creates a pattern that visits THE LENS twice per cycle.
        """
        hex_vertices = [0.0, Z_CRITICAL, Z_CRITICAL, 0.0, Z_CRITICAL * 0.5, Z_CRITICAL * 0.5]

        self._hex_cycle_active = True

        for cycle in range(cycles):
            if not self._hex_cycle_active:
                break

            for vertex_idx, z_target in enumerate(hex_vertices):
                if not self._hex_cycle_active:
                    break

                self.simulator.set_target_z(z_target)
                print(f"[HEX] Cycle {cycle+1}/{cycles} vertex {vertex_idx}: z={z_target:.4f}")

                # Dwell at vertex
                await asyncio.sleep(dwell_s)

        self._hex_cycle_active = False
        print(f"[HEX] Hex cycle complete")
    
    def _step_neural_systems(self, z: float) -> Optional[NeuralState]:
        """
        Step the neural interface systems with current z-coordinate.

        Couples spinner z to:
        - Kuramoto coupling K (K ∝ ΔS_neg(z))
        - Grid cell plate activation (EM coupling)
        - Training adaptation (negentropy-modulated)
        """
        if not self.neural_enabled:
            return None

        timestamp = self.current_state.timestamp_ms if self.current_state else 0

        # Step Kuramoto system with spinner z
        kuramoto_metrics = self.neural_trainer.training_step_update(z)

        # Step grid cell plates if enabled
        grid_coherence = 0.0
        grid_hex = 0.0
        grid_score = 0.0
        plate_activations = []
        resonance = 0.0

        if self.plate_coupling:
            plate_state = self.plate_coupling.forward_coupling(z)
            training_signal = self.plate_coupling.backward_coupling(plate_state)

            grid_coherence = plate_state.global_coherence
            grid_hex = plate_state.hexagonal_order
            grid_score = plate_state.grid_score
            plate_activations = plate_state.plate_activations
            resonance = self.plate_coupling.get_resonance_measure()

        # Build neural state
        self.current_neural_state = NeuralState(
            timestamp_ms=timestamp,
            kuramoto_r=kuramoto_metrics['r'],
            kuramoto_psi=0.0,  # Mean phase from system
            kuramoto_hex=kuramoto_metrics['r_hex'],
            kuramoto_K=kuramoto_metrics['K'],
            kuramoto_synchronized=kuramoto_metrics['synchronized'],
            grid_coherence=grid_coherence,
            grid_hex_order=grid_hex,
            grid_score=grid_score,
            plate_activations=plate_activations,
            learning_rate=kuramoto_metrics['learning_rate'],
            resonance=resonance,
            total_energy=kuramoto_metrics['total_energy']
        )

        # Update coherence for unified state
        self._kuramoto_coherence = kuramoto_metrics['r']

        return self.current_neural_state

    async def broadcast_state(self):
        """Send current state to all connected clients."""
        if not WEBSOCKETS_AVAILABLE:
            return

        if self.current_state and self.clients:
            # Step neural systems with current z
            if self.neural_enabled:
                self._step_neural_systems(self.current_state.z)

            # Standard spinner state message
            message = json.dumps({
                "type": "spinner_state",
                **asdict(self.current_state)
            })

            disconnected = set()
            for client in self.clients:
                try:
                    await client.send(message)
                except Exception:
                    disconnected.add(client)

            self.clients -= disconnected

            # Neural state broadcast at lower rate
            if self.neural_enabled and self.current_neural_state:
                now = self.current_state.timestamp_ms
                interval_ms = 1000 / self.config.neural_broadcast_hz
                if now - self.last_neural_broadcast >= interval_ms:
                    await self._broadcast_neural_state()
                    self.last_neural_broadcast = now

            # Unified state broadcast at lower rate
            if self.config.broadcast_unified_state:
                now = self.current_state.timestamp_ms
                interval_ms = 1000 / self.config.unified_state_rate_hz
                if now - self.last_unified_broadcast >= interval_ms:
                    await self._broadcast_unified_state()
                    self.last_unified_broadcast = now

    async def _broadcast_neural_state(self):
        """Broadcast neural system state."""
        if not self.current_neural_state or not self.clients:
            return

        n = self.current_neural_state
        neural_msg = {
            "type": "neural_state",
            "timestamp_ms": n.timestamp_ms,
            "kuramoto": {
                "r": n.kuramoto_r,
                "psi": n.kuramoto_psi,
                "hex_order": n.kuramoto_hex,
                "K": n.kuramoto_K,
                "synchronized": n.kuramoto_synchronized,
                "n_oscillators": 60,
                "angular_spacing_deg": 6.0,
            },
            "grid_cells": {
                "coherence": n.grid_coherence,
                "hex_order": n.grid_hex_order,
                "grid_score": n.grid_score,
                "plate_activations": n.plate_activations,
                "n_plates": 6,
                "cells_per_plate": 10,
            },
            "training": {
                "learning_rate": n.learning_rate,
                "resonance": n.resonance,
                "total_energy": n.total_energy,
            },
            "physics": {
                "z_critical": Z_CRITICAL,
                "sin_60": math.sqrt(3) / 2,
                "connection": "sin(60°) = √3/2 = z_c",
            }
        }

        message = json.dumps(neural_msg)
        for client in self.clients:
            try:
                await client.send(message)
            except Exception:
                pass

    async def _broadcast_unified_state(self):
        """Broadcast full unified state for integration."""
        if not self.current_state or not self.clients:
            return

        s = self.current_state
        n = self.current_neural_state

        # Build unified state message compatible with UnifiedStateBridge
        unified = {
            "type": "unified_state",
            "timestamp_ms": s.timestamp_ms,

            # Core spinner state
            "z": s.z,
            "delta_s_neg": s.delta_s_neg,
            "tier": s.tier,
            "phase": s.phase,
            "kappa": s.kappa,
            "eta": s.eta,

            # TRIAD state
            "triad": {
                "rank": s.rank,
                "k_formation": s.k_formation,
                "k_formation_duration_ms": s.k_formation_duration_ms,
                "lambda_decay": 1.0 - s.kappa,  # Conservation: κ + λ = 1
            },

            # Kuramoto state (from real 60-oscillator system if available)
            "kuramoto": {
                "coherence": n.kuramoto_r if n else self._compute_kuramoto_coherence(s),
                "mean_phase": n.kuramoto_psi if n else 0.0,
                "coupling_strength": n.kuramoto_K if n else s.delta_s_neg * 2.0,
                "hex_order": n.kuramoto_hex if n else 0.0,
                "synchronized": n.kuramoto_synchronized if n else False,
                "n_oscillators": 60,
            },

            # Grid cell neural plate state
            "grid_cells": {
                "coherence": n.grid_coherence if n else 0.0,
                "hex_order": n.grid_hex_order if n else 0.0,
                "grid_score": n.grid_score if n else 0.0,
                "plate_activations": n.plate_activations if n else [],
                "resonance": n.resonance if n else 0.0,
            },

            # GHMP state (simulated when no hardware)
            "ghmp": {
                "pattern_count": self._ghmp_pattern_count,
                "active_operators": self._get_active_operators(s),
                "parity": "even" if s.delta_s_neg > 0.5 else "odd",
            },

            # Training metrics
            "training": {
                "learning_rate": n.learning_rate if n else 0.01,
                "total_energy": n.total_energy if n else 0.0,
            },

            # Physics connection
            "physics": {
                "z_critical": Z_CRITICAL,
                "sin_60_deg": math.sqrt(3) / 2,
                "hex_connection": "sin(60°) = √3/2 = z_c",
            },
        }

        message = json.dumps(unified)
        for client in self.clients:
            try:
                await client.send(message)
            except Exception:
                pass

    def _compute_kuramoto_coherence(self, state: SpinnerState) -> float:
        """Estimate Kuramoto coherence from spinner state."""
        # At z_c, coherence peaks due to coupling peak
        # r ≈ delta_s_neg when near critical
        base = state.delta_s_neg * 0.8
        # Add smooth variation based on tier
        tier_bonus = state.tier / 10.0 * 0.2
        self._kuramoto_coherence = min(1.0, base + tier_bonus)
        return self._kuramoto_coherence

    def _get_active_operators(self, state: SpinnerState) -> list:
        """Get list of APL operators available at current tier."""
        # Tier-gated operators (from APL operator hierarchy)
        ALL_OPERATORS = ["∂", "+", "×", "÷", "⍴", "↓"]  # CLOSURE, FUSION, AMPLIFY, DECOHERE, GROUP, SEPARATE
        TIER_THRESHOLDS = [0, 2, 3, 4, 5, 6]  # Min tier for each operator

        available = []
        for op, min_tier in zip(ALL_OPERATORS, TIER_THRESHOLDS):
            if state.tier >= min_tier:
                available.append(op)

        # At K-formation, all operators become available
        if state.k_formation:
            available = ALL_OPERATORS.copy()

        return available
    
    async def handle_client(self, websocket, path):
        """Handle WebSocket client connection."""
        self.clients.add(websocket)
        print(f"[BRIDGE] Client connected ({len(self.clients)} total)")
        
        try:
            async for message in websocket:
                data = json.loads(message)
                if 'cmd' in data:
                    await self.send_command(data)
        except Exception:
            pass
        finally:
            self.clients.discard(websocket)
            print(f"[BRIDGE] Client disconnected ({len(self.clients)} total)")
    
    async def send_command(self, cmd: dict):
        """Send command to firmware or simulator."""
        if self.simulator:
            await self.command_queue.put(cmd)
        elif self.serial:
            message = json.dumps(cmd) + '\n'
            self.serial.write(message.encode('utf-8'))
        print(f"[BRIDGE] Command: {cmd}")
    
    async def run(self):
        """Main bridge service loop."""
        self.running = True
        
        if not await self.connect_serial():
            print("[BRIDGE] Failed to initialize")
            return
        
        # Start WebSocket server if available
        server = None
        if WEBSOCKETS_AVAILABLE:
            server = await websockets.serve(
                self.handle_client,
                self.config.websocket_host,
                self.config.websocket_port
            )
            print(f"[BRIDGE] WebSocket server on ws://{self.config.websocket_host}:{self.config.websocket_port}")
            if self.config.broadcast_unified_state:
                print(f"[BRIDGE] Unified state enabled at {self.config.unified_state_rate_hz} Hz")
        
        # Start serial/simulator reader
        serial_task = asyncio.create_task(self.read_serial())
        
        # Status printer
        async def print_status():
            while self.running:
                if self.current_state:
                    s = self.current_state
                    k_str = "★ K-FORMATION" if s.k_formation else ""
                    print(f"\r[STATE] z={s.z:.4f} ΔS={s.delta_s_neg:.4f} "
                          f"κ={s.kappa:.3f} {s.tier_name:10s} {s.phase:10s} {k_str}",
                          end='', flush=True)
                await asyncio.sleep(0.5)
        
        status_task = asyncio.create_task(print_status())
        
        try:
            print("[BRIDGE] Running... Press Ctrl+C to stop")
            await asyncio.Future()
        except KeyboardInterrupt:
            print("\n[BRIDGE] Shutting down...")
        finally:
            self.running = False
            serial_task.cancel()
            status_task.cancel()
            if server:
                server.close()
            if self.serial:
                self.serial.close()

# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Nuclear Spinner Bridge Service with Neural Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Neural Interface:
  The bridge includes a 60-oscillator Kuramoto system and 6 neural plates
  for grid cell dynamics. The Kuramoto coupling strength K is driven by
  the spinner's z-coordinate:

    dθᵢ/dt = ωᵢ + (K/N) Σⱼ sin(θⱼ - θᵢ)

  Key connection: sin(60°) = √3/2 = z_c (critical point)

  60 oscillators → 6° angular spacing → hexagonal symmetry
        """
    )
    parser.add_argument('--port', '-p', default=DEFAULT_SERIAL_PORT,
                        help=f'Serial port (default: {DEFAULT_SERIAL_PORT})')
    parser.add_argument('--simulate', '-s', action='store_true',
                        help='Run in simulation mode (no hardware)')
    parser.add_argument('--neural', '-n', action='store_true', default=True,
                        help='Enable neural interface (default: enabled)')
    parser.add_argument('--no-neural', action='store_true',
                        help='Disable neural interface')
    parser.add_argument('--no-em', action='store_true',
                        help='Disable electromagnetic training')
    parser.add_argument('--no-grid', action='store_true',
                        help='Disable grid cell plates')
    args = parser.parse_args()

    # Build configuration
    config = BridgeConfig(
        enable_neural=args.neural and not args.no_neural,
        enable_em_training=not args.no_em,
        enable_grid_cells=not args.no_grid,
    )

    # Print banner
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║        Nuclear Spinner Bridge with Neural Interface         ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print(f"║  z_c = √3/2 = {Z_CRITICAL:.6f}  |  sin(60°) = {math.sqrt(3)/2:.6f}        ║")
    print("║  60 Kuramoto oscillators  |  6 neural plates                ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    bridge = SpinnerBridge(serial_port=args.port, simulate=args.simulate,
                           config=config)
    asyncio.run(bridge.run())


if __name__ == "__main__":
    main()
