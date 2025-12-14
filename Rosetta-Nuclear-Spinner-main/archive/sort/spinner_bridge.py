#!/usr/bin/env python3
"""
spinner_bridge.py
─────────────────
Bridges Nuclear Spinner firmware to Rosetta-Helix via WebSocket.

Usage:
    python spinner_bridge.py [--port /dev/ttyACM0] [--simulate]
"""

import asyncio
import json
import argparse
import math
import random
from collections import deque
from dataclasses import dataclass, asdict
from typing import Optional, Set, Callable

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
    def __init__(self, serial_port: str = None, simulate: bool = False):
        self.serial_port = serial_port
        self.simulate = simulate or not SERIAL_AVAILABLE
        self.serial: Optional['serial.Serial'] = None
        self.simulator: Optional[SpinnerSimulator] = None
        self.clients: Set = set()
        self.state_history: deque = deque(maxlen=STATE_HISTORY_SIZE)
        self.current_state: Optional[SpinnerState] = None
        self.running = False
        self.command_queue: asyncio.Queue = asyncio.Queue()
    
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
                    await asyncio.sleep(1.0 / SIMULATION_RATE_HZ)
                    
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
        """Process command in simulation mode."""
        if cmd.get('cmd') == 'set_z':
            self.simulator.set_target_z(cmd.get('value', 0.5))
            print(f"[SIM] Target z set to {cmd.get('value')}")
        elif cmd.get('cmd') == 'stop':
            self.simulator.set_target_z(0.0)
            print("[SIM] Emergency stop")
    
    async def broadcast_state(self):
        """Send current state to all connected clients."""
        if not WEBSOCKETS_AVAILABLE:
            return
            
        if self.current_state and self.clients:
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
                WEBSOCKET_HOST,
                WEBSOCKET_PORT
            )
            print(f"[BRIDGE] WebSocket server on ws://{WEBSOCKET_HOST}:{WEBSOCKET_PORT}")
        
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
    parser = argparse.ArgumentParser(description="Nuclear Spinner Bridge Service")
    parser.add_argument('--port', '-p', default=DEFAULT_SERIAL_PORT,
                        help=f'Serial port (default: {DEFAULT_SERIAL_PORT})')
    parser.add_argument('--simulate', '-s', action='store_true',
                        help='Run in simulation mode (no hardware)')
    args = parser.parse_args()
    
    bridge = SpinnerBridge(serial_port=args.port, simulate=args.simulate)
    asyncio.run(bridge.run())

if __name__ == "__main__":
    main()
