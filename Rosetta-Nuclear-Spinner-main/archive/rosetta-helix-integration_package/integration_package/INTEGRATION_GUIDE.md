# Nuclear Spinner × Rosetta-Helix Integration Guide

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         INTEGRATED SYSTEM                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    HARDWARE LAYER                                    │   │
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐            │   │
│  │  │ Rotor Motor   │  │ RF Coils      │  │ B₀ Magnet     │            │   │
│  │  │ (0-10k RPM)   │  │ (Tx/Rx)       │  │ (Static)      │            │   │
│  │  └───────┬───────┘  └───────┬───────┘  └───────────────┘            │   │
│  │          │                  │                                        │   │
│  │          ▼                  ▼                                        │   │
│  │  ┌─────────────────────────────────────────────────────────────┐    │   │
│  │  │              STM32H7 Microcontroller                         │    │   │
│  │  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐            │    │   │
│  │  │  │ Rotor   │ │ Pulse   │ │Threshold│ │ Neural  │            │    │   │
│  │  │  │ Control │ │ Control │ │ Logic   │ │Interface│            │    │   │
│  │  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘            │    │   │
│  │  │                         │                                    │    │   │
│  │  │                    USB/UART                                  │    │   │
│  │  └─────────────────────────┼────────────────────────────────────┘    │   │
│  └────────────────────────────┼─────────────────────────────────────────┘   │
│                               │                                             │
│                               │ Serial Protocol (115200 baud)               │
│                               │ JSON messages                               │
│                               ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    SOFTWARE LAYER                                    │   │
│  │                                                                      │   │
│  │  ┌───────────────────────────────────────────────────────────────┐  │   │
│  │  │                  Bridge Service (Python)                       │  │   │
│  │  │  - Serial communication with firmware                          │  │   │
│  │  │  - State normalization                                         │  │   │
│  │  │  - WebSocket server for Rosetta-Helix                          │  │   │
│  │  └───────────────────────────┬───────────────────────────────────┘  │   │
│  │                              │                                       │   │
│  │                              │ WebSocket (localhost:8765)            │   │
│  │                              ▼                                       │   │
│  │  ┌───────────────────────────────────────────────────────────────┐  │   │
│  │  │                  Rosetta-Helix (Python/JS)                     │  │   │
│  │  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐              │  │   │
│  │  │  │  Heart  │ │  Brain  │ │  TRIAD  │ │   APL   │              │  │   │
│  │  │  │(Kuramoto│ │ (GHMP)  │ │ Tracker │ │Operators│              │  │   │
│  │  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘              │  │   │
│  │  └───────────────────────────────────────────────────────────────┘  │   │
│  │                                                                      │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Hardware Setup

### 1.1 Bill of Materials

| Component | Specification | Purpose |
|-----------|---------------|---------|
| STM32H743ZI Nucleo | 480 MHz Cortex-M7 | Main controller |
| BLDC Motor | 10,000 RPM max, encoder feedback | Rotor drive |
| Motor Driver | 48V, 10A, FOC capable | Motor control |
| Permanent Magnet | NdFeB, 0.5T uniform field | B₀ field |
| RF Coil Assembly | 31 MHz tuned (³¹P Larmor) | NMR Tx/Rx |
| USB-C Cable | USB 2.0 FS | Host communication |
| Power Supply | 48V 5A (motor), 5V 2A (logic) | System power |

### 1.2 Connections

```
STM32H743ZI Pinout:
──────────────────────────────────────────────────────
Motor Control:
  PA0  → Motor PWM (TIM2_CH1)
  PA1  → Motor Direction
  PB6  → Encoder A (TIM4_CH1)
  PB7  → Encoder B (TIM4_CH2)
  
RF Control:
  PA5  → RF Tx Enable
  PA6  → RF Rx Enable
  PB0  → ADC (NMR signal)
  
Neural Interface:
  PA4  → DAC1 (stimulus output)
  PA5  → Sync pulse output
  
Communication:
  PA9  → USART1_TX (to host)
  PA10 → USART1_RX (from host)
  
Status:
  PB0  → Status LED (heartbeat)
  PB14 → K-formation LED
──────────────────────────────────────────────────────
```

### 1.3 Firmware Flash

```bash
# Prerequisites
sudo apt install stlink-tools gcc-arm-none-eabi

# Clone firmware
git clone <repository>/nuclear_spinner_firmware
cd nuclear_spinner_firmware

# Build
make clean && make

# Flash to STM32
st-flash write build/nuclear_spinner.bin 0x08000000

# Verify
st-flash read readback.bin 0x08000000 0x10000
diff build/nuclear_spinner.bin readback.bin
```

---

## 2. Communication Protocol

### 2.1 Serial Configuration

```
Baud:     115200
Data:     8 bits
Parity:   None
Stop:     1 bit
Flow:     None
Encoding: UTF-8 JSON
```

### 2.2 Message Format

**Firmware → Host (State Updates, 100 Hz)**

```json
{
  "type": "state",
  "timestamp_ms": 1234567890,
  "z": 0.866025,
  "rpm": 8660,
  "delta_s_neg": 0.999999,
  "tier": 6,
  "tier_name": "UNIVERSAL",
  "phase": "THE_LENS",
  "kappa": 0.9234,
  "eta": 0.6543,
  "rank": 9,
  "k_formation": true,
  "k_formation_duration_ms": 1234
}
```

**Host → Firmware (Commands)**

```json
// Set target z
{"cmd": "set_z", "value": 0.866}

// Set target RPM directly
{"cmd": "set_rpm", "value": 8660}

// Start hexagonal cycling protocol
{"cmd": "hex_cycle", "dwell_s": 30.0, "cycles": 10}

// Dwell at z_c
{"cmd": "dwell_lens", "duration_s": 300.0}

// Emergency stop
{"cmd": "stop"}

// Query status
{"cmd": "status"}
```

**Firmware → Host (Responses)**

```json
{"type": "ack", "cmd": "set_z", "status": "ok"}
{"type": "error", "cmd": "set_z", "error": "value out of range"}
{"type": "k_formation_event", "timestamp_ms": 123456, "duration_ms": 5000}
```

### 2.3 Firmware Communication Code

Already implemented in `src/comm_protocol.c`. Key functions:

```c
// Send state at 100 Hz (call from main loop)
void CommProtocol_SendState(const SystemState_t *state);

// Process incoming command (call when UART data available)
void CommProtocol_ProcessCommand(const char *json_str);

// Register callback for command handling
void CommProtocol_RegisterCallback(CommCallback_t callback);
```

---

## 3. Bridge Service

### 3.1 Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install pyserial websockets numpy

# Clone bridge service
git clone <repository>/spinner_bridge
cd spinner_bridge
```

### 3.2 Bridge Service Code

```python
#!/usr/bin/env python3
"""
spinner_bridge.py
─────────────────
Bridges Nuclear Spinner firmware to Rosetta-Helix via WebSocket.
"""

import asyncio
import json
import serial
import websockets
from collections import deque
from dataclasses import dataclass, asdict
from typing import Optional, Set

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

SERIAL_PORT = "/dev/ttyACM0"  # Adjust for your system
SERIAL_BAUD = 115200
WEBSOCKET_HOST = "localhost"
WEBSOCKET_PORT = 8765
STATE_HISTORY_SIZE = 1000

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

# ═══════════════════════════════════════════════════════════════════════════
# BRIDGE SERVICE
# ═══════════════════════════════════════════════════════════════════════════

class SpinnerBridge:
    def __init__(self):
        self.serial: Optional[serial.Serial] = None
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        self.state_history: deque = deque(maxlen=STATE_HISTORY_SIZE)
        self.current_state: Optional[SpinnerState] = None
        self.running = False
    
    async def connect_serial(self):
        """Connect to firmware via serial port."""
        try:
            self.serial = serial.Serial(
                SERIAL_PORT, 
                SERIAL_BAUD, 
                timeout=0.1
            )
            print(f"[BRIDGE] Connected to {SERIAL_PORT}")
            return True
        except serial.SerialException as e:
            print(f"[BRIDGE] Serial error: {e}")
            return False
    
    async def read_serial(self):
        """Read and parse state updates from firmware."""
        while self.running:
            if self.serial and self.serial.in_waiting:
                try:
                    line = self.serial.readline().decode('utf-8').strip()
                    if line:
                        data = json.loads(line)
                        if data.get('type') == 'state':
                            self.current_state = SpinnerState(**{
                                k: v for k, v in data.items() 
                                if k != 'type'
                            })
                            self.state_history.append(self.current_state)
                            
                            # Broadcast to all WebSocket clients
                            await self.broadcast_state()
                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    print(f"[BRIDGE] Parse error: {e}")
            
            await asyncio.sleep(0.001)  # 1ms polling
    
    async def broadcast_state(self):
        """Send current state to all connected clients."""
        if self.current_state and self.clients:
            message = json.dumps({
                "type": "spinner_state",
                **asdict(self.current_state)
            })
            
            # Send to all clients
            disconnected = set()
            for client in self.clients:
                try:
                    await client.send(message)
                except websockets.ConnectionClosed:
                    disconnected.add(client)
            
            self.clients -= disconnected
    
    async def handle_client(self, websocket, path):
        """Handle WebSocket client connection."""
        self.clients.add(websocket)
        print(f"[BRIDGE] Client connected ({len(self.clients)} total)")
        
        try:
            async for message in websocket:
                # Forward commands to firmware
                data = json.loads(message)
                if 'cmd' in data:
                    await self.send_command(data)
        except websockets.ConnectionClosed:
            pass
        finally:
            self.clients.discard(websocket)
            print(f"[BRIDGE] Client disconnected ({len(self.clients)} total)")
    
    async def send_command(self, cmd: dict):
        """Send command to firmware via serial."""
        if self.serial:
            message = json.dumps(cmd) + '\n'
            self.serial.write(message.encode('utf-8'))
            print(f"[BRIDGE] Sent: {cmd}")
    
    async def run(self):
        """Main bridge service loop."""
        self.running = True
        
        # Connect to serial
        if not await self.connect_serial():
            print("[BRIDGE] Running in simulation mode (no hardware)")
        
        # Start WebSocket server
        server = await websockets.serve(
            self.handle_client,
            WEBSOCKET_HOST,
            WEBSOCKET_PORT
        )
        print(f"[BRIDGE] WebSocket server on ws://{WEBSOCKET_HOST}:{WEBSOCKET_PORT}")
        
        # Start serial reader
        serial_task = asyncio.create_task(self.read_serial())
        
        try:
            await asyncio.Future()  # Run forever
        except KeyboardInterrupt:
            print("[BRIDGE] Shutting down...")
        finally:
            self.running = False
            serial_task.cancel()
            server.close()
            if self.serial:
                self.serial.close()

# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    bridge = SpinnerBridge()
    asyncio.run(bridge.run())
```

### 3.3 Running the Bridge

```bash
# Start bridge service
python spinner_bridge.py

# Expected output:
# [BRIDGE] Connected to /dev/ttyACM0
# [BRIDGE] WebSocket server on ws://localhost:8765
```

---

## 4. Rosetta-Helix Integration

### 4.1 WebSocket Client

Add to Rosetta-Helix's main module:

```python
# rosetta_helix/spinner_client.py

import asyncio
import json
import websockets
from typing import Callable, Optional

class SpinnerClient:
    """
    WebSocket client for receiving Nuclear Spinner state.
    Feeds z-coordinate to Kuramoto Heart.
    """
    
    def __init__(
        self, 
        uri: str = "ws://localhost:8765",
        on_state: Optional[Callable] = None
    ):
        self.uri = uri
        self.on_state = on_state
        self.websocket = None
        self.connected = False
        self.latest_state = None
    
    async def connect(self):
        """Connect to bridge service."""
        try:
            self.websocket = await websockets.connect(self.uri)
            self.connected = True
            print(f"[ROSETTA] Connected to spinner at {self.uri}")
        except Exception as e:
            print(f"[ROSETTA] Connection failed: {e}")
            self.connected = False
    
    async def listen(self):
        """Listen for state updates."""
        while self.connected:
            try:
                message = await self.websocket.recv()
                data = json.loads(message)
                
                if data.get('type') == 'spinner_state':
                    self.latest_state = data
                    
                    if self.on_state:
                        await self.on_state(data)
                        
            except websockets.ConnectionClosed:
                print("[ROSETTA] Connection lost")
                self.connected = False
                break
    
    async def send_command(self, cmd: str, **kwargs):
        """Send command to spinner."""
        if self.connected:
            message = json.dumps({"cmd": cmd, **kwargs})
            await self.websocket.send(message)
    
    def get_z(self) -> float:
        """Get current z-coordinate."""
        if self.latest_state:
            return self.latest_state.get('z', 0.5)
        return 0.5
    
    def is_k_formation(self) -> bool:
        """Check if K-formation is active."""
        if self.latest_state:
            return self.latest_state.get('k_formation', False)
        return False
```

### 4.2 Heart Integration

Modify Rosetta-Helix Heart to accept spinner z:

```python
# rosetta_helix/heart.py (modification)

class Heart:
    """
    60 Kuramoto oscillators with spinner-driven coupling.
    """
    
    def __init__(self, n_oscillators: int = 60):
        self.n = n_oscillators
        self.phases = np.linspace(0, 2*np.pi, n_oscillators, endpoint=False)
        self.natural_freqs = 1.0 + 0.1 * (np.random.random(n_oscillators) - 0.5)
        
        # Spinner integration
        self.spinner_z = 0.5
        self.coupling_scale = 8.0  # Tuned for K-formation at z_c
    
    def set_spinner_z(self, z: float):
        """Update coupling from spinner z-coordinate."""
        self.spinner_z = z
    
    def compute_coupling(self) -> float:
        """
        Compute Kuramoto coupling K from spinner state.
        
        K = scale * z * ΔS_neg(z)
        
        This peaks at z = z_c = √3/2, creating resonance
        with the 60-oscillator hexagonal geometry.
        """
        Z_CRITICAL = 0.8660254037844387
        SIGMA = 36.0
        
        delta_s_neg = np.exp(-SIGMA * (self.spinner_z - Z_CRITICAL) ** 2)
        return self.coupling_scale * self.spinner_z * delta_s_neg
    
    def step(self, dt: float = 0.01) -> float:
        """Advance oscillators, return coherence."""
        K = self.compute_coupling()
        
        # Kuramoto dynamics
        z_complex = np.mean(np.exp(1j * self.phases))
        r = np.abs(z_complex)
        psi = np.angle(z_complex)
        
        d_phases = self.natural_freqs + K * r * np.sin(psi - self.phases)
        self.phases += d_phases * dt
        self.phases %= 2 * np.pi
        
        return r
```

### 4.3 Main Loop Integration

```python
# rosetta_helix/main.py

import asyncio
from spinner_client import SpinnerClient
from heart import Heart
from brain import Brain
from triad import TriadTracker

class RosettaHelixNode:
    """
    Integrated Rosetta-Helix node with spinner coupling.
    """
    
    def __init__(self):
        self.heart = Heart()
        self.brain = Brain()
        self.triad = TriadTracker()
        self.spinner = SpinnerClient(on_state=self.on_spinner_state)
        
        self.z = 0.5
        self.coherence = 0.0
        self.k_formation_active = False
    
    async def on_spinner_state(self, state: dict):
        """Callback when spinner state updates."""
        # Update heart coupling from spinner z
        spinner_z = state.get('z', 0.5)
        self.heart.set_spinner_z(spinner_z)
        
        # Check for spinner-side K-formation
        if state.get('k_formation', False):
            print(f"[NODE] Spinner K-formation active at z={spinner_z:.4f}")
    
    async def step(self):
        """Single simulation step."""
        # Advance heart
        self.coherence = self.heart.step()
        
        # Update internal z from coherence
        # (Could also use spinner z directly)
        self.z = 0.5 + 0.5 * self.coherence
        
        # Check K-formation (our side)
        if self.coherence >= 0.92:
            if not self.k_formation_active:
                print(f"[NODE] K-formation achieved! r={self.coherence:.4f}")
                self.k_formation_active = True
        else:
            self.k_formation_active = False
        
        # Update brain tier access
        tier = self.compute_tier()
        self.brain.set_accessible_tiers(tier)
        
        # TRIAD tracking
        self.triad.update(self.coherence)
    
    def compute_tier(self) -> int:
        """Map z to tier (0-9)."""
        TIER_BOUNDS = [0.0, 0.10, 0.20, 0.40, 0.60, 0.75, 0.866, 0.92, 0.97, 1.0]
        for i, bound in enumerate(TIER_BOUNDS):
            if self.z < bound:
                return max(0, i - 1)
        return 9
    
    async def run(self, steps: int = None):
        """Main loop."""
        # Connect to spinner
        await self.spinner.connect()
        
        # Start listening task
        listen_task = asyncio.create_task(self.spinner.listen())
        
        step_count = 0
        try:
            while steps is None or step_count < steps:
                await self.step()
                step_count += 1
                await asyncio.sleep(0.01)  # 100 Hz
        finally:
            listen_task.cancel()

# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

async def main():
    node = RosettaHelixNode()
    await node.run()

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 5. Deployment Sequence

### 5.1 Full Stack Startup

```bash
#!/bin/bash
# start_system.sh

echo "═══════════════════════════════════════════════════════════════"
echo "  Nuclear Spinner × Rosetta-Helix System Startup"
echo "═══════════════════════════════════════════════════════════════"

# 1. Check hardware connection
echo "[1/4] Checking hardware..."
if [ ! -e /dev/ttyACM0 ]; then
    echo "  WARNING: No spinner hardware detected"
    echo "  System will run in simulation mode"
else
    echo "  ✓ Spinner connected on /dev/ttyACM0"
fi

# 2. Start bridge service
echo "[2/4] Starting bridge service..."
cd spinner_bridge
python spinner_bridge.py &
BRIDGE_PID=$!
sleep 2

# 3. Start Rosetta-Helix
echo "[3/4] Starting Rosetta-Helix..."
cd ../rosetta_helix
python main.py &
ROSETTA_PID=$!
sleep 1

# 4. Start web visualizer (optional)
echo "[4/4] Starting visualizer..."
cd ../docs
python -m http.server 8000 &
VIZ_PID=$!

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  System Running"
echo "═══════════════════════════════════════════════════════════════"
echo "  Bridge:     PID $BRIDGE_PID (ws://localhost:8765)"
echo "  Rosetta:    PID $ROSETTA_PID"
echo "  Visualizer: PID $VIZ_PID (http://localhost:8000)"
echo ""
echo "  Press Ctrl+C to stop all services"
echo "═══════════════════════════════════════════════════════════════"

# Wait for interrupt
trap "kill $BRIDGE_PID $ROSETTA_PID $VIZ_PID 2>/dev/null" EXIT
wait
```

### 5.2 Verification Steps

```bash
# 1. Verify firmware communication
screen /dev/ttyACM0 115200
# Should see JSON state messages at 100 Hz

# 2. Verify bridge WebSocket
wscat -c ws://localhost:8765
# Should receive spinner_state messages

# 3. Verify Rosetta-Helix coupling
# Watch logs for:
#   [NODE] Spinner K-formation active at z=0.8660
#   [NODE] K-formation achieved! r=0.9234

# 4. Run integration test
python -c "
import asyncio
from spinner_client import SpinnerClient

async def test():
    client = SpinnerClient()
    await client.connect()
    await client.send_command('set_z', value=0.866)
    await asyncio.sleep(5)
    print(f'z={client.get_z()}, k={client.is_k_formation()}')

asyncio.run(test())
"
```

---

## 6. Operational Protocols

### 6.1 Standard Run: Approach z_c

```python
# Ramp to z_c over 30 seconds, dwell for 5 minutes
await spinner.send_command('set_z', value=0.866)
await asyncio.sleep(300)  # 5 min dwell
```

### 6.2 Hexagonal Cycling Protocol

```python
# 6-phase cycle, 30s per phase, 10 cycles
await spinner.send_command('hex_cycle', dwell_s=30.0, cycles=10)
# Total: 30 minutes
```

### 6.3 Z-Sweep Experiment

```python
# Sweep 0.5 → 1.0, measure K-formation rate at each z
for z in np.linspace(0.5, 1.0, 21):
    await spinner.send_command('set_z', value=z)
    await asyncio.sleep(180)  # 3 min per z
    # Log k_formation_rate
```

### 6.4 Emergency Stop

```python
# Immediate halt
await spinner.send_command('stop')
```

---

## 7. Data Logging

### 7.1 Log Format

```json
{
  "session_id": "2024-01-15_14-30-00",
  "samples": [
    {
      "t_ms": 0,
      "spinner": {"z": 0.500, "delta_s_neg": 0.008, "k_formation": false},
      "rosetta": {"coherence": 0.12, "tier": 2, "k_formation": false}
    },
    {
      "t_ms": 10,
      "spinner": {"z": 0.502, "delta_s_neg": 0.009, "k_formation": false},
      "rosetta": {"coherence": 0.13, "tier": 2, "k_formation": false}
    }
  ],
  "events": [
    {"t_ms": 28500, "type": "k_formation_start", "source": "spinner", "z": 0.866},
    {"t_ms": 28600, "type": "k_formation_start", "source": "rosetta", "coherence": 0.923}
  ],
  "summary": {
    "duration_s": 300,
    "k_formations": 47,
    "peak_coherence": 0.998,
    "time_at_lens_s": 180
  }
}
```

### 7.2 Analysis Script

```python
# analyze_session.py
import json
import numpy as np
import matplotlib.pyplot as plt

def analyze(log_file: str):
    with open(log_file) as f:
        data = json.load(f)
    
    samples = data['samples']
    t = np.array([s['t_ms'] for s in samples]) / 1000
    z = np.array([s['spinner']['z'] for s in samples])
    r = np.array([s['rosetta']['coherence'] for s in samples])
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    
    axes[0].plot(t, z, label='Spinner z')
    axes[0].axhline(0.866, color='r', linestyle='--', label='z_c')
    axes[0].set_ylabel('z')
    axes[0].legend()
    
    axes[1].plot(t, r, label='Rosetta coherence')
    axes[1].axhline(0.92, color='r', linestyle='--', label='K threshold')
    axes[1].set_ylabel('r')
    axes[1].set_xlabel('Time (s)')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(log_file.replace('.json', '.png'))
    
    # Summary
    print(f"Session: {data['session_id']}")
    print(f"K-formations: {data['summary']['k_formations']}")
    print(f"Peak coherence: {data['summary']['peak_coherence']:.4f}")
    print(f"Time at LENS: {data['summary']['time_at_lens_s']:.1f}s")
```

---

## 8. Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| No serial connection | Wrong port | `ls /dev/tty*` to find device |
| Bridge connects but no data | Firmware not running | Re-flash firmware |
| Low coherence at z_c | Coupling too weak | Increase `coupling_scale` |
| K-formation unstable | Noise in z | Add low-pass filter |
| WebSocket disconnects | Bridge crash | Check logs, restart bridge |

---

## 9. File Manifest

```
nuclear_spinner_firmware/
├── src/
│   ├── main.c
│   ├── rotor_control.c
│   ├── threshold_logic.c
│   ├── neural_interface.c
│   └── comm_protocol.c
├── include/
│   ├── physics_constants.h
│   └── *.h
└── Makefile

spinner_bridge/
├── spinner_bridge.py
└── requirements.txt

rosetta_helix/
├── main.py
├── spinner_client.py
├── heart.py
├── brain.py
└── triad.py

scripts/
├── start_system.sh
├── analyze_session.py
└── run_experiment.py
```

---

## 10. Summary

The integrated system operates as follows:

1. **Hardware** (Nuclear Spinner) generates physical z-coordinate from rotor RPM
2. **Firmware** (STM32H7) computes ΔS_neg, tier, phase, K-formation status
3. **Bridge** (Python) relays state via WebSocket at 100 Hz
4. **Rosetta-Helix** (Python) uses spinner z to drive Kuramoto coupling
5. **Heart** (60 oscillators) synchronizes, producing coherence r
6. **K-formation** emerges when both systems reach κ ≥ 0.92 at z_c

The key insight: **spinner z drives Kuramoto K**. When z = z_c = √3/2:
- ΔS_neg peaks (spinner)
- Coupling K peaks (rosetta)  
- Coherence r peaks (heart)
- K-formation triggers (both systems)

This is resonance engineering across physical and computational substrates.

---

*Δ|integration-guide|z_c=0.866|Ω*
