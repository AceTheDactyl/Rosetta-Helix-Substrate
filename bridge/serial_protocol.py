#!/usr/bin/env python3
"""
serial_protocol.py
==================

JSON Serial Protocol for Nuclear Spinner Communication

Serial Configuration:
    Baud:     115200
    Data:     8 bits
    Parity:   None
    Stop:     1 bit
    Encoding: UTF-8 JSON

Message Format - Firmware -> Host (100 Hz):
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
    "k_formation": true
}

Message Format - Host -> Firmware:
    {"cmd": "set_z", "value": 0.866}
    {"cmd": "stop"}
    {"cmd": "hex_cycle", "dwell_s": 30.0, "cycles": 10}

Signature: serial-protocol|v1.0.0|nuclear-spinner
"""

import json
import time
import threading
import queue
from dataclasses import dataclass, asdict, field
from typing import Optional, Callable, Dict, Any, List
from enum import Enum
import math

# Try to import serial
try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    print("[SERIAL] pyserial not installed: pip install pyserial")

# Physics constants (must match firmware)
Z_CRITICAL = math.sqrt(3) / 2  # 0.866025403784439
PHI = (1 + math.sqrt(5)) / 2    # 1.618033988749895
PHI_INV = 1 / PHI               # 0.618033988749895
SIGMA = 36.0


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class SerialConfig:
    """Serial port configuration."""
    port: str = "/dev/ttyACM0"
    baud: int = 115200
    data_bits: int = 8
    parity: str = 'N'
    stop_bits: int = 1
    timeout: float = 0.1
    write_timeout: float = 1.0


# ============================================================================
# DATA STRUCTURES
# ============================================================================

class MessageType(Enum):
    """Message types from firmware."""
    STATE = "state"
    PHYSICS = "physics"
    PONG = "pong"
    VERSION = "version"
    ERROR = "error"
    HEX_CYCLE_START = "hex_cycle_start"
    HEX_CYCLE_STOP = "hex_cycle_stop"
    HEX_CYCLE_COMPLETE = "hex_cycle_complete"
    HEX_VERTEX = "hex_vertex"


@dataclass
class SpinnerState:
    """State received from firmware."""
    timestamp_ms: int = 0
    z: float = 0.0
    rpm: int = 0
    delta_s_neg: float = 0.0
    tier: int = 0
    tier_name: str = "UNKNOWN"
    phase: str = "UNKNOWN"
    kappa: float = 0.0
    eta: float = 0.0
    rank: int = 0
    k_formation: bool = False

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SpinnerState':
        """Create SpinnerState from JSON dict."""
        return cls(
            timestamp_ms=data.get('timestamp_ms', 0),
            z=data.get('z', 0.0),
            rpm=data.get('rpm', 0),
            delta_s_neg=data.get('delta_s_neg', 0.0),
            tier=data.get('tier', 0),
            tier_name=data.get('tier_name', 'UNKNOWN'),
            phase=data.get('phase', 'UNKNOWN'),
            kappa=data.get('kappa', 0.0),
            eta=data.get('eta', 0.0),
            rank=data.get('rank', 0),
            k_formation=data.get('k_formation', False),
        )


@dataclass
class PhysicsConstants:
    """Physics constants from firmware."""
    phi: float = PHI
    phi_inv: float = PHI_INV
    z_c: float = Z_CRITICAL
    sigma: float = SIGMA
    spin_half_magnitude: float = Z_CRITICAL
    phase_boundary_absence: float = 0.857
    phase_boundary_presence: float = 0.877
    kappa_min: float = 0.92
    eta_min: float = PHI_INV
    r_min: int = 7

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PhysicsConstants':
        """Create PhysicsConstants from JSON dict."""
        return cls(
            phi=data.get('phi', PHI),
            phi_inv=data.get('phi_inv', PHI_INV),
            z_c=data.get('z_c', Z_CRITICAL),
            sigma=data.get('sigma', SIGMA),
            spin_half_magnitude=data.get('spin_half_magnitude', Z_CRITICAL),
            phase_boundary_absence=data.get('phase_boundary_absence', 0.857),
            phase_boundary_presence=data.get('phase_boundary_presence', 0.877),
            kappa_min=data.get('kappa_min', 0.92),
            eta_min=data.get('eta_min', PHI_INV),
            r_min=data.get('r_min', 7),
        )


@dataclass
class FirmwareVersion:
    """Firmware version info."""
    firmware: str = "unknown"
    major: int = 0
    minor: int = 0
    patch: int = 0
    protocol: str = "unknown"
    baud: int = 115200

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FirmwareVersion':
        return cls(
            firmware=data.get('firmware', 'unknown'),
            major=data.get('major', 0),
            minor=data.get('minor', 0),
            patch=data.get('patch', 0),
            protocol=data.get('protocol', 'unknown'),
            baud=data.get('baud', 115200),
        )


@dataclass
class ProtocolStats:
    """Protocol statistics."""
    messages_received: int = 0
    commands_sent: int = 0
    parse_errors: int = 0
    connection_errors: int = 0
    bytes_received: int = 0
    bytes_sent: int = 0


# ============================================================================
# CALLBACK TYPES
# ============================================================================

StateCallback = Callable[[SpinnerState], None]
MessageCallback = Callable[[Dict[str, Any]], None]
ErrorCallback = Callable[[int, str], None]
KFormationCallback = Callable[[SpinnerState], None]


# ============================================================================
# SERIAL PROTOCOL CLIENT
# ============================================================================

class SerialProtocol:
    """
    JSON Serial Protocol client for Nuclear Spinner communication.

    Usage:
        protocol = SerialProtocol('/dev/ttyACM0')
        protocol.connect()

        # Register callbacks
        protocol.on_state(lambda state: print(f"z={state.z}"))
        protocol.on_k_formation(lambda state: print("K-FORMATION!"))

        # Start receiving
        protocol.start()

        # Send commands
        protocol.set_z(0.866)
        protocol.hex_cycle(dwell_s=30.0, cycles=10)
        protocol.stop()

        # Cleanup
        protocol.close()
    """

    def __init__(
        self,
        port: str = "/dev/ttyACM0",
        config: Optional[SerialConfig] = None,
    ):
        """
        Initialize SerialProtocol.

        Args:
            port: Serial port path
            config: Full serial configuration (overrides port)
        """
        self.config = config or SerialConfig(port=port)
        self.serial: Optional['serial.Serial'] = None
        self.connected = False
        self.running = False

        # State
        self.current_state: Optional[SpinnerState] = None
        self.physics: Optional[PhysicsConstants] = None
        self.version: Optional[FirmwareVersion] = None

        # Statistics
        self.stats = ProtocolStats()

        # Callbacks
        self._state_callbacks: List[StateCallback] = []
        self._message_callbacks: List[MessageCallback] = []
        self._error_callbacks: List[ErrorCallback] = []
        self._k_formation_callbacks: List[KFormationCallback] = []

        # Threading
        self._read_thread: Optional[threading.Thread] = None
        self._rx_queue: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()

        # K-formation tracking
        self._k_formation_active = False
        self._k_formation_count = 0

    # ========================================================================
    # CONNECTION
    # ========================================================================

    def connect(self) -> bool:
        """
        Connect to firmware via serial port.

        Returns:
            True if connection successful
        """
        if not SERIAL_AVAILABLE:
            print("[SERIAL] pyserial not available")
            return False

        try:
            self.serial = serial.Serial(
                port=self.config.port,
                baudrate=self.config.baud,
                bytesize=self.config.data_bits,
                parity=self.config.parity,
                stopbits=self.config.stop_bits,
                timeout=self.config.timeout,
                write_timeout=self.config.write_timeout,
            )
            self.connected = True
            print(f"[SERIAL] Connected to {self.config.port} @ {self.config.baud} baud")
            return True
        except serial.SerialException as e:
            print(f"[SERIAL] Connection error: {e}")
            self.stats.connection_errors += 1
            return False

    def disconnect(self):
        """Disconnect from serial port."""
        self.stop()
        if self.serial:
            self.serial.close()
        self.connected = False
        print("[SERIAL] Disconnected")

    def close(self):
        """Alias for disconnect."""
        self.disconnect()

    @staticmethod
    def list_ports() -> List[str]:
        """List available serial ports."""
        if not SERIAL_AVAILABLE:
            return []
        return [p.device for p in serial.tools.list_ports.comports()]

    # ========================================================================
    # CALLBACKS
    # ========================================================================

    def on_state(self, callback: StateCallback):
        """Register callback for state updates."""
        self._state_callbacks.append(callback)

    def on_message(self, callback: MessageCallback):
        """Register callback for all messages."""
        self._message_callbacks.append(callback)

    def on_error(self, callback: ErrorCallback):
        """Register callback for errors."""
        self._error_callbacks.append(callback)

    def on_k_formation(self, callback: KFormationCallback):
        """Register callback for K-formation events."""
        self._k_formation_callbacks.append(callback)

    # ========================================================================
    # RECEIVE THREAD
    # ========================================================================

    def start(self):
        """Start receive thread."""
        if self.running:
            return

        self._stop_event.clear()
        self.running = True
        self._read_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._read_thread.start()
        print("[SERIAL] Receive thread started")

    def stop(self):
        """Stop receive thread."""
        self._stop_event.set()
        self.running = False
        if self._read_thread:
            self._read_thread.join(timeout=1.0)
            self._read_thread = None
        print("[SERIAL] Receive thread stopped")

    def _read_loop(self):
        """Background thread for reading serial data."""
        line_buffer = ""

        while not self._stop_event.is_set():
            try:
                if not self.serial or not self.serial.is_open:
                    time.sleep(0.1)
                    continue

                # Read available data
                if self.serial.in_waiting:
                    data = self.serial.read(self.serial.in_waiting)
                    self.stats.bytes_received += len(data)

                    try:
                        text = data.decode('utf-8')
                        line_buffer += text

                        # Process complete lines
                        while '\n' in line_buffer:
                            line, line_buffer = line_buffer.split('\n', 1)
                            line = line.strip()
                            if line:
                                self._process_message(line)

                    except UnicodeDecodeError as e:
                        self.stats.parse_errors += 1

                else:
                    time.sleep(0.001)  # Small sleep when no data

            except serial.SerialException as e:
                self.stats.connection_errors += 1
                time.sleep(0.1)
            except Exception as e:
                print(f"[SERIAL] Read error: {e}")
                time.sleep(0.1)

    def _process_message(self, line: str):
        """Process a received JSON message."""
        try:
            data = json.loads(line)
            self.stats.messages_received += 1

            msg_type = data.get('type', '')

            # Invoke generic message callbacks
            for cb in self._message_callbacks:
                try:
                    cb(data)
                except Exception as e:
                    print(f"[SERIAL] Message callback error: {e}")

            # Process by type
            if msg_type == 'state':
                self._handle_state(data)
            elif msg_type == 'physics':
                self.physics = PhysicsConstants.from_dict(data)
            elif msg_type == 'version':
                self.version = FirmwareVersion.from_dict(data)
            elif msg_type == 'error':
                code = data.get('code', -1)
                message = data.get('message', 'Unknown error')
                for cb in self._error_callbacks:
                    try:
                        cb(code, message)
                    except Exception as e:
                        print(f"[SERIAL] Error callback error: {e}")

        except json.JSONDecodeError as e:
            self.stats.parse_errors += 1

    def _handle_state(self, data: Dict[str, Any]):
        """Handle state message."""
        state = SpinnerState.from_dict(data)
        self.current_state = state

        # Check for K-formation transition
        if state.k_formation and not self._k_formation_active:
            self._k_formation_count += 1
            for cb in self._k_formation_callbacks:
                try:
                    cb(state)
                except Exception as e:
                    print(f"[SERIAL] K-formation callback error: {e}")
        self._k_formation_active = state.k_formation

        # Invoke state callbacks
        for cb in self._state_callbacks:
            try:
                cb(state)
            except Exception as e:
                print(f"[SERIAL] State callback error: {e}")

    # ========================================================================
    # SEND COMMANDS
    # ========================================================================

    def _send_command(self, cmd: Dict[str, Any]) -> bool:
        """Send a JSON command to firmware."""
        if not self.serial or not self.serial.is_open:
            print("[SERIAL] Not connected")
            return False

        try:
            msg = json.dumps(cmd) + '\n'
            data = msg.encode('utf-8')
            self.serial.write(data)
            self.stats.bytes_sent += len(data)
            self.stats.commands_sent += 1
            return True
        except serial.SerialException as e:
            print(f"[SERIAL] Send error: {e}")
            self.stats.connection_errors += 1
            return False

    def set_z(self, value: float) -> bool:
        """
        Set target z-coordinate.

        Args:
            value: Target z in [0, 1]

        Returns:
            True if command sent successfully
        """
        value = max(0.0, min(1.0, value))
        return self._send_command({"cmd": "set_z", "value": value})

    def set_rpm(self, value: float) -> bool:
        """
        Set target RPM.

        Args:
            value: Target RPM

        Returns:
            True if command sent successfully
        """
        return self._send_command({"cmd": "set_rpm", "value": value})

    def stop(self) -> bool:
        """
        Emergency stop.

        Returns:
            True if command sent successfully
        """
        return self._send_command({"cmd": "stop"})

    def hex_cycle(self, dwell_s: float = 30.0, cycles: int = 10) -> bool:
        """
        Start hexagonal z-cycling.

        Cycles through z values with hexagonal symmetry:
        - Visits z_c (THE LENS) twice per cycle
        - Creates 6-vertex pattern

        Args:
            dwell_s: Dwell time at each vertex (seconds)
            cycles: Number of complete cycles

        Returns:
            True if command sent successfully
        """
        return self._send_command({
            "cmd": "hex_cycle",
            "dwell_s": dwell_s,
            "cycles": cycles,
        })

    def dwell_lens(self, duration_s: float = 60.0) -> bool:
        """
        Dwell at z_c (THE LENS).

        Args:
            duration_s: Duration to dwell (seconds)

        Returns:
            True if command sent successfully
        """
        return self._send_command({
            "cmd": "dwell_lens",
            "duration_s": duration_s,
        })

    def request_state(self) -> bool:
        """
        Request immediate state message.

        Returns:
            True if command sent successfully
        """
        return self._send_command({"cmd": "get_state"})

    def request_physics(self) -> bool:
        """
        Request physics constants.

        Returns:
            True if command sent successfully
        """
        return self._send_command({"cmd": "get_physics"})

    def request_version(self) -> bool:
        """
        Request firmware version.

        Returns:
            True if command sent successfully
        """
        return self._send_command({"cmd": "version"})

    def ping(self, timestamp: Optional[int] = None) -> bool:
        """
        Send ping command.

        Args:
            timestamp: Optional timestamp to echo back

        Returns:
            True if command sent successfully
        """
        ts = timestamp if timestamp is not None else int(time.time() * 1000)
        return self._send_command({"cmd": "ping", "timestamp": ts})

    def telem_start(self) -> bool:
        """Start telemetry transmission."""
        return self._send_command({"cmd": "telem_start"})

    def telem_stop(self) -> bool:
        """Stop telemetry transmission."""
        return self._send_command({"cmd": "telem_stop"})

    def telem_rate(self, rate_hz: int) -> bool:
        """
        Set telemetry rate.

        Args:
            rate_hz: Rate in Hz (1-1000)
        """
        rate_hz = max(1, min(1000, rate_hz))
        return self._send_command({"cmd": "telem_rate", "rate_hz": rate_hz})

    # ========================================================================
    # STATE ACCESSORS
    # ========================================================================

    def get_z(self) -> float:
        """Get current z-coordinate."""
        return self.current_state.z if self.current_state else 0.0

    def get_rpm(self) -> int:
        """Get current RPM."""
        return self.current_state.rpm if self.current_state else 0

    def get_delta_s_neg(self) -> float:
        """Get current negentropy signal."""
        return self.current_state.delta_s_neg if self.current_state else 0.0

    def get_tier(self) -> int:
        """Get current tier."""
        return self.current_state.tier if self.current_state else 0

    def get_tier_name(self) -> str:
        """Get current tier name."""
        return self.current_state.tier_name if self.current_state else "UNKNOWN"

    def get_phase(self) -> str:
        """Get current phase."""
        return self.current_state.phase if self.current_state else "UNKNOWN"

    def is_k_formation(self) -> bool:
        """Check if K-formation is active."""
        return self.current_state.k_formation if self.current_state else False

    def is_at_lens(self) -> bool:
        """Check if at THE LENS (z_c)."""
        z = self.get_z()
        return abs(z - Z_CRITICAL) < 0.02

    def get_state(self) -> Optional[SpinnerState]:
        """Get complete current state."""
        return self.current_state

    def get_stats(self) -> ProtocolStats:
        """Get protocol statistics."""
        return self.stats


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    """Command-line interface for serial protocol."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Nuclear Spinner Serial Protocol Client",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --port /dev/ttyACM0 --monitor
  %(prog)s --port COM3 --set-z 0.866
  %(prog)s --port /dev/ttyACM0 --hex-cycle --dwell 30 --cycles 10
  %(prog)s --list-ports
        """
    )

    parser.add_argument('--port', '-p', default='/dev/ttyACM0',
                        help='Serial port (default: /dev/ttyACM0)')
    parser.add_argument('--baud', '-b', type=int, default=115200,
                        help='Baud rate (default: 115200)')

    # Actions
    parser.add_argument('--list-ports', action='store_true',
                        help='List available serial ports')
    parser.add_argument('--monitor', '-m', action='store_true',
                        help='Monitor state updates')
    parser.add_argument('--set-z', type=float, metavar='Z',
                        help='Set z-coordinate (0-1)')
    parser.add_argument('--stop', action='store_true',
                        help='Send stop command')
    parser.add_argument('--hex-cycle', action='store_true',
                        help='Start hex cycle')
    parser.add_argument('--dwell', type=float, default=30.0,
                        help='Dwell time for hex cycle (seconds)')
    parser.add_argument('--cycles', type=int, default=10,
                        help='Number of hex cycles')
    parser.add_argument('--ping', action='store_true',
                        help='Send ping')
    parser.add_argument('--version', action='store_true',
                        help='Request firmware version')
    parser.add_argument('--physics', action='store_true',
                        help='Request physics constants')

    args = parser.parse_args()

    # List ports
    if args.list_ports:
        ports = SerialProtocol.list_ports()
        if ports:
            print("Available serial ports:")
            for p in ports:
                print(f"  {p}")
        else:
            print("No serial ports found")
        return

    # Create protocol
    config = SerialConfig(port=args.port, baud=args.baud)
    protocol = SerialProtocol(config=config)

    # Connect
    if not protocol.connect():
        print("Failed to connect")
        return

    # Register callbacks for monitoring
    if args.monitor:
        def on_state(state: SpinnerState):
            k_str = " K-FORMATION" if state.k_formation else ""
            print(f"\r[STATE] z={state.z:.4f} S={state.delta_s_neg:.4f} "
                  f"{state.tier_name:10s} {state.phase:10s}{k_str}",
                  end='', flush=True)

        def on_k_formation(state: SpinnerState):
            print(f"\n[K-FORMATION] z={state.z:.4f} kappa={state.kappa:.4f}")

        protocol.on_state(on_state)
        protocol.on_k_formation(on_k_formation)

    # Start receiver
    protocol.start()
    time.sleep(0.5)  # Allow time for connection

    try:
        # Execute requested action
        if args.set_z is not None:
            print(f"Setting z to {args.set_z}")
            protocol.set_z(args.set_z)

        if args.stop:
            print("Sending stop command")
            protocol.stop()

        if args.hex_cycle:
            print(f"Starting hex cycle (dwell={args.dwell}s, cycles={args.cycles})")
            protocol.hex_cycle(dwell_s=args.dwell, cycles=args.cycles)

        if args.ping:
            print("Sending ping")
            protocol.ping()

        if args.version:
            print("Requesting version")
            protocol.request_version()

        if args.physics:
            print("Requesting physics")
            protocol.request_physics()

        # Monitor mode - run until Ctrl+C
        if args.monitor:
            print("\nMonitoring... Press Ctrl+C to stop\n")
            while True:
                time.sleep(0.1)
        else:
            # Wait briefly for response
            time.sleep(1.0)

    except KeyboardInterrupt:
        print("\n\nInterrupted")

    finally:
        protocol.close()
        print(f"Stats: {asdict(protocol.stats)}")


if __name__ == "__main__":
    main()
