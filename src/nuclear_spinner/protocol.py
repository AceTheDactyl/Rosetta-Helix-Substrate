#!/usr/bin/env python3
"""
Nuclear Spinner Communication Protocol
=======================================

Defines the communication protocol between host software and
firmware for the Nuclear Spinner.

Protocol Features:
- Command frame format with header, payload, and CRC
- Command types for all spinner operations
- Response frame format with status codes
- Encoding and decoding functions

Frame Format:
    | Header (1 byte) | Payload Length (2 bytes) | Payload (N bytes) | CRC (2 bytes) |

Signature: nuclear-spinner-protocol|v1.0.0|helix
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Union
from enum import IntEnum
import zlib


__all__ = [
    "CommandType",
    "ResponseStatus",
    "CommandFrame",
    "ResponseFrame",
    "encode_command",
    "decode_response",
    "encode_payload",
    "decode_payload",
    "compute_crc16",
    "PROTOCOL_VERSION",
]


# =============================================================================
# PROTOCOL CONSTANTS
# =============================================================================

PROTOCOL_VERSION = "1.0.0"
PROTOCOL_HEADER = 0xAA
MAX_PAYLOAD_SIZE = 256


# =============================================================================
# COMMAND TYPES
# =============================================================================

class CommandType(IntEnum):
    """Command type codes for spinner control."""
    # Core control
    SET_Z = 0x01            # Set target z-coordinate
    RUN_PULSE = 0x02        # Execute RF pulse
    GET_METRICS = 0x03      # Request current metrics
    SET_ROTOR_RATE = 0x04   # Set rotor speed

    # Recording control
    START_RECORDING = 0x05  # Start neural recording
    STOP_RECORDING = 0x06   # Stop neural recording

    # Operator control
    SET_OPERATOR_MASK = 0x07  # Set allowed operator mask
    APPLY_OPERATOR = 0x08     # Apply specific operator

    # Cross-frequency control
    CONFIGURE_RATIO = 0x09  # Configure cross-frequency ratio

    # System control
    INITIALIZE = 0x10       # Initialize system
    RESET = 0x11            # Reset to defaults
    EMERGENCY_STOP = 0x12   # Emergency stop

    # Configuration
    SET_FIELD = 0x20        # Set magnetic field strength
    SET_TEMPERATURE = 0x21  # Set target temperature

    # Data streaming
    STREAM_START = 0x30     # Start data streaming
    STREAM_STOP = 0x31      # Stop data streaming
    FETCH_DATA = 0x32       # Fetch buffered data


class ResponseStatus(IntEnum):
    """Response status codes."""
    OK = 0x00               # Success
    ERROR_INVALID_CMD = 0x01  # Invalid command code
    ERROR_INVALID_PARAM = 0x02  # Invalid parameter value
    ERROR_BUSY = 0x03       # System busy
    ERROR_SAFETY = 0x04     # Safety interlock triggered
    ERROR_CRC = 0x05        # CRC mismatch
    ERROR_TIMEOUT = 0x06    # Operation timeout
    ERROR_NOT_INIT = 0x07   # System not initialized


# =============================================================================
# FRAME STRUCTURES
# =============================================================================

@dataclass
class CommandFrame:
    """
    Command frame for sending commands to the spinner.

    Attributes:
        command: Command type code
        payload: Command-specific payload bytes
        sequence: Sequence number for tracking
    """
    command: CommandType
    payload: bytes = b''
    sequence: int = 0

    def to_bytes(self) -> bytes:
        """Encode frame to bytes for transmission."""
        return encode_command(self.command, self.payload, self.sequence)

    @classmethod
    def from_bytes(cls, data: bytes) -> 'CommandFrame':
        """Decode frame from received bytes."""
        if len(data) < 6:
            raise ValueError("Frame too short")

        header = data[0]
        if header != PROTOCOL_HEADER:
            raise ValueError(f"Invalid header: {header:#x}")

        payload_len = struct.unpack('<H', data[1:3])[0]
        if len(data) < 5 + payload_len:
            raise ValueError("Incomplete frame")

        command = CommandType(data[3])
        sequence = data[4]
        payload = data[5:5 + payload_len - 2]

        # Verify CRC
        received_crc = struct.unpack('<H', data[5 + payload_len - 2:5 + payload_len])[0]
        computed_crc = compute_crc16(data[:5 + payload_len - 2])

        if received_crc != computed_crc:
            raise ValueError("CRC mismatch")

        return cls(command=command, payload=payload, sequence=sequence)


@dataclass
class ResponseFrame:
    """
    Response frame from the spinner.

    Attributes:
        status: Response status code
        payload: Response data bytes
        sequence: Sequence number (matches command)
    """
    status: ResponseStatus
    payload: bytes = b''
    sequence: int = 0

    def to_bytes(self) -> bytes:
        """Encode frame to bytes."""
        # Header + payload length + status + sequence + payload + CRC
        payload_len = 2 + len(self.payload) + 2  # status + sequence + payload + crc
        frame = bytes([PROTOCOL_HEADER])
        frame += struct.pack('<H', payload_len)
        frame += bytes([self.status, self.sequence])
        frame += self.payload

        crc = compute_crc16(frame)
        frame += struct.pack('<H', crc)

        return frame

    @classmethod
    def from_bytes(cls, data: bytes) -> 'ResponseFrame':
        """Decode frame from received bytes."""
        if len(data) < 6:
            raise ValueError("Frame too short")

        header = data[0]
        if header != PROTOCOL_HEADER:
            raise ValueError(f"Invalid header: {header:#x}")

        payload_len = struct.unpack('<H', data[1:3])[0]
        if len(data) < 3 + payload_len:
            raise ValueError("Incomplete frame")

        status = ResponseStatus(data[3])
        sequence = data[4]
        payload = data[5:3 + payload_len - 2]

        # Verify CRC
        received_crc = struct.unpack('<H', data[3 + payload_len - 2:3 + payload_len])[0]
        computed_crc = compute_crc16(data[:3 + payload_len - 2])

        if received_crc != computed_crc:
            raise ValueError("CRC mismatch")

        return cls(status=status, payload=payload, sequence=sequence)


# =============================================================================
# ENCODING/DECODING FUNCTIONS
# =============================================================================

def compute_crc16(data: bytes) -> int:
    """
    Compute CRC-16/CCITT checksum.

    Args:
        data: Input bytes

    Returns:
        16-bit CRC value
    """
    # Use CRC-32 and truncate to 16 bits for simplicity
    crc32 = zlib.crc32(data) & 0xFFFFFFFF
    return crc32 & 0xFFFF


def encode_command(
    command: CommandType,
    payload: bytes = b'',
    sequence: int = 0
) -> bytes:
    """
    Encode a command into a frame for transmission.

    Frame format:
        Header (1) + PayloadLen (2) + Command (1) + Seq (1) + Payload (N) + CRC (2)

    Args:
        command: Command type
        payload: Command-specific payload
        sequence: Sequence number (0-255)

    Returns:
        Encoded frame bytes
    """
    if len(payload) > MAX_PAYLOAD_SIZE - 4:  # Reserve for cmd, seq, crc
        raise ValueError(f"Payload too large: {len(payload)} > {MAX_PAYLOAD_SIZE - 4}")

    # Payload length includes command, sequence, payload, and CRC
    payload_len = 2 + len(payload) + 2

    frame = bytes([PROTOCOL_HEADER])
    frame += struct.pack('<H', payload_len)
    frame += bytes([command, sequence & 0xFF])
    frame += payload

    crc = compute_crc16(frame)
    frame += struct.pack('<H', crc)

    return frame


def decode_response(data: bytes) -> ResponseFrame:
    """
    Decode a response frame from received bytes.

    Args:
        data: Received bytes

    Returns:
        Decoded ResponseFrame

    Raises:
        ValueError: If frame is invalid or CRC fails
    """
    return ResponseFrame.from_bytes(data)


# =============================================================================
# PAYLOAD ENCODING HELPERS
# =============================================================================

def encode_payload(data: Dict[str, Any]) -> bytes:
    """
    Encode a dictionary payload to bytes.

    Simple encoding: key-value pairs as type-length-value format.

    Supported types:
        - float: 4 bytes IEEE 754
        - int: 4 bytes signed
        - str: length-prefixed UTF-8
        - bytes: length-prefixed raw

    Args:
        data: Dictionary to encode

    Returns:
        Encoded bytes
    """
    result = b''

    for key, value in data.items():
        # Encode key as length-prefixed string
        key_bytes = key.encode('utf-8')
        result += bytes([len(key_bytes)])
        result += key_bytes

        # Encode value based on type
        if isinstance(value, float):
            result += bytes([0x01])  # Type: float
            result += struct.pack('<f', value)
        elif isinstance(value, int):
            result += bytes([0x02])  # Type: int
            result += struct.pack('<i', value)
        elif isinstance(value, str):
            result += bytes([0x03])  # Type: string
            str_bytes = value.encode('utf-8')
            result += struct.pack('<H', len(str_bytes))
            result += str_bytes
        elif isinstance(value, bytes):
            result += bytes([0x04])  # Type: bytes
            result += struct.pack('<H', len(value))
            result += value
        else:
            raise ValueError(f"Unsupported type: {type(value)}")

    return result


def decode_payload(data: bytes) -> Dict[str, Any]:
    """
    Decode bytes to a dictionary payload.

    Inverse of encode_payload.

    Args:
        data: Encoded bytes

    Returns:
        Decoded dictionary
    """
    result = {}
    pos = 0

    while pos < len(data):
        # Read key
        key_len = data[pos]
        pos += 1
        key = data[pos:pos + key_len].decode('utf-8')
        pos += key_len

        # Read type
        value_type = data[pos]
        pos += 1

        # Read value based on type
        if value_type == 0x01:  # float
            value = struct.unpack('<f', data[pos:pos + 4])[0]
            pos += 4
        elif value_type == 0x02:  # int
            value = struct.unpack('<i', data[pos:pos + 4])[0]
            pos += 4
        elif value_type == 0x03:  # string
            str_len = struct.unpack('<H', data[pos:pos + 2])[0]
            pos += 2
            value = data[pos:pos + str_len].decode('utf-8')
            pos += str_len
        elif value_type == 0x04:  # bytes
            bytes_len = struct.unpack('<H', data[pos:pos + 2])[0]
            pos += 2
            value = data[pos:pos + bytes_len]
            pos += bytes_len
        else:
            raise ValueError(f"Unknown value type: {value_type}")

        result[key] = value

    return result


# =============================================================================
# COMMAND-SPECIFIC PAYLOAD BUILDERS
# =============================================================================

def build_set_z_payload(z_target: float) -> bytes:
    """Build payload for SET_Z command."""
    return struct.pack('<f', z_target)


def build_pulse_payload(
    amplitude: float,
    phase: float,
    duration_us: float
) -> bytes:
    """Build payload for RUN_PULSE command."""
    return struct.pack('<fff', amplitude, phase, duration_us)


def build_rotor_rate_payload(rate_hz: float) -> bytes:
    """Build payload for SET_ROTOR_RATE command."""
    return struct.pack('<f', rate_hz)


def build_configure_ratio_payload(
    band_low: float,
    ratio: float
) -> bytes:
    """Build payload for CONFIGURE_RATIO command."""
    return struct.pack('<ff', band_low, ratio)


def build_operator_mask_payload(mask: int) -> bytes:
    """Build payload for SET_OPERATOR_MASK command."""
    return bytes([mask & 0xFF])


def build_apply_operator_payload(operator: str) -> bytes:
    """Build payload for APPLY_OPERATOR command."""
    # Encode operator as single byte
    op_codes = {
        "()": 0x01,
        "x": 0x02,
        "^": 0x03,
        "/": 0x04,
        "+": 0x05,
        "-": 0x06,
    }
    code = op_codes.get(operator, 0x00)
    return bytes([code])


# =============================================================================
# RESPONSE PARSERS
# =============================================================================

def parse_metrics_response(payload: bytes) -> Dict[str, float]:
    """Parse payload from GET_METRICS response."""
    if len(payload) < 20:
        return {}

    z, delta_s_neg, gradient, kappa, eta = struct.unpack('<fffff', payload[:20])

    return {
        "z": z,
        "delta_s_neg": delta_s_neg,
        "gradient": gradient,
        "kappa": kappa,
        "eta": eta,
    }


def parse_recording_response(payload: bytes) -> Dict[str, Any]:
    """Parse payload from recording data response."""
    if len(payload) < 4:
        return {"samples": []}

    n_samples = struct.unpack('<I', payload[:4])[0]
    samples = []

    pos = 4
    for _ in range(n_samples):
        if pos + 4 <= len(payload):
            sample = struct.unpack('<f', payload[pos:pos + 4])[0]
            samples.append(sample)
            pos += 4

    return {"samples": samples}
