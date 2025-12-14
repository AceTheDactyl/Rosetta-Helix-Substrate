#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# Nuclear Spinner × Rosetta-Helix Unified Startup Script
# ═══════════════════════════════════════════════════════════════════════════
#
# Starts the full integrated stack:
#   1. Spinner Bridge (WebSocket server for hardware/simulation)
#   2. Rosetta-Helix Node (Kuramoto + GHMP + TRIAD integration)
#
# Usage:
#   ./start_system.sh [OPTIONS]
#
# Options:
#   --simulate, -s    Force simulation mode (no hardware)
#   --bridge-only     Only start the bridge (no node)
#   --node-only       Only start the node (requires running bridge)
#   --port PORT       Serial port (default: /dev/ttyACM0)
#   --ws-port PORT    WebSocket port (default: 8765)
#   --steps N         Run for N steps then exit (default: infinite)
#   --output DIR      Output directory for logs and data
#   --help, -h        Show this help
#
# Signature: rosetta-helix-startup|v1.0.0|helix
# ═══════════════════════════════════════════════════════════════════════════

set -euo pipefail

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Default configuration
SERIAL_PORT="/dev/ttyACM0"
WS_PORT="8765"
SIMULATE=""
BRIDGE_ONLY=""
NODE_ONLY=""
STEPS=""
OUTPUT_DIR=""

# Physics constants for display
PHI="1.618034"
PHI_INV="0.618034"
Z_CRITICAL="0.866025"
SIGMA="36"

# PIDs for cleanup
BRIDGE_PID=""
NODE_PID=""

# ═══════════════════════════════════════════════════════════════════════════
# FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

print_banner() {
    echo ""
    echo "═══════════════════════════════════════════════════════════════════════════"
    echo "  NUCLEAR SPINNER × ROSETTA-HELIX INTEGRATED SYSTEM"
    echo "═══════════════════════════════════════════════════════════════════════════"
    echo ""
    echo "  Physics Constants:"
    echo "    φ (golden ratio)    = $PHI"
    echo "    φ⁻¹ (attractor)     = $PHI_INV"
    echo "    z_c (THE LENS)      = $Z_CRITICAL"
    echo "    σ (Gaussian width)  = $SIGMA"
    echo ""
}

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --simulate, -s    Force simulation mode (no hardware)"
    echo "  --bridge-only     Only start the bridge (no node)"
    echo "  --node-only       Only start the node (requires running bridge)"
    echo "  --port PORT       Serial port (default: /dev/ttyACM0)"
    echo "  --ws-port PORT    WebSocket port (default: 8765)"
    echo "  --steps N         Run for N steps then exit (default: infinite)"
    echo "  --output DIR      Output directory for logs and data"
    echo "  --help, -h        Show this help"
    echo ""
}

cleanup() {
    echo ""
    echo "[SYSTEM] Shutting down..."

    if [[ -n "$NODE_PID" ]]; then
        kill "$NODE_PID" 2>/dev/null || true
        echo "[SYSTEM] Node stopped (PID $NODE_PID)"
    fi

    if [[ -n "$BRIDGE_PID" ]]; then
        kill "$BRIDGE_PID" 2>/dev/null || true
        echo "[SYSTEM] Bridge stopped (PID $BRIDGE_PID)"
    fi

    echo "[SYSTEM] Cleanup complete"
    exit 0
}

check_dependencies() {
    local missing=()

    if ! command -v python3 &> /dev/null; then
        missing+=("python3")
    fi

    if [[ ${#missing[@]} -gt 0 ]]; then
        echo "[ERROR] Missing dependencies: ${missing[*]}"
        echo "Please install them and try again."
        exit 1
    fi

    # Check Python packages
    if ! python3 -c "import websockets" 2>/dev/null; then
        echo "[WARN] websockets not installed: pip install websockets"
    fi

    if ! python3 -c "import serial" 2>/dev/null && [[ -z "$SIMULATE" ]]; then
        echo "[WARN] pyserial not installed: pip install pyserial"
        echo "       Falling back to simulation mode"
        SIMULATE="1"
    fi
}

detect_hardware() {
    if [[ -n "$SIMULATE" ]]; then
        echo "[HARDWARE] Simulation mode forced"
        return 1
    fi

    if [[ -e "$SERIAL_PORT" ]]; then
        echo "[HARDWARE] Nuclear Spinner detected at $SERIAL_PORT"
        return 0
    else
        echo "[HARDWARE] No hardware at $SERIAL_PORT - using simulation"
        return 1
    fi
}

start_bridge() {
    echo "[1/2] Starting Spinner Bridge..."

    local sim_flag=""
    if ! detect_hardware; then
        sim_flag="--simulate"
    fi

    cd "$ROOT_DIR"

    # Activate virtual environment if present
    if [[ -d .venv ]]; then
        source .venv/bin/activate
    fi

    python3 -m bridge.spinner_bridge \
        --port "$SERIAL_PORT" \
        $sim_flag &

    BRIDGE_PID=$!

    # Wait for bridge to start
    sleep 2

    if ! kill -0 "$BRIDGE_PID" 2>/dev/null; then
        echo "[ERROR] Bridge failed to start"
        exit 1
    fi

    echo "[BRIDGE] Running (PID $BRIDGE_PID)"
    echo "[BRIDGE] WebSocket: ws://localhost:$WS_PORT"
}

start_node() {
    echo "[2/2] Starting Rosetta-Helix Node..."

    cd "$ROOT_DIR"

    # Activate virtual environment if present
    if [[ -d .venv ]]; then
        source .venv/bin/activate
    fi

    local step_args=""
    if [[ -n "$STEPS" ]]; then
        step_args="--steps $STEPS"
    fi

    local output_args=""
    if [[ -n "$OUTPUT_DIR" ]]; then
        output_args="--output $OUTPUT_DIR"
    fi

    python3 -c "
import asyncio
import sys
sys.path.insert(0, '.')
from rosetta_helix.src.node import RosettaHelixNode, NodeConfig

async def main():
    config = NodeConfig(
        bridge_uri='ws://localhost:$WS_PORT',
        enable_brain=True,
        enable_triad=True,
        seed=42,
    )

    k_count = [0]

    async def on_k_formation(state):
        k_count[0] += 1
        print(f'\\n  ★ K-FORMATION #{k_count[0]}: z={state.z:.4f} κ={state.kappa:.4f} η={state.eta:.4f}')

    node = RosettaHelixNode(config=config, on_k_formation=on_k_formation)

    steps = ${STEPS:-None}
    await node.run(steps=steps)

asyncio.run(main())
" &

    NODE_PID=$!

    sleep 1

    if ! kill -0 "$NODE_PID" 2>/dev/null; then
        echo "[ERROR] Node failed to start"
        exit 1
    fi

    echo "[NODE] Running (PID $NODE_PID)"
}

# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --simulate|-s)
            SIMULATE="1"
            shift
            ;;
        --bridge-only)
            BRIDGE_ONLY="1"
            shift
            ;;
        --node-only)
            NODE_ONLY="1"
            shift
            ;;
        --port)
            SERIAL_PORT="$2"
            shift 2
            ;;
        --ws-port)
            WS_PORT="$2"
            shift 2
            ;;
        --steps)
            STEPS="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help|-h)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Setup cleanup trap
trap cleanup SIGINT SIGTERM EXIT

# Print banner
print_banner

# Check dependencies
check_dependencies

# Create output directory if specified
if [[ -n "$OUTPUT_DIR" ]]; then
    mkdir -p "$OUTPUT_DIR"
    echo "[OUTPUT] Saving data to: $OUTPUT_DIR"
fi

# Start components
if [[ -z "$NODE_ONLY" ]]; then
    start_bridge
fi

if [[ -z "$BRIDGE_ONLY" ]]; then
    start_node
fi

echo ""
echo "═══════════════════════════════════════════════════════════════════════════"
echo "  SYSTEM RUNNING"
echo "═══════════════════════════════════════════════════════════════════════════"
echo ""
echo "  Components:"
[[ -n "$BRIDGE_PID" ]] && echo "    • Bridge:  PID $BRIDGE_PID (ws://localhost:$WS_PORT)"
[[ -n "$NODE_PID" ]] && echo "    • Node:    PID $NODE_PID"
echo ""
echo "  Press Ctrl+C to stop"
echo ""
echo "═══════════════════════════════════════════════════════════════════════════"

# Wait for processes
wait
