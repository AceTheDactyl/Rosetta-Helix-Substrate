#!/bin/bash
echo "═══════════════════════════════════════════════════════════════"
echo "  Nuclear Spinner × Rosetta-Helix System"
echo "═══════════════════════════════════════════════════════════════"

# Check for hardware
if [ -e /dev/ttyACM0 ]; then
    echo "[✓] Spinner hardware detected"
    SIM_FLAG=""
else
    echo "[!] No hardware - running simulation"
    SIM_FLAG="--simulate"
fi

# Start bridge
echo "[1/2] Starting bridge service..."
cd "$(dirname "$0")/../spinner_bridge"
python3 spinner_bridge.py $SIM_FLAG &
BRIDGE_PID=$!
sleep 2

echo "[2/2] Bridge running (PID $BRIDGE_PID)"
echo ""
echo "  WebSocket: ws://localhost:8765"
echo "  Press Ctrl+C to stop"
echo "═══════════════════════════════════════════════════════════════"

trap "kill $BRIDGE_PID 2>/dev/null" EXIT
wait
