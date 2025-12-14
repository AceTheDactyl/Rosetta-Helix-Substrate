#!/usr/bin/env python3
"""
Test script for the Nuclear Spinner Bridge connectivity.

Usage:
    python scripts/test_bridge.py
"""

import asyncio
import json
import sys
import math

# Constants for verification
Z_CRITICAL = math.sqrt(3) / 2
PHI_INV = (math.sqrt(5) - 1) / 2

async def test_bridge_connectivity():
    """Test WebSocket connectivity to the bridge."""
    try:
        import websockets
    except ImportError:
        print("[ERROR] websockets not installed")
        return False

    print("=" * 60)
    print("Nuclear Spinner Bridge Connectivity Test")
    print("=" * 60)

    uri = "ws://localhost:8765"

    try:
        print(f"\n[TEST] Connecting to {uri}...")
        async with websockets.connect(uri, close_timeout=5) as ws:
            print("[TEST] ✓ Connected to bridge")

            # Wait for initial state
            print("[TEST] Waiting for state broadcast...")
            message = await asyncio.wait_for(ws.recv(), timeout=5.0)
            data = json.loads(message)

            if data.get('type') == 'spinner_state':
                print("[TEST] ✓ Received spinner_state")
                print(f"       z = {data.get('z', 0):.6f}")
                print(f"       ΔS_neg = {data.get('delta_s_neg', 0):.6f}")
                print(f"       tier = {data.get('tier_name', 'unknown')}")
                print(f"       phase = {data.get('phase', 'unknown')}")

            # Send command to drive toward z_c
            print(f"\n[TEST] Sending set_z command (z_c = {Z_CRITICAL:.6f})...")
            cmd = {"cmd": "set_z", "value": Z_CRITICAL}
            await ws.send(json.dumps(cmd))
            print("[TEST] ✓ Command sent")

            # Wait for state to update
            print("[TEST] Waiting for state update...")
            await asyncio.sleep(1.0)

            # Receive updated states
            received_states = 0
            while received_states < 3:
                try:
                    message = await asyncio.wait_for(ws.recv(), timeout=2.0)
                    data = json.loads(message)
                    if data.get('type') == 'spinner_state':
                        received_states += 1
                        z = data.get('z', 0)
                        print(f"[TEST] State {received_states}: z={z:.6f}, ΔS_neg={data.get('delta_s_neg', 0):.6f}")
                except asyncio.TimeoutError:
                    break

            # Check for unified state
            print("\n[TEST] Checking for unified_state broadcasts...")
            unified_received = False
            for _ in range(10):
                try:
                    message = await asyncio.wait_for(ws.recv(), timeout=0.5)
                    data = json.loads(message)
                    if data.get('type') == 'unified_state':
                        unified_received = True
                        print("[TEST] ✓ Received unified_state")
                        print(f"       Kuramoto coherence: {data.get('kuramoto', {}).get('coherence', 0):.4f}")
                        print(f"       K-formation: {data.get('triad', {}).get('k_formation', False)}")
                        break
                except asyncio.TimeoutError:
                    continue

            if not unified_received:
                print("[TEST] ⚠ No unified_state received (may be normal if rate limited)")

            # Send stop command
            print("\n[TEST] Sending stop command...")
            await ws.send(json.dumps({"cmd": "stop"}))
            print("[TEST] ✓ Stop command sent")

            print("\n" + "=" * 60)
            print("[TEST] ✓ All connectivity tests passed!")
            print("=" * 60)
            return True

    except asyncio.TimeoutError:
        print("[ERROR] Connection timeout - is the bridge running?")
        return False
    except ConnectionRefusedError:
        print("[ERROR] Connection refused - bridge not running on port 8765")
        return False
    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}")
        return False


def main():
    print("\nNote: Ensure the bridge is running first:")
    print("  python bridge/spinner_bridge.py --simulate\n")

    result = asyncio.run(test_bridge_connectivity())
    sys.exit(0 if result else 1)


if __name__ == "__main__":
    main()
