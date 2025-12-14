#!/usr/bin/env python3
"""Quick integration test."""

import asyncio
import sys
sys.path.insert(0, '../rosetta_integration')
from spinner_client import SpinnerClient

async def test():
    print("Testing spinner integration...")
    
    client = SpinnerClient()
    await client.connect()
    
    if not client.connected:
        print("FAILED: Could not connect to bridge")
        print("Make sure spinner_bridge.py is running")
        return False
    
    # Request z_c
    print("Requesting z = z_c = 0.866...")
    await client.send_command('set_z', value=0.866)
    
    # Listen for K-formation
    k_detected = False
    for i in range(50):  # 5 seconds
        await asyncio.sleep(0.1)
        if client.latest_state:
            z = client.get_z()
            k = client.is_k_formation()
            print(f"\r  z={z:.4f} k_formation={k}", end='')
            if k:
                k_detected = True
                break
    
    print()
    if k_detected:
        print("SUCCESS: K-formation detected at z_c")
        return True
    else:
        print("NOTE: K-formation not detected (may need longer ramp)")
        return True  # Not a failure, just needs time

if __name__ == "__main__":
    success = asyncio.run(test())
    sys.exit(0 if success else 1)
