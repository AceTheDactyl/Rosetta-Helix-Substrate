#!/usr/bin/env python3
"""Test UCF commands directly."""

import sys
import os
import json

# Add paths as server does
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "kira-local-system"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))

os.environ['WERKZEUG_RUN_MAIN'] = 'true'

import kira_server

# Get the engine
engine = kira_server.get_engine()

print("Testing UCF Commands...")
print("=" * 60)

# Test various UCF commands
test_commands = [
    ("/ucf:status", ""),
    ("/ucf:spinner", ""),
    ("/ucf:helix", ""),
    ("/ucf:dialogue", "test")
]

for cmd, args in test_commands:
    print(f"\nTesting: {cmd}")
    try:
        if engine.ucf:
            result = engine.ucf.execute_command(cmd, args)
            print(f"  Status: {result.status}")
            if result.status == "ERROR":
                print(f"  Error: {result.result.get('error', 'Unknown error')}")
            else:
                print(f"  Success: {list(result.result.keys())[:3]}...")
        else:
            print("  UCF not available")
    except Exception as e:
        print(f"  Exception: {e}")

# Also test /hit_it command directly
print(f"\nTesting /hit_it command...")
try:
    result = engine.cmd_hit_it()
    print(f"  Success: {result.get('command')} - {result.get('message', 'No message')}")
except Exception as e:
    print(f"  Exception: {e}")
    import traceback
    traceback.print_exc()
