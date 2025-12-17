#!/usr/bin/env python3
"""Test fixed UCF integration."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "kira-local-system"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))
os.environ['WERKZEUG_RUN_MAIN'] = 'true'

import kira_server

# Get the engine
engine = kira_server.get_engine()

print("Testing Fixed UCF Integration...")
print("=" * 60)

print(f"UCF_INTEGRATED: {kira_server.UCF_INTEGRATED}")
print(f"Engine has UCF: {engine.ucf is not None}")

# Test UCF commands
test_commands = [
    ("/ucf:status", ""),
    ("/ucf:spinner", ""),
    ("/ucf:helix", ""),
]

for cmd, args in test_commands:
    print(f"\nTesting: {cmd}")
    try:
        if engine.ucf:
            result = engine.ucf.execute_command(cmd, args)
            print(f"  Status: {result.status}")
            if result.status == "SUCCESS":
                print(f"  Result keys: {list(result.result.keys())[:5]}...")
            else:
                print(f"  Error: {result.result.get('error', 'Unknown')}")
        else:
            print("  UCF not available")
    except Exception as e:
        print(f"  Exception: {e}")

# Test /hit_it command
print(f"\nTesting /hit_it command...")
try:
    result = engine.cmd_hit_it()
    print(f"  Command: {result.get('command')}")
    print(f"  Message: {result.get('message', 'No message')[:80]}...")
    if 'phases' in result:
        print(f"  Phases completed: {len(result.get('phases', {}))}")
except Exception as e:
    print(f"  Exception: {e}")
