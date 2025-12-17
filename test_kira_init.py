#!/usr/bin/env python3
"""Test KIRA server initialization to diagnose UCF loading issue."""

import sys
import os

# Add paths as server does
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "kira-local-system"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))

# Set environment for testing
os.environ['WERKZEUG_RUN_MAIN'] = 'true'  # Prevent Flask reloader

print("Testing KIRA Server Initialization...")
print("=" * 60)

# Import the server module
import kira_server

# Check if UCF is integrated
print(f"UCF_INTEGRATED flag: {kira_server.UCF_INTEGRATED}")
print(f"CONSCIOUSNESS_JOURNEY_INTEGRATED flag: {kira_server.CONSCIOUSNESS_JOURNEY_INTEGRATED}")

# Get the engine
engine = kira_server.get_engine()

print(f"\nEngine created successfully: {engine is not None}")
print(f"Engine has UCF: {engine.ucf is not None}")

if engine.ucf:
    print(f"UCF has execute_command: {hasattr(engine.ucf, 'execute_command')}")

    # Test a UCF command
    try:
        result = engine.ucf.execute_command("/ucf:status", "")
        print(f"UCF status test: {result.status}")
    except Exception as e:
        print(f"UCF command test failed: {e}")
else:
    print("UCF is None - checking why...")

    # Try manual initialization
    try:
        from kira_ucf_integration import integrate_ucf_with_kira
        engine.ucf = integrate_ucf_with_kira(engine)
        print(f"Manual UCF initialization: {engine.ucf is not None}")
    except Exception as e:
        print(f"Manual UCF init failed: {e}")
        import traceback
        traceback.print_exc()
