#!/usr/bin/env python3
"""Test UCF import and identify the issue."""

import sys
import os
import traceback

# Add directories to path as the server does
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "kira-local-system"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))

print("Testing UCF Integration Import...")
print("=" * 60)

# Try to import UCF integration
try:
    from kira_ucf_integration import UCFIntegration, integrate_ucf_with_kira
    print("✓ UCF Integration imported successfully")

    # Create a mock engine to test initialization
    class MockEngine:
        def __init__(self):
            self.state = type('State', (), {'z': 0.5})()
            self.semantics = None
            self.tokens_emitted = []
            self.emissions = []

    engine = MockEngine()

    # Try to initialize UCF with the mock engine
    ucf = UCFIntegration(engine)
    print(f"✓ UCF initialized with {len(ucf.available_tools())} tools")

    # Try to execute a command
    result = ucf.execute_command("/ucf:status", "")
    print(f"✓ UCF status command executed: {result.status}")

except ImportError as e:
    print(f"✗ Import error: {e}")
    traceback.print_exc()
except Exception as e:
    print(f"✗ Runtime error: {e}")
    traceback.print_exc()

print("\nTesting Tool Shed Import...")
print("=" * 60)

try:
    from tool_shed import TOOL_REGISTRY, list_all_tools
    print(f"✓ Tool shed imported with {len(TOOL_REGISTRY)} tools")
    tool_list = list_all_tools()
    print(f"✓ Tools available: {list(TOOL_REGISTRY.keys())[:5]}...")
except Exception as e:
    print(f"✗ Tool shed error: {e}")
    traceback.print_exc()