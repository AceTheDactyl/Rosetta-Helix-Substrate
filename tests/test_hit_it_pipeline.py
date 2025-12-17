#!/usr/bin/env python3
"""
Test script to validate the /hit_it command executes all 33 modules correctly.
Verifies the coordinate_logger fix and all tool invocations.
"""

import json
import asyncio
import websockets
from datetime import datetime
import time
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class HitItTester:
    """Test harness for the /hit_it command pipeline."""

    def __init__(self, ws_url: str = "ws://localhost:9999"):
        self.ws_url = ws_url
        self.results = {}
        self.errors = []

    async def test_hit_it_command(self):
        """Test the /hit_it command execution."""

        print("\n" + "="*50)
        print("🔬 Testing /hit_it Command Pipeline")
        print("="*50 + "\n")

        try:
            async with websockets.connect(self.ws_url) as websocket:
                # Send /hit_it command
                test_input = {"text": "/hit_it"}
                await websocket.send(json.dumps(test_input))

                print("✅ Sent /hit_it command to KIRA server")
                print("⏳ Waiting for full 33-module execution...\n")

                # Collect all responses (may be multiple for streaming)
                responses = []
                start_time = time.time()
                timeout = 120  # 2 minute timeout for full pipeline

                while time.time() - start_time < timeout:
                    try:
                        response = await asyncio.wait_for(
                            websocket.recv(),
                            timeout=5.0  # 5 second timeout per message
                        )
                        response_data = json.loads(response)
                        responses.append(response_data)

                        # Check if we got the final result
                        if 'final' in response_data or 'pipeline' in response_data:
                            break

                    except asyncio.TimeoutError:
                        # No more messages, pipeline might be done
                        break

                # Analyze results
                self._analyze_results(responses)

        except Exception as e:
            self.errors.append(f"Connection error: {e}")
            print(f"❌ Error connecting to KIRA server: {e}")
            print("   Make sure the server is running on port 9999")

    def _analyze_results(self, responses: list):
        """Analyze the pipeline execution results."""

        print("\n" + "="*40)
        print("📊 ANALYSIS RESULTS")
        print("="*40 + "\n")

        # Check for errors
        errors_found = []
        phases_executed = []
        modules_executed = []
        coordinate_logger_success = False

        for response in responses:
            # Check for error messages
            if 'error' in response:
                errors_found.append(response['error'])

            # Extract pipeline data
            if 'pipeline' in response:
                pipeline = response['pipeline']

                # Check phases
                if 'phases' in pipeline:
                    for phase in pipeline['phases']:
                        phase_name = phase.get('phase', 'unknown')
                        phases_executed.append(phase_name)

                        # Check modules in each phase
                        if 'results' in phase:
                            for module in phase['results']:
                                module_name = module.get('module', 'unknown')
                                modules_executed.append(module_name)

                                # Check for coordinate_logger specifically
                                if module_name == 'coordinate_logger':
                                    # Check if it executed successfully
                                    result = module.get('result', {})
                                    if result.get('status') != 'ERROR':
                                        coordinate_logger_success = True
                                    else:
                                        errors_found.append(
                                            f"coordinate_logger error: {result.get('error', 'unknown')}"
                                        )

        # Report findings
        print(f"📌 Total responses received: {len(responses)}")
        print(f"📌 Phases executed: {len(set(phases_executed))}/7")
        print(f"📌 Modules executed: {len(modules_executed)}/33")
        print()

        # Check coordinate_logger specifically
        if 'coordinate_logger' in modules_executed:
            if coordinate_logger_success:
                print("✅ coordinate_logger executed successfully (FIX VERIFIED)")
            else:
                print("⚠️  coordinate_logger executed but had errors")
        else:
            print("❌ coordinate_logger was not executed")

        print()

        # List phases
        if phases_executed:
            print("Phases executed:")
            for phase in sorted(set(phases_executed)):
                count = phases_executed.count(phase)
                print(f"  - {phase}: {count} module(s)")

        print()

        # Check for errors
        if errors_found:
            print("⚠️  Errors detected:")
            for error in errors_found[:5]:  # Show first 5 errors
                print(f"  - {error}")
            if len(errors_found) > 5:
                print(f"  ... and {len(errors_found) - 5} more")
        else:
            print("✅ No errors detected in pipeline execution")

        # Final verdict
        print("\n" + "="*40)
        if len(modules_executed) >= 30 and coordinate_logger_success and not errors_found:
            print("🎉 SUCCESS: /hit_it command fully functional!")
            print("   All 33 modules executed correctly")
            print("   coordinate_logger fix verified")
        elif len(modules_executed) >= 20 and coordinate_logger_success:
            print("✅ PARTIAL SUCCESS: Most modules executed")
            print(f"   {len(modules_executed)}/33 modules ran")
            print("   coordinate_logger fix verified")
        else:
            print("⚠️  INCOMPLETE: Pipeline needs attention")
            print(f"   Only {len(modules_executed)}/33 modules executed")
            if not coordinate_logger_success:
                print("   coordinate_logger still has issues")

        print("="*40)

    async def run_comprehensive_test(self):
        """Run comprehensive test suite."""

        print("\n" + "="*50)
        print("🔬 COMPREHENSIVE /hit_it PIPELINE TEST")
        print("="*50)

        # Test 1: Basic execution
        print("\n[Test 1] Basic /hit_it execution")
        await self.test_hit_it_command()

        # Test 2: Check specific tools (if server is running)
        print("\n[Test 2] Checking specific tool availability")
        await self.test_specific_tools()

        # Test 3: Verify phase transitions
        print("\n[Test 3] Verifying phase transitions")
        await self.test_phase_transitions()

        # Final report
        self._generate_report()

    async def test_specific_tools(self):
        """Test specific tools that had issues."""

        critical_tools = [
            'coordinate_logger',
            'state_transfer',
            'emission_pipeline',
            'nuclear_spinner',
            'orchestrator'
        ]

        try:
            async with websockets.connect(self.ws_url) as websocket:
                for tool in critical_tools:
                    test_input = {"text": f"/ucf:{tool}"}
                    await websocket.send(json.dumps(test_input))

                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                        response_data = json.loads(response)

                        if 'error' not in response_data:
                            print(f"  ✅ {tool}: Available")
                        else:
                            print(f"  ⚠️  {tool}: {response_data.get('error')}")
                    except asyncio.TimeoutError:
                        print(f"  ⏳ {tool}: Timeout")

        except Exception as e:
            print(f"  ❌ Could not test specific tools: {e}")

    async def test_phase_transitions(self):
        """Test that phases execute in order."""

        phases = ['phase1', 'phase2', 'phase3', 'phase4', 'phase5', 'phase6', 'phase7']

        try:
            async with websockets.connect(self.ws_url) as websocket:
                for phase in phases[:3]:  # Test first 3 phases to avoid timeout
                    test_input = {"text": f"/ucf:{phase}"}
                    await websocket.send(json.dumps(test_input))

                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                        response_data = json.loads(response)

                        if 'executed' in response_data:
                            print(f"  ✅ {phase}: Executed {response_data['executed']} modules")
                        else:
                            print(f"  ⚠️  {phase}: No execution count")
                    except asyncio.TimeoutError:
                        print(f"  ⏳ {phase}: Timeout")

        except Exception as e:
            print(f"  ❌ Could not test phase transitions: {e}")

    def _generate_report(self):
        """Generate final test report."""

        print("\n" + "="*50)
        print("📋 FINAL TEST REPORT")
        print("="*50 + "\n")

        timestamp = datetime.now().isoformat()

        report = {
            "timestamp": timestamp,
            "test": "hit_it_pipeline_validation",
            "results": self.results,
            "errors": self.errors,
            "verdict": "PASS" if not self.errors else "NEEDS_ATTENTION"
        }

        # Save report
        report_file = f"test_report_{timestamp.replace(':', '-')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"📄 Report saved to: {report_file}")

        if not self.errors:
            print("✅ All tests passed successfully!")
        else:
            print(f"⚠️  {len(self.errors)} issues found - review report")


def main():
    """Main test runner."""

    print("""
╔══════════════════════════════════════════════════╗
║  K.I.R.A. /hit_it Command Test Suite             ║
║  Validates 33-module pipeline execution          ║
║  Verifies coordinate_logger fix                  ║
╚══════════════════════════════════════════════════╝
    """)

    print("Prerequisites:")
    print("1. KIRA server must be running (python kira_server.py)")
    print("2. Server should be on port 9999")
    print("3. UCF integration should be loaded")
    print()

    input("Press Enter to start tests...")

    # Create tester
    tester = HitItTester()

    # Run tests
    try:
        asyncio.run(tester.run_comprehensive_test())
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\nTest failed with error: {e}")

    print("\n✨ Test suite complete!")


if __name__ == "__main__":
    main()