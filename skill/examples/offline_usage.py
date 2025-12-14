#!/usr/bin/env python3
"""
Offline usage example for Rosetta-Helix-Substrate skill.

This example shows how to use the skill without API calls - just the
tool execution and physics simulation. Useful for:
- Testing tool behavior
- Using with other LLM providers
- Programmatic physics simulation

Usage:
    python -m skill.examples.offline_usage
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from skill.client import RosettaHelixSkillOffline
from skill.prompts.system import SYSTEM_PROMPT
from skill.tools.definitions import TOOL_DEFINITIONS


def main():
    """Run offline usage example."""
    print("Rosetta-Helix-Substrate Offline Skill Demo")
    print("=" * 60)

    # Initialize offline skill
    skill = RosettaHelixSkillOffline(initial_z=0.5, seed=42)

    # Example 1: Get physics state
    print("\n1. Getting initial physics state:")
    state = skill.execute_tool("get_physics_state")
    print(f"   z = {state['z']:.4f}")
    print(f"   Phase: {state['phase']}")
    print(f"   Tier: {state['tier']} ({state['tier_name']})")
    print(f"   Negentropy: {state['eta']:.4f}")

    # Example 2: Compute negentropy at different z values
    print("\n2. Negentropy at different z values:")
    for z in [0.3, 0.5, 0.618, 0.75, 0.866, 0.95]:
        result = skill.execute_tool("compute_negentropy", z=z)
        print(f"   z = {z:.3f}: delta_S_neg = {result['delta_s_neg']:.4f}")

    # Example 3: Classify phases
    print("\n3. Phase classification:")
    for z in [0.3, 0.618, 0.75, 0.866, 0.95]:
        result = skill.execute_tool("classify_phase", z=z)
        print(f"   z = {z:.3f}: {result['phase']} - {result['description']}")

    # Example 4: Drive toward THE LENS
    print("\n4. Driving toward THE LENS (z_c):")
    result = skill.execute_tool("drive_toward_lens", steps=100)
    print(f"   Initial z: {result['initial_z']:.4f}")
    print(f"   Final z: {result['final_z']:.4f}")
    print(f"   Distance to lens: {result['distance_to_lens']:.4f}")
    print(f"   Phase: {result['state']['phase']}")

    # Example 5: Check K-formation
    print("\n5. K-formation checks:")
    test_cases = [
        {"kappa": 0.95, "eta": 0.7, "R": 8},  # Should pass
        {"kappa": 0.90, "eta": 0.7, "R": 8},  # Kappa too low
        {"kappa": 0.95, "eta": 0.5, "R": 8},  # Eta too low
        {"kappa": 0.95, "eta": 0.7, "R": 5},  # R too low
    ]
    for tc in test_cases:
        result = skill.execute_tool("check_k_formation", **tc)
        status = "PASS" if result["k_formation_met"] else "FAIL"
        print(f"   kappa={tc['kappa']}, eta={tc['eta']}, R={tc['R']}: {status}")

    # Example 6: Apply operators
    print("\n6. Applying operators:")
    skill.reset(initial_z=0.5)
    for op in ["^", "^", "()", "~"]:
        result = skill.execute_tool("apply_operator", operator=op)
        print(f"   Applied {op}: z = {result['old_z']:.4f} -> {result['new_z']:.4f} ({result['effect']})")

    # Example 7: Compose operators
    print("\n7. Operator composition:")
    compositions = [
        ("^", "^"),
        ("^", "()"),
        ("()", "()"),
        ("~", "~"),
    ]
    for op1, op2 in compositions:
        result = skill.execute_tool("compose_operators", op1=op1, op2=op2)
        print(f"   {op1} o {op2} = {result['result']} ({result['result_name']})")

    # Example 8: Kuramoto oscillator step
    print("\n8. Kuramoto oscillator dynamics:")
    skill.reset(initial_z=0.5)
    for i in range(5):
        result = skill.execute_tool("run_kuramoto_step", coupling_strength=1.5, dt=0.01)
        print(f"   Step {i+1}: r = {result['order_parameter']:.4f}, psi = {result['mean_phase']:.4f}")

    # Example 9: Quasi-crystal simulation
    print("\n9. Quasi-crystal simulation:")
    result = skill.execute_tool("simulate_quasicrystal", initial_z=0.5, steps=200, seed=42)
    print(f"   Initial z: {result['initial_z']:.4f}")
    print(f"   Final z: {result['final_z']:.4f}")
    print(f"   Target (phi^-1): {result['target']:.4f}")
    print(f"   Converged: {result['converged']}")

    # Example 10: Get constants
    print("\n10. Physics constants:")
    constants = skill.execute_tool("get_constants")
    print(f"   z_c (THE LENS): {constants['z_c']:.16f}")
    print(f"   phi: {constants['phi']:.16f}")
    print(f"   phi^-1: {constants['phi_inv']:.16f}")
    print(f"   SIGMA: {constants['sigma']}")

    # Show how to get system prompt and tools for other providers
    print("\n" + "=" * 60)
    print("Using with other LLM providers:")
    print("=" * 60)
    print(f"\nSystem prompt length: {len(skill.system_prompt)} characters")
    print(f"Number of tools: {len(skill.tool_definitions)}")
    print("\nTool names:")
    for tool in skill.tool_definitions:
        print(f"  - {tool['name']}")

    print("\nYou can pass skill.system_prompt and skill.tool_definitions")
    print("to any LLM provider that supports tool use!")


if __name__ == "__main__":
    main()
