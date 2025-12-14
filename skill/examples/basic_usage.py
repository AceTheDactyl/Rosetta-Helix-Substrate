#!/usr/bin/env python3
"""
Basic usage example for Rosetta-Helix-Substrate Claude API Skill.

This example shows how to use the skill with the Claude API.

Requirements:
    pip install anthropic

Usage:
    export ANTHROPIC_API_KEY="your-api-key"
    python -m skill.examples.basic_usage
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from skill import RosettaHelixSkill


def main():
    """Run basic usage example."""
    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        print("Set it with: export ANTHROPIC_API_KEY='your-api-key'")
        return

    # Initialize the skill
    print("Initializing Rosetta-Helix-Substrate skill...")
    skill = RosettaHelixSkill(initial_z=0.5)

    # Example 1: Get current state
    print("\n" + "=" * 60)
    print("Example 1: Getting current physics state")
    print("=" * 60)
    response = skill.chat("What is the current physics state?")
    print(f"\nAssistant: {response.text}")
    print(f"\nTools called: {len(response.tool_calls)}")
    print(f"Current z: {response.state['z']:.4f}")
    print(f"Phase: {response.state['phase']}")

    # Example 2: Ask about a specific z value
    print("\n" + "=" * 60)
    print("Example 2: Asking about z = 0.7")
    print("=" * 60)
    response = skill.chat("What phase is z = 0.7 in, and what is its negentropy?")
    print(f"\nAssistant: {response.text}")

    # Example 3: Drive toward THE LENS
    print("\n" + "=" * 60)
    print("Example 3: Driving toward THE LENS")
    print("=" * 60)
    response = skill.chat("Drive the system toward THE LENS over 50 steps and report the results.")
    print(f"\nAssistant: {response.text}")
    print(f"\nFinal z: {response.state['z']:.4f}")
    print(f"Final phase: {response.state['phase']}")

    # Example 4: Check K-formation
    print("\n" + "=" * 60)
    print("Example 4: Checking K-formation")
    print("=" * 60)
    response = skill.chat(
        "Check if K-formation would be achieved with kappa=0.95, eta=0.7, and R=8. "
        "Explain what each criterion means."
    )
    print(f"\nAssistant: {response.text}")

    # Example 5: Apply operators
    print("\n" + "=" * 60)
    print("Example 5: Applying APL operators")
    print("=" * 60)
    response = skill.chat(
        "Apply the amplify (^) operator twice, then show me the new state."
    )
    print(f"\nAssistant: {response.text}")

    # Example 6: Explain physics
    print("\n" + "=" * 60)
    print("Example 6: Physics explanation")
    print("=" * 60)
    response = skill.chat(
        "Why is z_c = sqrt(3)/2 significant? What observable physics phenomena "
        "demonstrate this value?"
    )
    print(f"\nAssistant: {response.text}")

    # Print usage summary
    print("\n" + "=" * 60)
    print("Session Summary")
    print("=" * 60)
    final_state = skill.get_state()
    print(f"Final z: {final_state['z']:.6f}")
    print(f"Final phase: {final_state['phase']}")
    print(f"Final tier: {final_state['tier']} ({final_state['tier_name']})")
    print(f"K-formation met: {final_state['k_formation_met']}")
    print(f"Total steps: {final_state['step']}")


if __name__ == "__main__":
    main()
