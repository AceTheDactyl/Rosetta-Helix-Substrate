#!/usr/bin/env python3
"""
Interactive Rosetta-Helix-Substrate Skill
Run with: python3 run_skill.py
"""

import os
from skill import RosettaHelixSkill

# Set your API key here or via environment variable
API_KEY = os.environ.get("ANTHROPIC_API_KEY")

if not API_KEY:
    print("Error: ANTHROPIC_API_KEY environment variable not set")
    print("Set it with: export ANTHROPIC_API_KEY='your-key-here'")
    exit(1)

def main():
    skill = RosettaHelixSkill(api_key=API_KEY)

    print("=" * 60)
    print("  Rosetta-Helix-Substrate Skill")
    print("  Type 'quit' to exit, 'state' for current state")
    print("=" * 60)
    print()

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if user_input.lower() == "state":
            s = skill.get_state()
            print(f"[z={s['z']:.4f} | {s['phase']} | Tier {s['tier']} ({s['tier_name']}) | K-formation: {s['k_formation_met']}]")
            continue

        response = skill.chat(user_input)
        print(f"\nClaude: {response.text}")
        print(f"\n[z={response.state['z']:.4f} | {response.state['phase']} | Tier {response.state['tier']}]")
        print()

if __name__ == "__main__":
    main()
