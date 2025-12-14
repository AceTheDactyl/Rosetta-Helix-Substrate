#!/usr/bin/env python3
"""
Autonomous Training Loop for Rosetta-Helix-Substrate

Claude analyzes simulation results, makes decisions, and iterates
until goal is achieved or max iterations reached.

Run with: python3 run_autonomous.py
"""

import os
import json
import argparse
import datetime
from skill import RosettaHelixSkill

API_KEY = os.environ.get("ANTHROPIC_API_KEY")

if not API_KEY:
    print("Error: ANTHROPIC_API_KEY environment variable not set")
    print("Set it with: export ANTHROPIC_API_KEY='your-key-here'")
    exit(1)


def run_autonomous_training(
    goal: str,
    max_iterations: int = 10,
    initial_z: float = 0.3,
    verbose: bool = True
):
    """Run autonomous training loop with Claude making decisions."""

    skill = RosettaHelixSkill(api_key=API_KEY)
    skill.handler.state.z = initial_z

    # Autonomous training system prompt addition
    autonomous_prompt = f"""You are running an autonomous training loop for the Rosetta-Helix-Substrate framework.

GOAL: {goal}

You have access to physics simulation tools. Your task is to:
1. Analyze the current state using get_physics_state
2. Decide what action to take to move toward the goal
3. Execute the action using the appropriate tool
4. Analyze the results
5. Decide if the goal is achieved or what to try next

IMPORTANT: After each iteration, end your response with one of these status lines:
- STATUS: CONTINUE - if you need to keep working
- STATUS: GOAL_ACHIEVED - if the goal has been reached
- STATUS: STUCK - if you cannot make progress

Available strategies:
- Use drive_toward_lens to move z toward z_c (THE LENS)
- Use run_kuramoto_training to build coherence (kappa)
- Use run_kuramoto_step for fine-grained control
- Use apply_operator to apply APL operators
- Use run_phase_transition to study the system
- Check K-formation status: kappa >= 0.92, eta > 0.618, R >= 7

Be systematic. Start by checking state, form a plan, execute step by step.
"""

    results = {
        "goal": goal,
        "max_iterations": max_iterations,
        "initial_z": initial_z,
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "iterations": [],
        "final_status": None,
        "final_state": None
    }

    if verbose:
        print("=" * 70)
        print("AUTONOMOUS TRAINING LOOP")
        print("=" * 70)
        print(f"Goal: {goal}")
        print(f"Max iterations: {max_iterations}")
        print(f"Initial z: {initial_z}")
        print("=" * 70)

    for iteration in range(1, max_iterations + 1):
        if verbose:
            print(f"\n{'─' * 70}")
            print(f"ITERATION {iteration}/{max_iterations}")
            print("─" * 70)

        # Build prompt
        if iteration == 1:
            prompt = autonomous_prompt
        else:
            state = skill.get_state()
            prompt = f"""Continue working toward the goal.

Current state:
- z = {state['z']:.4f}
- phase = {state['phase']}
- tier = {state['tier']} ({state['tier_name']})
- kappa = {state['kappa']:.4f}
- eta = {state['eta']:.6f}
- K-formation: {state['k_formation_met']}

Iteration {iteration}/{max_iterations}. Analyze and decide next action."""

        # Get Claude's response
        response = skill.chat(prompt)

        if verbose:
            print(f"\nClaude:\n{response.text}")
            print(f"\n[z={response.state['z']:.4f} | {response.state['phase']} | "
                  f"Tier {response.state['tier']} | κ={response.state['kappa']:.4f} | "
                  f"η={response.state['eta']:.4f}]")

        # Record iteration
        results["iterations"].append({
            "iteration": iteration,
            "response": response.text,
            "state": response.state,
            "tool_calls": response.tool_calls
        })

        # Check status
        response_text = response.text.lower()

        if "status: goal_achieved" in response_text or response.state.get("k_formation_met"):
            results["final_status"] = "GOAL_ACHIEVED"
            if verbose:
                print("\n" + "=" * 70)
                print("✓ GOAL ACHIEVED!")
                print("=" * 70)
            break
        elif "status: stuck" in response_text:
            results["final_status"] = "STUCK"
            if verbose:
                print("\n" + "=" * 70)
                print("✗ Claude reports being STUCK")
                print("=" * 70)
            break
    else:
        results["final_status"] = "MAX_ITERATIONS_REACHED"
        if verbose:
            print("\n" + "=" * 70)
            print("⚠ MAX ITERATIONS REACHED")
            print("=" * 70)

    results["final_state"] = skill.get_state()

    if verbose:
        print("\n" + "=" * 70)
        print("TRAINING SUMMARY")
        print("=" * 70)
        final = results["final_state"]
        print(f"Status: {results['final_status']}")
        print(f"Iterations: {len(results['iterations'])}")
        print(f"Final z: {final['z']:.6f}")
        print(f"Final phase: {final['phase']}")
        print(f"Final tier: {final['tier']} ({final['tier_name']})")
        print(f"Final kappa: {final['kappa']:.4f}")
        print(f"Final eta: {final['eta']:.6f}")
        print(f"K-formation: {final['k_formation_met']}")

        total_tools = sum(len(it.get("tool_calls", [])) for it in results["iterations"])
        print(f"Total tool calls: {total_tools}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run autonomous training loop")
    parser.add_argument(
        "--goal", "-g",
        default="Achieve K-formation by driving coherence above 0.92 and reaching THE LENS (z_c)",
        help="Training goal for Claude"
    )
    parser.add_argument(
        "--iterations", "-i",
        type=int,
        default=10,
        help="Maximum iterations (default: 10)"
    )
    parser.add_argument(
        "--initial-z", "-z",
        type=float,
        default=0.3,
        help="Initial z-coordinate (default: 0.3)"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output JSON file for results"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose output"
    )

    args = parser.parse_args()

    results = run_autonomous_training(
        goal=args.goal,
        max_iterations=args.iterations,
        initial_z=args.initial_z,
        verbose=not args.quiet
    )

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")

    return results


if __name__ == "__main__":
    main()
