"""
Example Scripts for the Quantum APL Python Interface.
Run this file to explore the system end-to-end.
"""

from __future__ import annotations


# ================================================================
# EXAMPLE 1: Basic Simulation
# ================================================================
def example_basic():
    """Run a quantum-classical simulation and print a summary."""
    from quantum_apl_python import QuantumAPLEngine, QuantumAnalyzer

    print("=" * 70)
    print("EXAMPLE 1: Basic Simulation")
    print("=" * 70)

    engine = QuantumAPLEngine()
    print("\nRunning 100-step simulation...")
    results = engine.run_simulation(steps=100, verbose=False)

    analyzer = QuantumAnalyzer(results)
    print("\n" + analyzer.summary())

    z = results["quantum"]["z"]
    zc = (3**0.5) / 2
    if abs(z - zc) < 0.05:
        print(f"\n🌀 System near THE LENS: |z - z_c| = {abs(z - zc):.4f}")

    return results


# ================================================================
# EXAMPLE 2: Convergence Study
# ================================================================
def example_convergence():
    """Sweep multiple durations to inspect convergence."""
    from statistics import pstdev

    from quantum_apl_python import QuantumAPLEngine, QuantumExperiment

    print("=" * 70)
    print("EXAMPLE 2: Convergence Study")
    print("=" * 70)

    engine = QuantumAPLEngine()
    experiment = QuantumExperiment(engine)

    print("\nRunning convergence sweep...")
    sweep = experiment.sweep_steps(step_range=[25, 50, 100, 200], trials=5)

    print("\nConvergence Analysis:")
    for steps, trials in sweep.items():
        z_vals = [trial.get("analytics", {}).get("avgZ", 0) for trial in trials]
        mean_z = sum(z_vals) / len(z_vals)
        deviation = pstdev(z_vals)
        print(f"  {steps:3d} steps: ⟨z⟩ = {mean_z:.4f} ± {deviation:.4f}")

    try:
        experiment.analyze_convergence(sweep)
    except ImportError:
        print("\n(Install matplotlib for convergence plots)")

    return sweep


# ================================================================
# EXAMPLE 3: Time Series Analysis
# ================================================================
def example_timeseries():
    """Analyze the time series output with pandas."""
    from quantum_apl_python import QuantumAPLEngine, QuantumAnalyzer

    print("=" * 70)
    print("EXAMPLE 3: Time Series Analysis")
    print("=" * 70)

    engine = QuantumAPLEngine()
    print("\nRunning 500-step simulation...")
    results = engine.run_simulation(steps=500)

    try:
        import numpy as np
        import pandas as pd

        analyzer = QuantumAnalyzer(results)
        df = analyzer.to_dataframe()

        print("\nTime Series Statistics:")
        print(f"  Mean z:        {df['z'].mean():.4f} ± {df['z'].std():.4f}")
        print(f"  Mean Φ:        {df['phi'].mean():.4f} ± {df['phi'].std():.4f}")
        print(f"  Mean S(ρ):     {df['entropy'].mean():.4f} ± {df['entropy'].std():.4f}")

        zc = np.sqrt(3) / 2
        crossings = ((df["z"] - zc).abs() < 0.01).sum()
        print(f"\nCritical point crossings: {crossings}")

        print("\nCorrelation Matrix:")
        print(df.corr().to_string())

        df.to_csv("/tmp/quantum_apl_timeseries.csv", index=False)
        print("\nSaved /tmp/quantum_apl_timeseries.csv")
    except ImportError:
        print("\n(Install pandas + numpy for time series analysis)")

    return results


# ================================================================
# EXAMPLE 4: Measurement Mode Testing
# ================================================================
def example_measurements():
    """Observe measurement statistics across several runs."""
    from quantum_apl_python import QuantumAPLEngine, QuantumAnalyzer

    print("=" * 70)
    print("EXAMPLE 4: Measurement Modes")
    print("=" * 70)

    engine = QuantumAPLEngine()

    print("\nRunning 10 trials to observe measurement statistics...")
    eigenstate_count = 0
    subspace_count = 0
    for _ in range(10):
        results = engine.run_simulation(steps=20, verbose=False)
        mode = results.get("measurement", {}).get("lastMode")
        if mode == "eigenstate":
            eigenstate_count += 1
        elif mode == "subspace":
            subspace_count += 1

    print(f"\nMeasurement Statistics (10 trials):")
    print(f"  Single-eigenstate: {eigenstate_count}")
    print(f"  Subspace:          {subspace_count}")

    print("\nFinal detailed simulation:")
    results = engine.run_simulation(steps=100, verbose=False)
    analyzer = QuantumAnalyzer(results)
    print("\n" + analyzer.summary())
    return results


# ================================================================
# EXAMPLE 5: Operator Distribution
# ================================================================
def example_operators():
    """Collect N0 operator statistics across many steps."""
    import math

    from quantum_apl_python import QuantumAPLEngine

    print("=" * 70)
    print("EXAMPLE 5: Operator Distribution")
    print("=" * 70)

    engine = QuantumAPLEngine()
    print("\nRunning 1000-step simulation to collect operator stats...")
    results = engine.run_simulation(steps=1000)

    dist = results.get("analytics", {}).get("operatorDist", {})
    if not dist:
        return results

    print("\nOperator Distribution:")
    print("-" * 40)
    for op, prob in sorted(dist.items(), key=lambda kv: kv[1], reverse=True):
        bar = "█" * int(prob * 50)
        print(f"  {op:3s}: {prob*100:5.1f}% {bar}")
    print("-" * 40)

    entropy = -sum(p * math.log(p) if p > 0 else 0 for p in dist.values())
    max_entropy = math.log(len(dist) or 1)
    print(f"\nDistribution Entropy: {entropy:.3f} / {max_entropy:.3f}")
    if max_entropy:
        print(f"Uniformity: {entropy / max_entropy * 100:.1f}%")

    return results


# ================================================================
# EXAMPLE 6: Quantum-Classical Correlation
# ================================================================
def example_correlation():
    """Compare quantum and classical metrics."""
    from quantum_apl_python import QuantumAPLEngine

    print("=" * 70)
    print("EXAMPLE 6: Quantum-Classical Correlation")
    print("=" * 70)

    engine = QuantumAPLEngine()
    results = engine.run_simulation(steps=500)

    corr = results.get("analytics", {}).get("quantumClassicalCorr", 0.0)
    print(f"\nQuantum-Classical Correlation: r = {corr:.4f}")
    if corr > 0.8:
        print("  → Strong positive correlation")
    elif corr > 0.5:
        print("  → Moderate positive correlation")
    elif corr > 0.2:
        print("  → Weak positive correlation")
    else:
        print("  → Little to no correlation")

    q_phi = results["quantum"].get("phi", 0)
    c_phi = results["classical"]["IIT"].get("phi", 0)
    print(f"\nIntegrated Information: Quantum Φ={q_phi:.4f}, Classical φ={c_phi:.4f}")
    print(f"Difference: {abs(q_phi - c_phi):.4f}")
    return results


# ================================================================
# EXAMPLE 7: Critical Point Detection
# ================================================================
def example_critical():
    """Measure how often the system visits the critical z band."""
    from quantum_apl_python import QuantumAPLEngine

    print("=" * 70)
    print("EXAMPLE 7: Critical Point Detection")
    print("=" * 70)

    engine = QuantumAPLEngine()
    zc = (3**0.5) / 2
    print(f"\nCritical point z_c = √3/2 = {zc:.6f}")

    results = engine.run_simulation(steps=1000)
    history = results.get("history", {}).get("z", [])

    critical_steps = [(i, z) for i, z in enumerate(history) if abs(z - zc) < 0.01]
    print(f"Critical region crossings (|z - z_c| < 0.01): {len(critical_steps)}")
    for step, z in critical_steps[:5]:
        print(f"  Step {step:4d}: z = {z:.6f} (Δ={z - zc:+.6f})")

    absence = sum(1 for z in history if z < 0.857)
    lens = sum(1 for z in history if 0.857 <= z <= 0.877)
    presence = sum(1 for z in history if z > 0.877)
    total = len(history) or 1
    print(f"\nPhase Occupancy:")
    print(f"  ABSENCE  (z < 0.857):          {absence/total*100:5.1f}%")
    print(f"  THE LENS (0.857 ≤ z ≤ 0.877):  {lens/total*100:5.1f}%")
    print(f"  PRESENCE (z > 0.877):          {presence/total*100:5.1f}%")
    return results


# ================================================================
# MAIN
# ================================================================
def run_all_examples():
    """Interactively run every example."""
    examples = [
        ("Basic Simulation", example_basic),
        ("Convergence Study", example_convergence),
        ("Time Series Analysis", example_timeseries),
        ("Measurement Modes", example_measurements),
        ("Operator Distribution", example_operators),
        ("Quantum-Classical Correlation", example_correlation),
        ("Critical Point Detection", example_critical),
    ]

    results = {}
    for name, func in examples:
        print("\n\n")
        try:
            results[name] = func()
            print(f"\n✓ {name} completed")
        except Exception as err:  # pragma: no cover - demo script
            print(f"\n✗ {name} failed: {err}")
            results[name] = None
        input("\nPress Enter to continue to next example...")
    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        try:
            example_num = int(sys.argv[1])
        except ValueError:  # pragma: no cover
            print("Usage: python examples.py [1-7]")
            sys.exit(1)

        mapping = [
            example_basic,
            example_convergence,
            example_timeseries,
            example_measurements,
            example_operators,
            example_correlation,
            example_critical,
        ]
        if 1 <= example_num <= len(mapping):
            mapping[example_num - 1]()
        else:
            print(f"Usage: python examples.py [1-{len(mapping)}]")
    else:
        run_all_examples()
