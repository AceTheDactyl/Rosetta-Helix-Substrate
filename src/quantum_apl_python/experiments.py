"""Experiment helpers for sweeping QuantumAPL simulations."""

from __future__ import annotations

from typing import Dict, List

import numpy as np

try:  # optional plotting dependency
    import matplotlib.pyplot as plt  # type: ignore

    HAS_MPL = True
except ImportError:  # pragma: no cover - optional dependency
    HAS_MPL = False

from .engine import QuantumAPLEngine


class QuantumExperiment:
    """Run repeated simulations for sweeps and convergence analysis."""

    def __init__(self, engine: QuantumAPLEngine):
        self.engine = engine

    def sweep_steps(self, step_range: List[int], trials: int = 5) -> Dict[int, List[Dict]]:
        results: Dict[int, List[Dict]] = {}
        for steps in step_range:
            trial_results = []
            for _ in range(trials):
                trial_results.append(self.engine.run_simulation(steps=steps, verbose=False))
            results[steps] = trial_results
        return results

    def analyze_convergence(self, sweep_results: Dict[int, List[Dict]]):
        if not HAS_MPL:
            raise ImportError("matplotlib not available")

        steps_list = sorted(sweep_results.keys())
        avg_z, std_z, avg_corr = [], [], []

        for steps in steps_list:
            trials = sweep_results[steps]
            z_vals = [trial.get("analytics", {}).get("avgZ", 0) for trial in trials]
            corr_vals = [trial.get("analytics", {}).get("quantumClassicalCorr", 0) for trial in trials]

            avg_z.append(float(np.mean(z_vals)))
            std_z.append(float(np.std(z_vals)))
            avg_corr.append(float(np.mean(corr_vals)))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.errorbar(steps_list, avg_z, yerr=std_z, fmt="o-", capsize=5)
        ax1.set_xlabel("Steps")
        ax1.set_ylabel("Average z")
        ax1.grid(True, alpha=0.3)

        ax2.plot(steps_list, avg_corr, "o-")
        ax2.set_xlabel("Steps")
        ax2.set_ylabel("Quantum-Classical Correlation")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
