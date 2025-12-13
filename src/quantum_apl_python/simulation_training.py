#!/usr/bin/env python3
"""
Simulation Training Data Generator
===================================

Generates training data from physics simulations for ML models.

Capabilities:
1. Quasicrystal formation trajectories
2. Omega point convergence dynamics
3. Cross-referenced validation runs
4. Batch generation with configurable parameters

Output format: JSON with full trajectory metadata.

@version 1.0.0
@author Claude (Anthropic) - Rosetta-Helix-Substrate Contribution
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from .extended_physics_constants import (
    # Simulation functions
    simulate_quasicrystal_formation,
    simulate_omega_convergence,
    # Validation
    validate_extended_physics,
    cross_reference_constants,
    # Holographic interpretation
    holographic_z_interpretation,
    # Quasicrystal functions
    fibonacci_ratio,
    penrose_tile_counts,
    quasicrystal_negentropy,
    # Constants
    SIGMA_S3,
)
from .constants import Z_CRITICAL, PHI, PHI_INV


# ============================================================================
# DATACLASSES FOR TRAINING OUTPUT
# ============================================================================

@dataclass
class TrajectoryMetadata:
    """Metadata for a simulation trajectory."""
    simulation_type: str  # "quasicrystal" or "omega_point"
    n_steps: int
    seed: int
    initial_state: Dict[str, float]
    final_state: Dict[str, float]
    parameters: Dict[str, Any]
    timestamp: str


@dataclass
class TrainingBatch:
    """A batch of training trajectories."""
    batch_id: str
    n_trajectories: int
    simulation_type: str
    trajectories: List[Dict[str, Any]]
    validation: Dict[str, bool]
    constants: Dict[str, Any]
    metadata: Dict[str, Any]


# ============================================================================
# TRAJECTORY GENERATION
# ============================================================================

def generate_quasicrystal_trajectory(
    n_steps: int = 100,
    initial_order: float = 0.3,
    noise_scale: float = 0.01,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Generate a single quasicrystal formation trajectory.

    Parameters
    ----------
    n_steps : int
        Number of simulation steps
    initial_order : float
        Initial order parameter
    noise_scale : float
        Thermal noise scale
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    Dict[str, Any]
        Trajectory data with metadata
    """
    if seed is None:
        seed = random.randint(0, 2**31 - 1)

    trajectory = simulate_quasicrystal_formation(
        n_steps=n_steps,
        initial_order=initial_order,
        noise_scale=noise_scale,
        seed=seed
    )

    # Extract trajectory as list of dicts
    traj_data = [
        {
            "step": i,
            "order": state.order,
            "delta_s_neg": state.delta_s_neg,
            "phason_strain": state.phason_strain,
            "generation": state.generation
        }
        for i, state in enumerate(trajectory)
    ]

    # Compute derived metrics
    initial = trajectory[0]
    final = trajectory[-1]
    convergence = abs(final.order - PHI_INV) < abs(initial.order - PHI_INV)

    return {
        "metadata": {
            "simulation_type": "quasicrystal",
            "n_steps": n_steps,
            "seed": seed,
            "initial_order": initial_order,
            "noise_scale": noise_scale,
            "timestamp": datetime.utcnow().isoformat()
        },
        "trajectory": traj_data,
        "summary": {
            "initial_order": initial.order,
            "final_order": final.order,
            "initial_negentropy": initial.delta_s_neg,
            "final_negentropy": final.delta_s_neg,
            "target_phi_inv": PHI_INV,
            "converged_toward_target": convergence,
            "final_phason_strain": final.phason_strain
        }
    }


def generate_omega_trajectory(
    n_steps: int = 500,
    alpha: float = 0.01,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Generate a single omega point convergence trajectory.

    Parameters
    ----------
    n_steps : int
        Number of simulation steps
    alpha : float
        Convergence rate parameter
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    Dict[str, Any]
        Trajectory data with metadata
    """
    if seed is None:
        seed = random.randint(0, 2**31 - 1)

    trajectory = simulate_omega_convergence(
        n_steps=n_steps,
        alpha=alpha,
        seed=seed
    )

    # Extract trajectory as list of dicts
    traj_data = [
        {
            "step": i,
            "tau_ratio": state.tau_ratio,
            "processing_rate": state.processing_rate,
            "cumulative_info": state.cumulative_info,
            "complexity": state.complexity
        }
        for i, state in enumerate(trajectory)
    ]

    # Compute derived metrics
    initial = trajectory[0]
    final = trajectory[-1]
    peak_complexity_idx = max(range(len(trajectory)),
                              key=lambda i: trajectory[i].complexity)

    return {
        "metadata": {
            "simulation_type": "omega_point",
            "n_steps": n_steps,
            "seed": seed,
            "alpha": alpha,
            "timestamp": datetime.utcnow().isoformat()
        },
        "trajectory": traj_data,
        "summary": {
            "initial_tau_ratio": initial.tau_ratio,
            "final_tau_ratio": final.tau_ratio,
            "initial_processing_rate": initial.processing_rate,
            "final_processing_rate": final.processing_rate,
            "cumulative_info": final.cumulative_info,
            "peak_complexity_step": peak_complexity_idx,
            "peak_complexity_value": trajectory[peak_complexity_idx].complexity,
            "target_z_c": Z_CRITICAL
        }
    }


# ============================================================================
# BATCH GENERATION
# ============================================================================

def generate_training_batch(
    n_trajectories: int = 100,
    simulation_type: str = "quasicrystal",
    n_steps: int = 100,
    base_seed: Optional[int] = None,
    **kwargs
) -> TrainingBatch:
    """
    Generate a batch of training trajectories.

    Parameters
    ----------
    n_trajectories : int
        Number of trajectories to generate
    simulation_type : str
        "quasicrystal" or "omega_point"
    n_steps : int
        Steps per trajectory
    base_seed : int, optional
        Base seed for reproducibility
    **kwargs
        Additional parameters passed to simulation function

    Returns
    -------
    TrainingBatch
        Complete batch with validation and constants
    """
    if base_seed is None:
        base_seed = random.randint(0, 2**31 - 1)

    trajectories = []

    for i in range(n_trajectories):
        seed = base_seed + i

        if simulation_type == "quasicrystal":
            # Vary initial conditions
            initial_order = random.uniform(0.2, 0.7)
            noise_scale = kwargs.get("noise_scale", 0.01)
            traj = generate_quasicrystal_trajectory(
                n_steps=n_steps,
                initial_order=initial_order,
                noise_scale=noise_scale,
                seed=seed
            )
        elif simulation_type == "omega_point":
            alpha = kwargs.get("alpha", 0.01)
            traj = generate_omega_trajectory(
                n_steps=n_steps,
                alpha=alpha,
                seed=seed
            )
        else:
            raise ValueError(f"Unknown simulation type: {simulation_type}")

        trajectories.append(traj)

    # Run validation
    validation = validate_extended_physics()

    # Get cross-reference constants
    xref = cross_reference_constants()
    constants = {
        "z_c": Z_CRITICAL,
        "phi": PHI,
        "phi_inv": PHI_INV,
        "sigma": SIGMA_S3,
        "cross_reference": {k: {kk: float(vv) for kk, vv in v.items()}
                           for k, v in xref.items()}
    }

    batch_id = f"{simulation_type}_{base_seed}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

    return TrainingBatch(
        batch_id=batch_id,
        n_trajectories=n_trajectories,
        simulation_type=simulation_type,
        trajectories=trajectories,
        validation=validation,
        constants=constants,
        metadata={
            "generated_at": datetime.utcnow().isoformat(),
            "base_seed": base_seed,
            "n_steps_per_trajectory": n_steps,
            "parameters": kwargs
        }
    )


# ============================================================================
# SERIALIZATION
# ============================================================================

def save_trajectories(
    filename: str,
    batch: TrainingBatch,
    runs_dir: Optional[str] = None
) -> str:
    """
    Save training batch to JSON file.

    Parameters
    ----------
    filename : str
        Output filename
    batch : TrainingBatch
        Batch to save
    runs_dir : str, optional
        Directory for output (default: ./runs)

    Returns
    -------
    str
        Full path to saved file
    """
    if runs_dir is None:
        runs_dir = "./runs"

    Path(runs_dir).mkdir(parents=True, exist_ok=True)

    filepath = os.path.join(runs_dir, filename)

    # Convert to serializable dict
    data = {
        "batch_id": batch.batch_id,
        "n_trajectories": batch.n_trajectories,
        "simulation_type": batch.simulation_type,
        "trajectories": batch.trajectories,
        "validation": batch.validation,
        "constants": batch.constants,
        "metadata": batch.metadata
    }

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)

    return filepath


def load_trajectories(filepath: str) -> TrainingBatch:
    """
    Load training batch from JSON file.

    Parameters
    ----------
    filepath : str
        Path to JSON file

    Returns
    -------
    TrainingBatch
        Loaded batch
    """
    with open(filepath, "r") as f:
        data = json.load(f)

    return TrainingBatch(**data)


# ============================================================================
# FULL PIPELINE
# ============================================================================

def run_full_training_pipeline(
    output_dir: str = "./runs",
    n_quasicrystal: int = 100,
    n_omega: int = 100,
    qc_steps: int = 100,
    omega_steps: int = 500,
    seed: Optional[int] = None
) -> Dict[str, str]:
    """
    Run complete training data generation pipeline.

    Parameters
    ----------
    output_dir : str
        Output directory for training data
    n_quasicrystal : int
        Number of quasicrystal trajectories
    n_omega : int
        Number of omega point trajectories
    qc_steps : int
        Steps per quasicrystal trajectory
    omega_steps : int
        Steps per omega trajectory
    seed : int, optional
        Master seed for reproducibility

    Returns
    -------
    Dict[str, str]
        Paths to generated files
    """
    if seed is None:
        seed = random.randint(0, 2**31 - 1)

    outputs = {}

    # Generate quasicrystal training data
    print(f"Generating {n_quasicrystal} quasicrystal trajectories...")
    qc_batch = generate_training_batch(
        n_trajectories=n_quasicrystal,
        simulation_type="quasicrystal",
        n_steps=qc_steps,
        base_seed=seed
    )
    qc_path = save_trajectories("quasicrystal_training_data.json", qc_batch, output_dir)
    outputs["quasicrystal"] = qc_path
    print(f"  Saved to: {qc_path}")

    # Generate omega point training data
    print(f"Generating {n_omega} omega point trajectories...")
    omega_batch = generate_training_batch(
        n_trajectories=n_omega,
        simulation_type="omega_point",
        n_steps=omega_steps,
        base_seed=seed + 1000000
    )
    omega_path = save_trajectories("omega_point_training_data.json", omega_batch, output_dir)
    outputs["omega_point"] = omega_path
    print(f"  Saved to: {omega_path}")

    # Generate combined summary
    summary = {
        "generated_at": datetime.utcnow().isoformat(),
        "master_seed": seed,
        "validation": validate_extended_physics(),
        "files": outputs,
        "statistics": {
            "quasicrystal": {
                "n_trajectories": n_quasicrystal,
                "n_steps": qc_steps,
                "converged_count": sum(
                    1 for t in qc_batch.trajectories
                    if t["summary"]["converged_toward_target"]
                )
            },
            "omega_point": {
                "n_trajectories": n_omega,
                "n_steps": omega_steps
            }
        }
    }

    summary_path = os.path.join(output_dir, "training_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    outputs["summary"] = summary_path
    print(f"  Summary: {summary_path}")

    return outputs


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    "TrajectoryMetadata",
    "TrainingBatch",
    "generate_quasicrystal_trajectory",
    "generate_omega_trajectory",
    "generate_training_batch",
    "save_trajectories",
    "load_trajectories",
    "run_full_training_pipeline",
]


# ============================================================================
# CLI INTERFACE
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate physics simulation training data"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="./runs",
        help="Output directory (default: ./runs)"
    )
    parser.add_argument(
        "--n-quasicrystal", "-nq",
        type=int, default=100,
        help="Number of quasicrystal trajectories (default: 100)"
    )
    parser.add_argument(
        "--n-omega", "-no",
        type=int, default=100,
        help="Number of omega point trajectories (default: 100)"
    )
    parser.add_argument(
        "--qc-steps",
        type=int, default=100,
        help="Steps per quasicrystal trajectory (default: 100)"
    )
    parser.add_argument(
        "--omega-steps",
        type=int, default=500,
        help="Steps per omega trajectory (default: 500)"
    )
    parser.add_argument(
        "--seed", "-s",
        type=int, default=None,
        help="Master seed for reproducibility"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("PHYSICS SIMULATION TRAINING DATA GENERATOR")
    print("=" * 60)

    outputs = run_full_training_pipeline(
        output_dir=args.output_dir,
        n_quasicrystal=args.n_quasicrystal,
        n_omega=args.n_omega,
        qc_steps=args.qc_steps,
        omega_steps=args.omega_steps,
        seed=args.seed
    )

    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    for name, path in outputs.items():
        print(f"  {name}: {path}")
