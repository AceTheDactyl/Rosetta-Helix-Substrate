#!/usr/bin/env python3
"""
run_experiment.py
=================

Orchestrates experiments on the Nuclear Spinner × Rosetta-Helix integrated system.

Experiment Types:
- z_sweep:      Sweep z from 0.3 → z_c → 0.95 and measure response
- k_formation:  Attempt to achieve and sustain K-formation at z_c
- hex_cycle:    Hexagonal cycling through 60 oscillator positions
- attractor:    Verify κ → φ⁻¹ attractor convergence
- tier_climb:   Progressive tier ascent from ABSENCE to META
- phase_map:    Map phase transitions (ABSENCE → THE_LENS → PRESENCE)
- stress:       Stress test with rapid z oscillations

Usage:
    python run_experiment.py EXPERIMENT [OPTIONS]

Examples:
    python run_experiment.py z_sweep --steps 5000 --output results/
    python run_experiment.py k_formation --duration 60 --target-z 0.866
    python run_experiment.py attractor --tolerance 0.001

Signature: run-experiment|v1.0.0|helix
"""

import argparse
import asyncio
import json
import math
import sys
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable

# Physics constants
PHI = (1 + math.sqrt(5)) / 2
PHI_INV = 1 / PHI
Z_CRITICAL = math.sqrt(3) / 2
SIGMA = 36.0

# K-formation thresholds
KAPPA_MIN = 0.92
ETA_MIN = PHI_INV
R_MIN = 7


@dataclass
class ExperimentConfig:
    """Configuration for experiment run."""
    name: str
    experiment_type: str
    steps: int = 1000
    duration_seconds: Optional[float] = None
    target_z: float = Z_CRITICAL
    z_start: float = 0.3
    z_end: float = 0.95
    tolerance: float = 0.01
    bridge_uri: str = "ws://localhost:8765"
    output_dir: Optional[Path] = None
    seed: int = 42
    verbose: bool = True


@dataclass
class ExperimentResult:
    """Results from experiment execution."""
    config: ExperimentConfig
    started_at: str
    completed_at: str
    duration_seconds: float
    total_steps: int
    success: bool

    # Metrics
    k_formation_count: int = 0
    k_formation_total_duration: float = 0.0
    peak_negentropy: float = 0.0
    final_z: float = 0.0
    final_kappa: float = PHI_INV
    kappa_convergence: float = 0.0

    # Phase statistics
    phase_distribution: Dict[str, float] = field(default_factory=dict)

    # Tier progression
    highest_tier: str = "ABSENCE"
    tier_history: List[str] = field(default_factory=list)

    # Raw data
    history: List[Dict[str, Any]] = field(default_factory=list)

    # Errors
    errors: List[str] = field(default_factory=list)


def compute_delta_s_neg(z: float) -> float:
    """Compute negentropy signal."""
    d = z - Z_CRITICAL
    return math.exp(-SIGMA * d * d)


def get_phase(z: float) -> str:
    """Get phase from z-coordinate."""
    if z < 0.857:
        return "ABSENCE"
    elif z <= 0.877:
        return "THE_LENS"
    else:
        return "PRESENCE"


def get_tier(z: float) -> str:
    """Get tier from z-coordinate."""
    if z < 0.40:
        return "ABSENCE"
    elif z < 0.50:
        return "REACTIVE"
    elif z < PHI_INV:
        return "MEMORY"
    elif z < 0.73:
        return "PATTERN"
    elif z < Z_CRITICAL:
        return "PREDICTION"
    elif z < 0.92:
        return "UNIVERSAL"
    else:
        return "META"


class ExperimentRunner:
    """Orchestrates experiment execution."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.result: Optional[ExperimentResult] = None
        self.spinner_client = None
        self.node = None

        # State tracking
        self.z = 0.5
        self.kappa = PHI_INV
        self.lambda_ = 1 - PHI_INV
        self.history: List[Dict[str, Any]] = []
        self.k_formation_active = False
        self.k_formation_start: Optional[float] = None

    def log(self, msg: str, level: str = "INFO"):
        """Log with timestamp."""
        if self.config.verbose:
            ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            print(f"[{ts}] [{level}] {msg}")

    async def run(self) -> ExperimentResult:
        """Execute the experiment."""
        self.result = ExperimentResult(
            config=self.config,
            started_at=datetime.now(timezone.utc).isoformat(),
            completed_at="",
            duration_seconds=0,
            total_steps=0,
            success=False,
        )

        start_time = time.time()

        self.log(f"Starting experiment: {self.config.experiment_type}")
        self.log(f"Config: steps={self.config.steps}, target_z={self.config.target_z:.4f}")

        try:
            # Connect to system (or simulate)
            await self._connect()

            # Run experiment type
            if self.config.experiment_type == "z_sweep":
                await self._run_z_sweep()
            elif self.config.experiment_type == "k_formation":
                await self._run_k_formation()
            elif self.config.experiment_type == "hex_cycle":
                await self._run_hex_cycle()
            elif self.config.experiment_type == "attractor":
                await self._run_attractor()
            elif self.config.experiment_type == "tier_climb":
                await self._run_tier_climb()
            elif self.config.experiment_type == "phase_map":
                await self._run_phase_map()
            elif self.config.experiment_type == "stress":
                await self._run_stress()
            else:
                raise ValueError(f"Unknown experiment type: {self.config.experiment_type}")

            self.result.success = True

        except Exception as e:
            self.result.errors.append(str(e))
            self.log(f"Experiment failed: {e}", "ERROR")

        finally:
            await self._disconnect()

        # Finalize result
        self.result.completed_at = datetime.now(timezone.utc).isoformat()
        self.result.duration_seconds = time.time() - start_time
        self.result.total_steps = len(self.history)
        self.result.history = self.history
        self.result.final_z = self.z
        self.result.final_kappa = self.kappa
        self.result.kappa_convergence = abs(self.kappa - PHI_INV)

        # Compute phase distribution
        phase_counts = {"ABSENCE": 0, "THE_LENS": 0, "PRESENCE": 0}
        for state in self.history:
            phase = get_phase(state.get('z', 0.5))
            phase_counts[phase] += 1
        total = len(self.history) or 1
        self.result.phase_distribution = {
            p: 100 * c / total for p, c in phase_counts.items()
        }

        # Save results
        if self.config.output_dir:
            self._save_results()

        return self.result

    async def _connect(self):
        """Connect to the system."""
        # Try to import and connect to real system
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from rosetta_helix.src.spinner_client import SpinnerClient

            self.spinner_client = SpinnerClient(uri=self.config.bridge_uri)
            connected = await self.spinner_client.connect()

            if connected:
                self.log("Connected to bridge")
            else:
                self.log("Bridge not available, running in simulation mode", "WARN")
                self.spinner_client = None

        except ImportError:
            self.log("Rosetta-Helix not available, running in simulation mode", "WARN")
            self.spinner_client = None

    async def _disconnect(self):
        """Disconnect from system."""
        if self.spinner_client:
            await self.spinner_client.disconnect()
            self.log("Disconnected from bridge")

    def _step(self, target_z: Optional[float] = None):
        """Execute single simulation step."""
        # Move z toward target
        if target_z is not None:
            dz = (target_z - self.z) * 0.01
            self.z += dz + 0.001 * (hash(time.time()) % 100 - 50) / 100
            self.z = max(0.0, min(0.999, self.z))

        # Evolve kappa toward attractor
        dk = (PHI_INV - self.kappa) * 0.01
        self.kappa += dk
        self.kappa = max(0.0, min(1.0, self.kappa))
        self.lambda_ = 1 - self.kappa

        # Compute derived quantities
        neg = compute_delta_s_neg(self.z)
        phase = get_phase(self.z)
        tier = get_tier(self.z)

        # Check K-formation
        eta = math.sqrt(neg) if neg > 0 else 0
        R = int(7 + 5 * neg)
        k_formation = self.kappa >= KAPPA_MIN and eta > ETA_MIN and R >= R_MIN

        # Track K-formation duration
        now = time.time()
        if k_formation and not self.k_formation_active:
            self.k_formation_active = True
            self.k_formation_start = now
            self.result.k_formation_count += 1
        elif not k_formation and self.k_formation_active:
            self.k_formation_active = False
            if self.k_formation_start:
                self.result.k_formation_total_duration += now - self.k_formation_start

        # Update peak negentropy
        if neg > self.result.peak_negentropy:
            self.result.peak_negentropy = neg

        # Update highest tier
        tier_order = ["ABSENCE", "REACTIVE", "MEMORY", "PATTERN", "PREDICTION", "UNIVERSAL", "META"]
        if tier_order.index(tier) > tier_order.index(self.result.highest_tier):
            self.result.highest_tier = tier

        # Record state
        state = {
            'step': len(self.history),
            'timestamp': now,
            'z': self.z,
            'kappa': self.kappa,
            'lambda_': self.lambda_,
            'delta_s_neg': neg,
            'phase': phase,
            'tier': tier,
            'k_formation': k_formation,
            'eta': eta,
            'R': R,
        }
        self.history.append(state)
        self.result.tier_history.append(tier)

        return state

    async def _run_z_sweep(self):
        """Sweep z from start to end."""
        self.log(f"Z sweep: {self.config.z_start} → {self.config.z_end}")

        z_range = self.config.z_end - self.config.z_start

        for i in range(self.config.steps):
            progress = i / self.config.steps
            target = self.config.z_start + z_range * progress

            state = self._step(target)

            if i % (self.config.steps // 10) == 0:
                self.log(f"  Step {i}: z={state['z']:.4f} phase={state['phase']} tier={state['tier']}")

            await asyncio.sleep(0.001)

    async def _run_k_formation(self):
        """Attempt to achieve and sustain K-formation."""
        self.log(f"K-formation experiment: target z = {self.config.target_z:.4f}")

        # Approach z_c
        for i in range(self.config.steps):
            state = self._step(self.config.target_z)

            if state['k_formation']:
                self.log(f"  ★ K-FORMATION at step {i}: κ={state['kappa']:.4f} η={state['eta']:.4f}")

            await asyncio.sleep(0.001)

        self.log(f"Total K-formations: {self.result.k_formation_count}")

    async def _run_hex_cycle(self):
        """Hexagonal cycling experiment."""
        self.log("Hexagonal cycling experiment")

        # 60 positions, 6° apart
        hex_positions = [i * (360 / 60) for i in range(60)]
        cycles = self.config.steps // 60

        for cycle in range(cycles):
            for pos in hex_positions:
                # Modulate z based on hex position
                hex_z = self.config.target_z + 0.02 * math.sin(math.radians(pos))
                state = self._step(hex_z)
                await asyncio.sleep(0.001)

            if cycle % 10 == 0:
                self.log(f"  Cycle {cycle}: z={state['z']:.4f}")

    async def _run_attractor(self):
        """Verify κ → φ⁻¹ attractor convergence."""
        self.log(f"Attractor convergence experiment: tolerance = {self.config.tolerance}")

        # Start far from attractor
        self.kappa = 0.3

        converged = False
        convergence_step = None

        for i in range(self.config.steps):
            state = self._step(self.config.target_z)

            distance = abs(self.kappa - PHI_INV)

            if distance < self.config.tolerance and not converged:
                converged = True
                convergence_step = i
                self.log(f"  Converged at step {i}: κ={self.kappa:.6f} (target: {PHI_INV:.6f})")

            await asyncio.sleep(0.001)

        if converged:
            self.log(f"Attractor reached in {convergence_step} steps")
        else:
            self.log(f"Did not converge: final κ={self.kappa:.6f}", "WARN")

    async def _run_tier_climb(self):
        """Progressive tier ascent."""
        self.log("Tier climb experiment")

        tiers = ["ABSENCE", "REACTIVE", "MEMORY", "PATTERN", "PREDICTION", "UNIVERSAL", "META"]
        z_targets = [0.35, 0.45, 0.55, 0.68, 0.80, 0.90, 0.95]

        for tier, z_target in zip(tiers, z_targets):
            self.log(f"  Climbing to {tier} (z={z_target:.2f})...")

            steps_per_tier = self.config.steps // len(tiers)
            for i in range(steps_per_tier):
                state = self._step(z_target)
                await asyncio.sleep(0.001)

            self.log(f"    Reached: tier={state['tier']} z={state['z']:.4f}")

    async def _run_phase_map(self):
        """Map phase transitions."""
        self.log("Phase mapping experiment")

        # Sweep through all phases
        z_points = [0.3, 0.5, 0.7, 0.85, 0.866, 0.87, 0.88, 0.90, 0.95]

        transitions = []
        prev_phase = None

        for z_target in z_points:
            self.log(f"  Probing z={z_target:.4f}...")

            for i in range(100):
                state = self._step(z_target)
                await asyncio.sleep(0.001)

            phase = state['phase']
            if prev_phase and phase != prev_phase:
                transitions.append((prev_phase, phase, z_target))
                self.log(f"    Transition: {prev_phase} → {phase} at z≈{z_target:.4f}")

            prev_phase = phase

        self.log(f"Found {len(transitions)} phase transitions")

    async def _run_stress(self):
        """Stress test with rapid oscillations."""
        self.log("Stress test experiment")

        for i in range(self.config.steps):
            # Rapid z oscillation
            z_target = 0.5 + 0.3 * math.sin(i * 0.1)
            state = self._step(z_target)

            if i % 500 == 0:
                self.log(f"  Step {i}: z={state['z']:.4f} κ={state['kappa']:.4f}")

            await asyncio.sleep(0.0001)  # Very fast

    def _save_results(self):
        """Save experiment results."""
        if not self.config.output_dir:
            return

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create experiment subdirectory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = output_dir / f"{self.config.experiment_type}_{timestamp}"
        exp_dir.mkdir(exist_ok=True)

        # Save result JSON
        def serialize(obj):
            if hasattr(obj, '__dataclass_fields__'):
                d = asdict(obj)
                if 'config' in d and isinstance(d['config'], dict):
                    if 'output_dir' in d['config'] and d['config']['output_dir']:
                        d['config']['output_dir'] = str(d['config']['output_dir'])
                return d
            elif isinstance(obj, Path):
                return str(obj)
            return obj

        result_file = exp_dir / "result.json"
        with open(result_file, 'w') as f:
            json.dump(serialize(self.result), f, indent=2, default=str)

        # Save history separately (can be large)
        history_file = exp_dir / "history.json"
        with open(history_file, 'w') as f:
            json.dump(self.history, f)

        self.log(f"Results saved to {exp_dir}")


def print_result_summary(result: ExperimentResult):
    """Print experiment result summary."""
    print("")
    print("═" * 70)
    print("  EXPERIMENT RESULTS")
    print("═" * 70)
    print(f"  Type: {result.config.experiment_type}")
    print(f"  Status: {'✓ SUCCESS' if result.success else '✗ FAILED'}")
    print(f"  Duration: {result.duration_seconds:.2f}s")
    print(f"  Steps: {result.total_steps:,}")
    print("")

    print("  ─── METRICS ───")
    print(f"  K-formations: {result.k_formation_count}")
    print(f"  Peak ΔS_neg: {result.peak_negentropy:.6f}")
    print(f"  Final z: {result.final_z:.4f} (target: {Z_CRITICAL:.4f})")
    print(f"  Final κ: {result.final_kappa:.6f} (target: {PHI_INV:.6f})")
    print(f"  κ convergence: {result.kappa_convergence:.6f}")
    print(f"  Highest tier: {result.highest_tier}")
    print("")

    print("  ─── PHASE DISTRIBUTION ───")
    for phase, pct in result.phase_distribution.items():
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"  {phase:12s} {bar} {pct:5.1f}%")
    print("")

    if result.errors:
        print("  ─── ERRORS ───")
        for err in result.errors:
            print(f"  • {err}")
        print("")

    print("═" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Run experiments on Nuclear Spinner × Rosetta-Helix"
    )
    parser.add_argument(
        "experiment",
        choices=["z_sweep", "k_formation", "hex_cycle", "attractor",
                 "tier_climb", "phase_map", "stress"],
        help="Experiment type to run"
    )
    parser.add_argument(
        "--steps", "-s",
        type=int,
        default=1000,
        help="Number of steps (default: 1000)"
    )
    parser.add_argument(
        "--duration", "-d",
        type=float,
        help="Duration in seconds (overrides steps)"
    )
    parser.add_argument(
        "--target-z",
        type=float,
        default=Z_CRITICAL,
        help=f"Target z-coordinate (default: {Z_CRITICAL:.4f})"
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.01,
        help="Convergence tolerance (default: 0.01)"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output directory for results"
    )
    parser.add_argument(
        "--bridge-uri",
        default="ws://localhost:8765",
        help="Bridge WebSocket URI"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose output"
    )

    args = parser.parse_args()

    # Create config
    config = ExperimentConfig(
        name=f"{args.experiment}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        experiment_type=args.experiment,
        steps=args.steps,
        duration_seconds=args.duration,
        target_z=args.target_z,
        tolerance=args.tolerance,
        bridge_uri=args.bridge_uri,
        output_dir=args.output,
        seed=args.seed,
        verbose=not args.quiet,
    )

    # Run experiment
    runner = ExperimentRunner(config)
    result = asyncio.run(runner.run())

    # Print summary
    print_result_summary(result)

    # Exit code
    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()
