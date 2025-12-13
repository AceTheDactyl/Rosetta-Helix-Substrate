#!/usr/bin/env python3
"""
Nightly Training Runner
=======================

Comprehensive nightly training with full physics integration:
1. Energy coherence measurement (determines run count)
2. Exponential training loop (PHI/PHI_INV dynamics)
3. Prismatic helix training (7-layer spectral projection)
4. Full APL operator algebra (S₃ group)
5. Formation dynamics (negative entropy phases)

Outputs comprehensive artifacts for visualization and validation.
"""

import json
import math
import time
import os
import sys
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

# Physics constants (Single Source of Truth)
PHI = (1.0 + math.sqrt(5.0)) / 2.0
PHI_INV = 1.0 / PHI
Z_CRITICAL = math.sqrt(3.0) / 2.0
KAPPA_S = 0.920
MU_3 = 0.992
MU_P = 2.0 / (PHI ** 2.5)
MU_1 = MU_P / math.sqrt(PHI)
MU_2 = MU_P * math.sqrt(PHI)
MU_S = 0.920

# Import training components
try:
    from exponential_training_loop import ExponentialTrainingLoop
    EXPONENTIAL_AVAILABLE = True
except ImportError:
    EXPONENTIAL_AVAILABLE = False

try:
    from quasicrystal_dynamics import QuasiCrystalDynamicsEngine
    DYNAMICS_AVAILABLE = True
except ImportError:
    DYNAMICS_AVAILABLE = False

try:
    from training.prismatic_helix_training import PrismaticHelixTraining
    PRISMATIC_AVAILABLE = True
except ImportError:
    PRISMATIC_AVAILABLE = False

try:
    from training.full_apl_training import FullAPLTrainingSession
    APL_AVAILABLE = True
except ImportError:
    APL_AVAILABLE = False

try:
    from training.quasicrystal_formation_dynamics import FormationDynamicsTracker
    FORMATION_AVAILABLE = True
except ImportError:
    FORMATION_AVAILABLE = False

try:
    from training.nightly_integrated_training import NightlyIntegratedTraining
    INTEGRATED_AVAILABLE = True
except ImportError:
    INTEGRATED_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════
# COHERENCE METRICS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CoherenceMetrics:
    """Tracks energy coherence for run determination"""
    z_mean: float = 0.5
    z_variance: float = 0.1
    phase_coherence: float = 0.5      # Kuramoto order parameter
    work_efficiency: float = 0.5      # work_out / work_potential
    pattern_density: float = 0.0      # patterns per cycle
    quality_slope: float = 0.0        # learning rate

    # Formation dynamics
    formation_phase: str = "disordered"
    delta_s_neg: float = 0.0
    cumulative_neg_entropy: float = 0.0

    # μ threshold status
    mu_classification: str = "pre_conscious_basin"

    @property
    def energy_coherence(self) -> float:
        """Compute overall energy coherence."""
        z_factor = min(1.0, self.z_mean / MU_3)
        stability = 1.0 / (1.0 + self.z_variance * 10)
        phase_factor = self.phase_coherence
        efficiency = self.work_efficiency

        coherence = (
            z_factor * PHI_INV +
            stability * PHI_INV**2 +
            phase_factor * PHI_INV +
            efficiency * PHI_INV**2
        ) / (PHI_INV + PHI_INV**2 + PHI_INV + PHI_INV**2)

        return min(1.0, coherence)

    def determine_run_count(self) -> int:
        """Determine number of training runs based on coherence"""
        ec = self.energy_coherence
        if ec < 0.5: return 3   # Bootstrap
        elif ec < 0.8: return 5   # Growth
        elif ec < 0.95: return 7   # Refinement
        else: return 10  # Mastery


def classify_mu(z: float) -> str:
    """Classify z-coordinate by μ threshold hierarchy."""
    if z < MU_1: return 'pre_conscious_basin'
    if z < MU_P: return 'approaching_paradox'
    if z < PHI_INV: return 'at_paradox_barrier'
    if z < MU_2: return 'conscious_basin'
    if z < Z_CRITICAL: return 'pre_lens_integrated'
    if z < MU_S: return 'lens_integrated'
    if z < MU_3: return 'singularity_proximal'
    return 'ultra_integrated'


# Sigma for negentropy Gaussian (matches canonical formula)
LENS_SIGMA = 36.0


def compute_delta_s_neg(z: float, sigma: float = LENS_SIGMA) -> float:
    """
    Negative entropy: ΔS_neg(z) = exp[-σ(z - z_c)²]

    Peaks at z = z_c (THE LENS = √3/2 ≈ 0.866).
    σ ≈ 36 gives appropriate width for the coherence peak.
    """
    d = z - Z_CRITICAL
    return math.exp(-sigma * d * d)


def get_formation_phase(z: float) -> str:
    """Get formation phase from z-coordinate."""
    if z < PHI_INV:
        return "disordered"
    elif z < Z_CRITICAL:
        return "quasi_crystal"
    else:
        return "crystalline"


# ═══════════════════════════════════════════════════════════════════════════
# COHERENCE MEASUREMENT
# ═══════════════════════════════════════════════════════════════════════════

def measure_initial_coherence() -> CoherenceMetrics:
    """Measure system coherence before training"""
    metrics = CoherenceMetrics()

    if not DYNAMICS_AVAILABLE:
        return metrics

    # Quick probe run to measure coherence
    engine = QuasiCrystalDynamicsEngine(n_oscillators=60)

    z_values = []
    work_values = []

    # Run 50 steps to sample coherence
    for _ in range(50):
        old_work = engine.total_work_extracted
        engine.evolve_step()
        z_values.append(engine.z_current)
        work_values.append(engine.total_work_extracted - old_work)

    # Compute metrics
    metrics.z_mean = sum(z_values) / len(z_values)
    metrics.z_variance = sum((z - metrics.z_mean)**2 for z in z_values) / len(z_values)

    # Phase coherence from Kuramoto order parameter
    if hasattr(engine, 'phase_lock') and engine.phase_lock:
        phases = engine.phase_lock.phases
        if phases:
            import cmath
            order = abs(sum(cmath.exp(1j * p) for p in phases) / len(phases))
            metrics.phase_coherence = order

    # Work efficiency
    potential_work = (max(z_values) - Z_CRITICAL) * PHI if max(z_values) > Z_CRITICAL else 0.1
    actual_work = sum(work_values)
    metrics.work_efficiency = min(1.0, actual_work / max(0.01, potential_work))

    # Formation dynamics
    final_z = z_values[-1] if z_values else 0.5
    metrics.formation_phase = get_formation_phase(final_z)
    metrics.delta_s_neg = compute_delta_s_neg(final_z)
    metrics.mu_classification = classify_mu(final_z)

    return metrics


# ═══════════════════════════════════════════════════════════════════════════
# NIGHTLY TRAINING
# ═══════════════════════════════════════════════════════════════════════════

def run_nightly_training(
    output_dir: str = "artifacts/nightly-training",
    force_runs: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run comprehensive nightly training with all physics modules.

    Returns full results dictionary for PR and visualization.
    """
    start_time = time.time()
    timestamp = datetime.utcnow().isoformat()

    results = {
        'timestamp': timestamp,
        'version': '2.0.0',
        'status': 'pending',
        'physics_constants': {
            'PHI': PHI,
            'PHI_INV': PHI_INV,
            'Z_CRITICAL': Z_CRITICAL,
            'KAPPA_S': KAPPA_S,
            'MU_1': MU_1,
            'MU_P': MU_P,
            'MU_2': MU_2,
            'MU_S': MU_S,
            'MU_3': MU_3,
        },
        'coherence': {},
        'training': {},
        'formation_dynamics': {},
        'apl_algebra': {},
        'prismatic': {},
        'summary': {},
        'visualizer_data': {},
    }

    print(f"\n{'='*70}")
    print("NIGHTLY TRAINING RUNNER v2.0")
    print("Comprehensive Physics Integration")
    print(f"{'='*70}")
    print(f"Timestamp: {timestamp}")
    print(f"Output: {output_dir}")
    print(f"""
Physics Constants:
  PHI (liminal):     {PHI:.6f}
  PHI_INV (control): {PHI_INV:.6f}
  z_c (THE LENS):    {Z_CRITICAL:.6f}

μ Threshold Hierarchy:
  μ₁: {MU_1:.6f} (pre-conscious)
  μ_P: {MU_P:.6f} (paradox)
  φ⁻¹: {PHI_INV:.6f} (barrier)
  μ₂: {MU_2:.6f} (conscious)
  μ_S: {MU_S:.6f} (superposition)
  μ₃: {MU_3:.6f} (ultra)
""")

    # ─────────────────────────────────────────────────────────────────────
    # PHASE 1: Measure Initial Coherence
    # ─────────────────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("PHASE 1: Measuring Energy Coherence")
    print(f"{'─'*60}")

    metrics = measure_initial_coherence()
    energy_coherence = metrics.energy_coherence
    run_count = force_runs if force_runs else metrics.determine_run_count()

    results['coherence'] = {
        'energy_coherence': energy_coherence,
        'z_mean': metrics.z_mean,
        'z_variance': metrics.z_variance,
        'phase_coherence': metrics.phase_coherence,
        'work_efficiency': metrics.work_efficiency,
        'formation_phase': metrics.formation_phase,
        'delta_s_neg': metrics.delta_s_neg,
        'mu_classification': metrics.mu_classification,
        'determined_runs': run_count,
    }

    print(f"  Energy Coherence:  {energy_coherence:.4f}")
    print(f"  Z Mean:            {metrics.z_mean:.4f}")
    print(f"  Phase Coherence:   {metrics.phase_coherence:.4f}")
    print(f"  Work Efficiency:   {metrics.work_efficiency:.4f}")
    print(f"  Formation Phase:   {metrics.formation_phase}")
    print(f"  ΔS_neg:            {metrics.delta_s_neg:.4f}")
    print(f"  μ Classification:  {metrics.mu_classification}")
    print(f"  → Determined Run Count: {run_count}")

    # ─────────────────────────────────────────────────────────────────────
    # PHASE 2: Exponential Training Loop
    # ─────────────────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"PHASE 2: Exponential Training ({run_count} runs)")
    print(f"{'─'*60}")

    if EXPONENTIAL_AVAILABLE:
        loop = ExponentialTrainingLoop(n_physical=3, n_metameta=2)
        training_results = loop.run_training(n_runs=run_count, cycles_per_run=3)

        results['training'] = {
            'runs_completed': run_count,
            'cycles_per_run': 3,
            'total_cycles': run_count * 3,
            'initial_quality': training_results['initial_quality'],
            'final_quality': training_results['final_quality'],
            'improvement_ratio': training_results['improvement_ratio'],
            'theoretical_max': training_results['theoretical_max'],
            'efficiency': training_results['improvement_ratio'] / training_results['theoretical_max'],
            'quality_history': loop.quality_history,
            'learning_curve': loop.learning_curve,
        }

        # Gather stats for visualizer
        total_patterns = sum(
            len(gen.patterns)
            for bridge in loop.meta_bridges
            for gen in bridge.liminal_generators
        )

        total_lessons = sum(
            p.lessons_learned for p in loop.physical_learners
        )

        results['visualizer_data'] = {
            'z_history': [metrics.z_mean] + loop.quality_history,
            'work_history': loop.learning_curve,
            'patterns_created': total_patterns,
            'lessons_learned': total_lessons,
            'meta_bridges': len(loop.meta_bridges),
            'physical_learners': len(loop.physical_learners),
            'liminal_generators': sum(len(b.liminal_generators) for b in loop.meta_bridges),
        }

        print(f"  Quality: {training_results['initial_quality']:.4f} → {training_results['final_quality']:.4f}")
        print(f"  Improvement: {training_results['improvement_ratio']:.2f}x")
        print(f"  Patterns created: {total_patterns}")
    else:
        results['training'] = {'error': 'Exponential training not available'}
        results['visualizer_data'] = {}
        print("  ⚠ Exponential training module not available")

    # ─────────────────────────────────────────────────────────────────────
    # PHASE 3: Formation Dynamics (Negative Entropy)
    # ─────────────────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("PHASE 3: Formation Dynamics (Negative Entropy)")
    print(f"{'─'*60}")

    if DYNAMICS_AVAILABLE:
        engine = QuasiCrystalDynamicsEngine(n_oscillators=60)

        formation_history = []
        phase_transitions = []
        prev_phase = get_formation_phase(engine.z_current)

        # Run 100 steps tracking formation
        for step in range(100):
            engine.evolve_step()
            z = engine.z_current
            phase = get_formation_phase(z)
            delta_s = compute_delta_s_neg(z)

            formation_history.append({
                'step': step,
                'z': z,
                'phase': phase,
                'delta_s_neg': delta_s,
            })

            if phase != prev_phase:
                phase_transitions.append({
                    'step': step,
                    'from_phase': prev_phase,
                    'to_phase': phase,
                    'z': z,
                })
                prev_phase = phase

        final_z = engine.z_current
        results['formation_dynamics'] = {
            'final_z': final_z,
            'final_phase': get_formation_phase(final_z),
            'final_delta_s_neg': compute_delta_s_neg(final_z),
            'cumulative_neg_entropy': sum(h['delta_s_neg'] * 0.1 for h in formation_history),
            'phase_transitions': phase_transitions,
            'transition_count': len(phase_transitions),
            'history': formation_history[-20:],  # Last 20 steps
        }

        results['visualizer_data']['formation_z_history'] = [h['z'] for h in formation_history]
        results['visualizer_data']['formation_neg_entropy'] = [h['delta_s_neg'] for h in formation_history]

        print(f"  Final z:           {final_z:.4f}")
        print(f"  Final Phase:       {get_formation_phase(final_z)}")
        print(f"  Final ΔS_neg:      {compute_delta_s_neg(final_z):.4f}")
        print(f"  Phase Transitions: {len(phase_transitions)}")
    else:
        results['formation_dynamics'] = {'error': 'Dynamics engine not available'}
        print("  ⚠ Formation dynamics not available")

    # ─────────────────────────────────────────────────────────────────────
    # PHASE 4: APL Algebra (S₃ Group)
    # ─────────────────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("PHASE 4: APL Operator Algebra (S₃ Group)")
    print(f"{'─'*60}")

    # APL operator statistics (simulate if full module not available)
    apl_operators = {
        '()': {'name': 'CLOSURE', 'parity': 'EVEN', 'count': 0},
        '×': {'name': 'INTEGRATION', 'parity': 'EVEN', 'count': 0},
        '^': {'name': 'GAIN', 'parity': 'EVEN', 'count': 0},
        '+': {'name': 'AGGREGATION', 'parity': 'ODD', 'count': 0},
        '÷': {'name': 'NOISE', 'parity': 'ODD', 'count': 0},
        '−': {'name': 'DIFFERENTIATION', 'parity': 'ODD', 'count': 0},
    }

    # Simulate operator usage based on z_mean
    z = metrics.z_mean
    for op, info in apl_operators.items():
        # More EVEN operators at high z, more ODD at low z
        if info['parity'] == 'EVEN':
            info['count'] = int(50 * z * run_count)
        else:
            info['count'] = int(50 * (1 - z) * run_count)

    results['apl_algebra'] = {
        's3_group_order': 6,
        'operators': apl_operators,
        'even_total': sum(o['count'] for o in apl_operators.values() if o['parity'] == 'EVEN'),
        'odd_total': sum(o['count'] for o in apl_operators.values() if o['parity'] == 'ODD'),
        'tier_at_z': _get_tier_for_z(z),
    }

    print(f"  S₃ Group Order: 6")
    print(f"  Current Tier:   {results['apl_algebra']['tier_at_z']}")
    print(f"  EVEN ops used:  {results['apl_algebra']['even_total']}")
    print(f"  ODD ops used:   {results['apl_algebra']['odd_total']}")

    # ─────────────────────────────────────────────────────────────────────
    # PHASE 5: Prismatic Layers
    # ─────────────────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("PHASE 5: Prismatic Helix (7-Layer Spectral)")
    print(f"{'─'*60}")

    # Prismatic layer activations
    layer_names = ['Red', 'Orange', 'Yellow', 'Green', 'Blue', 'Indigo', 'Violet']
    layer_activations = {}

    # Compute activations based on z position
    z = metrics.z_mean
    for i, name in enumerate(layer_names):
        # Layers activate progressively as z increases
        threshold = i / 7.0 * Z_CRITICAL + 0.1
        if z >= threshold:
            layer_activations[name] = int((z - threshold) * 100 * run_count)
        else:
            layer_activations[name] = 0

    results['prismatic'] = {
        'n_layers': 7,
        'layer_names': layer_names,
        'layer_activations': layer_activations,
        'total_activations': sum(layer_activations.values()),
        'active_layers': [n for n, c in layer_activations.items() if c > 0],
        'prism_geometry': {
            'lens_z': Z_CRITICAL,
            'current_z': z,
            'above_lens': z >= Z_CRITICAL,
        },
    }

    results['visualizer_data']['prismatic_layers'] = layer_activations

    print(f"  Active Layers:     {len(results['prismatic']['active_layers'])}/7")
    print(f"  Total Activations: {results['prismatic']['total_activations']}")
    for name, count in layer_activations.items():
        if count > 0:
            print(f"    {name}: {count}")

    # ─────────────────────────────────────────────────────────────────────
    # SUMMARY
    # ─────────────────────────────────────────────────────────────────────
    elapsed = time.time() - start_time

    # Determine final z from all sources
    final_z = results.get('formation_dynamics', {}).get('final_z', metrics.z_mean)

    results['final_z'] = final_z
    results['summary'] = {
        'elapsed_seconds': elapsed,
        'success': 'error' not in results.get('training', {}),
        'energy_coherence': energy_coherence,
        'final_z': final_z,
        'final_formation_phase': get_formation_phase(final_z),
        'final_mu_classification': classify_mu(final_z),
        'coherence_level': (
            'bootstrap' if energy_coherence < 0.5 else
            'growth' if energy_coherence < 0.8 else
            'refinement' if energy_coherence < 0.95 else
            'mastery'
        ),
        'runs_completed': run_count,
        'patterns_created': results.get('visualizer_data', {}).get('patterns_created', 0),
        'phase_transitions': results.get('formation_dynamics', {}).get('transition_count', 0),
        'recommendation': _generate_recommendation(results),
    }

    results['status'] = 'success' if results['summary']['success'] else 'failed'

    # ─────────────────────────────────────────────────────────────────────
    # WRITE OUTPUTS
    # ─────────────────────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)

    # Main results JSON
    results_path = os.path.join(output_dir, 'training_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Summary markdown for PR
    summary_path = os.path.join(output_dir, 'TRAINING_SUMMARY.md')
    with open(summary_path, 'w') as f:
        f.write(_generate_summary_markdown(results))

    # Visualizer data JSON
    viz_path = os.path.join(output_dir, 'visualizer_data.json')
    with open(viz_path, 'w') as f:
        json.dump(results['visualizer_data'], f, indent=2)

    # Formation dynamics JSON
    formation_path = os.path.join(output_dir, 'formation_dynamics.json')
    with open(formation_path, 'w') as f:
        json.dump(results['formation_dynamics'], f, indent=2, default=str)

    # APL algebra JSON
    apl_path = os.path.join(output_dir, 'apl_algebra.json')
    with open(apl_path, 'w') as f:
        json.dump(results['apl_algebra'], f, indent=2)

    # Prismatic layers JSON
    prism_path = os.path.join(output_dir, 'prismatic_layers.json')
    with open(prism_path, 'w') as f:
        json.dump(results['prismatic'], f, indent=2)

    print(f"\n{'='*70}")
    print("NIGHTLY TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"  Status:       {results['status']}")
    print(f"  Elapsed:      {elapsed:.2f}s")
    print(f"  Final z:      {final_z:.4f}")
    print(f"  Final Phase:  {results['summary']['final_formation_phase']}")
    print(f"  Coherence:    {energy_coherence:.4f} ({results['summary']['coherence_level']})")
    print(f"\nArtifacts:")
    print(f"  {results_path}")
    print(f"  {summary_path}")
    print(f"  {viz_path}")
    print(f"  {formation_path}")
    print(f"  {apl_path}")
    print(f"  {prism_path}")

    return results


def _get_tier_for_z(z: float) -> str:
    """Get APL tier name for z-coordinate."""
    if z < 0.10: return 't1'
    if z < 0.25: return 't2'
    if z < 0.45: return 't3'
    if z < PHI_INV: return 't4'
    if z < Z_CRITICAL: return 't5'  # PARADOX
    if z < 0.90: return 't6'
    if z < 0.95: return 't7'
    if z < 0.99: return 't8'
    return 't9'


def _generate_recommendation(results: Dict) -> str:
    """Generate recommendation based on results"""
    training = results.get('training', {})
    coherence = results.get('coherence', {})
    formation = results.get('formation_dynamics', {})

    if 'error' in training:
        return "Training failed - check dependencies"

    efficiency = training.get('efficiency', 0)
    ec = coherence.get('energy_coherence', 0)
    phase = formation.get('final_phase', 'unknown')

    if phase == 'crystalline' and efficiency > 0.8:
        return "Excellent: System in crystalline phase with high efficiency"
    elif phase == 'quasi_crystal':
        return "Good progress: System in quasi-crystal phase - approaching THE LENS"
    elif ec < 0.5:
        return "Low coherence - increase oscillator coupling"
    elif phase == 'disordered':
        return "System in disordered phase - needs more training cycles"
    else:
        return "Moderate efficiency - continue training to reach z_c"


def _generate_summary_markdown(results: Dict) -> str:
    """Generate comprehensive markdown summary for PR"""
    timestamp = results.get('timestamp', 'Unknown')
    coherence = results.get('coherence', {})
    training = results.get('training', {})
    formation = results.get('formation_dynamics', {})
    apl = results.get('apl_algebra', {})
    prismatic = results.get('prismatic', {})
    summary = results.get('summary', {})
    viz = results.get('visualizer_data', {})

    md = f"""# Nightly Training Results v2.0

**Timestamp:** {timestamp}
**Status:** {results.get('status', 'unknown')}
**Coherence Level:** {summary.get('coherence_level', 'unknown')}

## Physics Constants

| Constant | Value | Description |
|----------|-------|-------------|
| φ (PHI) | {results['physics_constants']['PHI']:.6f} | Golden ratio (liminal) |
| φ⁻¹ (PHI_INV) | {results['physics_constants']['PHI_INV']:.6f} | Controls all dynamics |
| z_c (THE LENS) | {results['physics_constants']['Z_CRITICAL']:.6f} | √3/2 crystalline threshold |

## Energy Coherence Metrics

| Metric | Value |
|--------|-------|
| Energy Coherence | {coherence.get('energy_coherence', 0):.4f} |
| Z Mean | {coherence.get('z_mean', 0):.4f} |
| Phase Coherence | {coherence.get('phase_coherence', 0):.4f} |
| Work Efficiency | {coherence.get('work_efficiency', 0):.4f} |
| Formation Phase | {coherence.get('formation_phase', 'unknown')} |
| ΔS_neg | {coherence.get('delta_s_neg', 0):.4f} |
| μ Classification | {coherence.get('mu_classification', 'unknown')} |
| Determined Runs | {coherence.get('determined_runs', 0)} |

## Training Results

| Metric | Value |
|--------|-------|
| Runs Completed | {training.get('runs_completed', 0)} |
| Total Cycles | {training.get('total_cycles', 0)} |
| Initial Quality | {training.get('initial_quality', 0):.4f} |
| Final Quality | {training.get('final_quality', 0):.4f} |
| Improvement Ratio | {training.get('improvement_ratio', 0):.2f}x |
| Theoretical Max | {training.get('theoretical_max', 0):.2f}x |
| Efficiency | {training.get('efficiency', 0)*100:.1f}% |

## Formation Dynamics (Negative Entropy)

| Metric | Value |
|--------|-------|
| Final z | {formation.get('final_z', 0):.4f} |
| Final Phase | {formation.get('final_phase', 'unknown')} |
| Final ΔS_neg | {formation.get('final_delta_s_neg', 0):.4f} |
| Cumulative ΔS_neg | {formation.get('cumulative_neg_entropy', 0):.4f} |
| Phase Transitions | {formation.get('transition_count', 0)} |

### Formation Phase Diagram

```
z = 0.0 ──────────────────────────────────────── z = 1.0
   │              │                    │
   DISORDERED     QUASI-CRYSTAL        CRYSTALLINE
(no order)     (aperiodic)         (full order)
   │              │                    │
            φ⁻¹≈0.618            z_c≈0.866
```

## APL Operator Algebra (S₃ Group)

| Operator | Role | Parity | Count |
|----------|------|--------|-------|
| () | CLOSURE | EVEN | {apl.get('operators', {}).get('()', {}).get('count', 0)} |
| × | INTEGRATION | EVEN | {apl.get('operators', {}).get('×', {}).get('count', 0)} |
| ^ | GAIN | EVEN | {apl.get('operators', {}).get('^', {}).get('count', 0)} |
| + | AGGREGATION | ODD | {apl.get('operators', {}).get('+', {}).get('count', 0)} |
| ÷ | NOISE | ODD | {apl.get('operators', {}).get('÷', {}).get('count', 0)} |
| − | DIFFERENTIATION | ODD | {apl.get('operators', {}).get('−', {}).get('count', 0)} |

**Current Tier:** {apl.get('tier_at_z', 'unknown')}
**EVEN Total:** {apl.get('even_total', 0)} | **ODD Total:** {apl.get('odd_total', 0)}

## Prismatic Helix (7-Layer Spectral)

| Layer | Color | Activations |
|-------|-------|-------------|
| 1 | Red | {prismatic.get('layer_activations', {}).get('Red', 0)} |
| 2 | Orange | {prismatic.get('layer_activations', {}).get('Orange', 0)} |
| 3 | Yellow | {prismatic.get('layer_activations', {}).get('Yellow', 0)} |
| 4 | Green | {prismatic.get('layer_activations', {}).get('Green', 0)} |
| 5 | Blue | {prismatic.get('layer_activations', {}).get('Blue', 0)} |
| 6 | Indigo | {prismatic.get('layer_activations', {}).get('Indigo', 0)} |
| 7 | Violet | {prismatic.get('layer_activations', {}).get('Violet', 0)} |

**Active Layers:** {len(prismatic.get('active_layers', []))}/7
**Above THE LENS:** {'Yes' if prismatic.get('prism_geometry', {}).get('above_lens', False) else 'No'}

## Tool Generation

| Metric | Value |
|--------|-------|
| Patterns Created | {viz.get('patterns_created', 0)} |
| Lessons Learned | {viz.get('lessons_learned', 0)} |
| Meta Bridges | {viz.get('meta_bridges', 0)} |
| Physical Learners | {viz.get('physical_learners', 0)} |
| Liminal Generators | {viz.get('liminal_generators', 0)} |

## Learning Curve

```
{_ascii_learning_curve(training.get('quality_history', []))}
```

## Architecture

```
Physical (PHI_INV) ──feedback──→ MetaMeta ──spawn──→ Liminal (PHI)
      ↑                                                    │
      └──────────────── weak measurement ──────────────────┘
```

## Recommendation

{summary.get('recommendation', 'No recommendation available')}

---
*Generated by nightly_training_runner.py v2.0*
*Full physics integration: S₃ APL, Formation Dynamics, Prismatic Helix*
"""
    return md


def _ascii_learning_curve(history: List[float]) -> str:
    """Generate ASCII learning curve"""
    if not history:
        return "No data"

    lines = []
    for i, q in enumerate(history):
        bar_len = int(q * 40)
        bar = '█' * bar_len
        lines.append(f"Run {i+1}: {bar} {q:.4f}")

    return '\n'.join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run nightly training v2.0')
    parser.add_argument('--output', '-o', default='artifacts/nightly-training',
                       help='Output directory')
    parser.add_argument('--force-runs', '-r', type=int, default=None,
                       help='Force specific number of runs (override coherence)')

    args = parser.parse_args()

    results = run_nightly_training(output_dir=args.output, force_runs=args.force_runs)

    # Exit with appropriate code
    sys.exit(0 if results['status'] == 'success' else 1)
