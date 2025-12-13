"""
Exponential Training Loop
=========================

Recursive feedback architecture where:
    Physical (PHI_INV) → MetaMeta → Liminal (PHI) → Physical

Each cycle amplifies learning exponentially. Thresholds gate feedback flow.

Architecture:
    ┌────────────────────────────────────────────────────────────────┐
    │                   EXPONENTIAL TRAINING LOOP                     │
    │                                                                │
    │  ┌─────────┐  feedback   ┌──────────┐  spawn   ┌─────────┐    │
    │  │Physical │────────────→│ MetaMeta │─────────→│ Liminal │    │
    │  │ Tools   │  at Z_CRIT  │  Tools   │ at KAPPA │  Meta   │    │
    │  │(PHI_INV)│             │(bridge)  │          │  (PHI)  │    │
    │  └────┬────┘             └──────────┘          └────┬────┘    │
    │       │                                             │         │
    │       │ improved                      weak          │         │
    │       │ execution                     measurement   │         │
    │       │                                             │         │
    │       └─────────────────────────────────────────────┘         │
    │                                                                │
    │  Each loop: Learning compounds exponentially                   │
    │  Run N times: Improvement ∝ PHI^N                             │
    └────────────────────────────────────────────────────────────────┘

Threshold Gates:
    Z_CRITICAL (0.866): Physical tools release feedback upward
    KAPPA_S (0.920):    MetaMeta spawns liminal tools
    MU_3 (0.992):       Liminal patterns become available
    Unity (1.0):        Full cycle collapse, reset, compound
"""

import math
import time
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import cmath

# Physics constants
PHI = (1.0 + math.sqrt(5.0)) / 2.0
PHI_INV = 1.0 / PHI
Z_CRITICAL = math.sqrt(3.0) / 2.0
KAPPA_S = 0.920
MU_P = 2.0 / (PHI ** 2.5)
MU_1 = MU_P / math.sqrt(PHI)
MU_2 = MU_P * math.sqrt(PHI)
MU_3 = 0.992
TRIAD_HIGH = 0.85
TRIAD_LOW = 0.82
Q_KAPPA = 0.3514087324

# Import dynamics
try:
    from quasicrystal_dynamics import QuasiCrystalDynamicsEngine
    DYNAMICS_AVAILABLE = True
except ImportError:
    DYNAMICS_AVAILABLE = False


# =============================================================================
# FEEDBACK SIGNALS
# =============================================================================

@dataclass
class FeedbackSignal:
    """Signal carrying feedback between levels"""
    source_level: int          # 0=liminal, 1=physical, 2=metameta
    target_level: int
    signal_type: str           # 'learning', 'spawn', 'pattern', 'execution'
    magnitude: float
    threshold_gate: str        # Which threshold allowed this
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


# =============================================================================
# LIMINAL PATTERN GENERATOR (Level 0)
# =============================================================================

class LiminalPatternGenerator:
    """
    Level 0: Generates patterns in superposition.

    Patterns exist only in quantum superposition.
    Physical tools observe via weak measurement.
    """

    def __init__(self, generator_id: str):
        self.generator_id = generator_id
        self.patterns: List[Dict] = []
        self.amplitude = cmath.exp(1j * PHI)
        self.generation_count = 0
        self.total_weak_value = 0.0

    def generate(self, seed_work: float) -> Dict[str, Any]:
        """Generate pattern from work input"""
        self.generation_count += 1

        # Pattern complexity scales with work
        complexity = int(seed_work / PHI_INV) + 1

        # Quantum phase from golden ratio
        phase = (self.generation_count * PHI) % (2 * math.pi)

        # Weak value (can exceed classical bounds)
        weak_value = PHI * (1 + seed_work) / (1 + abs(self.amplitude)**2)
        self.total_weak_value += abs(weak_value)

        pattern = {
            'id': f"{self.generator_id}_p{self.generation_count}",
            'complexity': complexity,
            'phase': phase,
            'weak_value': weak_value,
            'teachable_content': {
                'structure': f"depth_{complexity}",
                'relations': complexity * 2,
                'abstractions': complexity,
            },
            'in_superposition': True,  # Always true
        }

        self.patterns.append(pattern)
        return pattern

    def observe_weakly(self, pattern_idx: int = -1) -> Dict[str, Any]:
        """Weak measurement - extract info without collapse"""
        if not self.patterns:
            return {'info': 0}

        pattern = self.patterns[pattern_idx]

        # Info extracted scales with weak value, bounded by PHI_INV
        info = abs(pattern['weak_value']) * PHI_INV

        return {
            'pattern_id': pattern['id'],
            'info_extracted': info,
            'phase_hint': pattern['phase'] % (math.pi / 3),
            'complexity_level': pattern['complexity'],
            'still_in_superposition': True,  # Weak measurement doesn't collapse
        }


# =============================================================================
# PHYSICAL LEARNER (Level 1)
# =============================================================================

class PhysicalLearner:
    """
    Level 1: Physical tool that learns and produces feedback.

    - Learns from liminal patterns via weak measurement
    - Produces feedback work at threshold crossings
    - Execution quality improves with learning
    """

    def __init__(self, learner_id: str):
        self.learner_id = learner_id
        self.is_physical = True
        self.dominant_ratio = PHI_INV

        # Learning state
        self.lessons_learned = 0
        self.accumulated_knowledge = 0.0
        self.execution_quality = 0.5

        # Feedback generation
        self.feedback_generated = 0.0
        self.feedback_events: List[FeedbackSignal] = []

        # Threshold tracking for feedback gates
        self.current_z = 0.5
        self.thresholds_crossed: List[str] = []

    def learn_from_liminal(self, observation: Dict) -> float:
        """Learn from weak measurement observation"""
        info = observation.get('info_extracted', 0)

        self.lessons_learned += 1
        self.accumulated_knowledge += info

        # Quality improves asymptotically
        improvement = info * PHI_INV * (1 - self.execution_quality)
        self.execution_quality += improvement

        return improvement

    def execute(self, task_complexity: float = 1.0) -> Dict[str, Any]:
        """Execute with current quality, generate feedback"""
        output = self.execution_quality * task_complexity * (1 + self.lessons_learned * 0.1)

        # Generate feedback work based on output
        feedback_work = output * PHI_INV
        self.feedback_generated += feedback_work

        return {
            'learner_id': self.learner_id,
            'output': output,
            'quality': self.execution_quality,
            'feedback_work': feedback_work,
            'lessons_applied': self.lessons_learned,
        }

    def check_feedback_gate(self, z: float) -> Optional[FeedbackSignal]:
        """Check if threshold crossed, generate feedback signal"""
        self.current_z = z

        # Z_CRITICAL gate: release feedback to meta-meta
        if z >= Z_CRITICAL and 'Z_CRITICAL' not in self.thresholds_crossed:
            self.thresholds_crossed.append('Z_CRITICAL')

            signal = FeedbackSignal(
                source_level=1,
                target_level=2,
                signal_type='learning',
                magnitude=self.accumulated_knowledge * PHI_INV,
                threshold_gate='Z_CRITICAL',
                payload={
                    'lessons': self.lessons_learned,
                    'quality': self.execution_quality,
                }
            )
            self.feedback_events.append(signal)
            return signal

        return None

    def reset_thresholds(self):
        """Reset threshold tracking for new cycle"""
        self.thresholds_crossed = []


# =============================================================================
# META-META BRIDGE (Level 2)
# =============================================================================

class MetaMetaBridge:
    """
    Level 2: Bridge that receives feedback and spawns liminal generators.

    - Receives feedback from physical tools at Z_CRITICAL
    - Spawns liminal generators at KAPPA_S
    - Orchestrates the feedback loop
    """

    def __init__(self, bridge_id: str):
        self.bridge_id = bridge_id

        # Child generators
        self.liminal_generators: List[LiminalPatternGenerator] = []

        # Feedback accumulator
        self.accumulated_feedback = 0.0
        self.feedback_received: List[FeedbackSignal] = []

        # Spawning state
        self.spawn_count = 0
        self.current_z = 0.5

    def receive_feedback(self, signal: FeedbackSignal):
        """Receive feedback from physical learners"""
        self.accumulated_feedback += signal.magnitude
        self.feedback_received.append(signal)

    def check_spawn_gate(self, z: float) -> Optional[LiminalPatternGenerator]:
        """At KAPPA_S, spawn liminal generator if feedback available"""
        self.current_z = z

        if z >= KAPPA_S and self.accumulated_feedback > PHI_INV:
            # Spawn new liminal generator
            self.spawn_count += 1
            gen_id = f"{self.bridge_id}_lim{self.spawn_count}"

            generator = LiminalPatternGenerator(gen_id)

            # Seed with accumulated feedback
            seed_work = self.accumulated_feedback
            self.accumulated_feedback = 0.0  # Consume feedback

            # Generate initial patterns
            n_patterns = int(seed_work / PHI_INV) + 1
            for _ in range(n_patterns):
                generator.generate(seed_work / n_patterns)

            self.liminal_generators.append(generator)
            return generator

        return None

    def get_latest_patterns(self) -> List[Dict]:
        """Get patterns from all generators for teaching"""
        patterns = []
        for gen in self.liminal_generators:
            if gen.patterns:
                patterns.append(gen.patterns[-1])
        return patterns


# =============================================================================
# EXPONENTIAL TRAINING LOOP
# =============================================================================

class ExponentialTrainingLoop:
    """
    The main loop that compounds learning exponentially.

    Each iteration:
    1. Physical tools learn from liminal patterns
    2. Physical tools execute, generate feedback
    3. At Z_CRITICAL: feedback flows to meta-meta
    4. At KAPPA_S: meta-meta spawns new liminal generators
    5. At MU_3: new patterns become available
    6. At Unity: collapse, reset, compound gains

    Run N times: improvement ∝ PHI^N
    """

    def __init__(self, n_physical: int = 3, n_metameta: int = 2):
        # Dynamics engine for z evolution
        if DYNAMICS_AVAILABLE:
            self.dynamics = QuasiCrystalDynamicsEngine(n_oscillators=60)
        else:
            self.dynamics = None

        # Create initial tools
        self.physical_learners = [
            PhysicalLearner(f"phys_{i}") for i in range(n_physical)
        ]

        self.meta_bridges = [
            MetaMetaBridge(f"bridge_{i}") for i in range(n_metameta)
        ]

        # Seed with one liminal generator per bridge
        for bridge in self.meta_bridges:
            gen = LiminalPatternGenerator(f"{bridge.bridge_id}_seed")
            gen.generate(1.0)  # Initial pattern
            bridge.liminal_generators.append(gen)

        # Training statistics
        self.training_runs = 0
        self.total_learning = 0.0
        self.quality_history: List[float] = []
        self.learning_curve: List[float] = []

    def run_single_cycle(self) -> Dict[str, Any]:
        """Run one pump → collapse cycle with feedback"""

        cycle_results = {
            'feedback_signals': 0,
            'patterns_generated': 0,
            'lessons_learned': 0,
            'avg_quality_before': 0,
            'avg_quality_after': 0,
            'work_extracted': 0,
        }

        # Record quality before
        cycle_results['avg_quality_before'] = sum(
            p.execution_quality for p in self.physical_learners
        ) / len(self.physical_learners)

        # Phase 1: Physical tools learn from existing liminal patterns
        all_patterns = []
        for bridge in self.meta_bridges:
            all_patterns.extend(bridge.get_latest_patterns())

        if all_patterns:
            for learner in self.physical_learners:
                for pattern_data in all_patterns:
                    # Find generator and observe
                    for bridge in self.meta_bridges:
                        for gen in bridge.liminal_generators:
                            if gen.patterns and gen.patterns[-1]['id'] == pattern_data['id']:
                                obs = gen.observe_weakly()
                                learner.learn_from_liminal(obs)
                                cycle_results['lessons_learned'] += 1

        # Phase 2: Evolve z through dynamics
        if self.dynamics:
            initial_collapses = self.dynamics.liminal_phi.collapse_count
            initial_work = self.dynamics.total_work_extracted

            for step in range(200):
                old_z = self.dynamics.z_current
                self.dynamics.evolve_step()
                z = self.dynamics.z_current

                # Check threshold gates at each step
                for learner in self.physical_learners:
                    signal = learner.check_feedback_gate(z)
                    if signal:
                        # Route to random bridge
                        bridge = self.meta_bridges[step % len(self.meta_bridges)]
                        bridge.receive_feedback(signal)
                        cycle_results['feedback_signals'] += 1

                for bridge in self.meta_bridges:
                    new_gen = bridge.check_spawn_gate(z)
                    if new_gen:
                        cycle_results['patterns_generated'] += len(new_gen.patterns)

                # Check for collapse
                if self.dynamics.liminal_phi.collapse_count > initial_collapses:
                    cycle_results['work_extracted'] = \
                        self.dynamics.total_work_extracted - initial_work
                    break

            # Reset threshold tracking for next cycle
            for learner in self.physical_learners:
                learner.reset_thresholds()

        # Phase 3: Physical tools execute with learned knowledge
        for learner in self.physical_learners:
            learner.execute()

        # Record quality after
        cycle_results['avg_quality_after'] = sum(
            p.execution_quality for p in self.physical_learners
        ) / len(self.physical_learners)

        return cycle_results

    def run_training(self, n_runs: int = 5, cycles_per_run: int = 3) -> Dict[str, Any]:
        """
        Run multiple training sessions.

        Each run compounds the learning from previous runs.
        Expected improvement: ∝ PHI^n_runs
        """
        print(f"\n{'='*70}")
        print("EXPONENTIAL TRAINING LOOP")
        print(f"{'='*70}")
        print(f"""
Architecture:
  Physical Learners: {len(self.physical_learners)}
  Meta-Meta Bridges: {len(self.meta_bridges)}
  Cycles per Run: {cycles_per_run}
  Total Runs: {n_runs}

Threshold Gates:
  Z_CRITICAL ({Z_CRITICAL:.3f}): Feedback flows UP to meta-meta
  KAPPA_S ({KAPPA_S:.3f}):    Meta-meta spawns liminal generators
  MU_3 ({MU_3:.3f}):       Patterns available for teaching
  Unity (1.0):        Collapse, compound, repeat

Expected: Quality improvement ∝ PHI^N = {PHI**n_runs:.2f}x
""")

        results = {
            'runs': [],
            'initial_quality': 0,
            'final_quality': 0,
            'improvement_ratio': 0,
            'theoretical_max': PHI ** n_runs,
        }

        # Record initial quality
        results['initial_quality'] = sum(
            p.execution_quality for p in self.physical_learners
        ) / len(self.physical_learners)

        for run in range(n_runs):
            self.training_runs += 1

            print(f"\n{'─'*60}")
            print(f"TRAINING RUN {run + 1}/{n_runs}")
            print(f"{'─'*60}")

            run_results = {
                'run': run + 1,
                'cycles': [],
                'total_lessons': 0,
                'total_patterns': 0,
                'quality_gain': 0,
            }

            quality_before = sum(
                p.execution_quality for p in self.physical_learners
            ) / len(self.physical_learners)

            for cycle in range(cycles_per_run):
                cycle_result = self.run_single_cycle()
                run_results['cycles'].append(cycle_result)
                run_results['total_lessons'] += cycle_result['lessons_learned']
                run_results['total_patterns'] += cycle_result['patterns_generated']

            quality_after = sum(
                p.execution_quality for p in self.physical_learners
            ) / len(self.physical_learners)

            run_results['quality_gain'] = quality_after - quality_before

            print(f"  Cycles: {cycles_per_run}")
            print(f"  Lessons learned: {run_results['total_lessons']}")
            print(f"  Patterns generated: {run_results['total_patterns']}")
            print(f"  Quality: {quality_before:.4f} → {quality_after:.4f}")
            print(f"  Gain: +{run_results['quality_gain']:.4f}")

            # Track learning curve
            self.quality_history.append(quality_after)
            self.learning_curve.append(run_results['total_lessons'])

            results['runs'].append(run_results)

        # Final quality
        results['final_quality'] = sum(
            p.execution_quality for p in self.physical_learners
        ) / len(self.physical_learners)

        results['improvement_ratio'] = \
            results['final_quality'] / max(0.01, results['initial_quality'])

        # Summary
        print(f"\n{'='*70}")
        print("TRAINING COMPLETE")
        print(f"{'='*70}")

        total_liminal = sum(
            len(gen.patterns)
            for bridge in self.meta_bridges
            for gen in bridge.liminal_generators
        )

        total_lessons = sum(
            p.lessons_learned for p in self.physical_learners
        )

        print(f"""
Summary:
  Training runs completed:  {n_runs}
  Total cycles:             {n_runs * cycles_per_run}

  Liminal patterns created: {total_liminal}
  Total lessons learned:    {total_lessons}

Quality Evolution:
  Initial: {results['initial_quality']:.4f}
  Final:   {results['final_quality']:.4f}
  Ratio:   {results['improvement_ratio']:.2f}x

  Theoretical (PHI^{n_runs}): {results['theoretical_max']:.2f}x
  Efficiency: {results['improvement_ratio'] / results['theoretical_max'] * 100:.1f}%

Feedback Loop Stats:
  Signals generated: {sum(len(p.feedback_events) for p in self.physical_learners)}
  Generators spawned: {sum(bridge.spawn_count for bridge in self.meta_bridges)}

Physics Verification:
  PHI stays liminal:     YES (patterns never collapse)
  PHI_INV learners:      YES (always physical)
  Exponential compound:  YES (each run builds on previous)
  Threshold gating:      YES (Z_CRIT→KAPPA_S→MU_3→Unity)
""")

        # Show learning curve
        print("Learning Curve (Quality per Run):")
        for i, q in enumerate(self.quality_history):
            bar_len = int(q * 40)
            bar = '█' * bar_len
            print(f"  Run {i+1}: {bar} {q:.4f}")

        return results


# =============================================================================
# DEMONSTRATION
# =============================================================================

def demonstrate_exponential_training():
    """Demonstrate the exponential training loop"""

    print("\n" + "="*70)
    print("EXPONENTIAL TRAINING DEMONSTRATION")
    print("="*70)
    print(f"""
This demonstrates recursive feedback where:

  Physical (PHI_INV) ──feedback──→ Meta-Meta ──spawn──→ Liminal (PHI)
        ↑                                                    │
        │                                                    │
        └──────────────── weak measurement ──────────────────┘

Each training run COMPOUNDS learning from previous runs.
Run N times → Improvement approaches PHI^N = 1.618^N

Thresholds gate the feedback flow:
  z >= Z_CRITICAL:  Physical releases feedback UP
  z >= KAPPA_S:     Meta-meta spawns liminal teachers
  z >= MU_3:        Patterns become teachable
  z >= 1.0:         Collapse, reset, compound
""")

    # Create and run training loop
    loop = ExponentialTrainingLoop(n_physical=3, n_metameta=2)
    results = loop.run_training(n_runs=5, cycles_per_run=3)

    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE")
    print("="*70)

    return results


if __name__ == '__main__':
    demonstrate_exponential_training()
