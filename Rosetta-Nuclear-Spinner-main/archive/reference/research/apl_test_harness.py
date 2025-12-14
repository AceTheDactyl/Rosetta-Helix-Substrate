#!/usr/bin/env python3
"""
APL Seven Sentences Test Harness — Phase 3 Enhancement
=======================================================

Unified test framework for running and validating the APL Seven Sentences
test pack. Each sentence defines a hypothesis about physical system behavior
under specific APL operator conditions.

SENTENCES:
  A1: d()|Conductor|geometry     → Isotropic lattices under collapse
  A3: u^|Oscillator|wave         → Amplified vortex-rich waves
  A4: m×|Encoder|chemistry       → Helical information carriers
  A5: u×|Catalyst|chemistry      → Fractal polymer branching
  A6: u+|Reactor|wave            → Jet-like coherent grouping
  A7: u÷|Reactor|wave            → Stochastic decohered waves
  A8: m()|Filter|wave            → Adaptive boundary tuning

METHODOLOGY:
  - LHS condition: Apply the operator specified in the sentence
  - Control: Same setup without the key operator
  - Metric: Domain-specific measurement of the RHS regime
  - Validation: Statistical comparison (LHS vs Control)

Usage:
  python apl_test_harness.py --sentence A3 --trials 50
  python apl_test_harness.py --all --trials 100 --output results/

@version 2.0.0 (TRIAD Protocol Phase 3)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import random
import math

# Optional imports for statistics and visualization
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class APLSentence:
    """Representation of an APL test sentence."""
    sentence_id: str
    direction: str          # u (up/forward), d (down/collapse), m (modulation)
    operator: str           # APL operator symbol
    machine: str            # Processing context
    domain: str             # Physical domain
    predicted_regime: str   # Expected behavior/regime
    
    def token(self) -> str:
        """Render as compact APL token."""
        return f"{self.direction}{self.operator}|{self.machine}|{self.domain}"
    
    def __str__(self) -> str:
        return f"{self.sentence_id}: {self.token()} → {self.predicted_regime}"


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""
    sentence: APLSentence
    lhs_condition: bool     # True = apply operator, False = control
    trial_id: int
    seed: Optional[int] = None
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentResult:
    """Result from a single experiment run."""
    config: ExperimentConfig
    metric: float           # Primary outcome metric
    duration_ms: float      # Execution time
    metadata: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error: Optional[str] = None


@dataclass
class SentenceTestResult:
    """Aggregated results for a sentence across all trials."""
    sentence: APLSentence
    lhs_results: List[ExperimentResult]
    control_results: List[ExperimentResult]
    
    # Statistics (computed after trials)
    lhs_mean: Optional[float] = None
    lhs_std: Optional[float] = None
    control_mean: Optional[float] = None
    control_std: Optional[float] = None
    t_statistic: Optional[float] = None
    p_value: Optional[float] = None
    effect_size: Optional[float] = None  # Cohen's d
    
    def compute_statistics(self) -> None:
        """Compute aggregate statistics from trial results."""
        lhs_metrics = [r.metric for r in self.lhs_results if r.success]
        ctrl_metrics = [r.metric for r in self.control_results if r.success]
        
        if not lhs_metrics or not ctrl_metrics:
            return
        
        if HAS_NUMPY:
            self.lhs_mean = float(np.mean(lhs_metrics))
            self.lhs_std = float(np.std(lhs_metrics, ddof=1))
            self.control_mean = float(np.mean(ctrl_metrics))
            self.control_std = float(np.std(ctrl_metrics, ddof=1))
        else:
            # Pure Python fallback
            self.lhs_mean = sum(lhs_metrics) / len(lhs_metrics)
            self.control_mean = sum(ctrl_metrics) / len(ctrl_metrics)
            self.lhs_std = math.sqrt(sum((x - self.lhs_mean)**2 for x in lhs_metrics) / (len(lhs_metrics) - 1))
            self.control_std = math.sqrt(sum((x - self.control_mean)**2 for x in ctrl_metrics) / (len(ctrl_metrics) - 1))
        
        # T-test (requires scipy)
        if HAS_SCIPY and len(lhs_metrics) >= 2 and len(ctrl_metrics) >= 2:
            stat, pval = scipy_stats.ttest_ind(lhs_metrics, ctrl_metrics, equal_var=False)
            self.t_statistic = float(stat)
            self.p_value = float(pval)
        
        # Cohen's d effect size
        if self.lhs_std and self.control_std:
            pooled_std = math.sqrt((self.lhs_std**2 + self.control_std**2) / 2)
            if pooled_std > 0:
                self.effect_size = (self.lhs_mean - self.control_mean) / pooled_std
    
    def is_significant(self, alpha: float = 0.05) -> bool:
        """Check if difference is statistically significant."""
        return self.p_value is not None and self.p_value < alpha
    
    def supports_hypothesis(self, alpha: float = 0.05) -> bool:
        """Check if results support the APL hypothesis (LHS > Control)."""
        if self.lhs_mean is None or self.control_mean is None:
            return False
        return self.lhs_mean > self.control_mean and self.is_significant(alpha)


# ============================================================================
# APL SENTENCES REGISTRY
# ============================================================================

APL_SENTENCES: Dict[str, APLSentence] = {
    'A1': APLSentence('A1', 'd', '()', 'Conductor', 'geometry', 'Isotropic lattices under collapse'),
    'A3': APLSentence('A3', 'u', '^', 'Oscillator', 'wave', 'Amplified vortex-rich waves'),
    'A4': APLSentence('A4', 'm', '×', 'Encoder', 'chemistry', 'Helical information carriers'),
    'A5': APLSentence('A5', 'u', '×', 'Catalyst', 'chemistry', 'Fractal polymer branching'),
    'A6': APLSentence('A6', 'u', '+', 'Reactor', 'wave', 'Jet-like coherent grouping'),
    'A7': APLSentence('A7', 'u', '÷', 'Reactor', 'wave', 'Stochastic decohered waves'),
    'A8': APLSentence('A8', 'm', '()', 'Filter', 'wave', 'Adaptive boundary tuning'),
}


# ============================================================================
# DOMAIN SIMULATORS (Abstract Base)
# ============================================================================

class DomainSimulator(ABC):
    """Abstract base class for domain-specific simulators."""
    
    @abstractmethod
    def setup(self, config: ExperimentConfig) -> None:
        """Initialize simulation domain."""
        pass
    
    @abstractmethod
    def apply_operator(self, operator: str, direction: str, **kwargs) -> None:
        """Apply APL operator to the domain."""
        pass
    
    @abstractmethod
    def run(self, duration: float) -> Any:
        """Run simulation and return raw result."""
        pass
    
    @abstractmethod
    def compute_metric(self, result: Any, regime: str) -> float:
        """Compute regime-specific metric from result."""
        pass


class WaveSimulator(DomainSimulator):
    """
    Simulator for wave domain experiments (A3, A6, A7, A8).
    
    In a full implementation, this would interface with a CFD solver
    or wave equation integrator. For demonstration, we use a simplified
    stochastic model that captures the essential operator effects.
    """
    
    def __init__(self):
        self.state = None
        self.rng = random.Random()
        
    def setup(self, config: ExperimentConfig) -> None:
        if config.seed is not None:
            self.rng.seed(config.seed)
        
        # Initialize wave state
        self.state = {
            'amplitude': 1.0,
            'coherence': 0.5,
            'vorticity': 0.0,
            'noise_level': 0.1,
            'boundary_strength': 0.5
        }
        
        # Apply machine-specific setup
        machine = config.sentence.machine.lower()
        if machine == 'oscillator':
            self.state['coherence'] = 0.8
        elif machine == 'reactor':
            self.state['amplitude'] = 1.5
        elif machine == 'filter':
            self.state['boundary_strength'] = 0.8
    
    def apply_operator(self, operator: str, direction: str, **kwargs) -> None:
        """Apply APL operator effect to wave state."""
        intensity = kwargs.get('intensity', 1.0)
        
        if operator == '^':  # Amplify
            if direction == 'u':
                self.state['amplitude'] *= (1.0 + 0.5 * intensity)
                self.state['vorticity'] += 0.3 * intensity
            else:
                self.state['amplitude'] *= (1.0 - 0.3 * intensity)
        
        elif operator == '+':  # Group
            if direction == 'u':
                self.state['coherence'] += 0.2 * intensity
                self.state['coherence'] = min(1.0, self.state['coherence'])
        
        elif operator == '÷' or operator == '%':  # Decoherence
            if direction == 'u':
                self.state['noise_level'] += 0.3 * intensity
                self.state['coherence'] *= (1.0 - 0.2 * intensity)
        
        elif operator == '()':  # Boundary
            if direction == 'm':
                self.state['boundary_strength'] += 0.2 * intensity
                self.state['boundary_strength'] = min(1.0, self.state['boundary_strength'])
    
    def run(self, duration: float) -> Dict[str, float]:
        """Simulate wave evolution."""
        # Simplified stochastic evolution
        for _ in range(int(duration * 10)):
            # Add noise
            self.state['amplitude'] += self.rng.gauss(0, self.state['noise_level'] * 0.1)
            self.state['vorticity'] += self.rng.gauss(0, 0.05) * self.state['amplitude']
            
            # Boundary damping
            self.state['vorticity'] *= (1.0 - 0.1 * self.state['boundary_strength'])
            
            # Coherence evolution
            self.state['coherence'] *= (1.0 - 0.01 * self.state['noise_level'])
        
        return dict(self.state)
    
    def compute_metric(self, result: Dict[str, float], regime: str) -> float:
        """Compute regime-specific metric."""
        regime_lower = regime.lower()
        
        if 'vortex' in regime_lower or 'amplified' in regime_lower:
            # A3: Vorticity and amplitude
            return result['vorticity'] * result['amplitude']
        
        elif 'coherent' in regime_lower or 'jet' in regime_lower or 'grouping' in regime_lower:
            # A6: Coherence metric
            return result['coherence'] * result['amplitude']
        
        elif 'decohered' in regime_lower or 'stochastic' in regime_lower:
            # A7: Turbulence/noise metric (higher is more decohered)
            return result['noise_level'] / max(0.01, result['coherence'])
        
        elif 'boundary' in regime_lower or 'adaptive' in regime_lower or 'tuning' in regime_lower:
            # A8: Boundary effectiveness
            return result['boundary_strength'] * (1.0 - result['noise_level'])
        
        # Default: amplitude
        return result['amplitude']


class GeometrySimulator(DomainSimulator):
    """
    Simulator for geometry domain experiments (A1).
    
    Models lattice formation under boundary/collapse conditions.
    """
    
    def __init__(self):
        self.state = None
        self.rng = random.Random()
    
    def setup(self, config: ExperimentConfig) -> None:
        if config.seed is not None:
            self.rng.seed(config.seed)
        
        self.state = {
            'isotropy': 0.5,
            'lattice_order': 0.5,
            'defect_density': 0.2
        }
    
    def apply_operator(self, operator: str, direction: str, **kwargs) -> None:
        intensity = kwargs.get('intensity', 1.0)
        
        if operator == '()':  # Boundary
            if direction == 'd':  # Collapse
                self.state['isotropy'] += 0.3 * intensity
                self.state['lattice_order'] += 0.2 * intensity
                self.state['defect_density'] *= (1.0 - 0.2 * intensity)
    
    def run(self, duration: float) -> Dict[str, float]:
        # Simple relaxation dynamics
        for _ in range(int(duration * 10)):
            self.state['isotropy'] += self.rng.gauss(0, 0.02)
            self.state['isotropy'] = max(0, min(1, self.state['isotropy']))
            
            # Order increases with isotropy
            self.state['lattice_order'] += 0.01 * self.state['isotropy']
            self.state['lattice_order'] = min(1, self.state['lattice_order'])
        
        return dict(self.state)
    
    def compute_metric(self, result: Dict[str, float], regime: str) -> float:
        if 'isotropic' in regime.lower() or 'lattice' in regime.lower():
            return result['isotropy'] * result['lattice_order'] * (1 - result['defect_density'])
        return result['lattice_order']


class ChemistrySimulator(DomainSimulator):
    """
    Simulator for chemistry domain experiments (A4, A5).
    
    Models molecular assembly and polymer formation.
    """
    
    def __init__(self):
        self.state = None
        self.rng = random.Random()
    
    def setup(self, config: ExperimentConfig) -> None:
        if config.seed is not None:
            self.rng.seed(config.seed)
        
        machine = config.sentence.machine.lower()
        
        self.state = {
            'helicity': 0.0,
            'fractal_dim': 1.5,
            'branching': 0.0,
            'polymer_length': 1.0
        }
        
        if machine == 'encoder':
            self.state['helicity'] = 0.2
        elif machine == 'catalyst':
            self.state['branching'] = 0.3
    
    def apply_operator(self, operator: str, direction: str, **kwargs) -> None:
        intensity = kwargs.get('intensity', 1.0)
        
        if operator == '×':  # Fusion
            if direction == 'u':  # Forward
                self.state['branching'] += 0.4 * intensity
                self.state['fractal_dim'] += 0.3 * intensity
                self.state['polymer_length'] *= (1.0 + 0.5 * intensity)
            elif direction == 'm':  # Modulation
                self.state['helicity'] += 0.4 * intensity
    
    def run(self, duration: float) -> Dict[str, float]:
        for _ in range(int(duration * 10)):
            # Polymer growth
            self.state['polymer_length'] += self.rng.gauss(0.05, 0.02)
            
            # Branching evolution
            if self.state['branching'] > 0.5:
                self.state['fractal_dim'] += self.rng.gauss(0.01, 0.005)
            
            # Helicity stabilization
            self.state['helicity'] += self.rng.gauss(0, 0.01)
            self.state['helicity'] = max(0, min(1, self.state['helicity']))
        
        return dict(self.state)
    
    def compute_metric(self, result: Dict[str, float], regime: str) -> float:
        regime_lower = regime.lower()
        
        if 'helical' in regime_lower or 'information' in regime_lower:
            # A4: Helicity metric
            return result['helicity'] * result['polymer_length']
        
        elif 'fractal' in regime_lower or 'branching' in regime_lower:
            # A5: Fractal/branching metric
            return (result['fractal_dim'] - 1.0) * result['branching']
        
        return result['polymer_length']


# ============================================================================
# TEST HARNESS
# ============================================================================

class APLTestHarness:
    """
    Main test harness for APL Seven Sentences experiments.
    
    Orchestrates experiment execution, data collection, and analysis.
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path('results')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Domain simulators
        self.simulators: Dict[str, DomainSimulator] = {
            'wave': WaveSimulator(),
            'geometry': GeometrySimulator(),
            'chemistry': ChemistrySimulator()
        }
        
        self.results: Dict[str, SentenceTestResult] = {}
    
    def get_simulator(self, domain: str) -> DomainSimulator:
        """Get appropriate simulator for domain."""
        domain_lower = domain.lower()
        if domain_lower in self.simulators:
            return self.simulators[domain_lower]
        raise ValueError(f"No simulator for domain: {domain}")
    
    def run_experiment(self, config: ExperimentConfig) -> ExperimentResult:
        """
        Run a single experiment trial.
        
        @param config: Experiment configuration
        @returns: ExperimentResult with metric and metadata
        """
        import time
        start_time = time.time()
        
        try:
            # Get domain simulator
            sim = self.get_simulator(config.sentence.domain)
            
            # Setup domain
            sim.setup(config)
            
            # Apply operator if LHS condition
            if config.lhs_condition:
                sim.apply_operator(
                    config.sentence.operator,
                    config.sentence.direction,
                    intensity=config.parameters.get('intensity', 1.0)
                )
            
            # Run simulation
            duration = config.parameters.get('duration', 10.0)
            raw_result = sim.run(duration)
            
            # Compute metric
            metric = sim.compute_metric(raw_result, config.sentence.predicted_regime)
            
            duration_ms = (time.time() - start_time) * 1000
            
            return ExperimentResult(
                config=config,
                metric=metric,
                duration_ms=duration_ms,
                metadata={'raw_result': raw_result}
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return ExperimentResult(
                config=config,
                metric=0.0,
                duration_ms=duration_ms,
                success=False,
                error=str(e)
            )
    
    def run_sentence_test(
        self,
        sentence_id: str,
        n_trials: int = 50,
        base_seed: Optional[int] = None,
        **params
    ) -> SentenceTestResult:
        """
        Run complete test for a single APL sentence.
        
        @param sentence_id: Sentence identifier (e.g., 'A3')
        @param n_trials: Number of trials per condition
        @param base_seed: Base random seed for reproducibility
        @param params: Additional simulation parameters
        """
        sentence = APL_SENTENCES.get(sentence_id)
        if not sentence:
            raise ValueError(f"Unknown sentence: {sentence_id}")
        
        print(f"\n{'='*60}")
        print(f"Testing {sentence}")
        print(f"{'='*60}")
        
        lhs_results = []
        control_results = []
        
        # Run LHS condition trials
        print(f"\nRunning {n_trials} LHS trials (with operator)...")
        for i in range(n_trials):
            seed = (base_seed + i) if base_seed is not None else None
            config = ExperimentConfig(
                sentence=sentence,
                lhs_condition=True,
                trial_id=i,
                seed=seed,
                parameters=params
            )
            result = self.run_experiment(config)
            lhs_results.append(result)
            
            if (i + 1) % 10 == 0:
                print(f"  Completed {i + 1}/{n_trials}")
        
        # Run control trials
        print(f"\nRunning {n_trials} Control trials (without operator)...")
        for i in range(n_trials):
            seed = (base_seed + n_trials + i) if base_seed is not None else None
            config = ExperimentConfig(
                sentence=sentence,
                lhs_condition=False,
                trial_id=i,
                seed=seed,
                parameters=params
            )
            result = self.run_experiment(config)
            control_results.append(result)
            
            if (i + 1) % 10 == 0:
                print(f"  Completed {i + 1}/{n_trials}")
        
        # Aggregate results
        test_result = SentenceTestResult(
            sentence=sentence,
            lhs_results=lhs_results,
            control_results=control_results
        )
        test_result.compute_statistics()
        
        self.results[sentence_id] = test_result
        
        # Print summary
        self._print_result_summary(test_result)
        
        return test_result
    
    def run_all_sentences(self, n_trials: int = 50, base_seed: Optional[int] = None, **params):
        """Run tests for all APL sentences."""
        for sentence_id in APL_SENTENCES:
            self.run_sentence_test(sentence_id, n_trials, base_seed, **params)
        
        self._print_overall_summary()
    
    def _print_result_summary(self, result: SentenceTestResult) -> None:
        """Print summary for a single sentence test."""
        print(f"\n--- Results for {result.sentence.sentence_id} ---")
        print(f"LHS:     {result.lhs_mean:.4f} ± {result.lhs_std:.4f}")
        print(f"Control: {result.control_mean:.4f} ± {result.control_std:.4f}")
        
        if result.p_value is not None:
            sig = "***" if result.p_value < 0.001 else "**" if result.p_value < 0.01 else "*" if result.p_value < 0.05 else ""
            print(f"T-test:  t={result.t_statistic:.3f}, p={result.p_value:.4f} {sig}")
        
        if result.effect_size is not None:
            magnitude = "large" if abs(result.effect_size) > 0.8 else "medium" if abs(result.effect_size) > 0.5 else "small"
            print(f"Effect:  d={result.effect_size:.3f} ({magnitude})")
        
        verdict = "✓ SUPPORTS" if result.supports_hypothesis() else "✗ DOES NOT SUPPORT"
        print(f"Verdict: {verdict} hypothesis")
    
    def _print_overall_summary(self) -> None:
        """Print summary across all sentences."""
        print(f"\n{'='*60}")
        print("OVERALL SUMMARY")
        print(f"{'='*60}\n")
        
        supported = []
        not_supported = []
        
        for sid, result in self.results.items():
            if result.supports_hypothesis():
                supported.append(sid)
            else:
                not_supported.append(sid)
        
        print(f"Hypotheses supported:     {len(supported)}/{len(self.results)}")
        if supported:
            print(f"  Supported: {', '.join(supported)}")
        if not_supported:
            print(f"  Not supported: {', '.join(not_supported)}")
    
    def save_results(self, filename: Optional[str] = None) -> Path:
        """Save results to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"apl_test_results_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        # Convert results to serializable format
        output = {
            'timestamp': datetime.now().isoformat(),
            'sentences': {}
        }
        
        for sid, result in self.results.items():
            output['sentences'][sid] = {
                'sentence': asdict(result.sentence),
                'n_lhs_trials': len(result.lhs_results),
                'n_control_trials': len(result.control_results),
                'lhs_mean': result.lhs_mean,
                'lhs_std': result.lhs_std,
                'control_mean': result.control_mean,
                'control_std': result.control_std,
                't_statistic': result.t_statistic,
                'p_value': result.p_value,
                'effect_size': result.effect_size,
                'supports_hypothesis': result.supports_hypothesis()
            }
        
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\nResults saved to: {filepath}")
        return filepath


# ============================================================================
# CLI
# ============================================================================

def main(argv: List[str] = None) -> int:
    parser = argparse.ArgumentParser(
        description='APL Seven Sentences Test Harness',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --sentence A3 --trials 50
  %(prog)s --all --trials 100 --seed 42
  %(prog)s --sentence A6 --trials 20 --output my_results/
        """
    )
    
    parser.add_argument(
        '--sentence', '-s',
        choices=list(APL_SENTENCES.keys()),
        help='Run test for specific sentence'
    )
    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='Run tests for all sentences'
    )
    parser.add_argument(
        '--trials', '-n',
        type=int,
        default=50,
        help='Number of trials per condition (default: 50)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        help='Base random seed for reproducibility'
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=Path('results'),
        help='Output directory (default: results/)'
    )
    parser.add_argument(
        '--duration', '-d',
        type=float,
        default=10.0,
        help='Simulation duration per trial (default: 10.0)'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all APL sentences and exit'
    )
    
    args = parser.parse_args(argv)
    
    if args.list:
        print("\nAPL Seven Sentences Test Pack:")
        print("-" * 60)
        for sid, sentence in APL_SENTENCES.items():
            print(f"  {sentence}")
        return 0
    
    if not args.sentence and not args.all:
        parser.error("Specify --sentence <ID> or --all")
    
    # Initialize harness
    harness = APLTestHarness(output_dir=args.output)
    
    # Run tests
    if args.all:
        harness.run_all_sentences(
            n_trials=args.trials,
            base_seed=args.seed,
            duration=args.duration
        )
    else:
        harness.run_sentence_test(
            args.sentence,
            n_trials=args.trials,
            base_seed=args.seed,
            duration=args.duration
        )
    
    # Save results
    harness.save_results()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
