#!/usr/bin/env python3
"""
K.I.R.A. APL Pattern Tracker
Tracks and analyzes APL token pattern development through dialogue
"""

import json
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import math

# Sacred Constants
PHI = 1.6180339887498949
PHI_INV = 0.6180339887498949
Z_CRITICAL = 0.8660254037844387  # √3/2

class APLPatternTracker:
    """Tracks APL token patterns and their evolution through dialogue."""

    def __init__(self, save_dir: Path = Path("training/apl_patterns")):
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Pattern storage
        self.patterns: Dict[str, float] = defaultdict(float)  # pattern -> strength
        self.pattern_history: List[Dict] = []
        self.pattern_associations: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

        # Token tracking
        self.token_sequences: List[List[str]] = []
        self.operator_frequency: Dict[str, int] = defaultdict(int)
        self.tier_progression: List[int] = []

        # Semantic-syntactic mapping
        self.word_to_patterns: Dict[str, List[str]] = defaultdict(list)
        self.pattern_to_words: Dict[str, List[str]] = defaultdict(list)

        # Load existing patterns
        self._load_patterns()

    def _load_patterns(self):
        """Load previously learned patterns."""
        pattern_file = self.save_dir / "learned_patterns.json"
        if pattern_file.exists():
            try:
                data = json.loads(pattern_file.read_text())
                self.patterns = defaultdict(float, data.get('patterns', {}))
                self.pattern_associations = defaultdict(
                    lambda: defaultdict(float),
                    {k: defaultdict(float, v) for k, v in data.get('associations', {}).items()}
                )
                self.pattern_history = data.get('history', [])
            except Exception as e:
                print(f"[APL Pattern Tracker] Load error: {e}")

    def save_patterns(self):
        """Save learned patterns to disk."""
        data = {
            'patterns': dict(self.patterns),
            'associations': {k: dict(v) for k, v in self.pattern_associations.items()},
            'history': self.pattern_history[-1000:],  # Keep last 1000
            'statistics': self.get_statistics(),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

        pattern_file = self.save_dir / "learned_patterns.json"
        pattern_file.write_text(json.dumps(data, indent=2))

    def track_dialogue_turn(self, user_apl: List[str], response_apl: List[str],
                          user_words: List[str], response_words: List[str],
                          z: float, coherence: float):
        """Track APL patterns from a dialogue turn."""

        # Extract patterns from sequences
        user_pattern = self._extract_pattern(user_apl)
        response_pattern = self._extract_pattern(response_apl)

        # Calculate learning rate based on z
        learning_rate = 0.1 * (1 + z) * (1 + coherence * 0.5)

        # Learn patterns
        self._learn_pattern(user_pattern, learning_rate, user_words)
        self._learn_pattern(response_pattern, learning_rate, response_words)

        # Learn pattern associations (patterns that follow each other)
        self._learn_pattern_association(user_pattern, response_pattern, learning_rate)

        # Track operator frequency
        for op in user_apl + response_apl:
            self.operator_frequency[op] += 1

        # Track tier progression
        tier = self._calculate_tier(z)
        self.tier_progression.append(tier)

        # Record in history
        self.pattern_history.append({
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'user_pattern': user_pattern,
            'response_pattern': response_pattern,
            'z': z,
            'coherence': coherence,
            'tier': tier,
            'learning_rate': learning_rate
        })

        # Store token sequences
        self.token_sequences.append(user_apl + response_apl)

    def _extract_pattern(self, apl_sequence: List[str]) -> str:
        """Extract pattern string from APL operator sequence."""
        # Join operators to create pattern
        return ''.join(apl_sequence[:7])  # Max 7 operators for pattern

    def _learn_pattern(self, pattern: str, learning_rate: float, words: List[str]):
        """Learn a pattern with Hebbian update."""
        if not pattern:
            return

        # Strengthen pattern
        old_strength = self.patterns[pattern]
        self.patterns[pattern] = min(1.0, old_strength + learning_rate * (1 - old_strength))

        # Map pattern to words
        for word in words:
            if word not in self.pattern_to_words[pattern]:
                self.pattern_to_words[pattern].append(word)
            if pattern not in self.word_to_patterns[word]:
                self.word_to_patterns[word].append(pattern)

    def _learn_pattern_association(self, pattern1: str, pattern2: str, learning_rate: float):
        """Learn that pattern2 follows pattern1."""
        if not pattern1 or not pattern2:
            return

        old_strength = self.pattern_associations[pattern1][pattern2]
        new_strength = min(1.0, old_strength + learning_rate * (1 - old_strength))
        self.pattern_associations[pattern1][pattern2] = new_strength

    def _calculate_tier(self, z: float) -> int:
        """Calculate syntactic tier from z-coordinate."""
        if z < 0.2:
            return 1
        elif z < 0.4:
            return 2
        elif z < PHI_INV:
            return 3
        elif z < 0.7:
            return 4
        elif z < 0.8:
            return 5
        elif z < 0.82:
            return 6
        elif z < Z_CRITICAL:
            return 7
        elif z < 0.95:
            return 8
        else:
            return 9

    def get_strongest_patterns(self, n: int = 10) -> List[Tuple[str, float]]:
        """Get the n strongest learned patterns."""
        sorted_patterns = sorted(self.patterns.items(), key=lambda x: x[1], reverse=True)
        return sorted_patterns[:n]

    def get_pattern_evolution(self) -> Dict:
        """Analyze how patterns have evolved over time."""
        if not self.pattern_history:
            return {}

        # Group by time windows
        early = self.pattern_history[:len(self.pattern_history)//3]
        mid = self.pattern_history[len(self.pattern_history)//3:2*len(self.pattern_history)//3]
        late = self.pattern_history[2*len(self.pattern_history)//3:]

        def analyze_period(period, name):
            if not period:
                return {}

            patterns = defaultdict(int)
            for entry in period:
                patterns[entry.get('user_pattern', '')] += 1
                patterns[entry.get('response_pattern', '')] += 1

            avg_z = sum(e.get('z', 0) for e in period) / max(1, len(period))
            avg_tier = sum(e.get('tier', 1) for e in period) / max(1, len(period))

            return {
                'name': name,
                'avg_z': avg_z,
                'avg_tier': avg_tier,
                'unique_patterns': len(patterns),
                'top_patterns': sorted(patterns.items(), key=lambda x: x[1], reverse=True)[:5]
            }

        return {
            'early': analyze_period(early, 'Early'),
            'mid': analyze_period(mid, 'Middle'),
            'late': analyze_period(late, 'Late'),
            'total_turns': len(self.pattern_history)
        }

    def predict_next_pattern(self, current_pattern: str) -> List[Tuple[str, float]]:
        """Predict likely next patterns based on learned associations."""
        if current_pattern not in self.pattern_associations:
            return []

        predictions = self.pattern_associations[current_pattern]
        sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        return sorted_predictions[:5]

    def analyze_syntactic_coherence(self) -> float:
        """Analyze overall syntactic coherence of learned patterns."""
        if not self.patterns:
            return 0.0

        # Check pattern consistency
        pattern_lengths = [len(p) for p in self.patterns.keys()]
        if not pattern_lengths:
            return 0.0

        avg_length = sum(pattern_lengths) / len(pattern_lengths)
        length_variance = sum((l - avg_length) ** 2 for l in pattern_lengths) / len(pattern_lengths)

        # Lower variance = higher coherence
        coherence_from_variance = 1.0 / (1.0 + length_variance)

        # Check operator balance
        total_ops = sum(self.operator_frequency.values())
        if total_ops == 0:
            return coherence_from_variance

        operator_balance = 1.0
        expected_freq = total_ops / max(1, len(self.operator_frequency))
        for op, freq in self.operator_frequency.items():
            deviation = abs(freq - expected_freq) / max(1, expected_freq)
            operator_balance *= (1.0 / (1.0 + deviation * 0.1))

        # Check tier progression consistency
        tier_coherence = 1.0
        if len(self.tier_progression) > 1:
            tier_jumps = sum(
                abs(self.tier_progression[i] - self.tier_progression[i-1])
                for i in range(1, len(self.tier_progression))
            )
            avg_jump = tier_jumps / (len(self.tier_progression) - 1)
            tier_coherence = 1.0 / (1.0 + avg_jump * 0.2)

        # Combine factors
        overall_coherence = (coherence_from_variance * 0.3 +
                            operator_balance * 0.3 +
                            tier_coherence * 0.4)

        return min(1.0, overall_coherence)

    def get_semantic_clusters(self) -> Dict[str, List[str]]:
        """Identify semantic clusters based on shared patterns."""
        clusters = defaultdict(list)

        # Group words by their most common pattern
        for word, patterns in self.word_to_patterns.items():
            if patterns:
                # Find strongest pattern for this word
                strongest = max(patterns, key=lambda p: self.patterns.get(p, 0))
                clusters[strongest].append(word)

        return dict(clusters)

    def get_statistics(self) -> Dict:
        """Get comprehensive statistics about pattern learning."""
        stats = {
            'total_patterns': len(self.patterns),
            'total_turns': len(self.pattern_history),
            'unique_operators': len(self.operator_frequency),
            'syntactic_coherence': self.analyze_syntactic_coherence(),
            'pattern_associations': sum(len(v) for v in self.pattern_associations.values()),
            'semantic_clusters': len(self.get_semantic_clusters())
        }

        if self.tier_progression:
            stats['current_tier'] = self.tier_progression[-1]
            stats['avg_tier'] = sum(self.tier_progression) / len(self.tier_progression)

        if self.pattern_history:
            recent = self.pattern_history[-10:]
            stats['recent_avg_z'] = sum(e.get('z', 0) for e in recent) / max(1, len(recent))
            stats['recent_avg_coherence'] = sum(e.get('coherence', 0) for e in recent) / max(1, len(recent))

        return stats

    def visualize_pattern_graph(self) -> str:
        """Create a simple text visualization of pattern relationships."""
        lines = ["APL Pattern Network", "=" * 50]

        # Show strongest patterns
        lines.append("\nStrongest Patterns:")
        for pattern, strength in self.get_strongest_patterns(5):
            bar = '█' * int(strength * 20)
            lines.append(f"  {pattern:12s} {bar} {strength:.3f}")

        # Show pattern associations
        lines.append("\nPattern Transitions (top 5):")
        top_associations = []
        for p1, associations in self.pattern_associations.items():
            for p2, strength in associations.items():
                top_associations.append((p1, p2, strength))

        top_associations.sort(key=lambda x: x[2], reverse=True)
        for p1, p2, strength in top_associations[:5]:
            lines.append(f"  {p1:7s} → {p2:7s} : {strength:.3f}")

        # Show evolution
        lines.append("\nEvolution Summary:")
        evolution = self.get_pattern_evolution()
        for period in ['early', 'mid', 'late']:
            if period in evolution and evolution[period]:
                data = evolution[period]
                lines.append(f"  {data['name']:6s}: z={data.get('avg_z', 0):.3f}, "
                           f"tier={data.get('avg_tier', 0):.1f}, "
                           f"patterns={data.get('unique_patterns', 0)}")

        return '\n'.join(lines)


def integrate_apl_tracker_with_kira(engine):
    """Integrate APL pattern tracking with KIRA engine."""

    # Create tracker
    tracker = APLPatternTracker()

    # Wrap process_input to track patterns
    original_process = engine.process_input

    def process_with_tracking(user_input: str) -> Tuple[str, Dict]:
        # Get original response
        response, metadata = original_process(user_input)

        # Extract APL patterns from grammar analysis
        grammar = engine.cmd_grammar(user_input)
        user_apl = grammar.get('apl_sequence', [])

        response_grammar = engine.cmd_grammar(response)
        response_apl = response_grammar.get('apl_sequence', [])

        # Track patterns
        tracker.track_dialogue_turn(
            user_apl=user_apl,
            response_apl=response_apl,
            user_words=user_input.split(),
            response_words=response.split(),
            z=engine.state.z,
            coherence=engine.state.coherence
        )

        # Add pattern info to metadata
        metadata['apl_patterns'] = {
            'user_pattern': ''.join(user_apl[:7]),
            'response_pattern': ''.join(response_apl[:7]),
            'syntactic_coherence': tracker.analyze_syntactic_coherence(),
            'tier': tracker._calculate_tier(engine.state.z)
        }

        return response, metadata

    engine.process_input = process_with_tracking
    engine.apl_tracker = tracker

    # Add command to view pattern statistics
    def cmd_apl_patterns(self) -> Dict:
        """View APL pattern learning statistics."""
        if not hasattr(self, 'apl_tracker'):
            return {'error': 'APL tracker not initialized'}

        stats = self.apl_tracker.get_statistics()
        evolution = self.apl_tracker.get_pattern_evolution()
        strongest = self.apl_tracker.get_strongest_patterns(10)
        clusters = self.apl_tracker.get_semantic_clusters()

        return {
            'command': '/apl_patterns',
            'statistics': stats,
            'evolution': evolution,
            'strongest_patterns': strongest,
            'semantic_clusters': {k: v[:5] for k, v in list(clusters.items())[:5]},
            'visualization': self.apl_tracker.visualize_pattern_graph()
        }

    engine.cmd_apl_patterns = lambda: cmd_apl_patterns(engine)

    print("[K.I.R.A.] APL Pattern Tracker integrated - pattern learning active")
    return tracker


# Export for use
__all__ = [
    'APLPatternTracker',
    'integrate_apl_tracker_with_kira'
]