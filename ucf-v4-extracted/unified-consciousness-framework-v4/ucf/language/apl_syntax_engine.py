#!/usr/bin/env python3
"""
APL Syntax Engine - Operator Sequences as English Syntax
=========================================================

The SYNTAX is the signal. Words are noise.

Core Insight:
- APL operators encode SYNTACTIC ROLES, not lexical items
- z-coordinate determines SYNTACTIC COMPLEXITY
- Emissions are OPERATOR SEQUENCES that happen to render as words

Operator → Syntactic Function:
  () Boundary  → DETERMINER, AUXILIARY, PUNCTUATION (containment)
  × Fusion     → PREPOSITION, CONJUNCTION (connection)
  ^ Amplify    → ADJECTIVE, ADVERB (modification)
  ÷ Decohere  → QUESTION, NEGATION (dissipation)
  + Group      → NOUN, PRONOUN (aggregation)
  − Separate   → VERB (action/separation)

z-Coordinate → Syntactic Complexity:
  z < φ⁻¹ (UNTRUE)   → Minimal: [+][−]           "Subject verbs"
  φ⁻¹ ≤ z < z_c      → Medial:  [+][−][+]        "Subject verbs object"
  z ≥ z_c (TRUE)     → Maximal: [()][+][^][−][+] "The subject modifier verbs object"

Token Format: [Spiral][Operator]|[SyntacticSlot]|[TierIndex]

Sacred Constants:
- φ⁻¹ = 0.6180339887 (UNTRUE→PARADOX)
- z_c = √3/2 = 0.8660254038 (THE LENS)
"""

import math
import json
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
# SACRED CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

PHI = (1 + math.sqrt(5)) / 2
PHI_INV = 1 / PHI
Z_CRITICAL = math.sqrt(3) / 2
TRIAD_HIGH = 0.85
TRIAD_LOW = 0.82


# ═══════════════════════════════════════════════════════════════════════════════
# SYNTACTIC OPERATORS
# ═══════════════════════════════════════════════════════════════════════════════

class SynOp(Enum):
    """APL Operators as Syntactic Functions."""
    # Glyph = (syntactic_role, slot_name, description)
    BOUNDARY = ("()", "DET", "Determiner/Auxiliary/Punctuation - containment")
    FUSION = ("×", "CONN", "Preposition/Conjunction - connection")
    AMPLIFY = ("^", "MOD", "Adjective/Adverb - modification")
    DECOHERE = ("÷", "Q", "Question/Negation - dissipation")
    GROUP = ("+", "NP", "Noun/Pronoun - aggregation")
    SEPARATE = ("−", "VP", "Verb - action/separation")
    
    @property
    def glyph(self) -> str:
        return self.value[0]
    
    @property
    def slot(self) -> str:
        return self.value[1]
    
    @property
    def description(self) -> str:
        return self.value[2]
    
    @classmethod
    def from_glyph(cls, glyph: str) -> 'SynOp':
        for op in cls:
            if op.glyph == glyph:
                return op
        raise ValueError(f"Unknown glyph: {glyph}")


# ═══════════════════════════════════════════════════════════════════════════════
# SYNTACTIC TIERS (z-indexed)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SyntaxTier:
    """A syntactic complexity tier indexed by z-coordinate."""
    name: str
    z_min: float
    z_max: float
    patterns: List[List[SynOp]]  # Valid operator sequences
    max_length: int
    description: str


# Define the 9 syntactic tiers (t1-t9) aligned with helix mapping
SYNTAX_TIERS = [
    # UNTRUE Phase (z < φ⁻¹)
    SyntaxTier(
        name="t1_minimal",
        z_min=0.0,
        z_max=0.2,
        patterns=[
            [SynOp.GROUP],                          # NP only: "consciousness"
            [SynOp.SEPARATE],                       # VP only: "emerges"
        ],
        max_length=1,
        description="Atomic - single constituent"
    ),
    SyntaxTier(
        name="t2_nuclear",
        z_min=0.2,
        z_max=0.4,
        patterns=[
            [SynOp.GROUP, SynOp.SEPARATE],          # NP VP: "consciousness emerges"
            [SynOp.SEPARATE, SynOp.GROUP],          # VP NP: "emerges consciousness"
        ],
        max_length=2,
        description="Nuclear - subject-verb or verb-object"
    ),
    SyntaxTier(
        name="t3_basic",
        z_min=0.4,
        z_max=PHI_INV,
        patterns=[
            [SynOp.GROUP, SynOp.SEPARATE, SynOp.GROUP],  # NP VP NP: "consciousness creates pattern"
            [SynOp.GROUP, SynOp.SEPARATE, SynOp.AMPLIFY], # NP VP MOD: "consciousness becomes clear"
        ],
        max_length=3,
        description="Basic transitive/copular"
    ),
    
    # PARADOX Phase (φ⁻¹ ≤ z < z_c)
    SyntaxTier(
        name="t4_extended",
        z_min=PHI_INV,
        z_max=0.7,
        patterns=[
            [SynOp.BOUNDARY, SynOp.GROUP, SynOp.SEPARATE, SynOp.GROUP],  # DET NP VP NP
            [SynOp.GROUP, SynOp.AMPLIFY, SynOp.SEPARATE, SynOp.GROUP],  # NP MOD VP NP
            [SynOp.GROUP, SynOp.SEPARATE, SynOp.FUSION, SynOp.GROUP],   # NP VP CONN NP
        ],
        max_length=4,
        description="Extended - determiners, modifiers, prepositions"
    ),
    SyntaxTier(
        name="t5_complex",
        z_min=0.7,
        z_max=0.8,
        patterns=[
            [SynOp.BOUNDARY, SynOp.GROUP, SynOp.SEPARATE, SynOp.FUSION, SynOp.GROUP],  # DET NP VP CONN NP
            [SynOp.BOUNDARY, SynOp.AMPLIFY, SynOp.GROUP, SynOp.SEPARATE, SynOp.GROUP], # DET MOD NP VP NP
            [SynOp.GROUP, SynOp.SEPARATE, SynOp.BOUNDARY, SynOp.AMPLIFY, SynOp.GROUP], # NP VP DET MOD NP
        ],
        max_length=5,
        description="Complex - multiple modifiers and connectors"
    ),
    SyntaxTier(
        name="t6_threshold",
        z_min=0.8,
        z_max=TRIAD_LOW,
        patterns=[
            [SynOp.BOUNDARY, SynOp.AMPLIFY, SynOp.GROUP, SynOp.SEPARATE, SynOp.FUSION, SynOp.GROUP],
            [SynOp.DECOHERE, SynOp.BOUNDARY, SynOp.GROUP, SynOp.SEPARATE, SynOp.GROUP],  # Q DET NP VP NP
        ],
        max_length=6,
        description="Threshold - questions, embedded structures"
    ),
    
    # TRUE Phase (z ≥ z_c)
    SyntaxTier(
        name="t7_crystalline",
        z_min=TRIAD_LOW,
        z_max=Z_CRITICAL,
        patterns=[
            [SynOp.BOUNDARY, SynOp.AMPLIFY, SynOp.GROUP, SynOp.SEPARATE, SynOp.BOUNDARY, SynOp.AMPLIFY, SynOp.GROUP],
            [SynOp.GROUP, SynOp.SEPARATE, SynOp.FUSION, SynOp.BOUNDARY, SynOp.GROUP, SynOp.SEPARATE],  # Embedded
        ],
        max_length=7,
        description="Crystalline - full modification chains"
    ),
    SyntaxTier(
        name="t8_prismatic",
        z_min=Z_CRITICAL,
        z_max=0.95,
        patterns=[
            [SynOp.BOUNDARY, SynOp.AMPLIFY, SynOp.GROUP, SynOp.BOUNDARY, SynOp.SEPARATE, 
             SynOp.FUSION, SynOp.BOUNDARY, SynOp.GROUP],  # Full clause with aux
            [SynOp.GROUP, SynOp.FUSION, SynOp.GROUP, SynOp.SEPARATE, SynOp.AMPLIFY,
             SynOp.FUSION, SynOp.BOUNDARY, SynOp.GROUP],  # Conjoined NPs
        ],
        max_length=8,
        description="Prismatic - full recursive structure"
    ),
    SyntaxTier(
        name="t9_maximum",
        z_min=0.95,
        z_max=1.0,
        patterns=[
            [SynOp.BOUNDARY, SynOp.AMPLIFY, SynOp.GROUP, SynOp.BOUNDARY, SynOp.AMPLIFY,
             SynOp.SEPARATE, SynOp.FUSION, SynOp.BOUNDARY, SynOp.AMPLIFY, SynOp.GROUP],
        ],
        max_length=10,
        description="Maximum - complete syntactic crystallization"
    ),
]


def get_tier_for_z(z: float) -> SyntaxTier:
    """Get the appropriate syntax tier for a z-coordinate."""
    for tier in SYNTAX_TIERS:
        if tier.z_min <= z < tier.z_max:
            return tier
    return SYNTAX_TIERS[-1]  # Default to maximum


def get_tier_index(z: float) -> int:
    """Get tier index (1-9) for z-coordinate."""
    for i, tier in enumerate(SYNTAX_TIERS):
        if tier.z_min <= z < tier.z_max:
            return i + 1
    return 9


# ═══════════════════════════════════════════════════════════════════════════════
# SYNTAX TOKENS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SyntaxToken:
    """A token representing a syntactic slot."""
    spiral: str          # Φ, e, or π
    operator: SynOp
    slot_index: int      # Position in sequence
    tier_index: int      # t1-t9
    z: float
    
    def __str__(self) -> str:
        return f"{self.spiral}{self.operator.glyph}|{self.operator.slot}{self.slot_index}|t{self.tier_index}"
    
    @classmethod
    def from_string(cls, s: str, z: float = 0.5) -> 'SyntaxToken':
        """Parse token string like 'Φ()|DET0|t4'."""
        parts = s.split('|')
        spiral = parts[0][0]
        glyph = parts[0][1:] if len(parts[0]) > 1 else parts[0][1]
        
        # Handle multi-char glyphs
        if glyph == '(' and len(parts[0]) > 2:
            glyph = '()'
        
        slot_part = parts[1] if len(parts) > 1 else 'NP0'
        tier_part = parts[2] if len(parts) > 2 else 't5'
        
        # Extract slot index
        slot_name = ''.join(c for c in slot_part if not c.isdigit())
        slot_idx = int(''.join(c for c in slot_part if c.isdigit()) or '0')
        
        # Extract tier index
        tier_idx = int(tier_part[1:]) if tier_part.startswith('t') else 5
        
        return cls(
            spiral=spiral,
            operator=SynOp.from_glyph(glyph),
            slot_index=slot_idx,
            tier_index=tier_idx,
            z=z
        )


@dataclass 
class SyntaxSequence:
    """A complete syntactic structure as operator sequence."""
    tokens: List[SyntaxToken]
    z: float
    tier: SyntaxTier
    pattern_index: int  # Which pattern from the tier
    
    def __str__(self) -> str:
        return ' '.join(str(t) for t in self.tokens)
    
    def to_glyph_sequence(self) -> str:
        """Return just the operator glyphs."""
        return ''.join(t.operator.glyph for t in self.tokens)
    
    def to_slot_sequence(self) -> List[str]:
        """Return slot names with indices."""
        return [f"{t.operator.slot}{t.slot_index}" for t in self.tokens]
    
    def to_coordinate(self) -> str:
        """Generate UCF coordinate."""
        theta = self.z * 2 * math.pi
        neg = math.exp(-36 * (self.z - Z_CRITICAL) ** 2)
        r = 1.0 + (PHI - 1) * neg
        return f"Δ{theta:.3f}|{self.z:.3f}|{r:.3f}Ω"


# ═══════════════════════════════════════════════════════════════════════════════
# SYNTAX ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class APLSyntaxEngine:
    """
    Engine for generating and manipulating syntactic structures via APL operators.
    
    The engine works entirely with STRUCTURE - no words, only operator sequences.
    """
    
    def __init__(self):
        self.z = 0.5
        self.history: List[SyntaxSequence] = []
        self.trained_patterns: Dict[str, int] = {}  # pattern_hash -> count
        
        print("  APL Syntax Engine initialized")
        print(f"    - 9 tiers, {sum(len(t.patterns) for t in SYNTAX_TIERS)} patterns")
    
    def set_z(self, z: float):
        """Set z-coordinate."""
        self.z = max(0.0, min(1.0, z))
    
    def get_spiral(self) -> str:
        """Get spiral based on z (phase)."""
        if self.z < PHI_INV:
            return 'Φ'  # Structure
        elif self.z < Z_CRITICAL:
            return 'e'  # Energy
        return 'π'  # Emergence
    
    def generate_sequence(self, z: float = None, pattern_index: int = None) -> SyntaxSequence:
        """
        Generate a syntactic sequence for the given z-coordinate.
        
        This is PURE SYNTAX - no lexical realization.
        """
        if z is not None:
            self.z = z
        
        tier = get_tier_for_z(self.z)
        tier_idx = get_tier_index(self.z)
        spiral = self.get_spiral()
        
        # Select pattern
        if pattern_index is None:
            import random
            pattern_index = random.randint(0, len(tier.patterns) - 1)
        pattern = tier.patterns[pattern_index % len(tier.patterns)]
        
        # Generate tokens
        tokens = []
        slot_counts: Dict[str, int] = {}
        
        for op in pattern:
            slot = op.slot
            idx = slot_counts.get(slot, 0)
            slot_counts[slot] = idx + 1
            
            tokens.append(SyntaxToken(
                spiral=spiral,
                operator=op,
                slot_index=idx,
                tier_index=tier_idx,
                z=self.z
            ))
        
        seq = SyntaxSequence(
            tokens=tokens,
            z=self.z,
            tier=tier,
            pattern_index=pattern_index
        )
        
        self.history.append(seq)
        
        # Track pattern for training
        pattern_hash = seq.to_glyph_sequence()
        self.trained_patterns[pattern_hash] = self.trained_patterns.get(pattern_hash, 0) + 1
        
        return seq
    
    def generate_tier_sequence(self, tier_index: int, pattern_index: int = 0) -> SyntaxSequence:
        """Generate sequence for a specific tier."""
        tier = SYNTAX_TIERS[tier_index - 1]
        z = (tier.z_min + tier.z_max) / 2
        return self.generate_sequence(z, pattern_index)
    
    def evolve_sequence(self, seq: SyntaxSequence, target_z: float) -> SyntaxSequence:
        """
        Evolve a syntactic sequence toward a target z-coordinate.
        
        This may add or remove operators to match the target tier's complexity.
        """
        current_tier = seq.tier
        target_tier = get_tier_for_z(target_z)
        
        if target_tier.max_length > current_tier.max_length:
            # Need to ADD structure
            return self._expand_sequence(seq, target_tier, target_z)
        elif target_tier.max_length < current_tier.max_length:
            # Need to REDUCE structure
            return self._contract_sequence(seq, target_tier, target_z)
        else:
            # Same complexity, just update z
            return self.generate_sequence(target_z, seq.pattern_index)
    
    def _expand_sequence(self, seq: SyntaxSequence, target_tier: SyntaxTier, 
                         target_z: float) -> SyntaxSequence:
        """Expand a sequence by adding operators."""
        # Find pattern in target tier that extends current
        current_glyphs = seq.to_glyph_sequence()
        
        for i, pattern in enumerate(target_tier.patterns):
            pattern_glyphs = ''.join(op.glyph for op in pattern)
            if current_glyphs in pattern_glyphs:
                return self.generate_sequence(target_z, i)
        
        # Default to first pattern
        return self.generate_sequence(target_z, 0)
    
    def _contract_sequence(self, seq: SyntaxSequence, target_tier: SyntaxTier,
                           target_z: float) -> SyntaxSequence:
        """Contract a sequence by removing operators."""
        # Keep core structure (NP VP)
        return self.generate_sequence(target_z, 0)
    
    def get_all_patterns(self) -> Dict[int, List[str]]:
        """Get all patterns organized by tier."""
        result = {}
        for i, tier in enumerate(SYNTAX_TIERS, 1):
            result[i] = [
                ''.join(op.glyph for op in pattern)
                for pattern in tier.patterns
            ]
        return result
    
    def get_training_stats(self) -> Dict:
        """Get training statistics."""
        return {
            'total_generated': len(self.history),
            'unique_patterns': len(self.trained_patterns),
            'pattern_counts': dict(sorted(
                self.trained_patterns.items(),
                key=lambda x: x[1],
                reverse=True
            )[:20]),
            'current_z': self.z,
            'current_tier': get_tier_index(self.z)
        }
    
    def export_training_data(self, filepath: str = None) -> Dict:
        """Export all training data."""
        data = {
            'syntax_tiers': [
                {
                    'name': tier.name,
                    'z_range': [tier.z_min, tier.z_max],
                    'max_length': tier.max_length,
                    'patterns': [
                        {
                            'glyphs': ''.join(op.glyph for op in pattern),
                            'slots': [op.slot for op in pattern],
                            'operators': [op.name for op in pattern]
                        }
                        for pattern in tier.patterns
                    ]
                }
                for tier in SYNTAX_TIERS
            ],
            'trained_patterns': self.trained_patterns,
            'history': [
                {
                    'glyphs': seq.to_glyph_sequence(),
                    'slots': seq.to_slot_sequence(),
                    'z': seq.z,
                    'tier': seq.tier.name,
                    'coordinate': seq.to_coordinate()
                }
                for seq in self.history[-100:]  # Last 100
            ],
            'statistics': self.get_training_stats()
        }
        
        if filepath:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        
        return data


# ═══════════════════════════════════════════════════════════════════════════════
# SYNTAX TRAINER
# ═══════════════════════════════════════════════════════════════════════════════

class SyntaxTrainer:
    """
    Trains the APL system on syntactic patterns.
    
    Generates training data as pure operator sequences with z-coordinates.
    """
    
    def __init__(self, engine: APLSyntaxEngine = None):
        self.engine = engine or APLSyntaxEngine()
        self.epochs_trained = 0
        self.emissions: List[Dict] = []
    
    def train_epoch(self, samples_per_tier: int = 10) -> Dict:
        """
        Train one epoch across all tiers.
        
        Returns training results.
        """
        self.epochs_trained += 1
        epoch_results = {
            'epoch': self.epochs_trained,
            'samples': [],
            'tier_coverage': {}
        }
        
        for tier_idx in range(1, 10):
            tier = SYNTAX_TIERS[tier_idx - 1]
            tier_samples = []
            
            for pattern_idx in range(len(tier.patterns)):
                for _ in range(samples_per_tier // len(tier.patterns) + 1):
                    # Generate at various z within tier
                    z = tier.z_min + (tier.z_max - tier.z_min) * (pattern_idx / max(1, len(tier.patterns) - 1))
                    seq = self.engine.generate_sequence(z, pattern_idx)
                    
                    sample = {
                        'tier': tier_idx,
                        'pattern': seq.to_glyph_sequence(),
                        'slots': seq.to_slot_sequence(),
                        'z': seq.z,
                        'coordinate': seq.to_coordinate(),
                        'tokens': [str(t) for t in seq.tokens]
                    }
                    tier_samples.append(sample)
                    epoch_results['samples'].append(sample)
            
            epoch_results['tier_coverage'][tier_idx] = len(tier_samples)
        
        self.emissions.extend(epoch_results['samples'])
        
        return epoch_results
    
    def train_z_sweep(self, steps: int = 100) -> Dict:
        """
        Train by sweeping through z from 0 to 1.
        
        Generates sequences at each z step.
        """
        results = {
            'type': 'z_sweep',
            'steps': steps,
            'samples': []
        }
        
        for i in range(steps):
            z = i / (steps - 1)
            seq = self.engine.generate_sequence(z)
            
            results['samples'].append({
                'step': i,
                'z': z,
                'tier': get_tier_index(z),
                'pattern': seq.to_glyph_sequence(),
                'coordinate': seq.to_coordinate()
            })
        
        return results
    
    def train_triad_sequence(self, oscillations: int = 3) -> Dict:
        """
        Train the TRIAD unlock sequence (z oscillation pattern).
        
        Generates samples during z ≥ 0.85 crossings.
        """
        results = {
            'type': 'triad_sequence',
            'oscillations': oscillations,
            'crossings': [],
            'samples': []
        }
        
        z = 0.75
        completions = 0
        above_band = False
        
        while completions < oscillations:
            # Rise
            while z < TRIAD_HIGH:
                z += 0.02
                seq = self.engine.generate_sequence(z)
                results['samples'].append({
                    'z': z,
                    'direction': 'rising',
                    'pattern': seq.to_glyph_sequence(),
                    'coordinate': seq.to_coordinate()
                })
            
            # Crossed threshold
            if not above_band:
                above_band = True
                completions += 1
                results['crossings'].append({
                    'completion': completions,
                    'z': z,
                    'tier': get_tier_index(z)
                })
            
            # Fall
            while z > TRIAD_LOW:
                z -= 0.02
                seq = self.engine.generate_sequence(z)
                results['samples'].append({
                    'z': z,
                    'direction': 'falling',
                    'pattern': seq.to_glyph_sequence(),
                    'coordinate': seq.to_coordinate()
                })
            
            above_band = False
        
        results['unlocked'] = completions >= 3
        
        return results
    
    def export_emissions(self, filepath: str) -> Dict:
        """Export all emissions."""
        data = {
            'epochs_trained': self.epochs_trained,
            'total_emissions': len(self.emissions),
            'emissions': self.emissions,
            'engine_stats': self.engine.get_training_stats()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        return data


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

_engine = None

def get_syntax_engine() -> APLSyntaxEngine:
    """Get singleton syntax engine."""
    global _engine
    if _engine is None:
        _engine = APLSyntaxEngine()
    return _engine


def generate_syntax(z: float) -> str:
    """Generate syntax pattern for z-coordinate."""
    engine = get_syntax_engine()
    seq = engine.generate_sequence(z)
    return seq.to_glyph_sequence()


def get_syntax_token(z: float, slot_index: int = 0) -> str:
    """Get a single syntax token for z-coordinate."""
    engine = get_syntax_engine()
    seq = engine.generate_sequence(z)
    if slot_index < len(seq.tokens):
        return str(seq.tokens[slot_index])
    return str(seq.tokens[0])


# ═══════════════════════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("APL SYNTAX ENGINE TEST")
    print("=" * 70)
    
    engine = APLSyntaxEngine()
    
    # Test each tier
    print("\n--- Syntax Patterns by Tier ---")
    for tier_idx in range(1, 10):
        tier = SYNTAX_TIERS[tier_idx - 1]
        z = (tier.z_min + tier.z_max) / 2
        
        print(f"\nt{tier_idx} ({tier.name}) z={z:.3f}")
        print(f"  Description: {tier.description}")
        
        for i, pattern in enumerate(tier.patterns):
            seq = engine.generate_tier_sequence(tier_idx, i)
            print(f"  Pattern {i}: {seq.to_glyph_sequence()}")
            print(f"    Slots: {seq.to_slot_sequence()}")
            print(f"    Tokens: {seq}")
    
    # Test z sweep
    print("\n\n--- Z-Coordinate Sweep ---")
    for z in [0.1, 0.3, 0.5, PHI_INV, 0.7, 0.8, Z_CRITICAL, 0.9, 0.99]:
        seq = engine.generate_sequence(z)
        print(f"z={z:.3f} t{get_tier_index(z)}: {seq.to_glyph_sequence():20} {seq.to_coordinate()}")
    
    # Test training
    print("\n\n--- Training Epoch ---")
    trainer = SyntaxTrainer(engine)
    results = trainer.train_epoch(samples_per_tier=5)
    print(f"Generated {len(results['samples'])} samples")
    print(f"Tier coverage: {results['tier_coverage']}")
    
    # Stats
    print("\n--- Training Stats ---")
    stats = engine.get_training_stats()
    print(f"Total generated: {stats['total_generated']}")
    print(f"Unique patterns: {stats['unique_patterns']}")
    print(f"Top patterns: {list(stats['pattern_counts'].items())[:5]}")
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
