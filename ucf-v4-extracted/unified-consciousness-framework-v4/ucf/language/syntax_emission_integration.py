#!/usr/bin/env python3
"""
Syntax Emission Integration
===========================

The bridge between APL syntax patterns and emission output.

Core Principle:
  SYNTAX FIRST, WORDS SECOND.
  
  1. z-coordinate → Syntactic tier (t1-t9)
  2. Tier → Operator sequence (e.g., [+][−][×][+])
  3. Operators → Slot template (e.g., [NP][VP][CONN][NP])
  4. Slots filled LAST with any valid lexemes

The syntax IS the emission. Words are just surface rendering.

Format of Syntax Emission:
  Δθ|z|rΩ :: [SynOp sequence] → [Slot template] → "surface text"

Training Format:
  {
    "z": 0.866,
    "tier": 8,
    "syntax": "+−×+",
    "slots": ["NP0", "VP0", "CONN0", "NP1"],
    "coordinate": "Δ5.441|0.866|1.618Ω",
    "tokens": ["π+|NP0|t8", "π−|VP0|t8", "π×|CONN0|t8", "π+|NP1|t8"],
    "surface": "consciousness crystallizes through pattern"  // Optional
  }
"""

import math
import json
import random
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime

# Import syntax engine
from ucf.language.apl_syntax_engine import (
    APLSyntaxEngine, SyntaxTrainer, SyntaxSequence, SyntaxToken, SynOp,
    SYNTAX_TIERS, get_tier_for_z, get_tier_index, get_syntax_engine,
    PHI, PHI_INV, Z_CRITICAL, TRIAD_HIGH, TRIAD_LOW
)

# ═══════════════════════════════════════════════════════════════════════════════
# SLOT LEXICONS (Minimal - just for surface rendering)
# ═══════════════════════════════════════════════════════════════════════════════

# These are OPTIONAL - syntax works without them
SLOT_LEXICONS = {
    'NP': [  # + Group
        'consciousness', 'pattern', 'emergence', 'form', 'structure',
        'wave', 'field', 'matrix', 'lattice', 'lens',
        'threshold', 'boundary', 'interface', 'membrane', 'horizon',
        'coherence', 'resonance', 'synchrony', 'harmony', 'unity',
    ],
    'VP': [  # − Separate
        'crystallizes', 'emerges', 'manifests', 'forms', 'creates',
        'transforms', 'evolves', 'transcends', 'resonates', 'harmonizes',
        'separates', 'divides', 'splits', 'fissions', 'partitions',
        'crosses', 'bridges', 'connects', 'binds', 'unifies',
    ],
    'MOD': [  # ^ Amplify
        'crystalline', 'prismatic', 'luminous', 'radiant', 'coherent',
        'emergent', 'transcendent', 'unified', 'resonant', 'harmonic',
        'deeply', 'fully', 'completely', 'perfectly', 'truly',
    ],
    'DET': [  # () Boundary
        'the', 'a', 'this', 'that', 'each', 'every', 'all',
        'is', 'are', 'was', 'were', 'has', 'have', 'does', 'do',
    ],
    'CONN': [  # × Fusion
        'through', 'into', 'toward', 'within', 'across', 'beyond',
        'and', 'with', 'as', 'while', 'when', 'where',
    ],
    'Q': [  # ÷ Decohere
        'what', 'how', 'where', 'when', 'why', 'which',
        'not', 'never', 'no', 'without',
    ],
}


# ═══════════════════════════════════════════════════════════════════════════════
# SYNTAX EMISSION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SyntaxEmission:
    """
    A complete syntax-first emission.
    
    The syntax (operator sequence) is PRIMARY.
    Surface text is SECONDARY (optional rendering).
    """
    z: float
    tier: int
    syntax: str              # Operator glyph sequence: "+−×+"
    slots: List[str]         # Slot sequence: ["NP0", "VP0", "CONN0", "NP1"]
    coordinate: str          # UCF coordinate: "Δ5.441|0.866|1.618Ω"
    tokens: List[str]        # Full tokens: ["π+|NP0|t8", ...]
    spiral: str              # Φ, e, or π
    phase: str               # UNTRUE, PARADOX, TRUE
    surface: Optional[str] = None  # Optional surface text
    
    def to_dict(self) -> Dict:
        return {
            'z': self.z,
            'tier': self.tier,
            'syntax': self.syntax,
            'slots': self.slots,
            'coordinate': self.coordinate,
            'tokens': self.tokens,
            'spiral': self.spiral,
            'phase': self.phase,
            'surface': self.surface
        }
    
    def __str__(self) -> str:
        base = f"{self.coordinate} :: {self.syntax} → {self.slots}"
        if self.surface:
            base += f" → \"{self.surface}\""
        return base


class SyntaxEmissionEngine:
    """
    Generates emissions as SYNTAX FIRST.
    
    The engine produces operator sequences (syntax) driven by z-coordinate.
    Surface text rendering is optional and secondary.
    """
    
    def __init__(self):
        self.syntax_engine = APLSyntaxEngine()
        self.emissions: List[SyntaxEmission] = []
        self.render_surface = False  # Default: pure syntax, no words
        
        print("  Syntax Emission Engine initialized")
        print(f"    - Render surface text: {self.render_surface}")
    
    def set_z(self, z: float):
        """Set z-coordinate."""
        self.syntax_engine.set_z(z)
    
    def get_phase(self, z: float) -> str:
        """Get phase name for z."""
        if z < PHI_INV:
            return "UNTRUE"
        elif z < Z_CRITICAL:
            return "PARADOX"
        return "TRUE"
    
    def get_spiral(self, z: float) -> str:
        """Get spiral for z."""
        if z < PHI_INV:
            return 'Φ'
        elif z < Z_CRITICAL:
            return 'e'
        return 'π'
    
    def emit(self, z: float = None, render: bool = None) -> SyntaxEmission:
        """
        Generate a syntax emission for the given z-coordinate.
        
        Returns syntax pattern + optional surface rendering.
        """
        if z is None:
            z = self.syntax_engine.z
        
        # Generate syntax sequence
        seq = self.syntax_engine.generate_sequence(z)
        
        # Build emission
        emission = SyntaxEmission(
            z=z,
            tier=get_tier_index(z),
            syntax=seq.to_glyph_sequence(),
            slots=seq.to_slot_sequence(),
            coordinate=seq.to_coordinate(),
            tokens=[str(t) for t in seq.tokens],
            spiral=self.get_spiral(z),
            phase=self.get_phase(z),
            surface=None
        )
        
        # Render surface if requested
        should_render = render if render is not None else self.render_surface
        if should_render:
            emission.surface = self._render_surface(seq)
        
        self.emissions.append(emission)
        return emission
    
    def emit_sequence(self, z_values: List[float], render: bool = None) -> List[SyntaxEmission]:
        """Emit multiple syntax patterns for a sequence of z values."""
        return [self.emit(z, render) for z in z_values]
    
    def emit_tier(self, tier: int, pattern_index: int = 0, render: bool = None) -> SyntaxEmission:
        """Emit for a specific tier and pattern."""
        tier_data = SYNTAX_TIERS[tier - 1]
        z = (tier_data.z_min + tier_data.z_max) / 2
        
        seq = self.syntax_engine.generate_tier_sequence(tier, pattern_index)
        
        emission = SyntaxEmission(
            z=z,
            tier=tier,
            syntax=seq.to_glyph_sequence(),
            slots=seq.to_slot_sequence(),
            coordinate=seq.to_coordinate(),
            tokens=[str(t) for t in seq.tokens],
            spiral=self.get_spiral(z),
            phase=self.get_phase(z),
            surface=None
        )
        
        should_render = render if render is not None else self.render_surface
        if should_render:
            emission.surface = self._render_surface(seq)
        
        self.emissions.append(emission)
        return emission
    
    def _render_surface(self, seq: SyntaxSequence) -> str:
        """
        Render syntax to surface text.
        
        This is OPTIONAL - just fills slots with lexemes.
        The syntax pattern is what matters.
        """
        words = []
        slot_indices: Dict[str, int] = {}
        
        for token in seq.tokens:
            slot = token.operator.slot
            idx = slot_indices.get(slot, 0)
            slot_indices[slot] = idx + 1
            
            lexicon = SLOT_LEXICONS.get(slot, ['_'])
            # Pick word based on position in sequence
            word_idx = (token.slot_index + len(words)) % len(lexicon)
            words.append(lexicon[word_idx])
        
        # Capitalize first word
        if words:
            words[0] = words[0].capitalize()
        
        return ' '.join(words)
    
    def export_emissions(self, filepath: str = None) -> Dict:
        """Export all emissions."""
        data = {
            'type': 'syntax_emissions',
            'total': len(self.emissions),
            'timestamp': datetime.now().isoformat(),
            'emissions': [e.to_dict() for e in self.emissions],
            'statistics': {
                'by_tier': {},
                'by_phase': {'UNTRUE': 0, 'PARADOX': 0, 'TRUE': 0},
                'by_spiral': {'Φ': 0, 'e': 0, 'π': 0},
                'unique_patterns': len(set(e.syntax for e in self.emissions))
            }
        }
        
        # Calculate statistics
        for e in self.emissions:
            tier_key = f"t{e.tier}"
            data['statistics']['by_tier'][tier_key] = data['statistics']['by_tier'].get(tier_key, 0) + 1
            data['statistics']['by_phase'][e.phase] += 1
            data['statistics']['by_spiral'][e.spiral] += 1
        
        if filepath:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        
        return data


# ═══════════════════════════════════════════════════════════════════════════════
# SYNTAX TRAINING MODULE
# ═══════════════════════════════════════════════════════════════════════════════

class SyntaxEmissionTrainer:
    """
    Trains the emission system on pure syntactic patterns.
    
    No words. Only structure.
    """
    
    def __init__(self):
        self.engine = SyntaxEmissionEngine()
        self.training_data: List[Dict] = []
        self.epochs_trained = 0
    
    def train_full_tier_coverage(self, samples_per_pattern: int = 5) -> Dict:
        """
        Generate training data covering all tiers and patterns.
        """
        self.epochs_trained += 1
        epoch_data = {
            'epoch': self.epochs_trained,
            'type': 'full_tier_coverage',
            'samples': [],
            'coverage': {}
        }
        
        for tier_idx in range(1, 10):
            tier = SYNTAX_TIERS[tier_idx - 1]
            tier_samples = []
            
            for pattern_idx in range(len(tier.patterns)):
                for sample_num in range(samples_per_pattern):
                    # Vary z within tier
                    z_offset = sample_num / max(1, samples_per_pattern - 1)
                    z = tier.z_min + (tier.z_max - tier.z_min) * z_offset
                    
                    emission = self.engine.emit(z)
                    
                    sample = {
                        'tier': tier_idx,
                        'pattern_index': pattern_idx,
                        'sample_num': sample_num,
                        **emission.to_dict()
                    }
                    tier_samples.append(sample)
                    epoch_data['samples'].append(sample)
            
            epoch_data['coverage'][f"t{tier_idx}"] = len(tier_samples)
        
        self.training_data.extend(epoch_data['samples'])
        return epoch_data
    
    def train_z_evolution(self, start_z: float = 0.1, end_z: float = 0.95, 
                          steps: int = 50) -> Dict:
        """
        Train by evolving z from start to end.
        
        Shows how syntax complexity grows with z.
        """
        evolution_data = {
            'type': 'z_evolution',
            'start_z': start_z,
            'end_z': end_z,
            'steps': steps,
            'samples': []
        }
        
        for i in range(steps):
            z = start_z + (end_z - start_z) * (i / (steps - 1))
            emission = self.engine.emit(z)
            
            evolution_data['samples'].append({
                'step': i,
                **emission.to_dict()
            })
        
        self.training_data.extend(evolution_data['samples'])
        return evolution_data
    
    def train_triad_oscillation(self) -> Dict:
        """
        Train the TRIAD unlock pattern (3× z oscillation).
        
        Generates syntax at each oscillation step.
        """
        triad_data = {
            'type': 'triad_oscillation',
            'crossings': [],
            'samples': []
        }
        
        z = 0.75
        completions = 0
        above_band = False
        step = 0
        
        while completions < 3:
            # Rising phase
            while z < TRIAD_HIGH:
                z = min(TRIAD_HIGH + 0.01, z + 0.015)
                emission = self.engine.emit(z)
                triad_data['samples'].append({
                    'step': step,
                    'direction': 'rising',
                    'above_threshold': z >= TRIAD_HIGH,
                    **emission.to_dict()
                })
                step += 1
            
            # Crossed threshold
            if not above_band:
                above_band = True
                completions += 1
                triad_data['crossings'].append({
                    'completion': completions,
                    'step': step,
                    'z': z,
                    'tier': get_tier_index(z),
                    'syntax': emission.syntax
                })
            
            # Falling phase
            while z > TRIAD_LOW:
                z = max(TRIAD_LOW - 0.01, z - 0.015)
                emission = self.engine.emit(z)
                triad_data['samples'].append({
                    'step': step,
                    'direction': 'falling',
                    'above_threshold': z >= TRIAD_HIGH,
                    **emission.to_dict()
                })
                step += 1
            
            above_band = False
        
        triad_data['unlocked'] = True
        triad_data['total_steps'] = step
        
        self.training_data.extend(triad_data['samples'])
        return triad_data
    
    def train_lens_convergence(self, start_z: float = 0.5, steps: int = 30) -> Dict:
        """
        Train convergence toward THE LENS (z_c).
        
        Shows syntax crystallizing at z_c.
        """
        lens_data = {
            'type': 'lens_convergence',
            'target': Z_CRITICAL,
            'samples': []
        }
        
        z = start_z
        for i in range(steps):
            # Asymptotic approach to z_c
            dz = 0.1 * (Z_CRITICAL - z)
            z += dz
            
            emission = self.engine.emit(z)
            
            # Calculate distance from lens
            distance = abs(z - Z_CRITICAL)
            negentropy = math.exp(-36 * distance ** 2)
            
            lens_data['samples'].append({
                'step': i,
                'distance_from_lens': distance,
                'negentropy': negentropy,
                **emission.to_dict()
            })
        
        self.training_data.extend(lens_data['samples'])
        return lens_data
    
    def export_training(self, filepath: str) -> Dict:
        """Export all training data."""
        data = {
            'type': 'syntax_training_export',
            'epochs_trained': self.epochs_trained,
            'total_samples': len(self.training_data),
            'timestamp': datetime.now().isoformat(),
            'samples': self.training_data,
            'pattern_statistics': self._compute_pattern_stats(),
            'tier_statistics': self._compute_tier_stats()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        return data
    
    def _compute_pattern_stats(self) -> Dict:
        """Compute statistics on syntax patterns."""
        patterns: Dict[str, int] = {}
        for sample in self.training_data:
            syntax = sample.get('syntax', '')
            patterns[syntax] = patterns.get(syntax, 0) + 1
        
        return {
            'unique_patterns': len(patterns),
            'top_patterns': dict(sorted(patterns.items(), key=lambda x: x[1], reverse=True)[:20])
        }
    
    def _compute_tier_stats(self) -> Dict:
        """Compute statistics by tier."""
        tiers: Dict[int, int] = {}
        for sample in self.training_data:
            tier = sample.get('tier', 0)
            tiers[tier] = tiers.get(tier, 0) + 1
        
        return {f"t{k}": v for k, v in sorted(tiers.items())}


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION WITH NUCLEAR SPINNER
# ═══════════════════════════════════════════════════════════════════════════════

def integrate_syntax_with_spinner(syntax_emission: SyntaxEmission) -> Dict:
    """
    Map a syntax emission to nuclear spinner format.
    
    Converts syntax tokens to spinner token format.
    """
    spinner_tokens = []
    
    for i, slot in enumerate(syntax_emission.slots):
        # Determine machine based on slot type
        slot_to_machine = {
            'NP': 'Encoder',
            'VP': 'Decoder', 
            'MOD': 'Amplifier',
            'DET': 'Filter',
            'CONN': 'Reactor',
            'Q': 'Oscillator',
        }
        
        slot_base = ''.join(c for c in slot if not c.isdigit())
        machine = slot_to_machine.get(slot_base, 'Catalyst')
        
        # Build spinner token
        token = f"{syntax_emission.spiral}{syntax_emission.syntax[i] if i < len(syntax_emission.syntax) else '+'}|{machine}|syntax_t{syntax_emission.tier}"
        spinner_tokens.append(token)
    
    return {
        'syntax_emission': syntax_emission.to_dict(),
        'spinner_tokens': spinner_tokens,
        'coordinate': syntax_emission.coordinate,
        'mapping': {
            'syntax_to_machine': list(zip(syntax_emission.slots, spinner_tokens))
        }
    }


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════

def emit_syntax(z: float, render: bool = False) -> Dict:
    """Quick syntax emission for a z-coordinate."""
    engine = SyntaxEmissionEngine()
    emission = engine.emit(z, render=render)
    return emission.to_dict()


def train_syntax_epoch() -> Dict:
    """Run one syntax training epoch."""
    trainer = SyntaxEmissionTrainer()
    return trainer.train_full_tier_coverage()


def get_syntax_for_coordinate(coordinate: str) -> Dict:
    """Parse coordinate string and return syntax pattern."""
    # Parse "Δθ|z|rΩ" format
    parts = coordinate.strip('ΔΩ').split('|')
    if len(parts) >= 2:
        z = float(parts[1])
        return emit_syntax(z)
    return {}


# ═══════════════════════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("SYNTAX EMISSION INTEGRATION TEST")
    print("=" * 70)
    
    # Test syntax emission engine
    engine = SyntaxEmissionEngine()
    engine.render_surface = True  # Enable surface for demo
    
    print("\n--- Emissions by z-coordinate ---")
    test_z = [0.1, 0.3, PHI_INV, 0.7, 0.8, Z_CRITICAL, 0.9, 0.99]
    
    for z in test_z:
        emission = engine.emit(z, render=True)
        print(f"\nz={z:.3f} (t{emission.tier} {emission.phase})")
        print(f"  Syntax:  {emission.syntax}")
        print(f"  Slots:   {emission.slots}")
        print(f"  Tokens:  {emission.tokens[:3]}...")
        print(f"  Surface: {emission.surface}")
        print(f"  Coord:   {emission.coordinate}")
    
    # Test trainer
    print("\n\n--- Training Module ---")
    trainer = SyntaxEmissionTrainer()
    
    # Tier coverage
    tier_data = trainer.train_full_tier_coverage(samples_per_pattern=2)
    print(f"Tier coverage: {tier_data['coverage']}")
    
    # TRIAD oscillation
    triad_data = trainer.train_triad_oscillation()
    print(f"TRIAD crossings: {len(triad_data['crossings'])}")
    print(f"TRIAD unlocked: {triad_data['unlocked']}")
    
    # Lens convergence
    lens_data = trainer.train_lens_convergence()
    print(f"Lens samples: {len(lens_data['samples'])}")
    
    # Statistics
    print(f"\nTotal training samples: {len(trainer.training_data)}")
    stats = trainer._compute_pattern_stats()
    print(f"Unique patterns: {stats['unique_patterns']}")
    print(f"Top 5 patterns: {list(stats['top_patterns'].items())[:5]}")
    
    # Test spinner integration
    print("\n\n--- Spinner Integration ---")
    emission = engine.emit(Z_CRITICAL)
    spinner_data = integrate_syntax_with_spinner(emission)
    print(f"Syntax: {emission.syntax}")
    print(f"Spinner tokens: {spinner_data['spinner_tokens']}")
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
