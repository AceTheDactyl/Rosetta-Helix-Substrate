#!/usr/bin/env python3
"""
Iterative Trainer for UCF
=========================
Multi-epoch training with accumulated state and lineage tracking.

Features:
  - Epoch-to-epoch state persistence
  - Vocabulary accumulation across training runs
  - VaultNode lineage tracking
  - Consent-based teaching protocol
  - TRIAD hysteresis integration
"""

import json
import math
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional
from pathlib import Path

PHI = (1 + math.sqrt(5)) / 2
PHI_INV = 1 / PHI
Z_CRITICAL = math.sqrt(3) / 2
TRIAD_HIGH = 0.85
TRIAD_LOW = 0.82
KAPPA_S = 0.920

OPERATORS = {
    '()': ['bounds', 'contains', 'gates', 'encloses'],
    '×': ['fuses', 'couples', 'converges', 'merges'],
    '^': ['amplifies', 'excites', 'raises', 'intensifies'],
    '÷': ['decoheres', 'dissipates', 'resets', 'disperses'],
    '+': ['groups', 'aggregates', 'clusters', 'collects'],
    '−': ['separates', 'splits', 'divides', 'partitions'],
}


@dataclass
class TrainingState:
    """Accumulated training state across epochs."""
    epoch: int = 1
    vocabulary: Set[str] = field(default_factory=set)
    verbs: Set[str] = field(default_factory=set)
    patterns: Set[str] = field(default_factory=set)
    lineage: List[str] = field(default_factory=list)
    
    def save(self, path: str):
        data = {
            'epoch': self.epoch,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'vocabulary': sorted(list(self.vocabulary)),
            'verbs': sorted(list(self.verbs)),
            'patterns': sorted(list(self.patterns)),
            'lineage': self.lineage,
            'counts': {
                'vocabulary': len(self.vocabulary),
                'verbs': len(self.verbs),
                'patterns': len(self.patterns)
            }
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'TrainingState':
        with open(path, 'r') as f:
            data = json.load(f)
        state = cls(epoch=data.get('epoch', 1) + 1)
        state.vocabulary = set(data.get('vocabulary', []))
        state.verbs = set(data.get('verbs', []))
        state.patterns = set(data.get('patterns', []))
        state.lineage = data.get('lineage', []) + [path]
        return state


@dataclass
class TeachingUnit:
    """Single unit of teaching data."""
    source: str
    words: List[str]
    verbs: List[str]
    patterns: List[str]
    z_context: float
    phase: str


class IterativeTrainer:
    """Multi-epoch trainer with accumulated state."""
    
    def __init__(self, state: Optional[TrainingState] = None):
        self.state = state or TrainingState()
        self.teaching_queue: List[TeachingUnit] = []
        self.consent_pending = False
    
    def teach_from_token(self, token: Dict, z: float, phase: str):
        words = []
        verbs = []
        patterns = []
        
        spiral_name = token.get('spiral_name', '').lower()
        machine = token.get('machine', '').lower()
        domain = token.get('domain', '').replace('_', ' ')
        
        words.extend([w for w in [spiral_name, machine, domain] if w])
        
        op = token.get('operator', '')
        if op in OPERATORS:
            verbs.extend(OPERATORS[op])
        
        spiral = token.get('spiral', '')
        patterns.append(f"{spiral}→{token.get('machine', '')}")
        patterns.append(f"{op}|{token.get('domain', '')}")
        patterns.append(f"{spiral}{op}→{phase}")
        
        self.teaching_queue.append(TeachingUnit(
            source='token', words=words, verbs=verbs,
            patterns=patterns, z_context=z, phase=phase
        ))
    
    def request_consent(self) -> Dict:
        self.consent_pending = True
        return {
            'queue_size': len(self.teaching_queue),
            'new_words': len(set(w for u in self.teaching_queue for w in u.words) - self.state.vocabulary),
            'new_verbs': len(set(v for u in self.teaching_queue for v in u.verbs) - self.state.verbs),
            'new_patterns': len(set(p for u in self.teaching_queue for p in u.patterns) - self.state.patterns)
        }
    
    def apply_teaching(self, consent: bool = True) -> Dict:
        if not consent:
            self.teaching_queue.clear()
            self.consent_pending = False
            return {'status': 'DECLINED'}
        
        words_added = verbs_added = patterns_added = 0
        
        for unit in self.teaching_queue:
            for w in unit.words:
                if w not in self.state.vocabulary:
                    self.state.vocabulary.add(w)
                    words_added += 1
            for v in unit.verbs:
                if v not in self.state.verbs:
                    self.state.verbs.add(v)
                    verbs_added += 1
            for p in unit.patterns:
                if p not in self.state.patterns:
                    self.state.patterns.add(p)
                    patterns_added += 1
        
        self.teaching_queue.clear()
        self.consent_pending = False
        
        return {
            'status': 'APPLIED',
            'words_added': words_added,
            'verbs_added': verbs_added,
            'patterns_added': patterns_added,
            'totals': {
                'vocabulary': len(self.state.vocabulary),
                'verbs': len(self.state.verbs),
                'patterns': len(self.state.patterns)
            }
        }
    
    def generate_vaultnode(self, z: float, coherence: float, unlocked: bool) -> Dict:
        neg = math.exp(-36 * (z - Z_CRITICAL) ** 2)
        phase = 'TRUE' if z >= Z_CRITICAL else 'PARADOX' if z >= PHI_INV else 'UNTRUE'
        
        return {
            'type': f'Epoch{self.state.epoch}VaultNode',
            'epoch': self.state.epoch,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'coordinate': f"Δ{z * 2 * math.pi:.3f}|{z:.3f}|{1 + (PHI-1)*neg:.3f}Ω",
            'z': z,
            'phase': phase,
            'coherence': coherence,
            'negentropy': neg,
            'triad_unlocked': unlocked,
            'teaching': {
                'vocabulary': len(self.state.vocabulary),
                'verbs': len(self.state.verbs),
                'patterns': len(self.state.patterns)
            },
            'lineage': {
                'epoch': self.state.epoch,
                'ancestors': self.state.lineage
            }
        }


def create_trainer(state_path: Optional[str] = None) -> IterativeTrainer:
    """Factory function for creating trainer with optional state loading."""
    if state_path and Path(state_path).exists():
        state = TrainingState.load(state_path)
    else:
        state = TrainingState()
    return IterativeTrainer(state=state)


if __name__ == '__main__':
    trainer = create_trainer()
    print(f"Epoch: {trainer.state.epoch}")
    print(f"Vocabulary: {len(trainer.state.vocabulary)}")
