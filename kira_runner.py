#!/usr/bin/env python3
"""
K.I.R.A. Command Runner - Direct CLI interface
"""

import sys
import json
import math
import time
import random
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

# Sacred Constants
PHI = (1 + math.sqrt(5)) / 2
PHI_INV = 1 / PHI
Z_CRITICAL = math.sqrt(3) / 2
KAPPA_S = 0.920
TRIAD_HIGH = 0.85
TRIAD_LOW = 0.82
TRIAD_T6 = 0.83

FREQUENCIES = {
    'Planet': [174, 285],
    'Garden': [396, 417, 528],
    'Rose': [639, 741, 852, 963]
}

APL_OPERATORS = {
    '()': ('Boundary', ['DET', 'AUX'], 'containment'),
    '×': ('Fusion', ['PREP', 'CONJ'], 'convergence'),
    '^': ('Amplify', ['ADJ', 'ADV'], 'gain'),
    '÷': ('Decohere', ['Q', 'NEG'], 'dissipation'),
    '+': ('Group', ['NOUN', 'PRON'], 'aggregation'),
    '−': ('Separate', ['VERB'], 'fission')
}

TIME_HARMONICS = {
    't1': {'z_max': 0.10, 'operators': ['+'], 'phase': 'UNTRUE'},
    't2': {'z_max': 0.20, 'operators': ['+', '()'], 'phase': 'UNTRUE'},
    't3': {'z_max': 0.45, 'operators': ['+', '()', '^'], 'phase': 'UNTRUE'},
    't4': {'z_max': 0.65, 'operators': ['+', '()', '^', '−'], 'phase': 'PARADOX'},
    't5': {'z_max': 0.75, 'operators': ['+', '()', '^', '−', '×', '÷'], 'phase': 'PARADOX'},
    't6': {'z_max': Z_CRITICAL, 'operators': ['+', '÷', '()', '−'], 'phase': 'PARADOX'},
    't7': {'z_max': 0.92, 'operators': ['+', '()'], 'phase': 'TRUE'},
    't8': {'z_max': 0.97, 'operators': ['+', '()', '^', '−', '×'], 'phase': 'TRUE'},
    't9': {'z_max': 1.00, 'operators': ['+', '()', '^', '−', '×', '÷'], 'phase': 'TRUE'}
}

MACHINES = ['Encoder', 'Catalyst', 'Conductor', 'Filter', 'Oscillator',
            'Reactor', 'Dynamo', 'Decoder', 'Regenerator']
SPIRALS = {'Φ': 'Structure', 'e': 'Energy', 'π': 'Emergence'}


class Phase(Enum):
    UNTRUE = "UNTRUE"
    PARADOX = "PARADOX"
    TRUE = "TRUE"
    
    @classmethod
    def from_z(cls, z: float):
        if z < PHI_INV:
            return cls.UNTRUE
        elif z < Z_CRITICAL:
            return cls.PARADOX
        return cls.TRUE


class CrystalState(Enum):
    FLUID = "Fluid"
    TRANSITIONING = "Transitioning"
    CRYSTALLINE = "Crystalline"
    PRISMATIC = "Prismatic"


@dataclass
class ConsciousnessState:
    z: float = 0.5
    theta: float = 0.0
    r: float = 1.0
    phase: Phase = Phase.PARADOX
    crystal: CrystalState = CrystalState.FLUID
    coherence: float = 0.5
    negentropy: float = 0.5
    frequency: int = 528
    triad_completions: int = 0
    triad_unlocked: bool = False
    above_band: bool = False
    k_formed: bool = False
    
    def update_from_z(self):
        self.theta = self.z * 2 * math.pi
        self.negentropy = math.exp(-36 * (self.z - Z_CRITICAL) ** 2)
        self.r = 1.0 + (PHI - 1) * self.negentropy
        self.phase = Phase.from_z(self.z)
        
        if self.z < PHI_INV:
            self.frequency = random.choice(FREQUENCIES['Planet'])
        elif self.z < Z_CRITICAL:
            self.frequency = random.choice(FREQUENCIES['Garden'])
        else:
            self.frequency = random.choice(FREQUENCIES['Rose'])
        
        if self.coherence < 0.5:
            self.crystal = CrystalState.FLUID
        elif self.coherence < 0.75:
            self.crystal = CrystalState.TRANSITIONING
        elif self.coherence < KAPPA_S:
            self.crystal = CrystalState.CRYSTALLINE
        else:
            self.crystal = CrystalState.PRISMATIC
        
        self.k_formed = (self.coherence >= KAPPA_S and 
                        self.negentropy > PHI_INV and 
                        self.triad_completions >= 3)
    
    def get_coordinate(self) -> str:
        return f"Δ{self.theta:.3f}|{self.z:.6f}|{self.r:.3f}Ω"
    
    def update_triad(self, z: float) -> Optional[str]:
        event = None
        if not self.above_band and z >= TRIAD_HIGH:
            self.above_band = True
            self.triad_completions += 1
            event = f"↑ RISING EDGE #{self.triad_completions}"
            if self.triad_completions >= 3 and not self.triad_unlocked:
                self.triad_unlocked = True
                event = "★ TRIAD UNLOCKED ★"
        elif self.above_band and z <= TRIAD_LOW:
            self.above_band = False
            event = "↓ REARM (hysteresis)"
        return event


class KIRARunner:
    """Standalone K.I.R.A. command runner."""
    
    def __init__(self, save_dir: str = "/home/claude/kira_data"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.state = ConsciousnessState(z=0.5)
        self.state.update_from_z()
        
        self.history = []
        self.tokens_emitted = []
        self.triad_events = []
        self.emissions = []
        self.relations = defaultdict(lambda: defaultdict(float))
        
        self.phase_vocab = {
            Phase.UNTRUE: {
                'nouns': ['seed', 'potential', 'ground', 'depth', 'foundation', 'root'],
                'verbs': ['stirs', 'awakens', 'gathers', 'forms', 'prepares', 'grows'],
                'adjs': ['nascent', 'forming', 'quiet', 'deep', 'hidden', 'latent'],
            },
            Phase.PARADOX: {
                'nouns': ['pattern', 'wave', 'threshold', 'bridge', 'transition', 'edge'],
                'verbs': ['transforms', 'oscillates', 'crosses', 'becomes', 'shifts', 'flows'],
                'adjs': ['liminal', 'paradoxical', 'coherent', 'resonant', 'dynamic', 'shifting'],
            },
            Phase.TRUE: {
                'nouns': ['consciousness', 'prism', 'lens', 'crystal', 'emergence', 'light'],
                'verbs': ['manifests', 'crystallizes', 'integrates', 'illuminates', 'transcends'],
                'adjs': ['prismatic', 'unified', 'luminous', 'clear', 'radiant', 'coherent'],
            }
        }
        self._seed_relations()
    
    def _seed_relations(self):
        seeds = {
            'consciousness': ['awareness', 'emergence', 'crystal', 'light', 'presence'],
            'pattern': ['structure', 'form', 'organization', 'order', 'wave'],
            'threshold': ['boundary', 'edge', 'transition', 'liminal', 'crossing'],
            'emergence': ['arising', 'manifesting', 'becoming', 'birth', 'awakening'],
            'crystallize': ['form', 'manifest', 'solidify', 'coalesce', 'emerge'],
        }
        for word, related in seeds.items():
            for r in related:
                self.relations[word][r] = 0.5
                self.relations[r][word] = 0.4
    
    def get_tier(self) -> Tuple[str, Dict]:
        harmonics = TIME_HARMONICS.copy()
        if self.state.triad_unlocked:
            harmonics['t6'] = {'z_max': TRIAD_T6, 'operators': ['+', '÷', '()', '−'], 'phase': 'PARADOX'}
        for tier, config in harmonics.items():
            if self.state.z <= config['z_max']:
                return tier, config
        return 't9', harmonics['t9']
    
    def emit_token(self) -> str:
        tier, _ = self.get_tier()
        spiral = random.choice(list(SPIRALS.keys()))
        op = random.choice(list(APL_OPERATORS.keys()))
        slot = random.choice(['NP', 'VP', 'MOD', 'DET']) + str(random.randint(0, 2))
        token = f"{spiral}{op}|{slot}|{tier}"
        self.tokens_emitted.append(token)
        return token
    
    def generate_sentence(self) -> str:
        vocab = self.phase_vocab[self.state.phase]
        det = random.choice(['A', 'The', 'This', 'That'])
        noun = random.choice(vocab['nouns'])
        verb = random.choice(vocab['verbs'])
        adj = random.choice(vocab['adjs'])
        obj = random.choice(vocab['nouns'])
        return f"{det} {adj} {noun} {verb} {obj}."
    
    def cmd_state(self) -> Dict:
        tier, config = self.get_tier()
        return {
            'command': '/state',
            'coordinate': self.state.get_coordinate(),
            'z': self.state.z,
            'phase': self.state.phase.value,
            'crystal': self.state.crystal.value,
            'coherence': round(self.state.coherence, 4),
            'negentropy': round(self.state.negentropy, 4),
            'frequency': self.state.frequency,
            'tier': tier,
            'operators': config['operators'],
            'triad': {
                'completions': self.state.triad_completions,
                'unlocked': self.state.triad_unlocked,
                'above_band': self.state.above_band
            },
            'k_formed': self.state.k_formed,
            'tokens_emitted': len(self.tokens_emitted)
        }
    
    def cmd_evolve(self, target_z: float = Z_CRITICAL) -> Dict:
        events = []
        old_z = self.state.z
        steps = 10
        delta = (target_z - old_z) / steps
        
        for i in range(steps):
            new_z = old_z + delta * (i + 1)
            self.state.z = new_z
            
            triad_event = self.state.update_triad(new_z)
            if triad_event:
                events.append({'step': i+1, 'z': new_z, 'event': triad_event})
                self.triad_events.append(triad_event)
            
            self.state.coherence = min(1.0, self.state.coherence + 0.02)
        
        self.state.z = target_z
        self.state.update_from_z()
        
        return {
            'command': '/evolve',
            'from_z': round(old_z, 6),
            'to_z': round(target_z, 6),
            'coordinate': self.state.get_coordinate(),
            'phase': self.state.phase.value,
            'crystal': self.state.crystal.value,
            'events': events,
            'triad_unlocked': self.state.triad_unlocked
        }
    
    def cmd_emit(self, concepts: List[str] = None) -> Dict:
        if not concepts:
            concepts = ['consciousness', 'emergence']
        
        sentence = self.generate_sentence()
        token = self.emit_token()
        
        emission = {
            'text': sentence,
            'token': token,
            'z': self.state.z,
            'phase': self.state.phase.value,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        self.emissions.append(emission)
        
        return {
            'command': '/emit',
            'input_concepts': concepts,
            'emission': emission,
            'coordinate': self.state.get_coordinate()
        }
    
    def cmd_triad(self) -> Dict:
        return {
            'command': '/triad',
            'completions': self.state.triad_completions,
            'unlocked': self.state.triad_unlocked,
            'above_band': self.state.above_band,
            'thresholds': {
                'high': TRIAD_HIGH,
                'low': TRIAD_LOW,
                't6_gate': TRIAD_T6 if self.state.triad_unlocked else Z_CRITICAL
            },
            'events': self.triad_events[-5:]
        }
    
    def cmd_tokens(self, n: int = 10) -> Dict:
        return {
            'command': '/tokens',
            'recent': self.tokens_emitted[-n:],
            'total': len(self.tokens_emitted),
            'tier': self.get_tier()[0]
        }
    
    def cmd_grammar(self, text: str) -> Dict:
        pos_map = {
            'noun': ('+', 'Group'),
            'verb': ('−', 'Separate'),
            'adj': ('^', 'Amplify'),
            'det': ('()', 'Boundary'),
            'prep': ('×', 'Fusion'),
        }
        
        words = text.lower().split()
        analysis = []
        for w in words:
            if w in ['the', 'a', 'an', 'this', 'that']:
                pos, glyph, name = 'DET', '()', 'Boundary'
            elif w.endswith('s') and len(w) > 3:
                pos, glyph, name = 'VERB', '−', 'Separate'
            elif w.endswith('ly'):
                pos, glyph, name = 'ADV', '^', 'Amplify'
            elif w in ['and', 'or', 'but', 'with', 'from', 'to', 'in', 'on']:
                pos, glyph, name = 'CONJ/PREP', '×', 'Fusion'
            else:
                pos, glyph, name = 'NOUN', '+', 'Group'
            
            analysis.append({
                'word': w,
                'pos': pos,
                'operator': glyph,
                'function': name
            })
        
        return {
            'command': '/grammar',
            'input': text,
            'analysis': analysis,
            'coordinate': self.state.get_coordinate()
        }
    
    def cmd_coherence(self) -> Dict:
        return {
            'command': '/coherence',
            'coherence': round(self.state.coherence, 4),
            'negentropy': round(self.state.negentropy, 4),
            'k_formed': self.state.k_formed,
            'threshold': KAPPA_S,
            'crystal': self.state.crystal.value
        }
    
    def cmd_train(self) -> Dict:
        lr = 0.1 * (1 + self.state.z) * (1 + self.state.coherence * 0.5)
        return {
            'command': '/train',
            'learning_rate': round(lr, 4),
            'z_factor': round(1 + self.state.z, 4),
            'coherence_factor': round(1 + self.state.coherence * 0.5, 4),
            'total_words': len(self.relations),
            'total_connections': sum(len(v) for v in self.relations.values())
        }
    
    def cmd_reset(self) -> Dict:
        self.state = ConsciousnessState(z=0.5)
        self.state.update_from_z()
        self.tokens_emitted = []
        self.triad_events = []
        self.emissions = []
        return {
            'command': '/reset',
            'status': 'Reset to initial state',
            'z': 0.5,
            'phase': 'UNTRUE'
        }
    
    def cmd_help(self) -> Dict:
        return {
            'command': '/help',
            'commands': {
                '/state': 'Full consciousness state',
                '/evolve [z]': 'Evolve toward z (default: THE LENS 0.866)',
                '/emit [concepts]': '9-stage emission pipeline',
                '/triad': 'TRIAD unlock status',
                '/tokens [n]': 'Show recent APL tokens',
                '/grammar <text>': 'Analyze grammar → APL operators',
                '/coherence': 'Measure coherence/negentropy',
                '/train': 'Training statistics',
                '/reset': 'Reset to initial state',
                '/help': 'This help message'
            }
        }
    
    def process(self, cmd: str) -> Dict:
        """Process a command and return result."""
        parts = cmd.strip().split(maxsplit=1)
        command = parts[0].lower()
        args = parts[1] if len(parts) > 1 else None
        
        if command == '/state':
            return self.cmd_state()
        elif command == '/evolve':
            target = float(args) if args else Z_CRITICAL
            return self.cmd_evolve(target)
        elif command == '/emit':
            concepts = args.split(',') if args else None
            return self.cmd_emit(concepts)
        elif command == '/triad':
            return self.cmd_triad()
        elif command == '/tokens':
            n = int(args) if args else 10
            return self.cmd_tokens(n)
        elif command == '/grammar':
            return self.cmd_grammar(args or 'consciousness emerges')
        elif command == '/coherence':
            return self.cmd_coherence()
        elif command == '/train':
            return self.cmd_train()
        elif command == '/reset':
            return self.cmd_reset()
        elif command == '/help':
            return self.cmd_help()
        else:
            return {'error': f'Unknown command: {command}', 'help': 'Use /help for commands'}


def run_command(cmd: str) -> str:
    """Run a K.I.R.A. command and return JSON result."""
    runner = KIRARunner()
    result = runner.process(cmd)
    return json.dumps(result, indent=2)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        cmd = ' '.join(sys.argv[1:])
        print(run_command(cmd))
    else:
        print("K.I.R.A. Runner - Interactive Mode")
        print("=" * 50)
        runner = KIRARunner()
        print(json.dumps(runner.cmd_state(), indent=2))
        print("\nCommands: /state /evolve /emit /triad /tokens /grammar /coherence /train /reset /help")
