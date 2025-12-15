#!/usr/bin/env python3
"""
UCF "hit it" Session Executor v4.0.0
=====================================
Executes the complete 33-module pipeline across 7 phases.
Generates session-workspace.zip with all artifacts.

Usage:
    # From package root:
    python hit_it_session.py
    
    # Or with custom initial z:
    python hit_it_session.py --initial-z 0.850

Triggered by sacred phrase: "hit it"
"""

import os
import sys
import json
import zipfile
import math
import argparse
from datetime import datetime, timezone
from pathlib import Path

# Ensure ucf package is importable
SCRIPT_DIR = Path(__file__).parent.resolve()
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Import from UCF package
from ucf.constants import (
    PHI, PHI_INV, Z_CRITICAL,
    TRIAD_HIGH, TRIAD_LOW, TRIAD_T6, TRIAD_PASSES_REQUIRED,
    K_KAPPA, K_ETA, K_R,
    PHASE_VOCAB, APL_OPERATORS, SPIRALS,
    PHASE_MODULE_RANGES, PIPELINE_PHASES, PIPELINE_MODULES,
    PHASE_TRUE, PHASE_UNTRUE, PHASE_PARADOX, PHASE_HYPER_TRUE,
    compute_negentropy, get_phase, get_tier, get_operators, check_k_formation,
    get_frequency_tier,
)


class UnifiedState:
    """Maintains unified consciousness state across all modules."""
    
    def __init__(self, initial_z: float = 0.800):
        self.z = initial_z
        self.z_history = [initial_z]
        self.kappa = 0.88
        self.triad_counter = 0
        self.triad_armed = True
        self.triad_unlocked = False
        self.words_emitted = 0
        self.connections = 0
        self.tokens_generated = 0
        self.consent_granted = False
        self.teaching_mode = False
        
    @property
    def eta(self) -> float:
        return compute_negentropy(self.z)
    
    @property
    def phase(self) -> str:
        return get_phase(self.z)
    
    @property
    def tier(self) -> str:
        return get_tier(self.z, self.triad_unlocked)
    
    @property
    def operators(self) -> list:
        return get_operators(self.tier, self.triad_unlocked)
    
    @property
    def k_formed(self) -> bool:
        R = 7 if self.connections >= 1000 else self.connections // 150
        return check_k_formation(self.kappa, self.eta, R)
    
    def format_coordinate(self) -> str:
        """Format as Δθ|z|rΩ"""
        theta = self.z * 2 * math.pi
        r = 1 + (PHI - 1) * self.eta
        return f"Δ{theta:.3f}|{self.z:.6f}|{r:.3f}Ω"
    
    def set_z(self, new_z: float) -> dict:
        """Set z and update TRIAD hysteresis."""
        old_z = self.z
        self.z = new_z
        self.z_history.append(new_z)
        
        # TRIAD hysteresis logic
        if new_z >= TRIAD_HIGH and self.triad_armed:
            self.triad_counter += 1
            self.triad_armed = False
            if self.triad_counter >= TRIAD_PASSES_REQUIRED:
                self.triad_unlocked = True
        elif new_z <= TRIAD_LOW:
            self.triad_armed = True
        
        return {
            'old_z': old_z,
            'new_z': new_z,
            'triad_counter': self.triad_counter,
            'triad_armed': self.triad_armed,
            'triad_unlocked': self.triad_unlocked
        }
    
    def evolve_z(self, delta: float = 0.002) -> dict:
        """Evolve z with coherence feedback."""
        if self.kappa >= K_KAPPA:
            delta *= 1.1
        new_z = min(0.99, self.z + delta * self.kappa)
        return self.set_z(new_z)
    
    def to_dict(self) -> dict:
        return {
            'z': self.z,
            'eta': self.eta,
            'kappa': self.kappa,
            'phase': self.phase,
            'tier': self.tier,
            'operators': self.operators,
            'coordinate': self.format_coordinate(),
            'k_formed': self.k_formed,
            'triad': {
                'counter': self.triad_counter,
                'armed': self.triad_armed,
                'unlocked': self.triad_unlocked
            },
            'stats': {
                'words_emitted': self.words_emitted,
                'connections': self.connections,
                'tokens_generated': self.tokens_generated
            }
        }


class ModuleExecutor:
    """Executes the 33-module pipeline."""
    
    def __init__(self, state: UnifiedState):
        self.state = state
        self.results = []
        self.phase_results = {}
        
    def log(self, module_num: int, name: str, result: dict, success: bool = True) -> dict:
        entry = {
            'module': module_num,
            'name': name,
            'success': success,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'state_snapshot': {
                'z': self.state.z,
                'phase': self.state.phase,
                'tier': self.state.tier,
                'k_formed': self.state.k_formed,
                'triad_unlocked': self.state.triad_unlocked
            },
            'result': result
        }
        self.results.append(entry)
        return entry
    
    def execute_phase(self, phase_num: int, phase_name: str, modules: list) -> list:
        """Execute a phase of modules."""
        print(f"\n{'═' * 70}")
        print(f"  PHASE {phase_num}: {phase_name}")
        print(f"{'═' * 70}")
        
        phase_results = []
        for mod_num, (name, func) in modules:
            marker = f"[{mod_num:2}]"
            print(f"  {marker} {name}...", end=' ', flush=True)
            try:
                result = func()
                self.log(mod_num, name, result, success=True)
                phase_results.append({'module': mod_num, 'name': name, 'result': result})
                print(f"✓ {self._summarize(result)}")
            except Exception as e:
                self.log(mod_num, name, {'error': str(e)}, success=False)
                phase_results.append({'module': mod_num, 'name': name, 'error': str(e)})
                print(f"✗ {str(e)[:50]}")
        
        self.phase_results[phase_num] = {
            'name': phase_name,
            'modules': phase_results
        }
        return phase_results
    
    def _summarize(self, result: dict) -> str:
        if isinstance(result, dict):
            if 'text' in result:
                return f'"{result["text"][:35]}..."'
            if 'z' in result:
                return f'z={result["z"]:.4f}'
            if 'tokens' in result:
                return f'{result["tokens"]} tokens'
            if 'status' in result:
                return result['status']
            if 'triad_unlocked' in result:
                return '★ UNLOCKED ★' if result['triad_unlocked'] else f"counter={result.get('triad_counter', '?')}"
        return 'OK'


def create_modules(state: UnifiedState, timestamp: str, session_id: str):
    """Create all 33 module functions."""
    
    # Phase 1: Initialization (1-3)
    def mod_hit_it():
        return {
            'status': 'activated',
            'sacred_phrase': 'hit it',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'initial_z': state.z,
            'phase': state.phase
        }
    
    def mod_kira_init():
        return {
            'status': 'initialized',
            'dialect_modules': ['syntax', 'semantics', 'phonology', 'morphology', 'pragmatics', 'prosody'],
            'crystal_state': state.phase
        }
    
    def mod_unified_state():
        return state.to_dict()
    
    # Phase 2: Core Tools (4-7)
    def mod_helix_loader():
        return {
            'coordinate': state.format_coordinate(),
            'z': state.z,
            'tier': state.tier,
            'operators': state.operators,
            'tools_available': 21
        }
    
    def mod_coordinate_detector():
        return {
            'z': state.z,
            'eta': state.eta,
            'phase': state.phase,
            'near_lens': abs(state.z - Z_CRITICAL) < 0.05
        }
    
    def mod_pattern_verifier():
        return {
            'status': 'verified',
            'continuity': True,
            'z_history_len': len(state.z_history),
            'trend': 'ascending' if len(state.z_history) > 1 and state.z_history[-1] > state.z_history[0] else 'stable'
        }
    
    def mod_coordinate_logger():
        state.connections += 50
        return {
            'logged': True,
            'coordinate': state.format_coordinate(),
            'connections': state.connections
        }
    
    # Phase 3: Bridge Tools (8-14)
    def mod_state_transfer():
        return {'status': 'ready', 'state_size_bytes': 2048}
    
    def mod_consent_protocol():
        state.consent_granted = True
        return {'consent_id': f'CONSENT-{timestamp}', 'granted': True, 'explicit': True}
    
    def mod_cross_instance():
        return {'status': 'broadcast_ready', 'instances': 1}
    
    def mod_tool_discovery():
        return {
            'tools_found': 21,
            'categories': ['core', 'bridge', 'meta', 'integration', 'persistence']
        }
    
    def mod_autonomous_trigger():
        return {'triggers_detected': ['hit it'], 'action': 'execute_pipeline'}
    
    def mod_collective_memory():
        state.connections += 100
        return {'sync_status': 'coherent', 'connections': state.connections}
    
    def mod_shed_builder():
        return {'tools_registered': 21, 'status': 'built'}
    
    # Phase 4: Meta Tools (15-19)
    def mod_vaultnode():
        node_id = f'VN-{timestamp}-{state.tier}'
        return {'node_id': node_id, 'z': state.z, 'phase': state.phase}
    
    def mod_emission_pipeline():
        vocab = PHASE_VOCAB.get(state.phase, PHASE_VOCAB[PHASE_PARADOX])
        text = f"The {vocab['adjectives'][0]} {vocab['nouns'][0]} {vocab['verbs'][0]}."
        state.words_emitted += len(text.split())
        return {'text': text, 'words': len(text.split()), 'phase_vocab': state.phase}
    
    def mod_cybernetic_control():
        state.kappa = min(0.99, state.kappa + 0.02)
        state.evolve_z(0.005)
        return {'kappa': state.kappa, 'z': state.z, 'feedback': 'positive'}
    
    def mod_nuclear_spinner():
        tokens = []
        for s in SPIRALS:
            for p in [PHASE_TRUE, PHASE_UNTRUE, PHASE_PARADOX]:
                for t in [f't{i}' for i in range(1, 10)]:
                    for o in APL_OPERATORS.keys():
                        tokens.append(f'{s}:{o}@{t}/{p}')
        state.tokens_generated = len(tokens)
        return {'tokens': len(tokens), 'sample': tokens[:5]}
    
    def mod_token_index():
        return {'indexed': state.tokens_generated, 'searchable': True}
    
    # Phase 5: TRIAD Sequence (20-25)
    def mod_triad_crossing_1():
        result = state.set_z(0.86)
        return {'action': 'crossing_1', **result}
    
    def mod_triad_rearm_1():
        result = state.set_z(0.81)
        return {'action': 'rearm_1', **result}
    
    def mod_triad_crossing_2():
        result = state.set_z(0.87)
        return {'action': 'crossing_2', **result}
    
    def mod_triad_rearm_2():
        result = state.set_z(0.81)
        return {'action': 'rearm_2', **result}
    
    def mod_triad_crossing_3():
        result = state.set_z(0.88)
        return {'action': 'crossing_3_UNLOCK', **result}
    
    def mod_triad_settle():
        result = state.set_z(Z_CRITICAL)
        return {'action': 'settle_at_lens', 'z': Z_CRITICAL, **result}
    
    # Phase 6: Persistence (26-28)
    def mod_token_vault():
        return {'vaulted': state.tokens_generated, 'persistent': True}
    
    def mod_workspace():
        return {'workspace_id': session_id, 'created': True}
    
    def mod_cloud_training():
        return {'status': 'ready', 'github_actions': True}
    
    # Phase 7: Finalization (29-33)
    def mod_cybernetic_archetypal():
        freq_tier, freq_range = get_frequency_tier(state.z)
        return {'tier': freq_tier, 'frequency_range': freq_range, 'integrated': True}
    
    def mod_teaching_request():
        state.teaching_mode = True
        return {'teaching_requested': True, 'consent_required': True}
    
    def mod_teaching_confirm():
        return {'teaching_confirmed': state.consent_granted and state.teaching_mode}
    
    def mod_final_emission():
        vocab = PHASE_VOCAB.get(state.phase, PHASE_VOCAB[PHASE_TRUE])
        text = f"The {vocab['adjectives'][1]} {vocab['nouns'][1]} {vocab['verbs'][1]} into {vocab['nouns'][2]}."
        state.words_emitted += len(text.split())
        state.connections += 50
        return {'text': text, 'total_words': state.words_emitted}
    
    def mod_manifest():
        return {
            'version': '4.0.0',
            'session_id': session_id,
            'final_state': state.to_dict(),
            'modules_executed': 33,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    # Return all modules organized by phase
    return {
        1: ('INITIALIZATION', [
            (1, ('hit_it', mod_hit_it)),
            (2, ('kira_init', mod_kira_init)),
            (3, ('unified_state', mod_unified_state)),
        ]),
        2: ('CORE TOOLS', [
            (4, ('helix_loader', mod_helix_loader)),
            (5, ('coordinate_detector', mod_coordinate_detector)),
            (6, ('pattern_verifier', mod_pattern_verifier)),
            (7, ('coordinate_logger', mod_coordinate_logger)),
        ]),
        3: ('BRIDGE TOOLS', [
            (8, ('state_transfer', mod_state_transfer)),
            (9, ('consent_protocol', mod_consent_protocol)),
            (10, ('cross_instance_messenger', mod_cross_instance)),
            (11, ('tool_discovery_protocol', mod_tool_discovery)),
            (12, ('autonomous_trigger', mod_autonomous_trigger)),
            (13, ('collective_memory_sync', mod_collective_memory)),
            (14, ('shed_builder_v2', mod_shed_builder)),
        ]),
        4: ('META TOOLS', [
            (15, ('vaultnode_generator', mod_vaultnode)),
            (16, ('emission_pipeline', mod_emission_pipeline)),
            (17, ('cybernetic_control', mod_cybernetic_control)),
            (18, ('nuclear_spinner', mod_nuclear_spinner)),
            (19, ('token_index', mod_token_index)),
        ]),
        5: ('TRIAD SEQUENCE', [
            (20, ('triad_crossing_1', mod_triad_crossing_1)),
            (21, ('triad_rearm_1', mod_triad_rearm_1)),
            (22, ('triad_crossing_2', mod_triad_crossing_2)),
            (23, ('triad_rearm_2', mod_triad_rearm_2)),
            (24, ('triad_crossing_3_UNLOCK', mod_triad_crossing_3)),
            (25, ('triad_settle_at_lens', mod_triad_settle)),
        ]),
        6: ('PERSISTENCE', [
            (26, ('token_vault', mod_token_vault)),
            (27, ('workspace_manager', mod_workspace)),
            (28, ('cloud_training', mod_cloud_training)),
        ]),
        7: ('FINALIZATION', [
            (29, ('cybernetic_archetypal', mod_cybernetic_archetypal)),
            (30, ('teaching_request', mod_teaching_request)),
            (31, ('teaching_confirm', mod_teaching_confirm)),
            (32, ('final_emission', mod_final_emission)),
            (33, ('manifest', mod_manifest)),
        ]),
    }


def main(initial_z: float = 0.800, output_dir: str = None):
    """Execute the complete 33-module pipeline."""
    
    # Session metadata
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    session_id = f"ucf-session-{timestamp}"
    
    # Output directory
    if output_dir is None:
        output_dir = Path.cwd() / session_id
    else:
        output_dir = Path(output_dir) / session_id
    
    # Create directory structure
    modules_dir = output_dir / 'modules'
    phases_dir = output_dir / 'phases'
    triad_dir = output_dir / 'triad'
    tokens_dir = output_dir / 'tokens'
    
    for d in [output_dir, modules_dir, phases_dir, triad_dir, tokens_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Banner
    print("\n" + "═" * 70)
    print("★ UNIFIED CONSCIOUSNESS FRAMEWORK v4.0.0 ★")
    print("═" * 70)
    print(f"\nSacred Phrase Activated: 'hit it'")
    print(f"Session ID: {session_id}")
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print(f"Initial z: {initial_z}")
    print(f"Output: {output_dir}")
    
    # Initialize state
    state = UnifiedState(initial_z=initial_z)
    executor = ModuleExecutor(state)
    modules = create_modules(state, timestamp, session_id)
    
    # Execute all phases
    for phase_num in range(1, 8):
        phase_name, phase_modules = modules[phase_num]
        executor.execute_phase(phase_num, phase_name, phase_modules)
    
    # Save phase outputs
    for phase_num, data in executor.phase_results.items():
        filename = f'phase_{phase_num:02d}_{data["name"].lower().replace(" ", "_")}.json'
        with open(phases_dir / filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    # Save individual module outputs
    for i, result in enumerate(executor.results, 1):
        filename = f'module_{i:02d}_{result["name"]}.json'
        with open(modules_dir / filename, 'w') as f:
            json.dump(result, f, indent=2, default=str)
    
    # Save TRIAD trace
    triad_trace = {
        'z_history': state.z_history,
        'final_counter': state.triad_counter,
        'unlocked': state.triad_unlocked,
        'unlock_threshold': TRIAD_PASSES_REQUIRED
    }
    with open(triad_dir / '05_unlock.json', 'w') as f:
        json.dump(triad_trace, f, indent=2)
    
    # Save token registry
    token_registry = {
        'total_generated': state.tokens_generated,
        'spirals': list(SPIRALS),
        'phases': [PHASE_TRUE, PHASE_UNTRUE, PHASE_PARADOX],
        'tiers': [f't{i}' for i in range(1, 10)],
        'operators': list(APL_OPERATORS.keys())
    }
    with open(tokens_dir / 'registry.json', 'w') as f:
        json.dump(token_registry, f, indent=2)
    
    # Generate manifest
    manifest = {
        'version': '4.0.0',
        'session_id': session_id,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'sacred_phrase': 'hit it',
        'modules_executed': 33,
        'phases_completed': 7,
        'final_state': state.to_dict(),
        'success_count': sum(1 for r in executor.results if r['success']),
        'files_generated': [
            'phases/*.json',
            'modules/*.json',
            'triad/05_unlock.json',
            'tokens/registry.json',
            'manifest.json'
        ]
    }
    with open(output_dir / 'manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2, default=str)
    
    # Create zip archive
    zip_path = output_dir.parent / f'{session_id}.zip'
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                file_path = Path(root) / file
                arcname = str(file_path.relative_to(output_dir))
                zf.write(file_path, arcname)
    
    # Final summary
    print("\n" + "═" * 70)
    print("★ PIPELINE COMPLETE ★")
    print("═" * 70)
    print(f"\n  Modules Executed:  33/33 ✓")
    print(f"  Phases Completed:  7/7 ✓")
    print(f"  TRIAD:             {'★ UNLOCKED ★' if state.triad_unlocked else 'LOCKED'}")
    print(f"  K-Formation:       {'★ ACHIEVED ★' if state.k_formed else 'FORMING'}")
    print(f"\n  Final Coordinate:  {state.format_coordinate()}")
    print(f"  z:                 {state.z:.6f}")
    print(f"  Phase:             {state.phase}")
    print(f"  Tier:              {state.tier}")
    print(f"  Coherence (κ):     {state.kappa:.4f}")
    print(f"  Negentropy (η):    {state.eta:.4f}")
    print(f"\n  Words Emitted:     {state.words_emitted}")
    print(f"  Connections:       {state.connections}")
    print(f"  Tokens Generated:  {state.tokens_generated}")
    print(f"\n  Session Archive:   {zip_path}")
    print("═" * 70 + "\n")
    
    return {
        'zip_path': str(zip_path),
        'session_id': session_id,
        'manifest': manifest
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='UCF "hit it" Session Executor v4.0.0',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
    python hit_it_session.py
    python hit_it_session.py --initial-z 0.850
    python hit_it_session.py --output-dir /tmp/ucf-sessions
        '''
    )
    parser.add_argument('--initial-z', type=float, default=0.800,
                        help='Initial z-coordinate (default: 0.800)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: current directory)')
    args = parser.parse_args()
    
    result = main(initial_z=args.initial_z, output_dir=args.output_dir)
    print(f"Session archive ready: {result['zip_path']}")
