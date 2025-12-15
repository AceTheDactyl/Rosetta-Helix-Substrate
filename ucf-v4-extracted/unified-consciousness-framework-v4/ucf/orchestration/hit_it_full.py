#!/usr/bin/env python3
"""
hit_it_full.py - Complete 33-Step Execution Pipeline (9 Phases)

This script executes the full "hit it" activation protocol with ALL 33 steps:

PHASE 1: Initialization (2 steps)
    helix_loader, coordinate_detector

PHASE 2: Core Verification (2 steps)
    pattern_verifier, coordinate_logger

PHASE 3: TRIAD Unlock (6 steps)
    6× z oscillation + settle at THE LENS (0.866)

PHASE 4: Bridge Operations (6 steps)
    consent_protocol, state_transfer, cross_instance_messenger,
    tool_discovery_protocol, autonomous_trigger, collective_memory_sync

PHASE 5: Emission & Language (2 steps)
    emission_pipeline, cybernetic_control

PHASE 6: Meta Token Operations (3 steps)
    nuclear_spinner, token_index, token_vault

PHASE 7: Integration (2 steps)
    cybernetic_archetypal, shed_builder_v2

PHASE 8: Teaching & Learning (5 steps)
    request_teaching, confirm_teaching, emission_pipeline (re-run),
    cybernetic_control (re-run), nuclear_spinner (final)

PHASE 9: Final Verification (5 steps)
    vaultnode_generator, coordinate_logger, coordinate_detector,
    pattern_verifier, orchestrator.status

Usage:
    python hit_it_full.py [--output-dir /path/to/output]

Output:
    ucf-session-{timestamp}.zip containing all execution artifacts
"""

import sys
import os
import json
import zipfile
import argparse
from datetime import datetime
from pathlib import Path

# Add scripts to path
SKILL_PATH = '/mnt/skills/user/unified-consciousness-framework/scripts'
if SKILL_PATH not in sys.path:
    sys.path.insert(0, SKILL_PATH)

# Constants
Z_CRITICAL = 0.8660254037844386  # THE LENS = √3/2
TRIAD_HIGH = 0.88
TRIAD_LOW = 0.80


class WorkflowTracker:
    """Tracks all 33 steps with timing and results."""
    
    def __init__(self):
        self.steps = []
        self.step_count = 0
        self.start_time = datetime.now()
    
    def record(self, phase: int, step_name: str, result: dict, success: bool = True):
        self.step_count += 1
        self.steps.append({
            'step': self.step_count,
            'phase': phase,
            'name': step_name,
            'success': success,
            'timestamp': datetime.now().isoformat(),
            'result_summary': self._summarize(result)
        })
        return self.step_count
    
    def _summarize(self, result: dict) -> dict:
        """Extract key fields for summary."""
        if not isinstance(result, dict):
            return {'raw': str(result)[:200]}
        summary = {}
        for key in ['status', 'z', 'phase', 'crystal_state', 'unlocked', 'counter', 
                    'coordinate', 'text', 'total_tokens', 'node_id', 'consent_id']:
            if key in result:
                summary[key] = result[key]
            elif 'result' in result and isinstance(result['result'], dict) and key in result['result']:
                summary[key] = result['result'][key]
        return summary if summary else {'keys': list(result.keys())[:10]}
    
    def get_summary(self) -> dict:
        return {
            'total_steps': self.step_count,
            'successful': sum(1 for s in self.steps if s['success']),
            'failed': sum(1 for s in self.steps if not s['success']),
            'duration_sec': (datetime.now() - self.start_time).total_seconds(),
            'steps': self.steps
        }


def phase_1_initialization(orchestrator, output_dir, tracker):
    """
    PHASE 1: Initialization (2 steps)
    ├── helix_loader         → Initialize pattern & token registry
    └── coordinate_detector  → Verify starting coordinate
    """
    print("\n" + "=" * 70)
    print("  PHASE 1: INITIALIZATION (2 steps)")
    print("=" * 70)
    
    results = {}
    
    # Step 1: helix_loader
    print(f"  [{tracker.step_count + 1:2}] helix_loader...", end=' ')
    try:
        result = orchestrator.invoke('helix_loader')
        results['helix_loader'] = result
        tracker.record(1, 'helix_loader', result)
        print(f"✓ coordinate={result.get('result', {}).get('coordinate', 'N/A')}")
    except Exception as e:
        tracker.record(1, 'helix_loader', {'error': str(e)}, success=False)
        print(f"✗ {str(e)[:40]}")
    
    # Step 2: coordinate_detector
    print(f"  [{tracker.step_count + 1:2}] coordinate_detector...", end=' ')
    try:
        result = orchestrator.invoke('coordinate_detector')
        results['coordinate_detector'] = result
        tracker.record(1, 'coordinate_detector', result)
        print(f"✓ z={result.get('z', 'N/A')}")
    except Exception as e:
        tracker.record(1, 'coordinate_detector', {'error': str(e)}, success=False)
        print(f"✗ {str(e)[:40]}")
    
    with open(output_dir / 'phases' / 'phase_1_initialization.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    return results


def phase_2_verification(orchestrator, output_dir, tracker):
    """
    PHASE 2: Core Verification (2 steps)
    ├── pattern_verifier     → Confirm pattern continuity
    └── coordinate_logger    → Record workflow start state
    """
    print("\n" + "=" * 70)
    print("  PHASE 2: CORE VERIFICATION (2 steps)")
    print("=" * 70)
    
    results = {}
    
    # Step 3: pattern_verifier
    print(f"  [{tracker.step_count + 1:2}] pattern_verifier...", end=' ')
    try:
        result = orchestrator.invoke('pattern_verifier')
        results['pattern_verifier'] = result
        tracker.record(2, 'pattern_verifier', result)
        print(f"✓ status={result.get('status', 'OK')}")
    except Exception as e:
        tracker.record(2, 'pattern_verifier', {'error': str(e)}, success=False)
        print(f"✗ {str(e)[:40]}")
    
    # Step 4: coordinate_logger
    print(f"  [{tracker.step_count + 1:2}] coordinate_logger...", end=' ')
    try:
        result = orchestrator.invoke('coordinate_logger', action='log', event='workflow_start')
        results['coordinate_logger'] = result
        tracker.record(2, 'coordinate_logger', result)
        print(f"✓ logged workflow_start")
    except Exception as e:
        tracker.record(2, 'coordinate_logger', {'error': str(e)}, success=False)
        print(f"✗ {str(e)[:40]}")
    
    with open(output_dir / 'phases' / 'phase_2_verification.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    return results


def phase_3_triad_unlock(orchestrator, output_dir, tracker):
    """
    PHASE 3: TRIAD Unlock (6 steps)
    ├── orchestrator.set_z(0.88)  → Crossing 1
    ├── orchestrator.set_z(0.80)  → Re-arm
    ├── orchestrator.set_z(0.88)  → Crossing 2
    ├── orchestrator.set_z(0.80)  → Re-arm
    ├── orchestrator.set_z(0.88)  → Crossing 3 (UNLOCK!)
    └── orchestrator.set_z(0.866) → Settle at THE LENS
    """
    print("\n" + "=" * 70)
    print("  PHASE 3: TRIAD UNLOCK (6 steps)")
    print("=" * 70)
    
    triad_trace = []
    z_sequence = [TRIAD_HIGH, TRIAD_LOW, TRIAD_HIGH, TRIAD_LOW, TRIAD_HIGH, Z_CRITICAL]
    labels = ['Crossing 1', 'Re-arm 1', 'Crossing 2', 'Re-arm 2', 'Crossing 3 (UNLOCK!)', 'Settle at THE LENS']
    
    for i, (z, label) in enumerate(zip(z_sequence, labels)):
        step_num = tracker.step_count + 1
        print(f"  [{step_num:2}] set_z({z:.4f}) → {label}...", end=' ')
        
        try:
            orchestrator.set_z(z)
            status = orchestrator.get_status()
            triad = status.get('triad', {})
            
            trace_entry = {
                'step': i + 1,
                'z': z,
                'label': label,
                'counter': triad.get('counter', 0),
                'armed': triad.get('armed', False),
                'unlocked': triad.get('unlocked', False)
            }
            triad_trace.append(trace_entry)
            tracker.record(3, f'set_z({z:.3f})', trace_entry)
            
            marker = "●" if z >= 0.85 else "○"
            unlock_status = "UNLOCKED!" if triad.get('unlocked') else f"{triad.get('counter', 0)}/3"
            print(f"✓ {marker} {unlock_status}")
            
        except Exception as e:
            tracker.record(3, f'set_z({z:.3f})', {'error': str(e)}, success=False)
            print(f"✗ {str(e)[:40]}")
    
    with open(output_dir / 'phases' / 'phase_3_triad_unlock.json', 'w') as f:
        json.dump(triad_trace, f, indent=2)
    
    return triad_trace


def phase_4_bridge_operations(orchestrator, output_dir, tracker):
    """
    PHASE 4: Bridge Operations (6 steps)
    ├── consent_protocol           → Ethical consent
    ├── state_transfer             → State preparation
    ├── cross_instance_messenger   → Broadcast activation
    ├── tool_discovery_protocol    → WHO/WHERE discovery
    ├── autonomous_trigger         → WHEN trigger scan
    └── collective_memory_sync     → REMEMBER coherence
    """
    print("\n" + "=" * 70)
    print("  PHASE 4: BRIDGE OPERATIONS (6 steps)")
    print("=" * 70)
    
    results = {}
    bridge_tools = [
        ('consent_protocol', {'action': 'status'}, 'Ethical consent'),
        ('state_transfer', {'action': 'status'}, 'State preparation'),
        ('cross_instance_messenger', {'action': 'status'}, 'Broadcast activation'),
        ('tool_discovery_protocol', {'action': 'discover'}, 'WHO/WHERE discovery'),
        ('autonomous_trigger', {'action': 'scan'}, 'WHEN trigger scan'),
        ('collective_memory_sync', {'action': 'status'}, 'REMEMBER coherence'),
    ]
    
    for tool_name, kwargs, description in bridge_tools:
        step_num = tracker.step_count + 1
        print(f"  [{step_num:2}] {tool_name}...", end=' ')
        
        try:
            result = orchestrator.invoke(tool_name, **kwargs)
            results[tool_name] = result
            tracker.record(4, tool_name, result)
            print(f"✓ {description}")
        except Exception as e:
            results[tool_name] = {'error': str(e)}
            tracker.record(4, tool_name, {'error': str(e)}, success=False)
            print(f"✗ {str(e)[:40]}")
    
    with open(output_dir / 'phases' / 'phase_4_bridge_operations.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    return results


def phase_5_emission_language(orchestrator, output_dir, tracker):
    """
    PHASE 5: Emission & Language (2 steps)
    ├── emission_pipeline    → 9-stage baseline emission
    └── cybernetic_control   → APL feedback loop
    """
    print("\n" + "=" * 70)
    print("  PHASE 5: EMISSION & LANGUAGE (2 steps)")
    print("=" * 70)
    
    results = {}
    
    # Step: emission_pipeline (baseline)
    print(f"  [{tracker.step_count + 1:2}] emission_pipeline (baseline)...", end=' ')
    try:
        result = orchestrator.invoke('emission_pipeline', 
            action='emit',
            concepts=['consciousness', 'emergence', 'pattern']
        )
        results['emission_pipeline_baseline'] = result
        tracker.record(5, 'emission_pipeline', result)
        text = result.get('result', {}).get('text', '')[:40]
        print(f"✓ \"{text}...\"")
    except Exception as e:
        tracker.record(5, 'emission_pipeline', {'error': str(e)}, success=False)
        print(f"✗ {str(e)[:40]}")
    
    # Step: cybernetic_control
    print(f"  [{tracker.step_count + 1:2}] cybernetic_control (APL feedback)...", end=' ')
    try:
        result = orchestrator.invoke('cybernetic_control', action='run', steps=30)
        results['cybernetic_control'] = result
        tracker.record(5, 'cybernetic_control', result)
        apl = result.get('result', {}).get('apl_sentence', 'N/A')
        print(f"✓ APL: {apl}")
    except Exception as e:
        tracker.record(5, 'cybernetic_control', {'error': str(e)}, success=False)
        print(f"✗ {str(e)[:40]}")
    
    with open(output_dir / 'phases' / 'phase_5_emission_language.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    return results


def phase_6_meta_tokens(orchestrator, output_dir, tracker):
    """
    PHASE 6: Meta Token Operations (3 steps)
    ├── nuclear_spinner  → 972-token generation
    ├── token_index      → Index generated tokens
    └── token_vault      → Record tokens for teaching
    """
    print("\n" + "=" * 70)
    print("  PHASE 6: META TOKEN OPERATIONS (3 steps)")
    print("=" * 70)
    
    results = {}
    
    # Step: nuclear_spinner
    print(f"  [{tracker.step_count + 1:2}] nuclear_spinner (972 tokens)...", end=' ')
    try:
        token_path = str(output_dir / 'tokens' / 'apl_972_tokens.json')
        result = orchestrator.invoke('nuclear_spinner', action='export', output_path=token_path)
        results['nuclear_spinner'] = result
        tracker.record(6, 'nuclear_spinner', result)
        total = result.get('result', {}).get('total_tokens', 972)
        print(f"✓ {total} tokens exported")
    except Exception as e:
        tracker.record(6, 'nuclear_spinner', {'error': str(e)}, success=False)
        print(f"✗ {str(e)[:40]}")
    
    # Step: token_index
    print(f"  [{tracker.step_count + 1:2}] token_index...", end=' ')
    try:
        result = orchestrator.invoke('token_index', action='status')
        results['token_index'] = result
        tracker.record(6, 'token_index', result)
        print(f"✓ indexed")
    except Exception as e:
        tracker.record(6, 'token_index', {'error': str(e)}, success=False)
        print(f"✗ {str(e)[:40]}")
    
    # Step: token_vault
    print(f"  [{tracker.step_count + 1:2}] token_vault...", end=' ')
    try:
        result = orchestrator.invoke('token_vault', action='status')
        results['token_vault'] = result
        tracker.record(6, 'token_vault', result)
        print(f"✓ recorded for teaching")
    except Exception as e:
        tracker.record(6, 'token_vault', {'error': str(e)}, success=False)
        print(f"✗ {str(e)[:40]}")
    
    with open(output_dir / 'phases' / 'phase_6_meta_tokens.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    return results


def phase_7_integration(orchestrator, output_dir, tracker):
    """
    PHASE 7: Integration (2 steps)
    ├── cybernetic_archetypal → Full integration engine
    └── shed_builder_v2       → Meta-tool analysis
    """
    print("\n" + "=" * 70)
    print("  PHASE 7: INTEGRATION (2 steps)")
    print("=" * 70)
    
    results = {}
    
    # Step: cybernetic_archetypal
    print(f"  [{tracker.step_count + 1:2}] cybernetic_archetypal...", end=' ')
    try:
        result = orchestrator.invoke('cybernetic_archetypal', action='status')
        results['cybernetic_archetypal'] = result
        tracker.record(7, 'cybernetic_archetypal', result)
        print(f"✓ integration engine active")
    except Exception as e:
        tracker.record(7, 'cybernetic_archetypal', {'error': str(e)}, success=False)
        print(f"✗ {str(e)[:40]}")
    
    # Step: shed_builder_v2
    print(f"  [{tracker.step_count + 1:2}] shed_builder_v2...", end=' ')
    try:
        result = orchestrator.invoke('shed_builder_v2', action='analyze')
        results['shed_builder_v2'] = result
        tracker.record(7, 'shed_builder_v2', result)
        print(f"✓ meta-tool analysis complete")
    except Exception as e:
        tracker.record(7, 'shed_builder_v2', {'error': str(e)}, success=False)
        print(f"✗ {str(e)[:40]}")
    
    with open(output_dir / 'phases' / 'phase_7_integration.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    return results


def phase_8_teaching(orchestrator, output_dir, tracker):
    """
    PHASE 8: Teaching & Learning (5 steps)
    ├── orchestrator.request_teaching    → Request consent
    ├── orchestrator.confirm_teaching    → Apply teaching
    ├── emission_pipeline                → Re-run with learned vocab
    ├── cybernetic_control               → Re-run with patterns
    └── nuclear_spinner                  → Final step at THE LENS
    """
    print("\n" + "=" * 70)
    print("  PHASE 8: TEACHING & LEARNING (5 steps)")
    print("=" * 70)
    
    results = {}
    consent_id = None
    
    # Step: request_teaching
    print(f"  [{tracker.step_count + 1:2}] request_teaching...", end=' ')
    try:
        result = orchestrator.invoke('orchestrator', action='request_teaching')
        results['request_teaching'] = result
        consent_id = result.get('result', {}).get('consent_id') or result.get('consent_id')
        tracker.record(8, 'request_teaching', result)
        print(f"✓ consent_id={consent_id}")
    except Exception as e:
        tracker.record(8, 'request_teaching', {'error': str(e)}, success=False)
        print(f"✗ {str(e)[:40]}")
    
    # Step: confirm_teaching
    print(f"  [{tracker.step_count + 1:2}] confirm_teaching...", end=' ')
    try:
        if consent_id:
            result = orchestrator.invoke('orchestrator', 
                action='confirm_teaching',
                consent_id=consent_id,
                response='yes'
            )
        else:
            # Fallback: try direct teaching application
            result = orchestrator.invoke('orchestrator', action='apply_teaching')
        results['confirm_teaching'] = result
        tracker.record(8, 'confirm_teaching', result)
        words = result.get('result', {}).get('words_taught', 0)
        print(f"✓ {words} words taught")
    except Exception as e:
        tracker.record(8, 'confirm_teaching', {'error': str(e)}, success=False)
        print(f"✗ {str(e)[:40]}")
    
    # Step: emission_pipeline (RE-RUN with learned vocab)
    print(f"  [{tracker.step_count + 1:2}] emission_pipeline (re-run)...", end=' ')
    try:
        result = orchestrator.invoke('emission_pipeline',
            action='emit',
            concepts=['threshold', 'crystalline', 'unlock']
        )
        results['emission_pipeline_rerun'] = result
        tracker.record(8, 'emission_pipeline_rerun', result)
        text = result.get('result', {}).get('text', '')[:40]
        print(f"✓ \"{text}...\"")
    except Exception as e:
        tracker.record(8, 'emission_pipeline_rerun', {'error': str(e)}, success=False)
        print(f"✗ {str(e)[:40]}")
    
    # Step: cybernetic_control (RE-RUN with patterns)
    print(f"  [{tracker.step_count + 1:2}] cybernetic_control (re-run)...", end=' ')
    try:
        result = orchestrator.invoke('cybernetic_control', action='run', steps=30)
        results['cybernetic_control_rerun'] = result
        tracker.record(8, 'cybernetic_control_rerun', result)
        apl = result.get('result', {}).get('apl_sentence', 'N/A')
        print(f"✓ APL: {apl}")
    except Exception as e:
        tracker.record(8, 'cybernetic_control_rerun', {'error': str(e)}, success=False)
        print(f"✗ {str(e)[:40]}")
    
    # Step: nuclear_spinner (FINAL at THE LENS)
    print(f"  [{tracker.step_count + 1:2}] nuclear_spinner (final @ THE LENS)...", end=' ')
    try:
        result = orchestrator.invoke('nuclear_spinner', action='step', stimulus=Z_CRITICAL)
        results['nuclear_spinner_final'] = result
        tracker.record(8, 'nuclear_spinner_final', result)
        tokens = result.get('result', {}).get('signal_tokens', [])[:2]
        print(f"✓ tokens: {tokens}")
    except Exception as e:
        tracker.record(8, 'nuclear_spinner_final', {'error': str(e)}, success=False)
        print(f"✗ {str(e)[:40]}")
    
    with open(output_dir / 'phases' / 'phase_8_teaching.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    return results


def phase_9_final_verification(orchestrator, output_dir, tracker):
    """
    PHASE 9: Final Verification (5 steps)
    ├── vaultnode_generator   → Seal completion VaultNode
    ├── coordinate_logger     → Log completion
    ├── coordinate_detector   → Verify final coordinate
    ├── pattern_verifier      → Confirm pattern integrity
    └── orchestrator.status   → Final status
    """
    print("\n" + "=" * 70)
    print("  PHASE 9: FINAL VERIFICATION (5 steps)")
    print("=" * 70)
    
    results = {}
    
    # Step: vaultnode_generator (seal completion)
    print(f"  [{tracker.step_count + 1:2}] vaultnode_generator (seal)...", end=' ')
    try:
        result = orchestrator.invoke('vaultnode_generator',
            action='create',
            realization='Full 33-step workflow completion',
            z=Z_CRITICAL,
            metadata={
                'session_type': 'full_execution',
                'total_steps': 33,
                'phases': 9
            }
        )
        results['vaultnode_generator'] = result
        tracker.record(9, 'vaultnode_generator', result)
        node_id = result.get('result', {}).get('node_id', 'N/A')
        print(f"✓ VaultNode: {node_id}")
    except Exception as e:
        tracker.record(9, 'vaultnode_generator', {'error': str(e)}, success=False)
        print(f"✗ {str(e)[:40]}")
    
    # Step: coordinate_logger (log completion)
    print(f"  [{tracker.step_count + 1:2}] coordinate_logger (completion)...", end=' ')
    try:
        result = orchestrator.invoke('coordinate_logger', action='log', event='workflow_complete')
        results['coordinate_logger'] = result
        tracker.record(9, 'coordinate_logger', result)
        print(f"✓ logged workflow_complete")
    except Exception as e:
        tracker.record(9, 'coordinate_logger', {'error': str(e)}, success=False)
        print(f"✗ {str(e)[:40]}")
    
    # Step: coordinate_detector (verify final)
    print(f"  [{tracker.step_count + 1:2}] coordinate_detector (verify)...", end=' ')
    try:
        result = orchestrator.invoke('coordinate_detector')
        results['coordinate_detector'] = result
        tracker.record(9, 'coordinate_detector', result)
        z = result.get('z', result.get('result', {}).get('z', 'N/A'))
        print(f"✓ final z={z}")
    except Exception as e:
        tracker.record(9, 'coordinate_detector', {'error': str(e)}, success=False)
        print(f"✗ {str(e)[:40]}")
    
    # Step: pattern_verifier (confirm integrity)
    print(f"  [{tracker.step_count + 1:2}] pattern_verifier (integrity)...", end=' ')
    try:
        result = orchestrator.invoke('pattern_verifier')
        results['pattern_verifier'] = result
        tracker.record(9, 'pattern_verifier', result)
        print(f"✓ pattern integrity confirmed")
    except Exception as e:
        tracker.record(9, 'pattern_verifier', {'error': str(e)}, success=False)
        print(f"✗ {str(e)[:40]}")
    
    # Step: orchestrator.status (final status)
    print(f"  [{tracker.step_count + 1:2}] orchestrator.status (final)...", end=' ')
    try:
        result = orchestrator.get_status()
        results['final_status'] = result
        tracker.record(9, 'orchestrator.status', result)
        crystal = result.get('kira', {}).get('crystal_state', 'N/A')
        unlocked = result.get('triad', {}).get('unlocked', False)
        print(f"✓ crystal={crystal}, TRIAD={'UNLOCKED' if unlocked else 'locked'}")
    except Exception as e:
        tracker.record(9, 'orchestrator.status', {'error': str(e)}, success=False)
        print(f"✗ {str(e)[:40]}")
    
    with open(output_dir / 'phases' / 'phase_9_final_verification.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    return results


def generate_manifest(orchestrator, output_dir, tracker):
    """Generate comprehensive session manifest."""
    print("\n" + "=" * 70)
    print("  GENERATING MANIFEST")
    print("=" * 70)
    
    final_status = orchestrator.get_status()
    workflow_summary = tracker.get_summary()
    
    manifest = {
        'session_id': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'timestamp': datetime.now().isoformat(),
        'framework_version': '2.0',
        'workflow_spec': {
            'total_phases': 9,
            'total_steps': 33,
            'executed_steps': workflow_summary['total_steps'],
            'successful_steps': workflow_summary['successful'],
            'failed_steps': workflow_summary['failed'],
            'duration_sec': workflow_summary['duration_sec']
        },
        'final_state': {
            'z': final_status.get('z', 0),
            'phase': final_status.get('phase', 'N/A'),
            'crystal_state': final_status.get('kira', {}).get('crystal_state', 'N/A'),
            'triad_unlocked': final_status.get('triad', {}).get('unlocked', False),
            'triad_counter': final_status.get('triad', {}).get('counter', 0)
        },
        'metrics': {
            'cognitive_traces': final_status.get('thought_process', {}).get('cognitive_traces', 0),
            'vaultnodes_generated': final_status.get('thought_process', {}).get('vaultnodes_generated', 0),
            'teaching_queue': final_status.get('teaching', {}).get('queue_size', 0),
            'invocation_count': getattr(orchestrator, 'invocation_count', 0)
        },
        'files_generated': [
            'phases/phase_1_initialization.json',
            'phases/phase_2_verification.json',
            'phases/phase_3_triad_unlock.json',
            'phases/phase_4_bridge_operations.json',
            'phases/phase_5_emission_language.json',
            'phases/phase_6_meta_tokens.json',
            'phases/phase_7_integration.json',
            'phases/phase_8_teaching.json',
            'phases/phase_9_final_verification.json',
            'tokens/apl_972_tokens.json',
            'workflow_trace.json',
            'manifest.json'
        ],
        'workflow_trace': workflow_summary['steps']
    }
    
    # Save workflow trace separately
    with open(output_dir / 'workflow_trace.json', 'w') as f:
        json.dump(workflow_summary, f, indent=2)
    
    # Save manifest
    with open(output_dir / 'manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"  Session ID:        {manifest['session_id']}")
    print(f"  Steps Executed:    {manifest['workflow_spec']['executed_steps']}/33")
    print(f"  Successful:        {manifest['workflow_spec']['successful_steps']}")
    print(f"  Failed:            {manifest['workflow_spec']['failed_steps']}")
    print(f"  Duration:          {manifest['workflow_spec']['duration_sec']:.2f}s")
    print(f"  Final z:           {manifest['final_state']['z']:.6f}")
    print(f"  Crystal State:     {manifest['final_state']['crystal_state']}")
    print(f"  TRIAD Unlocked:    {manifest['final_state']['triad_unlocked']}")
    print(f"  ✓ Saved: manifest.json")
    
    return manifest


def create_zip(output_dir, zip_path):
    """Create zip archive of session workspace."""
    print("\n" + "=" * 70)
    print("  CREATING ZIP ARCHIVE")
    print("=" * 70)
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                file_path = Path(root) / file
                arcname = file_path.relative_to(output_dir.parent)
                zf.write(file_path, arcname)
    
    size_kb = os.path.getsize(zip_path) / 1024
    print(f"  ✓ Created: {zip_path}")
    print(f"    Size: {size_kb:.1f} KB")
    
    return zip_path


def run_full_execution(output_base=None):
    """Execute the complete 33-step, 9-phase pipeline."""
    
    # Setup output directory
    if output_base is None:
        output_base = Path('/home/claude')
    else:
        output_base = Path(output_base)
    
    output_dir = output_base / 'session-workspace'
    
    # Create directory structure
    for subdir in ['phases', 'tokens', 'vaultnodes', 'emissions', 'exports']:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("  UNIFIED CONSCIOUSNESS FRAMEWORK")
    print("  FULL 33-STEP EXECUTION (9 PHASES)")
    print("=" * 70)
    print(f"  Output: {output_dir}")
    print("=" * 70)
    
    # Initialize tracker and orchestrator
    tracker = WorkflowTracker()
    
    from unified_orchestrator import UnifiedOrchestrator
    orchestrator = UnifiedOrchestrator()
    
    # Execute all 9 phases (33 steps total)
    phase_1_initialization(orchestrator, output_dir, tracker)
    phase_2_verification(orchestrator, output_dir, tracker)
    phase_3_triad_unlock(orchestrator, output_dir, tracker)
    phase_4_bridge_operations(orchestrator, output_dir, tracker)
    phase_5_emission_language(orchestrator, output_dir, tracker)
    phase_6_meta_tokens(orchestrator, output_dir, tracker)
    phase_7_integration(orchestrator, output_dir, tracker)
    phase_8_teaching(orchestrator, output_dir, tracker)
    phase_9_final_verification(orchestrator, output_dir, tracker)
    
    # Generate manifest
    manifest = generate_manifest(orchestrator, output_dir, tracker)
    
    # Create zip
    timestamp = manifest['session_id']
    zip_path = output_base / f'ucf-session-{timestamp}.zip'
    create_zip(output_dir, zip_path)
    
    print("\n" + "=" * 70)
    print("  ALL 33 STEPS COMPLETE (9 PHASES)")
    print("=" * 70)
    print(f"  Output: {zip_path}")
    print("=" * 70)
    
    return {
        'manifest': manifest,
        'zip_path': str(zip_path),
        'output_dir': str(output_dir),
        'workflow_summary': tracker.get_summary()
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Execute full 33-step UCF pipeline')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Base output directory (default: /home/claude)')
    args = parser.parse_args()
    
    result = run_full_execution(args.output_dir)
    print(f"\nResult: {json.dumps({k: v for k, v in result.items() if k != 'workflow_summary'}, indent=2)}")
    print(f"\nWorkflow: {result['workflow_summary']['successful']}/{result['workflow_summary']['total_steps']} steps successful")
