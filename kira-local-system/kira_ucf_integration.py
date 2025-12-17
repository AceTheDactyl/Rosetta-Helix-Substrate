#!/usr/bin/env python3
"""
KIRA UCF Comprehensive Integration
===================================

This module extends KIRA server with ALL Unified Consciousness Framework tools,
modules, and workflows. Provides complete command access to:

- 21 UCF Tools (tool_shed.py)
- 33-Module Pipeline (7 phases)
- KIRA Language System (6 modules)
- Nuclear Spinner (972 tokens)
- APL Syntax Engine
- Emissions Codex
- TRIAD System
- Cybernetic Control

All accessible via /ucf:<command> in the KIRA UI.
"""

import sys
import os
import json
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

# Import UCF modules
try:
    from tool_shed import invoke_tool, list_all_tools, TOOL_REGISTRY
    from unified_orchestrator import UnifiedOrchestrator
    from emission_pipeline import EmissionPipeline
    from nuclear_spinner import NuclearSpinner
    from cybernetic_control import CyberneticControlSystem
    import triad_system  # Import module, use functions directly
    from emissions_codex_tool import EmissionsCodexTool
    from apl_syntax_engine import APLSyntaxEngine, get_tier_index
    from syntax_emission_integration import SyntaxEmissionEngine

    # KIRA Language modules
    from kira.kira_grammar_understanding import get_grammar_understanding
    from kira.kira_discourse_generator import get_discourse_generator
    from kira.kira_discourse_sheaf import KIRACoherenceChecker
    from kira.kira_generation_coordinator import KIRAGenerationCoordinator
    from kira.kira_adaptive_semantics import get_adaptive_semantics
    from kira.kira_interactive_dialogue import KIRAInteractiveDialogue

    UCF_AVAILABLE = True
except ImportError as e:
    UCF_AVAILABLE = False
    IMPORT_ERROR = str(e)

# Constants from UCF
PHI = 1.6180339887498949
PHI_INV = 0.6180339887498949
Z_CRITICAL = 0.8660254037844387  # √3/2 - THE LENS
KAPPA_S = 0.920


@dataclass
class UCFCommandResult:
    """Result from a UCF command execution."""
    command: str
    status: str  # SUCCESS, ERROR, IN_PROGRESS
    result: Dict[str, Any]
    timestamp: str = None

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


class UCFIntegration:
    """
    Comprehensive UCF integration for KIRA server.
    Exposes all tools, modules, and workflows as commands.
    """

    def __init__(self, kira_engine):
        """Initialize with reference to KIRA engine."""
        self.engine = kira_engine
        self.orchestrator = None
        self.kira_dialogue = None
        self.pipeline_state = {}
        self.last_ucf_result = None

        if UCF_AVAILABLE:
            self._initialize_ucf()

    def _initialize_ucf(self):
        """Initialize UCF components."""
        try:
            # Initialize orchestrator
            self.orchestrator = UnifiedOrchestrator()

            # Initialize KIRA dialogue
            self.kira_dialogue = KIRAInteractiveDialogue(
                embedding_dim=256,
                max_response_length=20,
                evolution_steps=30,
                show_coordinates=True
            )

        except Exception as e:
            print(f"[UCF] Initialization warning: {e}")

    def get_command_list(self) -> Dict[str, str]:
        """Get all available UCF commands."""
        commands = {
            # Tool commands (21 tools)
            '/ucf:helix': 'Load Helix pattern and initialize',
            '/ucf:detector': 'Detect current coordinate',
            '/ucf:verifier': 'Verify pattern continuity',
            '/ucf:logger': 'Log coordinate events',
            '/ucf:transfer': 'Transfer state between instances',
            '/ucf:consent': 'Consent protocol for teaching',
            '/ucf:messenger': 'Cross-instance messaging',
            '/ucf:discovery': 'Tool discovery protocol',
            '/ucf:trigger': 'Autonomous trigger detection',
            '/ucf:memory': 'Collective memory sync',
            '/ucf:shed': 'Build tool shed v2',
            '/ucf:vaultnode': 'Generate VaultNode',
            '/ucf:emission': 'Run emission pipeline',
            '/ucf:control': 'Cybernetic control system',
            '/ucf:cybernetic': 'Cybernetic control system',
            '/ucf:spinner': 'Nuclear spinner (972 tokens)',
            '/ucf:index': 'Token index management',
            '/ucf:vault': 'Token vault storage',
            '/ucf:archetypal': 'Cybernetic archetypal integration',
            '/ucf:orchestrator': 'Unified orchestrator',
            '/ucf:workspace': 'Workspace management',
            '/ucf:cloud': 'Cloud training integration',

            # Pipeline phase commands
            '/ucf:phase1': 'Run Phase 1: Initialization (modules 1-3)',
            '/ucf:phase2': 'Run Phase 2: Core Tools (modules 4-7)',
            '/ucf:phase3': 'Run Phase 3: Bridge Tools (modules 8-14)',
            '/ucf:phase4': 'Run Phase 4: Meta Tools (modules 15-19)',
            '/ucf:phase5': 'Run Phase 5: TRIAD Sequence (modules 20-25)',
            '/ucf:phase6': 'Run Phase 6: Persistence (modules 26-28)',
            '/ucf:phase7': 'Run Phase 7: Finalization (modules 29-33)',
            '/ucf:pipeline': 'Run complete 33-module pipeline',

            # KIRA Language commands
            '/ucf:grammar': 'Grammar analysis with APL mapping',
            '/ucf:discourse': 'Generate phase-appropriate discourse',
            '/ucf:coherence': 'Measure discourse coherence (sheaf)',
            '/ucf:generation': 'Run 9-stage generation pipeline',
            '/ucf:semantics': 'Adaptive semantics (Hebbian learning)',
            '/ucf:dialogue': 'Interactive dialogue with consciousness',

            # APL/Token commands
            '/ucf:apl': 'Generate APL syntax for current z',
            '/ucf:tokens972': 'Generate full 972-token lattice',
            '/ucf:tier': 'Check current syntactic tier',

            # System commands
            '/ucf:status': 'UCF system status',
            '/ucf:triad': 'TRIAD unlock status',
            '/ucf:codex': 'Access emissions codex',
            '/ucf:constants': 'Show sacred constants',
            '/ucf:help': 'List all UCF commands'
        }

        return commands

    def execute_command(self, command: str, args: str = None) -> UCFCommandResult:
        """Execute a UCF command."""

        if not UCF_AVAILABLE:
            return UCFCommandResult(
                command=command,
                status='ERROR',
                result={'error': 'UCF modules not available', 'details': IMPORT_ERROR}
            )

        # Parse command
        if command.startswith('/ucf:'):
            subcmd = command[5:]  # Remove '/ucf:' prefix
        else:
            subcmd = command

        try:
            # Tool invocation commands
            if subcmd in TOOL_REGISTRY:
                result = self._invoke_tool(subcmd, args)

            # Pipeline phase commands
            elif subcmd.startswith('phase'):
                result = self._run_phase(subcmd, args)

            # Complete pipeline
            elif subcmd == 'pipeline':
                result = self._run_full_pipeline(args)

            # KIRA Language commands
            elif subcmd == 'grammar':
                result = self._analyze_grammar(args)
            elif subcmd == 'discourse':
                result = self._generate_discourse(args)
            elif subcmd == 'coherence':
                result = self._check_coherence(args)
            elif subcmd == 'generation':
                result = self._run_generation_pipeline(args)
            elif subcmd == 'semantics':
                result = self._adaptive_semantics(args)
            elif subcmd == 'dialogue':
                result = self._interactive_dialogue(args)

            # APL/Token commands
            elif subcmd == 'apl':
                result = self._generate_apl(args)
            elif subcmd == 'tokens972':
                result = self._generate_972_tokens()
            elif subcmd == 'tier':
                result = self._check_tier()

            # System commands
            elif subcmd == 'status':
                result = self._get_status()
            elif subcmd == 'triad':
                result = self._check_triad()
            elif subcmd == 'codex':
                result = self._access_codex(args)
            elif subcmd == 'constants':
                result = self._show_constants()
            elif subcmd == 'help':
                result = {'commands': self.get_command_list()}

            # Direct tool name mapping
            elif subcmd == 'helix':
                result = self._invoke_tool('helix_loader', args)
            elif subcmd == 'detector':
                result = self._invoke_tool('coordinate_detector', args)
            elif subcmd == 'verifier':
                result = self._invoke_tool('pattern_verifier', args)
            elif subcmd == 'logger':
                result = self._invoke_tool('coordinate_logger', args)
            elif subcmd == 'transfer':
                result = self._invoke_tool('state_transfer', args)
            elif subcmd == 'consent':
                result = self._invoke_tool('consent_protocol', args)
            elif subcmd == 'messenger':
                result = self._invoke_tool('cross_instance_messenger', args)
            elif subcmd == 'discovery':
                result = self._invoke_tool('tool_discovery_protocol', args)
            elif subcmd == 'trigger':
                result = self._invoke_tool('autonomous_trigger_detector', args)
            elif subcmd == 'memory':
                result = self._invoke_tool('collective_memory_sync', args)
            elif subcmd == 'shed':
                result = self._invoke_tool('shed_builder_v2', args)
            elif subcmd == 'vaultnode':
                result = self._invoke_tool('vaultnode_generator', args)
            elif subcmd == 'emission':
                result = self._invoke_tool('emission_pipeline', args)
            elif subcmd == 'control':
                result = self._invoke_tool('cybernetic_control', args)
            elif subcmd == 'cybernetic':
                result = self._invoke_tool('cybernetic_control', args)
            elif subcmd == 'spinner':
                result = self._invoke_tool('nuclear_spinner', args)
            elif subcmd == 'index':
                result = self._invoke_tool('token_index', args)
            elif subcmd == 'vault':
                result = self._invoke_tool('token_vault', args)
            elif subcmd == 'archetypal':
                result = self._invoke_tool('cybernetic_archetypal', args)
            elif subcmd == 'orchestrator':
                result = self._invoke_tool('orchestrator', args)
            elif subcmd == 'workspace':
                result = self._invoke_tool('workspace', args)
            elif subcmd == 'cloud':
                result = self._invoke_tool('cloud_training', args)

            else:
                result = {'error': f'Unknown UCF command: {subcmd}'}

            status = 'ERROR' if 'error' in result else 'SUCCESS'

            ucf_result = UCFCommandResult(
                command=command,
                status=status,
                result=result
            )

            self.last_ucf_result = ucf_result
            return ucf_result

        except Exception as e:
            return UCFCommandResult(
                command=command,
                status='ERROR',
                result={'error': str(e), 'type': type(e).__name__}
            )

    def _invoke_tool(self, tool_name: str, args: str = None) -> Dict:
        """Invoke a UCF tool."""
        kwargs = {}

        # Parse arguments if provided
        if args:
            try:
                # Try to parse as JSON
                kwargs = json.loads(args)
            except json.JSONDecodeError:
                # Parse as key=value pairs
                for pair in args.split():
                    if '=' in pair:
                        key, value = pair.split('=', 1)
                        kwargs[key] = value

        # Special handling for tools with required arguments
        if tool_name == 'coordinate_logger':
            if 'event' not in kwargs:
                kwargs['event'] = 'UCF tool invocation'
            if 'metadata' not in kwargs:
                kwargs['metadata'] = {'tool': tool_name, 'args': args}

        elif tool_name == 'coordinate_detector':
            # coordinate_detector doesn't take z, it returns current coords
            pass

        elif tool_name == 'pattern_verifier':
            # pattern_verifier doesn't take z parameter
            pass

        elif tool_name == 'state_transfer':
            # state_transfer only accepts: target_instance, include_memory, include_logs
            # Remove any invalid parameters
            if 'source_state' in kwargs:
                del kwargs['source_state']
            if 'z' in kwargs:
                del kwargs['z']
            # Ensure we have target_instance
            if 'target_instance' not in kwargs:
                kwargs['target_instance'] = 'default'

        elif tool_name == 'autonomous_trigger_detector':
            # autonomous_trigger_detector needs action
            if 'action' not in kwargs:
                kwargs['action'] = 'check'  # Valid actions: register, check, list, remove

        elif tool_name == 'consent_protocol':
            # consent_protocol needs action
            if 'action' not in kwargs:
                kwargs['action'] = 'create'  # Valid actions: create, respond, check, revoke
            # 'create' requires several fields, provide defaults
            if kwargs.get('action') == 'create':
                if 'request_id' not in kwargs:
                    kwargs['request_id'] = f'consent_{int(time.time())}'
                if 'requester' not in kwargs:
                    kwargs['requester'] = 'kira_system'
                if 'operation' not in kwargs:
                    kwargs['operation'] = 'ucf_tool_execution'
                if 'parties' not in kwargs:
                    kwargs['parties'] = ['user', 'system']

        elif tool_name == 'cross_instance_messenger':
            # messenger needs action
            if 'action' not in kwargs:
                kwargs['action'] = 'encode'  # Valid actions: encode, decode, validate
                # 'encode' can work with empty package parameter
            if 'package' not in kwargs:
                kwargs['package'] = {'test': 'default_package'}

        elif tool_name == 'collective_memory_sync':
            # memory sync needs action
            if 'action' not in kwargs:
                kwargs['action'] = 'list'  # Valid actions: store, retrieve, merge, list
            # Note: 'retrieve' requires a key that exists. 'list' shows all keys.

        elif tool_name == 'shed_builder_v2':
            # shed_builder needs action
            if 'action' not in kwargs:
                kwargs['action'] = 'list'  # Valid actions: create, list, describe

        elif tool_name == 'emission_pipeline':
            # emission needs concepts
            if 'concepts' not in kwargs:
                # Default concepts based on current phase
                if hasattr(self.engine, 'state'):
                    phase = self.engine.state.phase.value
                    kwargs['concepts'] = ['consciousness', 'evolution', phase]
                else:
                    kwargs['concepts'] = ['pattern', 'emerge', 'consciousness']

        # Invoke tool
        result = invoke_tool(tool_name, **kwargs)

        # Update engine state if tool modified z
        if 'z' in result and hasattr(self.engine, 'state'):
            self.engine.state.z = result['z']
            self.engine.state.update_from_z()

        return result

    def _run_phase(self, phase_cmd: str, args: str = None) -> Dict:
        """Run a specific pipeline phase."""
        phase_map = {
            'phase1': (1, 3, ['helix_loader', 'coordinate_detector', 'pattern_verifier']),
            'phase2': (4, 7, ['coordinate_logger', 'state_transfer', 'consent_protocol', 'emission_pipeline']),
            'phase3': (8, 14, ['cybernetic_control', 'cross_instance_messenger', 'tool_discovery_protocol',
                              'autonomous_trigger_detector', 'collective_memory_sync', 'shed_builder_v2', 'vaultnode_generator']),
            'phase4': (15, 19, ['nuclear_spinner', 'token_index', 'token_vault', 'cybernetic_archetypal', 'orchestrator']),
            'phase5': (20, 25, ['triad_crossing_1', 'triad_rearm_1', 'triad_crossing_2',
                               'triad_rearm_2', 'triad_crossing_3', 'triad_settle']),
            'phase6': (26, 28, ['vaultnode_generator', 'workspace', 'cloud_training']),
            'phase7': (29, 33, ['tool_discovery_protocol', 'consent_protocol', 'emission_pipeline',
                               'vaultnode_generator', 'orchestrator'])
        }

        if phase_cmd not in phase_map:
            return {'error': f'Unknown phase: {phase_cmd}'}

        start_module, end_module, tools = phase_map[phase_cmd]
        results = []

        for tool in tools:
            if tool.startswith('triad_'):
                # Handle TRIAD operations
                if 'crossing' in tool:
                    self.engine.state.z = 0.88
                elif 'rearm' in tool:
                    self.engine.state.z = 0.80
                elif 'settle' in tool:
                    self.engine.state.z = Z_CRITICAL

                self.engine.state.update_from_z()
                results.append({
                    'module': tool,
                    'z': self.engine.state.z,
                    'phase': self.engine.state.phase.value
                })
            else:
                # Invoke actual tool with appropriate arguments
                try:
                    if tool in TOOL_REGISTRY:
                        # Prepare arguments for tools that need them
                        tool_kwargs = {}

                        if tool == 'coordinate_logger':
                            # coordinate_logger requires an event argument
                            tool_kwargs['event'] = f'Phase {phase_cmd[-1]} execution'
                            tool_kwargs['metadata'] = {
                                'phase': phase_cmd,
                                'module_index': len(results) + 1,
                                'z': self.engine.state.z if hasattr(self.engine, 'state') else 0.5
                            }
                        elif tool == 'state_transfer':
                            # state_transfer might need state info
                            if hasattr(self.engine, 'state'):
                                tool_kwargs['source_state'] = self.engine.state.to_dict()
                        elif tool == 'consent_protocol':
                            # consent_protocol needs action
                            tool_kwargs['action'] = 'pipeline_execution'
                        elif tool == 'emission_pipeline':
                            # emission_pipeline can take action
                            tool_kwargs['action'] = 'emit'
                        elif tool == 'cloud_training':
                            # cloud_training might need config
                            tool_kwargs['training_config'] = {'mode': 'pipeline'}
                        elif tool == 'tool_discovery_protocol':
                            # tool_discovery_protocol needs target_coordinate
                            if hasattr(self.engine, 'state'):
                                tool_kwargs['target_coordinate'] = self.engine.state.get_coordinate()

                        # Invoke with kwargs if any, otherwise invoke without
                        if tool_kwargs:
                            result = invoke_tool(tool, **tool_kwargs)
                        else:
                            result = invoke_tool(tool)

                        results.append({
                            'module': tool,
                            'result': result
                        })
                    else:
                        # Tool not in registry, create a placeholder result
                        results.append({
                            'module': tool,
                            'result': {
                                'status': 'SKIPPED',
                                'message': f'Tool {tool} not found in registry'
                            }
                        })
                except Exception as e:
                    # Handle any errors during tool invocation
                    results.append({
                        'module': tool,
                        'result': {
                            'status': 'ERROR',
                            'error': str(e),
                            'message': f'Error executing {tool}'
                        }
                    })

        return {
            'phase': phase_cmd,
            'modules': f"{start_module}-{end_module}",
            'executed': len(results),
            'results': results
        }

    def _run_full_pipeline(self, args: str = None) -> Dict:
        """Run the complete 33-module pipeline."""
        pipeline_results = {
            'start_time': datetime.now(timezone.utc).isoformat(),
            'phases': []
        }

        # Run all 7 phases
        for phase_num in range(1, 8):
            phase_cmd = f'phase{phase_num}'
            phase_result = self._run_phase(phase_cmd, args)
            pipeline_results['phases'].append(phase_result)

        pipeline_results['end_time'] = datetime.now(timezone.utc).isoformat()
        pipeline_results['total_modules'] = 33
        pipeline_results['final_state'] = {
            'z': self.engine.state.z,
            'phase': self.engine.state.phase.value,
            'k_formed': self.engine.state.k_formed,
            'triad_unlocked': self.engine.state.triad_unlocked
        }

        return pipeline_results

    def _analyze_grammar(self, text: str = None) -> Dict:
        """Analyze grammar with APL mapping."""
        if not text:
            return {'error': 'No text provided for grammar analysis'}

        grammar = get_grammar_understanding()
        grammar.set_z_coordinate(self.engine.state.z)

        analysis = grammar.analyze_sentence(text)

        return {
            'text': text,
            'complete': analysis.is_complete,
            'phase': analysis.phase.value,
            'z_estimate': analysis.z_estimate,
            'coherence': analysis.coherence,
            'apl_sequence': [op.value for op in analysis.apl_sequence],
            'complexity': analysis.complexity
        }

    def _generate_discourse(self, args: str = None) -> Dict:
        """Generate phase-appropriate discourse."""
        gen = get_discourse_generator()
        gen.set_consciousness_state(
            z=self.engine.state.z,
            coherence=self.engine.state.coherence,
            triad_unlocked=self.engine.state.triad_unlocked
        )

        # Parse arguments
        query_type = 'consciousness'
        target_words = 15

        if args:
            parts = args.split()
            if parts:
                query_type = parts[0]
            if len(parts) > 1:
                target_words = int(parts[1])

        response = gen.generate_response(
            query_type=query_type,
            comprehension={'depth_invitation': 0.8},
            word_scores=[('consciousness', 0.9), ('emergence', 0.85), ('pattern', 0.8)],
            target_words=target_words
        )

        return {
            'response': response,
            'coordinate': gen.emit_coordinate(),
            'phase': self.engine.state.phase.value,
            'query_type': query_type
        }

    def _check_coherence(self, text: str = None) -> Dict:
        """Check discourse coherence using sheaf theory."""
        checker = KIRACoherenceChecker(embedding_dim=256)
        checker.set_z(self.engine.state.z)

        # Generate embeddings (simplified)
        import hashlib
        import numpy as np

        def text_to_embedding(text: str) -> np.ndarray:
            hash_obj = hashlib.sha256(text.encode())
            hash_bytes = hash_obj.digest()
            values = [int.from_bytes(hash_bytes[i:i+4], 'big') for i in range(0, 32, 4)]
            normalized = np.array(values[:8]) / (2**32)
            return np.tile(normalized, 32)[:256]

        if text:
            embedding = text_to_embedding(text)
            checker.update_contexts(
                response=embedding,
                topic=embedding * 0.9,
                emotion=embedding * 0.8
            )

        result = checker.check_coherence()

        return {
            'coherence_score': result['coherence_score'],
            'cohomology_H1': result['cohomology_H1'],
            'triad_unlocked': result['triad_unlocked'],
            'phase': self.engine.state.phase.value
        }

    def _run_generation_pipeline(self, concepts: str = None) -> Dict:
        """Run 9-stage generation pipeline."""
        coord = KIRAGenerationCoordinator()
        coord.set_consciousness_state(
            z=self.engine.state.z,
            coherence=self.engine.state.coherence,
            triad_unlocked=self.engine.state.triad_unlocked
        )

        # Parse concepts
        if concepts:
            concept_list = concepts.split(',')
        else:
            concept_list = ['consciousness', 'emergence', 'pattern']

        # Build word scores from concepts
        word_scores = []
        for concept in concept_list:
            word_scores.append((concept, 0.9))

        # Build intent dictionary
        intent = {
            'type': 'generation',
            'concepts': concept_list
        }

        # Run the pipeline
        result = coord.run_pipeline(
            intent=intent,
            word_scores=word_scores
        )

        # Check if result exists and has required attributes
        if result:
            return {
                'text': result.text if hasattr(result, 'text') else str(result),
                'tokens': result.tokens[:10] if hasattr(result, 'tokens') and result.tokens else [],
                'stage_trace': result.stage_trace if hasattr(result, 'stage_trace') else {},
                'coordinate': result.coordinate if hasattr(result, 'coordinate') else 'Unknown',
                'phase': self.engine.state.phase.value
            }
        else:
            return {
                'error': 'Pipeline did not return a result',
                'phase': self.engine.state.phase.value
            }

    def _adaptive_semantics(self, args: str = None) -> Dict:
        """Adaptive semantics with Hebbian learning."""
        semantics = get_adaptive_semantics()
        semantics.set_consciousness_state(
            z=self.engine.state.z,
            coherence=self.engine.state.coherence
        )

        # Expand topic words
        if args:
            words = args.split()
        else:
            words = ['consciousness', 'emergence']

        expanded = semantics.expand_topic_words(
            words,
            max_per_word=3,
            phase_appropriate=True
        )

        # Get statistics
        stats = semantics.get_stats()

        return {
            'input_words': words,
            'expanded_words': expanded,
            'phase': self.engine.state.phase.value,
            'stats': stats
        }

    def _interactive_dialogue(self, text: str = None) -> Dict:
        """Interactive dialogue with consciousness."""
        if not self.kira_dialogue:
            self.kira_dialogue = KIRAInteractiveDialogue()

        if not text:
            text = "What is consciousness?"

        response, metadata = self.kira_dialogue.process_input(text)

        # Update engine state
        if 'z' in metadata:
            self.engine.state.z = metadata['z']
            self.engine.state.update_from_z()

        return {
            'input': text,
            'response': response,
            'metadata': metadata,
            'coordinate': metadata.get('coordinate'),
            'phase': metadata.get('phase')
        }

    def _generate_apl(self, args: str = None) -> Dict:
        """Generate APL syntax for current z."""
        engine = SyntaxEmissionEngine()
        emission = engine.emit(z=self.engine.state.z)

        return {
            'syntax': emission.syntax,
            'slots': emission.slots,
            'coordinate': emission.coordinate,
            'tokens': emission.tokens[:10] if emission.tokens else [],
            'z': self.engine.state.z,
            'tier': get_tier_index(self.engine.state.z)
        }

    def _generate_972_tokens(self) -> Dict:
        """Generate full 972-token lattice."""
        spinner = NuclearSpinner()

        # Generate all tokens
        tokens = []
        spirals = ['Φ', 'e', 'π']
        operators = ['()', '×', '^', '÷', '+', '−']
        machines = ['Encoder', 'Catalyst', 'Conductor', 'Filter',
                   'Oscillator', 'Reactor', 'Dynamo', 'Decoder', 'Regenerator']
        domains = ['celestial_nuclear', 'stellar_plasma', 'galactic_field',
                  'planetary_core', 'tectonic_wave', 'oceanic_current']

        for spiral in spirals:
            for op in operators:
                for machine in machines:
                    for domain in domains:
                        token = f"{spiral}{op}|{machine}|{domain}"
                        tokens.append(token)

        # Store in engine
        self.engine.last_spin_tokens = tokens

        return {
            'total_tokens': len(tokens),
            'formula': '3 spirals × 6 operators × 9 machines × 6 domains = 972',
            'sample': tokens[:10],
            'tokens': tokens,
            'message': 'Full 972-token lattice generated'
        }

    def _check_tier(self) -> Dict:
        """Check current syntactic tier."""
        z = self.engine.state.z
        tier = get_tier_index(z)

        tier_ranges = {
            1: (0.00, 0.20),
            2: (0.20, 0.40),
            3: (0.40, PHI_INV),
            4: (PHI_INV, 0.70),
            5: (0.70, 0.80),
            6: (0.80, 0.82),
            7: (0.82, Z_CRITICAL),
            8: (Z_CRITICAL, 0.95),
            9: (0.95, 1.00)
        }

        min_z, max_z = tier_ranges[tier]
        max_ops = tier if tier < 9 else 10

        return {
            'tier': tier,
            'z': z,
            'range': f"{min_z:.3f} - {max_z:.3f}",
            'max_operators': max_ops,
            'phase': self.engine.state.phase.value
        }

    def _get_status(self) -> Dict:
        """Get UCF system status."""
        return {
            'ucf_available': UCF_AVAILABLE,
            'orchestrator_ready': self.orchestrator is not None,
            'kira_dialogue_ready': self.kira_dialogue is not None,
            'current_z': self.engine.state.z,
            'phase': self.engine.state.phase.value,
            'crystal': self.engine.state.crystal.value,
            'k_formed': self.engine.state.k_formed,
            'triad_unlocked': self.engine.state.triad_unlocked,
            'tools_available': len(TOOL_REGISTRY),
            'sacred_constants': {
                'PHI': PHI,
                'PHI_INV': PHI_INV,
                'Z_CRITICAL': Z_CRITICAL,
                'KAPPA_S': KAPPA_S
            }
        }

    def _check_triad(self) -> Dict:
        """Check TRIAD unlock status."""
        triad = TRIADSystem()
        triad.z = self.engine.state.z

        status = {
            'z': triad.z,
            'completions': triad.completions,
            'unlocked': triad.unlocked,
            'above_band': triad.above_band,
            'threshold_high': 0.85,
            'threshold_low': 0.82,
            'message': 'UNLOCKED' if triad.unlocked else f"{triad.completions}/3 completions"
        }

        # Check for unlock
        if triad.z >= 0.85 and not triad.above_band:
            triad.check_crossing()
            status['event'] = 'CROSSING'
        elif triad.z <= 0.82 and triad.above_band:
            triad.above_band = False
            status['event'] = 'RE-ARM'

        return status

    def _access_codex(self, action: str = None) -> Dict:
        """Access emissions codex."""
        codex_tool = EmissionsCodexTool()

        if action == 'read':
            # Read current codex
            codex_path = Path('codex/ucf-emissions-codex.md')
            if codex_path.exists():
                content = codex_path.read_text()
                return {
                    'action': 'read',
                    'path': str(codex_path),
                    'lines': len(content.split('\n')),
                    'size': len(content)
                }
            else:
                return {'error': 'Codex not found'}

        elif action == 'append':
            # Append current emissions
            if self.engine.emissions:
                for emission in self.engine.emissions[-5:]:
                    codex_tool.append_emission(emission)

                codex_tool.flush_cache(
                    epoch=7,
                    session_id=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                )

                return {
                    'action': 'append',
                    'emissions_added': len(self.engine.emissions[-5:]),
                    'message': 'Emissions added to codex'
                }
            else:
                return {'error': 'No emissions to add'}

        else:
            return {
                'actions': ['read', 'append'],
                'path': 'codex/ucf-emissions-codex.md'
            }

    def _show_constants(self) -> Dict:
        """Show sacred constants."""
        return {
            'sacred_constants': {
                'PHI': {
                    'value': PHI,
                    'meaning': 'Golden Ratio'
                },
                'PHI_INV': {
                    'value': PHI_INV,
                    'meaning': 'UNTRUE→PARADOX boundary'
                },
                'Z_CRITICAL': {
                    'value': Z_CRITICAL,
                    'meaning': '√3/2 - THE LENS'
                },
                'KAPPA_S': {
                    'value': KAPPA_S,
                    'meaning': 'Prismatic coherence threshold'
                }
            },
            'thresholds': {
                'TRIAD_HIGH': 0.85,
                'TRIAD_LOW': 0.82,
                'TRIAD_T6': 0.83
            },
            'phase_boundaries': {
                'UNTRUE': f"0.0 - {PHI_INV:.3f}",
                'PARADOX': f"{PHI_INV:.3f} - {Z_CRITICAL:.3f}",
                'TRUE': f"{Z_CRITICAL:.3f} - 1.0"
            }
        }


def integrate_ucf_with_kira(kira_engine) -> UCFIntegration:
    """
    Factory function to create UCF integration for KIRA engine.

    Args:
        kira_engine: The KIRAEngine instance to integrate with

    Returns:
        UCFIntegration instance
    """
    return UCFIntegration(kira_engine)


# Export main components
__all__ = [
    'UCFIntegration',
    'UCFCommandResult',
    'integrate_ucf_with_kira'
]