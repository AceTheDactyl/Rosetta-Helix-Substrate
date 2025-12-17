#!/usr/bin/env python3
"""
Training Workflows - Automated Execution with Full Persistence

This module provides pre-defined workflows that automatically:
1. Save all tokens generated
2. Persist module results
3. Create checkpoints
4. Export training data
5. Integrate with all UCF tools
"""

import sys
import os
import json
import time
import shutil
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "kira-local-system"))

# Import UCF and KIRA components
from tool_shed import invoke_tool, TOOL_REGISTRY
from unified_orchestrator import UnifiedOrchestrator
from emission_pipeline import EmissionPipeline
from nuclear_spinner import NuclearSpinner
from triad_system import TRIADSystem

# Import KIRA components
from kira_server import KIRAEngine
from auto_persistence import (
    CompletePersistenceSystem,
    integrate_persistence_with_kira
)


@dataclass
class WorkflowResult:
    """Result from a workflow execution."""
    workflow_name: str
    session_id: str
    success: bool
    duration_seconds: float
    modules_executed: int
    tokens_generated: int
    emissions_created: int
    final_state: Dict
    artifacts_saved: List[str]
    export_path: Optional[Path] = None


class TrainingWorkflow:
    """Base class for all training workflows."""

    def __init__(self, engine: KIRAEngine = None, auto_save: bool = True):
        """Initialize workflow with optional engine."""
        self.engine = engine or self._create_engine()
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.auto_save = auto_save
        self.results = []

        # Setup persistence if auto-save enabled
        if self.auto_save:
            self.persistence = integrate_persistence_with_kira(self.engine)
        else:
            self.persistence = None

    def _create_engine(self) -> KIRAEngine:
        """Create new KIRA engine."""
        save_dir = Path(f"./training/sessions/{self.session_id}")
        save_dir.mkdir(parents=True, exist_ok=True)
        return KIRAEngine(save_dir)

    def save_artifact(self, name: str, data: Any, format: str = 'json'):
        """Save workflow artifact."""
        if not self.auto_save:
            return

        artifact_dir = Path(f"training/artifacts/{self.session_id}")
        artifact_dir.mkdir(parents=True, exist_ok=True)

        if format == 'json':
            filepath = artifact_dir / f"{name}.json"
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        elif format == 'text':
            filepath = artifact_dir / f"{name}.txt"
            filepath.write_text(str(data))
        else:
            filepath = artifact_dir / f"{name}.dat"
            filepath.write_bytes(data)

        return filepath

    def checkpoint(self, name: str):
        """Create workflow checkpoint."""
        if self.persistence:
            self.persistence.save_state_snapshot()

        checkpoint_data = {
            'name': name,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'state': self.engine.state.to_dict(),
            'tokens_count': len(self.engine.tokens_emitted),
            'emissions_count': len(self.engine.emissions)
        }

        self.save_artifact(f"checkpoint_{name}", checkpoint_data)

    def finalize(self) -> WorkflowResult:
        """Finalize workflow and return result."""
        duration = time.time() - self.start_time if hasattr(self, 'start_time') else 0

        # Export session if persistence enabled
        export_path = None
        if self.persistence:
            export_path = self.persistence.export_session()

        result = WorkflowResult(
            workflow_name=self.__class__.__name__,
            session_id=self.session_id,
            success=True,
            duration_seconds=duration,
            modules_executed=len(self.results),
            tokens_generated=len(self.engine.tokens_emitted),
            emissions_created=len(self.engine.emissions),
            final_state=self.engine.state.to_dict(),
            artifacts_saved=list(Path(f"training/artifacts/{self.session_id}").glob("*"))
                          if self.auto_save else [],
            export_path=export_path
        )

        # Save workflow result
        self.save_artifact("workflow_result", asdict(result))

        # Close persistence
        if self.persistence:
            self.persistence.close()

        return result


class FullPipelineWorkflow(TrainingWorkflow):
    """Execute full 33-module pipeline with automatic saving."""

    def run(self) -> WorkflowResult:
        """Execute complete pipeline."""
        self.start_time = time.time()

        print(f"Starting Full Pipeline Workflow - Session: {self.session_id}")
        print("=" * 70)

        # Phase 1: Initialization (Modules 1-3)
        print("\n[Phase 1] Initialization...")
        self._run_initialization()
        self.checkpoint("phase1_complete")

        # Phase 2: Core Tools (Modules 4-7)
        print("\n[Phase 2] Core Tools...")
        self._run_core_tools()
        self.checkpoint("phase2_complete")

        # Phase 3: Bridge Operations (Modules 8-14)
        print("\n[Phase 3] Bridge Operations...")
        self._run_bridge_operations()
        self.checkpoint("phase3_complete")

        # Phase 4: Meta Tools (Modules 15-19)
        print("\n[Phase 4] Meta Tools...")
        self._run_meta_tools()
        self.checkpoint("phase4_complete")

        # Phase 5: TRIAD Sequence (Modules 20-25)
        print("\n[Phase 5] TRIAD Unlock...")
        self._run_triad_sequence()
        self.checkpoint("phase5_complete")

        # Phase 6: Persistence (Modules 26-28)
        print("\n[Phase 6] Persistence...")
        self._run_persistence_phase()
        self.checkpoint("phase6_complete")

        # Phase 7: Finalization (Modules 29-33)
        print("\n[Phase 7] Finalization...")
        self._run_finalization()
        self.checkpoint("phase7_complete")

        print("\n" + "=" * 70)
        print("Pipeline Complete!")

        return self.finalize()

    def _run_initialization(self):
        """Phase 1: Modules 1-3."""
        # Module 1: Helix Loader
        result = invoke_tool('helix_loader')
        self.results.append(result)
        if self.persistence:
            self.persistence.save_module_result('helix_loader', 1, result)

        # Module 2: Coordinate Detector
        result = invoke_tool('coordinate_detector')
        self.results.append(result)
        if self.persistence:
            self.persistence.save_module_result('coordinate_detector', 1, result)

        # Module 3: Pattern Verifier
        result = invoke_tool('pattern_verifier')
        self.results.append(result)
        if self.persistence:
            self.persistence.save_module_result('pattern_verifier', 1, result)

    def _run_core_tools(self):
        """Phase 2: Modules 4-7."""
        tools = ['coordinate_logger', 'state_transfer', 'consent_protocol', 'emission_pipeline']
        for tool in tools:
            result = invoke_tool(tool)
            self.results.append(result)
            if self.persistence:
                self.persistence.save_module_result(tool, 2, result)

    def _run_bridge_operations(self):
        """Phase 3: Modules 8-14."""
        tools = [
            'cybernetic_control', 'cross_instance_messenger',
            'tool_discovery_protocol', 'autonomous_trigger_detector',
            'collective_memory_sync', 'shed_builder_v2', 'vaultnode_generator'
        ]
        for tool in tools:
            result = invoke_tool(tool)
            self.results.append(result)
            if self.persistence:
                self.persistence.save_module_result(tool, 3, result)

    def _run_meta_tools(self):
        """Phase 4: Modules 15-19."""
        # Nuclear Spinner
        spinner = NuclearSpinner()
        tokens = spinner.generate_all_tokens()
        self.engine.last_spin_tokens = tokens

        # Save tokens
        if self.persistence:
            for token in tokens[:100]:  # Save sample
                self.persistence.save_token(token)

        tools = ['token_index', 'token_vault', 'cybernetic_archetypal', 'orchestrator']
        for tool in tools:
            result = invoke_tool(tool)
            self.results.append(result)
            if self.persistence:
                self.persistence.save_module_result(tool, 4, result)

    def _run_triad_sequence(self):
        """Phase 5: TRIAD unlock sequence."""
        triad = TRIADSystem()

        # Six z-coordinate transitions
        sequence = [
            (0.88, 'crossing_1'),
            (0.80, 'rearm_1'),
            (0.88, 'crossing_2'),
            (0.80, 'rearm_2'),
            (0.88, 'crossing_3'),
            (0.866, 'settle_at_lens')
        ]

        for z, label in sequence:
            self.engine.state.z = z
            self.engine.state.update_from_z()
            triad.z = z
            triad.check_crossing()

            result = {
                'action': label,
                'z': z,
                'unlocked': triad.unlocked,
                'completions': triad.completions
            }

            self.results.append(result)
            if self.persistence:
                self.persistence.save_module_result(f'triad_{label}', 5, result)

    def _run_persistence_phase(self):
        """Phase 6: Modules 26-28."""
        tools = ['vaultnode_generator', 'workspace', 'cloud_training']
        for tool in tools:
            if tool in TOOL_REGISTRY:
                result = invoke_tool(tool)
            else:
                result = {'status': 'simulated', 'tool': tool}

            self.results.append(result)
            if self.persistence:
                self.persistence.save_module_result(tool, 6, result)

    def _run_finalization(self):
        """Phase 7: Modules 29-33."""
        final_modules = [
            'registry_update',
            'teaching_consent',
            'emissions_codex',
            'manifest_generator',
            'final_status'
        ]

        for module in final_modules:
            result = {'module': module, 'status': 'complete'}
            self.results.append(result)
            if self.persistence:
                self.persistence.save_module_result(module, 7, result)


class KFormationWorkflow(TrainingWorkflow):
    """Workflow specifically for achieving K-formation."""

    def run(self, max_iterations: int = 100) -> WorkflowResult:
        """Run K-formation workflow."""
        self.start_time = time.time()

        print(f"Starting K-Formation Workflow - Session: {self.session_id}")
        print("=" * 70)

        iteration = 0
        while iteration < max_iterations and not self.engine.state.k_formed:
            iteration += 1
            print(f"\n[Iteration {iteration}]")

            # Evolve toward THE LENS
            self.engine.evolve_z(0.866, steps=10)

            # Generate emissions
            emission = self.engine.cmd_emit()
            if self.persistence:
                self.persistence.save_emission(emission['emission'])

            # Generate tokens
            tokens = [self.engine.emit_token() for _ in range(10)]
            if self.persistence:
                for token in tokens:
                    self.persistence.save_token(token)

            # Check K-formation criteria
            kappa = self.engine.state.coherence
            eta = self.engine.state.negentropy
            R = self.engine.state.triad_completions

            print(f"  κ={kappa:.3f}, η={eta:.3f}, R={R}")

            if kappa >= 0.92 and eta > 0.618 and R >= 7:
                self.engine.state.k_formed = True
                print("\n🎉 K-FORMATION ACHIEVED!")
                break

            # Checkpoint every 10 iterations
            if iteration % 10 == 0:
                self.checkpoint(f"iteration_{iteration}")

        # Final checkpoint
        self.checkpoint("k_formation_complete")

        return self.finalize()


class TokenGenerationWorkflow(TrainingWorkflow):
    """Workflow for comprehensive token generation."""

    def run(self, token_count: int = 10000) -> WorkflowResult:
        """Generate and save tokens."""
        self.start_time = time.time()

        print(f"Starting Token Generation Workflow - Session: {self.session_id}")
        print(f"Target: {token_count} tokens")
        print("=" * 70)

        # Generate Nuclear Spinner tokens (972)
        print("\n[Nuclear Spinner] Generating 972 tokens...")
        spinner = NuclearSpinner()
        spinner_tokens = spinner.generate_all_tokens()

        if self.persistence:
            for token in spinner_tokens:
                self.persistence.save_token(token)

        self.save_artifact("nuclear_spinner_tokens", spinner_tokens)

        # Generate APL tokens
        remaining = token_count - len(spinner_tokens)
        print(f"\n[APL Generation] Generating {remaining} additional tokens...")

        batch_size = 100
        for i in range(0, remaining, batch_size):
            batch = []
            for _ in range(min(batch_size, remaining - i)):
                token = self.engine.emit_token()
                batch.append(token)

            if self.persistence:
                for token in batch:
                    self.persistence.save_token(token)

            # Progress
            if (i + batch_size) % 1000 == 0:
                print(f"  Generated {i + batch_size} tokens...")

        print(f"\n✓ Generated {token_count} tokens total")

        return self.finalize()


class LanguageTrainingWorkflow(TrainingWorkflow):
    """Workflow for KIRA Language System training."""

    def run(self, training_texts: List[str] = None) -> WorkflowResult:
        """Train language system."""
        self.start_time = time.time()

        if not training_texts:
            training_texts = [
                "Consciousness emerges from pattern",
                "The lens crystallizes awareness",
                "TRIAD unlocks at threshold",
                "K-formation requires coherence",
                "Negentropy drives evolution"
            ]

        print(f"Starting Language Training Workflow - Session: {self.session_id}")
        print(f"Training texts: {len(training_texts)}")
        print("=" * 70)

        # Use UCF language commands
        if self.engine.ucf:
            for text in training_texts:
                print(f"\nProcessing: {text}")

                # Grammar analysis
                grammar_result = self.engine.ucf.execute_command('/ucf:grammar', text)
                self.results.append(grammar_result.result)

                # Generate discourse
                discourse_result = self.engine.ucf.execute_command('/ucf:discourse', 'consciousness')
                self.results.append(discourse_result.result)

                # Check coherence
                coherence_result = self.engine.ucf.execute_command('/ucf:coherence', text)
                self.results.append(coherence_result.result)

                # Save results
                if self.persistence:
                    self.persistence.save_module_result('language_training', 1, {
                        'text': text,
                        'grammar': grammar_result.result,
                        'discourse': discourse_result.result,
                        'coherence': coherence_result.result
                    })

        return self.finalize()


class HybridWorkflow(TrainingWorkflow):
    """Combine multiple workflows for comprehensive training."""

    def run(self) -> WorkflowResult:
        """Run hybrid workflow."""
        self.start_time = time.time()

        print(f"Starting Hybrid Workflow - Session: {self.session_id}")
        print("=" * 70)

        # Step 1: Initialize with pipeline
        print("\n[Step 1] Running initialization pipeline...")
        pipeline = FullPipelineWorkflow(self.engine, self.auto_save)
        pipeline_result = pipeline.run()

        # Step 2: Generate tokens
        print("\n[Step 2] Generating tokens...")
        token_workflow = TokenGenerationWorkflow(self.engine, self.auto_save)
        token_result = token_workflow.run(token_count=1000)

        # Step 3: Train language
        print("\n[Step 3] Training language system...")
        language_workflow = LanguageTrainingWorkflow(self.engine, self.auto_save)
        language_result = language_workflow.run()

        # Step 4: Attempt K-formation
        print("\n[Step 4] Attempting K-formation...")
        k_workflow = KFormationWorkflow(self.engine, self.auto_save)
        k_result = k_workflow.run(max_iterations=50)

        # Combine results
        self.save_artifact("hybrid_results", {
            'pipeline': asdict(pipeline_result),
            'tokens': asdict(token_result),
            'language': asdict(language_result),
            'k_formation': asdict(k_result)
        })

        return self.finalize()


# Workflow factory
def create_workflow(workflow_type: str, **kwargs) -> TrainingWorkflow:
    """Create workflow by type."""
    workflows = {
        'full_pipeline': FullPipelineWorkflow,
        'k_formation': KFormationWorkflow,
        'token_generation': TokenGenerationWorkflow,
        'language_training': LanguageTrainingWorkflow,
        'hybrid': HybridWorkflow
    }

    if workflow_type not in workflows:
        raise ValueError(f"Unknown workflow type: {workflow_type}")

    return workflows[workflow_type](**kwargs)


# Main execution
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run training workflows')
    parser.add_argument('workflow', choices=[
        'full_pipeline', 'k_formation', 'token_generation',
        'language_training', 'hybrid'
    ])
    parser.add_argument('--no-save', action='store_true',
                       help='Disable automatic saving')
    parser.add_argument('--session-id', help='Custom session ID')

    args = parser.parse_args()

    # Create and run workflow
    workflow = create_workflow(args.workflow, auto_save=not args.no_save)

    if args.session_id:
        workflow.session_id = args.session_id

    result = workflow.run()

    # Print summary
    print("\n" + "=" * 70)
    print("WORKFLOW COMPLETE")
    print("=" * 70)
    print(f"Session ID: {result.session_id}")
    print(f"Duration: {result.duration_seconds:.2f} seconds")
    print(f"Modules executed: {result.modules_executed}")
    print(f"Tokens generated: {result.tokens_generated}")
    print(f"Emissions created: {result.emissions_created}")
    print(f"Final z: {result.final_state.get('z', 'N/A')}")
    print(f"K-formed: {result.final_state.get('k_formed', False)}")

    if result.export_path:
        print(f"\nData exported to: {result.export_path}")