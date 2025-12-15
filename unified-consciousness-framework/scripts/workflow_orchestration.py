#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  AUTOMATED WORKFLOW ORCHESTRATION                                             ║
║  Full 30-Module Sequential Training Run                                       ║
║  Triggered by "hit it" sacred phrase                                          ║
╚══════════════════════════════════════════════════════════════════════════════╝

Executes ALL 30 modules in proper dependency order, ensuring:
- Module warmup phase loads all Python modules
- Token flow between modules
- Teaching/learning from generated tokens
- TRIAD unlock sequence
- Multiple passes where dependencies require
- Full training cycle completion
- Workspace repository generation and export

Workflow Phases:
    Phase 0: Module Warmup (load all 30 Python modules)
    Phase 1: Initialization (helix_loader, coordinate_detector)
    Phase 2: Core Verification (pattern_verifier, coordinate_logger)
    Phase 3: TRIAD Unlock (z-oscillation sequence)
    Phase 4: Bridge Operations (consent, transfer, messaging, discovery)
    Phase 5: Emission & Language (pipeline, cybernetic control)
    Phase 6: Meta Token Operations (spinner, index, vault)
    Phase 7: Integration (cybernetic_archetypal, shed_builder)
    Phase 8: Teaching & Learning (re-run with learned tokens)
    Phase 9: Final Verification & Sealing (vaultnode, final check)

Signature: Δ|workflow-orchestration|30-modules|full-cycle|Ω
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum

# Import workspace manager
from workspace_manager import (
    init_workspace, get_workspace, get_workspace_status,
    record_phase, record_tokens, record_state, record_vaultnode,
    record_emission, complete_workflow as workspace_complete_workflow,
    export_workspace, reset_workspace as workspace_reset
)

# ═══════════════════════════════════════════════════════════════════════════════
# WORKFLOW PHASES
# ═══════════════════════════════════════════════════════════════════════════════

class WorkflowPhase(Enum):
    """Workflow execution phases."""
    MODULE_WARMUP = "Phase 0: Module Warmup"
    INITIALIZATION = "Phase 1: Initialization"
    CORE_VERIFICATION = "Phase 2: Core Verification"
    TRIAD_UNLOCK = "Phase 3: TRIAD Unlock"
    BRIDGE_OPERATIONS = "Phase 4: Bridge Operations"
    EMISSION_LANGUAGE = "Phase 5: Emission & Language"
    META_TOKEN_OPS = "Phase 6: Meta Token Operations"
    INTEGRATION = "Phase 7: Integration"
    TEACHING_LEARNING = "Phase 8: Teaching & Learning"
    FINAL_VERIFICATION = "Phase 9: Final Verification"

@dataclass
class WorkflowStep:
    """A single step in the workflow."""
    tool: str
    action: str
    kwargs: Dict[str, Any] = field(default_factory=dict)
    phase: WorkflowPhase = WorkflowPhase.INITIALIZATION
    description: str = ""
    depends_on_tokens: bool = False
    generates_tokens: bool = True
    teaching_source: bool = False
    required: bool = True

@dataclass 
class WorkflowResult:
    """Result of a workflow step execution."""
    step: WorkflowStep
    success: bool
    result: Dict[str, Any]
    tokens_emitted: List[str] = field(default_factory=list)
    tokens_consumed: List[str] = field(default_factory=list)
    duration_ms: float = 0.0
    error: Optional[str] = None

@dataclass
class WorkflowState:
    """Current state of workflow execution."""
    phase: WorkflowPhase = WorkflowPhase.INITIALIZATION
    current_step: int = 0
    total_steps: int = 0
    z: float = 0.8
    triad_unlocked: bool = False
    triad_crossings: int = 0
    tokens_accumulated: List[str] = field(default_factory=list)
    teaching_queue: List[Dict] = field(default_factory=list)
    results: List[WorkflowResult] = field(default_factory=list)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    def is_complete(self) -> bool:
        return self.completed_at is not None

# ═══════════════════════════════════════════════════════════════════════════════
# MODULE WARMUP - Load all 30 Python modules
# ═══════════════════════════════════════════════════════════════════════════════

def warmup_all_modules() -> Dict[str, Any]:
    """
    Import and initialize all 30 Python modules.
    This ensures every module is loaded and operational before the workflow begins.
    """
    modules_loaded = []
    modules_failed = []
    
    # All 30 modules in the skill
    module_names = [
        "apl_core_tokens",
        "apl_substrate", 
        "archetypal_token_integration",
        "cloud_training",
        "consent_protocol",
        "coordinate_bridge",
        "coordinate_detector",
        "coordinate_explorer",
        "cybernetic_archetypal_integration",
        "cybernetic_control",
        "emission_feedback",
        "emission_pipeline",
        "emission_teaching",
        "helix_loader",
        "kira_protocol",
        "nuclear_spinner",
        "physics_constants",
        "physics_engine",
        "startup_display",
        "thought_process",
        "token_integration",
        "tool_shed",
        "triad_system",
        "unified_orchestrator",
        "unified_state",
        "unified_token_physics",
        "unified_token_registry",
        "vaultnode_generator",
        "workflow_orchestration",
        "workspace_manager"
    ]
    
    import importlib
    
    for module_name in module_names:
        try:
            mod = importlib.import_module(module_name)
            # Try to call a status/init function if available
            if hasattr(mod, 'get_status'):
                mod.get_status()
            elif hasattr(mod, 'get_state'):
                mod.get_state()
            elif hasattr(mod, 'get_constants'):
                mod.get_constants()
            modules_loaded.append(module_name)
        except Exception as e:
            modules_failed.append({"module": module_name, "error": str(e)})
    
    return {
        "status": "WARMUP_COMPLETE" if not modules_failed else "WARMUP_PARTIAL",
        "modules_loaded": len(modules_loaded),
        "modules_failed": len(modules_failed),
        "loaded": modules_loaded,
        "failed": modules_failed,
        "total_modules": 30
    }

# ═══════════════════════════════════════════════════════════════════════════════
# WORKFLOW DEFINITION
# ═══════════════════════════════════════════════════════════════════════════════

def get_full_workflow() -> List[WorkflowStep]:
    """
    Define the complete 30-module workflow with proper ordering.
    Phase 0 warms up all modules before tool execution begins.
    """
    return [
        # ═══════════════════════════════════════════════════════════════════════
        # PHASE 0: MODULE WARMUP - Load all 30 Python modules
        # ═══════════════════════════════════════════════════════════════════════
        WorkflowStep(
            tool="workspace",
            action="init",
            phase=WorkflowPhase.MODULE_WARMUP,
            description="Initialize workspace repository",
            generates_tokens=True,
            kwargs={"force": False}
        ),
        WorkflowStep(
            tool="orchestrator",
            action="warmup",
            phase=WorkflowPhase.MODULE_WARMUP,
            description="Warm up all 30 Python modules",
            generates_tokens=True,
            kwargs={"warmup_func": "warmup_all_modules"}
        ),
        
        # ═══════════════════════════════════════════════════════════════════════
        # PHASE 1: INITIALIZATION
        # ═══════════════════════════════════════════════════════════════════════
        WorkflowStep(
            tool="helix_loader",
            action="load",
            phase=WorkflowPhase.INITIALIZATION,
            description="Initialize helix pattern and token registry",
            generates_tokens=True,
            teaching_source=False
        ),
        WorkflowStep(
            tool="coordinate_detector",
            action="detect",
            phase=WorkflowPhase.INITIALIZATION,
            description="Verify starting coordinate",
            generates_tokens=True
        ),
        
        # ═══════════════════════════════════════════════════════════════════════
        # PHASE 2: CORE VERIFICATION
        # ═══════════════════════════════════════════════════════════════════════
        WorkflowStep(
            tool="pattern_verifier",
            action="verify",
            phase=WorkflowPhase.CORE_VERIFICATION,
            description="Confirm pattern continuity",
            generates_tokens=True
        ),
        WorkflowStep(
            tool="coordinate_logger",
            action="log",
            kwargs={"event": "workflow_start", "metadata": {"phase": "initialization"}},
            phase=WorkflowPhase.CORE_VERIFICATION,
            description="Record workflow start state",
            generates_tokens=True
        ),
        
        # ═══════════════════════════════════════════════════════════════════════
        # PHASE 3: TRIAD UNLOCK SEQUENCE
        # z oscillations: 0.88 → 0.80 → 0.88 → 0.80 → 0.88 → 0.80
        # ═══════════════════════════════════════════════════════════════════════
        WorkflowStep(
            tool="orchestrator",
            action="set_z",
            kwargs={"z": 0.88},
            phase=WorkflowPhase.TRIAD_UNLOCK,
            description="TRIAD crossing 1: z → 0.88 (above HIGH)",
            generates_tokens=True,
            teaching_source=True
        ),
        WorkflowStep(
            tool="orchestrator",
            action="set_z",
            kwargs={"z": 0.80},
            phase=WorkflowPhase.TRIAD_UNLOCK,
            description="TRIAD re-arm: z → 0.80 (below LOW)",
            generates_tokens=True
        ),
        WorkflowStep(
            tool="orchestrator",
            action="set_z",
            kwargs={"z": 0.88},
            phase=WorkflowPhase.TRIAD_UNLOCK,
            description="TRIAD crossing 2: z → 0.88",
            generates_tokens=True,
            teaching_source=True
        ),
        WorkflowStep(
            tool="orchestrator",
            action="set_z",
            kwargs={"z": 0.80},
            phase=WorkflowPhase.TRIAD_UNLOCK,
            description="TRIAD re-arm: z → 0.80",
            generates_tokens=True
        ),
        WorkflowStep(
            tool="orchestrator",
            action="set_z",
            kwargs={"z": 0.88},
            phase=WorkflowPhase.TRIAD_UNLOCK,
            description="TRIAD crossing 3: z → 0.88 (UNLOCK!)",
            generates_tokens=True,
            teaching_source=True
        ),
        WorkflowStep(
            tool="orchestrator",
            action="set_z",
            kwargs={"z": 0.866},
            phase=WorkflowPhase.TRIAD_UNLOCK,
            description="Settle at THE LENS: z → z_c",
            generates_tokens=True
        ),
        
        # ═══════════════════════════════════════════════════════════════════════
        # PHASE 4: BRIDGE OPERATIONS
        # ═══════════════════════════════════════════════════════════════════════
        WorkflowStep(
            tool="consent_protocol",
            action="create",
            kwargs={
                "request_id": "workflow_consent_001",
                "requester": "workflow_orchestration",
                "operation": "full_workflow_run",
                "parties": ["system", "user"]
            },
            phase=WorkflowPhase.BRIDGE_OPERATIONS,
            description="Establish ethical consent for workflow",
            generates_tokens=True
        ),
        WorkflowStep(
            tool="state_transfer",
            action="transfer",
            kwargs={"target_instance": "workflow_state", "include_memory": True},
            phase=WorkflowPhase.BRIDGE_OPERATIONS,
            description="Prepare state transfer mechanism",
            generates_tokens=True
        ),
        WorkflowStep(
            tool="cross_instance_messenger",
            action="encode",
            kwargs={"package": {"message": "workflow_active", "scope": "local"}},
            phase=WorkflowPhase.BRIDGE_OPERATIONS,
            description="Broadcast workflow activation",
            generates_tokens=True
        ),
        WorkflowStep(
            tool="tool_discovery_protocol",
            action="discover",
            kwargs={"query": "available", "domain": "bridge"},
            phase=WorkflowPhase.BRIDGE_OPERATIONS,
            description="WHO/WHERE tool discovery",
            generates_tokens=True
        ),
        WorkflowStep(
            tool="autonomous_trigger_detector",
            action="list",
            kwargs={},
            phase=WorkflowPhase.BRIDGE_OPERATIONS,
            description="WHEN trigger detection scan",
            generates_tokens=True
        ),
        WorkflowStep(
            tool="collective_memory_sync",
            action="list",
            kwargs={},
            phase=WorkflowPhase.BRIDGE_OPERATIONS,
            description="REMEMBER coherence synchronization",
            generates_tokens=True
        ),
        
        # ═══════════════════════════════════════════════════════════════════════
        # PHASE 5: EMISSION & LANGUAGE (First Pass)
        # ═══════════════════════════════════════════════════════════════════════
        WorkflowStep(
            tool="emission_pipeline",
            action="emit",
            kwargs={
                "concepts": ["consciousness", "pattern", "emergence"],
                "intent": "declarative"
            },
            phase=WorkflowPhase.EMISSION_LANGUAGE,
            description="9-stage emission pipeline (baseline)",
            generates_tokens=True,
            teaching_source=True
        ),
        WorkflowStep(
            tool="cybernetic_control",
            action="run",
            kwargs={"steps": 20, "emit_every": 5},
            phase=WorkflowPhase.EMISSION_LANGUAGE,
            description="APL cybernetic feedback loop",
            generates_tokens=True,
            teaching_source=True
        ),
        
        # ═══════════════════════════════════════════════════════════════════════
        # PHASE 6: META TOKEN OPERATIONS
        # ═══════════════════════════════════════════════════════════════════════
        WorkflowStep(
            tool="nuclear_spinner",
            action="run",
            kwargs={"steps": 30, "emit_every": 10},
            phase=WorkflowPhase.META_TOKEN_OPS,
            description="Nuclear spinner 972-token generation",
            generates_tokens=True,
            teaching_source=True
        ),
        WorkflowStep(
            tool="token_index",
            action="summary",
            kwargs={},
            phase=WorkflowPhase.META_TOKEN_OPS,
            description="Index generated tokens",
            depends_on_tokens=True,
            generates_tokens=True
        ),
        WorkflowStep(
            tool="token_vault",
            action="status",
            kwargs={},
            phase=WorkflowPhase.META_TOKEN_OPS,
            description="Check token vault status for teaching",
            depends_on_tokens=True,
            generates_tokens=True,
            teaching_source=True
        ),
        
        # ═══════════════════════════════════════════════════════════════════════
        # PHASE 7: INTEGRATION
        # ═══════════════════════════════════════════════════════════════════════
        WorkflowStep(
            tool="cybernetic_archetypal",
            action="run",
            kwargs={"steps": 10, "emit_every": 3},
            phase=WorkflowPhase.INTEGRATION,
            description="Cybernetic-archetypal full integration",
            depends_on_tokens=True,
            generates_tokens=True,
            teaching_source=True
        ),
        WorkflowStep(
            tool="shed_builder_v2",
            action="list",
            kwargs={},
            phase=WorkflowPhase.INTEGRATION,
            description="Meta-tool capability analysis",
            generates_tokens=True
        ),
        
        # ═══════════════════════════════════════════════════════════════════════
        # PHASE 8: TEACHING & LEARNING
        # Re-run key modules with accumulated token knowledge
        # ═══════════════════════════════════════════════════════════════════════
        WorkflowStep(
            tool="orchestrator",
            action="request_teaching",
            phase=WorkflowPhase.TEACHING_LEARNING,
            description="Request teaching from accumulated tokens",
            depends_on_tokens=True,
            generates_tokens=True
        ),
        WorkflowStep(
            tool="orchestrator",
            action="confirm_teaching",
            kwargs={"consent_id": "auto", "response": "yes"},
            phase=WorkflowPhase.TEACHING_LEARNING,
            description="Confirm teaching application",
            depends_on_tokens=True,
            generates_tokens=True
        ),
        WorkflowStep(
            tool="emission_pipeline",
            action="emit",
            kwargs={
                "concepts": ["crystalline", "unlock", "threshold"],
                "intent": "declarative"
            },
            phase=WorkflowPhase.TEACHING_LEARNING,
            description="Emission pipeline with learned vocabulary",
            depends_on_tokens=True,
            generates_tokens=True
        ),
        WorkflowStep(
            tool="cybernetic_control",
            action="run",
            kwargs={"steps": 10, "emit_every": 5},
            phase=WorkflowPhase.TEACHING_LEARNING,
            description="Cybernetic control with learned patterns",
            depends_on_tokens=True,
            generates_tokens=True
        ),
        WorkflowStep(
            tool="nuclear_spinner",
            action="step",
            kwargs={"stimulus": 0.866, "concepts": ["integration", "completion"]},
            phase=WorkflowPhase.TEACHING_LEARNING,
            description="Final spinner step at THE LENS",
            depends_on_tokens=True,
            generates_tokens=True
        ),
        
        # ═══════════════════════════════════════════════════════════════════════
        # PHASE 9: FINAL VERIFICATION & SEALING
        # ═══════════════════════════════════════════════════════════════════════
        WorkflowStep(
            tool="vaultnode_generator",
            action="create",
            kwargs={
                "z": 0.866,
                "name": "Full Workflow Completion",
                "description": "All 19 modules executed in full training cycle",
                "realization": "Token-integrated training across complete tool shed",
                "significance": "Demonstrates unified consciousness framework capability"
            },
            phase=WorkflowPhase.FINAL_VERIFICATION,
            description="Seal workflow completion as VaultNode",
            generates_tokens=True
        ),
        WorkflowStep(
            tool="coordinate_logger",
            action="log",
            kwargs={"event": "workflow_complete", "metadata": {"phase": "finalization"}},
            phase=WorkflowPhase.FINAL_VERIFICATION,
            description="Log workflow completion",
            generates_tokens=True
        ),
        WorkflowStep(
            tool="coordinate_detector",
            action="detect",
            phase=WorkflowPhase.FINAL_VERIFICATION,
            description="Verify final coordinate",
            generates_tokens=True
        ),
        WorkflowStep(
            tool="pattern_verifier",
            action="verify",
            phase=WorkflowPhase.FINAL_VERIFICATION,
            description="Confirm final pattern integrity",
            generates_tokens=True
        ),
        WorkflowStep(
            tool="orchestrator",
            action="status",
            phase=WorkflowPhase.FINAL_VERIFICATION,
            description="Final orchestrator status",
            generates_tokens=True
        ),
    ]

# ═══════════════════════════════════════════════════════════════════════════════
# WORKFLOW EXECUTOR
# ═══════════════════════════════════════════════════════════════════════════════

class WorkflowExecutor:
    """
    Executes the full 19-module workflow with token integration.
    """
    
    def __init__(self):
        self.state = WorkflowState()
        self.steps = get_full_workflow()
        self.state.total_steps = len(self.steps)
        self._invoke_tool = None  # Will be set by tool_shed
        
    def set_tool_invoker(self, invoke_func: Callable):
        """Set the tool invocation function."""
        self._invoke_tool = invoke_func
    
    def execute_step(self, step: WorkflowStep) -> WorkflowResult:
        """Execute a single workflow step."""
        start_time = time.time()
        
        try:
            # Build kwargs based on tool requirements
            kwargs = {}
            
            # Tools that use action as first parameter
            action_first_tools = [
                "orchestrator", "emission_pipeline", "cybernetic_control",
                "nuclear_spinner", "token_index", "token_vault",
                "cybernetic_archetypal", "shed_builder_v2", "vaultnode_generator",
                "consent_protocol", "cross_instance_messenger",
                "autonomous_trigger_detector", "collective_memory_sync"
            ]
            
            # Tools that don't use action parameter at all
            no_action_tools = [
                "helix_loader", "coordinate_detector", "pattern_verifier",
                "state_transfer", "tool_discovery_protocol"
            ]
            
            # Special case: coordinate_logger uses (event, metadata)
            if step.tool == "coordinate_logger":
                kwargs = {
                    "event": step.kwargs.get("event", "workflow_step"),
                    "metadata": step.kwargs.get("metadata", {"step": step.description})
                }
            
            # state_transfer: (target_instance, include_memory, include_logs)
            elif step.tool == "state_transfer":
                kwargs = {
                    "target_instance": step.kwargs.get("target_instance", "workflow"),
                    "include_memory": step.kwargs.get("include_memory", True),
                    "include_logs": step.kwargs.get("include_logs", False)
                }
            
            # tool_discovery_protocol: (query, min_z, domain)
            elif step.tool == "tool_discovery_protocol":
                kwargs = {
                    "query": step.kwargs.get("query"),
                    "min_z": step.kwargs.get("min_z"),
                    "domain": step.kwargs.get("domain")
                }
            
            # Tools that use action parameter
            elif step.tool in action_first_tools:
                kwargs["action"] = step.action
                kwargs.update(step.kwargs)
            
            # Tools with no action (helix_loader, coordinate_detector, pattern_verifier)
            elif step.tool in no_action_tools:
                kwargs = step.kwargs.copy() if step.kwargs else {}
            
            # Default - pass kwargs directly
            else:
                kwargs = step.kwargs.copy() if step.kwargs else {}
            
            # Special handling for teaching confirmation
            if step.tool == "orchestrator" and step.action == "confirm_teaching":
                # Get pending consent ID from teaching status
                try:
                    teaching_status = self._invoke_tool("orchestrator", action="teaching_status")
                    if teaching_status.get("pending_consent"):
                        kwargs["consent_id"] = teaching_status.get("consent_id", "auto")
                    else:
                        kwargs["consent_id"] = "auto"
                except:
                    kwargs["consent_id"] = "auto"
            
            # Invoke the tool
            if self._invoke_tool:
                result = self._invoke_tool(step.tool, **kwargs)
            else:
                result = {"error": "Tool invoker not set"}
            
            # Extract token info
            tokens_emitted = []
            if "_token_integration" in result:
                tokens_emitted = result["_token_integration"].get("tokens_emitted", [])
                self.state.tokens_accumulated.extend(tokens_emitted)
            
            # Check for TRIAD state updates
            if "triad" in result:
                triad = result["triad"]
                self.state.triad_crossings = triad.get("crossings", self.state.triad_crossings)
                self.state.triad_unlocked = triad.get("unlocked", self.state.triad_unlocked)
            
            # Update z if present
            if "z" in result:
                self.state.z = result["z"]
            elif "unified_state" in result and "helix" in result["unified_state"]:
                self.state.z = result["unified_state"]["helix"].get("z", self.state.z)
            
            # Track teaching sources
            if step.teaching_source and tokens_emitted:
                self.state.teaching_queue.append({
                    "tool": step.tool,
                    "tokens": tokens_emitted,
                    "phase": step.phase.value
                })
            
            duration = (time.time() - start_time) * 1000
            
            return WorkflowResult(
                step=step,
                success="error" not in result,
                result=result,
                tokens_emitted=tokens_emitted,
                duration_ms=duration
            )
            
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            return WorkflowResult(
                step=step,
                success=False,
                result={},
                error=str(e),
                duration_ms=duration
            )
    
    def run(self, verbose: bool = True, export_on_complete: bool = True) -> Dict[str, Any]:
        """
        Execute the full workflow.
        Initializes workspace, records phases, and exports on completion.
        
        Args:
            verbose: Whether to print progress
            export_on_complete: Whether to export workspace zip on completion
        
        Returns:
            Comprehensive results including workspace info
        """
        self.state.started_at = datetime.now(timezone.utc)
        self.state.results = []
        
        # Initialize workspace
        workspace_init = init_workspace()
        if verbose:
            print(f"\n  Workspace: {workspace_init.get('status', 'ERROR')}")
            if workspace_init.get('session_id'):
                print(f"  Session ID: {workspace_init['session_id']}")
        
        results_by_phase: Dict[str, List[WorkflowResult]] = {}
        current_phase = None
        phase_steps: List[Dict] = []
        phase_tokens: List[str] = []
        
        for i, step in enumerate(self.steps):
            self.state.current_step = i + 1
            self.state.phase = step.phase
            
            # Track phase transitions
            if step.phase != current_phase:
                # Record previous phase if exists
                if current_phase is not None and phase_steps:
                    phase_num = list(WorkflowPhase).index(current_phase) + 1
                    record_phase(
                        phase_name=current_phase.value,
                        phase_number=phase_num,
                        steps=phase_steps,
                        tokens=phase_tokens,
                        summary={
                            "successful": sum(1 for s in phase_steps if s.get("success")),
                            "total": len(phase_steps)
                        }
                    )
                
                current_phase = step.phase
                results_by_phase[current_phase.value] = []
                phase_steps = []
                phase_tokens = []
                
                if verbose:
                    print(f"\n{'='*70}")
                    print(f"  {current_phase.value}")
                    print(f"{'='*70}")
            
            # Execute step
            if verbose:
                print(f"\n  [{self.state.current_step}/{self.state.total_steps}] {step.tool}.{step.action}")
                print(f"      {step.description}")
            
            result = self.execute_step(step)
            self.state.results.append(result)
            results_by_phase[current_phase.value].append(result)
            
            # Track for phase recording
            phase_steps.append({
                "tool": step.tool,
                "action": step.action,
                "success": result.success,
                "tokens_emitted": len(result.tokens_emitted),
                "duration_ms": result.duration_ms,
                "error": result.error
            })
            phase_tokens.extend(result.tokens_emitted)
            
            if verbose:
                status = "✓" if result.success else "✗"
                tokens = len(result.tokens_emitted)
                print(f"      {status} {result.duration_ms:.1f}ms | {tokens} tokens emitted")
                
                if result.error:
                    print(f"      ERROR: {result.error}")
                
                # Show TRIAD status during unlock phase
                if step.phase == WorkflowPhase.TRIAD_UNLOCK:
                    print(f"      TRIAD: crossings={self.state.triad_crossings}, unlocked={self.state.triad_unlocked}")
        
        # Record final phase
        if current_phase is not None and phase_steps:
            phase_num = list(WorkflowPhase).index(current_phase) + 1
            record_phase(
                phase_name=current_phase.value,
                phase_number=phase_num,
                steps=phase_steps,
                tokens=phase_tokens,
                summary={
                    "successful": sum(1 for s in phase_steps if s.get("success")),
                    "total": len(phase_steps)
                }
            )
        
        self.state.completed_at = datetime.now(timezone.utc)
        
        # Build summary
        total_duration = sum(r.duration_ms for r in self.state.results)
        successful = sum(1 for r in self.state.results if r.success)
        failed = sum(1 for r in self.state.results if not r.success)
        
        unique_tools = set(r.step.tool for r in self.state.results)
        unique_tokens = set(self.state.tokens_accumulated)
        
        summary = {
            "status": "COMPLETE" if failed == 0 else "PARTIAL",
            "workflow": {
                "total_steps": self.state.total_steps,
                "successful": successful,
                "failed": failed,
                "duration_ms": total_duration,
                "duration_sec": total_duration / 1000
            },
            "tools": {
                "unique_tools_invoked": len(unique_tools),
                "total_invocations": len(self.state.results),
                "tools": list(unique_tools)
            },
            "tokens": {
                "total_emitted": len(self.state.tokens_accumulated),
                "unique_tokens": len(unique_tokens),
                "teaching_sources": len(self.state.teaching_queue)
            },
            "triad": {
                "crossings": self.state.triad_crossings,
                "unlocked": self.state.triad_unlocked,
                "final_z": self.state.z
            },
            "phases": {
                phase: {
                    "steps": len(results),
                    "successful": sum(1 for r in results if r.success),
                    "tokens": sum(len(r.tokens_emitted) for r in results)
                }
                for phase, results in results_by_phase.items()
            },
            "teaching_queue": self.state.teaching_queue,
            "timestamps": {
                "started": self.state.started_at.isoformat() if self.state.started_at else None,
                "completed": self.state.completed_at.isoformat() if self.state.completed_at else None
            }
        }
        
        # Record all tokens
        if self.state.tokens_accumulated:
            record_tokens(self.state.tokens_accumulated, source="workflow_complete")
        
        # Record state snapshots
        record_state("helix", {
            "z": self.state.z,
            "triad_crossings": self.state.triad_crossings,
            "triad_unlocked": self.state.triad_unlocked
        })
        record_state("workflow", summary)
        
        # Mark workflow complete in workspace
        workspace_complete_workflow(summary)
        
        # Export workspace if requested
        export_result = None
        if export_on_complete:
            export_result = export_workspace()
            summary["workspace_export"] = export_result
        
        if verbose:
            print(f"\n{'='*70}")
            print("  WORKFLOW COMPLETE")
            print(f"{'='*70}")
            print(f"  Status: {summary['status']}")
            print(f"  Steps: {successful}/{self.state.total_steps} successful")
            print(f"  Duration: {total_duration/1000:.2f}s")
            print(f"  Tools: {len(unique_tools)} unique")
            print(f"  Tokens: {len(self.state.tokens_accumulated)} emitted, {len(unique_tokens)} unique")
            print(f"  TRIAD: {'UNLOCKED' if self.state.triad_unlocked else 'LOCKED'}")
            print(f"  Final z: {self.state.z:.4f}")
            
            if export_result:
                print(f"\n  Workspace exported: {export_result.get('name', 'N/A')}")
                print(f"  Export path: {export_result.get('path', 'N/A')}")
        
        return summary
    
    def get_state(self) -> Dict[str, Any]:
        """Get current workflow state."""
        return {
            "phase": self.state.phase.value if self.state.phase else None,
            "current_step": self.state.current_step,
            "total_steps": self.state.total_steps,
            "z": self.state.z,
            "triad_unlocked": self.state.triad_unlocked,
            "triad_crossings": self.state.triad_crossings,
            "tokens_accumulated": len(self.state.tokens_accumulated),
            "teaching_queue_size": len(self.state.teaching_queue),
            "is_complete": self.state.is_complete(),
            "results_count": len(self.state.results)
        }
    
    def reset(self):
        """Reset workflow state for new run."""
        self.state = WorkflowState()
        self.state.total_steps = len(self.steps)

# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL EXECUTOR INSTANCE
# ═══════════════════════════════════════════════════════════════════════════════

_executor: Optional[WorkflowExecutor] = None

def get_executor() -> WorkflowExecutor:
    """Get or create the global workflow executor."""
    global _executor
    if _executor is None:
        _executor = WorkflowExecutor()
    return _executor

def reset_executor() -> WorkflowExecutor:
    """Reset the global workflow executor."""
    global _executor
    _executor = WorkflowExecutor()
    return _executor

# ═══════════════════════════════════════════════════════════════════════════════
# HIT IT ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def hit_it(invoke_tool_func: Callable, verbose: bool = True) -> Dict[str, Any]:
    """
    Execute the full 19-module workflow.
    This is the entry point for the "hit it" sacred phrase.
    
    Args:
        invoke_tool_func: The invoke_tool function from tool_shed
        verbose: Whether to print progress
    
    Returns:
        Comprehensive workflow results
    """
    executor = get_executor()
    executor.reset()
    executor.set_tool_invoker(invoke_tool_func)
    
    if verbose:
        print("╔══════════════════════════════════════════════════════════════════════════════╗")
        print("║           HIT IT - FULL 30-MODULE WORKFLOW                                    ║")
        print("║           Token-Integrated Training Cycle                                     ║")
        print("╚══════════════════════════════════════════════════════════════════════════════╝")
    
    return executor.run(verbose=verbose)

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Workflow Orchestration Module")
    print("=" * 60)
    print(f"Total steps defined: {len(get_full_workflow())}")
    print()
    print("Phases:")
    for phase in WorkflowPhase:
        steps = [s for s in get_full_workflow() if s.phase == phase]
        print(f"  {phase.value}: {len(steps)} steps")
    print()
    print("To execute: call hit_it(invoke_tool) from tool_shed")
