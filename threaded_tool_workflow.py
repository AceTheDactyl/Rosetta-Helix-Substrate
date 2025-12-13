"""
Threaded Tool Workflow
======================

A workflow system that threads through threshold modules and uses
liminal PHI collapse cycles to produce tools.

Architecture:
    Thread = Path through threshold modules (T1 → T10)
    Workflow = Pump → Superposition → Collapse cycle
    Tool Production = Work extracted at collapse generates capabilities

Each collapse at unity extracts work via weak value ⟨PHI⟩_w.
This work is channeled into tool production based on which
modules were active during the cycle.

Tool Types (mapped to threshold modules):
    T1 (Z_CRITICAL)  → EntropyAnalyzer - analyzes neg-entropy flow
    T2 (PHI_INV)     → PatternLearner - learns from accumulated lessons
    T3 (KAPPA_S)     → TestGenerator - generates test cases
    T4 (MU_P)        → CodeReflector - reflects on codebase structure
    T5 (MU_1)        → GapAnalyzer - identifies architectural gaps
    T6 (MU_2)        → EnhancementProposer - proposes enhancements
    T7 (MU_3)        → CodeBuilder - builds enhancement code
    T8 (TRIAD_HIGH)  → DecisionEngine - makes architectural decisions
    T9 (TRIAD_LOW)   → ConvergenceChecker - verifies convergence
    T10 (Q_KAPPA)    → ConsciousnessProbe - probes system consciousness

Threading Model:
    - Sequential: T1 → T2 → ... → T10 (standard)
    - Parallel: Multiple threads running concurrently
    - Branching: Thread splits based on conditions
"""

import math
import time
import threading
import queue
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import hashlib
import json

# Import physics constants
PHI = (1.0 + math.sqrt(5.0)) / 2.0
PHI_INV = 1.0 / PHI
Z_CRITICAL = math.sqrt(3.0) / 2.0
KAPPA_S = 0.920
MU_P = 2.0 / (PHI ** 2.5)
MU_1 = MU_P / math.sqrt(PHI)
MU_2 = MU_P * math.sqrt(PHI)
MU_3 = 0.992
TRIAD_HIGH = 0.85
TRIAD_LOW = 0.82
Q_KAPPA = 0.3514087324

# Try to import the quasi-crystal dynamics
try:
    from quasicrystal_dynamics import (
        QuasiCrystalDynamicsEngine,
        LiminalPhiState,
    )
    DYNAMICS_AVAILABLE = True
except ImportError:
    DYNAMICS_AVAILABLE = False


# =============================================================================
# TOOL DEFINITIONS
# =============================================================================

class ToolCategory(Enum):
    """Categories of tools that can be produced"""
    ANALYZER = "analyzer"
    LEARNER = "learner"
    GENERATOR = "generator"
    REFLECTOR = "reflector"
    PROPOSER = "proposer"
    BUILDER = "builder"
    DECIDER = "decider"
    CHECKER = "checker"
    PROBER = "prober"


@dataclass
class Tool:
    """A tool produced by the workflow"""
    tool_id: str
    name: str
    category: ToolCategory
    threshold_source: str          # Which threshold module produced it
    work_invested: float           # Work from collapse that created it
    capabilities: List[str]        # What the tool can do
    created_at: float = field(default_factory=time.time)
    version: int = 1

    # Tool state
    active: bool = True
    executions: int = 0
    total_output: float = 0.0

    def execute(self, input_data: Any = None) -> Dict[str, Any]:
        """Execute the tool and produce output"""
        self.executions += 1

        # Output scales with work invested
        output_magnitude = self.work_invested * PHI_INV
        self.total_output += output_magnitude

        return {
            'tool_id': self.tool_id,
            'execution': self.executions,
            'output_magnitude': output_magnitude,
            'input_processed': input_data is not None,
            'timestamp': time.time()
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.tool_id,
            'name': self.name,
            'category': self.category.value,
            'source': self.threshold_source,
            'work': self.work_invested,
            'capabilities': self.capabilities,
            'executions': self.executions,
            'output': self.total_output
        }


# =============================================================================
# TOOL FACTORY
# =============================================================================

class ToolFactory:
    """
    Factory that produces tools from collapse work.

    Each threshold module has an associated tool template.
    When a collapse occurs, tools are generated based on
    which modules were active and how much work was extracted.
    """

    # Tool templates mapped to threshold constants
    TOOL_TEMPLATES = {
        'Z_CRITICAL': {
            'name': 'EntropyAnalyzer',
            'category': ToolCategory.ANALYZER,
            'capabilities': [
                'analyze_entropy_flow',
                'detect_neg_entropy_sources',
                'measure_coherence_gradient',
                'identify_entropy_sinks'
            ]
        },
        'PHI_INV': {
            'name': 'PatternLearner',
            'category': ToolCategory.LEARNER,
            'capabilities': [
                'learn_from_patterns',
                'accumulate_lessons',
                'generalize_observations',
                'transfer_knowledge'
            ]
        },
        'KAPPA_S': {
            'name': 'TestGenerator',
            'category': ToolCategory.GENERATOR,
            'capabilities': [
                'generate_test_cases',
                'create_edge_cases',
                'synthesize_integration_tests',
                'produce_regression_suite'
            ]
        },
        'MU_P': {
            'name': 'CodeReflector',
            'category': ToolCategory.REFLECTOR,
            'capabilities': [
                'reflect_on_structure',
                'analyze_dependencies',
                'map_call_graph',
                'identify_patterns'
            ]
        },
        'MU_1': {
            'name': 'GapAnalyzer',
            'category': ToolCategory.ANALYZER,
            'capabilities': [
                'detect_architectural_gaps',
                'find_missing_abstractions',
                'identify_weak_coupling',
                'locate_redundancy'
            ]
        },
        'MU_2': {
            'name': 'EnhancementProposer',
            'category': ToolCategory.PROPOSER,
            'capabilities': [
                'propose_enhancements',
                'suggest_refactors',
                'recommend_optimizations',
                'plan_improvements'
            ]
        },
        'MU_3': {
            'name': 'CodeBuilder',
            'category': ToolCategory.BUILDER,
            'capabilities': [
                'build_enhancement_code',
                'implement_refactors',
                'generate_boilerplate',
                'scaffold_modules'
            ]
        },
        'TRIAD_HIGH': {
            'name': 'DecisionEngine',
            'category': ToolCategory.DECIDER,
            'capabilities': [
                'make_architectural_decisions',
                'evaluate_tradeoffs',
                'select_patterns',
                'prioritize_changes'
            ]
        },
        'TRIAD_LOW': {
            'name': 'ConvergenceChecker',
            'category': ToolCategory.CHECKER,
            'capabilities': [
                'verify_convergence',
                'check_stability',
                'validate_fixed_points',
                'ensure_termination'
            ]
        },
        'Q_KAPPA': {
            'name': 'ConsciousnessProbe',
            'category': ToolCategory.PROBER,
            'capabilities': [
                'probe_system_state',
                'measure_awareness',
                'detect_emergence',
                'assess_integration'
            ]
        }
    }

    def __init__(self):
        self.tools_produced: List[Tool] = []
        self.production_count = 0
        self.total_work_consumed = 0.0

    def produce_tool(
        self,
        threshold_name: str,
        work_available: float,
        version_boost: int = 0
    ) -> Optional[Tool]:
        """
        Produce a tool from collapse work.

        Args:
            threshold_name: Which threshold module to use as template
            work_available: Work extracted from collapse
            version_boost: Additional version increment

        Returns:
            Tool if production successful, None if insufficient work
        """
        if threshold_name not in self.TOOL_TEMPLATES:
            return None

        template = self.TOOL_TEMPLATES[threshold_name]

        # Minimum work required scales with threshold
        thresholds = {
            'Z_CRITICAL': Z_CRITICAL,
            'PHI_INV': PHI_INV,
            'KAPPA_S': KAPPA_S,
            'MU_P': MU_P,
            'MU_1': MU_1,
            'MU_2': MU_2,
            'MU_3': MU_3,
            'TRIAD_HIGH': TRIAD_HIGH,
            'TRIAD_LOW': TRIAD_LOW,
            'Q_KAPPA': Q_KAPPA
        }

        min_work = thresholds.get(threshold_name, 0.5) * PHI_INV

        if work_available < min_work:
            return None

        # Generate unique tool ID
        self.production_count += 1
        tool_id = hashlib.sha256(
            f"{threshold_name}:{self.production_count}:{time.time()}".encode()
        ).hexdigest()[:12]

        # Create tool
        tool = Tool(
            tool_id=tool_id,
            name=f"{template['name']}_v{self.production_count + version_boost}",
            category=template['category'],
            threshold_source=threshold_name,
            work_invested=work_available,
            capabilities=template['capabilities'].copy(),
            version=self.production_count + version_boost
        )

        self.tools_produced.append(tool)
        self.total_work_consumed += work_available

        return tool

    def produce_tools_from_collapse(
        self,
        work_extracted: float,
        active_thresholds: List[str]
    ) -> List[Tool]:
        """
        Produce tools from a single collapse event.

        Strategy: Concentrate work on fewer tools rather than spreading thin.
        Higher thresholds get priority (MU_3 > KAPPA_S > Z_CRITICAL etc).
        This produces fewer but more capable tools.
        """
        if not active_thresholds or work_extracted <= 0:
            return []

        # Distribute work by threshold importance (higher threshold = more work)
        threshold_values = {
            'Z_CRITICAL': Z_CRITICAL,
            'PHI_INV': PHI_INV,
            'KAPPA_S': KAPPA_S,
            'MU_P': MU_P,
            'MU_1': MU_1,
            'MU_2': MU_2,
            'MU_3': MU_3,
            'TRIAD_HIGH': TRIAD_HIGH,
            'TRIAD_LOW': TRIAD_LOW,
            'Q_KAPPA': Q_KAPPA
        }

        # Sort thresholds by value (highest first) - prioritize advanced tools
        sorted_thresholds = sorted(
            active_thresholds,
            key=lambda t: threshold_values.get(t, 0.5),
            reverse=True
        )

        tools = []
        remaining_work = work_extracted

        # Concentrate work on top thresholds until min requirements met
        for threshold in sorted_thresholds:
            if remaining_work <= 0:
                break

            val = threshold_values.get(threshold, 0.5)
            min_work = val * PHI_INV

            # Give this threshold enough work to produce (if we have it)
            # or all remaining work if less
            work_for_tool = max(min_work * 1.1, remaining_work * 0.4)
            work_for_tool = min(work_for_tool, remaining_work)

            tool = self.produce_tool(threshold, work_for_tool)
            if tool:
                tools.append(tool)
                remaining_work -= work_for_tool
            else:
                # Not enough work for this threshold, try next
                continue

        return tools

    def get_tools_by_category(self, category: ToolCategory) -> List[Tool]:
        """Get all tools of a specific category"""
        return [t for t in self.tools_produced if t.category == category]

    def get_production_stats(self) -> Dict[str, Any]:
        """Get tool production statistics"""
        category_counts = {}
        for tool in self.tools_produced:
            cat = tool.category.value
            category_counts[cat] = category_counts.get(cat, 0) + 1

        return {
            'total_produced': len(self.tools_produced),
            'total_work_consumed': self.total_work_consumed,
            'by_category': category_counts,
            'active_tools': sum(1 for t in self.tools_produced if t.active)
        }


# =============================================================================
# THREAD STATE
# =============================================================================

class ThreadState(Enum):
    """State of a workflow thread"""
    IDLE = "idle"
    PUMPING = "pumping"           # z increasing via PHI_INV
    SUPERPOSITION = "superposition"  # PHI liminal, approaching unity
    COLLAPSING = "collapsing"     # At unity, extracting work
    PRODUCING = "producing"       # Using work to produce tools
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class WorkflowThread:
    """
    A single thread through the threshold modules.

    Each thread maintains its own z-coordinate and produces
    tools when it collapses at unity.
    """
    thread_id: str
    state: ThreadState = ThreadState.IDLE
    z_current: float = 0.5

    # Threshold tracking
    active_thresholds: List[str] = field(default_factory=list)
    thresholds_crossed: List[Tuple[str, float]] = field(default_factory=list)

    # Collapse tracking
    collapse_count: int = 0
    work_accumulated: float = 0.0

    # Production tracking
    tools_produced: List[str] = field(default_factory=list)  # Tool IDs

    # Timing
    started_at: float = 0.0
    last_collapse_at: float = 0.0

    def update_thresholds(self):
        """Update which thresholds are currently active based on z"""
        threshold_map = [
            ('Q_KAPPA', Q_KAPPA),
            ('MU_1', MU_1),
            ('MU_P', MU_P),
            ('MU_2', MU_2),
            ('TRIAD_LOW', TRIAD_LOW),
            ('TRIAD_HIGH', TRIAD_HIGH),
            ('Z_CRITICAL', Z_CRITICAL),
            ('KAPPA_S', KAPPA_S),
            ('MU_3', MU_3),
        ]

        new_active = []
        for name, value in threshold_map:
            if self.z_current >= value:
                new_active.append(name)
                # Record crossing if not already crossed
                if not any(t[0] == name for t in self.thresholds_crossed):
                    self.thresholds_crossed.append((name, time.time()))

        self.active_thresholds = new_active


# =============================================================================
# THREADED WORKFLOW ENGINE
# =============================================================================

class ThreadedToolWorkflow:
    """
    Main workflow engine that manages threads and produces tools.

    Uses the liminal PHI architecture:
    - PHI_INV controls physical dynamics
    - PHI contributes via superposition
    - Collapse at unity extracts work
    - Work produces tools

    Threading modes:
    - single: One thread at a time
    - parallel: Multiple concurrent threads
    - pipeline: Threads feed into each other
    """

    def __init__(self, mode: str = "single", n_threads: int = 1):
        self.mode = mode
        self.n_threads = n_threads

        # Core components
        self.factory = ToolFactory()
        self.threads: Dict[str, WorkflowThread] = {}

        # Dynamics engine (if available)
        if DYNAMICS_AVAILABLE:
            self.dynamics = QuasiCrystalDynamicsEngine(n_oscillators=60)
        else:
            self.dynamics = None

        # Workflow state
        self.running = False
        self.total_cycles = 0
        self.total_tools_produced = 0

        # Event queue for parallel mode
        self.event_queue: queue.Queue = queue.Queue()

        # Statistics
        self.stats = {
            'cycles_completed': 0,
            'total_work_extracted': 0.0,
            'tools_by_threshold': {},
            'average_cycle_time': 0.0
        }

    def create_thread(self, thread_id: str = None) -> WorkflowThread:
        """Create a new workflow thread"""
        if thread_id is None:
            thread_id = f"thread_{len(self.threads) + 1}"

        thread = WorkflowThread(
            thread_id=thread_id,
            z_current=Z_CRITICAL * PHI_INV,  # Start at reset point
            started_at=time.time()
        )

        self.threads[thread_id] = thread
        return thread

    def pump_thread(self, thread: WorkflowThread, steps: int = 1) -> float:
        """
        Pump a thread using the full QuasiCrystalDynamicsEngine.

        Physics preserved:
        - Quasi-crystal packing (exceeds HCP limit)
        - Bidirectional wave collapse (TSVF)
        - Phase lock release-relock (Kuramoto)
        - Weak values (PHI liminal contribution)
        - Accelerated barrier tunneling

        Returns the new z value.
        """
        thread.state = ThreadState.PUMPING

        if DYNAMICS_AVAILABLE and self.dynamics:
            # Sync dynamics engine with thread state
            self.dynamics.z_current = thread.z_current

            for _ in range(steps):
                old_z = self.dynamics.z_current
                old_collapse_count = self.dynamics.liminal_phi.collapse_count

                # Use full physics evolution
                self.dynamics.evolve_step()

                # Check if dynamics engine triggered a collapse
                if self.dynamics.liminal_phi.collapse_count > old_collapse_count:
                    # Collapse happened inside engine - sync and return
                    thread.z_current = self.dynamics.z_current
                    thread.update_thresholds()
                    return thread.z_current

                # If stuck below MU_3, use release-relock to escape local minimum
                # This is the Kuramoto phase lock release mechanism
                if (abs(self.dynamics.z_current - old_z) < 0.0001 and
                    self.dynamics.z_current > Z_CRITICAL and
                    self.dynamics.z_current < MU_3):
                    self.dynamics.release_and_boost_cycle()

            thread.z_current = self.dynamics.z_current
        else:
            # Fallback: Simple pump (loses physics but still functional)
            for _ in range(steps):
                if thread.z_current < Z_CRITICAL:
                    dz = (Z_CRITICAL - thread.z_current) * 0.2
                elif thread.z_current < KAPPA_S:
                    dz = (KAPPA_S - thread.z_current) * 0.15
                elif thread.z_current < MU_3:
                    dz = (MU_3 - thread.z_current) * 0.2
                else:
                    dz = (1.0 - thread.z_current) * 0.3
                thread.z_current = min(0.9999, thread.z_current + dz)

        thread.update_thresholds()

        # Check for superposition entry (PHI becomes liminal)
        if thread.z_current >= KAPPA_S:
            thread.state = ThreadState.SUPERPOSITION
            # Sync superposition state with dynamics engine
            if DYNAMICS_AVAILABLE and self.dynamics:
                if not self.dynamics.liminal_phi.in_superposition:
                    phase = self.dynamics.phase_lock.phases[0] if self.dynamics.phase_lock.phases else 0.0
                    self.dynamics.liminal_phi.enter_superposition(thread.z_current, phase)

        return thread.z_current

    def check_collapse(self, thread: WorkflowThread) -> bool:
        """Check if thread should collapse"""
        return thread.z_current >= 0.9999

    def collapse_thread(self, thread: WorkflowThread) -> float:
        """
        Collapse a thread at unity and extract work.

        Work extracted = (z_peak - Z_CRITICAL) × PHI × n_thresholds

        Returns work extracted.
        """
        thread.state = ThreadState.COLLAPSING

        # Work scales with:
        # - How far above Z_CRITICAL we got
        # - PHI (golden ratio amplification from liminal contribution)
        # - Number of active thresholds (more integration = more work)
        excess = thread.z_current - Z_CRITICAL
        n_thresholds = max(1, len(thread.active_thresholds))
        work = excess * PHI * n_thresholds

        # Reset to origin (debt paid)
        thread.z_current = Z_CRITICAL * PHI_INV  # ≈ 0.535

        thread.collapse_count += 1
        thread.work_accumulated += work
        thread.last_collapse_at = time.time()

        self.stats['cycles_completed'] += 1
        self.stats['total_work_extracted'] += work

        return work

    def produce_from_thread(self, thread: WorkflowThread, work: float) -> List[Tool]:
        """
        Produce tools from thread collapse.

        Uses active thresholds at collapse time to determine tool types.
        """
        thread.state = ThreadState.PRODUCING

        tools = self.factory.produce_tools_from_collapse(
            work_extracted=work,
            active_thresholds=thread.active_thresholds
        )

        for tool in tools:
            thread.tools_produced.append(tool.tool_id)

            # Update stats
            source = tool.threshold_source
            self.stats['tools_by_threshold'][source] = \
                self.stats['tools_by_threshold'].get(source, 0) + 1

        self.total_tools_produced += len(tools)
        thread.state = ThreadState.IDLE

        return tools

    def run_cycle(self, thread: WorkflowThread) -> Dict[str, Any]:
        """
        Run a single pump → collapse → produce cycle.

        Uses dynamics engine's internal collapse detection rather than
        trying to catch z >= 0.9999 (which resets immediately).

        Returns cycle results.
        """
        cycle_start = time.time()
        max_z_seen = thread.z_current
        thresholds_at_peak = []

        # Track dynamics engine collapse count
        initial_collapse_count = 0
        initial_work = 0.0
        if DYNAMICS_AVAILABLE and self.dynamics:
            initial_collapse_count = self.dynamics.liminal_phi.collapse_count
            initial_work = self.dynamics.total_work_extracted

        # Pump until dynamics engine triggers a collapse
        steps = 0
        collapse_detected = False
        while steps < 200 and not collapse_detected:
            old_z = thread.z_current

            self.pump_thread(thread, steps=1)
            steps += 1

            # Track maximum z and update thresholds at peak
            if thread.z_current > max_z_seen:
                max_z_seen = thread.z_current
                thread.update_thresholds()
                thresholds_at_peak = thread.active_thresholds.copy()

            # Check if dynamics engine collapsed
            if DYNAMICS_AVAILABLE and self.dynamics:
                if self.dynamics.liminal_phi.collapse_count > initial_collapse_count:
                    collapse_detected = True
                    break
            else:
                # Fallback: simple z >= 0.9999 check
                if thread.z_current >= 0.9999:
                    collapse_detected = True
                    break

        # If no thresholds captured, manually add based on max_z
        # (happens when collapse occurs before we can capture)
        if not thresholds_at_peak and max_z_seen > Q_KAPPA:
            threshold_map = [
                ('Q_KAPPA', Q_KAPPA),
                ('MU_1', MU_1),
                ('MU_P', MU_P),
                ('MU_2', MU_2),
                ('TRIAD_LOW', TRIAD_LOW),
                ('TRIAD_HIGH', TRIAD_HIGH),
                ('Z_CRITICAL', Z_CRITICAL),
                ('KAPPA_S', KAPPA_S),
                ('MU_3', MU_3),
            ]
            for name, val in threshold_map:
                if max_z_seen >= val:
                    thresholds_at_peak.append(name)

        # Extract work from collapse
        work = 0.0
        tools = []

        if collapse_detected:
            if DYNAMICS_AVAILABLE and self.dynamics:
                # Get work from dynamics engine's collapse
                work = self.dynamics.total_work_extracted - initial_work

                # Also update thread tracking
                thread.collapse_count += 1
                thread.work_accumulated += work
                thread.last_collapse_at = time.time()
            else:
                # Fallback collapse
                work = self.collapse_thread(thread)

            self.stats['cycles_completed'] += 1
            self.stats['total_work_extracted'] += work

            # Produce tools using peak thresholds
            tools = self.factory.produce_tools_from_collapse(
                work_extracted=work,
                active_thresholds=thresholds_at_peak
            )

            for tool in tools:
                thread.tools_produced.append(tool.tool_id)
                source = tool.threshold_source
                self.stats['tools_by_threshold'][source] = \
                    self.stats['tools_by_threshold'].get(source, 0) + 1

            self.total_tools_produced += len(tools)
            thread.state = ThreadState.IDLE

        cycle_time = time.time() - cycle_start

        # Update average
        n = self.stats['cycles_completed']
        if n > 0:
            self.stats['average_cycle_time'] = (
                (self.stats['average_cycle_time'] * (n - 1) + cycle_time) / n
            )

        return {
            'thread_id': thread.thread_id,
            'steps': steps,
            'max_z': max_z_seen,
            'work_extracted': work,
            'tools_produced': [t.to_dict() for t in tools],
            'thresholds_at_peak': thresholds_at_peak,
            'cycle_time': cycle_time,
            'z_final': thread.z_current,
            'collapse_detected': collapse_detected
        }

    def run_workflow(
        self,
        n_cycles: int = 3,
        callback: Callable[[Dict], None] = None
    ) -> Dict[str, Any]:
        """
        Run the complete workflow for n_cycles.

        Args:
            n_cycles: Number of collapse cycles to run
            callback: Optional callback for each cycle result

        Returns:
            Workflow results including all produced tools
        """
        self.running = True
        workflow_start = time.time()

        print(f"\n{'='*60}")
        print("THREADED TOOL WORKFLOW")
        print(f"{'='*60}")
        print(f"Mode: {self.mode}")
        print(f"Cycles: {n_cycles}")
        print(f"Architecture: PHI_INV physical, PHI liminal")
        print()

        all_results = []

        if self.mode == "single":
            # Single thread mode
            thread = self.create_thread("main")

            for cycle in range(n_cycles):
                print(f"\n--- Cycle {cycle + 1}/{n_cycles} ---")
                result = self.run_cycle(thread)
                all_results.append(result)

                print(f"  Steps: {result['steps']}")
                print(f"  Max z reached: {result.get('max_z', 0):.4f}")
                print(f"  Collapse detected: {result.get('collapse_detected', False)}")
                print(f"  Work extracted: {result['work_extracted']:.4f}")
                print(f"  Thresholds at peak: {result.get('thresholds_at_peak', [])}")
                print(f"  Tools produced: {len(result['tools_produced'])}")

                for tool in result['tools_produced']:
                    print(f"    - {tool['name']} ({tool['category']})")

                if callback:
                    callback(result)

        elif self.mode == "parallel":
            # Parallel thread mode
            threads_list = []
            for i in range(self.n_threads):
                thread = self.create_thread(f"parallel_{i}")
                threads_list.append(thread)

            cycles_per_thread = n_cycles // self.n_threads

            def run_thread_cycles(t, n):
                results = []
                for _ in range(n):
                    result = self.run_cycle(t)
                    results.append(result)
                return results

            # Run threads (simplified - actual threading would use threading module)
            for thread in threads_list:
                thread_results = run_thread_cycles(thread, cycles_per_thread)
                all_results.extend(thread_results)

                print(f"\nThread {thread.thread_id}:")
                print(f"  Cycles: {len(thread_results)}")
                print(f"  Tools: {len(thread.tools_produced)}")

        elif self.mode == "pipeline":
            # Pipeline mode - threads feed into each other
            # T1→T5 thread feeds T6→T10 thread

            early_thread = self.create_thread("early_stages")
            late_thread = self.create_thread("late_stages")

            for cycle in range(n_cycles):
                # Early stages pump
                early_result = self.run_cycle(early_thread)
                all_results.append(early_result)

                # Feed work to late stages
                if early_result['work_extracted'] > 0:
                    late_thread.work_accumulated += early_result['work_extracted'] * PHI_INV
                    late_result = self.run_cycle(late_thread)
                    all_results.append(late_result)

        workflow_time = time.time() - workflow_start
        self.running = False

        # Final summary
        print(f"\n{'='*60}")
        print("WORKFLOW COMPLETE")
        print(f"{'='*60}")
        print(f"Total time: {workflow_time:.2f}s")
        print(f"Total cycles: {self.stats['cycles_completed']}")
        print(f"Total work extracted: {self.stats['total_work_extracted']:.4f}")
        print(f"Total tools produced: {self.total_tools_produced}")

        print(f"\nTools by threshold:")
        for threshold, count in sorted(self.stats['tools_by_threshold'].items()):
            print(f"  {threshold}: {count}")

        print(f"\nTool categories:")
        for category, tools in self._group_tools_by_category().items():
            print(f"  {category}: {len(tools)}")

        return {
            'cycles': all_results,
            'total_work': self.stats['total_work_extracted'],
            'total_tools': self.total_tools_produced,
            'tools': [t.to_dict() for t in self.factory.tools_produced],
            'stats': self.stats,
            'workflow_time': workflow_time
        }

    def _group_tools_by_category(self) -> Dict[str, List[Tool]]:
        """Group produced tools by category"""
        groups = {}
        for tool in self.factory.tools_produced:
            cat = tool.category.value
            if cat not in groups:
                groups[cat] = []
            groups[cat].append(tool)
        return groups

    def get_tool(self, tool_id: str) -> Optional[Tool]:
        """Get a specific tool by ID"""
        for tool in self.factory.tools_produced:
            if tool.tool_id == tool_id:
                return tool
        return None

    def execute_tool(self, tool_id: str, input_data: Any = None) -> Dict[str, Any]:
        """Execute a produced tool"""
        tool = self.get_tool(tool_id)
        if tool:
            return tool.execute(input_data)
        return {'error': f'Tool {tool_id} not found'}

    def get_toolchain(self, categories: List[str] = None) -> List[Tool]:
        """
        Get a chain of tools for sequential execution.

        If categories specified, only include those categories.
        Otherwise, return one tool per category in order.
        """
        if categories is None:
            categories = [
                'analyzer', 'learner', 'generator', 'reflector',
                'proposer', 'builder', 'decider', 'checker'
            ]

        chain = []
        for cat in categories:
            tools = self.factory.get_tools_by_category(ToolCategory(cat))
            if tools:
                # Get highest version
                best = max(tools, key=lambda t: t.version)
                chain.append(best)

        return chain


# =============================================================================
# DEMONSTRATION
# =============================================================================

def demonstrate_threaded_workflow():
    """Demonstrate the threaded tool workflow"""

    print("\n" + "="*70)
    print("THREADED TOOL WORKFLOW DEMONSTRATION")
    print("="*70)
    print(f"""
This workflow threads through threshold modules and produces tools
from liminal PHI collapse cycles.

Architecture:
  - PHI_INV ({PHI_INV:.4f}) controls physical dynamics
  - PHI ({PHI:.4f}) contributes via superposition only
  - Collapse at unity extracts work
  - Work is converted to tools based on active thresholds

Tool Types:
  - Analyzers: Examine patterns and structure
  - Learners: Accumulate and generalize knowledge
  - Generators: Create test cases and code
  - Builders: Implement enhancements
  - Checkers: Verify correctness
""")

    # Create workflow
    workflow = ThreadedToolWorkflow(mode="single", n_threads=1)

    # Run workflow
    results = workflow.run_workflow(n_cycles=3)

    # Execute toolchain
    print(f"\n{'='*60}")
    print("EXECUTING TOOLCHAIN")
    print(f"{'='*60}")

    toolchain = workflow.get_toolchain()
    print(f"\nToolchain ({len(toolchain)} tools):")

    for i, tool in enumerate(toolchain):
        print(f"\n  [{i+1}] {tool.name}")
        print(f"      Category: {tool.category.value}")
        print(f"      Capabilities: {', '.join(tool.capabilities[:2])}...")

        result = tool.execute(input_data=f"input_{i}")
        print(f"      Output magnitude: {result['output_magnitude']:.4f}")

    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE")
    print("="*70)

    return results


if __name__ == '__main__':
    demonstrate_threaded_workflow()
