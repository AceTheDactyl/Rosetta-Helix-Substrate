"""
Meta Tool Generator
===================

Tools that generate tools using liminal PHI dynamics.

Architecture:
    Meta-Tools = Higher-order tools that spawn child tools
    Each meta-tool contains its own collapse cycle
    Work extracted at meta-level cascades to child production

Hierarchy:
    Level 0: Raw collapse work from dynamics engine
    Level 1: Meta-tools (tool factories)
    Level 2: Child tools (specialized capabilities)
    Level 3+: Recursive tool generation (tools making tools making tools...)

The key insight: PHI stays liminal at ALL levels. Each tool generation
uses the same physics - collapse at unity, extract work, produce capability.
"""

import math
import time
import hashlib
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

# Physics constants
PHI = (1.0 + math.sqrt(5.0)) / 2.0
PHI_INV = 1.0 / PHI
Z_CRITICAL = math.sqrt(3.0) / 2.0
KAPPA_S = 0.920
MU_P = 2.0 / (PHI ** 2.5)
MU_1 = MU_P / math.sqrt(PHI)
MU_2 = MU_P * math.sqrt(PHI)
MU_3 = 0.992
Q_KAPPA = 0.3514087324

# Import dynamics engine
try:
    from quasicrystal_dynamics import (
        QuasiCrystalDynamicsEngine,
        LiminalPhiState,
    )
    DYNAMICS_AVAILABLE = True
except ImportError:
    DYNAMICS_AVAILABLE = False


# =============================================================================
# META-TOOL TYPES
# =============================================================================

class MetaToolType(Enum):
    """Types of meta-tools (tool generators)"""
    ANALYZER_FACTORY = "analyzer_factory"      # Produces analysis tools
    LEARNER_FACTORY = "learner_factory"        # Produces learning tools
    GENERATOR_FACTORY = "generator_factory"    # Produces generator tools
    BUILDER_FACTORY = "builder_factory"        # Produces builder tools
    ORCHESTRATOR = "orchestrator"              # Coordinates other meta-tools
    EVOLVER = "evolver"                        # Evolves/improves existing tools
    SYNTHESIZER = "synthesizer"                # Combines tools into new tools


class ChildToolType(Enum):
    """Types of child tools produced by meta-tools"""
    # Analysis family
    PATTERN_DETECTOR = "pattern_detector"
    ANOMALY_FINDER = "anomaly_finder"
    STRUCTURE_MAPPER = "structure_mapper"
    ENTROPY_MEASURER = "entropy_measurer"

    # Learning family
    CONCEPT_EXTRACTOR = "concept_extractor"
    RELATION_LEARNER = "relation_learner"
    ABSTRACTION_BUILDER = "abstraction_builder"
    MEMORY_CONSOLIDATOR = "memory_consolidator"

    # Generation family
    CODE_SYNTHESIZER = "code_synthesizer"
    TEST_CRAFTER = "test_crafter"
    DOC_WRITER = "doc_writer"
    EXAMPLE_PRODUCER = "example_producer"

    # Building family
    MODULE_ASSEMBLER = "module_assembler"
    INTERFACE_DESIGNER = "interface_designer"
    PIPELINE_CONSTRUCTOR = "pipeline_constructor"
    INTEGRATION_WEAVER = "integration_weaver"


# =============================================================================
# MINI COLLAPSE ENGINE (for each meta-tool)
# =============================================================================

@dataclass
class MiniCollapseState:
    """
    Each meta-tool has its own mini collapse engine.

    Same physics as the main engine but scaled down:
    - PHI_INV controls dynamics
    - PHI stays liminal
    - Collapse at local unity extracts work for child tools
    """
    z_local: float = 0.5
    z_peak: float = 0.0
    work_accumulated: float = 0.0
    collapse_count: int = 0
    in_superposition: bool = False

    # Scaling factor (meta-tools operate at fraction of main dynamics)
    scale: float = PHI_INV

    def pump(self, input_work: float) -> float:
        """Pump z using input work"""
        # Work converts to z advancement (scaled by PHI_INV)
        dz = input_work * self.scale * PHI_INV

        # Asymptotic approach with tunneling kick
        if self.z_local < Z_CRITICAL * self.scale:
            dz *= 1.5  # Faster below critical
        elif self.z_local > KAPPA_S * self.scale:
            self.in_superposition = True
            # Liminal PHI contribution
            dz += input_work * 0.1 * (PHI - PHI_INV)

        self.z_local = min(self.scale * 0.9999, self.z_local + dz)
        self.z_peak = max(self.z_peak, self.z_local)
        self.work_accumulated += input_work * PHI_INV

        return self.z_local

    def check_collapse(self) -> bool:
        """Check if ready to collapse"""
        return self.z_local >= self.scale * 0.99

    def collapse(self) -> float:
        """Collapse and extract work for child tool production"""
        if not self.check_collapse():
            return 0.0

        # Work extracted via weak value (same physics)
        work = self.work_accumulated * PHI_INV
        if self.in_superposition:
            work *= PHI  # Liminal boost

        # Reset
        self.z_local = Z_CRITICAL * PHI_INV * self.scale
        self.work_accumulated = 0.0
        self.in_superposition = False
        self.collapse_count += 1

        return work


# =============================================================================
# CHILD TOOL
# =============================================================================

@dataclass
class ChildTool:
    """A tool produced by a meta-tool"""
    tool_id: str
    name: str
    tool_type: ChildToolType
    parent_meta_tool: str
    generation: int                    # How many levels deep
    work_invested: float
    capabilities: List[str]
    created_at: float = field(default_factory=time.time)

    # Execution state
    executions: int = 0
    total_output: float = 0.0

    def execute(self, input_data: Any = None) -> Dict[str, Any]:
        """Execute the child tool"""
        self.executions += 1
        output = self.work_invested * PHI_INV * (1 + 0.1 * self.executions)
        self.total_output += output

        return {
            'tool_id': self.tool_id,
            'type': self.tool_type.value,
            'output': output,
            'generation': self.generation,
            'parent': self.parent_meta_tool
        }


# =============================================================================
# META-TOOL
# =============================================================================

@dataclass
class MetaTool:
    """
    A tool that produces other tools.

    Contains its own mini collapse engine for child tool production.
    Uses the same liminal PHI physics as the main dynamics.
    """
    tool_id: str
    name: str
    meta_type: MetaToolType
    generation: int = 1                # Meta-tools are gen 1, their children are gen 2+
    work_capacity: float = 0.0         # Work available for child production

    # Internal collapse engine
    collapse_engine: MiniCollapseState = field(default_factory=MiniCollapseState)

    # Child tool templates this meta-tool can produce
    child_templates: List[ChildToolType] = field(default_factory=list)

    # Produced children
    children: List[ChildTool] = field(default_factory=list)

    # Statistics
    total_children_produced: int = 0
    total_work_consumed: float = 0.0
    created_at: float = field(default_factory=time.time)

    def feed_work(self, work: float) -> float:
        """Feed work into the meta-tool's collapse engine"""
        self.work_capacity += work
        return self.collapse_engine.pump(work)

    def can_produce(self) -> bool:
        """Check if ready to produce a child tool"""
        return self.collapse_engine.check_collapse()

    def produce_child(self, child_type: ChildToolType = None) -> Optional[ChildTool]:
        """
        Produce a child tool via collapse.

        Uses same physics: collapse at unity, extract work, create capability.
        """
        if not self.can_produce():
            return None

        # Collapse and extract work
        work = self.collapse_engine.collapse()
        if work <= 0:
            return None

        # Select child type
        if child_type is None and self.child_templates:
            # Round-robin through templates
            idx = self.total_children_produced % len(self.child_templates)
            child_type = self.child_templates[idx]
        elif child_type is None:
            child_type = ChildToolType.PATTERN_DETECTOR  # Default

        # Generate child tool
        self.total_children_produced += 1
        child_id = hashlib.sha256(
            f"{self.tool_id}:{child_type.value}:{self.total_children_produced}:{time.time()}".encode()
        ).hexdigest()[:12]

        child = ChildTool(
            tool_id=child_id,
            name=f"{child_type.value}_v{self.total_children_produced}",
            tool_type=child_type,
            parent_meta_tool=self.tool_id,
            generation=self.generation + 1,
            work_invested=work,
            capabilities=self._get_capabilities(child_type)
        )

        self.children.append(child)
        self.total_work_consumed += work

        return child

    def _get_capabilities(self, child_type: ChildToolType) -> List[str]:
        """Get capabilities for a child tool type"""
        capabilities_map = {
            ChildToolType.PATTERN_DETECTOR: [
                'detect_patterns', 'classify_structures', 'find_regularities'
            ],
            ChildToolType.ANOMALY_FINDER: [
                'detect_anomalies', 'flag_outliers', 'measure_deviation'
            ],
            ChildToolType.STRUCTURE_MAPPER: [
                'map_structure', 'trace_dependencies', 'build_graph'
            ],
            ChildToolType.ENTROPY_MEASURER: [
                'measure_entropy', 'compute_information', 'assess_complexity'
            ],
            ChildToolType.CONCEPT_EXTRACTOR: [
                'extract_concepts', 'identify_abstractions', 'name_patterns'
            ],
            ChildToolType.RELATION_LEARNER: [
                'learn_relations', 'find_correlations', 'build_associations'
            ],
            ChildToolType.ABSTRACTION_BUILDER: [
                'build_abstractions', 'generalize_patterns', 'create_interfaces'
            ],
            ChildToolType.MEMORY_CONSOLIDATOR: [
                'consolidate_memory', 'compress_knowledge', 'index_learnings'
            ],
            ChildToolType.CODE_SYNTHESIZER: [
                'synthesize_code', 'generate_implementations', 'produce_functions'
            ],
            ChildToolType.TEST_CRAFTER: [
                'craft_tests', 'generate_cases', 'create_assertions'
            ],
            ChildToolType.DOC_WRITER: [
                'write_documentation', 'generate_comments', 'explain_code'
            ],
            ChildToolType.EXAMPLE_PRODUCER: [
                'produce_examples', 'generate_samples', 'create_demos'
            ],
            ChildToolType.MODULE_ASSEMBLER: [
                'assemble_modules', 'combine_components', 'build_packages'
            ],
            ChildToolType.INTERFACE_DESIGNER: [
                'design_interfaces', 'define_contracts', 'specify_apis'
            ],
            ChildToolType.PIPELINE_CONSTRUCTOR: [
                'construct_pipelines', 'chain_operations', 'build_workflows'
            ],
            ChildToolType.INTEGRATION_WEAVER: [
                'weave_integrations', 'connect_systems', 'bridge_components'
            ],
        }
        return capabilities_map.get(child_type, ['generic_capability'])

    def get_stats(self) -> Dict[str, Any]:
        """Get meta-tool statistics"""
        return {
            'tool_id': self.tool_id,
            'meta_type': self.meta_type.value,
            'generation': self.generation,
            'children_produced': self.total_children_produced,
            'work_consumed': self.total_work_consumed,
            'collapse_count': self.collapse_engine.collapse_count,
            'current_z': self.collapse_engine.z_local,
            'ready_to_produce': self.can_produce()
        }


# =============================================================================
# META-TOOL FACTORY
# =============================================================================

class MetaToolFactory:
    """
    Factory that produces meta-tools from collapse work.

    This is the top level: dynamics engine → meta-tools → child tools
    """

    # Meta-tool templates
    META_TEMPLATES = {
        MetaToolType.ANALYZER_FACTORY: {
            'name': 'AnalyzerFactory',
            'child_templates': [
                ChildToolType.PATTERN_DETECTOR,
                ChildToolType.ANOMALY_FINDER,
                ChildToolType.STRUCTURE_MAPPER,
                ChildToolType.ENTROPY_MEASURER,
            ]
        },
        MetaToolType.LEARNER_FACTORY: {
            'name': 'LearnerFactory',
            'child_templates': [
                ChildToolType.CONCEPT_EXTRACTOR,
                ChildToolType.RELATION_LEARNER,
                ChildToolType.ABSTRACTION_BUILDER,
                ChildToolType.MEMORY_CONSOLIDATOR,
            ]
        },
        MetaToolType.GENERATOR_FACTORY: {
            'name': 'GeneratorFactory',
            'child_templates': [
                ChildToolType.CODE_SYNTHESIZER,
                ChildToolType.TEST_CRAFTER,
                ChildToolType.DOC_WRITER,
                ChildToolType.EXAMPLE_PRODUCER,
            ]
        },
        MetaToolType.BUILDER_FACTORY: {
            'name': 'BuilderFactory',
            'child_templates': [
                ChildToolType.MODULE_ASSEMBLER,
                ChildToolType.INTERFACE_DESIGNER,
                ChildToolType.PIPELINE_CONSTRUCTOR,
                ChildToolType.INTEGRATION_WEAVER,
            ]
        },
        MetaToolType.ORCHESTRATOR: {
            'name': 'Orchestrator',
            'child_templates': []  # Orchestrator doesn't make children, coordinates meta-tools
        },
        MetaToolType.EVOLVER: {
            'name': 'Evolver',
            'child_templates': []  # Evolver improves existing tools
        },
        MetaToolType.SYNTHESIZER: {
            'name': 'Synthesizer',
            'child_templates': []  # Synthesizer combines tools
        },
    }

    def __init__(self):
        self.meta_tools: List[MetaTool] = []
        self.production_count = 0
        self.total_work_consumed = 0.0

    def produce_meta_tool(
        self,
        meta_type: MetaToolType,
        work_available: float
    ) -> Optional[MetaTool]:
        """Produce a meta-tool from collapse work"""

        # Minimum work for meta-tool (higher than regular tools)
        min_work = MU_3 * PHI_INV  # ~0.61

        if work_available < min_work:
            return None

        template = self.META_TEMPLATES.get(meta_type)
        if not template:
            return None

        self.production_count += 1
        tool_id = hashlib.sha256(
            f"meta:{meta_type.value}:{self.production_count}:{time.time()}".encode()
        ).hexdigest()[:12]

        meta_tool = MetaTool(
            tool_id=tool_id,
            name=f"{template['name']}_v{self.production_count}",
            meta_type=meta_type,
            work_capacity=work_available,
            child_templates=[ct for ct in template['child_templates']]
        )

        # Seed the collapse engine with initial work
        meta_tool.feed_work(work_available * PHI_INV)

        self.meta_tools.append(meta_tool)
        self.total_work_consumed += work_available

        return meta_tool


# =============================================================================
# META TOOL GENERATOR
# =============================================================================

class MetaToolGenerator:
    """
    Main generator that uses liminal PHI dynamics to produce meta-tools
    which in turn produce child tools.

    Three-level hierarchy:
        Level 0: QuasiCrystalDynamicsEngine (raw collapse work)
        Level 1: Meta-tools (tool factories)
        Level 2+: Child tools (specialized capabilities)

    Each level uses the same physics:
        - PHI_INV controls dynamics
        - PHI stays liminal (superposition)
        - Collapse at unity extracts work
    """

    def __init__(self):
        # Level 0: Main dynamics engine
        if DYNAMICS_AVAILABLE:
            self.dynamics = QuasiCrystalDynamicsEngine(n_oscillators=60)
        else:
            self.dynamics = None

        # Level 1: Meta-tool factory
        self.meta_factory = MetaToolFactory()

        # Statistics
        self.total_collapses = 0
        self.total_work_extracted = 0.0
        self.generation_tree: Dict[str, List[str]] = {}  # parent -> children

    def run_generation_cycle(self, n_collapses: int = 3) -> Dict[str, Any]:
        """
        Run a full generation cycle:
        1. Pump dynamics engine to collapse
        2. Use work to create meta-tools
        3. Feed remaining work to meta-tools to create children

        Returns generation results.
        """
        print(f"\n{'='*60}")
        print("META TOOL GENERATOR")
        print(f"{'='*60}")
        print("Hierarchy: Dynamics → Meta-Tools → Child Tools")
        print("Physics: Liminal PHI at ALL levels")
        print()

        results = {
            'collapses': [],
            'meta_tools_created': [],
            'child_tools_created': [],
            'total_work': 0.0
        }

        if not DYNAMICS_AVAILABLE or not self.dynamics:
            print("ERROR: Dynamics engine not available")
            return results

        # Run collapse cycles
        for cycle in range(n_collapses):
            print(f"\n--- Generation Cycle {cycle + 1}/{n_collapses} ---")

            # Pump to collapse
            initial_collapse_count = self.dynamics.liminal_phi.collapse_count
            initial_work = self.dynamics.total_work_extracted

            steps = 0
            while steps < 200:
                self.dynamics.evolve_step()
                steps += 1

                if self.dynamics.liminal_phi.collapse_count > initial_collapse_count:
                    break

            # Extract work from collapse
            work = self.dynamics.total_work_extracted - initial_work

            if work <= 0:
                print(f"  No collapse in {steps} steps")
                continue

            self.total_collapses += 1
            self.total_work_extracted += work
            results['total_work'] += work

            print(f"  Collapse at step {steps}")
            print(f"  Work extracted: {work:.4f}")

            # Distribute work: 60% to new meta-tool, 40% to existing meta-tools
            meta_work = work * 0.6
            child_work = work * 0.4

            # Create meta-tool
            meta_types = list(MetaToolType)
            meta_type = meta_types[cycle % len(meta_types)]

            meta_tool = self.meta_factory.produce_meta_tool(meta_type, meta_work)

            if meta_tool:
                print(f"  Created meta-tool: {meta_tool.name} ({meta_tool.meta_type.value})")
                results['meta_tools_created'].append(meta_tool.get_stats())
                self.generation_tree[meta_tool.tool_id] = []

            # Feed work to existing meta-tools to produce children
            if self.meta_factory.meta_tools and child_work > 0:
                work_per_meta = child_work / len(self.meta_factory.meta_tools)

                for mt in self.meta_factory.meta_tools:
                    mt.feed_work(work_per_meta)

                    # Try to produce child
                    if mt.can_produce():
                        child = mt.produce_child()
                        if child:
                            print(f"    → {mt.name} produced: {child.name}")
                            results['child_tools_created'].append({
                                'tool_id': child.tool_id,
                                'name': child.name,
                                'type': child.tool_type.value,
                                'parent': mt.tool_id,
                                'generation': child.generation,
                                'work': child.work_invested
                            })

                            if mt.tool_id in self.generation_tree:
                                self.generation_tree[mt.tool_id].append(child.tool_id)

            results['collapses'].append({
                'cycle': cycle + 1,
                'steps': steps,
                'work': work,
                'meta_tool': meta_tool.tool_id if meta_tool else None
            })

        # Final summary
        print(f"\n{'='*60}")
        print("GENERATION COMPLETE")
        print(f"{'='*60}")
        print(f"Total collapses: {self.total_collapses}")
        print(f"Total work extracted: {self.total_work_extracted:.4f}")
        print(f"Meta-tools created: {len(results['meta_tools_created'])}")
        print(f"Child tools created: {len(results['child_tools_created'])}")

        print(f"\nGeneration Tree:")
        for meta_id, children in self.generation_tree.items():
            meta = next((m for m in self.meta_factory.meta_tools if m.tool_id == meta_id), None)
            if meta:
                print(f"  {meta.name} ({meta.meta_type.value})")
                for child_id in children:
                    child = next((c for m in self.meta_factory.meta_tools
                                 for c in m.children if c.tool_id == child_id), None)
                    if child:
                        print(f"    └─ {child.name} ({child.tool_type.value})")

        return results

    def cascade_generation(self, depth: int = 3, work_seed: float = 5.0) -> Dict[str, Any]:
        """
        Cascade generation: tools making tools making tools.

        Each level uses PHI_INV portion of parent's work.
        Demonstrates recursive tool generation.
        """
        print(f"\n{'='*60}")
        print("CASCADE GENERATION")
        print(f"{'='*60}")
        print(f"Depth: {depth} levels")
        print(f"Work seed: {work_seed}")
        print(f"Cascade factor: PHI_INV = {PHI_INV:.4f}")
        print()

        all_tools = []
        work_at_level = [work_seed]

        for level in range(depth):
            print(f"\n--- Level {level + 1} ---")
            print(f"  Work available: {work_at_level[level]:.4f}")

            tools_at_level = []
            work_for_next = 0.0

            if level == 0:
                # Level 0: Create meta-tools
                n_meta = int(work_at_level[level] / (MU_3 * PHI_INV))
                work_per_meta = work_at_level[level] / max(1, n_meta)

                for i in range(max(1, n_meta)):
                    meta_type = list(MetaToolType)[i % len(MetaToolType)]
                    meta = self.meta_factory.produce_meta_tool(meta_type, work_per_meta)
                    if meta:
                        tools_at_level.append(meta)
                        work_for_next += work_per_meta * PHI_INV
                        print(f"  Created: {meta.name}")
            else:
                # Level 1+: Feed work to meta-tools, produce children
                meta_tools = self.meta_factory.meta_tools
                if meta_tools:
                    work_per = work_at_level[level] / len(meta_tools)

                    for mt in meta_tools:
                        # Pump until can produce
                        remaining = work_per
                        while remaining > 0.1 and not mt.can_produce():
                            mt.feed_work(0.1)
                            remaining -= 0.1

                        if mt.can_produce():
                            child = mt.produce_child()
                            if child:
                                tools_at_level.append(child)
                                work_for_next += child.work_invested * PHI_INV
                                print(f"  {mt.name} → {child.name}")

            all_tools.extend(tools_at_level)
            print(f"  Tools created: {len(tools_at_level)}")

            if work_for_next > 0.1:
                work_at_level.append(work_for_next)
            else:
                print(f"  Work depleted at level {level + 1}")
                break

        print(f"\n{'='*60}")
        print("CASCADE COMPLETE")
        print(f"{'='*60}")
        print(f"Levels reached: {len(work_at_level)}")
        print(f"Total tools created: {len(all_tools)}")

        # Show hierarchy
        print(f"\nHierarchy:")
        for i, work in enumerate(work_at_level):
            indent = "  " * i
            n_tools = sum(1 for t in all_tools
                         if (hasattr(t, 'generation') and t.generation == i + 1) or
                            (hasattr(t, 'meta_type') and i == 0))
            print(f"{indent}Level {i}: {n_tools} tools ({work:.4f} work)")

        return {
            'depth_reached': len(work_at_level),
            'tools_created': len(all_tools),
            'work_by_level': work_at_level
        }


# =============================================================================
# DEMONSTRATION
# =============================================================================

def demonstrate_meta_tool_generator():
    """Demonstrate the meta tool generator"""

    print("\n" + "="*70)
    print("META TOOL GENERATOR DEMONSTRATION")
    print("="*70)
    print(f"""
Tools that develop tools using liminal PHI dynamics.

Architecture:
  Level 0: QuasiCrystalDynamicsEngine
           PHI_INV = {PHI_INV:.4f} (physical)
           PHI = {PHI:.4f} (liminal/superposition)
           Collapse at unity → extract work

  Level 1: Meta-Tools (tool factories)
           Same physics, scaled down
           Work from L0 → create meta-tools

  Level 2+: Child Tools (specialized)
           Work from meta-tool collapses
           Recursive generation possible

Key: PHI stays liminal at ALL levels.
     No PHI flip - instant collapse, work extraction, reset.
""")

    # Create generator
    generator = MetaToolGenerator()

    # Run generation cycles
    print("\n" + "-"*70)
    print("PHASE 1: Generation Cycles")
    print("-"*70)
    results = generator.run_generation_cycle(n_collapses=5)

    # Show cascade generation
    print("\n" + "-"*70)
    print("PHASE 2: Cascade Generation")
    print("-"*70)
    cascade_results = generator.cascade_generation(depth=4, work_seed=3.0)

    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE")
    print("="*70)

    return {
        'generation': results,
        'cascade': cascade_results
    }


if __name__ == '__main__':
    demonstrate_meta_tool_generator()
