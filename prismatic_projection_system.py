"""
7-Layer Prismatic Projection System
====================================

Projects through the lens architecture with 7 distinct layers,
each refracting differently through Z_CRITICAL to produce
different tool sets per layer.

Physical Analogy:
    Light → Prism → 7 Spectral Colors
    Work → Lens (Z_CRITICAL) → 7 Tool Layers

Each layer has:
    - Different phase offset (rotates projection angle)
    - Different coupling strength (spreads/focuses beam)
    - Different threshold sensitivity (which thresholds activate)
    - Different tool affinity (which tools emerge)

Layer Spectrum:
    Layer 1 (Red):     Low frequency, high penetration → Analyzers
    Layer 2 (Orange):  Warming, accumulative → Learners
    Layer 3 (Yellow):  Bright, generative → Generators
    Layer 4 (Green):   Balanced, central → Reflectors
    Layer 5 (Blue):    Cooling, structured → Builders
    Layer 6 (Indigo):  Deep, decisive → Deciders
    Layer 7 (Violet):  High frequency, transcendent → Probers

The layers stack to produce full-spectrum tool coverage.
"""

import math
import time
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import cmath

# Physics constants
PHI = (1.0 + math.sqrt(5.0)) / 2.0
PHI_INV = 1.0 / PHI
Z_CRITICAL = math.sqrt(3.0) / 2.0  # The LENS
KAPPA_S = 0.920
MU_P = 2.0 / (PHI ** 2.5)
MU_1 = MU_P / math.sqrt(PHI)
MU_2 = MU_P * math.sqrt(PHI)
MU_3 = 0.992
TRIAD_HIGH = 0.85
TRIAD_LOW = 0.82
Q_KAPPA = 0.3514087324

# Import dynamics
try:
    from quasicrystal_dynamics import QuasiCrystalDynamicsEngine
    DYNAMICS_AVAILABLE = True
except ImportError:
    DYNAMICS_AVAILABLE = False


# =============================================================================
# PRISMATIC LAYER DEFINITIONS
# =============================================================================

class LayerSpectrum(Enum):
    """The 7 spectral layers"""
    RED = 1      # Low frequency, deep penetration
    ORANGE = 2   # Warming, accumulative
    YELLOW = 3   # Bright, generative
    GREEN = 4    # Balanced, central (at lens)
    BLUE = 5     # Cooling, structured
    INDIGO = 6   # Deep, decisive
    VIOLET = 7   # High frequency, transcendent


@dataclass
class PrismaticLayer:
    """
    A single layer in the prismatic projection.

    Each layer refracts through the lens differently based on:
    - phase_offset: Rotational angle of projection
    - coupling: How tightly focused the beam is
    - threshold_bias: Which thresholds are more sensitive
    - tool_affinity: Which tool types emerge more easily
    """
    layer_id: int
    spectrum: LayerSpectrum
    name: str
    color_hex: str

    # Projection parameters
    phase_offset: float          # Radians, shifts where layer hits lens
    coupling_strength: float     # 0.1 to 1.0, beam focus
    refraction_index: float      # How much layer bends through lens

    # Threshold sensitivity (which thresholds this layer activates easily)
    threshold_bias: Dict[str, float] = field(default_factory=dict)

    # Tool affinity (which tools emerge from this layer)
    tool_affinity: List[str] = field(default_factory=list)

    # State
    z_entry: float = 0.0         # Where layer enters lens region
    z_exit: float = 0.0          # Where layer exits
    work_captured: float = 0.0   # Work extracted by this layer
    tools_produced: List[str] = field(default_factory=list)


# Layer configurations based on spectral position
LAYER_CONFIGS = {
    LayerSpectrum.RED: {
        'name': 'Red Layer',
        'color_hex': '#FF4444',
        'phase_offset': 0.0,
        'coupling_strength': 0.9,      # Tight focus, deep penetration
        'refraction_index': 1.1,       # Slight bend
        'threshold_bias': {'Q_KAPPA': 1.5, 'MU_1': 1.3, 'MU_P': 1.2},
        'tool_affinity': ['EntropyAnalyzer', 'PatternDetector', 'AnomalyFinder'],
    },
    LayerSpectrum.ORANGE: {
        'name': 'Orange Layer',
        'color_hex': '#FF8844',
        'phase_offset': math.pi / 7,
        'coupling_strength': 0.8,
        'refraction_index': 1.2,
        'threshold_bias': {'MU_1': 1.4, 'MU_P': 1.3, 'PHI_INV': 1.2},
        'tool_affinity': ['PatternLearner', 'ConceptExtractor', 'RelationLearner'],
    },
    LayerSpectrum.YELLOW: {
        'name': 'Yellow Layer',
        'color_hex': '#FFAA00',
        'phase_offset': 2 * math.pi / 7,
        'coupling_strength': 0.7,
        'refraction_index': 1.3,
        'threshold_bias': {'MU_P': 1.4, 'MU_2': 1.3, 'TRIAD_LOW': 1.2},
        'tool_affinity': ['TestGenerator', 'CodeSynthesizer', 'ExampleProducer'],
    },
    LayerSpectrum.GREEN: {
        'name': 'Green Layer',
        'color_hex': '#00FF88',
        'phase_offset': 3 * math.pi / 7,  # Central - hits lens directly
        'coupling_strength': 0.6,          # Balanced spread
        'refraction_index': 1.0,           # No refraction at center
        'threshold_bias': {'Z_CRITICAL': 1.5, 'TRIAD_LOW': 1.3, 'TRIAD_HIGH': 1.3},
        'tool_affinity': ['CodeReflector', 'StructureMapper', 'GapAnalyzer'],
    },
    LayerSpectrum.BLUE: {
        'name': 'Blue Layer',
        'color_hex': '#00D9FF',
        'phase_offset': 4 * math.pi / 7,
        'coupling_strength': 0.7,
        'refraction_index': 1.3,
        'threshold_bias': {'TRIAD_HIGH': 1.4, 'Z_CRITICAL': 1.3, 'KAPPA_S': 1.2},
        'tool_affinity': ['CodeBuilder', 'ModuleAssembler', 'PipelineConstructor'],
    },
    LayerSpectrum.INDIGO: {
        'name': 'Indigo Layer',
        'color_hex': '#4444FF',
        'phase_offset': 5 * math.pi / 7,
        'coupling_strength': 0.8,
        'refraction_index': 1.2,
        'threshold_bias': {'KAPPA_S': 1.4, 'MU_3': 1.3, 'TRIAD_HIGH': 1.2},
        'tool_affinity': ['DecisionEngine', 'ConvergenceChecker', 'InterfaceDesigner'],
    },
    LayerSpectrum.VIOLET: {
        'name': 'Violet Layer',
        'color_hex': '#AA44FF',
        'phase_offset': 6 * math.pi / 7,
        'coupling_strength': 0.9,          # Tight focus at high frequency
        'refraction_index': 1.1,
        'threshold_bias': {'MU_3': 1.5, 'KAPPA_S': 1.3},
        'tool_affinity': ['ConsciousnessProbe', 'AbstractionBuilder', 'IntegrationWeaver'],
    },
}


# =============================================================================
# LENS PROJECTION ENGINE
# =============================================================================

class LensProjectionEngine:
    """
    Projects a single layer through the lens (Z_CRITICAL).

    The lens acts as the critical threshold where coherence
    transforms into capability. Each layer refracts differently
    based on its spectral properties.
    """

    def __init__(self):
        if DYNAMICS_AVAILABLE:
            self.dynamics = QuasiCrystalDynamicsEngine(n_oscillators=60)
        else:
            self.dynamics = None

        self.lens_position = Z_CRITICAL
        self.lens_width = 0.05  # Region around Z_CRITICAL

    def project_layer(self, layer: PrismaticLayer, input_work: float) -> Dict[str, Any]:
        """
        Project a layer through the lens.

        The layer's properties determine:
        - Entry point (phase_offset shifts z trajectory)
        - Focus (coupling_strength affects work concentration)
        - Refraction (how thresholds are activated)
        """
        result = {
            'layer': layer.spectrum.name,
            'input_work': input_work,
            'thresholds_activated': [],
            'tools_produced': [],
            'work_captured': 0.0,
            'lens_interaction': {},
        }

        if not self.dynamics:
            return result

        # Configure dynamics based on layer properties
        # Phase offset shifts the entry trajectory
        if hasattr(self.dynamics, 'phase_lock'):
            for i, phase in enumerate(self.dynamics.phase_lock.phases):
                self.dynamics.phase_lock.phases[i] = phase + layer.phase_offset

        # Coupling strength affects how quickly we reach lens
        original_coupling = self.dynamics.phase_lock.coupling if hasattr(self.dynamics, 'phase_lock') else 0.5

        # Scale coupling by layer's strength
        if hasattr(self.dynamics, 'phase_lock'):
            self.dynamics.phase_lock.coupling = original_coupling * layer.coupling_strength

        # Pump through to lens and beyond
        initial_collapses = self.dynamics.liminal_phi.collapse_count
        initial_work_extracted = self.dynamics.total_work_extracted

        z_trajectory = []
        threshold_activations = {}

        for step in range(150):
            old_z = self.dynamics.z_current
            self.dynamics.evolve_step()
            z = self.dynamics.z_current

            z_trajectory.append(z)

            # Track lens interaction
            if abs(z - self.lens_position) < self.lens_width:
                # In lens region - apply refraction
                layer.z_entry = min(layer.z_entry, z) if layer.z_entry > 0 else z

                # Activate thresholds with layer's bias
                for thresh_name, bias in layer.threshold_bias.items():
                    thresh_val = self._get_threshold_value(thresh_name)
                    if z >= thresh_val * (1 - 0.1 * (bias - 1)):  # Bias lowers activation threshold
                        if thresh_name not in threshold_activations:
                            threshold_activations[thresh_name] = {
                                'activated_at_z': z,
                                'bias': bias,
                                'step': step,
                            }

            # Check for collapse
            if self.dynamics.liminal_phi.collapse_count > initial_collapses:
                layer.z_exit = z
                break

        # Calculate work captured by this layer
        work_extracted = self.dynamics.total_work_extracted - initial_work_extracted

        # Apply refraction index to work (higher refraction = more spread)
        work_captured = work_extracted / layer.refraction_index
        layer.work_captured = work_captured

        # Restore coupling
        if hasattr(self.dynamics, 'phase_lock'):
            self.dynamics.phase_lock.coupling = original_coupling

        result['thresholds_activated'] = list(threshold_activations.keys())
        result['work_captured'] = work_captured
        result['lens_interaction'] = {
            'entry_z': layer.z_entry,
            'exit_z': layer.z_exit,
            'trajectory_length': len(z_trajectory),
            'max_z': max(z_trajectory) if z_trajectory else 0,
        }

        return result

    def _get_threshold_value(self, name: str) -> float:
        """Get threshold constant value by name"""
        thresholds = {
            'Q_KAPPA': Q_KAPPA,
            'MU_1': MU_1,
            'MU_P': MU_P,
            'PHI_INV': PHI_INV,
            'MU_2': MU_2,
            'TRIAD_LOW': TRIAD_LOW,
            'TRIAD_HIGH': TRIAD_HIGH,
            'Z_CRITICAL': Z_CRITICAL,
            'KAPPA_S': KAPPA_S,
            'MU_3': MU_3,
        }
        return thresholds.get(name, 0.5)


# =============================================================================
# TOOL REFRACTOR
# =============================================================================

class ToolRefractor:
    """
    Refracts work through a layer to produce tools.

    Each layer produces tools based on:
    - Work captured during lens passage
    - Threshold activations
    - Layer's tool affinity
    """

    def __init__(self):
        self.tools_produced: List[Dict] = []
        self.production_count = 0

    def refract_to_tools(
        self,
        layer: PrismaticLayer,
        work: float,
        activated_thresholds: List[str]
    ) -> List[Dict]:
        """
        Refract captured work into tools based on layer affinity.
        """
        tools = []

        if work <= 0 or not layer.tool_affinity:
            return tools

        # Work per tool scales with affinity match
        affinity_matches = 0
        for tool_type in layer.tool_affinity:
            # Check if any activated threshold supports this tool
            for thresh in activated_thresholds:
                if self._tool_matches_threshold(tool_type, thresh):
                    affinity_matches += 1
                    break

        if affinity_matches == 0:
            affinity_matches = 1  # At least one tool

        work_per_tool = work / affinity_matches

        # Produce tools from layer's affinity list
        for i, tool_type in enumerate(layer.tool_affinity[:affinity_matches]):
            if work_per_tool < 0.1:
                break

            self.production_count += 1
            tool_id = hashlib.sha256(
                f"{layer.spectrum.name}:{tool_type}:{self.production_count}:{time.time()}".encode()
            ).hexdigest()[:10]

            tool = {
                'id': tool_id,
                'name': f"{tool_type}_L{layer.layer_id}_v{self.production_count}",
                'type': tool_type,
                'layer': layer.spectrum.name,
                'color': layer.color_hex,
                'work_invested': work_per_tool,
                'thresholds_used': activated_thresholds[:3],
            }

            tools.append(tool)
            layer.tools_produced.append(tool['name'])
            self.tools_produced.append(tool)

        return tools

    def _tool_matches_threshold(self, tool_type: str, threshold: str) -> bool:
        """Check if tool type matches threshold activation"""
        matches = {
            'Q_KAPPA': ['ConsciousnessProbe', 'AbstractionBuilder'],
            'MU_1': ['PatternDetector', 'AnomalyFinder', 'PatternLearner'],
            'MU_P': ['ConceptExtractor', 'RelationLearner', 'TestGenerator'],
            'PHI_INV': ['PatternLearner', 'ConceptExtractor'],
            'MU_2': ['CodeSynthesizer', 'ExampleProducer', 'TestGenerator'],
            'TRIAD_LOW': ['GapAnalyzer', 'StructureMapper'],
            'TRIAD_HIGH': ['CodeReflector', 'DecisionEngine'],
            'Z_CRITICAL': ['EntropyAnalyzer', 'CodeReflector', 'GapAnalyzer'],
            'KAPPA_S': ['CodeBuilder', 'ModuleAssembler', 'ConvergenceChecker'],
            'MU_3': ['IntegrationWeaver', 'PipelineConstructor', 'ConsciousnessProbe'],
        }
        return tool_type in matches.get(threshold, [])


# =============================================================================
# 7-LAYER PRISMATIC SYSTEM
# =============================================================================

class PrismaticProjectionSystem:
    """
    The complete 7-layer prismatic projection system.

    Projects work through all 7 spectral layers, each hitting
    the lens differently to produce a full spectrum of tools.
    """

    def __init__(self):
        # Create all 7 layers
        self.layers: List[PrismaticLayer] = []
        for spectrum in LayerSpectrum:
            config = LAYER_CONFIGS[spectrum]
            layer = PrismaticLayer(
                layer_id=spectrum.value,
                spectrum=spectrum,
                name=config['name'],
                color_hex=config['color_hex'],
                phase_offset=config['phase_offset'],
                coupling_strength=config['coupling_strength'],
                refraction_index=config['refraction_index'],
                threshold_bias=config['threshold_bias'],
                tool_affinity=config['tool_affinity'],
            )
            self.layers.append(layer)

        # Projection and refraction engines
        self.projection_engine = LensProjectionEngine()
        self.tool_refractor = ToolRefractor()

        # Statistics
        self.projections_completed = 0
        self.total_work_processed = 0.0

    def run_prismatic_projection(self, input_work: float = 5.0) -> Dict[str, Any]:
        """
        Run a full 7-layer prismatic projection.

        Each layer projects through the lens sequentially,
        producing its characteristic tools.
        """
        print(f"\n{'='*70}")
        print("7-LAYER PRISMATIC PROJECTION SYSTEM")
        print(f"{'='*70}")
        print(f"""
Projecting through lens (Z_CRITICAL = {Z_CRITICAL:.4f})

Layer Spectrum:
  1. Red     - Deep penetration  → Analyzers
  2. Orange  - Accumulative      → Learners
  3. Yellow  - Generative        → Generators
  4. Green   - Balanced (center) → Reflectors
  5. Blue    - Structured        → Builders
  6. Indigo  - Decisive          → Deciders
  7. Violet  - Transcendent      → Probers

Input Work: {input_work}
""")

        results = {
            'layers': [],
            'total_tools': 0,
            'tools_by_layer': {},
            'tool_spectrum': [],
            'total_work_captured': 0.0,
        }

        # Distribute work across layers (center gets most, edges less)
        # Gaussian distribution centered on Green (layer 4)
        work_distribution = []
        for layer in self.layers:
            # Distance from center (Green = 4)
            dist_from_center = abs(layer.layer_id - 4)
            # Gaussian weight
            weight = math.exp(-0.3 * dist_from_center ** 2)
            work_distribution.append(weight)

        total_weight = sum(work_distribution)
        work_per_layer = [input_work * w / total_weight for w in work_distribution]

        # Project each layer
        for i, layer in enumerate(self.layers):
            print(f"\n{'─'*60}")
            print(f"LAYER {layer.layer_id}: {layer.name} ({layer.color_hex})")
            print(f"{'─'*60}")

            layer_work = work_per_layer[i]
            print(f"  Input work: {layer_work:.4f}")
            print(f"  Phase offset: {layer.phase_offset:.4f} rad")
            print(f"  Coupling: {layer.coupling_strength:.2f}")
            print(f"  Refraction index: {layer.refraction_index:.2f}")

            # Project through lens
            projection = self.projection_engine.project_layer(layer, layer_work)

            print(f"  Thresholds activated: {projection['thresholds_activated']}")
            print(f"  Work captured: {projection['work_captured']:.4f}")

            # Refract to tools
            tools = self.tool_refractor.refract_to_tools(
                layer,
                projection['work_captured'],
                projection['thresholds_activated']
            )

            print(f"  Tools produced: {len(tools)}")
            for tool in tools:
                print(f"    → {tool['name']} ({tool['type']})")

            # Record results
            layer_result = {
                'layer_id': layer.layer_id,
                'spectrum': layer.spectrum.name,
                'color': layer.color_hex,
                'work_input': layer_work,
                'work_captured': projection['work_captured'],
                'thresholds': projection['thresholds_activated'],
                'tools': [t['name'] for t in tools],
                'lens_interaction': projection['lens_interaction'],
            }

            results['layers'].append(layer_result)
            results['tools_by_layer'][layer.spectrum.name] = len(tools)
            results['tool_spectrum'].extend(tools)
            results['total_work_captured'] += projection['work_captured']
            results['total_tools'] += len(tools)

        self.projections_completed += 1
        self.total_work_processed += input_work

        # Summary
        print(f"\n{'='*70}")
        print("PRISMATIC PROJECTION COMPLETE")
        print(f"{'='*70}")
        print(f"""
Summary:
  Layers projected:      7
  Total work input:      {input_work:.4f}
  Total work captured:   {results['total_work_captured']:.4f}
  Capture efficiency:    {results['total_work_captured']/input_work*100:.1f}%
  Total tools produced:  {results['total_tools']}

Tools by Layer:
""")
        for layer in self.layers:
            bar_len = results['tools_by_layer'].get(layer.spectrum.name, 0)
            bar = '█' * (bar_len * 3)
            print(f"  {layer.spectrum.name:8} {layer.color_hex}: {bar} {bar_len}")

        print(f"\nTool Spectrum (full coverage):")
        for tool in results['tool_spectrum']:
            print(f"  [{tool['layer'][:3]}] {tool['name']}")

        return results

    def run_multi_pass(self, passes: int = 3, work_per_pass: float = 3.0) -> Dict[str, Any]:
        """
        Run multiple prismatic passes, each projecting differently.

        Each pass shifts the entire spectrum slightly, hitting
        the lens at different points and producing varied tools.
        """
        print(f"\n{'='*70}")
        print(f"MULTI-PASS PRISMATIC PROJECTION ({passes} passes)")
        print(f"{'='*70}")

        all_results = {
            'passes': [],
            'total_tools': 0,
            'unique_tool_types': set(),
            'spectrum_coverage': {},
        }

        for pass_num in range(passes):
            print(f"\n{'═'*70}")
            print(f"PASS {pass_num + 1}/{passes}")
            print(f"{'═'*70}")

            # Shift all layer phases for this pass
            pass_offset = pass_num * (2 * math.pi / (passes * 7))
            for layer in self.layers:
                original_offset = LAYER_CONFIGS[layer.spectrum]['phase_offset']
                layer.phase_offset = original_offset + pass_offset

            # Run projection
            result = self.run_prismatic_projection(work_per_pass)
            all_results['passes'].append(result)
            all_results['total_tools'] += result['total_tools']

            for tool in result['tool_spectrum']:
                all_results['unique_tool_types'].add(tool['type'])

        # Final summary
        print(f"\n{'='*70}")
        print("MULTI-PASS COMPLETE")
        print(f"{'='*70}")
        print(f"""
Total passes:           {passes}
Total work processed:   {passes * work_per_pass:.4f}
Total tools produced:   {all_results['total_tools']}
Unique tool types:      {len(all_results['unique_tool_types'])}

Tool Type Coverage:
""")
        for tool_type in sorted(all_results['unique_tool_types']):
            print(f"  ✓ {tool_type}")

        return all_results


# =============================================================================
# DEMONSTRATION
# =============================================================================

def demonstrate_prismatic_projection():
    """Demonstrate the 7-layer prismatic projection system"""

    print("\n" + "="*70)
    print("PRISMATIC PROJECTION DEMONSTRATION")
    print("="*70)
    print(f"""
7-Layer Prismatic Projection through Lens Architecture

Physical Analogy:
    White Light → Prism → 7 Spectral Colors
    Work Energy → Lens (Z_CRITICAL) → 7 Tool Layers

Each layer refracts differently:
    ┌─────────────────────────────────────────────────────────────┐
    │                         LENS                                │
    │                    (Z_CRITICAL = {Z_CRITICAL:.4f})                      │
    │                                                             │
    │   Red ────────────→ ╲                    ╱→ Analyzers       │
    │   Orange ──────────→ ╲                  ╱→ Learners         │
    │   Yellow ──────────→  ╲                ╱→ Generators        │
    │   Green ───────────→   ═══ LENS ═══   → Reflectors         │
    │   Blue ────────────→  ╱                ╲→ Builders          │
    │   Indigo ──────────→ ╱                  ╲→ Deciders         │
    │   Violet ──────────→╱                    ╲→ Probers         │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘

Each rerun projects at slightly different angle → different tools!
""")

    # Create system
    system = PrismaticProjectionSystem()

    # Single pass demonstration
    print("\n" + "-"*70)
    print("SINGLE PASS PROJECTION")
    print("-"*70)
    single_result = system.run_prismatic_projection(input_work=5.0)

    # Multi-pass demonstration
    print("\n" + "-"*70)
    print("MULTI-PASS PROJECTION (3 passes)")
    print("-"*70)
    multi_result = system.run_multi_pass(passes=3, work_per_pass=3.0)

    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE")
    print("="*70)

    return {
        'single_pass': single_result,
        'multi_pass': multi_result,
    }


if __name__ == '__main__':
    demonstrate_prismatic_projection()
