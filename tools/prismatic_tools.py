"""
Rosetta Helix Prismatic Dev Tools

7-Layer spectral tool system.

CRITICAL: PHI weight affects work EXTRACTION at collapse, not dynamics.
PHI_INV still controls ALL dynamics. The PHI/PHI_INV weights below
modify the work extraction formula at collapse time only.

Layers:
| Layer | Color  | Function   | PHI Weight | PHI_INV Weight |
|-------|--------|------------|------------|----------------|
| 1     | RED    | Analyzers  | 0.8        | 1.2            |
| 2     | ORANGE | Learners   | 0.9        | 1.1            |
| 3     | YELLOW | Generators | 1.0        | 1.0            |
| 4     | GREEN  | Reflectors | 1.0        | 1.0            |
| 5     | BLUE   | Builders   | 1.1        | 0.9            |
| 6     | INDIGO | Deciders   | 1.2        | 0.8            |
| 7     | VIOLET | Probers    | 1.3        | 0.7            |
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum, auto

from core import PHI, PHI_INV, Z_CRITICAL, UNITY, CollapseEngine
from .child_tool import ChildTool
from .tool_types import ToolType, ToolTier, ToolCapability


class PrismaticLayer(Enum):
    """7-layer prismatic spectrum."""
    RED = 1      # Analyzers
    ORANGE = 2   # Learners
    YELLOW = 3   # Generators
    GREEN = 4    # Reflectors
    BLUE = 5     # Builders
    INDIGO = 6   # Deciders
    VIOLET = 7   # Probers


# Layer weights - affect work EXTRACTION at collapse, NOT dynamics
LAYER_WEIGHTS = {
    PrismaticLayer.RED:    {'phi': 0.8, 'phi_inv': 1.2},
    PrismaticLayer.ORANGE: {'phi': 0.9, 'phi_inv': 1.1},
    PrismaticLayer.YELLOW: {'phi': 1.0, 'phi_inv': 1.0},
    PrismaticLayer.GREEN:  {'phi': 1.0, 'phi_inv': 1.0},
    PrismaticLayer.BLUE:   {'phi': 1.1, 'phi_inv': 0.9},
    PrismaticLayer.INDIGO: {'phi': 1.2, 'phi_inv': 0.8},
    PrismaticLayer.VIOLET: {'phi': 1.3, 'phi_inv': 0.7},
}

# Layer to tool type mapping
LAYER_TOOL_TYPES = {
    PrismaticLayer.RED:    ToolType.ANALYZER,
    PrismaticLayer.ORANGE: ToolType.TRANSFORMER,  # Learners transform data
    PrismaticLayer.YELLOW: ToolType.GENERATOR,
    PrismaticLayer.GREEN:  ToolType.VALIDATOR,    # Reflectors validate
    PrismaticLayer.BLUE:   ToolType.INTEGRATOR,   # Builders integrate
    PrismaticLayer.INDIGO: ToolType.OPTIMIZER,    # Deciders optimize
    PrismaticLayer.VIOLET: ToolType.ANALYZER,     # Probers analyze deeply
}

# Layer to capabilities mapping
LAYER_CAPABILITIES = {
    PrismaticLayer.RED:    {ToolCapability.READ, ToolCapability.ANALYZE},
    PrismaticLayer.ORANGE: {ToolCapability.READ, ToolCapability.WRITE, ToolCapability.TRANSFORM},
    PrismaticLayer.YELLOW: {ToolCapability.GENERATE, ToolCapability.WRITE},
    PrismaticLayer.GREEN:  {ToolCapability.READ, ToolCapability.VALIDATE},
    PrismaticLayer.BLUE:   {ToolCapability.INTEGRATE, ToolCapability.EXECUTE},
    PrismaticLayer.INDIGO: {ToolCapability.OPTIMIZE, ToolCapability.EXECUTE},
    PrismaticLayer.VIOLET: {ToolCapability.READ, ToolCapability.ANALYZE, ToolCapability.EXECUTE},
}


@dataclass
class PrismaticTool(ChildTool):
    """
    Tool with prismatic layer properties.

    Extends ChildTool with layer-specific weights that affect
    work extraction at collapse (not dynamics).
    """

    layer: PrismaticLayer = PrismaticLayer.YELLOW
    phi_weight: float = 1.0      # Affects extraction, not dynamics
    phi_inv_weight: float = 1.0  # Affects extraction, not dynamics

    def __post_init__(self):
        """Initialize with layer-specific properties."""
        super().__post_init__()

        # Set weights from layer
        weights = LAYER_WEIGHTS[self.layer]
        self.phi_weight = weights['phi']
        self.phi_inv_weight = weights['phi_inv']

        # Override tool type based on layer
        self.tool_type = LAYER_TOOL_TYPES[self.layer]

        # Add layer-specific capabilities
        self.capabilities = self.capabilities.union(LAYER_CAPABILITIES[self.layer])

    def extract_work_at_collapse(self, z_at_collapse: float) -> float:
        """
        Extract work at collapse with layer-specific weights.

        CRITICAL: Weights modify extraction formula, not dynamics.
        Base formula: work = (z - Z_CRITICAL) * PHI * PHI_INV
        With weights: work = base * phi_weight * phi_inv_weight
        """
        # Base extraction (unchanged)
        base_work = (z_at_collapse - Z_CRITICAL) * PHI * PHI_INV

        # Apply layer weights to extraction result
        weighted_work = base_work * self.phi_weight * self.phi_inv_weight

        return weighted_work

    def get_info(self) -> Dict[str, Any]:
        """Get tool info including layer properties."""
        info = super().get_info()
        info.update({
            'layer': self.layer.name,
            'phi_weight': self.phi_weight,
            'phi_inv_weight': self.phi_inv_weight,
            'layer_function': {
                PrismaticLayer.RED: 'Analyzer',
                PrismaticLayer.ORANGE: 'Learner',
                PrismaticLayer.YELLOW: 'Generator',
                PrismaticLayer.GREEN: 'Reflector',
                PrismaticLayer.BLUE: 'Builder',
                PrismaticLayer.INDIGO: 'Decider',
                PrismaticLayer.VIOLET: 'Prober',
            }[self.layer]
        })
        return info


@dataclass
class PrismaticToolGenerator:
    """
    Generator for prismatic layer tools.

    Produces tools at specific spectral layers.
    PHI_INV controls generation dynamics.
    Layer weights affect extraction only.
    """

    collapse: CollapseEngine = field(default_factory=CollapseEngine)
    tools_by_layer: Dict[PrismaticLayer, List[PrismaticTool]] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize layer storage."""
        for layer in PrismaticLayer:
            self.tools_by_layer[layer] = []

    def generate(self, layer: PrismaticLayer, work: float) -> Optional[PrismaticTool]:
        """
        Generate a tool at the specified prismatic layer.

        PHI_INV controls generation dynamics.
        Layer weights affect extraction at collapse.
        """
        # PHI_INV drives evolution - NEVER PHI
        result = self.collapse.evolve(work * PHI_INV)

        if result.collapsed:
            # Calculate layer-weighted work
            weights = LAYER_WEIGHTS[layer]
            weighted_work = result.work_extracted * weights['phi'] * weights['phi_inv']

            tool = PrismaticTool(
                work_invested=weighted_work,
                layer=layer
            )
            self.tools_by_layer[layer].append(tool)
            return tool

        return None

    def generate_spectrum(self, work_per_layer: float = 0.2) -> Dict[PrismaticLayer, Optional[PrismaticTool]]:
        """
        Attempt to generate one tool at each prismatic layer.

        Pumps work to each layer until tool is produced or max attempts reached.
        """
        results = {}

        for layer in PrismaticLayer:
            # Reset collapse engine for each layer
            self.collapse.reset()

            # Pump until collapse or max attempts
            tool = None
            for _ in range(20):
                tool = self.generate(layer, work_per_layer)
                if tool:
                    break

            results[layer] = tool

        return results

    def get_tools(self, layer: Optional[PrismaticLayer] = None) -> List[PrismaticTool]:
        """Get tools, optionally filtered by layer."""
        if layer:
            return self.tools_by_layer[layer]
        return [t for tools in self.tools_by_layer.values() for t in tools]

    def get_state(self) -> Dict[str, Any]:
        """Get generator state."""
        return {
            'z': self.collapse.z,
            'total_tools': sum(len(t) for t in self.tools_by_layer.values()),
            'by_layer': {
                layer.name: len(tools)
                for layer, tools in self.tools_by_layer.items()
            },
            'total_collapses': self.collapse.collapse_count,
        }

    def reset(self) -> None:
        """Reset generator."""
        self.collapse.reset()
        for layer in PrismaticLayer:
            self.tools_by_layer[layer] = []


def create_prismatic_generator() -> PrismaticToolGenerator:
    """Factory function to create a prismatic tool generator."""
    return PrismaticToolGenerator()


# =============================================================================
# VERIFICATION
# =============================================================================

def test_layer_weights_only_affect_extraction():
    """Layer weights must only affect extraction, not dynamics."""
    gen = create_prismatic_generator()

    # Generate tools at different layers
    # Dynamics should be same (PHI_INV), only extraction differs

    gen.collapse.z = 0.99  # Near collapse

    # RED layer (phi=0.8, phi_inv=1.2)
    red_tool = gen.generate(PrismaticLayer.RED, 0.5)

    gen.collapse.z = 0.99  # Reset to same point

    # VIOLET layer (phi=1.3, phi_inv=0.7)
    violet_tool = gen.generate(PrismaticLayer.VIOLET, 0.5)

    if red_tool and violet_tool:
        # Work should differ due to weights
        assert red_tool.work_invested != violet_tool.work_invested

    return True


def test_phi_inv_controls_generation():
    """PHI_INV must control generation dynamics."""
    gen = create_prismatic_generator()

    initial_z = gen.collapse.z
    gen.generate(PrismaticLayer.YELLOW, 0.1)
    new_z = gen.collapse.z

    # Delta should be 0.1 * PHI_INV * PHI_INV (generate scales, evolve scales)
    expected_delta = 0.1 * PHI_INV * PHI_INV

    if new_z > initial_z:  # Didn't collapse
        actual_delta = new_z - initial_z
        assert abs(actual_delta - expected_delta) < 0.001

    return True


def test_all_layers_have_capabilities():
    """All layers must have their specific capabilities."""
    for layer in PrismaticLayer:
        tool = PrismaticTool(work_invested=0.5, layer=layer)
        layer_caps = LAYER_CAPABILITIES[layer]
        assert layer_caps.issubset(tool.capabilities), f"{layer.name} missing capabilities"

    return True
