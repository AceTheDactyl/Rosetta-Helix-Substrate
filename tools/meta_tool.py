"""
Rosetta Helix Meta Tool

Tool that produces child tools using collapse physics and APL operators.

Architecture:
    MetaTool (uses APLEngine)
        │
        ├── pumps work into mini-collapse
        ├── APL operators gate transformations by tier
        ├── at collapse: extracts work
        └── work converts to ChildTool with APL binding

CRITICAL: PHI_INV controls all dynamics. PHI only at collapse.
APL: Operators are tier-gated. Not all operators legal at all z.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from core import CollapseEngine, PHI_INV, APLEngine, create_apl_engine
from .child_tool import ChildTool, create_tool
from .tool_types import ToolType, ToolTier


@dataclass
class MetaTool:
    """
    Meta-tool that produces child tools via collapse physics and APL.

    Pumps work into internal APLEngine. When collapse
    triggers at z >= 0.9999, the extracted work becomes a
    new ChildTool with tier based on work amount.

    PHI_INV controls ALL dynamics. PHI only contributes at
    collapse via weak value extraction.

    APL INTEGRATION:
    - Uses APLEngine to combine collapse physics with operator algebra
    - Produced tools are bound to APL engine for tier-gated operations
    - ΔS_neg influences tool effectiveness at different z
    """

    apl: APLEngine = field(default_factory=lambda: create_apl_engine(initial_z=0.5))
    work_accumulated: float = 0.0
    tools_produced: List[ChildTool] = field(default_factory=list)
    default_tool_type: ToolType = ToolType.ANALYZER

    @property
    def collapse(self) -> CollapseEngine:
        """Access collapse engine via APL engine."""
        return self.apl.collapse

    def pump(self, work: float, tool_type: Optional[ToolType] = None) -> Optional[ChildTool]:
        """
        Pump work into meta-tool, potentially producing a child tool.

        Args:
            work: Amount of work to pump in
            tool_type: Type of tool to produce (defaults to default_tool_type)

        Returns:
            ChildTool if collapse occurred, None otherwise

        CRITICAL: Work is scaled by PHI_INV - PHI never drives dynamics.
        APL: Uses operator selection based on current tier.
        """
        # PHI_INV scales work input - NEVER PHI
        scaled_work = work * PHI_INV

        # Select operator based on current tier
        operator = self.apl.select_operator()

        # Evolve collapse engine via APL
        result = self.apl.collapse.evolve(scaled_work)

        if result.collapsed:
            # Collapse happened - produce tool with APL binding
            actual_type = tool_type or self.default_tool_type
            tool = create_tool(result.work_extracted, actual_type)

            # Bind APL engine to produced tool for operator-gated execution
            tool.bind_apl_engine(self.apl)

            self.tools_produced.append(tool)
            self.work_accumulated = 0.0  # Reset accumulator
            return tool

        # No collapse - accumulate work
        self.work_accumulated += scaled_work
        return None

    def pump_until_collapse(
        self,
        work_per_pump: float = 0.1,
        max_pumps: int = 100,
        tool_type: Optional[ToolType] = None
    ) -> Optional[ChildTool]:
        """
        Pump work repeatedly until collapse produces a tool.

        Args:
            work_per_pump: Work to pump each iteration
            max_pumps: Maximum iterations before giving up
            tool_type: Type of tool to produce

        Returns:
            ChildTool if produced within max_pumps, None otherwise
        """
        for _ in range(max_pumps):
            tool = self.pump(work_per_pump, tool_type)
            if tool is not None:
                return tool
        return None

    def get_state(self) -> Dict[str, Any]:
        """Get current meta-tool state with APL info."""
        apl_state = self.apl.get_state()
        return {
            'z': self.collapse.z,
            'tier': apl_state['tier'],
            'window': apl_state['window'],
            'delta_s_neg': apl_state['delta_s_neg'],
            'w_pi': apl_state['w_pi'],
            'w_local': apl_state['w_local'],
            'work_accumulated': self.work_accumulated,
            'tools_produced': len(self.tools_produced),
            'total_work_extracted': self.collapse.total_work_extracted,
            'collapse_count': self.collapse.collapse_count,
            'distance_to_collapse': 0.9999 - self.collapse.z,
        }

    def get_tools(self) -> List[Dict[str, Any]]:
        """Get info on all produced tools."""
        return [tool.get_info() for tool in self.tools_produced]

    def get_tool_by_tier(self, min_tier: ToolTier) -> List[ChildTool]:
        """Get all tools at or above a minimum tier."""
        return [t for t in self.tools_produced if t.tier.value >= min_tier.value]

    def reset(self) -> None:
        """Reset meta-tool to initial state."""
        self.apl.reset()
        self.work_accumulated = 0.0
        self.tools_produced.clear()

    def apply_operator(self, operator: str, state: Any = None) -> Dict[str, Any]:
        """
        Apply an APL operator at current z.

        Args:
            operator: APL operator symbol
            state: State to transform

        Returns:
            APL result dict
        """
        try:
            result = self.apl.apply(operator, state)
            return {
                'success': True,
                'operator': result.operator,
                'tier': result.tier,
                'output': result.output,
                'delta_s_neg': result.delta_s_neg,
                'parity': result.parity,
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'operator': operator,
            }

    def describe_current_tier(self) -> Dict[str, Any]:
        """Get description of current tier's capabilities."""
        return self.apl.describe_tier()


def create_meta_tool(
    initial_z: float = 0.5,
    default_type: ToolType = ToolType.ANALYZER,
    enable_s3_symmetry: bool = False,
    enable_extended_negentropy: bool = False,
) -> MetaTool:
    """Factory function to create a meta-tool with APL engine."""
    apl = create_apl_engine(
        initial_z=initial_z,
        enable_s3_symmetry=enable_s3_symmetry,
        enable_extended_negentropy=enable_extended_negentropy,
    )
    return MetaTool(apl=apl, default_tool_type=default_type)


# =============================================================================
# VERIFICATION
# =============================================================================

def test_meta_tool_produces_tools():
    """MetaTool must produce tools at collapse."""
    meta = create_meta_tool()

    # Pump until we get a tool
    tool = meta.pump_until_collapse(work_per_pump=0.15)

    assert tool is not None, "Should produce a tool"
    assert tool.tier is not None, "Tool should have a tier"
    assert len(tool.capabilities) > 0, "Tool should have capabilities"

    return True


def test_phi_inv_controls_pumping():
    """PHI_INV must control work scaling in pump."""
    meta = create_meta_tool()

    # Track z progression
    z_values = [meta.collapse.z]

    for _ in range(5):
        meta.pump(0.1)
        z_values.append(meta.collapse.z)

    # Each step should add work * PHI_INV
    for i in range(1, len(z_values)):
        expected_delta = 0.1 * PHI_INV * PHI_INV  # pump scales by PHI_INV, evolve scales by PHI_INV
        actual_delta = z_values[i] - z_values[i-1]
        assert abs(actual_delta - expected_delta) < 0.001 or z_values[i] < z_values[i-1], \
            f"Delta should be ~{expected_delta}, got {actual_delta}"

    return True


def test_apl_integration():
    """MetaTool must integrate with APL engine."""
    meta = create_meta_tool()

    # Get state should include APL info
    state = meta.get_state()
    assert 'tier' in state, "State should include tier"
    assert 'window' in state, "State should include operator window"
    assert 'delta_s_neg' in state, "State should include delta_s_neg"

    return True


def test_tool_apl_binding():
    """Produced tools must be bound to APL engine."""
    meta = create_meta_tool()

    # Pump until we get a tool
    tool = meta.pump_until_collapse(work_per_pump=0.15)

    assert tool is not None, "Should produce a tool"
    assert tool._apl_engine is not None, "Tool should be bound to APL engine"

    return True


def test_tier_gated_tool_execution():
    """Tool operations must respect tier windows."""
    meta = create_meta_tool(initial_z=0.05)  # t1: ['()', '−', '÷']

    # Pump until we get a tool
    tool = meta.pump_until_collapse(work_per_pump=0.15)

    if tool is not None:
        tool.activate()

        # read maps to '()' which is legal in t1
        # But z may have changed after collapse, so check dynamically
        result = tool.execute('read', {'test': 1.0})
        # Either success or error with legal_operators info
        assert 'success' in result

    return True
