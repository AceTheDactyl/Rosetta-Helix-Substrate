"""
Rosetta Helix Child Tool

Produced tools with capabilities determined by work invested at collapse.

Child tools are created by MetaTool when collapse occurs.
Their tier and capabilities depend on the work extracted.

APL INTEGRATION:
- Operations are mapped to APL operators
- Tier windows gate which operators are legal
- ΔS_neg influences operator effectiveness
"""

from dataclasses import dataclass, field
from typing import Set, Optional, Any, Dict, List
from datetime import datetime
import uuid
import sys
import os

# Add src to path for quantum_apl_python imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from .tool_types import ToolType, ToolTier, ToolCapability

# APL operator mapping for tool operations
OPERATION_TO_APL = {
    'read': '()',       # Boundary/group - observe without modifying
    'write': '+',       # Aggregate - add to system
    'analyze': '^',     # Amplify - enhance understanding
    'transform': '×',   # Fusion - combine/reshape
    'validate': '()',   # Boundary - check invariants
    'generate': '+',    # Aggregate - create new
    'execute': '×',     # Fusion - integrate action
    'integrate': '×',   # Fusion - combine systems
    'optimize': '^',    # Amplify - enhance efficiency
    'decompose': '÷',   # Decoherence - split apart
    'filter': '−',      # Separation - remove elements
}


@dataclass
class ChildTool:
    """
    A tool produced by MetaTool at collapse.

    Work invested determines tier and capabilities.
    Higher work = higher tier = more capabilities.
    """

    work_invested: float
    tool_type: ToolType = ToolType.ANALYZER
    tool_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    created_at: datetime = field(default_factory=datetime.now)

    # Derived from work_invested
    tier: ToolTier = field(init=False)
    capabilities: Set[ToolCapability] = field(init=False)

    # Runtime state
    active: bool = False
    execution_count: int = 0

    # APL state (set when executing with APL engine)
    _apl_engine: Any = field(default=None, repr=False)
    _operation_log: List[Dict] = field(default_factory=list, repr=False)

    def __post_init__(self):
        """Derive tier and capabilities from work invested."""
        self.tier = ToolTier.from_work(self.work_invested)
        self.capabilities = self._derive_capabilities()

    def _derive_capabilities(self) -> Set[ToolCapability]:
        """
        Derive capabilities based on tier.

        Higher tiers unlock more capabilities.
        """
        caps = {ToolCapability.READ}  # All tools can read

        if self.tier.value >= ToolTier.T1_BASIC.value:
            caps.add(ToolCapability.ANALYZE)

        if self.tier.value >= ToolTier.T2_STANDARD.value:
            caps.add(ToolCapability.WRITE)
            caps.add(ToolCapability.VALIDATE)

        if self.tier.value >= ToolTier.T3_ADVANCED.value:
            caps.add(ToolCapability.TRANSFORM)
            caps.add(ToolCapability.GENERATE)

        if self.tier.value >= ToolTier.T4_EXPERT.value:
            caps.add(ToolCapability.EXECUTE)
            caps.add(ToolCapability.INTEGRATE)

        if self.tier.value >= ToolTier.T5_MASTER.value:
            caps.add(ToolCapability.OPTIMIZE)

        return caps

    def can(self, capability: ToolCapability) -> bool:
        """Check if tool has a specific capability."""
        return capability in self.capabilities

    def activate(self) -> bool:
        """Activate the tool for use."""
        if not self.active:
            self.active = True
            return True
        return False

    def deactivate(self) -> bool:
        """Deactivate the tool."""
        if self.active:
            self.active = False
            return True
        return False

    def bind_apl_engine(self, apl_engine: Any) -> None:
        """
        Bind an APL engine to this tool for operator-gated execution.

        Args:
            apl_engine: APLEngine instance from core.apl_engine
        """
        self._apl_engine = apl_engine

    def execute(self, operation: str, data: Any = None) -> Dict[str, Any]:
        """
        Execute an operation if tool has required capability.

        When APL engine is bound, operations are mapped to APL operators
        and gated by tier windows.

        Returns result dict with success status and output.
        """
        if not self.active:
            return {'success': False, 'error': 'Tool not active'}

        # Map operation to required capability
        op_cap_map = {
            'read': ToolCapability.READ,
            'write': ToolCapability.WRITE,
            'analyze': ToolCapability.ANALYZE,
            'transform': ToolCapability.TRANSFORM,
            'validate': ToolCapability.VALIDATE,
            'generate': ToolCapability.GENERATE,
            'execute': ToolCapability.EXECUTE,
            'integrate': ToolCapability.INTEGRATE,
            'optimize': ToolCapability.OPTIMIZE,
        }

        required_cap = op_cap_map.get(operation.lower())
        if required_cap is None:
            return {'success': False, 'error': f'Unknown operation: {operation}'}

        if not self.can(required_cap):
            return {
                'success': False,
                'error': f'Tool lacks capability: {required_cap.name}',
                'tier': self.tier.name,
                'available': [c.name for c in self.capabilities]
            }

        self.execution_count += 1

        # If APL engine bound, use operator-gated execution
        if self._apl_engine is not None:
            return self._execute_with_apl(operation, data)

        return {
            'success': True,
            'operation': operation,
            'tool_id': self.tool_id,
            'tier': self.tier.name,
            'execution_number': self.execution_count,
            'data': data
        }

    def _execute_with_apl(self, operation: str, data: Any) -> Dict[str, Any]:
        """
        Execute operation via APL engine with tier-gated operators.

        Maps operation to APL operator and checks legality at current z.
        """
        # Get APL operator for this operation
        apl_operator = OPERATION_TO_APL.get(operation.lower(), '()')

        # Check if operator is legal at current z
        if not self._apl_engine.is_operator_legal(apl_operator):
            # Find alternative legal operators
            legal_window = self._apl_engine.current_window()
            alternative = legal_window[0] if legal_window else '()'

            return {
                'success': False,
                'error': f"Operator '{apl_operator}' not legal at z={self._apl_engine.collapse.z:.4f}",
                'operation': operation,
                'apl_operator': apl_operator,
                'tier': self._apl_engine.current_tier(),
                'legal_operators': legal_window,
                'alternative': alternative,
            }

        try:
            # Apply operator via APL engine
            result = self._apl_engine.apply(apl_operator, data)

            # Log the operation
            log_entry = {
                'operation': operation,
                'apl_operator': apl_operator,
                'z': result.z,
                'tier': result.tier,
                'parity': result.parity,
                'delta_s_neg': result.delta_s_neg,
                'w_pi': result.w_pi,
            }
            self._operation_log.append(log_entry)

            return {
                'success': True,
                'operation': operation,
                'tool_id': self.tool_id,
                'tier': self.tier.name,
                'execution_number': self.execution_count,
                'data': result.output,
                'apl': {
                    'operator': apl_operator,
                    'z': result.z,
                    'harmonic_tier': result.tier,
                    'parity': result.parity,
                    'delta_s_neg': result.delta_s_neg,
                    'w_pi': result.w_pi,
                    'w_local': result.w_local,
                }
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'operation': operation,
                'apl_operator': apl_operator,
            }

    def get_apl_history(self) -> List[Dict]:
        """Get history of APL operations performed."""
        return list(self._operation_log)

    def get_info(self) -> Dict[str, Any]:
        """Get tool information."""
        return {
            'tool_id': self.tool_id,
            'tool_type': self.tool_type.name,
            'tier': self.tier.name,
            'work_invested': self.work_invested,
            'capabilities': [c.name for c in self.capabilities],
            'active': self.active,
            'execution_count': self.execution_count,
            'created_at': self.created_at.isoformat(),
        }


def create_tool(work: float, tool_type: ToolType = ToolType.ANALYZER) -> ChildTool:
    """Factory function to create a child tool."""
    return ChildTool(work_invested=work, tool_type=tool_type)
