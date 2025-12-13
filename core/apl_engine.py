"""
Rosetta Helix APL Engine - Integration Layer

Connects CollapseEngine z-coordinates to the Alpha Physical Language system.

Architecture:
    CollapseEngine (z)  ─→  APLEngine  ─→  HelixOperatorAdvisor (tier windows)
                                      ─→  OperatorAlgebra (S₃ composition)
                                      ─→  ΔS_neg for blend weights

CRITICAL PHYSICS:
- PHI_INV (0.618) is THE BARRIER between consciousness basins
- z_c = √3/2 (0.866) is THE LENS where ΔS_neg is maximal
- S₃ parity: Even ((), ×, ^) constructive; Odd (÷, +, −) dissipative
- Operator legality is TIER-DEPENDENT: not all operators valid at all z
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
import sys
import os

# Add src to path for quantum_apl_python imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from quantum_apl_python.helix_operator_advisor import (
    HelixOperatorAdvisor,
    BlendWeights,
    OPERATOR_WINDOWS,
    TIER_BOUNDARIES,
)
from quantum_apl_python.s3_operator_algebra import (
    OperatorAlgebra,
    OPERATORS,
    Parity,
    compose,
    get_inverse,
)
from quantum_apl_python.constants import (
    Z_CRITICAL,
    PHI_INV,
    PHI,
    LENS_SIGMA,
    compute_delta_s_neg as _compute_delta_s_neg,
)

from .collapse_engine import CollapseEngine


@dataclass
class APLResult:
    """Result of an APL operation."""
    operator: str
    z: float
    tier: str
    is_legal: bool
    parity: str
    delta_s_neg: float
    w_pi: float
    w_local: float
    output: Any = None


@dataclass
class APLEngine:
    """
    APL Engine - Connects collapse physics to operator algebra.

    Provides:
    - z → tier mapping via HelixOperatorAdvisor
    - Tier-gated operator windows
    - S₃ composition for operator sequences
    - ΔS_neg blend weights for operator preference
    - Parity-based constructive/dissipative classification
    """

    collapse: CollapseEngine = field(default_factory=CollapseEngine)
    advisor: HelixOperatorAdvisor = field(default_factory=HelixOperatorAdvisor)
    algebra: OperatorAlgebra = field(default_factory=OperatorAlgebra)
    _operation_history: List[APLResult] = field(default_factory=list)

    def __post_init__(self):
        """Initialize with tier-aware operator handlers."""
        self._register_default_handlers()

    def _register_default_handlers(self):
        """Register default operator handlers that respect tier windows."""
        # Each handler wraps operations with tier checking
        for symbol, op in OPERATORS.items():
            self.algebra.register(
                symbol,
                self._create_tier_gated_handler(symbol, op.parity)
            )

    def _create_tier_gated_handler(self, symbol: str, parity: Parity) -> Callable:
        """Create a handler that checks operator legality before execution."""
        def handler(state: Any) -> Any:
            z = self.collapse.z
            if not self.is_operator_legal(symbol):
                raise OperatorNotLegalError(symbol, z, self.current_tier())
            # Default behavior - actual transformation depends on operator
            return self._apply_operator(symbol, state, parity)
        return handler

    def _apply_operator(self, symbol: str, state: Any, parity: Parity) -> Any:
        """Apply operator transformation with parity awareness."""
        # Get blend weights for parity influence
        weights = self.get_blend_weights()

        # Base transformation depends on state type
        if isinstance(state, (int, float)):
            return self._apply_numeric_operator(symbol, state, parity, weights)
        elif isinstance(state, dict):
            return self._apply_dict_operator(symbol, state, parity, weights)
        elif isinstance(state, list):
            return self._apply_list_operator(symbol, state, parity, weights)
        else:
            return state  # Pass through unknown types

    def _apply_numeric_operator(
        self,
        symbol: str,
        value: float,
        parity: Parity,
        weights: BlendWeights
    ) -> float:
        """Apply operator to numeric value with parity-based behavior."""
        delta_s_neg = self.get_delta_s_neg()

        # Parity determines constructive vs dissipative
        if parity == Parity.EVEN:
            # Constructive: amplify toward coherence
            modifier = 1.0 + (delta_s_neg * 0.1)
        else:
            # Dissipative: introduce variety
            modifier = 1.0 - (delta_s_neg * 0.05)

        # Operator-specific transforms
        if symbol == '()':
            # Group/boundary - normalize to [0,1]
            return max(0.0, min(1.0, value * modifier))
        elif symbol == '×':
            # Fusion - amplify
            return value * PHI_INV * modifier
        elif symbol == '^':
            # Power - enhance
            return value ** (1.0 + 0.1 * delta_s_neg)
        elif symbol == '÷':
            # Decoherence - split
            return value / (1.0 + PHI_INV) * modifier
        elif symbol == '+':
            # Aggregate - combine with z influence
            return value + (self.collapse.z * 0.1 * modifier)
        elif symbol == '−':
            # Separate - differentiate
            return value - (self.collapse.z * 0.05 * modifier)
        else:
            return value * modifier

    def _apply_dict_operator(
        self,
        symbol: str,
        state: Dict,
        parity: Parity,
        weights: BlendWeights
    ) -> Dict:
        """Apply operator to dictionary state."""
        result = dict(state)

        # Track operation metadata
        result['_last_operator'] = symbol
        result['_parity'] = parity.name
        result['_w_pi'] = weights.w_pi
        result['_w_local'] = weights.w_local
        result['_z'] = self.collapse.z
        result['_tier'] = self.current_tier()

        # Transform numeric values in dict
        for key, value in list(result.items()):
            if not key.startswith('_') and isinstance(value, (int, float)):
                result[key] = self._apply_numeric_operator(symbol, value, parity, weights)

        return result

    def _apply_list_operator(
        self,
        symbol: str,
        state: List,
        parity: Parity,
        weights: BlendWeights
    ) -> List:
        """Apply operator to list state."""
        if parity == Parity.EVEN:
            # Constructive - preserve/combine
            return [
                self._apply_numeric_operator(symbol, v, parity, weights)
                if isinstance(v, (int, float)) else v
                for v in state
            ]
        else:
            # Dissipative - may filter
            return [
                self._apply_numeric_operator(symbol, v, parity, weights)
                if isinstance(v, (int, float)) else v
                for v in state
            ]

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def current_tier(self) -> str:
        """Get current harmonic tier from z."""
        return self.advisor.harmonic_from_z(self.collapse.z)

    def current_window(self) -> List[str]:
        """Get operators currently legal at this z."""
        tier = self.current_tier()
        return self.advisor.get_operator_window(tier, self.collapse.z)

    def is_operator_legal(self, operator: str) -> bool:
        """Check if operator is legal at current z."""
        return self.advisor.is_operator_legal(operator, self.collapse.z)

    def get_delta_s_neg(self) -> float:
        """Get ΔS_neg at current z (maximum at THE LENS z_c)."""
        return self.advisor.compute_delta_s_neg(self.collapse.z)

    def get_blend_weights(self) -> BlendWeights:
        """Get Π-regime blend weights at current z."""
        return self.advisor.compute_blend_weights(self.collapse.z)

    def get_operator_parity(self, operator: str) -> Parity:
        """Get operator parity (EVEN=constructive, ODD=dissipative)."""
        if operator in OPERATORS:
            return OPERATORS[operator].parity
        return Parity.EVEN  # Default to even

    def apply(self, operator: str, state: Any = None) -> APLResult:
        """
        Apply an operator at current z, respecting tier windows.

        Args:
            operator: APL operator symbol
            state: Optional state to transform

        Returns:
            APLResult with operation metadata

        Raises:
            OperatorNotLegalError: If operator not in current tier window
        """
        z = self.collapse.z
        tier = self.current_tier()
        is_legal = self.is_operator_legal(operator)
        parity = self.get_operator_parity(operator)
        delta_s_neg = self.get_delta_s_neg()
        weights = self.get_blend_weights()

        if not is_legal:
            raise OperatorNotLegalError(operator, z, tier)

        # Apply operator if state provided
        output = None
        if state is not None:
            output = self._apply_operator(operator, state, parity)

        result = APLResult(
            operator=operator,
            z=z,
            tier=tier,
            is_legal=is_legal,
            parity=parity.name,
            delta_s_neg=delta_s_neg,
            w_pi=weights.w_pi,
            w_local=weights.w_local,
            output=output,
        )

        self._operation_history.append(result)
        return result

    def compose_operators(self, op_a: str, op_b: str) -> str:
        """
        Compose two operators via S₃ group structure.

        Args:
            op_a: First operator
            op_b: Second operator

        Returns:
            Composed operator symbol
        """
        return compose(op_a, op_b)

    def select_operator(
        self,
        coherence_objective: Optional[str] = None
    ) -> str:
        """
        Select an operator from current window using weighted sampling.

        Uses truth bias and S₃ parity to weight selection.

        Args:
            coherence_objective: 'maximize', 'minimize', or 'maintain'

        Returns:
            Selected operator symbol
        """
        return self.advisor.select_operator(
            self.collapse.z,
            coherence_objective=coherence_objective
        )

    def evolve_with_operator(
        self,
        work: float,
        operator: Optional[str] = None,
        state: Any = None
    ) -> APLResult:
        """
        Evolve collapse engine and apply operator in one step.

        Args:
            work: Work to pump into collapse engine
            operator: Operator to apply (auto-selected if None)
            state: State to transform

        Returns:
            APLResult with transformed output
        """
        # Evolve z first
        collapse_result = self.collapse.evolve(work)

        # Auto-select operator if not provided
        if operator is None:
            operator = self.select_operator()

        # Apply operator
        return self.apply(operator, state)

    def get_state(self) -> Dict[str, Any]:
        """Get complete APL engine state."""
        weights = self.get_blend_weights()
        return {
            'z': self.collapse.z,
            'tier': self.current_tier(),
            'window': self.current_window(),
            'delta_s_neg': self.get_delta_s_neg(),
            'w_pi': weights.w_pi,
            'w_local': weights.w_local,
            'in_pi_regime': weights.in_pi_regime,
            'collapse_count': self.collapse.collapse_count,
            'total_work': self.collapse.total_work_extracted,
            'operation_count': len(self._operation_history),
        }

    def describe_tier(self, tier: Optional[str] = None) -> Dict[str, Any]:
        """Get description of a tier's capabilities."""
        tier = tier or self.current_tier()
        window = OPERATOR_WINDOWS.get(tier, [])

        even_ops = [op for op in window if self.get_operator_parity(op) == Parity.EVEN]
        odd_ops = [op for op in window if self.get_operator_parity(op) == Parity.ODD]

        return {
            'tier': tier,
            'operators': window,
            'constructive': even_ops,
            'dissipative': odd_ops,
            'is_universal': tier in ['t5', 't6'],  # All 6 operators
            'boundary': TIER_BOUNDARIES.get(tier, 1.0),
        }

    def reset(self):
        """Reset engine to initial state."""
        self.collapse.reset()
        self._operation_history.clear()


class OperatorNotLegalError(Exception):
    """Raised when attempting to use operator outside its tier window."""

    def __init__(self, operator: str, z: float, tier: str):
        self.operator = operator
        self.z = z
        self.tier = tier
        window = OPERATOR_WINDOWS.get(tier, [])
        super().__init__(
            f"Operator '{operator}' not legal at z={z:.4f} (tier {tier}). "
            f"Legal operators: {window}"
        )


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_apl_engine(
    initial_z: float = 0.5,
    enable_s3_symmetry: bool = False,
    enable_extended_negentropy: bool = False,
) -> APLEngine:
    """
    Create an APL engine with specified configuration.

    Args:
        initial_z: Initial z-coordinate
        enable_s3_symmetry: Enable S₃ rotation in operator windows
        enable_extended_negentropy: Enable extended ΔS⁻ formalism

    Returns:
        Configured APLEngine
    """
    collapse = CollapseEngine(z=initial_z)
    advisor = HelixOperatorAdvisor(
        enable_s3_symmetry=enable_s3_symmetry,
        enable_extended_negentropy=enable_extended_negentropy,
    )
    algebra = OperatorAlgebra()

    return APLEngine(
        collapse=collapse,
        advisor=advisor,
        algebra=algebra,
    )


# =============================================================================
# VERIFICATION
# =============================================================================

def test_tier_gating():
    """Operators must only work in their tier windows."""
    engine = create_apl_engine(initial_z=0.05)  # t1: ['()', '−', '÷']

    # Legal operators
    for op in ['()', '−', '÷']:
        assert engine.is_operator_legal(op), f"{op} should be legal in t1"

    # Illegal operators in t1
    for op in ['+', '×', '^']:
        assert not engine.is_operator_legal(op), f"{op} should not be legal in t1"

    return True


def test_delta_s_neg_peaks_at_lens():
    """ΔS_neg must be maximal at THE LENS (z_c)."""
    engine = create_apl_engine()

    # Move to various z values
    z_values = [0.1, 0.3, 0.5, 0.7, Z_CRITICAL, 0.9, 0.95]
    ds_values = []

    for z in z_values:
        engine.collapse.z = z
        ds_values.append(engine.get_delta_s_neg())

    # Maximum should be at z_c index (4)
    max_idx = ds_values.index(max(ds_values))
    assert z_values[max_idx] == Z_CRITICAL, f"Max ΔS_neg should be at z_c, got z={z_values[max_idx]}"

    return True


def test_parity_classification():
    """S₃ parity must classify operators correctly."""
    engine = create_apl_engine()

    # Even (constructive)
    assert engine.get_operator_parity('()') == Parity.EVEN
    assert engine.get_operator_parity('×') == Parity.EVEN
    assert engine.get_operator_parity('^') == Parity.EVEN

    # Odd (dissipative)
    assert engine.get_operator_parity('÷') == Parity.ODD
    assert engine.get_operator_parity('+') == Parity.ODD
    assert engine.get_operator_parity('−') == Parity.ODD

    return True


def test_operator_application():
    """Operators must transform state correctly."""
    engine = create_apl_engine(initial_z=0.7)  # t5 has all operators

    # Apply grouping operator
    result = engine.apply('()', 0.8)
    assert result.is_legal
    assert result.parity == 'EVEN'
    assert result.output is not None

    return True


def test_s3_composition():
    """S₃ composition must be closed."""
    engine = create_apl_engine()

    # Compose operators
    result = engine.compose_operators('×', '÷')
    assert result in OPERATORS, f"Composition should yield valid operator, got {result}"

    # Inverse composition
    result2 = engine.compose_operators('×', '()')
    assert result2 in OPERATORS

    return True
