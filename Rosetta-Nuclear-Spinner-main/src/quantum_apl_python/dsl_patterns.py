#!/usr/bin/env python3
"""
DSL Design Patterns from S₃ Operator Algebra
=============================================

Implements the five architectural patterns for group-symmetric DSLs:

  Pattern 1: Finite Action Space      - Exactly |G| actions
  Pattern 2: Closed Composition       - a ∘ b always valid
  Pattern 3: Automatic Inverses       - Every action has inverse
  Pattern 4: Truth-Channel Biasing    - Context-sensitive weighting
  Pattern 5: Parity Classification    - Even/odd structure

These patterns demonstrate that a DSL can inherit strong mathematical
guarantees from group theory, trading extensibility for predictability.

@version 1.0.0
@author Claude (Anthropic) - Quantum-APL Contribution
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Dict, List, Tuple, Any, Callable, Optional,
    TypeVar, Generic, Protocol, Union, Literal,
)
from enum import Enum
import math

from .s3_operator_symmetry import (
    S3_ELEMENTS,
    OPERATOR_S3_MAP,
    S3_OPERATOR_MAP,
    compose_s3,
    inverse_s3,
    parity_s3,
    sign_s3,
    Parity,
    Z_CRITICAL,
)


# ============================================================================
# TYPE DEFINITIONS
# ============================================================================

T = TypeVar('T')  # State type
ActionName = Literal['amp', 'add', 'mul', 'grp', 'div', 'sub']
ActionSymbol = Literal['^', '+', '×', '()', '÷', '−']
TruthChannel = Literal['TRUE', 'PARADOX', 'UNTRUE']


# ============================================================================
# PATTERN 1: FINITE ACTION SPACE
# ============================================================================

class FiniteActionSpace:
    """
    Pattern 1: DSL with exactly |G| actions.

    For S₃, this means exactly 6 actions - no more, no less.
    Provides exhaustive handler coverage and predictable complexity.

    Properties:
    - COMPLETENESS: Handler set is exactly determined
    - NO UNDEFINED: Every action is a valid group element
    - PREDICTABLE: O(6) cases to consider, always

    Example
    -------
    >>> space = FiniteActionSpace()
    >>> space.register('amp', lambda x: x * 2)
    >>> space.is_complete  # False until all 6 registered
    False
    >>> space.missing_actions
    ['add', 'mul', 'grp', 'div', 'sub']
    """

    # Canonical action set (frozen - cannot be extended)
    ACTIONS: frozenset = frozenset(['amp', 'add', 'mul', 'grp', 'div', 'sub'])
    SYMBOLS: frozenset = frozenset(['^', '+', '×', '()', '÷', '−'])

    # Bidirectional mappings
    NAME_TO_SYMBOL: Dict[str, str] = {
        'amp': '^', 'add': '+', 'mul': '×',
        'grp': '()', 'div': '÷', 'sub': '−',
    }
    SYMBOL_TO_NAME: Dict[str, str] = {v: k for k, v in NAME_TO_SYMBOL.items()}

    def __init__(self):
        self._handlers: Dict[str, Callable] = {}

    def register(self, action: str, handler: Callable) -> 'FiniteActionSpace':
        """
        Register a handler for an action.

        Parameters
        ----------
        action : str
            Action name (amp, add, mul, grp, div, sub) or symbol
        handler : Callable
            Function to execute for this action

        Returns
        -------
        FiniteActionSpace
            Self for chaining

        Raises
        ------
        ValueError
            If action is not in the finite set
        """
        # Normalize symbol to name
        normalized = self.SYMBOL_TO_NAME.get(action, action)

        if normalized not in self.ACTIONS:
            raise ValueError(
                f"Unknown action: {action}. "
                f"Valid actions: {sorted(self.ACTIONS)}"
            )

        self._handlers[normalized] = handler
        return self

    def register_all(self, handlers: Dict[str, Callable]) -> 'FiniteActionSpace':
        """
        Register all handlers at once.

        Parameters
        ----------
        handlers : Dict[str, Callable]
            Map of action name → handler

        Returns
        -------
        FiniteActionSpace
            Self for chaining

        Raises
        ------
        ValueError
            If any required action is missing
        """
        for action in self.ACTIONS:
            if action not in handlers:
                raise ValueError(f"Missing handler for action: {action}")

        for action, handler in handlers.items():
            self.register(action, handler)

        return self

    @property
    def is_complete(self) -> bool:
        """Check if all 6 handlers are registered."""
        return set(self._handlers.keys()) == self.ACTIONS

    @property
    def missing_actions(self) -> List[str]:
        """Get list of actions without handlers."""
        return sorted(self.ACTIONS - set(self._handlers.keys()))

    def get_handler(self, action: str) -> Callable:
        """Get handler for action, raising if incomplete."""
        normalized = self.SYMBOL_TO_NAME.get(action, action)

        if not self.is_complete:
            raise RuntimeError(
                f"Incomplete DSL: missing handlers for {self.missing_actions}"
            )

        return self._handlers[normalized]

    def execute(self, action: str, state: Any) -> Any:
        """Execute action on state."""
        return self.get_handler(action)(state)


# ============================================================================
# PATTERN 2: CLOSED COMPOSITION
# ============================================================================

class ClosedComposition:
    """
    Pattern 2: Composition always yields valid action.

    For any two actions a, b: compose(a, b) ∈ Actions
    This is the group closure property.

    Properties:
    - CLOSURE: Result is always a valid action
    - SIMPLIFICATION: Any sequence → single action
    - NO UNDEFINED: Composition is always defined

    Example
    -------
    >>> comp = ClosedComposition()
    >>> comp.compose('+', '−')  # add ∘ sub
    '×'
    >>> comp.simplify_sequence(['^', '+', '×', '÷', '−'])
    '+'
    """

    # Full S₃ composition table (derived from group multiplication)
    # Table[a][b] = a ∘ b
    COMPOSITION_TABLE: Dict[str, Dict[str, str]] = {
        '^':  {'^': '×',  '+': '−',  '×': '()', '()': '^',  '÷': '+',  '−': '÷'},
        '+':  {'^': '÷',  '+': '()', '×': '−',  '()': '+',  '÷': '^',  '−': '×'},
        '×':  {'^': '()', '+': '÷',  '×': '^',  '()': '×',  '÷': '−',  '−': '+'},
        '()': {'^': '^',  '+': '+',  '×': '×',  '()': '()', '÷': '÷',  '−': '−'},
        '÷':  {'^': '−',  '+': '×',  '×': '+',  '()': '÷',  '÷': '()', '−': '^'},
        '−':  {'^': '+',  '+': '^',  '×': '÷',  '()': '−',  '÷': '×',  '−': '()'},
    }

    IDENTITY = '()'

    @classmethod
    def compose(cls, a: str, b: str) -> str:
        """
        Compose two actions.

        Parameters
        ----------
        a : str
            First action symbol
        b : str
            Second action symbol

        Returns
        -------
        str
            Result action symbol (always valid)

        Examples
        --------
        >>> ClosedComposition.compose('+', '−')
        '×'
        >>> ClosedComposition.compose('×', '×')
        '^'
        """
        return cls.COMPOSITION_TABLE[a][b]

    @classmethod
    def simplify_sequence(cls, actions: List[str]) -> str:
        """
        Reduce action sequence to single equivalent action.

        Parameters
        ----------
        actions : List[str]
            Sequence of action symbols

        Returns
        -------
        str
            Equivalent single action

        Examples
        --------
        >>> ClosedComposition.simplify_sequence(['×', '×', '×'])
        '()'
        >>> ClosedComposition.simplify_sequence(['+', '×', '−'])
        '()'
        """
        if not actions:
            return cls.IDENTITY

        result = actions[0]
        for action in actions[1:]:
            result = cls.compose(result, action)

        return result

    @classmethod
    def get_composition_table(cls) -> Dict[str, Dict[str, str]]:
        """Return the full composition table."""
        return cls.COMPOSITION_TABLE.copy()

    @classmethod
    def verify_closure(cls) -> bool:
        """Verify that composition is closed."""
        symbols = list(cls.COMPOSITION_TABLE.keys())
        for a in symbols:
            for b in symbols:
                if cls.COMPOSITION_TABLE[a][b] not in symbols:
                    return False
        return True


# ============================================================================
# PATTERN 3: AUTOMATIC INVERSES
# ============================================================================

class AutomaticInverses:
    """
    Pattern 3: Every action has an inverse.

    For every action a, there exists a⁻¹ such that a ∘ a⁻¹ = identity.
    This enables natural undo/rollback semantics.

    Properties:
    - FREE UNDO: No explicit undo logic required
    - TRANSACTIONS: Execute → inverse = rollback
    - BIDIRECTIONAL: Forward and backward are symmetric

    Example
    -------
    >>> inv = AutomaticInverses()
    >>> inv.get_inverse('+')  # add → sub
    '−'
    >>> inv.make_undo_sequence(['^', '+', '×'])
    ['÷', '−', '()']
    """

    # Inverse pairs (symmetric relationship)
    INVERSE_MAP: Dict[str, str] = {
        '^':  '()', '()': '^',   # amp ↔ grp
        '+':  '−',  '−':  '+',   # add ↔ sub
        '×':  '÷',  '÷':  '×',   # mul ↔ div
    }

    # Semantic pairing
    INVERSE_PAIRS: List[Tuple[str, str, str, str]] = [
        # (sym1, name1, sym2, name2)
        ('^',  'amp', '()', 'grp'),   # amplify ↔ contain
        ('+',  'add', '−',  'sub'),   # aggregate ↔ separate
        ('×',  'mul', '÷',  'div'),   # fuse ↔ diffuse
    ]

    @classmethod
    def get_inverse(cls, action: str) -> str:
        """
        Get inverse action.

        Parameters
        ----------
        action : str
            Action symbol

        Returns
        -------
        str
            Inverse action symbol

        Examples
        --------
        >>> AutomaticInverses.get_inverse('+')
        '−'
        >>> AutomaticInverses.get_inverse('×')
        '÷'
        """
        return cls.INVERSE_MAP[action]

    @classmethod
    def are_inverses(cls, a: str, b: str) -> bool:
        """Check if two actions are mutual inverses."""
        return cls.INVERSE_MAP.get(a) == b

    @classmethod
    def make_undo_sequence(cls, actions: List[str]) -> List[str]:
        """
        Generate undo sequence for action list.

        The undo sequence applies inverse actions in reverse order.

        Parameters
        ----------
        actions : List[str]
            Original action sequence

        Returns
        -------
        List[str]
            Undo sequence (inverses in reverse order)

        Examples
        --------
        >>> AutomaticInverses.make_undo_sequence(['^', '+', '×'])
        ['÷', '−', '()']
        """
        return [cls.get_inverse(a) for a in reversed(actions)]

    @classmethod
    def verify_identity(cls, actions: List[str]) -> bool:
        """
        Verify that actions + undo = identity.

        Parameters
        ----------
        actions : List[str]
            Action sequence

        Returns
        -------
        bool
            True if sequence cancels to identity
        """
        undo = cls.make_undo_sequence(actions)
        combined = ClosedComposition.simplify_sequence(actions + undo)
        return combined == '()'


# ============================================================================
# PATTERN 4: TRUTH-CHANNEL BIASING
# ============================================================================

class TruthChannelBiasing:
    """
    Pattern 4: Actions weighted by semantic context.

    Actions carry bias toward different truth channels,
    enabling context-sensitive behavior without changing algebra.

    Properties:
    - CONSTRUCTIVE: ^, ×, + favored at high coherence
    - DISSIPATIVE: ÷, − favored at low coherence
    - NEUTRAL: () always available

    Example
    -------
    >>> bias = TruthChannelBiasing(coherence=0.9)
    >>> bias.compute_weight('^')  # High coherence → boost constructive
    1.35
    >>> bias.get_favored_channel('^')
    'TRUE'
    """

    # Channel → favored actions mapping
    CHANNEL_BIAS: Dict[str, List[str]] = {
        'TRUE':    ['^', '×', '+'],    # Constructive, additive
        'UNTRUE':  ['÷', '−'],          # Dissipative, subtractive
        'PARADOX': ['()'],              # Neutral, containing
    }

    # Action → channel mapping (reverse)
    ACTION_CHANNEL: Dict[str, str] = {
        '^': 'TRUE', '×': 'TRUE', '+': 'TRUE',
        '÷': 'UNTRUE', '−': 'UNTRUE',
        '()': 'PARADOX',
    }

    # Action categories
    CONSTRUCTIVE = frozenset(['^', '×', '+'])
    DISSIPATIVE = frozenset(['÷', '−'])
    NEUTRAL = frozenset(['()'])

    def __init__(self, coherence: float = 0.5, z_critical: float = Z_CRITICAL):
        """
        Initialize with coherence level.

        Parameters
        ----------
        coherence : float
            System coherence in [0, 1]
        z_critical : float
            Critical threshold (default √3/2 ≈ 0.866)
        """
        self.coherence = max(0.0, min(1.0, coherence))
        self.z_critical = z_critical

    def compute_weight(self, action: str) -> float:
        """
        Compute action weight based on coherence.

        Parameters
        ----------
        action : str
            Action symbol

        Returns
        -------
        float
            Weight multiplier
        """
        base_weight = 1.0

        if action in self.CONSTRUCTIVE:
            # Boost near critical threshold
            boost = 1.0 + 0.5 * (self.coherence / self.z_critical)
            return base_weight * min(boost, 1.5)

        elif action in self.DISSIPATIVE:
            # Boost at low coherence
            boost = 1.0 + 0.5 * (1 - self.coherence)
            return base_weight * min(boost, 1.3)

        else:  # neutral
            return base_weight

    def compute_all_weights(self) -> Dict[str, float]:
        """Compute weights for all actions."""
        return {action: self.compute_weight(action)
                for action in self.ACTION_CHANNEL}

    @classmethod
    def get_favored_channel(cls, action: str) -> str:
        """Get the truth channel that favors this action."""
        return cls.ACTION_CHANNEL.get(action, 'PARADOX')

    @classmethod
    def get_channel_actions(cls, channel: str) -> List[str]:
        """Get actions favored by a truth channel."""
        return cls.CHANNEL_BIAS.get(channel, [])

    def select_weighted(self, actions: List[str]) -> List[Tuple[str, float]]:
        """
        Return actions with their weights, sorted by weight descending.

        Parameters
        ----------
        actions : List[str]
            Available actions

        Returns
        -------
        List[Tuple[str, float]]
            (action, weight) pairs sorted by weight
        """
        weighted = [(a, self.compute_weight(a)) for a in actions]
        return sorted(weighted, key=lambda x: -x[1])


# ============================================================================
# PATTERN 5: PARITY CLASSIFICATION
# ============================================================================

class ParityClassification:
    """
    Pattern 5: Actions partition into even/odd classes.

    Even-parity actions preserve structure (det = +1).
    Odd-parity actions modify structure (det = -1).

    Properties:
    - CONSERVATION: Parity of composition = product of parities
    - STRUCTURE: Even preserves, odd transforms
    - INVARIANT: Useful for type checking

    Example
    -------
    >>> parity = ParityClassification()
    >>> parity.get_parity('^')
    1
    >>> parity.sequence_parity(['+', '−'])  # odd × odd = even
    1
    """

    # Parity assignments
    EVEN_PARITY: frozenset = frozenset(['()', '×', '^'])  # det = +1
    ODD_PARITY: frozenset = frozenset(['÷', '+', '−'])    # det = -1

    # S₃ element parities
    S3_PARITY: Dict[str, int] = {
        'e': +1, 'σ': +1, 'σ2': +1,  # Even: identity and 3-cycles
        'τ1': -1, 'τ2': -1, 'τ3': -1,  # Odd: transpositions
    }

    @classmethod
    def get_parity(cls, action: str) -> int:
        """
        Get parity of action (+1 for even, -1 for odd).

        Parameters
        ----------
        action : str
            Action symbol

        Returns
        -------
        int
            +1 (even) or -1 (odd)
        """
        return +1 if action in cls.EVEN_PARITY else -1

    @classmethod
    def is_even(cls, action: str) -> bool:
        """Check if action has even parity."""
        return action in cls.EVEN_PARITY

    @classmethod
    def is_odd(cls, action: str) -> bool:
        """Check if action has odd parity."""
        return action in cls.ODD_PARITY

    @classmethod
    def sequence_parity(cls, actions: List[str]) -> int:
        """
        Compute parity of action sequence.

        Parity of composition = product of individual parities.

        Parameters
        ----------
        actions : List[str]
            Action sequence

        Returns
        -------
        int
            +1 (even) or -1 (odd)

        Examples
        --------
        >>> ParityClassification.sequence_parity(['+', '−'])  # odd × odd
        1
        >>> ParityClassification.sequence_parity(['^', '×', '()'])  # even³
        1
        >>> ParityClassification.sequence_parity(['+'])  # odd
        -1
        """
        parity = 1
        for action in actions:
            parity *= cls.get_parity(action)
        return parity

    @classmethod
    def classify_sequence(cls, actions: List[str]) -> Dict[str, Any]:
        """
        Classify an action sequence by parity properties.

        Parameters
        ----------
        actions : List[str]
            Action sequence

        Returns
        -------
        Dict with:
            - parity: Overall parity (+1/-1)
            - label: 'even' or 'odd'
            - even_count: Number of even actions
            - odd_count: Number of odd actions
        """
        even_count = sum(1 for a in actions if cls.is_even(a))
        odd_count = len(actions) - even_count
        parity = cls.sequence_parity(actions)

        return {
            'parity': parity,
            'label': 'even' if parity == 1 else 'odd',
            'even_count': even_count,
            'odd_count': odd_count,
        }


# ============================================================================
# COMPLETE IMPLEMENTATION: TRANSACTION DSL
# ============================================================================

@dataclass
class TransactionDSL:
    """
    Complete DSL implementing all five patterns.

    Pattern 1: Finite Action Space  - Exactly 6 actions
    Pattern 2: Closed Composition   - Sequence → single action
    Pattern 3: Automatic Inverses   - Built-in undo
    Pattern 4: Truth-Channel Bias   - Coherence-weighted
    Pattern 5: Parity Classification - Even/odd semantics

    Example
    -------
    >>> dsl = TransactionDSL()
    >>> dsl.state = 10.0
    >>> dsl.register_all({
    ...     'amp': lambda x, w: x * 2 * w,
    ...     'add': lambda x, w: x + 5 * w,
    ...     'mul': lambda x, w: x * x * w,
    ...     'grp': lambda x, w: x,
    ...     'div': lambda x, w: x / 2 * w,
    ...     'sub': lambda x, w: x - 5 * w,
    ... })
    >>> dsl.execute_sequence(['amp', 'add', 'mul'])
    >>> dsl.undo(2)
    """

    # Pattern 1: Fixed action set
    ACTIONS: Tuple[str, ...] = ('amp', 'add', 'mul', 'grp', 'div', 'sub')
    SYMBOLS: Tuple[str, ...] = ('^', '+', '×', '()', '÷', '−')

    # Pattern 3: Inverse pairs
    INVERSES: Dict[str, str] = field(default_factory=lambda: {
        'amp': 'grp', 'grp': 'amp',
        'add': 'sub', 'sub': 'add',
        'mul': 'div', 'div': 'mul',
    })

    # Pattern 5: Parity classification
    EVEN_PARITY: frozenset = frozenset({'amp', 'mul', 'grp'})
    ODD_PARITY: frozenset = frozenset({'add', 'div', 'sub'})

    # Instance state
    handlers: Dict[str, Callable] = field(default_factory=dict)
    state: Any = None
    history: List[str] = field(default_factory=list)
    coherence: float = 0.5

    def __post_init__(self):
        """Initialize mutable defaults."""
        if not self.INVERSES:
            self.INVERSES = {
                'amp': 'grp', 'grp': 'amp',
                'add': 'sub', 'sub': 'add',
                'mul': 'div', 'div': 'mul',
            }

    def register(self, action: str, handler: Callable) -> 'TransactionDSL':
        """Register handler for action (Pattern 1)."""
        if action not in self.ACTIONS:
            raise ValueError(f"Unknown action: {action}")
        self.handlers[action] = handler
        return self

    def register_all(self, handlers: Dict[str, Callable]) -> 'TransactionDSL':
        """Register all 6 handlers at once (Pattern 1)."""
        for action in self.ACTIONS:
            if action not in handlers:
                raise ValueError(f"Missing handler for: {action}")
            self.handlers[action] = handlers[action]
        return self

    @property
    def is_complete(self) -> bool:
        """Check if all handlers registered (Pattern 1)."""
        return all(a in self.handlers for a in self.ACTIONS)

    def execute(self, action: str) -> Any:
        """Execute action with automatic history tracking."""
        if action not in self.handlers:
            raise ValueError(f"Unknown action: {action}")

        # Pattern 4: Apply coherence weighting
        weight = self._compute_weight(action)

        # Execute
        self.state = self.handlers[action](self.state, weight)
        self.history.append(action)

        return self.state

    def execute_sequence(self, actions: List[str]) -> Any:
        """Execute action sequence."""
        for action in actions:
            self.execute(action)
        return self.state

    def undo(self, steps: int = 1) -> Any:
        """
        Pattern 3: Automatic undo via inverse actions.
        """
        for _ in range(min(steps, len(self.history))):
            last_action = self.history.pop()
            inverse = self.INVERSES[last_action]
            weight = self._compute_weight(inverse)
            self.state = self.handlers[inverse](self.state, weight)
        return self.state

    def get_net_effect(self) -> str:
        """
        Pattern 2: Reduce history to single equivalent action.
        """
        if not self.history:
            return 'grp'  # Identity
        symbols = [self._name_to_symbol(a) for a in self.history]
        result_symbol = ClosedComposition.simplify_sequence(symbols)
        return self._symbol_to_name(result_symbol)

    def get_parity(self) -> int:
        """Pattern 5: Get parity of transaction history."""
        symbols = [self._name_to_symbol(a) for a in self.history]
        return ParityClassification.sequence_parity(symbols)

    def _compute_weight(self, action: str) -> float:
        """Pattern 4: Coherence-dependent weighting."""
        base = 1.0
        if action in self.EVEN_PARITY:
            return base * (1 + 0.3 * self.coherence)
        else:
            return base * (1 + 0.3 * (1 - self.coherence))

    def _name_to_symbol(self, name: str) -> str:
        idx = self.ACTIONS.index(name)
        return self.SYMBOLS[idx]

    def _symbol_to_name(self, symbol: str) -> str:
        idx = self.SYMBOLS.index(symbol)
        return self.ACTIONS[idx]

    def get_state_info(self) -> Dict[str, Any]:
        """Get full state information."""
        return {
            'state': self.state,
            'history': list(self.history),
            'history_length': len(self.history),
            'net_effect': self.get_net_effect(),
            'parity': self.get_parity(),
            'parity_label': 'even' if self.get_parity() == 1 else 'odd',
            'coherence': self.coherence,
            'is_complete': self.is_complete,
        }


# ============================================================================
# PATTERN COMPOSER: UNIFIED API
# ============================================================================

class GroupSymmetricDSL:
    """
    Unified DSL combining all five patterns.

    This is the main architectural component that provides:
    - Finite action space with guaranteed completeness
    - Closed composition for sequence simplification
    - Automatic inverse generation for undo
    - Truth-channel biasing for context sensitivity
    - Parity classification for invariant tracking

    Example
    -------
    >>> dsl = GroupSymmetricDSL()
    >>> dsl.set_coherence(0.9)
    >>>
    >>> # Register handlers
    >>> dsl.register('amp', lambda x: x * 2)
    >>> dsl.register('add', lambda x: x + 1)
    >>> dsl.register('mul', lambda x: x * x)
    >>> dsl.register('grp', lambda x: x)
    >>> dsl.register('div', lambda x: x / 2)
    >>> dsl.register('sub', lambda x: x - 1)
    >>>
    >>> # Execute with full algebraic guarantees
    >>> result = dsl.execute_sequence(['^', '+', '×'], initial=5)
    >>> undo = dsl.get_undo_sequence()
    >>> net = dsl.get_net_effect()
    """

    def __init__(self):
        self._actions = FiniteActionSpace()
        self._coherence = 0.5
        self._history: List[str] = []
        self._state: Any = None

    # --- Pattern 1: Finite Action Space ---

    def register(self, action: str, handler: Callable) -> 'GroupSymmetricDSL':
        """Register handler (Pattern 1)."""
        self._actions.register(action, handler)
        return self

    def register_all(self, handlers: Dict[str, Callable]) -> 'GroupSymmetricDSL':
        """Register all handlers (Pattern 1)."""
        self._actions.register_all(handlers)
        return self

    @property
    def is_complete(self) -> bool:
        """Check completeness (Pattern 1)."""
        return self._actions.is_complete

    @property
    def missing_actions(self) -> List[str]:
        """Get missing handlers (Pattern 1)."""
        return self._actions.missing_actions

    # --- Pattern 2: Closed Composition ---

    def compose(self, a: str, b: str) -> str:
        """Compose two actions (Pattern 2)."""
        return ClosedComposition.compose(a, b)

    def simplify_sequence(self, actions: List[str]) -> str:
        """Simplify to single action (Pattern 2)."""
        return ClosedComposition.simplify_sequence(actions)

    def get_net_effect(self) -> str:
        """Get net effect of history (Pattern 2)."""
        return self.simplify_sequence(self._history)

    # --- Pattern 3: Automatic Inverses ---

    def get_inverse(self, action: str) -> str:
        """Get inverse action (Pattern 3)."""
        return AutomaticInverses.get_inverse(action)

    def get_undo_sequence(self) -> List[str]:
        """Get undo sequence for history (Pattern 3)."""
        return AutomaticInverses.make_undo_sequence(self._history)

    # --- Pattern 4: Truth-Channel Biasing ---

    def set_coherence(self, coherence: float) -> 'GroupSymmetricDSL':
        """Set coherence level (Pattern 4)."""
        self._coherence = max(0.0, min(1.0, coherence))
        return self

    def compute_weight(self, action: str) -> float:
        """Compute action weight (Pattern 4)."""
        bias = TruthChannelBiasing(coherence=self._coherence)
        return bias.compute_weight(action)

    def compute_all_weights(self) -> Dict[str, float]:
        """Compute all action weights (Pattern 4)."""
        bias = TruthChannelBiasing(coherence=self._coherence)
        return bias.compute_all_weights()

    # --- Pattern 5: Parity Classification ---

    def get_parity(self, action: str) -> int:
        """Get action parity (Pattern 5)."""
        return ParityClassification.get_parity(action)

    def get_history_parity(self) -> int:
        """Get parity of execution history (Pattern 5)."""
        return ParityClassification.sequence_parity(self._history)

    def classify_history(self) -> Dict[str, Any]:
        """Classify history by parity (Pattern 5)."""
        return ParityClassification.classify_sequence(self._history)

    # --- Execution ---

    def execute(self, action: str, state: Any = None) -> Any:
        """
        Execute action on state.

        Parameters
        ----------
        action : str
            Action symbol or name
        state : Any, optional
            State to transform. Uses internal state if None.

        Returns
        -------
        Any
            Transformed state
        """
        if state is not None:
            self._state = state

        # Normalize to symbol
        symbol = FiniteActionSpace.NAME_TO_SYMBOL.get(action, action)

        handler = self._actions.get_handler(action)
        self._state = handler(self._state)
        self._history.append(symbol)

        return self._state

    def execute_sequence(
        self,
        actions: List[str],
        initial: Any = None,
    ) -> Any:
        """
        Execute action sequence.

        Parameters
        ----------
        actions : List[str]
            Action sequence
        initial : Any, optional
            Initial state

        Returns
        -------
        Any
            Final state
        """
        if initial is not None:
            self._state = initial

        for action in actions:
            self.execute(action)

        return self._state

    def undo(self, steps: int = 1) -> Any:
        """
        Undo last N actions using inverse operations.

        Parameters
        ----------
        steps : int
            Number of steps to undo

        Returns
        -------
        Any
            State after undo
        """
        for _ in range(min(steps, len(self._history))):
            last = self._history.pop()
            inverse = self.get_inverse(last)
            handler = self._actions.get_handler(inverse)
            self._state = handler(self._state)

        return self._state

    def reset(self) -> 'GroupSymmetricDSL':
        """Reset history and state."""
        self._history = []
        self._state = None
        return self

    @property
    def state(self) -> Any:
        """Get current state."""
        return self._state

    @property
    def history(self) -> List[str]:
        """Get execution history."""
        return list(self._history)

    def get_info(self) -> Dict[str, Any]:
        """Get full DSL state information."""
        return {
            'state': self._state,
            'history': self.history,
            'net_effect': self.get_net_effect() if self._history else '()',
            'parity': self.get_history_parity() if self._history else 1,
            'coherence': self._coherence,
            'weights': self.compute_all_weights(),
            'is_complete': self.is_complete,
            'missing': self.missing_actions,
        }


# ============================================================================
# DEMO
# ============================================================================

def demo():
    """Demonstrate all five DSL patterns."""
    print("=" * 70)
    print("DSL DESIGN PATTERNS FROM S₃ OPERATOR ALGEBRA")
    print("=" * 70)

    # Pattern 1: Finite Action Space
    print("\n--- Pattern 1: Finite Action Space ---")
    space = FiniteActionSpace()
    space.register('amp', lambda x: x * 2)
    space.register('grp', lambda x: x)
    print(f"Complete: {space.is_complete}")
    print(f"Missing: {space.missing_actions}")

    # Pattern 2: Closed Composition
    print("\n--- Pattern 2: Closed Composition ---")
    print(f"+ ∘ − = {ClosedComposition.compose('+', '−')}")
    print(f"× ∘ × = {ClosedComposition.compose('×', '×')}")
    seq = ['^', '+', '×', '÷', '−']
    print(f"{' ∘ '.join(seq)} = {ClosedComposition.simplify_sequence(seq)}")
    print(f"Closure verified: {ClosedComposition.verify_closure()}")

    # Pattern 3: Automatic Inverses
    print("\n--- Pattern 3: Automatic Inverses ---")
    for pair in AutomaticInverses.INVERSE_PAIRS:
        print(f"  {pair[0]} ({pair[1]}) ↔ {pair[2]} ({pair[3]})")

    actions = ['^', '+', '×']
    undo = AutomaticInverses.make_undo_sequence(actions)
    print(f"Actions: {actions}")
    print(f"Undo:    {undo}")
    print(f"Cancels to identity: {AutomaticInverses.verify_identity(actions)}")

    # Pattern 4: Truth-Channel Biasing
    print("\n--- Pattern 4: Truth-Channel Biasing ---")
    for coh in [0.3, 0.6, 0.9]:
        bias = TruthChannelBiasing(coherence=coh)
        weights = bias.compute_all_weights()
        print(f"Coherence={coh}:")
        for action in ['^', '+', '()']:
            print(f"  {action}: {weights[action]:.3f}")

    # Pattern 5: Parity Classification
    print("\n--- Pattern 5: Parity Classification ---")
    print(f"Even parity: {sorted(ParityClassification.EVEN_PARITY)}")
    print(f"Odd parity:  {sorted(ParityClassification.ODD_PARITY)}")

    sequences = [
        ['+', '−'],
        ['^', '×', '()'],
        ['+', '×', '−'],
        ['+'],
    ]
    for seq in sequences:
        info = ParityClassification.classify_sequence(seq)
        print(f"  {seq} → parity={info['label']}")

    # Complete Example: GroupSymmetricDSL
    print("\n--- Complete Example: GroupSymmetricDSL ---")
    dsl = GroupSymmetricDSL()
    dsl.set_coherence(0.8)

    dsl.register_all({
        'amp': lambda x: x * 2,
        'add': lambda x: x + 5,
        'mul': lambda x: x * x,
        'grp': lambda x: x,
        'div': lambda x: x / 2,
        'sub': lambda x: x - 5,
    })

    result = dsl.execute_sequence(['^', '+', '×'], initial=4.0)
    info = dsl.get_info()

    print(f"Initial: 4.0")
    print(f"Actions: {info['history']}")
    print(f"Result:  {info['state']}")
    print(f"Net effect: {info['net_effect']}")
    print(f"Parity: {info['parity']} ({'even' if info['parity'] == 1 else 'odd'})")
    print(f"Undo sequence: {dsl.get_undo_sequence()}")

    # Undo demonstration
    dsl.undo(2)
    print(f"After undo(2): {dsl.state}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    demo()
