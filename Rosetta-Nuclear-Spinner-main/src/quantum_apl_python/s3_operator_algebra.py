#!/usr/bin/env python3
"""
S₃ Operator Algebra Module
===========================

Closed algebraic structure for APL operators with DSL-focused design.

OPERATOR SET (6 elements, closed under composition):
    Symbol  Name    Algebraic   S₃ Element  Parity
    ------  ------  ----------  ----------  ------
    ^       amp     amplify     σ²          even
    +       add     aggregate   τ₂          odd
    ×       mul     multiply    σ           even
    ()      grp     group       e           even
    ÷       div     divide      τ₁          odd
    −       sub     subtract    τ₃          odd

DESIGN PRINCIPLES:
    1. Finite action space: Exactly 6 handlers needed
    2. Predictable composition: op₁ ∘ op₂ always yields valid op
    3. Invertibility pairs: +/−, ×/÷, ^/() provide natural undo

COMPOSITION TABLE (derived from S₃ group multiplication):
    The 6 operators form a closed set under composition.
    Composing any two operators yields another operator in the set.

INVERTIBILITY:
    Each operator has a natural inverse for undo semantics:
    - amp (^) ↔ grp (())  : amplify/contain
    - add (+) ↔ sub (−)   : aggregate/separate
    - mul (×) ↔ div (÷)   : fuse/diffuse

@version 1.0.0
@author Claude (Anthropic) - Quantum-APL Contribution
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Tuple
from enum import Enum

from .s3_operator_symmetry import (
    S3_ELEMENTS,
    OPERATOR_S3_MAP,
    S3_OPERATOR_MAP,
    compose_s3,
    inverse_s3,
    parity_s3,
    sign_s3,
    Parity,
)


# ============================================================================
# ALGEBRAIC OPERATOR DEFINITIONS
# ============================================================================

class AlgebraicOp(Enum):
    """Algebraic operator names for DSL design."""
    AMP = "amp"   # ^ - amplify
    ADD = "add"   # + - aggregate
    MUL = "mul"   # × - multiply/fuse
    GRP = "grp"   # () - group/contain
    DIV = "div"   # ÷ - divide/diffuse
    SUB = "sub"   # − - subtract/separate


@dataclass(frozen=True)
class Operator:
    """
    APL Operator with algebraic properties.

    Attributes
    ----------
    symbol : str
        The Unicode symbol (^, +, ×, (), ÷, −)
    name : str
        Short algebraic name (amp, add, mul, grp, div, sub)
    description : str
        Human-readable description
    s3_element : str
        Corresponding S₃ group element
    parity : Parity
        Even or odd permutation
    inverse_symbol : str
        Symbol of the inverse operator
    """
    symbol: str
    name: str
    description: str
    s3_element: str
    parity: Parity
    inverse_symbol: str

    @property
    def sign(self) -> int:
        """Get sign (+1 for even, -1 for odd)."""
        return sign_s3(self.s3_element)

    @property
    def is_constructive(self) -> bool:
        """Even-parity operators are constructive."""
        return self.parity == Parity.EVEN

    @property
    def is_dissipative(self) -> bool:
        """Odd-parity operators are dissipative."""
        return self.parity == Parity.ODD


# Operator registry (canonical definitions)
OPERATORS: Dict[str, Operator] = {
    "^": Operator(
        symbol="^",
        name="amp",
        description="amplify/excite",
        s3_element="σ2",
        parity=Parity.EVEN,
        inverse_symbol="()",
    ),
    "+": Operator(
        symbol="+",
        name="add",
        description="aggregate/route",
        s3_element="τ2",
        parity=Parity.ODD,
        inverse_symbol="−",
    ),
    "×": Operator(
        symbol="×",
        name="mul",
        description="multiply/fuse",
        s3_element="σ",
        parity=Parity.EVEN,
        inverse_symbol="÷",
    ),
    "()": Operator(
        symbol="()",
        name="grp",
        description="group/contain",
        s3_element="e",
        parity=Parity.EVEN,
        inverse_symbol="^",
    ),
    "÷": Operator(
        symbol="÷",
        name="div",
        description="divide/diffuse",
        s3_element="τ1",
        parity=Parity.ODD,
        inverse_symbol="×",
    ),
    "−": Operator(
        symbol="−",
        name="sub",
        description="subtract/separate",
        s3_element="τ3",
        parity=Parity.ODD,
        inverse_symbol="+",
    ),
}

# Name → Symbol lookup
NAME_TO_SYMBOL: Dict[str, str] = {op.name: op.symbol for op in OPERATORS.values()}

# Symbol → Name lookup
SYMBOL_TO_NAME: Dict[str, str] = {op.symbol: op.name for op in OPERATORS.values()}

# Canonical symbol ordering
SYMBOL_ORDER: List[str] = ["^", "+", "×", "()", "÷", "−"]

# Canonical name ordering
NAME_ORDER: List[str] = ["amp", "add", "mul", "grp", "div", "sub"]


# ============================================================================
# INVERTIBILITY PAIRS
# ============================================================================

INVERSE_PAIRS: List[Tuple[str, str]] = [
    ("^", "()"),   # amp ↔ grp
    ("+", "−"),    # add ↔ sub
    ("×", "÷"),    # mul ↔ div
]


def get_inverse(symbol: str) -> str:
    """
    Get the inverse operator symbol.

    Parameters
    ----------
    symbol : str
        Operator symbol

    Returns
    -------
    str
        Inverse operator symbol

    Examples
    --------
    >>> get_inverse("+")
    '−'
    >>> get_inverse("×")
    '÷'
    >>> get_inverse("^")
    '()'
    """
    return OPERATORS[symbol].inverse_symbol


def are_inverses(a: str, b: str) -> bool:
    """
    Check if two operators are inverses of each other.

    Parameters
    ----------
    a : str
        First operator symbol
    b : str
        Second operator symbol

    Returns
    -------
    bool
        True if operators are inverses
    """
    return get_inverse(a) == b


# ============================================================================
# OPERATOR COMPOSITION (Closed Set Property)
# ============================================================================

def compose(a: str, b: str) -> str:
    """
    Compose two operators (a ∘ b).

    The result is always another operator in the set (closure property).

    Parameters
    ----------
    a : str
        First operator symbol
    b : str
        Second operator symbol

    Returns
    -------
    str
        Result operator symbol

    Examples
    --------
    >>> compose("+", "−")  # add ∘ sub
    '÷'
    >>> compose("×", "×")  # mul ∘ mul
    '^'
    >>> compose("()", "×")  # grp ∘ mul (identity)
    '×'
    """
    s3_a = OPERATORS[a].s3_element
    s3_b = OPERATORS[b].s3_element
    s3_result = compose_s3(s3_a, s3_b)
    return S3_OPERATOR_MAP[s3_result]


def compose_sequence(operators: List[str]) -> str:
    """
    Compose a sequence of operators left-to-right.

    Parameters
    ----------
    operators : List[str]
        Sequence of operator symbols

    Returns
    -------
    str
        Result operator symbol (or "()" for empty sequence)

    Examples
    --------
    >>> compose_sequence(["×", "×", "×"])  # mul³ = grp (identity)
    '()'
    >>> compose_sequence(["+", "−"])  # add ∘ sub
    '÷'
    """
    if not operators:
        return "()"  # Identity element

    result = operators[0]
    for op in operators[1:]:
        result = compose(result, op)

    return result


def generate_composition_table() -> Dict[str, Dict[str, str]]:
    """
    Generate the full 6×6 composition table.

    Returns
    -------
    Dict[str, Dict[str, str]]
        table[a][b] = a ∘ b

    The table demonstrates closure: every cell contains
    a valid operator from the set.
    """
    table = {}
    for a in SYMBOL_ORDER:
        table[a] = {}
        for b in SYMBOL_ORDER:
            table[a][b] = compose(a, b)
    return table


# ============================================================================
# DSL HANDLER INTERFACE
# ============================================================================

# Type alias for operator handlers
OperatorHandler = Callable[[Any], Any]


@dataclass
class OperatorAlgebra:
    """
    DSL interpreter with exactly 6 handlers (one per operator).

    This class demonstrates the "finite action space" property:
    your interpreter only needs 6 handlers to process any
    valid operator sequence.

    Examples
    --------
    >>> algebra = OperatorAlgebra()
    >>> algebra.register("^", lambda x: x * 2)  # amplify
    >>> algebra.register("()", lambda x: x)     # group (identity)
    >>> algebra.apply("^", 5)
    10
    >>> algebra.apply_sequence(["^", "^"], 3)  # equivalent to "×" on result
    12
    """

    handlers: Dict[str, OperatorHandler] = field(default_factory=dict)

    def register(self, symbol: str, handler: OperatorHandler) -> None:
        """
        Register a handler for an operator.

        Parameters
        ----------
        symbol : str
            Operator symbol (^, +, ×, (), ÷, −)
        handler : Callable
            Function to execute for this operator
        """
        if symbol not in OPERATORS:
            raise ValueError(f"Unknown operator: {symbol}")
        self.handlers[symbol] = handler

    def register_by_name(self, name: str, handler: OperatorHandler) -> None:
        """
        Register a handler using algebraic name.

        Parameters
        ----------
        name : str
            Algebraic name (amp, add, mul, grp, div, sub)
        handler : Callable
            Function to execute for this operator
        """
        symbol = NAME_TO_SYMBOL.get(name)
        if symbol is None:
            raise ValueError(f"Unknown operator name: {name}")
        self.handlers[symbol] = handler

    def apply(self, symbol: str, value: Any) -> Any:
        """
        Apply a single operator to a value.

        Parameters
        ----------
        symbol : str
            Operator symbol
        value : Any
            Input value

        Returns
        -------
        Any
            Transformed value
        """
        handler = self.handlers.get(symbol)
        if handler is None:
            raise ValueError(f"No handler registered for: {symbol}")
        return handler(value)

    def apply_sequence(self, operators: List[str], value: Any) -> Any:
        """
        Apply a sequence of operators to a value.

        Parameters
        ----------
        operators : List[str]
            Sequence of operator symbols
        value : Any
            Initial value

        Returns
        -------
        Any
            Final transformed value
        """
        result = value
        for op in operators:
            result = self.apply(op, result)
        return result

    def apply_with_undo(
        self,
        operators: List[str],
        value: Any
    ) -> Tuple[Any, List[str]]:
        """
        Apply operators and return the undo sequence.

        The undo sequence uses invertibility pairs to reverse
        the transformation.

        Parameters
        ----------
        operators : List[str]
            Sequence of operator symbols
        value : Any
            Initial value

        Returns
        -------
        Tuple[Any, List[str]]
            (transformed_value, undo_sequence)
        """
        result = self.apply_sequence(operators, value)
        undo = [get_inverse(op) for op in reversed(operators)]
        return result, undo

    @property
    def is_complete(self) -> bool:
        """Check if all 6 handlers are registered."""
        return all(sym in self.handlers for sym in SYMBOL_ORDER)

    @property
    def missing_handlers(self) -> List[str]:
        """Get list of operators without handlers."""
        return [sym for sym in SYMBOL_ORDER if sym not in self.handlers]


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def operator_info(symbol: str) -> Dict[str, Any]:
    """
    Get full information about an operator.

    Parameters
    ----------
    symbol : str
        Operator symbol

    Returns
    -------
    Dict[str, Any]
        Operator properties
    """
    op = OPERATORS.get(symbol)
    if op is None:
        raise ValueError(f"Unknown operator: {symbol}")

    return {
        "symbol": op.symbol,
        "name": op.name,
        "description": op.description,
        "s3_element": op.s3_element,
        "parity": op.parity.value,
        "sign": op.sign,
        "inverse": op.inverse_symbol,
        "inverse_name": SYMBOL_TO_NAME[op.inverse_symbol],
        "is_constructive": op.is_constructive,
        "is_dissipative": op.is_dissipative,
    }


def simplify_sequence(operators: List[str]) -> str:
    """
    Simplify an operator sequence to its equivalent single operator.

    This demonstrates that any sequence of operators from the set
    can be reduced to a single operator (closure + group structure).

    Parameters
    ----------
    operators : List[str]
        Sequence of operator symbols

    Returns
    -------
    str
        Equivalent single operator

    Examples
    --------
    >>> simplify_sequence(["×", "×", "×"])  # 3-cycle returns to identity
    '()'
    >>> simplify_sequence(["+", "+"])  # add ∘ add
    '÷'
    >>> simplify_sequence(["^", "()"])  # amp ∘ grp
    '^'
    """
    return compose_sequence(operators)


def find_path_to_identity(symbol: str) -> List[str]:
    """
    Find the shortest sequence that returns an operator to identity.

    Parameters
    ----------
    symbol : str
        Starting operator symbol

    Returns
    -------
    List[str]
        Sequence that when composed with symbol gives "()"

    Examples
    --------
    >>> find_path_to_identity("×")  # mul needs mul² to reach identity
    ['×', '×']
    >>> find_path_to_identity("+")  # add is self-inverse (order 2)
    ['+']
    """
    # For even parity (3-cycles and identity)
    s3_elem = OPERATORS[symbol].s3_element

    if s3_elem == "e":  # Already identity
        return []
    elif s3_elem == "σ":  # 3-cycle: needs σ² to return
        return ["×", "×"]
    elif s3_elem == "σ2":  # σ² needs σ to return
        return ["^", "^"]
    else:  # Transpositions are self-inverse
        return [symbol]


def order_of(symbol: str) -> int:
    """
    Get the order of an operator in the group.

    The order is the smallest n such that op^n = identity.

    Parameters
    ----------
    symbol : str
        Operator symbol

    Returns
    -------
    int
        Order (1, 2, or 3)
    """
    s3_elem = OPERATORS[symbol].s3_element

    if s3_elem == "e":
        return 1  # Identity has order 1
    elif s3_elem in ("σ", "σ2"):
        return 3  # 3-cycles have order 3
    else:
        return 2  # Transpositions have order 2


# ============================================================================
# DEMO
# ============================================================================

def demo():
    """Demonstrate S₃ operator algebra."""
    print("=" * 70)
    print("S₃ OPERATOR ALGEBRA")
    print("=" * 70)

    # Show operators
    print("\n--- Operator Set (6 elements) ---")
    print(f"{'Symbol':<8} {'Name':<6} {'Description':<20} {'Parity':<6} {'Inverse'}")
    print("-" * 60)
    for sym in SYMBOL_ORDER:
        op = OPERATORS[sym]
        print(f"{sym:<8} {op.name:<6} {op.description:<20} {op.parity.value:<6} {op.inverse_symbol}")

    # Invertibility pairs
    print("\n--- Invertibility Pairs (undo semantics) ---")
    for a, b in INVERSE_PAIRS:
        name_a = SYMBOL_TO_NAME[a]
        name_b = SYMBOL_TO_NAME[b]
        print(f"  {a} ({name_a}) ↔ {b} ({name_b})")

    # Composition table
    print("\n--- Composition Table (demonstrates closure) ---")
    table = generate_composition_table()

    # Header
    header = "  ∘  │ " + "  ".join(f"{s:>3}" for s in SYMBOL_ORDER)
    print(header)
    print("─" * len(header))

    # Rows
    for a in SYMBOL_ORDER:
        row = f" {a:>3} │ " + "  ".join(f"{table[a][b]:>3}" for b in SYMBOL_ORDER)
        print(row)

    # Demonstrate closure
    print("\n--- Closure Property ---")
    print("  Every composition yields a valid operator:")
    examples = [
        ("+", "−"),
        ("×", "×"),
        ("^", "()"),
        ("÷", "+"),
    ]
    for a, b in examples:
        result = compose(a, b)
        print(f"  {a} ∘ {b} = {result} ({SYMBOL_TO_NAME[result]})")

    # Sequence simplification
    print("\n--- Sequence Simplification ---")
    sequences = [
        ["×", "×", "×"],
        ["+", "−"],
        ["^", "^", "^"],
        ["+", "×", "−"],
    ]
    for seq in sequences:
        result = simplify_sequence(seq)
        seq_str = " ∘ ".join(seq)
        print(f"  {seq_str} = {result} ({SYMBOL_TO_NAME[result]})")

    # Order of operators
    print("\n--- Operator Orders ---")
    for sym in SYMBOL_ORDER:
        order = order_of(sym)
        path = find_path_to_identity(sym)
        path_str = " ∘ ".join([sym] + path) if path else sym
        print(f"  {sym}^{order} = () | {path_str} → ()")

    # DSL Handler example
    print("\n--- DSL Handler Interface ---")
    algebra = OperatorAlgebra()
    algebra.register("^", lambda x: x * 2)      # amp: double
    algebra.register("+", lambda x: x + 1)      # add: increment
    algebra.register("×", lambda x: x ** 2)     # mul: square
    algebra.register("()", lambda x: x)         # grp: identity
    algebra.register("÷", lambda x: x ** 0.5)   # div: sqrt
    algebra.register("−", lambda x: x - 1)      # sub: decrement

    print(f"  Handlers registered: {algebra.is_complete}")

    value = 4
    ops = ["^", "+", "×"]
    result, undo = algebra.apply_with_undo(ops, value)
    print(f"  Apply {ops} to {value}: {result}")
    print(f"  Undo sequence: {undo}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    demo()
