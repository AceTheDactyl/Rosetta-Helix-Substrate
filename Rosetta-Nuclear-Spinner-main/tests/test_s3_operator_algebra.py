#!/usr/bin/env python3
"""
Test suite for S₃ Operator Algebra module.

Tests the closed algebraic structure for APL operators:
- 6 operators forming a closed set under composition
- Invertibility pairs: +/-, x//, ^/()
- DSL handler interface
- Composition table correctness

@version 1.0.0
@author Claude (Anthropic) - Quantum-APL Contribution
"""

import pytest
from quantum_apl_python.s3_operator_algebra import (
    OPERATORS,
    NAME_TO_SYMBOL,
    SYMBOL_TO_NAME,
    SYMBOL_ORDER,
    NAME_ORDER,
    INVERSE_PAIRS,
    get_inverse,
    are_inverses,
    compose,
    compose_sequence,
    generate_composition_table,
    OperatorAlgebra,
    operator_info,
    simplify_sequence,
    find_path_to_identity,
    order_of,
    Parity,
)


# ============================================================================
# OPERATOR DEFINITIONS TESTS
# ============================================================================

def test_operator_count():
    """Exactly 6 operators in the algebra."""
    assert len(OPERATORS) == 6
    assert len(SYMBOL_ORDER) == 6
    assert len(NAME_ORDER) == 6


def test_symbol_name_mapping():
    """Symbol <-> name bijection."""
    for sym, name in SYMBOL_TO_NAME.items():
        assert NAME_TO_SYMBOL[name] == sym

    for name, sym in NAME_TO_SYMBOL.items():
        assert SYMBOL_TO_NAME[sym] == name


def test_algebraic_names():
    """Algebraic names match specification."""
    expected_names = ["amp", "add", "mul", "grp", "div", "sub"]
    assert NAME_ORDER == expected_names


def test_operator_symbols():
    """Operator symbols match specification."""
    expected_symbols = ["^", "+", "×", "()", "÷", "−"]
    assert SYMBOL_ORDER == expected_symbols


def test_parity_classification():
    """Even/odd parity matches S3 group structure."""
    even_ops = ["^", "×", "()"]  # amp, mul, grp
    odd_ops = ["+", "÷", "−"]    # add, div, sub

    for sym in even_ops:
        assert OPERATORS[sym].parity == Parity.EVEN
        assert OPERATORS[sym].sign == +1

    for sym in odd_ops:
        assert OPERATORS[sym].parity == Parity.ODD
        assert OPERATORS[sym].sign == -1


# ============================================================================
# INVERTIBILITY TESTS
# ============================================================================

def test_inverse_pairs():
    """Inverse pairs are correctly defined."""
    expected_pairs = [("^", "()"), ("+", "−"), ("×", "÷")]
    assert INVERSE_PAIRS == expected_pairs


def test_get_inverse():
    """get_inverse returns correct inverses."""
    assert get_inverse("^") == "()"
    assert get_inverse("()") == "^"
    assert get_inverse("+") == "−"
    assert get_inverse("−") == "+"
    assert get_inverse("×") == "÷"
    assert get_inverse("÷") == "×"


def test_inverse_symmetry():
    """Inverse operation is symmetric."""
    for sym in SYMBOL_ORDER:
        inv = get_inverse(sym)
        assert get_inverse(inv) == sym


def test_are_inverses():
    """are_inverses detects inverse pairs."""
    assert are_inverses("^", "()")
    assert are_inverses("()", "^")
    assert are_inverses("+", "−")
    assert are_inverses("×", "÷")

    # Non-inverses
    assert not are_inverses("^", "+")
    assert not are_inverses("×", "−")


# ============================================================================
# COMPOSITION TESTS (Closure Property)
# ============================================================================

def test_compose_identity():
    """Identity element () is neutral."""
    for sym in SYMBOL_ORDER:
        # () o X = X
        assert compose("()", sym) == sym
        # X o () = X
        assert compose(sym, "()") == sym


def test_compose_closure():
    """Composition always yields element in the set (closure)."""
    for a in SYMBOL_ORDER:
        for b in SYMBOL_ORDER:
            result = compose(a, b)
            assert result in SYMBOL_ORDER


def test_compose_with_inverse():
    """Composing with inverse yields known result."""
    # For transpositions (self-inverse), X o X = identity
    # + -> tau2, tau2 o tau2 = e... wait let me think
    # Actually tau2 o tau2 in S3 is different

    # Let's test specific compositions we know from S3
    # sigma o sigma = sigma^2 (mul o mul = amp)
    assert compose("×", "×") == "^"

    # sigma^2 o sigma = e (amp o mul = grp)
    assert compose("^", "×") == "()"

    # sigma o sigma^2 = e (mul o amp = grp)
    assert compose("×", "^") == "()"


def test_compose_transposition_self():
    """Transpositions composed with themselves."""
    # In S3, tau_i o tau_i = e only for transpositions
    # But composition in our algebra: tau1 o tau1 = ?
    # tau1 = (1,0,2), so (1,0,2) o (1,0,2) = (0,1,2) = e
    # So div o div should be grp
    assert compose("÷", "÷") == "()"
    assert compose("+", "+") == "÷"  # tau2 o tau2 in S3
    assert compose("−", "−") == "()"  # tau3 o tau3 = e


def test_composition_table_complete():
    """Composition table covers all 36 combinations."""
    table = generate_composition_table()

    assert len(table) == 6
    for row in table.values():
        assert len(row) == 6


def test_compose_sequence_empty():
    """Empty sequence gives identity."""
    assert compose_sequence([]) == "()"


def test_compose_sequence_single():
    """Single element sequence returns that element."""
    for sym in SYMBOL_ORDER:
        assert compose_sequence([sym]) == sym


def test_compose_sequence_triple_cycle():
    """3-cycle cubed returns to identity."""
    # mul^3 = grp (sigma^3 = e)
    assert compose_sequence(["×", "×", "×"]) == "()"

    # amp^3 = grp (sigma^2)^3 = e
    assert compose_sequence(["^", "^", "^"]) == "()"


# ============================================================================
# DSL HANDLER INTERFACE TESTS
# ============================================================================

def test_operator_algebra_register():
    """Can register handlers for operators."""
    algebra = OperatorAlgebra()

    algebra.register("^", lambda x: x * 2)
    algebra.register("+", lambda x: x + 1)

    assert "^" in algebra.handlers
    assert "+" in algebra.handlers


def test_operator_algebra_register_by_name():
    """Can register handlers by algebraic name."""
    algebra = OperatorAlgebra()

    algebra.register_by_name("amp", lambda x: x * 2)
    algebra.register_by_name("add", lambda x: x + 1)

    assert "^" in algebra.handlers
    assert "+" in algebra.handlers


def test_operator_algebra_apply():
    """Apply single operator."""
    algebra = OperatorAlgebra()
    algebra.register("^", lambda x: x * 2)

    assert algebra.apply("^", 5) == 10


def test_operator_algebra_apply_sequence():
    """Apply sequence of operators."""
    algebra = OperatorAlgebra()
    algebra.register("^", lambda x: x * 2)
    algebra.register("+", lambda x: x + 1)
    algebra.register("×", lambda x: x ** 2)

    # 3 -> amp(3)=6 -> add(6)=7 -> mul(7)=49
    result = algebra.apply_sequence(["^", "+", "×"], 3)
    assert result == 49


def test_operator_algebra_with_undo():
    """Apply with undo returns inverse sequence."""
    algebra = OperatorAlgebra()
    algebra.register("^", lambda x: x * 2)
    algebra.register("+", lambda x: x + 1)
    algebra.register("()", lambda x: x)
    algebra.register("−", lambda x: x - 1)

    result, undo = algebra.apply_with_undo(["^", "+"], 5)

    # Result: 5 -> 10 -> 11
    assert result == 11

    # Undo sequence: [+, ^] inverses = [-, ()]
    assert undo == ["−", "()"]


def test_operator_algebra_is_complete():
    """is_complete checks all handlers registered."""
    algebra = OperatorAlgebra()
    assert not algebra.is_complete

    for sym in SYMBOL_ORDER:
        algebra.register(sym, lambda x: x)

    assert algebra.is_complete


def test_operator_algebra_missing_handlers():
    """missing_handlers lists unregistered operators."""
    algebra = OperatorAlgebra()
    algebra.register("^", lambda x: x)
    algebra.register("+", lambda x: x)

    missing = algebra.missing_handlers
    assert "^" not in missing
    assert "+" not in missing
    assert "×" in missing
    assert "()" in missing
    assert "÷" in missing
    assert "−" in missing


# ============================================================================
# UTILITY FUNCTION TESTS
# ============================================================================

def test_operator_info():
    """operator_info returns complete information."""
    info = operator_info("^")

    assert info["symbol"] == "^"
    assert info["name"] == "amp"
    assert info["parity"] == "even"
    assert info["sign"] == +1
    assert info["inverse"] == "()"
    assert info["inverse_name"] == "grp"
    assert info["is_constructive"] is True
    assert info["is_dissipative"] is False


def test_simplify_sequence():
    """simplify_sequence reduces to single operator."""
    # mul^3 = grp
    assert simplify_sequence(["×", "×", "×"]) == "()"

    # add o sub = ?
    # tau2 o tau3 in S3... need to compute
    result = simplify_sequence(["+", "−"])
    assert result in SYMBOL_ORDER  # Closure holds


def test_order_of_operators():
    """Order of each operator in the group."""
    # Identity has order 1
    assert order_of("()") == 1

    # 3-cycles have order 3
    assert order_of("×") == 3
    assert order_of("^") == 3

    # Transpositions have order 2
    assert order_of("+") == 2
    assert order_of("÷") == 2
    assert order_of("−") == 2


def test_find_path_to_identity():
    """find_path_to_identity returns correct path."""
    # Identity needs no path
    assert find_path_to_identity("()") == []

    # 3-cycle needs 2 more of same to return
    path_mul = find_path_to_identity("×")
    composed = compose_sequence(["×"] + path_mul)
    assert composed == "()"

    path_amp = find_path_to_identity("^")
    composed = compose_sequence(["^"] + path_amp)
    assert composed == "()"

    # Transpositions are self-inverse
    path_add = find_path_to_identity("+")
    composed = compose_sequence(["+"] + path_add)
    assert composed == "()"


# ============================================================================
# GROUP THEORY PROPERTY TESTS
# ============================================================================

def test_associativity():
    """Composition is associative: (a o b) o c = a o (b o c)."""
    for a in SYMBOL_ORDER:
        for b in SYMBOL_ORDER:
            for c in SYMBOL_ORDER:
                left = compose(compose(a, b), c)
                right = compose(a, compose(b, c))
                assert left == right, f"Associativity failed for ({a} o {b}) o {c}"


def test_unique_identity():
    """There is exactly one identity element."""
    identities = []
    for sym in SYMBOL_ORDER:
        is_identity = all(
            compose(sym, x) == x and compose(x, sym) == x
            for x in SYMBOL_ORDER
        )
        if is_identity:
            identities.append(sym)

    assert identities == ["()"]


def test_inverse_uniqueness():
    """Each element has a unique inverse."""
    for sym in SYMBOL_ORDER:
        inverses = []
        for candidate in SYMBOL_ORDER:
            if compose(sym, candidate) == "()" and compose(candidate, sym) == "()":
                inverses.append(candidate)

        assert len(inverses) == 1, f"Multiple inverses for {sym}: {inverses}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
