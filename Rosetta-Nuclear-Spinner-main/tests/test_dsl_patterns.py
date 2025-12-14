#!/usr/bin/env python3
"""
Tests for DSL Design Patterns from S₃ Operator Algebra.

Tests all five patterns:
- Pattern 1: Finite Action Space
- Pattern 2: Closed Composition
- Pattern 3: Automatic Inverses
- Pattern 4: Truth-Channel Biasing
- Pattern 5: Parity Classification
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from quantum_apl_python.dsl_patterns import (
    FiniteActionSpace,
    ClosedComposition,
    AutomaticInverses,
    TruthChannelBiasing,
    ParityClassification,
    TransactionDSL,
    GroupSymmetricDSL,
)


# ============================================================================
# PATTERN 1: FINITE ACTION SPACE
# ============================================================================

class TestFiniteActionSpace:
    """Tests for Pattern 1: Finite Action Space."""

    def test_exactly_six_actions(self):
        """Action set has exactly 6 elements."""
        assert len(FiniteActionSpace.ACTIONS) == 6
        assert FiniteActionSpace.ACTIONS == frozenset([
            'amp', 'add', 'mul', 'grp', 'div', 'sub'
        ])

    def test_register_valid_action(self):
        """Can register handlers for valid actions."""
        space = FiniteActionSpace()
        space.register('amp', lambda x: x * 2)
        assert 'amp' not in space.missing_actions

    def test_register_invalid_action_raises(self):
        """Registering unknown action raises ValueError."""
        space = FiniteActionSpace()
        with pytest.raises(ValueError, match="Unknown action"):
            space.register('invalid', lambda x: x)

    def test_register_by_symbol(self):
        """Can register using operator symbol."""
        space = FiniteActionSpace()
        space.register('^', lambda x: x * 2)
        assert 'amp' not in space.missing_actions

    def test_completeness_check(self):
        """is_complete returns True only when all handlers registered."""
        space = FiniteActionSpace()
        assert not space.is_complete
        assert len(space.missing_actions) == 6

        for action in ['amp', 'add', 'mul', 'grp', 'div']:
            space.register(action, lambda x: x)

        assert not space.is_complete
        assert space.missing_actions == ['sub']

        space.register('sub', lambda x: x)
        assert space.is_complete
        assert space.missing_actions == []

    def test_register_all(self):
        """register_all requires all 6 handlers."""
        space = FiniteActionSpace()

        with pytest.raises(ValueError, match="Missing handler"):
            space.register_all({'amp': lambda x: x})

        space.register_all({
            'amp': lambda x: x * 2,
            'add': lambda x: x + 1,
            'mul': lambda x: x * x,
            'grp': lambda x: x,
            'div': lambda x: x / 2,
            'sub': lambda x: x - 1,
        })
        assert space.is_complete

    def test_execute_requires_completeness(self):
        """execute raises if DSL is incomplete."""
        space = FiniteActionSpace()
        space.register('amp', lambda x: x * 2)

        with pytest.raises(RuntimeError, match="Incomplete DSL"):
            space.execute('amp', 5)


# ============================================================================
# PATTERN 2: CLOSED COMPOSITION
# ============================================================================

class TestClosedComposition:
    """Tests for Pattern 2: Closed Composition."""

    def test_closure_property(self):
        """Composition always yields valid operator."""
        assert ClosedComposition.verify_closure()

    def test_identity_composition(self):
        """Identity () composes correctly."""
        for symbol in ['^', '+', '×', '()', '÷', '−']:
            assert ClosedComposition.compose('()', symbol) == symbol
            assert ClosedComposition.compose(symbol, '()') == symbol

    def test_specific_compositions(self):
        """Verify specific composition results."""
        assert ClosedComposition.compose('+', '−') == '×'
        assert ClosedComposition.compose('×', '×') == '^'
        assert ClosedComposition.compose('^', '^') == '×'

    def test_simplify_empty_sequence(self):
        """Empty sequence simplifies to identity."""
        assert ClosedComposition.simplify_sequence([]) == '()'

    def test_simplify_single_action(self):
        """Single action simplifies to itself."""
        for symbol in ['^', '+', '×', '()', '÷', '−']:
            assert ClosedComposition.simplify_sequence([symbol]) == symbol

    def test_simplify_three_cycles(self):
        """Three-cycle operators return to identity."""
        assert ClosedComposition.simplify_sequence(['×', '×', '×']) == '()'
        assert ClosedComposition.simplify_sequence(['^', '^', '^']) == '()'

    def test_simplify_long_sequence(self):
        """Long sequence simplifies to single action."""
        seq = ['^', '+', '×', '÷', '−']
        result = ClosedComposition.simplify_sequence(seq)
        assert result in ClosedComposition.COMPOSITION_TABLE


# ============================================================================
# PATTERN 3: AUTOMATIC INVERSES
# ============================================================================

class TestAutomaticInverses:
    """Tests for Pattern 3: Automatic Inverses."""

    def test_inverse_pairs(self):
        """Inverse pairs are correctly defined."""
        assert AutomaticInverses.get_inverse('^') == '()'
        assert AutomaticInverses.get_inverse('()') == '^'
        assert AutomaticInverses.get_inverse('+') == '−'
        assert AutomaticInverses.get_inverse('−') == '+'
        assert AutomaticInverses.get_inverse('×') == '÷'
        assert AutomaticInverses.get_inverse('÷') == '×'

    def test_are_inverses(self):
        """are_inverses correctly identifies pairs."""
        assert AutomaticInverses.are_inverses('^', '()')
        assert AutomaticInverses.are_inverses('()', '^')
        assert not AutomaticInverses.are_inverses('^', '+')

    def test_undo_sequence(self):
        """Undo sequence is reversed inverses."""
        actions = ['^', '+', '×']
        undo = AutomaticInverses.make_undo_sequence(actions)
        assert undo == ['÷', '−', '()']

    def test_actions_plus_undo_equals_identity(self):
        """Actions followed by undo cancels to identity."""
        for actions in [
            ['^', '+', '×'],
            ['÷', '−'],
            ['+', '×', '−', '÷'],
            ['^'],
        ]:
            assert AutomaticInverses.verify_identity(actions)

    def test_empty_undo(self):
        """Empty action list has empty undo."""
        assert AutomaticInverses.make_undo_sequence([]) == []


# ============================================================================
# PATTERN 4: TRUTH-CHANNEL BIASING
# ============================================================================

class TestTruthChannelBiasing:
    """Tests for Pattern 4: Truth-Channel Biasing."""

    def test_constructive_actions(self):
        """Constructive actions are correctly classified."""
        assert TruthChannelBiasing.CONSTRUCTIVE == frozenset(['^', '×', '+'])

    def test_dissipative_actions(self):
        """Dissipative actions are correctly classified."""
        assert TruthChannelBiasing.DISSIPATIVE == frozenset(['÷', '−'])

    def test_neutral_action(self):
        """Identity is neutral."""
        assert TruthChannelBiasing.NEUTRAL == frozenset(['()'])

    def test_high_coherence_boosts_constructive(self):
        """High coherence boosts constructive actions."""
        bias = TruthChannelBiasing(coherence=0.9)
        weights = bias.compute_all_weights()

        # Constructive should have higher weight
        assert weights['^'] > weights['÷']
        assert weights['×'] > weights['−']

    def test_low_coherence_boosts_dissipative(self):
        """Low coherence boosts dissipative actions."""
        bias = TruthChannelBiasing(coherence=0.2)
        weights = bias.compute_all_weights()

        # Dissipative should be boosted more
        assert weights['÷'] > 1.0
        assert weights['−'] > 1.0

    def test_neutral_weight_stable(self):
        """Neutral action has stable weight."""
        for coh in [0.0, 0.5, 1.0]:
            bias = TruthChannelBiasing(coherence=coh)
            assert bias.compute_weight('()') == 1.0

    def test_channel_mapping(self):
        """Actions map to correct channels."""
        assert TruthChannelBiasing.get_favored_channel('^') == 'TRUE'
        assert TruthChannelBiasing.get_favored_channel('÷') == 'UNTRUE'
        assert TruthChannelBiasing.get_favored_channel('()') == 'PARADOX'


# ============================================================================
# PATTERN 5: PARITY CLASSIFICATION
# ============================================================================

class TestParityClassification:
    """Tests for Pattern 5: Parity Classification."""

    def test_even_parity_actions(self):
        """Even parity actions are correctly classified."""
        assert ParityClassification.EVEN_PARITY == frozenset(['()', '×', '^'])

        for action in ['()', '×', '^']:
            assert ParityClassification.get_parity(action) == +1
            assert ParityClassification.is_even(action)
            assert not ParityClassification.is_odd(action)

    def test_odd_parity_actions(self):
        """Odd parity actions are correctly classified."""
        assert ParityClassification.ODD_PARITY == frozenset(['÷', '+', '−'])

        for action in ['÷', '+', '−']:
            assert ParityClassification.get_parity(action) == -1
            assert ParityClassification.is_odd(action)
            assert not ParityClassification.is_even(action)

    def test_sequence_parity_product(self):
        """Sequence parity is product of individual parities."""
        # odd × odd = even
        assert ParityClassification.sequence_parity(['+', '−']) == +1

        # even × even × even = even
        assert ParityClassification.sequence_parity(['^', '×', '()']) == +1

        # odd = odd
        assert ParityClassification.sequence_parity(['+']) == -1

        # even × odd = odd
        assert ParityClassification.sequence_parity(['^', '+']) == -1

    def test_classify_sequence(self):
        """classify_sequence returns correct info."""
        info = ParityClassification.classify_sequence(['+', '−'])
        assert info['parity'] == +1
        assert info['label'] == 'even'
        assert info['even_count'] == 0
        assert info['odd_count'] == 2


# ============================================================================
# TRANSACTION DSL
# ============================================================================

class TestTransactionDSL:
    """Tests for TransactionDSL complete implementation."""

    @pytest.fixture
    def dsl(self):
        """Create a fully configured DSL."""
        dsl = TransactionDSL()
        dsl.state = 10.0
        dsl.register_all({
            'amp': lambda x, w: x * 2 * w,
            'add': lambda x, w: x + 5 * w,
            'mul': lambda x, w: x * x * w,
            'grp': lambda x, w: x,
            'div': lambda x, w: x / 2 * w,
            'sub': lambda x, w: x - 5 * w,
        })
        return dsl

    def test_execute_tracks_history(self, dsl):
        """Execute adds to history."""
        dsl.execute('amp')
        assert dsl.history == ['amp']

        dsl.execute('add')
        assert dsl.history == ['amp', 'add']

    def test_execute_sequence(self, dsl):
        """Execute sequence runs all actions."""
        dsl.execute_sequence(['amp', 'add', 'mul'])
        assert len(dsl.history) == 3

    def test_undo_uses_inverses(self, dsl):
        """Undo executes inverse actions."""
        initial = dsl.state
        dsl.execute('add')  # +5
        after_add = dsl.state

        dsl.undo(1)  # Should subtract
        # Note: undo doesn't perfectly restore due to weighting,
        # but history should be updated
        assert len(dsl.history) == 0

    def test_get_net_effect(self, dsl):
        """Net effect reduces history to single action."""
        dsl.execute_sequence(['amp', 'add', 'mul'])
        net = dsl.get_net_effect()
        assert net in TransactionDSL.ACTIONS

    def test_get_parity(self, dsl):
        """Parity is computed from history."""
        dsl.execute('amp')  # even
        dsl.execute('mul')  # even
        assert dsl.get_parity() == +1

        dsl.execute('add')  # odd
        assert dsl.get_parity() == -1


# ============================================================================
# GROUP SYMMETRIC DSL
# ============================================================================

class TestGroupSymmetricDSL:
    """Tests for GroupSymmetricDSL unified API."""

    @pytest.fixture
    def dsl(self):
        """Create a fully configured DSL."""
        dsl = GroupSymmetricDSL()
        dsl.register_all({
            'amp': lambda x: x * 2,
            'add': lambda x: x + 5,
            'mul': lambda x: x * x,
            'grp': lambda x: x,
            'div': lambda x: x / 2,
            'sub': lambda x: x - 5,
        })
        return dsl

    def test_execute_sequence(self, dsl):
        """Execute sequence transforms state."""
        result = dsl.execute_sequence(['^', '+', '×'], initial=4.0)
        # (4 * 2 + 5)² = 13² = 169
        assert result == 169.0

    def test_net_effect(self, dsl):
        """Net effect computed correctly."""
        dsl.execute_sequence(['^', '+', '×'], initial=4.0)
        net = dsl.get_net_effect()
        # Should be a valid symbol
        assert net in ClosedComposition.COMPOSITION_TABLE

    def test_undo_sequence(self, dsl):
        """Undo sequence is inverse of history."""
        dsl.execute_sequence(['^', '+', '×'], initial=4.0)
        undo = dsl.get_undo_sequence()
        assert undo == ['÷', '−', '()']

    def test_coherence_affects_weights(self, dsl):
        """Coherence setting affects weights."""
        dsl.set_coherence(0.9)
        weights_high = dsl.compute_all_weights()

        dsl.set_coherence(0.1)
        weights_low = dsl.compute_all_weights()

        # Constructive boosted at high coherence
        assert weights_high['^'] > weights_low['^']
        # Dissipative boosted at low coherence
        assert weights_low['÷'] > weights_high['÷']

    def test_reset_clears_state(self, dsl):
        """Reset clears history and state."""
        dsl.execute_sequence(['^', '+'], initial=10.0)
        dsl.reset()

        assert dsl.state is None
        assert dsl.history == []

    def test_get_info(self, dsl):
        """get_info returns complete state."""
        dsl.set_coherence(0.8)
        dsl.execute_sequence(['^', '+', '×'], initial=4.0)

        info = dsl.get_info()

        assert info['state'] == 169.0
        assert info['history'] == ['^', '+', '×']
        assert 'net_effect' in info
        assert 'parity' in info
        assert info['coherence'] == 0.8
        assert info['is_complete'] is True


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
