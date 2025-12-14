/**
 * Test suite for S₃ Operator Algebra module (JavaScript)
 *
 * Tests the closed algebraic structure for APL operators:
 * - 6 operators forming a closed set under composition
 * - Invertibility pairs: +/-, x//, ^/()
 * - DSL handler interface
 * - Composition table correctness
 *
 * @version 1.0.0
 * @author Claude (Anthropic) - Quantum-APL Contribution
 */

'use strict';

const {
  OPERATORS,
  NAME_TO_SYMBOL,
  SYMBOL_TO_NAME,
  SYMBOL_ORDER,
  NAME_ORDER,
  INVERSE_PAIRS,
  getInverse,
  areInverses,
  compose,
  composeSequence,
  generateCompositionTable,
  OperatorAlgebra,
  operatorInfo,
  simplifySequence,
  findPathToIdentity,
  orderOf,
  verifyClosure,
} = require('../src/s3_operator_algebra');

// ============================================================================
// TEST UTILITIES
// ============================================================================

let passed = 0;
let failed = 0;
const errors = [];

function test(name, fn) {
  try {
    fn();
    passed++;
    console.log(`  ✓ ${name}`);
  } catch (e) {
    failed++;
    errors.push({ name, error: e.message });
    console.log(`  ✗ ${name}`);
    console.log(`    → ${e.message}`);
  }
}

function assert(condition, message) {
  if (!condition) {
    throw new Error(message || 'Assertion failed');
  }
}

function assertEquals(actual, expected, message) {
  if (actual !== expected) {
    throw new Error(message || `Expected ${expected}, got ${actual}`);
  }
}

function assertArrayEquals(actual, expected, message) {
  if (JSON.stringify(actual) !== JSON.stringify(expected)) {
    throw new Error(message || `Expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}`);
  }
}

function section(title) {
  console.log(`\n${'─'.repeat(60)}`);
  console.log(`  ${title}`);
  console.log(`${'─'.repeat(60)}`);
}

// ============================================================================
// OPERATOR DEFINITIONS TESTS
// ============================================================================

section('Operator Definitions');

test('Exactly 6 operators in the algebra', () => {
  assertEquals(Object.keys(OPERATORS).length, 6);
  assertEquals(SYMBOL_ORDER.length, 6);
  assertEquals(NAME_ORDER.length, 6);
});

test('Symbol <-> name bijection', () => {
  for (const [sym, name] of Object.entries(SYMBOL_TO_NAME)) {
    assertEquals(NAME_TO_SYMBOL[name], sym);
  }
  for (const [name, sym] of Object.entries(NAME_TO_SYMBOL)) {
    assertEquals(SYMBOL_TO_NAME[sym], name);
  }
});

test('Algebraic names match specification', () => {
  assertArrayEquals(NAME_ORDER, ['amp', 'add', 'mul', 'grp', 'div', 'sub']);
});

test('Operator symbols match specification', () => {
  assertArrayEquals(SYMBOL_ORDER, ['^', '+', '×', '()', '÷', '−']);
});

test('Parity classification', () => {
  const evenOps = ['^', '×', '()'];
  const oddOps = ['+', '÷', '−'];

  for (const sym of evenOps) {
    assertEquals(OPERATORS[sym].parity, 'even');
    assertEquals(OPERATORS[sym].sign, +1);
  }
  for (const sym of oddOps) {
    assertEquals(OPERATORS[sym].parity, 'odd');
    assertEquals(OPERATORS[sym].sign, -1);
  }
});

// ============================================================================
// INVERTIBILITY TESTS
// ============================================================================

section('Invertibility');

test('Inverse pairs are correctly defined', () => {
  assertArrayEquals(INVERSE_PAIRS, [['^', '()'], ['+', '−'], ['×', '÷']]);
});

test('getInverse returns correct inverses', () => {
  assertEquals(getInverse('^'), '()');
  assertEquals(getInverse('()'), '^');
  assertEquals(getInverse('+'), '−');
  assertEquals(getInverse('−'), '+');
  assertEquals(getInverse('×'), '÷');
  assertEquals(getInverse('÷'), '×');
});

test('Inverse operation is symmetric', () => {
  for (const sym of SYMBOL_ORDER) {
    const inv = getInverse(sym);
    assertEquals(getInverse(inv), sym);
  }
});

test('areInverses detects inverse pairs', () => {
  assert(areInverses('^', '()'));
  assert(areInverses('()', '^'));
  assert(areInverses('+', '−'));
  assert(areInverses('×', '÷'));
  assert(!areInverses('^', '+'));
  assert(!areInverses('×', '−'));
});

// ============================================================================
// COMPOSITION TESTS (Closure Property)
// ============================================================================

section('Composition (Closure Property)');

test('Identity element () is neutral', () => {
  for (const sym of SYMBOL_ORDER) {
    assertEquals(compose('()', sym), sym, `() ∘ ${sym} should be ${sym}`);
    assertEquals(compose(sym, '()'), sym, `${sym} ∘ () should be ${sym}`);
  }
});

test('Composition always yields element in the set (closure)', () => {
  assert(verifyClosure(), 'Closure property violated');

  for (const a of SYMBOL_ORDER) {
    for (const b of SYMBOL_ORDER) {
      const result = compose(a, b);
      assert(SYMBOL_ORDER.includes(result), `${a} ∘ ${b} = ${result} not in set`);
    }
  }
});

test('Specific compositions from S3 group', () => {
  // sigma ∘ sigma = sigma^2 (mul ∘ mul = amp)
  assertEquals(compose('×', '×'), '^');

  // sigma^2 ∘ sigma = e (amp ∘ mul = grp)
  assertEquals(compose('^', '×'), '()');

  // sigma ∘ sigma^2 = e (mul ∘ amp = grp)
  assertEquals(compose('×', '^'), '()');
});

test('Transposition self-composition', () => {
  // tau1 ∘ tau1 = e (div ∘ div = grp)
  assertEquals(compose('÷', '÷'), '()');

  // tau3 ∘ tau3 = e (sub ∘ sub = grp)
  assertEquals(compose('−', '−'), '()');
});

test('Composition table is complete (36 entries)', () => {
  const table = generateCompositionTable();

  assertEquals(Object.keys(table).length, 6);
  for (const row of Object.values(table)) {
    assertEquals(Object.keys(row).length, 6);
  }
});

test('composeSequence empty gives identity', () => {
  assertEquals(composeSequence([]), '()');
});

test('composeSequence single element', () => {
  for (const sym of SYMBOL_ORDER) {
    assertEquals(composeSequence([sym]), sym);
  }
});

test('3-cycle cubed returns to identity', () => {
  // mul^3 = grp (sigma^3 = e)
  assertEquals(composeSequence(['×', '×', '×']), '()');

  // amp^3 = grp (sigma^2)^3 = e
  assertEquals(composeSequence(['^', '^', '^']), '()');
});

// ============================================================================
// DSL HANDLER INTERFACE TESTS
// ============================================================================

section('DSL Handler Interface');

test('Can register handlers for operators', () => {
  const algebra = new OperatorAlgebra();

  algebra.register('^', (x) => x * 2);
  algebra.register('+', (x) => x + 1);

  assert('^' in algebra.handlers);
  assert('+' in algebra.handlers);
});

test('Can register handlers by algebraic name', () => {
  const algebra = new OperatorAlgebra();

  algebra.registerByName('amp', (x) => x * 2);
  algebra.registerByName('add', (x) => x + 1);

  assert('^' in algebra.handlers);
  assert('+' in algebra.handlers);
});

test('Apply single operator', () => {
  const algebra = new OperatorAlgebra();
  algebra.register('^', (x) => x * 2);

  assertEquals(algebra.apply('^', 5), 10);
});

test('Apply sequence of operators', () => {
  const algebra = new OperatorAlgebra();
  algebra.register('^', (x) => x * 2);
  algebra.register('+', (x) => x + 1);
  algebra.register('×', (x) => x ** 2);

  // 3 -> amp(3)=6 -> add(6)=7 -> mul(7)=49
  assertEquals(algebra.applySequence(['^', '+', '×'], 3), 49);
});

test('Apply with undo returns inverse sequence', () => {
  const algebra = new OperatorAlgebra();
  algebra.register('^', (x) => x * 2);
  algebra.register('+', (x) => x + 1);
  algebra.register('()', (x) => x);
  algebra.register('−', (x) => x - 1);

  const { result, undoSequence } = algebra.applyWithUndo(['^', '+'], 5);

  // Result: 5 -> 10 -> 11
  assertEquals(result, 11);

  // Undo sequence: [+, ^] inverses = [-, ()]
  assertArrayEquals(undoSequence, ['−', '()']);
});

test('isComplete checks all handlers registered', () => {
  const algebra = new OperatorAlgebra();
  assert(!algebra.isComplete());

  for (const sym of SYMBOL_ORDER) {
    algebra.register(sym, (x) => x);
  }

  assert(algebra.isComplete());
});

test('missingHandlers lists unregistered operators', () => {
  const algebra = new OperatorAlgebra();
  algebra.register('^', (x) => x);
  algebra.register('+', (x) => x);

  const missing = algebra.missingHandlers();
  assert(!missing.includes('^'));
  assert(!missing.includes('+'));
  assert(missing.includes('×'));
  assert(missing.includes('()'));
  assert(missing.includes('÷'));
  assert(missing.includes('−'));
});

// ============================================================================
// UTILITY FUNCTION TESTS
// ============================================================================

section('Utility Functions');

test('operatorInfo returns complete information', () => {
  const info = operatorInfo('^');

  assertEquals(info.symbol, '^');
  assertEquals(info.name, 'amp');
  assertEquals(info.parity, 'even');
  assertEquals(info.sign, +1);
  assertEquals(info.inverse, '()');
  assertEquals(info.inverseName, 'grp');
  assertEquals(info.isConstructive, true);
  assertEquals(info.isDissipative, false);
});

test('simplifySequence reduces to single operator', () => {
  // mul^3 = grp
  assertEquals(simplifySequence(['×', '×', '×']), '()');

  // Any sequence yields valid operator
  const result = simplifySequence(['+', '−']);
  assert(SYMBOL_ORDER.includes(result));
});

test('Order of operators in the group', () => {
  // Identity has order 1
  assertEquals(orderOf('()'), 1);

  // 3-cycles have order 3
  assertEquals(orderOf('×'), 3);
  assertEquals(orderOf('^'), 3);

  // Transpositions have order 2
  assertEquals(orderOf('+'), 2);
  assertEquals(orderOf('÷'), 2);
  assertEquals(orderOf('−'), 2);
});

test('findPathToIdentity returns correct path', () => {
  // Identity needs no path
  assertArrayEquals(findPathToIdentity('()'), []);

  // 3-cycle needs 2 more of same to return
  const pathMul = findPathToIdentity('×');
  assertEquals(composeSequence(['×', ...pathMul]), '()');

  const pathAmp = findPathToIdentity('^');
  assertEquals(composeSequence(['^', ...pathAmp]), '()');

  // Transpositions are self-inverse
  const pathAdd = findPathToIdentity('+');
  assertEquals(composeSequence(['+', ...pathAdd]), '()');
});

// ============================================================================
// GROUP THEORY PROPERTY TESTS
// ============================================================================

section('Group Theory Properties');

test('Associativity: (a ∘ b) ∘ c = a ∘ (b ∘ c)', () => {
  for (const a of SYMBOL_ORDER) {
    for (const b of SYMBOL_ORDER) {
      for (const c of SYMBOL_ORDER) {
        const left = compose(compose(a, b), c);
        const right = compose(a, compose(b, c));
        assertEquals(left, right, `Associativity failed for (${a} ∘ ${b}) ∘ ${c}`);
      }
    }
  }
});

test('Unique identity element', () => {
  const identities = SYMBOL_ORDER.filter((sym) =>
    SYMBOL_ORDER.every((x) => compose(sym, x) === x && compose(x, sym) === x)
  );

  assertArrayEquals(identities, ['()']);
});

test('Each element has unique inverse', () => {
  for (const sym of SYMBOL_ORDER) {
    const inverses = SYMBOL_ORDER.filter(
      (candidate) =>
        compose(sym, candidate) === '()' && compose(candidate, sym) === '()'
    );
    assertEquals(inverses.length, 1, `Multiple inverses for ${sym}: ${inverses}`);
  }
});

// ============================================================================
// SUMMARY
// ============================================================================

console.log(`\n${'='.repeat(60)}`);
console.log(`RESULTS: ${passed} passed, ${failed} failed`);
if (errors.length > 0) {
  console.log('Failures:');
  for (const { name, error } of errors) {
    console.log(`  - ${name}: ${error}`);
  }
}
console.log('='.repeat(60));

process.exit(failed === 0 ? 0 : 1);
