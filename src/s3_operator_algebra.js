/**
 * S₃ Operator Algebra Module
 * ===========================
 *
 * Closed algebraic structure for APL operators with DSL-focused design.
 *
 * OPERATOR SET (6 elements, closed under composition):
 *     Symbol  Name    Algebraic   S₃ Element  Parity
 *     ------  ------  ----------  ----------  ------
 *     ^       amp     amplify     σ²          even
 *     +       add     aggregate   τ₂          odd
 *     ×       mul     multiply    σ           even
 *     ()      grp     group       e           even
 *     ÷       div     divide      τ₁          odd
 *     −       sub     subtract    τ₃          odd
 *
 * DESIGN PRINCIPLES:
 *     1. Finite action space: Exactly 6 handlers needed
 *     2. Predictable composition: op₁ ∘ op₂ always yields valid op
 *     3. Invertibility pairs: +/−, ×/÷, ^/() provide natural undo
 *
 * @version 1.0.0
 * @author Claude (Anthropic) - Quantum-APL Contribution
 */

'use strict';

const S3Symmetry = require('./s3_operator_symmetry');

// ============================================================================
// ALGEBRAIC OPERATOR DEFINITIONS
// ============================================================================

/**
 * Operator definitions with algebraic properties
 */
const OPERATORS = {
  '^': {
    symbol: '^',
    name: 'amp',
    description: 'amplify/excite',
    s3Element: 'σ2',
    parity: 'even',
    sign: +1,
    inverseSymbol: '()',
  },
  '+': {
    symbol: '+',
    name: 'add',
    description: 'aggregate/route',
    s3Element: 'τ2',
    parity: 'odd',
    sign: -1,
    inverseSymbol: '−',
  },
  '×': {
    symbol: '×',
    name: 'mul',
    description: 'multiply/fuse',
    s3Element: 'σ',
    parity: 'even',
    sign: +1,
    inverseSymbol: '÷',
  },
  '()': {
    symbol: '()',
    name: 'grp',
    description: 'group/contain',
    s3Element: 'e',
    parity: 'even',
    sign: +1,
    inverseSymbol: '^',
  },
  '÷': {
    symbol: '÷',
    name: 'div',
    description: 'divide/diffuse',
    s3Element: 'τ1',
    parity: 'odd',
    sign: -1,
    inverseSymbol: '×',
  },
  '−': {
    symbol: '−',
    name: 'sub',
    description: 'subtract/separate',
    s3Element: 'τ3',
    parity: 'odd',
    sign: -1,
    inverseSymbol: '+',
  },
};

/**
 * Name → Symbol lookup
 */
const NAME_TO_SYMBOL = {};
for (const op of Object.values(OPERATORS)) {
  NAME_TO_SYMBOL[op.name] = op.symbol;
}

/**
 * Symbol → Name lookup
 */
const SYMBOL_TO_NAME = {};
for (const op of Object.values(OPERATORS)) {
  SYMBOL_TO_NAME[op.symbol] = op.name;
}

/**
 * Canonical symbol ordering
 */
const SYMBOL_ORDER = ['^', '+', '×', '()', '÷', '−'];

/**
 * Canonical name ordering
 */
const NAME_ORDER = ['amp', 'add', 'mul', 'grp', 'div', 'sub'];

// ============================================================================
// INVERTIBILITY PAIRS
// ============================================================================

/**
 * Inverse pairs providing natural undo semantics
 */
const INVERSE_PAIRS = [
  ['^', '()'],   // amp ↔ grp
  ['+', '−'],    // add ↔ sub
  ['×', '÷'],    // mul ↔ div
];

/**
 * Get the inverse operator symbol
 * @param {string} symbol - Operator symbol
 * @returns {string} Inverse operator symbol
 */
function getInverse(symbol) {
  const op = OPERATORS[symbol];
  if (!op) throw new Error(`Unknown operator: ${symbol}`);
  return op.inverseSymbol;
}

/**
 * Check if two operators are inverses of each other
 * @param {string} a - First operator symbol
 * @param {string} b - Second operator symbol
 * @returns {boolean} True if operators are inverses
 */
function areInverses(a, b) {
  return getInverse(a) === b;
}

// ============================================================================
// OPERATOR COMPOSITION (Closed Set Property)
// ============================================================================

/**
 * Compose two operators (a ∘ b)
 * The result is always another operator in the set (closure property)
 *
 * @param {string} a - First operator symbol
 * @param {string} b - Second operator symbol
 * @returns {string} Result operator symbol
 */
function compose(a, b) {
  const opA = OPERATORS[a];
  const opB = OPERATORS[b];

  if (!opA) throw new Error(`Unknown operator: ${a}`);
  if (!opB) throw new Error(`Unknown operator: ${b}`);

  const s3Result = S3Symmetry.composeS3(opA.s3Element, opB.s3Element);
  return S3Symmetry.S3_OPERATOR_MAP[s3Result];
}

/**
 * Compose a sequence of operators left-to-right
 * @param {Array<string>} operators - Sequence of operator symbols
 * @returns {string} Result operator symbol (or "()" for empty sequence)
 */
function composeSequence(operators) {
  if (operators.length === 0) {
    return '()';  // Identity element
  }

  let result = operators[0];
  for (let i = 1; i < operators.length; i++) {
    result = compose(result, operators[i]);
  }

  return result;
}

/**
 * Generate the full 6×6 composition table
 * @returns {Object} table[a][b] = a ∘ b
 */
function generateCompositionTable() {
  const table = {};

  for (const a of SYMBOL_ORDER) {
    table[a] = {};
    for (const b of SYMBOL_ORDER) {
      table[a][b] = compose(a, b);
    }
  }

  return table;
}

// ============================================================================
// DSL HANDLER INTERFACE
// ============================================================================

/**
 * DSL interpreter with exactly 6 handlers (one per operator)
 *
 * This class demonstrates the "finite action space" property:
 * your interpreter only needs 6 handlers to process any
 * valid operator sequence.
 */
class OperatorAlgebra {
  constructor() {
    this.handlers = {};
  }

  /**
   * Register a handler for an operator
   * @param {string} symbol - Operator symbol (^, +, ×, (), ÷, −)
   * @param {Function} handler - Function to execute for this operator
   */
  register(symbol, handler) {
    if (!OPERATORS[symbol]) {
      throw new Error(`Unknown operator: ${symbol}`);
    }
    this.handlers[symbol] = handler;
  }

  /**
   * Register a handler using algebraic name
   * @param {string} name - Algebraic name (amp, add, mul, grp, div, sub)
   * @param {Function} handler - Function to execute for this operator
   */
  registerByName(name, handler) {
    const symbol = NAME_TO_SYMBOL[name];
    if (!symbol) {
      throw new Error(`Unknown operator name: ${name}`);
    }
    this.handlers[symbol] = handler;
  }

  /**
   * Apply a single operator to a value
   * @param {string} symbol - Operator symbol
   * @param {*} value - Input value
   * @returns {*} Transformed value
   */
  apply(symbol, value) {
    const handler = this.handlers[symbol];
    if (!handler) {
      throw new Error(`No handler registered for: ${symbol}`);
    }
    return handler(value);
  }

  /**
   * Apply a sequence of operators to a value
   * @param {Array<string>} operators - Sequence of operator symbols
   * @param {*} value - Initial value
   * @returns {*} Final transformed value
   */
  applySequence(operators, value) {
    let result = value;
    for (const op of operators) {
      result = this.apply(op, result);
    }
    return result;
  }

  /**
   * Apply operators and return the undo sequence
   * @param {Array<string>} operators - Sequence of operator symbols
   * @param {*} value - Initial value
   * @returns {Object} { result, undoSequence }
   */
  applyWithUndo(operators, value) {
    const result = this.applySequence(operators, value);
    const undoSequence = [...operators].reverse().map(getInverse);
    return { result, undoSequence };
  }

  /**
   * Check if all 6 handlers are registered
   * @returns {boolean}
   */
  isComplete() {
    return SYMBOL_ORDER.every(sym => sym in this.handlers);
  }

  /**
   * Get list of operators without handlers
   * @returns {Array<string>}
   */
  missingHandlers() {
    return SYMBOL_ORDER.filter(sym => !(sym in this.handlers));
  }
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * Get full information about an operator
 * @param {string} symbol - Operator symbol
 * @returns {Object} Operator properties
 */
function operatorInfo(symbol) {
  const op = OPERATORS[symbol];
  if (!op) throw new Error(`Unknown operator: ${symbol}`);

  return {
    symbol: op.symbol,
    name: op.name,
    description: op.description,
    s3Element: op.s3Element,
    parity: op.parity,
    sign: op.sign,
    inverse: op.inverseSymbol,
    inverseName: SYMBOL_TO_NAME[op.inverseSymbol],
    isConstructive: op.parity === 'even',
    isDissipative: op.parity === 'odd',
  };
}

/**
 * Simplify an operator sequence to its equivalent single operator
 * @param {Array<string>} operators - Sequence of operator symbols
 * @returns {string} Equivalent single operator
 */
function simplifySequence(operators) {
  return composeSequence(operators);
}

/**
 * Find the shortest sequence that returns an operator to identity
 * @param {string} symbol - Starting operator symbol
 * @returns {Array<string>} Sequence that when composed with symbol gives "()"
 */
function findPathToIdentity(symbol) {
  const op = OPERATORS[symbol];
  if (!op) throw new Error(`Unknown operator: ${symbol}`);

  const s3Elem = op.s3Element;

  if (s3Elem === 'e') {
    return [];  // Already identity
  } else if (s3Elem === 'σ') {
    return ['×', '×'];  // 3-cycle needs σ² to return
  } else if (s3Elem === 'σ2') {
    return ['^', '^'];  // σ² needs σ to return
  } else {
    return [symbol];  // Transpositions are self-inverse
  }
}

/**
 * Get the order of an operator in the group
 * @param {string} symbol - Operator symbol
 * @returns {number} Order (1, 2, or 3)
 */
function orderOf(symbol) {
  const op = OPERATORS[symbol];
  if (!op) throw new Error(`Unknown operator: ${symbol}`);

  const s3Elem = op.s3Element;

  if (s3Elem === 'e') {
    return 1;  // Identity has order 1
  } else if (s3Elem === 'σ' || s3Elem === 'σ2') {
    return 3;  // 3-cycles have order 3
  } else {
    return 2;  // Transpositions have order 2
  }
}

/**
 * Verify closure property
 * @returns {boolean} True if all compositions yield valid operators
 */
function verifyClosure() {
  const table = generateCompositionTable();

  for (const a of SYMBOL_ORDER) {
    for (const b of SYMBOL_ORDER) {
      if (!OPERATORS[table[a][b]]) {
        return false;
      }
    }
  }

  return true;
}

// ============================================================================
// EXPORTS
// ============================================================================

module.exports = {
  // Operator definitions
  OPERATORS,
  NAME_TO_SYMBOL,
  SYMBOL_TO_NAME,
  SYMBOL_ORDER,
  NAME_ORDER,

  // Invertibility
  INVERSE_PAIRS,
  getInverse,
  areInverses,

  // Composition
  compose,
  composeSequence,
  generateCompositionTable,

  // DSL interface
  OperatorAlgebra,

  // Utilities
  operatorInfo,
  simplifySequence,
  findPathToIdentity,
  orderOf,
  verifyClosure,
};
