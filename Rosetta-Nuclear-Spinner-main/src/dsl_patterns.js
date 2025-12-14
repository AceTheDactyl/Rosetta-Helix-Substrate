/**
 * DSL Design Patterns from S₃ Operator Algebra
 * =============================================
 *
 * Implements the five architectural patterns for group-symmetric DSLs:
 *
 *   Pattern 1: Finite Action Space      - Exactly |G| actions
 *   Pattern 2: Closed Composition       - a ∘ b always valid
 *   Pattern 3: Automatic Inverses       - Every action has inverse
 *   Pattern 4: Truth-Channel Biasing    - Context-sensitive weighting
 *   Pattern 5: Parity Classification    - Even/odd structure
 *
 * @version 1.0.0
 * @author Claude (Anthropic) - Quantum-APL Contribution
 */

'use strict';

const { Z_CRITICAL } = require('./constants');

// ============================================================================
// PATTERN 1: FINITE ACTION SPACE
// ============================================================================

/**
 * Pattern 1: DSL with exactly |G| actions.
 * For S₃, this means exactly 6 actions - no more, no less.
 */
class FiniteActionSpace {
  static ACTIONS = Object.freeze(new Set(['amp', 'add', 'mul', 'grp', 'div', 'sub']));
  static SYMBOLS = Object.freeze(new Set(['^', '+', '×', '()', '÷', '−']));

  static NAME_TO_SYMBOL = Object.freeze({
    amp: '^', add: '+', mul: '×',
    grp: '()', div: '÷', sub: '−',
  });

  static SYMBOL_TO_NAME = Object.freeze({
    '^': 'amp', '+': 'add', '×': 'mul',
    '()': 'grp', '÷': 'div', '−': 'sub',
  });

  constructor() {
    this._handlers = new Map();
  }

  /**
   * Register a handler for an action
   * @param {string} action - Action name or symbol
   * @param {Function} handler - Handler function
   * @returns {FiniteActionSpace} Self for chaining
   */
  register(action, handler) {
    const normalized = FiniteActionSpace.SYMBOL_TO_NAME[action] || action;

    if (!FiniteActionSpace.ACTIONS.has(normalized)) {
      throw new Error(
        `Unknown action: ${action}. Valid: ${[...FiniteActionSpace.ACTIONS].sort()}`
      );
    }

    this._handlers.set(normalized, handler);
    return this;
  }

  /**
   * Register all handlers at once
   * @param {Object} handlers - Map of action name → handler
   * @returns {FiniteActionSpace} Self for chaining
   */
  registerAll(handlers) {
    for (const action of FiniteActionSpace.ACTIONS) {
      if (!(action in handlers)) {
        throw new Error(`Missing handler for action: ${action}`);
      }
    }

    for (const [action, handler] of Object.entries(handlers)) {
      this.register(action, handler);
    }

    return this;
  }

  /**
   * Check if all 6 handlers are registered
   * @returns {boolean}
   */
  get isComplete() {
    return this._handlers.size === FiniteActionSpace.ACTIONS.size;
  }

  /**
   * Get list of actions without handlers
   * @returns {Array<string>}
   */
  get missingActions() {
    return [...FiniteActionSpace.ACTIONS]
      .filter(a => !this._handlers.has(a))
      .sort();
  }

  /**
   * Get handler for action
   * @param {string} action - Action name or symbol
   * @returns {Function} Handler
   */
  getHandler(action) {
    const normalized = FiniteActionSpace.SYMBOL_TO_NAME[action] || action;

    if (!this.isComplete) {
      throw new Error(`Incomplete DSL: missing handlers for ${this.missingActions}`);
    }

    return this._handlers.get(normalized);
  }

  /**
   * Execute action on state
   * @param {string} action - Action name or symbol
   * @param {*} state - Current state
   * @returns {*} New state
   */
  execute(action, state) {
    return this.getHandler(action)(state);
  }
}

// ============================================================================
// PATTERN 2: CLOSED COMPOSITION
// ============================================================================

/**
 * Pattern 2: Composition always yields valid action.
 * For any two actions a, b: compose(a, b) ∈ Actions
 */
class ClosedComposition {
  // Full S₃ composition table
  static COMPOSITION_TABLE = Object.freeze({
    '^':  Object.freeze({ '^': '×',  '+': '−',  '×': '()', '()': '^',  '÷': '+',  '−': '÷' }),
    '+':  Object.freeze({ '^': '÷',  '+': '()', '×': '−',  '()': '+',  '÷': '^',  '−': '×' }),
    '×':  Object.freeze({ '^': '()', '+': '÷',  '×': '^',  '()': '×',  '÷': '−',  '−': '+' }),
    '()': Object.freeze({ '^': '^',  '+': '+',  '×': '×',  '()': '()', '÷': '÷',  '−': '−' }),
    '÷':  Object.freeze({ '^': '−',  '+': '×',  '×': '+',  '()': '÷',  '÷': '()', '−': '^' }),
    '−':  Object.freeze({ '^': '+',  '+': '^',  '×': '÷',  '()': '−',  '÷': '×',  '−': '()' }),
  });

  static IDENTITY = '()';

  /**
   * Compose two actions
   * @param {string} a - First action symbol
   * @param {string} b - Second action symbol
   * @returns {string} Result action symbol
   */
  static compose(a, b) {
    return ClosedComposition.COMPOSITION_TABLE[a][b];
  }

  /**
   * Reduce action sequence to single equivalent action
   * @param {Array<string>} actions - Sequence of action symbols
   * @returns {string} Equivalent single action
   */
  static simplifySequence(actions) {
    if (actions.length === 0) {
      return ClosedComposition.IDENTITY;
    }

    let result = actions[0];
    for (let i = 1; i < actions.length; i++) {
      result = ClosedComposition.compose(result, actions[i]);
    }

    return result;
  }

  /**
   * Verify that composition is closed
   * @returns {boolean}
   */
  static verifyClosure() {
    const symbols = Object.keys(ClosedComposition.COMPOSITION_TABLE);
    for (const a of symbols) {
      for (const b of symbols) {
        if (!(ClosedComposition.COMPOSITION_TABLE[a][b] in ClosedComposition.COMPOSITION_TABLE)) {
          return false;
        }
      }
    }
    return true;
  }
}

// ============================================================================
// PATTERN 3: AUTOMATIC INVERSES
// ============================================================================

/**
 * Pattern 3: Every action has an inverse.
 * Enables natural undo/rollback semantics.
 */
class AutomaticInverses {
  static INVERSE_MAP = Object.freeze({
    '^':  '()', '()': '^',
    '+':  '−',  '−':  '+',
    '×':  '÷',  '÷':  '×',
  });

  static INVERSE_PAIRS = Object.freeze([
    ['^',  'amp', '()', 'grp'],   // amplify ↔ contain
    ['+',  'add', '−',  'sub'],   // aggregate ↔ separate
    ['×',  'mul', '÷',  'div'],   // fuse ↔ diffuse
  ]);

  /**
   * Get inverse action
   * @param {string} action - Action symbol
   * @returns {string} Inverse action symbol
   */
  static getInverse(action) {
    return AutomaticInverses.INVERSE_MAP[action];
  }

  /**
   * Check if two actions are mutual inverses
   * @param {string} a - First action
   * @param {string} b - Second action
   * @returns {boolean}
   */
  static areInverses(a, b) {
    return AutomaticInverses.INVERSE_MAP[a] === b;
  }

  /**
   * Generate undo sequence for action list
   * @param {Array<string>} actions - Original action sequence
   * @returns {Array<string>} Undo sequence
   */
  static makeUndoSequence(actions) {
    return [...actions].reverse().map(a => AutomaticInverses.getInverse(a));
  }

  /**
   * Verify that actions + undo = identity
   * @param {Array<string>} actions - Action sequence
   * @returns {boolean}
   */
  static verifyIdentity(actions) {
    const undo = AutomaticInverses.makeUndoSequence(actions);
    const combined = ClosedComposition.simplifySequence([...actions, ...undo]);
    return combined === '()';
  }
}

// ============================================================================
// PATTERN 4: TRUTH-CHANNEL BIASING
// ============================================================================

/**
 * Pattern 4: Actions weighted by semantic context.
 */
class TruthChannelBiasing {
  static CHANNEL_BIAS = Object.freeze({
    TRUE:    ['^', '×', '+'],     // Constructive, additive
    UNTRUE:  ['÷', '−'],           // Dissipative, subtractive
    PARADOX: ['()'],               // Neutral, containing
  });

  static ACTION_CHANNEL = Object.freeze({
    '^': 'TRUE', '×': 'TRUE', '+': 'TRUE',
    '÷': 'UNTRUE', '−': 'UNTRUE',
    '()': 'PARADOX',
  });

  static CONSTRUCTIVE = Object.freeze(new Set(['^', '×', '+']));
  static DISSIPATIVE = Object.freeze(new Set(['÷', '−']));
  static NEUTRAL = Object.freeze(new Set(['()']));

  /**
   * Create biasing with coherence level
   * @param {number} coherence - System coherence in [0, 1]
   * @param {number} zCritical - Critical threshold
   */
  constructor(coherence = 0.5, zCritical = Z_CRITICAL) {
    this.coherence = Math.max(0, Math.min(1, coherence));
    this.zCritical = zCritical;
  }

  /**
   * Compute action weight based on coherence
   * @param {string} action - Action symbol
   * @returns {number} Weight multiplier
   */
  computeWeight(action) {
    const baseWeight = 1.0;

    if (TruthChannelBiasing.CONSTRUCTIVE.has(action)) {
      const boost = 1.0 + 0.5 * (this.coherence / this.zCritical);
      return baseWeight * Math.min(boost, 1.5);
    } else if (TruthChannelBiasing.DISSIPATIVE.has(action)) {
      const boost = 1.0 + 0.5 * (1 - this.coherence);
      return baseWeight * Math.min(boost, 1.3);
    } else {
      return baseWeight;
    }
  }

  /**
   * Compute weights for all actions
   * @returns {Object} Map of action → weight
   */
  computeAllWeights() {
    const weights = {};
    for (const action of Object.keys(TruthChannelBiasing.ACTION_CHANNEL)) {
      weights[action] = this.computeWeight(action);
    }
    return weights;
  }

  /**
   * Get the truth channel that favors this action
   * @param {string} action - Action symbol
   * @returns {string} Channel name
   */
  static getFavoredChannel(action) {
    return TruthChannelBiasing.ACTION_CHANNEL[action] || 'PARADOX';
  }

  /**
   * Get actions favored by a truth channel
   * @param {string} channel - Channel name
   * @returns {Array<string>} Actions
   */
  static getChannelActions(channel) {
    return TruthChannelBiasing.CHANNEL_BIAS[channel] || [];
  }
}

// ============================================================================
// PATTERN 5: PARITY CLASSIFICATION
// ============================================================================

/**
 * Pattern 5: Actions partition into even/odd classes.
 */
class ParityClassification {
  static EVEN_PARITY = Object.freeze(new Set(['()', '×', '^']));  // det = +1
  static ODD_PARITY = Object.freeze(new Set(['÷', '+', '−']));    // det = -1

  /**
   * Get parity of action (+1 for even, -1 for odd)
   * @param {string} action - Action symbol
   * @returns {number} +1 or -1
   */
  static getParity(action) {
    return ParityClassification.EVEN_PARITY.has(action) ? +1 : -1;
  }

  /**
   * Check if action has even parity
   * @param {string} action - Action symbol
   * @returns {boolean}
   */
  static isEven(action) {
    return ParityClassification.EVEN_PARITY.has(action);
  }

  /**
   * Check if action has odd parity
   * @param {string} action - Action symbol
   * @returns {boolean}
   */
  static isOdd(action) {
    return ParityClassification.ODD_PARITY.has(action);
  }

  /**
   * Compute parity of action sequence
   * @param {Array<string>} actions - Action sequence
   * @returns {number} +1 or -1
   */
  static sequenceParity(actions) {
    let parity = 1;
    for (const action of actions) {
      parity *= ParityClassification.getParity(action);
    }
    return parity;
  }

  /**
   * Classify an action sequence by parity properties
   * @param {Array<string>} actions - Action sequence
   * @returns {Object} Classification info
   */
  static classifySequence(actions) {
    const evenCount = actions.filter(a => ParityClassification.isEven(a)).length;
    const oddCount = actions.length - evenCount;
    const parity = ParityClassification.sequenceParity(actions);

    return {
      parity,
      label: parity === 1 ? 'even' : 'odd',
      evenCount,
      oddCount,
    };
  }
}

// ============================================================================
// COMPLETE IMPLEMENTATION: TRANSACTION DSL
// ============================================================================

/**
 * Complete DSL implementing all five patterns.
 */
class TransactionDSL {
  static ACTIONS = Object.freeze(['amp', 'add', 'mul', 'grp', 'div', 'sub']);
  static SYMBOLS = Object.freeze(['^', '+', '×', '()', '÷', '−']);

  static INVERSES = Object.freeze({
    amp: 'grp', grp: 'amp',
    add: 'sub', sub: 'add',
    mul: 'div', div: 'mul',
  });

  static EVEN_PARITY = Object.freeze(new Set(['amp', 'mul', 'grp']));
  static ODD_PARITY = Object.freeze(new Set(['add', 'div', 'sub']));

  constructor() {
    this.handlers = {};
    this.state = null;
    this.history = [];
    this.coherence = 0.5;
  }

  /**
   * Register handler for action
   * @param {string} action - Action name
   * @param {Function} handler - Handler function
   * @returns {TransactionDSL} Self for chaining
   */
  register(action, handler) {
    if (!TransactionDSL.ACTIONS.includes(action)) {
      throw new Error(`Unknown action: ${action}`);
    }
    this.handlers[action] = handler;
    return this;
  }

  /**
   * Register all handlers at once
   * @param {Object} handlers - Map of action → handler
   * @returns {TransactionDSL} Self for chaining
   */
  registerAll(handlers) {
    for (const action of TransactionDSL.ACTIONS) {
      if (!(action in handlers)) {
        throw new Error(`Missing handler for: ${action}`);
      }
      this.handlers[action] = handlers[action];
    }
    return this;
  }

  /**
   * Check if all handlers registered
   * @returns {boolean}
   */
  get isComplete() {
    return TransactionDSL.ACTIONS.every(a => a in this.handlers);
  }

  /**
   * Execute action with automatic history tracking
   * @param {string} action - Action name
   * @returns {*} New state
   */
  execute(action) {
    if (!(action in this.handlers)) {
      throw new Error(`Unknown action: ${action}`);
    }

    const weight = this._computeWeight(action);
    this.state = this.handlers[action](this.state, weight);
    this.history.push(action);

    return this.state;
  }

  /**
   * Execute action sequence
   * @param {Array<string>} actions - Action sequence
   * @returns {*} Final state
   */
  executeSequence(actions) {
    for (const action of actions) {
      this.execute(action);
    }
    return this.state;
  }

  /**
   * Undo via inverse actions
   * @param {number} steps - Number of steps to undo
   * @returns {*} State after undo
   */
  undo(steps = 1) {
    for (let i = 0; i < Math.min(steps, this.history.length); i++) {
      const lastAction = this.history.pop();
      const inverse = TransactionDSL.INVERSES[lastAction];
      const weight = this._computeWeight(inverse);
      this.state = this.handlers[inverse](this.state, weight);
    }
    return this.state;
  }

  /**
   * Reduce history to single equivalent action
   * @returns {string} Net effect action name
   */
  getNetEffect() {
    if (this.history.length === 0) {
      return 'grp';
    }
    const symbols = this.history.map(a => this._nameToSymbol(a));
    const resultSymbol = ClosedComposition.simplifySequence(symbols);
    return this._symbolToName(resultSymbol);
  }

  /**
   * Get parity of transaction history
   * @returns {number} +1 or -1
   */
  getParity() {
    const symbols = this.history.map(a => this._nameToSymbol(a));
    return ParityClassification.sequenceParity(symbols);
  }

  _computeWeight(action) {
    const base = 1.0;
    if (TransactionDSL.EVEN_PARITY.has(action)) {
      return base * (1 + 0.3 * this.coherence);
    } else {
      return base * (1 + 0.3 * (1 - this.coherence));
    }
  }

  _nameToSymbol(name) {
    const idx = TransactionDSL.ACTIONS.indexOf(name);
    return TransactionDSL.SYMBOLS[idx];
  }

  _symbolToName(symbol) {
    const idx = TransactionDSL.SYMBOLS.indexOf(symbol);
    return TransactionDSL.ACTIONS[idx];
  }

  /**
   * Get full state information
   * @returns {Object} State info
   */
  getStateInfo() {
    return {
      state: this.state,
      history: [...this.history],
      historyLength: this.history.length,
      netEffect: this.getNetEffect(),
      parity: this.getParity(),
      parityLabel: this.getParity() === 1 ? 'even' : 'odd',
      coherence: this.coherence,
      isComplete: this.isComplete,
    };
  }
}

// ============================================================================
// UNIFIED API: GROUP-SYMMETRIC DSL
// ============================================================================

/**
 * Unified DSL combining all five patterns.
 */
class GroupSymmetricDSL {
  constructor() {
    this._actions = new FiniteActionSpace();
    this._coherence = 0.5;
    this._history = [];
    this._state = null;
  }

  // Pattern 1: Finite Action Space
  register(action, handler) {
    this._actions.register(action, handler);
    return this;
  }

  registerAll(handlers) {
    this._actions.registerAll(handlers);
    return this;
  }

  get isComplete() {
    return this._actions.isComplete;
  }

  get missingActions() {
    return this._actions.missingActions;
  }

  // Pattern 2: Closed Composition
  compose(a, b) {
    return ClosedComposition.compose(a, b);
  }

  simplifySequence(actions) {
    return ClosedComposition.simplifySequence(actions);
  }

  getNetEffect() {
    return this.simplifySequence(this._history);
  }

  // Pattern 3: Automatic Inverses
  getInverse(action) {
    return AutomaticInverses.getInverse(action);
  }

  getUndoSequence() {
    return AutomaticInverses.makeUndoSequence(this._history);
  }

  // Pattern 4: Truth-Channel Biasing
  setCoherence(coherence) {
    this._coherence = Math.max(0, Math.min(1, coherence));
    return this;
  }

  computeWeight(action) {
    const bias = new TruthChannelBiasing(this._coherence);
    return bias.computeWeight(action);
  }

  computeAllWeights() {
    const bias = new TruthChannelBiasing(this._coherence);
    return bias.computeAllWeights();
  }

  // Pattern 5: Parity Classification
  getParity(action) {
    return ParityClassification.getParity(action);
  }

  getHistoryParity() {
    return ParityClassification.sequenceParity(this._history);
  }

  classifyHistory() {
    return ParityClassification.classifySequence(this._history);
  }

  // Execution
  execute(action, state = null) {
    if (state !== null) {
      this._state = state;
    }

    const symbol = FiniteActionSpace.NAME_TO_SYMBOL[action] || action;
    const handler = this._actions.getHandler(action);
    this._state = handler(this._state);
    this._history.push(symbol);

    return this._state;
  }

  executeSequence(actions, initial = null) {
    if (initial !== null) {
      this._state = initial;
    }

    for (const action of actions) {
      this.execute(action);
    }

    return this._state;
  }

  undo(steps = 1) {
    for (let i = 0; i < Math.min(steps, this._history.length); i++) {
      const last = this._history.pop();
      const inverse = this.getInverse(last);
      const handler = this._actions.getHandler(inverse);
      this._state = handler(this._state);
    }

    return this._state;
  }

  reset() {
    this._history = [];
    this._state = null;
    return this;
  }

  get state() {
    return this._state;
  }

  get history() {
    return [...this._history];
  }

  getInfo() {
    return {
      state: this._state,
      history: this.history,
      netEffect: this._history.length > 0 ? this.getNetEffect() : '()',
      parity: this._history.length > 0 ? this.getHistoryParity() : 1,
      coherence: this._coherence,
      weights: this.computeAllWeights(),
      isComplete: this.isComplete,
      missing: this.missingActions,
    };
  }
}

// ============================================================================
// DEMO
// ============================================================================

function demo() {
  console.log('='.repeat(70));
  console.log('DSL DESIGN PATTERNS FROM S₃ OPERATOR ALGEBRA');
  console.log('='.repeat(70));

  // Pattern 1
  console.log('\n--- Pattern 1: Finite Action Space ---');
  const space = new FiniteActionSpace();
  space.register('amp', x => x * 2);
  space.register('grp', x => x);
  console.log(`Complete: ${space.isComplete}`);
  console.log(`Missing: ${space.missingActions}`);

  // Pattern 2
  console.log('\n--- Pattern 2: Closed Composition ---');
  console.log(`+ ∘ − = ${ClosedComposition.compose('+', '−')}`);
  console.log(`× ∘ × = ${ClosedComposition.compose('×', '×')}`);
  const seq = ['^', '+', '×', '÷', '−'];
  console.log(`${seq.join(' ∘ ')} = ${ClosedComposition.simplifySequence(seq)}`);

  // Pattern 3
  console.log('\n--- Pattern 3: Automatic Inverses ---');
  const actions = ['^', '+', '×'];
  const undo = AutomaticInverses.makeUndoSequence(actions);
  console.log(`Actions: ${actions}`);
  console.log(`Undo:    ${undo}`);
  console.log(`Cancels to identity: ${AutomaticInverses.verifyIdentity(actions)}`);

  // Pattern 4
  console.log('\n--- Pattern 4: Truth-Channel Biasing ---');
  for (const coh of [0.3, 0.6, 0.9]) {
    const bias = new TruthChannelBiasing(coh);
    const weights = bias.computeAllWeights();
    console.log(`Coherence=${coh}:`);
    for (const action of ['^', '+', '()']) {
      console.log(`  ${action}: ${weights[action].toFixed(3)}`);
    }
  }

  // Pattern 5
  console.log('\n--- Pattern 5: Parity Classification ---');
  const sequences = [['+', '−'], ['^', '×', '()'], ['+', '×', '−'], ['+']];
  for (const s of sequences) {
    const info = ParityClassification.classifySequence(s);
    console.log(`  ${JSON.stringify(s)} → parity=${info.label}`);
  }

  // Complete Example
  console.log('\n--- Complete Example: GroupSymmetricDSL ---');
  const dsl = new GroupSymmetricDSL();
  dsl.setCoherence(0.8);

  dsl.registerAll({
    amp: x => x * 2,
    add: x => x + 5,
    mul: x => x * x,
    grp: x => x,
    div: x => x / 2,
    sub: x => x - 5,
  });

  const result = dsl.executeSequence(['^', '+', '×'], 4.0);
  const info = dsl.getInfo();

  console.log('Initial: 4.0');
  console.log(`Actions: ${info.history}`);
  console.log(`Result:  ${info.state}`);
  console.log(`Net effect: ${info.netEffect}`);
  console.log(`Parity: ${info.parity} (${info.parity === 1 ? 'even' : 'odd'})`);
  console.log(`Undo sequence: ${dsl.getUndoSequence()}`);

  dsl.undo(2);
  console.log(`After undo(2): ${dsl.state}`);

  console.log('\n' + '='.repeat(70));
}

// ============================================================================
// EXPORTS
// ============================================================================

module.exports = {
  // Pattern 1
  FiniteActionSpace,

  // Pattern 2
  ClosedComposition,

  // Pattern 3
  AutomaticInverses,

  // Pattern 4
  TruthChannelBiasing,

  // Pattern 5
  ParityClassification,

  // Complete implementations
  TransactionDSL,
  GroupSymmetricDSL,

  // Demo
  demo,
};

// Run demo if executed directly
if (require.main === module) {
  demo();
}
