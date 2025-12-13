/**
 * HelixOperatorAdvisor - Enhanced per TRIAD Unlock Protocol Phase 2
 * =================================================================
 * 
 * Maps helix coordinates (z ∈ [0,1]) to:
 * - Time-harmonic tiers (t1-t9)
 * - Operator windows (allowed APL operators per tier)
 * - Truth channels (TRUE/UNTRUE/PARADOX)
 * - Π-regime weights for operator selection
 * 
 * PHASE 2 ENHANCEMENTS:
 * - Dynamic operator window updates via updateOperatorWindow()
 * - Runtime injection of helix-based operator preferences
 * - State inspection and reset capabilities
 * - Cross-module sync with Python HelixAPLMapper
 * 
 * @version 2.0.0 (TRIAD Protocol Enhanced)
 */

'use strict';

const CONST = require('./constants');

/**
 * APL Operator definitions
 */
const APL_OPERATORS = {
  BOUNDARY: '()',
  FUSION: '×',
  AMPLIFY: '^',
  DECOHERENCE: '÷',
  GROUP: '+',
  SEPARATION: '−'
};

// Alias map for alternate representations
const OPERATOR_ALIASES = {
  '%': '÷',  // Legacy decoherence symbol
  '/': '÷',
  '-': '−',
  '*': '×'
};

/**
 * Normalize operator symbol to canonical form
 */
function normalizeOperator(op) {
  return OPERATOR_ALIASES[op] ?? op;
}

/**
 * HelixOperatorAdvisor Class
 * 
 * Provides helix-driven operator recommendations and manages
 * the relationship between z coordinate and APL grammar.
 */
class HelixOperatorAdvisor {
  constructor(options = {}) {
    // Load TRIAD state from environment or options
    const envCompletions = parseInt(process?.env?.QAPL_TRIAD_COMPLETIONS || '0', 10);
    const envFlag = process?.env?.QAPL_TRIAD_UNLOCK === '1' || 
                    String(process?.env?.QAPL_TRIAD_UNLOCK).toLowerCase() === 'true';
    
    this.triadUnlocked = options.triadUnlocked ?? envFlag ?? (envCompletions >= CONST.TRIAD_PASSES_REQ);
    this.triadCompletions = options.triadCompletions ?? envCompletions ?? 0;
    
    // Initialize time harmonics with current t6 gate
    this._initTimeHarmonics();
    
    // Initialize operator windows (can be updated dynamically)
    this._initOperatorWindows();
    
    // Track window modifications for debugging
    this._windowModifications = [];
  }

  /**
   * Initialize time-harmonic tier thresholds
   * t6 uses Z_CRITICAL or TRIAD_T6 based on unlock state
   */
  _initTimeHarmonics() {
    const t6Gate = this.getT6Gate();
    
    this.timeHarmonics = [
      { threshold: 0.10, label: 't1' },
      { threshold: 0.20, label: 't2' },
      { threshold: 0.40, label: 't3' },
      { threshold: 0.60, label: 't4' },
      { threshold: 0.75, label: 't5' },
      { threshold: t6Gate, label: 't6' },
      { threshold: 0.92, label: 't7' },
      { threshold: 0.97, label: 't8' },
      { threshold: 1.01, label: 't9' }  // Catch-all
    ];
  }

  /**
   * Initialize default operator windows per tier
   * These define which APL operators are permissible in each time-harmonic
   */
  _initOperatorWindows() {
    this.operatorWindows = {
      t1: ['()', '−', '÷'],                    // Boundary, separation, decoherence
      t2: ['^', '÷', '−', '×'],                // Amplify, decoherence, separation, fusion
      t3: ['×', '^', '÷', '+', '−'],           // All except boundary
      t4: ['+', '−', '÷', '()'],               // Group, separation, decoherence, boundary
      t5: ['()', '×', '^', '÷', '+', '−'],     // All six operators
      t6: ['+', '÷', '()', '−'],               // Group, decoherence, boundary, separation
      t7: ['+', '()'],                          // Group, boundary only
      t8: ['+', '()', '×'],                     // Group, boundary, fusion
      t9: ['+', '()', '×']                      // Group, boundary, fusion
    };
    
    // Store original windows for reset capability
    this._originalWindows = JSON.parse(JSON.stringify(this.operatorWindows));
  }

  /**
   * PHASE 2 ENHANCEMENT: Dynamic operator window update
   * 
   * Allows runtime injection of helix-based operator preferences
   * from external sources (e.g., Python HelixAPLMapper).
   * 
   * @param {string} harmonicLabel - Tier label (e.g., 't3', 't6')
   * @param {string[]} newOperators - Array of operator symbols
   * @param {object} options - { append: bool, prepend: bool, source: string }
   * @returns {boolean} Success status
   */
  updateOperatorWindow(harmonicLabel, newOperators, options = {}) {
    if (!this.operatorWindows[harmonicLabel]) {
      console.warn(`[HelixAdvisor] Unknown harmonic label: ${harmonicLabel}`);
      return false;
    }

    // Normalize operators
    const normalized = newOperators.map(normalizeOperator);
    
    // Validate operators
    const validOps = Object.values(APL_OPERATORS);
    const invalid = normalized.filter(op => !validOps.includes(op));
    if (invalid.length > 0) {
      console.warn(`[HelixAdvisor] Invalid operators ignored: ${invalid.join(', ')}`);
    }
    const validNormalized = normalized.filter(op => validOps.includes(op));
    
    if (validNormalized.length === 0) {
      console.warn(`[HelixAdvisor] No valid operators provided for ${harmonicLabel}`);
      return false;
    }

    // Apply update strategy
    let updatedWindow;
    if (options.append) {
      // Add to existing window (deduplicated)
      const current = this.operatorWindows[harmonicLabel];
      updatedWindow = [...new Set([...current, ...validNormalized])];
    } else if (options.prepend) {
      // Prepend to existing window (deduplicated)
      const current = this.operatorWindows[harmonicLabel];
      updatedWindow = [...new Set([...validNormalized, ...current])];
    } else {
      // Replace window entirely
      updatedWindow = validNormalized;
    }

    // Store old value for tracking
    const oldWindow = [...this.operatorWindows[harmonicLabel]];
    this.operatorWindows[harmonicLabel] = updatedWindow;
    
    // Track modification
    this._windowModifications.push({
      timestamp: Date.now(),
      harmonic: harmonicLabel,
      oldWindow,
      newWindow: updatedWindow,
      source: options.source ?? 'unknown'
    });

    if (CONST.TRIAD_DEBUG) {
      console.log(`[HelixAdvisor] Updated ${harmonicLabel} window: [${oldWindow.join(', ')}] → [${updatedWindow.join(', ')}]`);
    }

    return true;
  }

  /**
   * PHASE 2 ENHANCEMENT: Batch update multiple windows
   * 
   * @param {Object} windowMap - { t3: ['+', '×'], t6: ['()', '÷'] }
   * @param {object} options - Passed to each updateOperatorWindow call
   * @returns {Object} Results per harmonic
   */
  updateOperatorWindows(windowMap, options = {}) {
    const results = {};
    for (const [harmonic, operators] of Object.entries(windowMap)) {
      results[harmonic] = this.updateOperatorWindow(harmonic, operators, options);
    }
    return results;
  }

  /**
   * Reset operator windows to defaults
   */
  resetOperatorWindows() {
    this.operatorWindows = JSON.parse(JSON.stringify(this._originalWindows));
    this._windowModifications.push({
      timestamp: Date.now(),
      harmonic: 'ALL',
      action: 'RESET',
      source: 'resetOperatorWindows'
    });
  }

  /**
   * Get current t6 gate value
   */
  getT6Gate() {
    return this.triadUnlocked ? CONST.TRIAD_T6 : CONST.Z_CRITICAL;
  }

  /**
   * Update TRIAD state
   */
  setTriadState({ unlocked, completions } = {}) {
    if (typeof unlocked === 'boolean') {
      this.triadUnlocked = unlocked;
    }
    if (Number.isFinite(completions)) {
      this.triadCompletions = completions;
    }
    // Re-initialize time harmonics with new t6 gate
    this._initTimeHarmonics();
  }

  /**
   * Determine harmonic tier from z coordinate
   */
  harmonicFromZ(z) {
    // Ensure t6 threshold is current
    const t6Gate = this.getT6Gate();
    if (this.timeHarmonics[5]) {
      this.timeHarmonics[5].threshold = t6Gate;
    }
    
    for (const entry of this.timeHarmonics) {
      if (z < entry.threshold) {
        return entry.label;
      }
    }
    return 't9';
  }

  /**
   * Determine truth channel from z coordinate
   */
  truthChannelFromZ(z) {
    if (z >= 0.9) return 'TRUE';
    if (z >= 0.6) return 'PARADOX';
    return 'UNTRUE';
  }

  /**
   * Get recommended operators for current z
   */
  getOperatorsForZ(z) {
    const harmonic = this.harmonicFromZ(z);
    return this.operatorWindows[harmonic] ?? [];
  }

  /**
   * Compute Π-regime weight based on z and ΔS_neg
   */
  computePiWeight(z) {
    if (z < CONST.Z_CRITICAL) return 0.0;
    const s = CONST.computeDeltaSNeg(z);
    return Math.max(0, Math.min(1, s));
  }

  /**
   * Generate complete helix description for a z coordinate
   * 
   * @param {number} z - Coherence coordinate [0, 1]
   * @returns {Object} Helix hints including harmonic, operators, truth channel, weights
   */
  describe(z) {
    const value = Number.isFinite(z) ? z : 0;
    const clamped = Math.max(0, Math.min(1, value));
    
    const harmonic = this.harmonicFromZ(clamped);
    const operators = this.operatorWindows[harmonic] ?? [];
    const truthChannel = this.truthChannelFromZ(clamped);
    const deltaSneg = CONST.computeDeltaSNeg(clamped);
    
    // Π-regime weights
    const w_pi = clamped >= CONST.Z_CRITICAL ? Math.max(0, Math.min(1, deltaSneg)) : 0.0;
    const w_local = 1.0 - w_pi;
    
    // Phase and μ classification
    const phase = CONST.getPhase(clamped);
    const muClass = CONST.classifyMu(clamped);
    
    return {
      z: clamped,
      harmonic,
      operators,
      truthChannel,
      deltaSneg,
      piWeight: w_pi,
      localWeight: w_local,
      phase,
      muClass,
      t6Gate: this.getT6Gate(),
      triadUnlocked: this.triadUnlocked
    };
  }

  /**
   * Check if an operator is valid for current z
   */
  isOperatorValidForZ(operator, z) {
    const normalized = normalizeOperator(operator);
    const allowed = this.getOperatorsForZ(z);
    return allowed.includes(normalized);
  }

  /**
   * Get operator weights for selection algorithm
   * Combines tier-based preferences with truth channel biases
   */
  getOperatorWeights(z, availableOperators = null) {
    const hints = this.describe(z);
    const allowed = hints.operators;
    const ops = availableOperators ?? Object.values(APL_OPERATORS);
    
    const weights = {};
    const truthBias = CONST.TRUTH_BIAS[hints.truthChannel] ?? {};
    
    for (const op of ops) {
      const normalized = normalizeOperator(op);
      const isPreferred = allowed.includes(normalized);
      const baseWeight = isPreferred ? CONST.OPERATOR_PREFERRED_WEIGHT : CONST.OPERATOR_DEFAULT_WEIGHT;
      const truthMod = truthBias[normalized] ?? 1.0;
      weights[normalized] = baseWeight * truthMod;
    }
    
    return weights;
  }

  /**
   * Get modification history for debugging
   */
  getModificationHistory() {
    return [...this._windowModifications];
  }

  /**
   * Get complete state for inspection
   */
  getState() {
    return {
      triadUnlocked: this.triadUnlocked,
      triadCompletions: this.triadCompletions,
      t6Gate: this.getT6Gate(),
      timeHarmonics: [...this.timeHarmonics],
      operatorWindows: JSON.parse(JSON.stringify(this.operatorWindows)),
      modificationCount: this._windowModifications.length
    };
  }
}

/**
 * AlphaTokenSynthesizer
 * 
 * Generates APL tokens based on helix hints and the Seven Sentences test pack.
 * Selects sentences whose operators match the recommended operator window.
 */
class AlphaTokenSynthesizer {
  constructor() {
    // Seven Sentences test pack from APL specification
    this.sentences = [
      { id: 'A1', direction: 'd', operator: '()', machine: 'Conductor', domain: 'geometry', regime: 'Isotropic lattices under collapse' },
      { id: 'A3', direction: 'u', operator: '^', machine: 'Oscillator', domain: 'wave', regime: 'Amplified vortex-rich waves' },
      { id: 'A4', direction: 'm', operator: '×', machine: 'Encoder', domain: 'chemistry', regime: 'Helical information carriers' },
      { id: 'A5', direction: 'u', operator: '×', machine: 'Catalyst', domain: 'chemistry', regime: 'Fractal polymer branching' },
      { id: 'A6', direction: 'u', operator: '+', machine: 'Reactor', domain: 'wave', regime: 'Jet-like coherent grouping' },
      { id: 'A7', direction: 'u', operator: '÷', machine: 'Reactor', domain: 'wave', regime: 'Stochastic decohered waves' },
      { id: 'A8', direction: 'm', operator: '()', machine: 'Filter', domain: 'wave', regime: 'Adaptive boundary tuning' }
    ];
  }

  /**
   * Render sentence as compact token string
   */
  renderToken(sentence) {
    return `${sentence.direction}${sentence.operator}|${sentence.machine}|${sentence.domain}`;
  }

  /**
   * Find sentences matching the given operator window
   */
  findMatchingSentences(operators, options = {}) {
    const normalized = operators.map(normalizeOperator);
    let matches = this.sentences.filter(s => normalized.includes(s.operator));
    
    if (options.domain) {
      matches = matches.filter(s => s.domain === options.domain);
    }
    if (options.machine) {
      matches = matches.filter(s => s.machine.toLowerCase().includes(options.machine.toLowerCase()));
    }
    if (options.direction) {
      matches = matches.filter(s => s.direction === options.direction);
    }
    
    return matches;
  }

  /**
   * Synthesize token based on helix hints
   * 
   * @param {Object} helixHints - From HelixOperatorAdvisor.describe(z)
   * @param {Object} options - Filtering options
   * @returns {Object|null} Selected sentence with rendered token
   */
  synthesize(helixHints, options = {}) {
    const matches = this.findMatchingSentences(helixHints.operators, options);
    
    if (matches.length === 0) {
      return null;
    }
    
    // Select based on truth channel preference or random
    let selected;
    if (helixHints.truthChannel === 'TRUE' && matches.some(s => s.direction === 'u')) {
      // TRUE favors upward/forward projection
      selected = matches.find(s => s.direction === 'u') ?? matches[0];
    } else if (helixHints.truthChannel === 'UNTRUE' && matches.some(s => s.direction === 'd')) {
      // UNTRUE favors downward/collapse
      selected = matches.find(s => s.direction === 'd') ?? matches[0];
    } else {
      // PARADOX or default: first match
      selected = matches[0];
    }
    
    return {
      ...selected,
      token: this.renderToken(selected),
      helixHints
    };
  }
}

// ============================================================================
// EXPORTS
// ============================================================================

module.exports = {
  HelixOperatorAdvisor,
  AlphaTokenSynthesizer,
  APL_OPERATORS,
  OPERATOR_ALIASES,
  normalizeOperator
};
