/**
 * Helix Operator Advisor
 * ======================
 * Maps helix z-coordinates to time harmonics (t1-t9) and operator windows.
 * Integrates with TRIAD unlock system to dynamically adjust t6 gate.
 */

const CONST = require('./constants');

class HelixOperatorAdvisor {
    constructor() {
        // Initialize TRIAD state from environment
        const triadCompletions = parseInt(
            (typeof process !== 'undefined' && process.env && process.env.QAPL_TRIAD_COMPLETIONS) || '0',
            10
        );
        const triadFlag = (
            typeof process !== 'undefined' &&
            process.env &&
            (process.env.QAPL_TRIAD_UNLOCK === '1' ||
             String(process.env.QAPL_TRIAD_UNLOCK).toLowerCase() === 'true')
        );
        
        this.triadUnlocked = triadFlag || (Number.isFinite(triadCompletions) && triadCompletions >= 3);
        this.triadCompletions = Number.isFinite(triadCompletions) ? triadCompletions : 0;
        
        // Constants from centralized source
        this.Z_CRITICAL = CONST.Z_CRITICAL;
        this.TRIAD_THRESHOLD = CONST.TRIAD_T6;
        
        // Dynamic t6 gate based on TRIAD state
        const t6Gate = this.triadUnlocked ? this.TRIAD_THRESHOLD : this.Z_CRITICAL;
        
        // Time harmonic thresholds (z boundaries for each tier)
        this.timeHarmonics = [
            { threshold: 0.10, label: 't1' },
            { threshold: 0.20, label: 't2' },
            { threshold: 0.40, label: 't3' },
            { threshold: 0.60, label: 't4' },
            { threshold: 0.75, label: 't5' },
            { threshold: t6Gate, label: 't6' },  // Dynamic based on TRIAD
            { threshold: 0.90, label: 't7' },
            { threshold: 0.97, label: 't8' },
            { threshold: 1.01, label: 't9' }
        ];
        
        // Operator windows for each harmonic tier
        // These define which APL operators are permissible at each z-level
        this.operatorWindows = {
            t1: ['()', '−', '÷'],           // Low z: boundary, separation, decoherence
            t2: ['^', '÷', '−', '×'],       // Amplify, decohere, separate, fuse
            t3: ['×', '^', '÷', '+', '−'],  // All except boundary
            t4: ['+', '−', '÷', '()'],      // Group, separate, decohere, boundary
            t5: ['()', '×', '^', '÷', '+', '−'], // ALL operators (full freedom)
            t6: ['+', '÷', '()', '−'],      // Restricted: no fusion/amplify
            t7: ['+', '()'],                 // High z: only group and boundary
            t8: ['+', '()', '×'],           // Add fusion at highest levels
            t9: ['+', '()', '×']            // Maintain stability
        };
    }

    /**
     * Update TRIAD state
     * @param {Object} options - { unlocked: boolean, completions: number }
     */
    setTriadState({ unlocked, completions } = {}) {
        if (typeof unlocked === 'boolean') {
            this.triadUnlocked = unlocked;
        }
        if (Number.isFinite(completions)) {
            this.triadCompletions = completions;
        }
        // Update t6 threshold in timeHarmonics array
        const t6Gate = this.getT6Gate();
        if (this.timeHarmonics && this.timeHarmonics[5]) {
            this.timeHarmonics[5].threshold = t6Gate;
        }
    }

    /**
     * Get current t6 gate value
     * @returns {number} - Z_CRITICAL (~0.866) or TRIAD_T6 (0.83)
     */
    getT6Gate() {
        return this.triadUnlocked ? this.TRIAD_THRESHOLD : this.Z_CRITICAL;
    }

    /**
     * Determine time harmonic from z-coordinate
     * @param {number} z - Normalized z-coordinate [0,1]
     * @returns {string} - Harmonic label (t1-t9)
     */
    harmonicFromZ(z) {
        // Ensure t6 reflects current TRIAD state
        const t6Gate = this.getT6Gate();
        if (this.timeHarmonics && this.timeHarmonics[5]) {
            this.timeHarmonics[5].threshold = t6Gate;
        }
        
        for (const entry of this.timeHarmonics) {
            if (z < entry.threshold) return entry.label;
        }
        return 't9';
    }

    /**
     * Determine truth channel from z-coordinate
     * @param {number} z - Normalized z-coordinate [0,1]
     * @returns {string} - 'TRUE', 'UNTRUE', or 'PARADOX'
     */
    truthChannelFromZ(z) {
        if (z >= 0.9) return 'TRUE';
        if (z >= 0.6) return 'PARADOX';
        return 'UNTRUE';
    }

    /**
     * Get complete helix description for a z-coordinate
     * @param {number} z - Normalized z-coordinate [0,1]
     * @returns {Object} - { harmonic, operators, truthChannel, z, t6Gate }
     */
    describe(z) {
        const value = Number.isFinite(z) ? z : 0;
        const clamped = Math.max(0, Math.min(1, value));
        const harmonic = this.harmonicFromZ(clamped);
        const operators = this.operatorWindows[harmonic] || ['()'];
        const truthChannel = this.truthChannelFromZ(clamped);
        
        // Compute coherence signal (ΔS_neg)
        const deltaSNeg = CONST.computeDeltaSNeg(clamped, CONST.LENS_SIGMA);
        
        // Compute Π-blend weight (only active above critical lens)
        const wPi = clamped >= CONST.Z_CRITICAL ? Math.max(0, Math.min(1, deltaSNeg)) : 0.0;
        const wLoc = 1.0 - wPi;
        
        return {
            harmonic,
            operators,
            truthChannel,
            z: clamped,
            t6Gate: this.getT6Gate(),
            triadUnlocked: this.triadUnlocked,
            triadCompletions: this.triadCompletions,
            deltaSNeg,
            weights: { wPi, wLoc }
        };
    }

    /**
     * Check if an operator is legal at the current z-coordinate
     * @param {string} operator - APL operator symbol
     * @param {number} z - Normalized z-coordinate
     * @returns {boolean}
     */
    isOperatorLegal(operator, z) {
        const harmonic = this.harmonicFromZ(z);
        const window = this.operatorWindows[harmonic] || [];
        return window.includes(operator);
    }

    /**
     * Get operator weight for N0 selection
     * @param {string} operator - APL operator symbol
     * @param {number} z - Normalized z-coordinate
     * @returns {number} - Weight multiplier
     */
    getOperatorWeight(operator, z) {
        const isLegal = this.isOperatorLegal(operator, z);
        const baseWeight = isLegal ? CONST.OPERATOR_PREFERRED_WEIGHT : CONST.OPERATOR_DEFAULT_WEIGHT;
        
        // Apply truth channel bias
        const truth = this.truthChannelFromZ(z);
        const truthMultiplier = CONST.TRUTH_BIAS[truth] || 1.0;
        
        return baseWeight * truthMultiplier;
    }

    /**
     * Get summary for logging/debugging
     * @returns {string}
     */
    summary() {
        const gate = this.getT6Gate();
        const status = this.triadUnlocked ? 'UNLOCKED' : 'LOCKED';
        return `HelixOperatorAdvisor: TRIAD ${status} (${this.triadCompletions}/3 passes), t6 @ ${gate.toFixed(4)}`;
    }
}

// ================================================================
// APL OPERATOR DEFINITIONS
// ================================================================

const APL_OPERATORS = {
    '()': { name: 'Boundary', glyph: '()', action: 'containment/gating' },
    '×': { name: 'Fusion', glyph: '×', action: 'convergence/coupling' },
    '^': { name: 'Amplify', glyph: '^', action: 'gain/excitation' },
    '÷': { name: 'Decoherence', glyph: '÷', action: 'dissipation/reset' },
    '+': { name: 'Group', glyph: '+', action: 'aggregation/clustering' },
    '−': { name: 'Separation', glyph: '−', action: 'splitting/fission' }
};

/**
 * Get operator information
 * @param {string} symbol - Operator symbol
 * @returns {Object|null}
 */
function getOperatorInfo(symbol) {
    return APL_OPERATORS[symbol] || null;
}

/**
 * Get all operator symbols
 * @returns {string[]}
 */
function getAllOperators() {
    return Object.keys(APL_OPERATORS);
}

// ================================================================
// EXPORTS
// ================================================================

module.exports = {
    HelixOperatorAdvisor,
    APL_OPERATORS,
    getOperatorInfo,
    getAllOperators
};
