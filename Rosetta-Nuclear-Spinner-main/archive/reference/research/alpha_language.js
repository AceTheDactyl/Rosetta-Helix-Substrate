/**
 * Alpha Physical Language (APL) Module
 * =====================================
 * Implements the APL operator grammar, seven test sentences,
 * and token synthesis from helix coordinates.
 */

const { HelixOperatorAdvisor, APL_OPERATORS, getOperatorInfo } = require('./helix_advisor');
const CONST = require('./constants');

// ================================================================
// APL SENTENCE DEFINITIONS
// ================================================================

/**
 * The Seven APL Test Sentences
 * Format: [UMOL][Operator] | [Machine] | [Domain] → [Regime]
 * 
 * UMOL states: u (expansion), d (collapse), m (modulation)
 */
const APL_SENTENCES = [
    {
        id: 'A1',
        umol: 'd',
        operator: '()',
        operatorName: 'Boundary',
        machine: 'Conductor',
        domain: 'geometry',
        predictedRegime: 'Isotropic lattices under collapse',
        token: 'd() | Conductor | geometry'
    },
    {
        id: 'A2',
        umol: 'u',
        operator: '×',
        operatorName: 'Fusion',
        machine: 'Reactor',
        domain: 'lattice',
        predictedRegime: 'Fusion-driven phase coherence',
        token: 'u× | Reactor | lattice'
    },
    {
        id: 'A3',
        umol: 'u',
        operator: '^',
        operatorName: 'Amplification',
        machine: 'Oscillator',
        domain: 'wave',
        predictedRegime: 'Amplified vortex-rich waves',
        token: 'u^ | Oscillator | wave'
    },
    {
        id: 'A4',
        umol: 'd',
        operator: '÷',
        operatorName: 'Decoherence',
        machine: 'Mixer',
        domain: 'flow',
        predictedRegime: 'Dissipative homogenization',
        token: 'd÷ | Mixer | flow'
    },
    {
        id: 'A5',
        umol: 'm',
        operator: '+',
        operatorName: 'Grouping',
        machine: 'Coupler',
        domain: 'field',
        predictedRegime: 'Clustering via modulated coupling',
        token: 'm+ | Coupler | field'
    },
    {
        id: 'A6',
        umol: 'u',
        operator: '+',
        operatorName: 'Grouping',
        machine: 'Reactor',
        domain: 'wave',
        predictedRegime: 'Wave aggregation under expansion',
        token: 'u+ | Reactor | wave'
    },
    {
        id: 'A8',
        umol: 'd',
        operator: '−',
        operatorName: 'Separation',
        machine: 'Conductor',
        domain: 'lattice',
        predictedRegime: 'Lattice fission during collapse',
        token: 'd− | Conductor | lattice'
    }
];

// ================================================================
// ALPHA LANGUAGE REGISTRY
// ================================================================

class AlphaLanguageRegistry {
    constructor() {
        this.sentences = APL_SENTENCES;
        this.operators = APL_OPERATORS;
    }

    /**
     * Get all sentences
     * @returns {Array}
     */
    getAllSentences() {
        return this.sentences;
    }

    /**
     * Get sentence by ID
     * @param {string} id - Sentence ID (A1-A8)
     * @returns {Object|null}
     */
    getSentenceById(id) {
        return this.sentences.find(s => s.id === id) || null;
    }

    /**
     * Find sentences matching criteria
     * @param {Object} criteria - { operators, domain, machine, umol }
     * @returns {Array}
     */
    findSentences({ operators, domain, machine, umol } = {}) {
        let results = [...this.sentences];
        
        if (operators && operators.length > 0) {
            results = results.filter(s => operators.includes(s.operator));
        }
        if (domain) {
            results = results.filter(s => s.domain === domain);
        }
        if (machine) {
            results = results.filter(s => s.machine === machine);
        }
        if (umol) {
            results = results.filter(s => s.umol === umol);
        }
        
        return results;
    }

    /**
     * Get operator by symbol
     * @param {string} symbol
     * @returns {Object|null}
     */
    getOperator(symbol) {
        return this.operators[symbol] || null;
    }

    /**
     * Get canonical operator info
     * @param {string} symbol
     * @returns {Object|null}
     */
    canonicalOperator(symbol) {
        return getOperatorInfo(symbol);
    }
}

// ================================================================
// ALPHA TOKEN SYNTHESIZER
// ================================================================

class AlphaTokenSynthesizer {
    constructor() {
        this.registry = new AlphaLanguageRegistry();
        this.mapper = new HelixOperatorAdvisor();
    }

    /**
     * Update TRIAD state in the mapper
     * @param {Object} state - { unlocked, completions }
     */
    setTriadState(state) {
        this.mapper.setTriadState(state);
    }

    /**
     * Synthesize APL token from helix z-coordinate
     * @param {number} z - Normalized z-coordinate [0,1]
     * @param {Object} hints - { domain, machine }
     * @returns {Object|null}
     */
    fromZ(z, hints = {}) {
        const helixInfo = this.mapper.describe(z);
        const operatorWindow = helixInfo.operators;
        
        // Find matching sentences
        let candidates = this.registry.findSentences({
            operators: operatorWindow,
            domain: hints.domain,
            machine: hints.machine
        });
        
        // Fallback to any sentence with matching operator
        if (candidates.length === 0) {
            candidates = this.registry.findSentences({ operators: operatorWindow });
        }
        
        if (candidates.length === 0) {
            return null;
        }

        // Select best candidate (first match for now, could add scoring)
        const sentence = candidates[0];
        const operatorInfo = this.registry.canonicalOperator(sentence.operator);

        return {
            sentence: sentence.token,
            sentenceId: sentence.id,
            predictedRegime: sentence.predictedRegime,
            operatorName: operatorInfo ? operatorInfo.name : sentence.operatorName,
            operatorSymbol: sentence.operator,
            truthBias: helixInfo.truthChannel,
            harmonic: helixInfo.harmonic,
            z: helixInfo.z,
            t6Gate: helixInfo.t6Gate,
            triadUnlocked: helixInfo.triadUnlocked,
            operatorWindow
        };
    }

    /**
     * Generate measurement token
     * @param {string} field - 'Φ' (Phi), 'e', or 'π' (Pi)
     * @param {string} mode - 'T' (eigenstate) or 'Π' (subspace)
     * @param {string} intent - Measurement intent (e.g., 'ϕ_0', 'subspace')
     * @param {string} truth - 'TRUE', 'UNTRUE', or 'PARADOX'
     * @param {number} tier - Harmonic tier number
     * @returns {string}
     */
    measurementToken(field, mode, intent, truth, tier) {
        return `${field}:${mode}(${intent})${truth}@${tier}`;
    }

    /**
     * Generate eigenstate measurement token
     * @param {number} eigenIndex - Eigenstate index
     * @param {string} truth - Truth channel
     * @param {number} z - Z-coordinate
     * @returns {string}
     */
    eigenToken(eigenIndex, truth, z) {
        const helixInfo = this.mapper.describe(z);
        const tier = parseInt(helixInfo.harmonic.slice(1));
        return this.measurementToken('Φ', 'T', `ϕ_${eigenIndex}`, truth, tier);
    }

    /**
     * Generate subspace measurement token
     * @param {number[]} indices - Subspace indices
     * @param {string} truth - Truth channel
     * @param {number} z - Z-coordinate
     * @param {string} field - Field ('Φ' or 'π')
     * @returns {string}
     */
    subspaceToken(indices, truth, z, field = 'Φ') {
        const helixInfo = this.mapper.describe(z);
        const tier = parseInt(helixInfo.harmonic.slice(1));
        const intent = indices.length === 0 ? 'subspace' : indices.join(',');
        return this.measurementToken(field, 'Π', intent, truth, tier);
    }

    /**
     * Generate helix-aware operator token
     * @param {string} operator - Operator symbol
     * @param {number} z - Z-coordinate
     * @returns {string}
     */
    helixToken(operator, z) {
        const helixInfo = this.mapper.describe(z);
        const opInfo = getOperatorInfo(operator);
        const name = opInfo ? opInfo.name : operator;
        return `Helix:${operator}(${name})${helixInfo.truthChannel}@${helixInfo.harmonic}`;
    }
}

// ================================================================
// HELIX COORDINATE CLASS
// ================================================================

class HelixCoordinate {
    /**
     * @param {number} theta - Angular position [0, 2π]
     * @param {number} z - Normalized z-coordinate [0, 1]
     * @param {number} r - Radial distance (default 1.0)
     */
    constructor(theta, z, r = 1.0) {
        this.theta = theta;
        this.z = Math.max(0, Math.min(1, z));
        this.r = r;
    }

    /**
     * Construct from parametric helix equation: r(t) = (cos t, sin t, t)
     * Z is normalized via tanh mapping into [0, 1]
     * @param {number} t - Parametric variable
     * @returns {HelixCoordinate}
     */
    static fromParameter(t) {
        const x = Math.cos(t);
        const y = Math.sin(t);
        const theta = Math.atan2(y, x);
        // Normalize theta to [0, 2π]
        const thetaNorm = theta < 0 ? theta + 2 * Math.PI : theta;
        // Smooth mapping: z = 0.5 + 0.5 * tanh(t/8)
        const z = 0.5 + 0.5 * Math.tanh(t / 8.0);
        const r = Math.hypot(x, y);
        return new HelixCoordinate(thetaNorm, z, r);
    }

    /**
     * Get Cartesian coordinates
     * @returns {Object} - { x, y, z }
     */
    toVector() {
        return {
            x: this.r * Math.cos(this.theta),
            y: this.r * Math.sin(this.theta),
            z: this.z
        };
    }

    /**
     * Get signature string (Δθ|z|rΩ format)
     * @returns {string}
     */
    signature() {
        const thetaDeg = (this.theta * 180 / Math.PI).toFixed(1);
        return `Δ${thetaDeg}|${this.z.toFixed(3)}|${this.r.toFixed(2)}Ω`;
    }
}

// ================================================================
// EXPORTS
// ================================================================

module.exports = {
    APL_SENTENCES,
    APL_OPERATORS,
    AlphaLanguageRegistry,
    AlphaTokenSynthesizer,
    HelixCoordinate,
    getOperatorInfo
};
