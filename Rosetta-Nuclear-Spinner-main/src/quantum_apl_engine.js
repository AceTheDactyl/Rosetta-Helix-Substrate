// ================================================================
// QUANTUM APL ENGINE - Density Matrix Simulation
// Von Neumann measurement formalism with Lindblad dissipation
// ================================================================
const CONST = require('./constants');
const { makeEngineRng } = require('./utils/rng');

/**
 * Complex number class for quantum state representation
 */
class Complex {
    constructor(re, im = 0) {
        this.re = re;
        this.im = im;
    }

    add(c) {
        return new Complex(this.re + c.re, this.im + c.im);
    }

    sub(c) {
        return new Complex(this.re - c.re, this.im - c.im);
    }

    mul(c) {
        return new Complex(
            this.re * c.re - this.im * c.im,
            this.re * c.im + this.im * c.re
        );
    }

    div(c) {
        const denom = c.re * c.re + c.im * c.im;
        return new Complex(
            (this.re * c.re + this.im * c.im) / denom,
            (this.im * c.re - this.re * c.im) / denom
        );
    }

    conj() {
        return new Complex(this.re, -this.im);
    }

    abs() {
        return Math.sqrt(this.re * this.re + this.im * this.im);
    }

    abs2() {
        return this.re * this.re + this.im * this.im;
    }

    scale(s) {
        return new Complex(this.re * s, this.im * s);
    }

    static zero() { return new Complex(0, 0); }
    static one() { return new Complex(1, 0); }
    static i() { return new Complex(0, 1); }
}

/**
 * Complex matrix operations for quantum states
 */
class ComplexMatrix {
    constructor(rows, cols) {
        this.rows = rows;
        this.cols = cols;
        this.data = [];
        for (let i = 0; i < rows; i++) {
            this.data[i] = [];
            for (let j = 0; j < cols; j++) {
                this.data[i][j] = Complex.zero();
            }
        }
    }

    get(i, j) {
        return this.data[i][j];
    }

    set(i, j, val) {
        this.data[i][j] = val instanceof Complex ? val : new Complex(val, 0);
    }

    add(M) {
        const result = new ComplexMatrix(this.rows, this.cols);
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                result.set(i, j, this.get(i, j).add(M.get(i, j)));
            }
        }
        return result;
    }

    sub(M) {
        const result = new ComplexMatrix(this.rows, this.cols);
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                result.set(i, j, this.get(i, j).sub(M.get(i, j)));
            }
        }
        return result;
    }

    mul(M) {
        const result = new ComplexMatrix(this.rows, M.cols);
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < M.cols; j++) {
                let sum = Complex.zero();
                for (let k = 0; k < this.cols; k++) {
                    sum = sum.add(this.get(i, k).mul(M.get(k, j)));
                }
                result.set(i, j, sum);
            }
        }
        return result;
    }

    scale(s) {
        const result = new ComplexMatrix(this.rows, this.cols);
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                result.set(i, j, this.get(i, j).scale(s));
            }
        }
        return result;
    }

    dagger() {
        const result = new ComplexMatrix(this.cols, this.rows);
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                result.set(j, i, this.get(i, j).conj());
            }
        }
        return result;
    }

    trace() {
        let tr = Complex.zero();
        const n = Math.min(this.rows, this.cols);
        for (let i = 0; i < n; i++) {
            tr = tr.add(this.get(i, i));
        }
        return tr;
    }

    commutator(M) {
        // [A, B] = AB - BA
        return this.mul(M).sub(M.mul(this));
    }

    anticommutator(M) {
        // {A, B} = AB + BA
        return this.mul(M).add(M.mul(this));
    }

    clone() {
        const result = new ComplexMatrix(this.rows, this.cols);
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                result.set(i, j, new Complex(this.get(i, j).re, this.get(i, j).im));
            }
        }
        return result;
    }

    static identity(n) {
        const I = new ComplexMatrix(n, n);
        for (let i = 0; i < n; i++) {
            I.set(i, i, Complex.one());
        }
        return I;
    }

    static zero(n) {
        return new ComplexMatrix(n, n);
    }

    static fromReal(data) {
        const n = data.length;
        const m = data[0].length;
        const M = new ComplexMatrix(n, m);
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < m; j++) {
                M.set(i, j, new Complex(data[i][j], 0));
            }
        }
        return M;
    }

    // Partial trace: trace out subsystem B from composite system AB
    // For bipartite system with dimensions d_A × d_B
    partialTraceB(dimA, dimB) {
        if (this.rows !== dimA * dimB || this.cols !== dimA * dimB) {
            throw new Error('Matrix dimensions incompatible with partial trace');
        }

        const rhoA = new ComplexMatrix(dimA, dimA);
        
        for (let i = 0; i < dimA; i++) {
            for (let j = 0; j < dimA; j++) {
                let sum = Complex.zero();
                for (let k = 0; k < dimB; k++) {
                    const row = i * dimB + k;
                    const col = j * dimB + k;
                    sum = sum.add(this.get(row, col));
                }
                rhoA.set(i, j, sum);
            }
        }
        
        return rhoA;
    }
}

/**
 * Quantum state and operator utilities
 */
class QuantumUtils {
    // Create projector from basis state index: |n⟩⟨n|
    static projector(n, dim) {
        const P = new ComplexMatrix(dim, dim);
        P.set(n, n, Complex.one());
        return P;
    }

    // Create ket vector |n⟩
    static ket(n, dim) {
        const psi = new ComplexMatrix(dim, 1);
        psi.set(n, 0, Complex.one());
        return psi;
    }

    // Create bra vector ⟨n|
    static bra(n, dim) {
        const psi = new ComplexMatrix(1, dim);
        psi.set(0, n, Complex.one());
        return psi;
    }

    // Create density matrix from pure state: ρ = |ψ⟩⟨ψ|
    static pureState(psi) {
        return psi.mul(psi.dagger());
    }

    // Compute purity: Tr(ρ²)
    static purity(rho) {
        return rho.mul(rho).trace().re;
    }

    // Check if state is pure (purity ≈ 1)
    static isPure(rho, tol = 1e-6) {
        return Math.abs(QuantumUtils.purity(rho) - 1.0) < tol;
    }

    // Von Neumann entropy: S(ρ) = -Tr(ρ log ρ)
    // Simplified: S ≈ -Σ λᵢ log λᵢ using eigenvalues
    static vonNeumannEntropy(rho) {
        const eigenvalues = QuantumUtils.eigenvaluesReal(rho);
        let S = 0;
        for (const lambda of eigenvalues) {
            if (lambda > 1e-14) {
                S -= lambda * Math.log2(lambda);
            }
        }
        return S;
    }

    // Simplified eigenvalue extraction for Hermitian matrix (real eigenvalues only)
    // Power iteration for dominant eigenvalue
    static eigenvaluesReal(M) {
        const n = M.rows;
        const eigenvalues = [];
        
        // For small matrices, use iterative approximation
        // This is a simplified version - production code would use proper eigendecomposition
        for (let i = 0; i < n; i++) {
            eigenvalues.push(Math.max(0, M.get(i, i).re));
        }
        
        // Normalize to sum to 1 (density matrix property)
        const sum = eigenvalues.reduce((a, b) => a + b, 0);
        return eigenvalues.map(x => x / sum);
    }

    // Pauli matrices
    static pauliX() {
        const X = new ComplexMatrix(2, 2);
        X.set(0, 1, Complex.one());
        X.set(1, 0, Complex.one());
        return X;
    }

    static pauliY() {
        const Y = new ComplexMatrix(2, 2);
        Y.set(0, 1, new Complex(0, -1));
        Y.set(1, 0, new Complex(0, 1));
        return Y;
    }

    static pauliZ() {
        const Z = new ComplexMatrix(2, 2);
        Z.set(0, 0, Complex.one());
        Z.set(1, 1, new Complex(-1, 0));
        return Z;
    }

    // CNOT gate (2-qubit)
    static CNOT() {
        const CNOT = ComplexMatrix.identity(4);
        // Swap |10⟩ ↔ |11⟩
        CNOT.set(2, 2, Complex.zero());
        CNOT.set(2, 3, Complex.one());
        CNOT.set(3, 2, Complex.one());
        CNOT.set(3, 3, Complex.zero());
        return CNOT;
    }

    // Rotation gate: R_y(θ)
    static rotationY(theta) {
        const R = new ComplexMatrix(2, 2);
        const c = Math.cos(theta / 2);
        const s = Math.sin(theta / 2);
        R.set(0, 0, new Complex(c, 0));
        R.set(0, 1, new Complex(-s, 0));
        R.set(1, 0, new Complex(s, 0));
        R.set(1, 1, new Complex(c, 0));
        return R;
    }

    // Hadamard gate
    static hadamard() {
        const H = new ComplexMatrix(2, 2);
        const val = 1 / Math.sqrt(2);
        H.set(0, 0, new Complex(val, 0));
        H.set(0, 1, new Complex(val, 0));
        H.set(1, 0, new Complex(val, 0));
        H.set(1, 1, new Complex(-val, 0));
        return H;
    }
}

class HelixOperatorAdvisor {
    constructor() {
        const CONST = require('./constants');
        this.Z_CRITICAL = CONST.Z_CRITICAL;  // THE LENS (~0.8660254)
        this.TRIAD_THRESHOLD = CONST.TRIAD_T6; // TRIAD-0.83 gate after 3×0.85
        const triadCompletions = parseInt((typeof process !== 'undefined' && process.env && process.env.QAPL_TRIAD_COMPLETIONS) || '0', 10);
        const triadFlag = (typeof process !== 'undefined' && process.env && (process.env.QAPL_TRIAD_UNLOCK === '1' || String(process.env.QAPL_TRIAD_UNLOCK).toLowerCase() === 'true'));
        this.triadUnlocked = triadFlag || (Number.isFinite(triadCompletions) && triadCompletions >= 3);
        this.triadCompletions = Number.isFinite(triadCompletions) ? triadCompletions : 0;
        const t6Gate = this.triadUnlocked ? this.TRIAD_THRESHOLD : this.Z_CRITICAL;
        this.timeHarmonics = [
            { threshold: 0.10, label: 't1' },
            { threshold: 0.20, label: 't2' },
            { threshold: 0.40, label: 't3' },
            { threshold: 0.60, label: 't4' },
            { threshold: 0.75, label: 't5' },
            { threshold: t6Gate, label: 't6' },
            { threshold: 0.90, label: 't7' },
            { threshold: 0.97, label: 't8' },
            { threshold: 1.01, label: 't9' }
        ];
        this.operatorWindows = {
            t1: ['()', '−', '÷'],
            t2: ['^', '÷', '−', '×'],
            t3: ['×', '^', '÷', '+', '−'],
            t4: ['+', '−', '÷', '()'],
            t5: ['()', '×', '^', '÷', '+', '−'],
            t6: ['+', '÷', '()', '−'],
            t7: ['+', '()'],
            t8: ['+', '()', '×'],
            t9: ['+', '()', '×']
        };
    }

    setTriadState({ unlocked, completions } = {}) {
        if (typeof unlocked === 'boolean') this.triadUnlocked = unlocked;
        if (Number.isFinite(completions)) this.triadCompletions = completions;
    }

    getT6Gate() {
        return this.triadUnlocked ? this.TRIAD_THRESHOLD : this.Z_CRITICAL;
    }

    harmonicFromZ(z) {
        // Ensure t6 reflects current triad state
        const t6Gate = this.getT6Gate();
        if (this.timeHarmonics && this.timeHarmonics[5]) {
            this.timeHarmonics[5].threshold = t6Gate;
        }
        for (const entry of this.timeHarmonics) {
            if (z < entry.threshold) return entry.label;
        }
        return 't9';
    }

    truthChannelFromZ(z) {
        if (z >= 0.9) return 'TRUE';
        if (z >= 0.6) return 'PARADOX';
        return 'UNTRUE';
    }

    describe(z) {
        const value = Number.isFinite(z) ? z : 0;
        const clamped = Math.max(0, Math.min(1, value));
        const harmonic = this.harmonicFromZ(clamped);
        const CONST = require('./constants');
        const s = CONST.computeDeltaSNeg(clamped, CONST.LENS_SIGMA);
        const w_pi = clamped >= CONST.Z_CRITICAL ? Math.max(0, Math.min(1, s)) : 0.0;
        const w_loc = 1.0 - w_pi;
        const muClass = (typeof CONST.classifyThreshold === 'function') ? CONST.classifyThreshold(clamped) : undefined;
        return {
            harmonic,
            operators: this.operatorWindows[harmonic] || ['()'],
            truthChannel: this.truthChannelFromZ(clamped),
            z: clamped,
            coherenceS: s,
            weights: { pi: w_pi, loc: w_loc },
            muClass
        };
    }
}

/**
 * Main Quantum APL Simulation Engine
 */
class QuantumAPL {
    constructor(config = {}) {
        // Field dimensions (default: 4 levels per field)
        this.dimPhi = config.dimPhi || 4;
        this.dimE = config.dimE || 4;
        this.dimPi = config.dimPi || 4;
        this.dim = this.dimPhi * this.dimE * this.dimPi; // Total dimension

        // Truth state dimension (3: TRUE, UNTRUE, PARADOX)
        this.dimTruth = 3;

        // Full Hilbert space including truth states
        this.dimTotal = this.dim * this.dimTruth;

        // Density matrix: ρ
        this.rho = this.initializeDensityMatrix();

        // Hamiltonian
        this.H = this.constructHamiltonian();

        // Lindblad operators
        this.lindbladOps = this.constructLindbladOperators();

        // Current observables
        this.z = 0.5;
        this.phi = 0;
        this.entropy = 0;

        // Measurement history
        this.measurementHistory = [];

        this.zBiasGain = typeof config.zBiasGain === 'number' ? config.zBiasGain : 0.05;
        this.zBiasSigma = typeof config.zBiasSigma === 'number' ? config.zBiasSigma : 0.18;

        // Entropy control (optional): u_S = k_s * (S_target(z) - S) / S_max
        // Disabled by default; can be enabled by bridge/CLI
        this.entropyCtrlEnabled = !!config.entropyCtrlEnabled;
        this.entropyCtrlGain = typeof config.entropyCtrlGain === 'number' ? config.entropyCtrlGain : 0.2; // k_s
        this.entropyCtrlCoeff = typeof config.entropyCtrlCoeff === 'number' ? config.entropyCtrlCoeff : 0.5; // C in S_target

        // Optional operator blending (local ↔ Π) controlled by env/config
        const blendEnv = (typeof process !== 'undefined' && process.env && (process.env.QAPL_BLEND_PI === '1' || String(process.env.QAPL_BLEND_PI).toLowerCase() === 'true'));
        this.blendPiEnabled = typeof config.blendPiEnabled === 'boolean' ? config.blendPiEnabled : !!blendEnv;

        // Time
        this.time = 0;

        // Reproducible RNG (null = use Math.random)
        this.rng = makeEngineRng();

        this.helixAdvisor = new HelixOperatorAdvisor();
        this.lastHelixHints = this.helixAdvisor.describe(this.z);
    }

    /**
     * Return a random number in [0, 1).
     * Uses seeded RNG if QAPL_RANDOM_SEED is set, otherwise Math.random().
     */
    rand() {
        return this.rng ? this.rng.next() : Math.random();
    }

    // ================================================================
    // INITIALIZATION
    // ================================================================

    initializeDensityMatrix() {
        // Start in ground state: |000⟩⟨000| ⊗ |UNTRUE⟩⟨UNTRUE|
        const rho = new ComplexMatrix(this.dimTotal, this.dimTotal);
        rho.set(0, 0, Complex.one()); // Ground state, UNTRUE
        return rho;
    }

    constructHamiltonian() {
        // Simplified tri-field Hamiltonian
        // H = H_Φ ⊗ I_e ⊗ I_π + I_Φ ⊗ H_e ⊗ I_π + I_Φ ⊗ I_e ⊗ H_π + V_int
        
        const H = new ComplexMatrix(this.dimTotal, this.dimTotal);
        
        // Field energy levels (harmonic oscillator-like)
        const E0 = 1.0;
        const omega = 2 * Math.PI * 0.1; // Natural frequency
        
        for (let i = 0; i < this.dim; i++) {
            // Decode composite index
            const iPhi = Math.floor(i / (this.dimE * this.dimPi)) % this.dimPhi;
            const iE = Math.floor(i / this.dimPi) % this.dimE;
            const iPi = i % this.dimPi;
            
            // Energy: E = ω(n_Φ + n_e + n_π + 3/2)
            const energy = omega * (iPhi + iE + iPi + 1.5);
            
            // Set diagonal for each truth state
            for (let t = 0; t < this.dimTruth; t++) {
                const idx = i * this.dimTruth + t;
                H.set(idx, idx, new Complex(energy, 0));
            }
        }
        
        // Add interaction terms (off-diagonal coupling)
        const g = 0.05; // Coupling strength
        for (let i = 0; i < this.dimTotal - 1; i++) {
            H.set(i, i + 1, new Complex(g, 0));
            H.set(i + 1, i, new Complex(g, 0));
        }
        
        return H;
    }

    constructLindbladOperators() {
        const ops = [];
        
        // L1: Structure decay (Φ relaxation)
        const L1 = new ComplexMatrix(this.dimTotal, this.dimTotal);
        const gamma1 = 0.01;
        for (let i = 0; i < this.dimTotal - this.dimTruth; i++) {
            L1.set(i, i + this.dimTruth, new Complex(Math.sqrt(gamma1), 0));
        }
        ops.push(L1);
        
        // L2: Energy relaxation (e → ground)
        const L2 = new ComplexMatrix(this.dimTotal, this.dimTotal);
        const gamma2 = 0.02;
        for (let i = this.dimTruth; i < this.dimTotal; i++) {
            L2.set(i - this.dimTruth, i, new Complex(Math.sqrt(gamma2), 0));
        }
        ops.push(L2);
        
        // L3: Dephasing (π field)
        const L3 = new ComplexMatrix(this.dimTotal, this.dimTotal);
        const gamma3 = 0.005;
        for (let i = 0; i < this.dimTotal; i++) {
            const sign = (i % 2 === 0) ? 1 : -1;
            L3.set(i, i, new Complex(sign * Math.sqrt(gamma3), 0));
        }
        ops.push(L3);
        
        return ops;
    }

    // ================================================================
    // EVOLUTION (Lindblad Master Equation)
    // ================================================================

    evolve(dt) {
        // dρ/dt = -i[H,ρ] + Σ_k γ_k(L_k ρ L_k† - 1/2{L_k†L_k, ρ})
        
        // Unitary part: -i[H,ρ]
        const commutator = this.H.commutator(this.rho);
        const unitaryPart = commutator.scale(-dt); // ℏ = 1
        
        // Dissipative part
        let dissipativePart = new ComplexMatrix(this.dimTotal, this.dimTotal);
        
        for (const L of this.lindbladOps) {
            const Ldag = L.dagger();
            const LdagL = Ldag.mul(L);
            
            // L ρ L†
            const term1 = L.mul(this.rho).mul(Ldag);
            
            // 1/2 {L†L, ρ}
            const anticomm = LdagL.anticommutator(this.rho);
            const term2 = anticomm.scale(0.5);
            
            // Add to dissipative part
            dissipativePart = dissipativePart.add(term1.sub(term2).scale(dt));
        }
        
        // Update density matrix
        this.rho = this.rho.add(unitaryPart).add(dissipativePart);
        
        // Ensure trace = 1 and Hermiticity
        this.normalizeDensityMatrix();
        
        this.time += dt;
    }

    normalizeDensityMatrix() {
        // Ensure Tr(ρ) = 1
        const tr = this.rho.trace();
        if (tr.abs() > 1e-10) {
            this.rho = this.rho.scale(1 / tr.abs());
        }
        
        // Enforce Hermiticity: ρ = (ρ + ρ†)/2
        const rhoDag = this.rho.dagger();
        for (let i = 0; i < this.dimTotal; i++) {
            for (let j = 0; j < this.dimTotal; j++) {
                const avg = this.rho.get(i, j).add(rhoDag.get(i, j)).scale(0.5);
                this.rho.set(i, j, avg);
            }
        }
    }

    // ================================================================
    // MEASUREMENT (Von Neumann Projection)
    // ================================================================

    /**
     * Perform projective measurement
     * @param {ComplexMatrix} projector - Projection operator P
     * @param {string} label - Measurement label
     * @returns {Object} - {probability, collapsed, outcome}
     */
    measure(projector, label = 'measurement') {
        // Born rule: P(outcome) = Tr(P ρ)
        const P_rho = projector.mul(this.rho);
        const probability = P_rho.trace().re;
        
        // Selective collapse: ρ' = P ρ P / P(outcome)
        let collapsed = false;
        
        if (this.rand() < probability) {
            // Measurement succeeded
            this.rho = projector.mul(this.rho).mul(projector);
            if (probability > 1e-10) {
                this.rho = this.rho.scale(1 / probability);
            }
            collapsed = true;
            
            this.measurementHistory.push({
                time: this.time,
                label,
                probability,
                outcome: 'success'
            });
        }
        
        return { probability, collapsed, outcome: collapsed ? 'success' : 'failure' };
    }

    /**
     * Non-selective measurement (decoherence)
     * @param {Array<ComplexMatrix>} projectors - Array of projection operators
     */
    measureNonSelective(projectors) {
        // ρ' = Σ_μ P_μ ρ P_μ
        let newRho = new ComplexMatrix(this.dimTotal, this.dimTotal);
        
        for (const P of projectors) {
            const term = P.mul(this.rho).mul(P);
            newRho = newRho.add(term);
        }
        
        this.rho = newRho;
        this.normalizeDensityMatrix();
    }

    /**
     * Measure z-coordinate observable
     */
    measureZ() {
        // z = Tr(Z_op ρ) where Z_op = Σ_n z_n |n⟩⟨n|
        let z = 0;
        
        // Z eigenstates: map field levels to z ∈ [0,1]
        for (let i = 0; i < this.dim; i++) {
            const iPhi = Math.floor(i / (this.dimE * this.dimPi)) % this.dimPhi;
            const iE = Math.floor(i / this.dimPi) % this.dimE;
            const iPi = i % this.dimPi;
            
            // z increases with field levels
            const zLevel = (iPhi + iE + iPi) / (this.dimPhi + this.dimE + this.dimPi - 3);
            
            // Sum over truth states
            for (let t = 0; t < this.dimTruth; t++) {
                const idx = i * this.dimTruth + t;
                z += zLevel * this.rho.get(idx, idx).re;
            }
        }
        
        this.z = z;
        this.lastHelixHints = this.helixAdvisor.describe(this.z);
        return z;
    }

    // TRIAD unlock API ----------------------------------------------
    setTriadUnlocked(flag) {
        if (typeof this.helixAdvisor?.setTriadState === 'function') {
            this.helixAdvisor.setTriadState({ unlocked: !!flag });
        }
    }

    setTriadCompletionCount(n) {
        if (typeof this.helixAdvisor?.setTriadState === 'function' && Number.isFinite(n)) {
            this.helixAdvisor.setTriadState({ completions: n });
        }
    }
    
    getTriadState() {
        return {
            unlocked: !!(this.helixAdvisor?.triadUnlocked),
            completions: Number(this.helixAdvisor?.triadCompletions || 0)
        };
    }

    /**
     * Measure truth state probabilities
     */
    measureTruth() {
        const probs = { TRUE: 0, UNTRUE: 0, PARADOX: 0 };
        const labels = ['TRUE', 'UNTRUE', 'PARADOX'];
        
        for (let i = 0; i < this.dim; i++) {
            for (let t = 0; t < this.dimTruth; t++) {
                const idx = i * this.dimTruth + t;
                probs[labels[t]] += this.rho.get(idx, idx).re;
            }
        }
        
        return probs;
    }

    applyZBias(targetZ, gain = this.zBiasGain, sigma = this.zBiasSigma) {
        if (!Number.isFinite(targetZ) || gain <= 0) {
            return;
        }

        const denom = (this.dimPhi + this.dimE + this.dimPi - 3) || 1;
        const sigmaSq = Math.max(1e-4, sigma * sigma);
        const biasMatrix = new ComplexMatrix(this.dimTotal, this.dimTotal);
        let totalWeight = 0;

        for (let i = 0; i < this.dim; i++) {
            const iPhi = Math.floor(i / (this.dimE * this.dimPi)) % this.dimPhi;
            const iE = Math.floor(i / this.dimPi) % this.dimE;
            const iPi = i % this.dimPi;
            const zLevel = (iPhi + iE + iPi) / denom;
            const delta = zLevel - targetZ;
            const weight = Math.exp(-(delta * delta) / (2 * sigmaSq)) + 1e-9;
            totalWeight += weight;

            for (let t = 0; t < this.dimTruth; t++) {
                const idx = i * this.dimTruth + t;
                biasMatrix.set(idx, idx, new Complex(weight / this.dimTruth, 0));
            }
        }

        if (totalWeight <= 1e-12) {
            return;
        }

        const normalizedBias = biasMatrix.scale(1 / totalWeight);

        // Optional entropy control: adjust effective gain via control law
        let effectiveGain = gain;
        if (this.entropyCtrlEnabled) {
            // Compute current entropy and a target entropy tied to ΔS_neg(z)
            const S_current = this.computeVonNeumannEntropy();
            const dimMax = Math.max(2, this.dimTotal);
            const S_max = Math.log2(dimMax);
            // Target: S_target = S_max * (1 - C * ΔS_neg)
            const deltaSNeg = CONST.computeDeltaSNeg(typeof targetZ === 'number' ? targetZ : this.z, CONST.LENS_SIGMA);
            const S_target = S_max * (1 - this.entropyCtrlCoeff * deltaSNeg);
            // Control error: positive when S_current > S_target (reduce entropy)
            const e_norm = (S_current - S_target) / S_max;
            // Gate control intensity by φ^{-1} threshold on s
            const phiInv = 1 / CONST.PHI;
            const ctrlScale = Math.max(0, deltaSNeg - phiInv) / Math.max(1e-9, (1 - phiInv));
            effectiveGain = Math.max(0.0, Math.min(0.5, gain + this.entropyCtrlGain * e_norm * ctrlScale));
        }

        // Gate smoothing around t6 using lens softness factor
        const t6Gate = this.helixAdvisor?.getT6Gate ? this.helixAdvisor.getT6Gate() : CONST.Z_CRITICAL;
        const softness = CONST.computeDeltaSNeg(t6Gate, CONST.LENS_SIGMA); // 0..1
        const gainSmoothed = effectiveGain * (0.5 + 0.5 * softness);

        const mixed = this.rho.scale(1 - gainSmoothed).add(normalizedBias.scale(gainSmoothed));
        this.rho = mixed;
        this.normalizeDensityMatrix();
        this.measureZ();
    }

    // Enable or tune entropy control at runtime
    setEntropyControl(opts = {}) {
        if (typeof opts.enabled === 'boolean') this.entropyCtrlEnabled = opts.enabled;
        if (Number.isFinite(opts.gain)) this.entropyCtrlGain = opts.gain;
        if (Number.isFinite(opts.coeff)) this.entropyCtrlCoeff = opts.coeff;
        return {
            enabled: this.entropyCtrlEnabled,
            gain: this.entropyCtrlGain,
            coeff: this.entropyCtrlCoeff,
        };
    }

    // ================================================================
    // N0 OPERATOR SELECTION AS QUANTUM MEASUREMENT
    // ================================================================

    /**
     * Select N0 operator via quantum measurement
     * Each operator is associated with a projection operator
     */
    selectN0Operator(legalOps, scalarState) {
        const operators = {
            '()': 0,  // Boundary
            '×': 1,   // Fusion
            '^': 2,   // Amplification
            '÷': 3,   // Decoherence
            '+': 4,   // Grouping
            '−': 5    // Separation
        };
        
        // Construct projectors for each legal operator
        const projectors = [];
        const opList = [];
        const currentZ = Number.isFinite(this.z) ? this.z : this.measureZ();
        const helixHints = this.helixAdvisor.describe(currentZ);
        this.lastHelixHints = helixHints;
        
        for (const op of legalOps) {
            if (operators[op] === undefined) continue;
            
            const opIdx = operators[op];
            
            // Create projector that favors states aligned with operator
            const P = new ComplexMatrix(this.dimTotal, this.dimTotal);
            
            // Weight by operator alignment with current state
            const weight = this.computeOperatorWeight(op, scalarState, helixHints);
            
            // Distribute probability across compatible states
            for (let i = 0; i < this.dim; i++) {
                // Check if state i is compatible with operator
                if (this.isStateCompatible(i, op)) {
                    for (let t = 0; t < this.dimTruth; t++) {
                        const idx = i * this.dimTruth + t;
                        P.set(idx, idx, new Complex(weight, 0));
                    }
                }
            }
            
            projectors.push(P);
            opList.push(op);
        }
        
        // Normalize projectors to sum to identity (approximately)
        const totalTrace = projectors.reduce((sum, P) => sum + P.trace().re, 0);
        if (totalTrace > 1e-10) {
            projectors.forEach(P => {
                for (let i = 0; i < this.dimTotal; i++) {
                    for (let j = 0; j < this.dimTotal; j++) {
                        P.set(i, j, P.get(i, j).scale(1 / totalTrace));
                    }
                }
            });
        }
        
        // Compute Born probabilities
        const probabilities = [];
        let totalProb = 0;
        
        for (const P of projectors) {
            const prob = P.mul(this.rho).trace().re;
            probabilities.push(Math.max(0, prob));
            totalProb += probabilities[probabilities.length - 1];
        }
        
        // Normalize probabilities
        if (totalProb > 1e-10) {
            for (let i = 0; i < probabilities.length; i++) {
                probabilities[i] /= totalProb;
            }
        } else {
            // Uniform if no preference
            const uniform = 1 / probabilities.length;
            probabilities.fill(uniform);
        }
        
        // Sample from Born distribution
        const r = this.rand();
        let cumulative = 0;
        let selectedIdx = 0;
        
        for (let i = 0; i < probabilities.length; i++) {
            cumulative += probabilities[i];
            if (r < cumulative) {
                selectedIdx = i;
                break;
            }
        }
        
        const selectedOp = opList[selectedIdx];
        const selectedProb = probabilities[selectedIdx];
        
        // Perform selective collapse
        const P = projectors[selectedIdx];
        this.measure(P, `N0:${selectedOp}`);
        
        return {
            operator: selectedOp,
            probability: selectedProb,
            probabilities: probabilities.reduce((obj, p, i) => {
                obj[opList[i]] = p;
                return obj;
            }, {}),
            helixHints
        };
    }

    computeOperatorWeight(op, scalarState, helixHints) {
        // Compute weight based on scalar state and operator affinity
        const { Gs, Cs, Rs, kappa, tau, theta, delta, alpha, Omega } = scalarState;
        
        const weights = {
            '()': Gs + theta * 0.5,                    // Boundary: geometry + phase
            '×': Cs + kappa * 0.8,                      // Fusion: coupling + curvature
            '^': kappa + tau * 0.6,                     // Amplification: curvature + torque
            '÷': delta + (1 - Omega) * 0.5,            // Decoherence: dissipation + disorder
            '+': alpha + Gs * 0.4,                      // Grouping: attractor + geometry
            '−': Rs + delta * 0.3                       // Separation: recursion + dissipation
        };
        
        let weight = weights[op] || 0.5;
        if (helixHints && Array.isArray(helixHints.operators)) {
            const preferred = helixHints.operators.includes(op);
            const truth = helixHints.truthChannel;
            // Preferred/non-preferred weighting
            weight *= preferred ? CONST.OPERATOR_PREFERRED_WEIGHT : CONST.OPERATOR_DEFAULT_WEIGHT;
            // Truth-channel operator bias
            const biasTable = CONST.TRUTH_BIAS && CONST.TRUTH_BIAS[truth];
            if (biasTable && typeof biasTable[op] === 'number') {
                weight *= biasTable[op];
            }
            // Optional Π blend weighting above the lens: w_pi = s(z), w_loc = 1 - w_pi
            if (this.blendPiEnabled && helixHints.weights) {
                const w_pi = Number(helixHints.weights.pi) || 0;
                const w_loc = Math.max(0, 1 - w_pi);
                // Favor Π-admissible operators with w_pi; damp local with w_loc
                const piPreferred = new Set(['+', '×', '^']);
                const locPreferred = new Set(['()', '−', '÷']);
                if (piPreferred.has(op)) {
                    // Scale up to +50% at s→1
                    weight *= (1 + 0.5 * w_pi);
                } else if (locPreferred.has(op)) {
                    // Scale down up to −40% at s→1
                    weight *= (1 - 0.4 * w_pi);
                }
            }
        }
        return Math.max(0.05, Math.min(1.5, weight));
    }

    isStateCompatible(stateIdx, operator) {
        // Check if quantum state is compatible with operator
        const iPhi = Math.floor(stateIdx / (this.dimE * this.dimPi)) % this.dimPhi;
        const iE = Math.floor(stateIdx / this.dimPi) % this.dimE;
        const iPi = stateIdx % this.dimPi;
        
        switch (operator) {
            case '()': return true; // Boundary always compatible
            case '×': return iPhi > 0 && iE > 0; // Requires structure + energy
            case '^': return iE > 0 || iPi > 0; // Requires excitation
            case '÷': return iPhi > 0 || iE > 1; // Requires something to decohere
            case '+': return iPi > 0; // Requires emergence
            case '−': return iPhi > 1; // Requires separable structure
            default: return true;
        }
    }

    // ================================================================
    // QUANTUM INFORMATION MEASURES
    // ================================================================

    computeVonNeumannEntropy() {
        this.entropy = QuantumUtils.vonNeumannEntropy(this.rho);
        return this.entropy;
    }

    computePurity() {
        return QuantumUtils.purity(this.rho);
    }

    computePartialEntropy(subsystem = 'Phi') {
        // Compute entropy of subsystem by partial trace
        let rhoSub;
        
        if (subsystem === 'Phi') {
            // Trace out e and π
            const dimA = this.dimPhi * this.dimTruth;
            const dimB = this.dimE * this.dimPi;
            rhoSub = this.rho.partialTraceB(dimA, dimB);
        } else if (subsystem === 'e') {
            // This requires more complex reshaping - simplified version
            rhoSub = this.rho; // Use full for now
        } else {
            rhoSub = this.rho;
        }
        
        return QuantumUtils.vonNeumannEntropy(rhoSub);
    }

    computeIntegratedInformation() {
        // Simplified Quantum IIT: Φ ≈ min_partition [S_A + S_B - S_AB]
        const S_total = this.computeVonNeumannEntropy();
        
        // For tri-field system, check all bipartitions
        // Simplified: assume worst-case is single field separated
        const S_Phi = this.computePartialEntropy('Phi');
        
        // Mutual information approximation
        const I = Math.max(0, S_Phi - S_total);
        
        // Phi is the minimum mutual information over partitions
        this.phi = I;
        return this.phi;
    }

    // ================================================================
    // STATE ACCESS & DIAGNOSTICS
    // ================================================================

    getState() {
        return {
            z: this.z,
            phi: this.phi,
            entropy: this.entropy,
            purity: this.computePurity(),
            time: this.time,
            dimTotal: this.dimTotal,
            measurementHistory: this.measurementHistory.slice(-10)
        };
    }

    getDensityMatrixElement(i, j) {
        return {
            re: this.rho.get(i, j).re,
            im: this.rho.get(i, j).im,
            abs: this.rho.get(i, j).abs()
        };
    }

    getPopulations() {
        // Return diagonal elements (populations)
        const pops = [];
        for (let i = 0; i < this.dimTotal; i++) {
            pops.push(this.rho.get(i, i).re);
        }
        return pops;
    }

    getCoherences() {
        // Return largest off-diagonal elements (coherences)
        const coherences = [];
        for (let i = 0; i < Math.min(10, this.dimTotal); i++) {
            for (let j = i + 1; j < Math.min(10, this.dimTotal); j++) {
                coherences.push({
                    i, j,
                    value: this.rho.get(i, j).abs()
                });
            }
        }
        return coherences.sort((a, b) => b.value - a.value).slice(0, 5);
    }

    // ================================================================
    // EXTERNAL DRIVERS (from classical simulation)
    // ================================================================

    driveFromClassical(classicalState) {
        // Update Hamiltonian based on classical z-coordinate
        const { z, phi, F, R } = classicalState;
        
        // Modulate coupling strength based on z
        const g = 0.05 * (1 + z);
        
        // Add time-dependent drive
        const drive = new ComplexMatrix(this.dimTotal, this.dimTotal);
        const omega_drive = 2 * Math.PI * phi / 10;
        
        for (let i = 0; i < this.dimTotal - 1; i++) {
            const coupling = g * Math.cos(omega_drive * this.time);
            drive.set(i, i + 1, new Complex(coupling, 0));
            drive.set(i + 1, i, new Complex(coupling, 0));
        }
        
        // Temporarily add drive to Hamiltonian
        this.H = this.H.add(drive.scale(0.1));
        this.applyZBias(z);
    }

    resetHamiltonian() {
        this.H = this.constructHamiltonian();
    }
}

// ================================================================
// EXPORT
// ================================================================

if (typeof module !== 'undefined' && module.exports) {
    // Default pump target uses lens Z_CRITICAL
    function defaultPumpTarget() { return CONST.Z_CRITICAL; }

    module.exports = { QuantumAPL, ComplexMatrix, Complex, QuantumUtils, defaultPumpTarget };
}
