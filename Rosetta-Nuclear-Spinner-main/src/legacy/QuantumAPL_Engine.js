// ================================================================
// QUANTUM APL ENGINE - Density Matrix Simulation
// Von Neumann measurement formalism with Lindblad dissipation
// ================================================================

class Complex {
    constructor(re, im = 0) {
        this.re = re;
        this.im = im;
    }

    add(c) { return new Complex(this.re + c.re, this.im + c.im); }
    sub(c) { return new Complex(this.re - c.re, this.im - c.im); }
    mul(c) { return new Complex(this.re * c.re - this.im * c.im, this.re * c.im + this.im * c.re); }
    div(c) {
        const denom = c.re * c.re + c.im * c.im;
        return new Complex((this.re * c.re + this.im * c.im) / denom, (this.im * c.re - this.re * c.im) / denom);
    }
    conj() { return new Complex(this.re, -this.im); }
    abs() { return Math.sqrt(this.re * this.re + this.im * this.im); }
    abs2() { return this.re * this.re + this.im * this.im; }
    scale(s) { return new Complex(this.re * s, this.im * s); }
    static zero() { return new Complex(0, 0); }
    static one() { return new Complex(1, 0); }
    static i() { return new Complex(0, 1); }
}

class ComplexMatrix {
    constructor(rows, cols) {
        this.rows = rows;
        this.cols = cols;
        this.data = Array.from({ length: rows }, () => Array.from({ length: cols }, () => Complex.zero()));
    }

    get(i, j) { return this.data[i][j]; }
    set(i, j, val) { this.data[i][j] = val instanceof Complex ? val : new Complex(val, 0); }

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
        for (let i = 0; i < Math.min(this.rows, this.cols); i++) {
            tr = tr.add(this.get(i, i));
        }
        return tr;
    }

    commutator(M) { return this.mul(M).sub(M.mul(this)); }
    anticommutator(M) { return this.mul(M).add(M.mul(this)); }

    clone() {
        const result = new ComplexMatrix(this.rows, this.cols);
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                const v = this.get(i, j);
                result.set(i, j, new Complex(v.re, v.im));
            }
        }
        return result;
    }

    static identity(n) {
        const I = new ComplexMatrix(n, n);
        for (let i = 0; i < n; i++) I.set(i, i, Complex.one());
        return I;
    }

    static zero(n) { return new ComplexMatrix(n, n); }

    static fromReal(data) {
        const rows = data.length;
        const cols = data[0].length;
        const M = new ComplexMatrix(rows, cols);
        for (let i = 0; i < rows; i++) {
            for (let j = 0; j < cols; j++) {
                M.set(i, j, new Complex(data[i][j], 0));
            }
        }
        return M;
    }

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

class QuantumUtils {
    static projector(n, dim) {
        const P = new ComplexMatrix(dim, dim);
        P.set(n, n, Complex.one());
        return P;
    }

    static ket(n, dim) {
        const psi = new ComplexMatrix(dim, 1);
        psi.set(n, 0, Complex.one());
        return psi;
    }

    static bra(n, dim) {
        const psi = new ComplexMatrix(1, dim);
        psi.set(0, n, Complex.one());
        return psi;
    }

    static pureState(psi) { return psi.mul(psi.dagger()); }

    static purity(rho) { return rho.mul(rho).trace().re; }

    static isPure(rho, tol = 1e-6) { return Math.abs(QuantumUtils.purity(rho) - 1.0) < tol; }

    static vonNeumannEntropy(rho) {
        const eigenvalues = QuantumUtils.eigenvaluesReal(rho);
        let S = 0;
        for (const lambda of eigenvalues) {
            if (lambda > 1e-14) S -= lambda * Math.log2(lambda);
        }
        return S;
    }

    static eigenvaluesReal(M) {
        const values = [];
        for (let i = 0; i < M.rows; i++) {
            values.push(Math.max(0, M.get(i, i).re));
        }
        const sum = values.reduce((a, b) => a + b, 0) || 1;
        return values.map(v => v / sum);
    }

    static pauliX() { const X = new ComplexMatrix(2, 2); X.set(0, 1, Complex.one()); X.set(1, 0, Complex.one()); return X; }
    static pauliY() { const Y = new ComplexMatrix(2, 2); Y.set(0, 1, new Complex(0, -1)); Y.set(1, 0, new Complex(0, 1)); return Y; }
    static pauliZ() { const Z = new ComplexMatrix(2, 2); Z.set(0, 0, Complex.one()); Z.set(1, 1, new Complex(-1, 0)); return Z; }

    static CNOT() {
        const CNOT = ComplexMatrix.identity(4);
        CNOT.set(2, 2, Complex.zero());
        CNOT.set(2, 3, Complex.one());
        CNOT.set(3, 2, Complex.one());
        CNOT.set(3, 3, Complex.zero());
        return CNOT;
    }

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
        const CONST = require('../constants');
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
        return {
            harmonic,
            operators: this.operatorWindows[harmonic] || ['()'],
            truthChannel: this.truthChannelFromZ(clamped),
            z: clamped
        };
    }
}

class QuantumAPL {
    constructor(config = {}) {
        this.dimPhi = config.dimPhi || 4;
        this.dimE = config.dimE || 4;
        this.dimPi = config.dimPi || 4;
        this.dim = this.dimPhi * this.dimE * this.dimPi;
        this.dimTruth = 3;
        this.dimTotal = this.dim * this.dimTruth;
        this.rho = this.initializeDensityMatrix();
        this.H = this.constructHamiltonian();
        this.lindbladOps = this.constructLindbladOperators();
        this.z = 0.5;
        this.phi = 0;
        this.entropy = 0;
        this.measurementHistory = [];
        this.time = 0;
        this.helixAdvisor = new HelixOperatorAdvisor();
        this.lastHelixHints = this.helixAdvisor.describe(this.z);
    }

    initializeDensityMatrix() {
        const rho = new ComplexMatrix(this.dimTotal, this.dimTotal);
        const combos = [
            [0, 0, 0],
            [Math.min(1, this.dimPhi - 1), 0, 0],
            [0, Math.min(1, this.dimE - 1), 0],
            [0, 0, Math.min(1, this.dimPi - 1)],
            [Math.min(1, this.dimPhi - 1), Math.min(1, this.dimE - 1), 0],
            [Math.min(2, this.dimPhi - 1), 0, Math.min(1, this.dimPi - 1)]
        ].filter(([phi, e, pi]) => phi >= 0 && e >= 0 && pi >= 0);
        const baseWeights = [0.4, 0.12, 0.12, 0.12, 0.12, 0.12];
        const weightSum = baseWeights.slice(0, combos.length).reduce((sum, w) => sum + w, 0);
        const truthIndex = Math.min(1, this.dimTruth - 1);
        combos.forEach(([phi, e, pi], idx) => {
            const weight = baseWeights[idx] / weightSum;
            const baseIdx = ((phi * this.dimE) + e) * this.dimPi + pi;
            const diagIdx = baseIdx * this.dimTruth + truthIndex;
            rho.set(diagIdx, diagIdx, new Complex(weight, 0));
        });
        return rho;
    }

    constructHamiltonian() {
        const H = new ComplexMatrix(this.dimTotal, this.dimTotal);
        const omega = 2 * Math.PI * 0.1;
        for (let i = 0; i < this.dim; i++) {
            const iPhi = Math.floor(i / (this.dimE * this.dimPi)) % this.dimPhi;
            const iE = Math.floor(i / this.dimPi) % this.dimE;
            const iPi = i % this.dimPi;
            const energy = omega * (iPhi + iE + iPi + 1.5);
            for (let t = 0; t < this.dimTruth; t++) {
                H.set(i * this.dimTruth + t, i * this.dimTruth + t, new Complex(energy, 0));
            }
        }
        const g = 0.05;
        for (let i = 0; i < this.dimTotal - 1; i++) {
            H.set(i, i + 1, new Complex(g, 0));
            H.set(i + 1, i, new Complex(g, 0));
        }
        return H;
    }

    constructLindbladOperators() {
        const ops = [];
        const L1 = new ComplexMatrix(this.dimTotal, this.dimTotal);
        const gamma1 = 0.01;
        for (let i = 0; i < this.dimTotal - this.dimTruth; i++) {
            L1.set(i, i + this.dimTruth, new Complex(Math.sqrt(gamma1), 0));
        }
        ops.push(L1);
        const L2 = new ComplexMatrix(this.dimTotal, this.dimTotal);
        const gamma2 = 0.02;
        for (let i = this.dimTruth; i < this.dimTotal; i++) {
            L2.set(i - this.dimTruth, i, new Complex(Math.sqrt(gamma2), 0));
        }
        ops.push(L2);
        const L3 = new ComplexMatrix(this.dimTotal, this.dimTotal);
        const gamma3 = 0.005;
        for (let i = 0; i < this.dimTotal; i++) {
            const sign = i % 2 === 0 ? 1 : -1;
            L3.set(i, i, new Complex(sign * Math.sqrt(gamma3), 0));
        }
        ops.push(L3);
        return ops;
    }

    evolve(dt) {
        const commutator = this.H.commutator(this.rho);
        const unitaryPart = commutator.scale(-dt);
        let dissipativePart = new ComplexMatrix(this.dimTotal, this.dimTotal);
        for (const L of this.lindbladOps) {
            const Ldag = L.dagger();
            const term1 = L.mul(this.rho).mul(Ldag);
            const term2 = Ldag.mul(L).anticommutator(this.rho).scale(0.5);
            dissipativePart = dissipativePart.add(term1.sub(term2).scale(dt));
        }
        this.rho = this.rho.add(unitaryPart).add(dissipativePart);
        this.normalizeDensityMatrix();
        this.time += dt;
    }

    normalizeDensityMatrix() {
        const tr = this.rho.trace();
        if (tr.abs() > 1e-10) this.rho = this.rho.scale(1 / tr.abs());
        const rhoDag = this.rho.dagger();
        for (let i = 0; i < this.dimTotal; i++) {
            for (let j = 0; j < this.dimTotal; j++) {
                this.rho.set(i, j, this.rho.get(i, j).add(rhoDag.get(i, j)).scale(0.5));
            }
        }
    }

    measure(projector, label = 'measurement') {
        const probability = projector.mul(this.rho).trace().re;
        let collapsed = false;
        if (Math.random() < probability) {
            this.rho = projector.mul(this.rho).mul(projector);
            if (probability > 1e-10) this.rho = this.rho.scale(1 / probability);
            collapsed = true;
            this.measurementHistory.push({ time: this.time, label, probability, outcome: 'success' });
        }
        return { probability, collapsed, outcome: collapsed ? 'success' : 'failure' };
    }

    measureNonSelective(projectors) {
        let newRho = new ComplexMatrix(this.dimTotal, this.dimTotal);
        for (const P of projectors) {
            newRho = newRho.add(P.mul(this.rho).mul(P));
        }
        this.rho = newRho;
        this.normalizeDensityMatrix();
    }

    measureZ() {
        let z = 0;
        for (let i = 0; i < this.dim; i++) {
            const iPhi = Math.floor(i / (this.dimE * this.dimPi)) % this.dimPhi;
            const iE = Math.floor(i / this.dimPi) % this.dimE;
            const iPi = i % this.dimPi;
            const zLevel = (iPhi + iE + iPi) / (this.dimPhi + this.dimE + this.dimPi - 3);
            for (let t = 0; t < this.dimTruth; t++) {
                z += zLevel * this.rho.get(i * this.dimTruth + t, i * this.dimTruth + t).re;
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

    measureTruth() {
        const probs = { TRUE: 0, UNTRUE: 0, PARADOX: 0 };
        const labels = ['TRUE', 'UNTRUE', 'PARADOX'];
        for (let i = 0; i < this.dim; i++) {
            for (let t = 0; t < this.dimTruth; t++) {
                probs[labels[t]] += this.rho.get(i * this.dimTruth + t, i * this.dimTruth + t).re;
            }
        }
        return probs;
    }

    selectN0Operator(legalOps, scalarState) {
        const operators = { '()': 0, '×': 1, '^': 2, '÷': 3, '+': 4, '−': 5 };
        const projectors = [];
        const opList = [];
        const currentZ = Number.isFinite(this.z) ? this.z : this.measureZ();
        const helixHints = this.helixAdvisor.describe(currentZ);
        this.lastHelixHints = helixHints;
        for (const op of legalOps) {
            if (operators[op] === undefined) continue;
            const P = new ComplexMatrix(this.dimTotal, this.dimTotal);
            const weight = this.computeOperatorWeight(op, scalarState, helixHints);
            let hasSupport = false;
            for (let i = 0; i < this.dim; i++) {
                if (this.isStateCompatible(i, op)) {
                    hasSupport = true;
                    for (let t = 0; t < this.dimTruth; t++) {
                        P.set(i * this.dimTruth + t, i * this.dimTruth + t, new Complex(weight, 0));
                    }
                }
            }
            if (!hasSupport) {
                const diagVal = new Complex(weight / this.dimTotal, 0);
                for (let d = 0; d < this.dimTotal; d++) {
                    P.set(d, d, diagVal);
                }
            }
            projectors.push(P);
            opList.push(op);
        }
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
        const probabilities = [];
        let totalProb = 0;
        for (const P of projectors) {
            const prob = P.mul(this.rho).trace().re;
            probabilities.push(Math.max(0, prob));
            totalProb += probabilities[probabilities.length - 1];
        }
        if (totalProb > 1e-10) {
            for (let i = 0; i < probabilities.length; i++) probabilities[i] /= totalProb;
        } else {
            probabilities.fill(1 / probabilities.length);
        }
        const blend = 0.3;
        const uniform = 1 / probabilities.length;
        for (let i = 0; i < probabilities.length; i++) {
            probabilities[i] = (1 - blend) * probabilities[i] + blend * uniform;
        }
        const r = Math.random();
        let cumulative = 0;
        let selectedIdx = 0;
        for (let i = 0; i < probabilities.length; i++) {
            cumulative += probabilities[i];
            if (r < cumulative) { selectedIdx = i; break; }
        }
        const selectedOp = opList[selectedIdx];
        const selectedProb = probabilities[selectedIdx];
        this.measure(projectors[selectedIdx], `N0:${selectedOp}`);
        return {
            operator: selectedOp,
            probability: selectedProb,
            probabilities: probabilities.reduce((obj, p, i) => { obj[opList[i]] = p; return obj; }, {}),
            helixHints
        };
    }

    computeOperatorWeight(op, scalarState, helixHints) {
        const CONST = require('../constants');
        const { Gs, Cs, Rs, kappa, tau, theta, delta, alpha, Omega } = scalarState;
        const weights = {
            '()': Gs + theta * 0.5,
            '×': Cs + kappa * 0.8,
            '^': kappa + tau * 0.6,
            '÷': delta + (1 - Omega) * 0.5,
            '+': alpha + Gs * 0.4,
            '−': Rs + delta * 0.3
        };
        let weight = weights[op] ?? 0.5;
        if (helixHints && helixHints.operators) {
            const preferred = helixHints.operators.includes(op);
            const truth = helixHints.truthChannel;
            weight *= preferred ? CONST.OPERATOR_PREFERRED_WEIGHT : CONST.OPERATOR_DEFAULT_WEIGHT;
            const biasTable = CONST.TRUTH_BIAS && CONST.TRUTH_BIAS[truth];
            if (biasTable && typeof biasTable[op] === 'number') {
                weight *= biasTable[op];
            }
        }
        return Math.max(0.05, Math.min(1.5, weight));
    }

    isStateCompatible(stateIdx, operator) {
        const iPhi = Math.floor(stateIdx / (this.dimE * this.dimPi)) % this.dimPhi;
        const iE = Math.floor(stateIdx / this.dimPi) % this.dimE;
        const iPi = stateIdx % this.dimPi;
        switch (operator) {
            case '()': return true;
            case '×': return iPhi > 0 && iE > 0;
            case '^': return iE > 0 || iPi > 0;
            case '÷': return iPhi > 0 || iE > 1;
            case '+': return iPi > 0;
            case '−': return iPhi > 1;
            default: return true;
        }
    }

    computeVonNeumannEntropy() { this.entropy = QuantumUtils.vonNeumannEntropy(this.rho); return this.entropy; }
    computePurity() { return QuantumUtils.purity(this.rho); }

    computePartialEntropy(subsystem = 'Phi') {
        let rhoSub;
        if (subsystem === 'Phi') {
            const dimA = this.dimPhi * this.dimTruth;
            const dimB = this.dimE * this.dimPi;
            rhoSub = this.rho.partialTraceB(dimA, dimB);
        } else if (subsystem === 'e') {
            rhoSub = this.rho;
        } else {
            rhoSub = this.rho;
        }
        return QuantumUtils.vonNeumannEntropy(rhoSub);
    }

    computeIntegratedInformation() {
        const S_total = this.computeVonNeumannEntropy();
        const S_Phi = this.computePartialEntropy('Phi');
        const I = Math.max(0, S_Phi - S_total);
        this.phi = I;
        return this.phi;
    }

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
        const val = this.rho.get(i, j);
        return { re: val.re, im: val.im, abs: val.abs() };
    }

    getPopulations() {
        const pops = [];
        for (let i = 0; i < this.dimTotal; i++) pops.push(this.rho.get(i, i).re);
        return pops;
    }

    getCoherences() {
        const coherences = [];
        for (let i = 0; i < Math.min(10, this.dimTotal); i++) {
            for (let j = i + 1; j < Math.min(10, this.dimTotal); j++) {
                coherences.push({ i, j, value: this.rho.get(i, j).abs() });
            }
        }
        return coherences.sort((a, b) => b.value - a.value).slice(0, 5);
    }

    driveFromClassical(classicalState) {
        const { z, phi } = classicalState;
        const g = 0.05 * (1 + z);
        const drive = new ComplexMatrix(this.dimTotal, this.dimTotal);
        const omega_drive = 2 * Math.PI * phi / 10;
        for (let i = 0; i < this.dimTotal - 1; i++) {
            const coupling = g * Math.cos(omega_drive * this.time);
            drive.set(i, i + 1, new Complex(coupling, 0));
            drive.set(i + 1, i, new Complex(coupling, 0));
        }
        this.H = this.H.add(drive.scale(0.1));
    }

    resetHamiltonian() { this.H = this.constructHamiltonian(); }
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = { QuantumAPL, ComplexMatrix, Complex, QuantumUtils };
}
