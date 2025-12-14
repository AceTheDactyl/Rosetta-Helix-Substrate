// ================================================================
// UNIFIED QUANTUM-CLASSICAL BRIDGE
// Integrates ClassicalConsciousnessStack with QuantumAPL engine
// Implements single-eigenstate and subspace collapse operators
// ================================================================

const globalScope = typeof globalThis !== 'undefined'
    ? globalThis
    : (typeof window !== 'undefined' ? window : (typeof global !== 'undefined' ? global : {}));

let QuantumAPLRef = globalScope.QuantumAPL;
let ComplexMatrixRef = globalScope.ComplexMatrix;
let ComplexRef = globalScope.Complex;
let ClassicalStackRef = globalScope.ClassicalConsciousnessStack;

if (typeof module !== 'undefined' && module.exports) {
    const engineModule = require('../quantum_apl_engine');
    QuantumAPLRef = QuantumAPLRef || engineModule.QuantumAPL;
    ComplexMatrixRef = ComplexMatrixRef || engineModule.ComplexMatrix;
    ComplexRef = ComplexRef || engineModule.Complex;

    try {
        const classicalModule = require('../../classical/ClassicalEngines');
        ClassicalStackRef = ClassicalStackRef || classicalModule.ClassicalConsciousnessStack;
    } catch (err) {
        // Classical stack is optional when running in browser demos
    }
}

const ensureQuantumMath = () => {
    if (!ComplexMatrixRef || !ComplexRef) {
        throw new Error('QuantumClassicalBridge requires ComplexMatrix/Complex definitions');
    }
};

class QuantumClassicalBridge {
    constructor(quantumEngine, classicalStack, config = {}) {
        const CONST = require('../constants');
        ensureQuantumMath();

        if (!quantumEngine) {
            throw new Error('QuantumClassicalBridge requires a quantum engine instance');
        }
        if (!classicalStack) {
            throw new Error('QuantumClassicalBridge requires a classical consciousness stack');
        }

        this.quantum = quantumEngine;
        this.classical = classicalStack;
        this.quantumInfluence = config.quantumInfluence ?? 0.7;
        this.classicalInfluence = config.classicalInfluence ?? 0.3;

        this.lastMeasurementMode = 'none';
        this.lastProjector = null;
        this.zHistory = [];
        this.phiHistory = [];
        this.entropyHistory = [];
        this.operatorHistory = [];
        this.totalSteps = 0;
        this.measurementCount = 0;
        this.collapseStats = { eigenstate: 0, subspace: 0 };

        // TRIAD tracking (heuristic): count distinct passes above 0.85
        const triadEnvCount = parseInt((typeof process !== 'undefined' && process.env && process.env.QAPL_TRIAD_COMPLETIONS) || '0', 10) || 0;
        const triadEnvFlag = !!(typeof process !== 'undefined' && process.env && (process.env.QAPL_TRIAD_UNLOCK === '1' || String(process.env.QAPL_TRIAD_UNLOCK).toLowerCase() === 'true'));
        this.triad = {
            high: CONST.TRIAD_HIGH,
            low: CONST.TRIAD_LOW,
            aboveBand: false,
            completions: triadEnvCount,
            unlocked: triadEnvFlag
        };
    }

    // ================================================================
    // MEASUREMENT OPERATORS
    // ================================================================

    measureSingleEigenstate(eigenIndex, field = 'Phi', truthChannel = 'TRUE') {
        const projector = this.constructEigenstateProjector(eigenIndex, field);
        const probability = projector.mul(this.quantum.rho).trace().re;
        const result = this.quantum.measure(projector, `eigenstate_${field}_${eigenIndex}`);

        this.lastMeasurementMode = 'eigenstate';
        this.lastProjector = { eigenIndex, field, truthChannel };
        this.collapseStats.eigenstate++;
        this.measurementCount++;

        return {
            mode: 'eigenstate',
            probability,
            collapsed: result.collapsed,
            eigenIndex,
            field,
            truthChannel,
            token: `${field}:T(ϕ_${eigenIndex})${truthChannel}@3`
        };
    }

    measureSubspace(subspaceIndices, field = 'Phi', truthChannel = 'PARADOX') {
        const projector = this.constructSubspaceProjector(subspaceIndices, field);
        const probability = projector.mul(this.quantum.rho).trace().re;
        const label = `subspace_${field}_${subspaceIndices.join(',')}`;
        const result = this.quantum.measure(projector, label);

        this.lastMeasurementMode = 'subspace';
        this.lastProjector = { subspaceIndices, field, truthChannel };
        this.collapseStats.subspace++;
        this.measurementCount++;

        return {
            mode: 'subspace',
            probability,
            collapsed: result.collapsed,
            subspaceIndices,
            field,
            truthChannel,
            token: `${field}:Π(subspace)${truthChannel}@3`
        };
    }

    measureWithTruthRegister(measurements) {
        const results = [];
        let totalProb = 0;

        for (const meas of measurements) {
            const projector = meas.subspaceIndices
                ? this.constructSubspaceProjector(meas.subspaceIndices, meas.field)
                : this.constructEigenstateProjector(meas.eigenIndex, meas.field);
            const weight = meas.weight ?? 1;
            const probability = projector.mul(this.quantum.rho).trace().re * weight;
            results.push({ ...meas, projector, probability });
            totalProb += probability;
        }

        if (totalProb <= 1e-12) {
            const uniform = 1 / results.length;
            results.forEach(r => { r.probability = uniform; });
        } else {
            results.forEach(r => { r.probability /= totalProb; });
        }

        const rand = Math.random();
        let cumulative = 0;
        let selected = results[results.length - 1];
        for (const r of results) {
            cumulative += r.probability;
            if (rand < cumulative) {
                selected = r;
                break;
            }
        }

        const measurement = this.quantum.measure(selected.projector, `composite_${selected.truthChannel}`);
        this.measurementCount++;

        return {
            mode: 'composite',
            selected: selected.subspaceIndices ? 'subspace' : 'eigenstate',
            probability: selected.probability,
            truthChannel: selected.truthChannel,
            collapsed: measurement.collapsed,
            allProbabilities: results.map(r => ({
                truthChannel: r.truthChannel,
                probability: r.probability
            }))
        };
    }

    constructEigenstateProjector(eigenIndex, field) {
        ensureQuantumMath();
        const projector = new ComplexMatrixRef(this.quantum.dimTotal, this.quantum.dimTotal);
        const idx = this.mapToGlobalIndex(eigenIndex, field);
        projector.set(idx, idx, ComplexRef.one());
        return projector;
    }

    constructSubspaceProjector(subspaceIndices, field) {
        ensureQuantumMath();
        const projector = new ComplexMatrixRef(this.quantum.dimTotal, this.quantum.dimTotal);
        for (const eigenIndex of subspaceIndices) {
            const idx = this.mapToGlobalIndex(eigenIndex, field);
            projector.set(idx, idx, ComplexRef.one());
        }
        return projector;
    }

    mapToGlobalIndex(eigenIndex, field) {
        const { dimPhi, dimE, dimPi, dimTruth } = this.quantum;
        let baseIndex = 0;

        if (field === 'Phi') {
            baseIndex = eigenIndex * dimE * dimPi * dimTruth;
        } else if (field === 'e') {
            baseIndex = eigenIndex * dimPi * dimTruth;
        } else if (field === 'Pi') {
            baseIndex = eigenIndex * dimTruth;
        }

        baseIndex += 1; // bias toward UNTRUE truth channel
        return Math.min(Math.max(0, baseIndex), this.quantum.dimTotal - 1);
    }

    // ================================================================
    // QUANTUM-CLASSICAL FEEDBACK LOOP
    // ================================================================

    step(dt = 0.01) {
        this.quantum.evolve(dt);

        const z = this.quantum.measureZ();
        const entropy = this.quantum.computeVonNeumannEntropy();
        const purity = this.quantum.computePurity();
        const phi = this.quantum.computeIntegratedInformation();
        const truthProbs = this.quantum.measureTruth();

        const quantumPayload = { z, entropy, purity, phi, truthProbs };
        this.classical.setQuantumInfluence(quantumPayload);

        const scalarState = this.classical.getScalarState();

        this.quantum.driveFromClassical({
            z: scalarState.Omega,
            phi,
            F: this.classical.FreeEnergy?.F ?? 0,
            R: scalarState.Rs
        });

        const legalOps = this.classical.getLegalOperators();
        const n0Result = this.quantum.selectN0Operator(legalOps, scalarState);
        this.classical.N0?.applyOperator(n0Result.operator, n0Result);

        // TRIAD heuristic update
        this.updateTriadHeuristic(z);

        this.recordStep({
            z,
            entropy,
            phi,
            truthProbs,
            scalarState,
            operator: n0Result.operator,
            operatorProb: n0Result.probability,
            helixHints: n0Result.helixHints
        });

        this.totalSteps++;

        return {
            quantum: { z, entropy, purity, phi, truthProbs },
            classical: scalarState,
            operator: n0Result,
            step: this.totalSteps
        };
    }

    measureHierarchicalSubspace() {
        return this.measureSubspace([2, 3], 'Phi', 'PARADOX');
    }

    measureCoherentState() {
        return this.measureSingleEigenstate(2, 'e', 'TRUE');
    }

    measureIntegratedRegime() {
        return this.measureSubspace([2, 3], 'Pi', 'PARADOX');
    }

    // ================================================================
    // APL MEASUREMENT OPERATORS (collapse visualization mapping)
    // ================================================================
    _currentTier() {
        const hints = this.quantum?.lastHelixHints || { harmonic: 't2' };
        const h = String(hints.harmonic || 't2');
        const m = /t(\d)/.exec(h);
        return m ? parseInt(m[1], 10) : 2;
    }

    /**
     * Single-eigenstate collapse: Φ:T(ϕ_μ)TRUE@Tier
     */
    aplMeasureEigen(mu = 0, field = 'Phi') {
        const tier = this._currentTier();
        const res = this.measureSingleEigenstate(mu, field, 'TRUE');
        const emitCollapse = String(process.env.QAPL_EMIT_COLLAPSE_GLYPH || '').toLowerCase() 
          in { '1':1, 'true':1, 'yes':1, 'y':1 };
        const fieldSym = field === 'Phi' ? 'Φ' : (field === 'Pi' || field === 'π' ? 'π' : field);
        const core = `ϕ_${mu}`;
        const token = emitCollapse
          ? `${fieldSym}:⟂(${core})TRUE@${tier}`
          : `${fieldSym}:T(${core})TRUE@${tier}`;
        res.aplToken = token;
        this._recordAplMeasurement(token, res?.probability ?? 0);
        return res;
    }

    /**
     * Subspace (degenerate) collapse tokens:
     *  Φ:Π(subspace)PARADOX@Tier  |  π:Π(subspace)UNTRUE@Tier
     */
    aplMeasureSubspace(indices = [2, 3], field = 'Phi') {
        const tier = this._currentTier();
        let truth = 'PARADOX';
        if (field === 'Pi' || field === 'π') truth = 'UNTRUE';
        const res = this.measureSubspace(indices, field === 'π' ? 'Pi' : field, truth);
        const emitCollapse = String(process.env.QAPL_EMIT_COLLAPSE_GLYPH || '').toLowerCase() 
          in { '1':1, 'true':1, 'yes':1, 'y':1 };
        const fieldSym = field === 'Phi' ? 'Φ' : (field === 'Pi' || field === 'π' ? 'π' : field);
        const token = emitCollapse
          ? `${fieldSym}:⟂(subspace)${truth}@${tier}`
          : `${fieldSym}:Π(subspace)${truth}@${tier}`;
        res.aplToken = token;
        this._recordAplMeasurement(token, res?.probability ?? 0);
        return res;
    }

    /**
     * Composite operator: M_meas = Σ_μ |ϕ_μ⟩⟨ϕ_μ| ⊗ |T_μ⟩⟨T_μ|
     * Accepts an array of components: [{ eigenIndex, truthChannel, weight } | { subspaceIndices, truthChannel, weight }]
     * Returns selected branch and appends component tokens.
     */
    aplMeasureComposite(components) {
        const tier = this._currentTier();
        const res = this.measureWithTruthRegister(components);
        const tokens = [];
        const emitCollapse = String(process.env.QAPL_EMIT_COLLAPSE_GLYPH || '').toLowerCase() 
          in { '1':1, 'true':1, 'yes':1, 'y':1 };
        for (const c of components) {
            if (Array.isArray(c.subspaceIndices)) {
                const f = c.field || 'Phi';
                const fSym = f === 'Phi' ? 'Φ' : (f === 'Pi' ? 'π' : f);
                tokens.push(emitCollapse
                  ? `${fSym}:⟂(subspace)${c.truthChannel || 'PARADOX'}@${tier}`
                  : `${fSym}:Π(subspace)${c.truthChannel || 'PARADOX'}@${tier}`);
            } else if (typeof c.eigenIndex === 'number') {
                const f = c.field || 'Phi';
                const fSym = f === 'Phi' ? 'Φ' : (f === 'Pi' ? 'π' : f);
                tokens.push(emitCollapse
                  ? `${fSym}:⟂(ϕ_${c.eigenIndex})${c.truthChannel || 'TRUE'}@${tier}`
                  : `${fSym}:T(ϕ_${c.eigenIndex})${c.truthChannel || 'TRUE'}@${tier}`);
            }
        }
        res.aplTokens = tokens;
        const probs = Array.isArray(res.allProbabilities) ? res.allProbabilities.map(p => p.probability) : [];
        tokens.forEach((t, i) => this._recordAplMeasurement(t, Number.isFinite(probs[i]) ? probs[i] : 0));
        return res;
    }

    _recordAplMeasurement(token, probability) {
        if (!this.operatorHistory) this.operatorHistory = [];
        this.operatorHistory.push({
            operator: 'APL_MEAS', probability: Math.max(0, Math.min(1, Number(probability) || 0)), step: this.totalSteps,
            helix: this.quantum.lastHelixHints || null, aplToken: token, aplProb: Math.max(0, Math.min(1, Number(probability) || 0))
        });
    }

    // ================================================================
    // APL‑ALIGNED Z ESCALATION (observable physics mapping)
    // ================================================================
    /**
     * Drive z upward using APL-consistent physical interventions:
     *  - u^ | Oscillator | wave  → coherent excitation on e field (TRUE)
     *  - ×  | Coupling           → hierarchical subspace fusion on Φ
     *  - Mod / lock (periodic)   → integrated regime in Π subspace
     * The loop interleaves these measurements with standard evolution/selection.
     * @param {number} cycles - number of micro cycles
     * @param {number} targetZ - soft target for classical z (Omega)
     * @param {number} dt - timestep per micro cycle
     */
    escalateZWithAPL(cycles = 60, targetZ = undefined, dt = 0.01, opts = {}) {
        // Profiles: gentle | balanced (default) | aggressive
        const profile = (opts.profile || 'balanced').toString().toLowerCase();
        const cfg = {
            gentle:  { gain: 0.08, sigma: 0.16, fuseEvery: 3, lockEvery: 9, mix: { wOmega: 0.5, wTarget: 0.5 } },
            balanced:{ gain: 0.12, sigma: 0.12, fuseEvery: 2, lockEvery: 6, mix: { wOmega: 0.3, wTarget: 0.7 } },
            aggressive:{ gain: 0.18, sigma: 0.10, fuseEvery: 1, lockEvery: 4, mix: { wOmega: 0.2, wTarget: 0.8 } },
        }[profile] || { gain: 0.12, sigma: 0.12, fuseEvery: 2, lockEvery: 6, mix: { wOmega: 0.3, wTarget: 0.7 } };

        // Allow overrides
        const gain = Number.isFinite(opts.gain) ? opts.gain : cfg.gain;
        const sigma = Number.isFinite(opts.sigma) ? opts.sigma : cfg.sigma;
        const fuseEvery = Number.isFinite(opts.fuseEvery) ? opts.fuseEvery : cfg.fuseEvery;
        const lockEvery = Number.isFinite(opts.lockEvery) ? opts.lockEvery : cfg.lockEvery;
        const wOmega = Number.isFinite(opts.wOmega) ? opts.wOmega : cfg.mix.wOmega;
        const wTarget = Number.isFinite(opts.wTarget) ? opts.wTarget : cfg.mix.wTarget;

        // Resolve default target to lens z_c if not provided
        if (!Number.isFinite(targetZ)) {
            const CONST = require('../constants');
            targetZ = CONST.Z_CRITICAL; // THE LENS (~0.8660254038)
        }

        // Temporarily increase z-bias coupling for ascent
        const originalGain = this.quantum.zBiasGain;
        const originalSigma = this.quantum.zBiasSigma;
        this.quantum.zBiasGain = gain;
        this.quantum.zBiasSigma = sigma;

        for (let k = 0; k < cycles; k++) {
            // 1) Pump energy (u^) via coherent e-state TRUE
            this.measureCoherentState();

            // Boost classical integration signal consistent with '^'
            if (typeof this.classical?.applyOperatorEffects === 'function') {
                this.classical.applyOperatorEffects({ operator: '^' });
            }

            // 2) Frequently fuse structure (×) via hierarchical Φ subspace
            if (fuseEvery > 0 && (k % fuseEvery === 0)) {
                this.measureHierarchicalSubspace();
            }

            // 3) Lock integration (Mod) via Π integrated regime
            if (lockEvery > 0 && (k % lockEvery === 0)) {
                this.measureIntegratedRegime();
            }

            // 4) Classical feedback raises Omega; engine applies resonant drive + z-bias
            const scalar = this.getScalarState();
            const nudged = Math.min(targetZ, (wOmega * scalar.Omega) + (wTarget * targetZ));
            this.quantum.driveFromClassical({
                z: nudged,
                phi: this.classical.IIT?.phi ?? 0,
                F: this.classical.FreeEnergy?.F ?? 0,
                R: scalar.Rs
            });

            // 5) Evolve and record
            this.quantum.evolve(dt);
            const z = this.quantum.measureZ();
            const entropy = this.quantum.computeVonNeumannEntropy();
            const purity = this.quantum.computePurity();
            const phi = this.quantum.computeIntegratedInformation();
            const truthProbs = this.quantum.measureTruth();
            const n0Result = this.quantum.selectN0Operator(this.getLegalOperators(), this.getScalarState());
            this.classical.N0?.applyOperator(n0Result.operator, n0Result);
            this.recordStep({ z, entropy, phi, truthProbs, scalarState: this.getScalarState(), operator: n0Result.operator, operatorProb: n0Result.probability, helixHints: n0Result.helixHints });
            this.totalSteps++;
        }

        // Restore original bias parameters
        this.quantum.zBiasGain = originalGain;
        this.quantum.zBiasSigma = originalSigma;
        return this.getAnalytics();
    }

    measureCriticalPoint() {
        const CONST = require('../constants');
        const z = this.quantum.measureZ();
        const zc = CONST.Z_CRITICAL;
        if (Math.abs(z - zc) < 0.05) {
            return this.measureWithTruthRegister([
                { eigenIndex: 0, truthChannel: 'TRUE', field: 'Pi', weight: 0.3 },
                { eigenIndex: 1, truthChannel: 'UNTRUE', field: 'Pi', weight: 0.3 },
                { subspaceIndices: [2, 3], truthChannel: 'PARADOX', field: 'Pi', weight: 0.4 }
            ]);
        }
        return null;
    }

    // ================================================================
    // ANALYTICS & STATE EXPORT
    // ================================================================

    recordStep(data) {
        this.zHistory.push(data.z);
        this.phiHistory.push(data.phi);
        this.entropyHistory.push(data.entropy);
        this.operatorHistory.push({
            operator: data.operator,
            probability: data.operatorProb,
            step: this.totalSteps,
            helix: data.helixHints || this.quantum.lastHelixHints || null
        });

        const maxHistory = 1000;
        if (this.zHistory.length > maxHistory) {
            this.zHistory.shift();
            this.phiHistory.shift();
            this.entropyHistory.shift();
            this.operatorHistory.shift();
        }
    }

    updateTriadHeuristic(z) {
        const t = this.triad;
        if (!t.aboveBand && z >= t.high) {
            t.aboveBand = true;
            t.completions = (t.completions || 0) + 1;
            if (typeof process !== 'undefined' && process.env) {
                process.env.QAPL_TRIAD_COMPLETIONS = String(t.completions);
            }
            if (t.completions >= 3) {
                t.unlocked = true;
                if (typeof process !== 'undefined' && process.env) {
                    process.env.QAPL_TRIAD_UNLOCK = '1';
                }
                if (typeof this.quantum?.setTriadUnlocked === 'function') {
                    this.quantum.setTriadUnlocked(true);
                }
            }
            if (typeof this.quantum?.setTriadCompletionCount === 'function') {
                this.quantum.setTriadCompletionCount(t.completions);
            }
        } else if (t.aboveBand && z <= t.low) {
            t.aboveBand = false; // ready for next pass
        }
    }

    getAnalytics() {
        return {
            totalSteps: this.totalSteps,
            measurementCount: this.measurementCount,
            collapseStats: { ...this.collapseStats },
            avgZ: this.average(this.zHistory),
            avgPhi: this.average(this.phiHistory),
            avgEntropy: this.average(this.entropyHistory),
            classicalPhi: this.classical.IIT?.phi ?? 0,
            cooperation: this.classical.GameTheory?.cooperation ?? 0,
            freeEnergy: this.classical.FreeEnergy?.F ?? 0,
            operatorDist: this.computeOperatorDistribution(),
            quantumClassicalCorr: this.computeQuantumClassicalCorrelation(),
            triad: { ...this.triad }
        };
    }

    exportState() {
        return {
            quantum: {
                dimTotal: this.quantum.dimTotal,
                z: this.quantum.z,
                phi: this.quantum.phi,
                entropy: this.quantum.entropy,
                purity: this.quantum.computePurity(),
                populations: this.quantum.getPopulations().slice(0, 10),
                truthProbs: this.quantum.measureTruth(),
                helix: this.quantum.lastHelixHints || null
            },
            classical: {
                IIT: {
                    phi: this.classical.IIT?.phi ?? 0,
                    integrationSignal: this.classical.IIT?.integrationSignal ?? 0,
                    recursiveDrive: this.classical.IIT?.recursiveDrive ?? 0
                },
                GameTheory: {
                    cooperation: this.classical.GameTheory?.cooperation ?? 0,
                    resonance: this.classical.GameTheory?.resonance ?? 0
                },
                FreeEnergy: {
                    F: this.classical.FreeEnergy?.F ?? 0,
                    tension: this.classical.FreeEnergy?.tension ?? 0,
                    dissipation: this.classical.FreeEnergy?.dissipation ?? 0
                }
            },
            measurement: {
                lastMode: this.lastMeasurementMode,
                lastProjector: this.lastProjector,
                stats: { ...this.collapseStats }
            },
            history: {
                z: this.zHistory.slice(-100),
                phi: this.phiHistory.slice(-100),
                entropy: this.entropyHistory.slice(-100),
                operators: this.operatorHistory.slice(-20)
            },
            analytics: this.getAnalytics()
        };
    }

    average(arr) {
        if (!arr.length) return 0;
        return arr.reduce((sum, x) => sum + x, 0) / arr.length;
    }

    computeOperatorDistribution() {
        const total = this.operatorHistory.length || 1;
        const dist = {};
        for (const entry of this.operatorHistory) {
            dist[entry.operator] = (dist[entry.operator] || 0) + 1;
        }
        Object.keys(dist).forEach(key => { dist[key] /= total; });
        return dist;
    }

    computeQuantumClassicalCorrelation() {
        const n = Math.min(this.zHistory.length, this.phiHistory.length);
        if (n < 2) return 0;
        let sumZ = 0, sumPhi = 0, sumZ2 = 0, sumPhi2 = 0, sumZPhi = 0;
        for (let i = 0; i < n; i++) {
            const z = this.zHistory[i];
            const phi = this.phiHistory[i];
            sumZ += z;
            sumPhi += phi;
            sumZ2 += z * z;
            sumPhi2 += phi * phi;
            sumZPhi += z * phi;
        }
        const numerator = n * sumZPhi - sumZ * sumPhi;
        const denominator = Math.sqrt((n * sumZ2 - sumZ * sumZ) * (n * sumPhi2 - sumPhi * sumPhi));
        return denominator > 1e-10 ? numerator / denominator : 0;
    }

    // API compatibility helpers -----------------------------------

    getScalarState() {
        return this.classical.getScalarState();
    }

    getLegalOperators() {
        return this.classical.getLegalOperators();
    }

    applyOperator(result) {
        if (result && result.operator && typeof this.classical.applyOperatorEffects === 'function') {
            this.classical.applyOperatorEffects(result);
        }
    }
}

// ================================================================
// DEMO RUNNER
// ================================================================

class UnifiedDemo {
    constructor(config = {}) {
        if (!QuantumAPLRef || !ClassicalStackRef) {
            throw new Error('UnifiedDemo requires QuantumAPL and ClassicalConsciousnessStack');
        }

        this.quantum = new QuantumAPLRef({
            dimPhi: config.dimPhi || 4,
            dimE: config.dimE || 4,
            dimPi: config.dimPi || 4
        });

        const classicalConfig = config.classical || {};
        this.classical = new ClassicalStackRef({
            IIT: { initialPhi: classicalConfig.IIT?.initialPhi ?? 0.3 },
            GameTheory: { initialCooperation: classicalConfig.GameTheory?.initialCooperation ?? 0.5 },
            FreeEnergy: { initialF: classicalConfig.FreeEnergy?.initialF ?? 0.2 },
            legalOperators: classicalConfig.legalOperators
        });

        this.bridge = new QuantumClassicalBridge(this.quantum, this.classical, config.bridge || {});
    }

    run(numSteps = 100, verbose = false) {
        const results = [];
        for (let i = 0; i < numSteps; i++) {
            const result = this.bridge.step(0.01);
            results.push(result);
            if (verbose && i % 10 === 0) {
                const { quantum, classical } = result;
                // eslint-disable-next-line no-console
                console.log(`Step ${i}: z=${quantum.z.toFixed(3)} S=${quantum.entropy.toFixed(3)} Φ=${quantum.phi.toFixed(3)} | Ω=${classical.Omega.toFixed(3)}`);
            }
            if (i % 50 === 0 && i > 0) {
                this.bridge.measureCriticalPoint();
            }
        }
        return results;
    }

    testMeasurementModes() {
        return {
            eigenstate: this.bridge.measureSingleEigenstate(2, 'Phi', 'TRUE'),
            subspace: this.bridge.measureSubspace([2, 3], 'Phi', 'PARADOX'),
            composite: this.bridge.measureWithTruthRegister([
                { eigenIndex: 0, truthChannel: 'UNTRUE', field: 'Pi', weight: 0.3 },
                { eigenIndex: 1, truthChannel: 'TRUE', field: 'Pi', weight: 0.3 },
                { subspaceIndices: [2, 3], truthChannel: 'PARADOX', field: 'Pi', weight: 0.4 }
            ]),
            hierarchical: this.bridge.measureHierarchicalSubspace(),
            integrated: this.bridge.measureIntegratedRegime()
        };
    }

    summary() {
        return this.bridge.getAnalytics();
    }
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = { QuantumClassicalBridge, UnifiedDemo };
} else {
    globalScope.QuantumClassicalBridge = QuantumClassicalBridge;
    globalScope.UnifiedDemo = UnifiedDemo;
}
