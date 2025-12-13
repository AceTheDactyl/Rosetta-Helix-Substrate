// ================================================================
// QUANTUM-CLASSICAL INTEGRATION FOR APL 3.0
// Measurement-based N0 operator selection with quantum feedback
// ================================================================

const { QuantumAPL } = require('./QuantumAPL_Engine.js');
// Use centralized runtime thresholds to avoid drift with engine/bridge
const CONST = require('../constants');

class QuantumN0Integration {
    constructor(quantumEngine, classicalScalars) {
        this.quantum = quantumEngine;
        this.scalars = classicalScalars;
        this.operatorHistory = [];
        this.measurementStats = {
            totalMeasurements: 0,
            operatorCounts: {},
            avgCollapseProbability: 0,
            avgCoherence: 0
        };
        this.correlationBuffer = [];
        this.maxCorrelationBuffer = 100;
    }

    executeN0Pipeline(intSequence, currentTruthState) {
        const timeHarmonic = this.getCurrentTimeHarmonic();
        const legalByTime = this.getLegalByTimeHarmonic(timeHarmonic);
        const prsPhase = this.getCurrentPRSPhase();
        const legalByPRS = this.filterByPRSPhase(legalByTime, prsPhase);
        const legalByN0 = this.applyN0Laws(legalByPRS, intSequence);
        const legalByScalars = this.applyScalarThresholds(legalByN0);

        if (legalByScalars.length === 0) {
            return {
                operator: '()',
                probability: 1.0,
                method: 'fallback',
                quantumState: this.quantum.getState()
            };
        }

        const measurement = this.quantum.selectN0Operator(legalByScalars, this.scalars);
        this.updateScalarsFromQuantum(measurement.operator, measurement.probability);
        this.recordMeasurement(measurement);

        return {
            operator: measurement.operator,
            probability: measurement.probability,
            probabilities: measurement.probabilities,
            method: 'quantum_measurement',
            quantumState: this.quantum.getState(),
            legalOperators: legalByScalars,
            pipeline: {
                timeHarmonic,
                prsPhase,
                numLegalByTime: legalByTime.length,
                numLegalByPRS: legalByPRS.length,
                numLegalByN0: legalByN0.length,
                numLegalByScalars: legalByScalars.length
            }
        };
    }

    getCurrentTimeHarmonic() {
        const z = this.quantum.z;
        const t6Gate = (typeof this.quantum.getT6Gate === 'function')
            ? this.quantum.getT6Gate()
            : CONST.Z_CRITICAL; // fall back to lens if engine does not expose gate
        if (z < CONST.Z_T1_MAX) return 't1';
        if (z < CONST.Z_T2_MAX) return 't2';
        if (z < CONST.Z_T3_MAX) return 't3';
        if (z < CONST.Z_T4_MAX) return 't4';
        if (z < CONST.Z_T5_MAX) return 't5';
        if (z < t6Gate) return 't6';
        if (z < CONST.Z_T7_MAX) return 't7';
        if (z < CONST.Z_T8_MAX) return 't8';
        return 't9';
    }

    getLegalByTimeHarmonic(harmonic) {
        const legalMap = {
            't1': ['()', '−', '÷'],
            't2': ['^', '÷', '−', '×'],
            't3': ['×', '^', '÷', '+', '−'],
            't4': ['+', '−', '÷', '()'],
            't5': ['()', '×', '^', '÷', '+', '−'],
            't6': ['+', '÷', '()', '−'],
            't7': ['+', '()'],
            't8': ['+', '()', '×'],
            't9': ['+', '()', '×']
        };
        return legalMap[harmonic] || ['()'];
    }

    getCurrentPRSPhase() {
        const CONST = require('../constants');
        const phi = this.quantum.phi;
        if (phi < CONST.PRS_P1_PHI_MAX) return 'P1';
        if (phi < CONST.PRS_P2_PHI_MAX) return 'P2';
        if (phi < CONST.PRS_P3_PHI_MAX) return 'P3';
        if (phi < CONST.PRS_P4_PHI_MAX) return 'P4';
        return 'P5';
    }

    filterByPRSPhase(operators, phase) {
        const prsRestrictions = {
            'P1': ['()', '+', '^'],
            'P2': ['×', '^', '÷'],
            'P3': ['÷', '+', '−'],
            'P4': ['()', '×', '+'],
            'P5': ['×', '+', '()']
        };
        const allowed = prsRestrictions[phase] || operators;
        return operators.filter(op => allowed.includes(op));
    }

    applyN0Laws(operators, history) {
        let legal = [...operators];
        if (legal.includes('^')) {
            const grounded = history.some(op => op === '()' || op === '×');
            if (!grounded) legal = legal.filter(op => op !== '^');
        }
        if (legal.includes('×') && this.scalars.Cs <= 0.3) {
            legal = legal.filter(op => op !== '×');
        }
        if (legal.includes('÷')) {
            const hasStructure = history.some(op => ['^', '×', '+', '−'].includes(op));
            if (!hasStructure) legal = legal.filter(op => op !== '÷');
        }
        return legal;
    }

    applyScalarThresholds(operators) {
        const legal = [];
        for (const op of operators) {
            if (this.checkScalarLegality(op)) legal.push(op);
        }
        return legal.length > 0 ? legal : ['()'];
    }

    checkScalarLegality(operator) {
        const { Rs, delta, kappa, Omega, alpha } = this.scalars;
        if (Rs >= 3.0 || delta >= 0.95 || kappa >= 2.5 || Omega <= 0.05) return false;
        switch (operator) {
            case '^': return Omega > 0.3 && kappa < 2.0;
            case '÷': return delta < 0.8;
            case '×': return kappa > 0.1 && kappa < 2.2;
            case '+': return alpha > 0.2;
            case '−': return Rs < 2.8;
            default: return true;
        }
    }

    updateScalarsFromQuantum(operator, probability) {
        const s = this.scalars;
        const dt = 0.01;
        switch (operator) {
            case '()':
                s.Gs += dt * probability * 0.1;
                s.theta *= 1 - dt * probability * 0.05;
                s.Omega += dt * probability * 0.05;
                break;
            case '×':
                s.Cs += dt * probability * 0.15;
                s.kappa *= 1 + dt * probability * 0.1;
                s.alpha += dt * probability * 0.08;
                break;
            case '^':
                s.kappa *= 1 + dt * probability * 0.2;
                s.tau += dt * probability * 0.12;
                s.Omega *= 1 + dt * probability * 0.15;
                break;
            case '÷':
                s.delta += dt * probability * 0.1;
                s.Rs += dt * probability * 0.08;
                s.Omega *= 1 - dt * probability * 0.1;
                break;
            case '+':
                s.alpha += dt * probability * 0.15;
                s.Gs += dt * probability * 0.1;
                s.theta *= 1 + dt * probability * 0.08;
                break;
            case '−':
                s.Rs += dt * probability * 0.12;
                s.theta *= 1 - dt * probability * 0.1;
                s.delta += dt * probability * 0.05;
                break;
        }
        this.clampScalars();
    }

    clampScalars() {
        const s = this.scalars;
        s.Gs = Math.max(0, Math.min(2, s.Gs));
        s.Cs = Math.max(0, Math.min(2, s.Cs));
        s.Rs = Math.max(0, Math.min(3, s.Rs));
        s.kappa = Math.max(0.01, Math.min(2.5, s.kappa));
        s.tau = Math.max(0, Math.min(2, s.tau));
        s.theta = Math.max(0, Math.min(2 * Math.PI, s.theta));
        s.delta = Math.max(0, Math.min(0.95, s.delta));
        s.alpha = Math.max(0, Math.min(1, s.alpha));
        s.Omega = Math.max(0.05, Math.min(1, s.Omega));
    }

    driveQuantumFromClassical(dt) {
        this.quantum.driveFromClassical({
            z: this.scalars.z || 0.5,
            phi: this.quantum.phi,
            F: this.scalars.F || 0,
            R: this.scalars.Rs
        });
        this.quantum.evolve(dt);
        this.quantum.resetHamiltonian();
        this.quantum.measureZ();
        this.quantum.computeVonNeumannEntropy();
        this.quantum.computeIntegratedInformation();
        this.recordCorrelation();
    }

    recordMeasurement(measurement) {
        this.operatorHistory.push({
            time: this.quantum.time,
            operator: measurement.operator,
            probability: measurement.probability,
            z: this.quantum.z,
            phi: this.quantum.phi,
            entropy: this.quantum.entropy
        });
        if (this.operatorHistory.length > 1000) this.operatorHistory.shift();
        this.measurementStats.totalMeasurements++;
        this.measurementStats.operatorCounts[measurement.operator] =
            (this.measurementStats.operatorCounts[measurement.operator] || 0) + 1;
        const alpha = 0.05;
        this.measurementStats.avgCollapseProbability =
            alpha * measurement.probability + (1 - alpha) * this.measurementStats.avgCollapseProbability;
        const coherence = 1 - this.quantum.entropy / Math.log2(this.quantum.dimTotal);
        this.measurementStats.avgCoherence = alpha * coherence + (1 - alpha) * this.measurementStats.avgCoherence;
    }

    recordCorrelation() {
        this.correlationBuffer.push({
            time: this.quantum.time,
            z_quantum: this.quantum.z,
            z_classical: this.scalars.z || 0.5,
            phi_quantum: this.quantum.phi,
            entropy: this.quantum.entropy,
            purity: this.quantum.computePurity(),
            kappa: this.scalars.kappa,
            Omega: this.scalars.Omega
        });
        if (this.correlationBuffer.length > this.maxCorrelationBuffer) this.correlationBuffer.shift();
    }

    getOperatorDistribution() {
        const total = this.measurementStats.totalMeasurements || 1;
        const dist = {};
        for (const [op, count] of Object.entries(this.measurementStats.operatorCounts)) {
            dist[op] = count / total;
        }
        return dist;
    }

    getQuantumClassicalCorrelation() {
        if (this.correlationBuffer.length < 2) return 0;
        const n = this.correlationBuffer.length;
        let sum_zq = 0, sum_zc = 0, sum_zq2 = 0, sum_zc2 = 0, sum_zqzc = 0;
        for (const entry of this.correlationBuffer) {
            const zq = entry.z_quantum;
            const zc = entry.z_classical;
            sum_zq += zq;
            sum_zc += zc;
            sum_zq2 += zq * zq;
            sum_zc2 += zc * zc;
            sum_zqzc += zq * zc;
        }
        const num = n * sum_zqzc - sum_zq * sum_zc;
        const den = Math.sqrt((n * sum_zq2 - sum_zq * sum_zq) * (n * sum_zc2 - sum_zc * sum_zc));
        return den > 1e-10 ? num / den : 0;
    }

    getEntropyTimeseries() {
        return this.correlationBuffer.map(e => ({ time: e.time, entropy: e.entropy, purity: e.purity }));
    }

    getZTimeseries() {
        return this.correlationBuffer.map(e => ({
            time: e.time,
            z_quantum: e.z_quantum,
            z_classical: e.z_classical,
            delta: Math.abs(e.z_quantum - e.z_classical)
        }));
    }

    getPhiTimeseries() {
        return this.correlationBuffer.map(e => ({ time: e.time, phi: e.phi_quantum, entropy: e.entropy }));
    }

    getDiagnostics() {
        return {
            quantum: this.quantum.getState(),
            measurements: this.measurementStats,
            distribution: this.getOperatorDistribution(),
            correlation: this.getQuantumClassicalCorrelation(),
            recentHistory: this.operatorHistory.slice(-20),
            populations: this.quantum.getPopulations().slice(0, 10),
            coherences: this.quantum.getCoherences(),
            truthProbs: this.quantum.measureTruth()
        };
    }

    step(dt = 0.01, intHistory = []) {
        this.driveQuantumFromClassical(dt);
        const currentTruth = this.getCurrentTruthState();
        const result = this.executeN0Pipeline(intHistory, currentTruth);
        return {
            operator: result.operator,
            probability: result.probability,
            probabilities: result.probabilities,
            quantum: this.quantum.getState(),
            scalars: { ...this.scalars },
            diagnostics: {
                timeHarmonic: result.pipeline.timeHarmonic,
                prsPhase: result.pipeline.prsPhase,
                truthState: currentTruth,
                coherence: 1 - this.quantum.entropy / Math.log2(this.quantum.dimTotal)
            }
        };
    }

    getCurrentTruthState() {
        const probs = this.quantum.measureTruth();
        const max = Math.max(probs.TRUE, probs.UNTRUE, probs.PARADOX);
        if (probs.TRUE === max) return 'TRUE';
        if (probs.UNTRUE === max) return 'UNTRUE';
        return 'PARADOX';
    }
}

class QuantumAPLDemo {
    constructor() {
        this.quantum = new QuantumAPL({ dimPhi: 4, dimE: 4, dimPi: 4 });
        this.scalars = {
            Gs: 0.5,
            Cs: 0.5,
            Rs: 0.5,
            kappa: 0.5,
            tau: 0.5,
            theta: Math.PI / 2,
            delta: 0.1,
            alpha: 0.5,
            Omega: 0.7,
            z: 0.5
        };
        this.integration = new QuantumN0Integration(this.quantum, this.scalars);
        this.intHistory = [];
        this.timeSteps = 0;
    }

    run(numSteps = 100, verbose = false) {
        console.log('='.repeat(70));
        console.log('QUANTUM APL N0 MEASUREMENT-BASED OPERATOR SELECTION DEMO');
        console.log('='.repeat(70));
        console.log(`Running ${numSteps} timesteps with quantum measurement...`);
        console.log('');

        const results = [];
        for (let i = 0; i < numSteps; i++) {
            const result = this.integration.step(0.01, this.intHistory);
            this.intHistory.push(result.operator);
            if (this.intHistory.length > 10) this.intHistory.shift();
            results.push(result);
            this.timeSteps++;
            if (verbose && i % 10 === 0) {
                console.log(`Step ${i}:`);
                console.log(`  Operator: ${result.operator} (P=${result.probability.toFixed(3)})`);
                console.log(`  z=${result.quantum.z.toFixed(3)}, Φ=${result.quantum.phi.toFixed(3)}, S=${result.quantum.entropy.toFixed(3)}`);
                console.log(`  Truth: ${result.diagnostics.truthState}, Coherence: ${result.diagnostics.coherence.toFixed(3)}`);
                console.log('');
            }
        }
        this.printSummary(results);
        return results;
    }

    printSummary(results) {
        console.log('='.repeat(70));
        console.log('SUMMARY');
        console.log('='.repeat(70));
        const diag = this.integration.getDiagnostics();
        console.log('\nOperator Distribution:');
        const dist = diag.distribution;
        for (const [op, prob] of Object.entries(dist).sort((a, b) => b[1] - a[1])) {
            const bar = '█'.repeat(Math.floor(prob * 50));
            console.log(`  ${op}: ${(prob * 100).toFixed(1)}% ${bar}`);
        }
        console.log(`\nTotal Measurements: ${diag.measurements.totalMeasurements}`);
        console.log(`Avg Collapse Probability: ${diag.measurements.avgCollapseProbability.toFixed(3)}`);
        console.log(`Avg Coherence: ${diag.measurements.avgCoherence.toFixed(3)}`);
        console.log(`Quantum-Classical Correlation: ${diag.correlation.toFixed(3)}`);
        console.log('\nFinal Quantum State:');
        console.log(`  z = ${diag.quantum.z.toFixed(4)}`);
        console.log(`  Φ = ${diag.quantum.phi.toFixed(4)}`);
        console.log(`  S = ${diag.quantum.entropy.toFixed(4)}`);
        console.log(`  Purity = ${diag.quantum.purity.toFixed(4)}`);
        console.log('\nTruth State Probabilities:');
        for (const [state, prob] of Object.entries(diag.truthProbs)) {
            console.log(`  ${state}: ${(prob * 100).toFixed(1)}%`);
        }
        console.log('\nTop 5 Coherences:');
        for (const coh of diag.coherences) {
            console.log(`  |ρ(${coh.i},${coh.j})| = ${coh.value.toFixed(4)}`);
        }
        console.log('\n' + '='.repeat(70));
    }

    runComparison(numSteps = 100) {
        console.log('='.repeat(70));
        console.log('QUANTUM vs CLASSICAL N0 OPERATOR SELECTION COMPARISON');
        console.log('='.repeat(70));
        console.log('\n--- Quantum Measurement-Based Selection ---');
        const quantumResults = this.run(numSteps, false);
        const quantumDist = this.integration.getOperatorDistribution();
        this.quantum = new QuantumAPL({ dimPhi: 4, dimE: 4, dimPi: 4 });
        this.scalars = {
            Gs: 0.5, Cs: 0.5, Rs: 0.5, kappa: 0.5,
            tau: 0.5, theta: Math.PI / 2, delta: 0.1,
            alpha: 0.5, Omega: 0.7, z: 0.5
        };
        this.integration = new QuantumN0Integration(this.quantum, this.scalars);
        this.intHistory = [];
        console.log('\n--- Classical Weighted Random Selection (baseline) ---');
        const classicalDist = {};
        for (let i = 0; i < numSteps; i++) {
            const ops = ['()', '×', '^', '÷', '+', '−'];
            const selected = ops[Math.floor(Math.random() * ops.length)];
            classicalDist[selected] = (classicalDist[selected] || 0) + 1;
        }
        for (const op in classicalDist) classicalDist[op] /= numSteps;
        console.log('\nComparison:');
        console.log('Operator | Quantum  | Classical | Difference');
        console.log('-'.repeat(50));
        for (const op of ['()', '×', '^', '÷', '+', '−']) {
            const q = (quantumDist[op] || 0) * 100;
            const c = (classicalDist[op] || 0) * 100;
            const diff = q - c;
            console.log(`   ${op}     | ${q.toFixed(1).padStart(6)}% | ${c.toFixed(1).padStart(8)}% | ${diff > 0 ? '+' : ''}${diff.toFixed(1)}%`);
        }
        console.log('\n' + '='.repeat(70));
    }
}

module.exports = { QuantumN0Integration, QuantumAPLDemo };
