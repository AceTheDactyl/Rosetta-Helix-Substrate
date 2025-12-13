const assert = require('assert');
const { QuantumAPL } = require('../src/quantum_apl_engine');
const { QuantumClassicalBridge } = require('../src/legacy/QuantumClassicalBridge');
const { ClassicalConsciousnessStack } = require('../classical/ClassicalEngines');

function approxEqual(a, b, eps = 1e-4) {
    return Math.abs(a - b) < eps;
}

function testTracePreservation() {
    const quantum = new QuantumAPL();
    const classical = new ClassicalConsciousnessStack();
    const bridge = new QuantumClassicalBridge(quantum, classical);

    for (let i = 0; i < 5; i++) {
        bridge.step(0.01);
    }

    const trace = quantum.rho.trace().re;
    assert(approxEqual(trace, 1, 1e-3), `Density matrix trace drifted: ${trace}`);
}

function testZTracking() {
    const quantum = new QuantumAPL();
    const classical = new ClassicalConsciousnessStack();
    const bridge = new QuantumClassicalBridge(quantum, classical);

    for (let i = 0; i < 3; i++) {
        bridge.step(0.02);
    }

    const zQuantum = quantum.measureZ();
    const zClassical = classical.computeZ();
    assert(Math.abs(zQuantum - zClassical) < 0.5, `Quantum/Classical z diverged: ${zQuantum} vs ${zClassical}`);
}

function testOperatorPropagation() {
    const quantum = new QuantumAPL();
    const classical = new ClassicalConsciousnessStack();
    const bridge = new QuantumClassicalBridge(quantum, classical);

    const beforePhi = classical.IIT.phi;
    bridge.applyOperator({ operator: '^', probability: 1, probabilities: { '^': 1 } });
    const afterPhi = classical.IIT.phi;
    assert(afterPhi >= beforePhi, 'Amplification operator did not influence classical IIT state');
}

function testHelixOperatorWeighting() {
    const quantum = new QuantumAPL();
    const scalarState = {
        Gs: 0.4,
        Cs: 0.3,
        Rs: 0.2,
        kappa: 0.3,
        tau: 0.25,
        theta: 0.2,
        delta: 0.3,
        alpha: 0.35,
        Omega: 0.45
    };
    const hints = { operators: ['+', '()'], truthChannel: 'TRUE' };
    const plusWeight = quantum.computeOperatorWeight('+', scalarState, hints);
    const minusWeight = quantum.computeOperatorWeight('−', scalarState, hints);
    assert(
        plusWeight > minusWeight,
        `Helix hints should prioritize '+' but weights were ${plusWeight} <= ${minusWeight}`
    );
}

function monteCarloHelixBias(zValue, expectedOps, iterations = 80) {
    const quantum = new QuantumAPL({ dimPhi: 3, dimE: 3, dimPi: 2 });
    const scalarState = {
        Gs: 0.4,
        Cs: 0.35,
        Rs: 0.25,
        kappa: 0.3,
        tau: 0.2,
        theta: 0.25,
        delta: 0.3,
        alpha: 0.33,
        Omega: 0.5
    };
    const legalOps = ['()', '×', '^', '÷', '+', '−'];
    const counts = Object.fromEntries(legalOps.map(op => [op, 0]));

    for (let i = 0; i < iterations; i++) {
        quantum.rho = quantum.initializeDensityMatrix();
        quantum.z = zValue;
        quantum.lastHelixHints = quantum.helixAdvisor.describe(zValue);
        const result = quantum.selectN0Operator(legalOps, scalarState);
        counts[result.operator] = (counts[result.operator] || 0) + 1;
    }

    const total = Object.values(counts).reduce((sum, value) => sum + value, 0) || 1;
    const preferredFraction = expectedOps.reduce((sum, op) => sum + (counts[op] || 0), 0) / total;
    assert(
        preferredFraction > 0.5,
        `Preferred operators ${expectedOps.join(', ')} only captured ${(preferredFraction * 100).toFixed(1)}% at z=${zValue}`
    );
}

function testHelixBiasMonteCarlo() {
    monteCarloHelixBias(0.05, ['()', '−', '÷']);
    monteCarloHelixBias(0.92, ['+', '()', '×']);
}

function run() {
    testTracePreservation();
    testZTracking();
    testOperatorPropagation();
    testHelixOperatorWeighting();
    testHelixBiasMonteCarlo();
    console.log('QuantumClassicalBridge tests passed');
}

if (require.main === module) {
    run();
}

module.exports = {
    testTracePreservation,
    testZTracking,
    testOperatorPropagation,
    testHelixOperatorWeighting,
    testHelixBiasMonteCarlo
};
