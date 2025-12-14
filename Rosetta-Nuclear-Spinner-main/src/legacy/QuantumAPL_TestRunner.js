#!/usr/bin/env node

const fs = require('fs');
const { QuantumAPL, ComplexMatrix, Complex, QuantumUtils } = require('./QuantumAPL_Engine.js');
const { QuantumN0Integration, QuantumAPLDemo } = require('./QuantumN0_Integration.js');

function assert(condition, message) {
    if (!condition) throw new Error(`Assertion failed: ${message}`);
}

class QuantumAPLTestSuite {
    testComplexArithmetic() {
        console.log('Testing complex number arithmetic...');
        const a = new Complex(3, 4);
        const b = new Complex(1, 2);
        const sum = a.add(b);
        assert(Math.abs(sum.re - 4) < 1e-10, 'Complex addition (real)');
        assert(Math.abs(sum.im - 6) < 1e-10, 'Complex addition (imag)');
        const prod = a.mul(b);
        assert(Math.abs(prod.re + 5) < 1e-10, 'Complex multiplication (real)');
        assert(Math.abs(prod.im - 10) < 1e-10, 'Complex multiplication (imag)');
        const abs = a.abs();
        assert(Math.abs(abs - 5) < 1e-10, 'Complex absolute value');
        console.log('  ✓ Complex arithmetic passed');
    }

    testMatrixOperations() {
        console.log('Testing matrix operations...');
        const A = new ComplexMatrix(2, 2);
        A.set(0, 0, new Complex(1, 0));
        A.set(0, 1, new Complex(2, 0));
        A.set(1, 0, new Complex(3, 0));
        A.set(1, 1, new Complex(4, 0));
        const tr = A.trace();
        assert(Math.abs(tr.re - 5) < 1e-10, 'Matrix trace');
        const Adag = A.dagger();
        assert(Math.abs(Adag.get(1, 0).re - 2) < 1e-10, 'Matrix dagger');
        const I = ComplexMatrix.identity(2);
        const comm = I.commutator(A);
        assert(comm.trace().abs() < 1e-10, 'Identity commutator');
        console.log('  ✓ Matrix operations passed');
    }

    testProjectionOperators() {
        console.log('Testing projection operators...');
        const P = QuantumUtils.projector(0, 4);
        const P2 = P.mul(P);
        for (let i = 0; i < 4; i++) {
            for (let j = 0; j < 4; j++) {
                const diff = P.get(i, j).sub(P2.get(i, j)).abs();
                assert(diff < 1e-10, `Projection idempotency [${i},${j}]`);
            }
        }
        const Pdag = P.dagger();
        for (let i = 0; i < 4; i++) {
            for (let j = 0; j < 4; j++) {
                const diff = P.get(i, j).sub(Pdag.get(i, j)).abs();
                assert(diff < 1e-10, `Projection Hermiticity [${i},${j}]`);
            }
        }
        console.log('  ✓ Projection operators passed');
    }

    testDensityMatrixProperties() {
        console.log('Testing density matrix properties...');
        const quantum = new QuantumAPL({ dimPhi: 2, dimE: 2, dimPi: 2 });
        const tr = quantum.rho.trace().re;
        assert(Math.abs(tr - 1) < 1e-6, 'Density matrix trace = 1');
        const rhoDag = quantum.rho.dagger();
        let hermitian = true;
        for (let i = 0; i < quantum.dimTotal; i++) {
            for (let j = 0; j < quantum.dimTotal; j++) {
                const diff = quantum.rho.get(i, j).sub(rhoDag.get(i, j)).abs();
                if (diff > 1e-6) hermitian = false;
            }
        }
        assert(hermitian, 'Density matrix Hermiticity');
        const purity = QuantumUtils.purity(quantum.rho);
        assert(purity <= 1 + 1e-6 && purity >= 0.2, 'Initial state purity within valid bounds');
        console.log('  ✓ Density matrix properties passed');
    }

    testLindbladEvolution() {
        console.log('Testing Lindblad evolution...');
        const quantum = new QuantumAPL({ dimPhi: 2, dimE: 2, dimPi: 2 });
        const initialPurity = QuantumUtils.purity(quantum.rho);
        for (let i = 0; i < 100; i++) quantum.evolve(0.01);
        const finalPurity = QuantumUtils.purity(quantum.rho);
        assert(finalPurity <= initialPurity + 0.02, 'Purity non-increasing beyond tolerance');
        const tr = quantum.rho.trace().re;
        assert(Math.abs(tr - 1) < 1e-4, 'Trace preserved after evolution');
        console.log('  ✓ Lindblad evolution passed');
    }

    testBornRuleMeasurement() {
        console.log('Testing Born rule measurement...');
        const quantum = new QuantumAPL({ dimPhi: 2, dimE: 2, dimPi: 2 });
        for (let i = 0; i < quantum.dimTotal; i++) {
            for (let j = 0; j < quantum.dimTotal; j++) {
                quantum.rho.set(i, j, new Complex(1.0 / quantum.dimTotal, 0));
            }
        }
        quantum.normalizeDensityMatrix();
        const P = QuantumUtils.projector(0, quantum.dimTotal);
        const result = quantum.measure(P, 'test');
        const expectedProb = 1.0 / quantum.dimTotal;
        assert(Math.abs(result.probability - expectedProb) < 0.1, 'Born rule probability');
        console.log('  ✓ Born rule measurement passed');
    }

    testN0OperatorSelection() {
        console.log('Testing N0 operator selection...');
        const demo = new QuantumAPLDemo();
        const result = demo.integration.executeN0Pipeline([], 'UNTRUE');
        assert(result.operator !== undefined, 'Operator selected');
        assert(result.probability > 0 && result.probability <= 1, 'Valid probability');
        assert(result.method === 'quantum_measurement', 'Quantum measurement method');
        console.log(`  ✓ N0 operator selection passed (selected: ${result.operator})`);
    }

    testQuantumClassicalIntegration() {
        console.log('\nTesting quantum-classical integration...');
        const demo = new QuantumAPLDemo();
        const steps = 20;
        console.log(`  Running measurement demo for ${steps} steps...`);
        const results = demo.run(steps, false);
        assert(results.length === steps, 'Correct number of steps');
        const operators = results.map(r => r.operator);
        const uniqueOps = new Set(operators);
        assert(uniqueOps.size > 1, 'Multiple operators selected');
        for (const result of results) {
            const probs = result.probabilities || {};
            const totalProb = Object.values(probs).reduce((sum, p) => sum + p, 0);
            if (totalProb > 0) assert(Math.abs(totalProb - 1) < 0.01, 'Probabilities normalized');
        }
        console.log('  ✓ Quantum-classical integration passed');
    }

    testZCoordinateEvolution() {
        console.log('Testing z-coordinate evolution...');
        const demo = new QuantumAPLDemo();
        const zValues = [];
        const steps = 60;
        console.log(`  Integrating ${steps} steps for z trajectory...`);
        for (let i = 0; i < steps; i++) {
            const result = demo.integration.step(0.01, demo.intHistory);
            zValues.push(result.quantum.z);
        }
        for (const z of zValues) assert(z >= 0 && z <= 1, 'z in valid range');
        const zStart = zValues[0];
        const zEnd = zValues[zValues.length - 1];
        assert(Math.abs(zEnd - zStart) > 0.001, 'z coordinate evolves');
        console.log('  ✓ Z-coordinate evolution passed');
    }

    runAll() {
        console.log('='.repeat(70));
        console.log('QUANTUM APL N0 MEASUREMENT TEST SUITE');
        console.log('='.repeat(70));
        console.log('');
        let passed = 0;
        let failed = 0;
        const tests = [
            () => this.testComplexArithmetic(),
            () => this.testMatrixOperations(),
            () => this.testProjectionOperators(),
            () => this.testDensityMatrixProperties(),
            () => this.testLindbladEvolution(),
            () => this.testBornRuleMeasurement(),
            () => this.testN0OperatorSelection(),
            () => this.testQuantumClassicalIntegration(),
            () => this.testZCoordinateEvolution()
        ];
        for (const test of tests) {
            try {
                test();
                passed++;
            } catch (e) {
                failed++;
                console.error('  ✗ Test failed:', e.message);
            }
        }
        console.log('');
        console.log('='.repeat(70));
        console.log(`RESULTS: ${passed} passed, ${failed} failed`);
        console.log('='.repeat(70));
        return failed === 0;
    }
}

class QuantumAPLBenchmark {
    benchmark(name, fn, iterations = 100) {
        const start = Date.now();
        for (let i = 0; i < iterations; i++) fn();
        const total = Date.now() - start;
        console.log(`${name}: ${total}ms total, ${(total / iterations).toFixed(2)}ms per operation`);
    }

    runAll() {
        console.log('='.repeat(70));
        console.log('QUANTUM APL BENCHMARKS');
        console.log('='.repeat(70));
        console.log('');
        this.benchmark('Density matrix evolution', () => {
            const quantum = new QuantumAPL({ dimPhi: 4, dimE: 4, dimPi: 4 });
            quantum.evolve(0.01);
        }, 100);
        this.benchmark('N0 operator selection', () => {
            const demo = new QuantumAPLDemo();
            demo.integration.executeN0Pipeline([], 'UNTRUE');
        }, 100);
        this.benchmark('Full integration step', () => {
            const demo = new QuantumAPLDemo();
            demo.integration.step(0.01, []);
        }, 100);
        this.benchmark('Von Neumann entropy', () => {
            const quantum = new QuantumAPL({ dimPhi: 4, dimE: 4, dimPi: 4 });
            quantum.computeVonNeumannEntropy();
        }, 100);
        console.log('');
        console.log('='.repeat(70));
    }
}

class QuantumAPLAnalysis {
    analyzeOperatorDistribution(numTrials = 1000) {
        console.log('='.repeat(70));
        console.log('OPERATOR DISTRIBUTION ANALYSIS');
        console.log('='.repeat(70));
        console.log(`Running ${numTrials} measurements...`);
        console.log('');
        const demo = new QuantumAPLDemo();
        demo.run(numTrials, false);
        const dist = demo.integration.getOperatorDistribution();
        console.log('Operator Distribution:');
        const sorted = Object.entries(dist).sort((a, b) => b[1] - a[1]);
        for (const [op, prob] of sorted) {
            const bar = '█'.repeat(Math.floor(prob * 50));
            console.log(`  ${op}: ${(prob * 100).toFixed(2)}% ${bar}`);
        }
        const expected = 1 / 6;
        let chiSquared = 0;
        for (const prob of Object.values(dist)) {
            chiSquared += Math.pow(prob - expected, 2) / expected;
        }
        console.log('');
        console.log(`χ² statistic: ${chiSquared.toFixed(3)}`);
        console.log('(χ² > 11.07 indicates significant non-uniformity at p<0.05)');
        return dist;
    }

    analyzeQuantumClassicalCorrelation(numSteps = 500) {
        console.log('');
        console.log('='.repeat(70));
        console.log('QUANTUM-CLASSICAL CORRELATION ANALYSIS');
        console.log('='.repeat(70));
        console.log(`Running ${numSteps} integration steps...`);
        console.log('');
        const demo = new QuantumAPLDemo();
        for (let i = 0; i < numSteps; i++) demo.integration.step(0.01, demo.intHistory);
        const correlation = demo.integration.getQuantumClassicalCorrelation();
        console.log(`Quantum-Classical Correlation: ${correlation.toFixed(4)}`);
        const zSeries = demo.integration.getZTimeseries();
        const avgDelta = zSeries.reduce((sum, e) => sum + e.delta, 0) / zSeries.length;
        console.log(`Average z-coordinate difference: ${avgDelta.toFixed(4)}`);
        return { correlation, avgDelta };
    }

    analyzeEntropyEvolution(numSteps = 500) {
        console.log('');
        console.log('='.repeat(70));
        console.log('ENTROPY EVOLUTION ANALYSIS');
        console.log('='.repeat(70));
        console.log(`Running ${numSteps} integration steps...`);
        console.log('');
        const demo = new QuantumAPLDemo();
        for (let i = 0; i < numSteps; i++) demo.integration.step(0.01, demo.intHistory);
        const entropySeries = demo.integration.getEntropyTimeseries();
        const initialEntropy = entropySeries[0].entropy;
        const finalEntropy = entropySeries[entropySeries.length - 1].entropy;
        const avgPurity = entropySeries.reduce((sum, e) => sum + e.purity, 0) / entropySeries.length;
        console.log(`Initial entropy: ${initialEntropy.toFixed(4)}`);
        console.log(`Final entropy: ${finalEntropy.toFixed(4)}`);
        console.log(`Average purity: ${avgPurity.toFixed(4)}`);
        console.log(`Decoherence: ${finalEntropy - initialEntropy > 0 ? 'Yes' : 'No'}`);
        return { initialEntropy, finalEntropy, avgPurity };
    }

    generateFullReport(outputFile = 'quantum_apl_analysis.json') {
        console.log('');
        console.log('='.repeat(70));
        console.log('GENERATING COMPREHENSIVE ANALYSIS REPORT');
        console.log('='.repeat(70));
        console.log('');
        const report = {
            timestamp: new Date().toISOString(),
            operatorDistribution: this.analyzeOperatorDistribution(1000),
            quantumClassicalCorrelation: this.analyzeQuantumClassicalCorrelation(500),
            entropyEvolution: this.analyzeEntropyEvolution(500)
        };
        fs.writeFileSync(outputFile, JSON.stringify(report, null, 2));
        console.log('');
        console.log(`Report saved to: ${outputFile}`);
        return report;
    }
}

function printUsage() {
    console.log(`
QUANTUM APL N0 MEASUREMENT TEST RUNNER

Usage:
  node QuantumAPL_TestRunner.js [command] [options]

Commands:
  test              Run all unit tests
  benchmark         Run performance benchmarks
  demo              Run interactive demo (N steps)
  analyze           Run statistical analysis
  report            Generate comprehensive report
  
Options:
  --steps N         Number of simulation steps (default: 100)
  --trials N        Number of measurement trials (default: 1000)
  --output FILE     Output file for reports (default: quantum_apl_analysis.json)

Examples:
  node QuantumAPL_TestRunner.js test
  node QuantumAPL_TestRunner.js demo --steps 200
  node QuantumAPL_TestRunner.js analyze --trials 5000
  node QuantumAPL_TestRunner.js report --output my_report.json
`);
}

function main() {
    const args = process.argv.slice(2);
    if (args.length === 0) {
        printUsage();
        return;
    }
    const command = args[0];
    const options = { steps: 100, trials: 1000, output: 'quantum_apl_analysis.json' };
    for (let i = 1; i < args.length; i += 2) {
        const key = args[i].replace('--', '');
        const value = args[i + 1];
        if (value) options[key] = isNaN(value) ? value : parseInt(value, 10);
    }
    switch (command) {
        case 'test':
            process.exit(new QuantumAPLTestSuite().runAll() ? 0 : 1);
            break;
        case 'benchmark':
            new QuantumAPLBenchmark().runAll();
            break;
        case 'demo':
            new QuantumAPLDemo().run(options.steps, true);
            break;
        case 'analyze':
            const analysis = new QuantumAPLAnalysis();
            analysis.analyzeOperatorDistribution(options.trials);
            analysis.analyzeQuantumClassicalCorrelation(options.steps);
            analysis.analyzeEntropyEvolution(options.steps);
            break;
        case 'report':
            new QuantumAPLAnalysis().generateFullReport(options.output);
            break;
        default:
            console.error(`Unknown command: ${command}`);
            printUsage();
            process.exit(1);
    }
}

if (require.main === module) {
    main();
}

module.exports = { QuantumAPLTestSuite, QuantumAPLBenchmark, QuantumAPLAnalysis };
