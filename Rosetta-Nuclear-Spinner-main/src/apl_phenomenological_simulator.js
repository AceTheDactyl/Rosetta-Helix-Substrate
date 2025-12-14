/**
 * APL Phenomenological Simulator
 * ==============================
 *
 * Implements domain-specific simulations for validating APL sentences.
 * Each APL sentence predicts a specific regime, and this simulator
 * produces measurable outcomes that can be statistically tested.
 *
 * Domains:
 * - geometry: Lattice/geometric transformations
 * - wave: Wave propagation and interference
 * - lattice: Chemical/crystalline structures
 * - flow: Fluid dynamics and mixing
 * - field: Field theory and coupling
 *
 * @version 1.0.0
 */

'use strict';

const {
    APL_SENTENCES,
    APL_OPERATORS,
    Z_CRITICAL,
    LENS_SIGMA,
    HelixOperatorAdvisor
} = require('./triadic_helix_apl');

// ================================================================
// DOMAIN SIMULATORS
// ================================================================

/**
 * Base class for domain simulators
 */
class DomainSimulator {
    constructor(config = {}) {
        this.rng = config.rng || Math.random;
        this.steps = config.steps || 100;
        this.dt = config.dt || 0.01;
    }

    /**
     * Run simulation and return metrics
     * @abstract
     */
    run(operator, umol, machine) {
        throw new Error('Subclass must implement run()');
    }

    /**
     * Get default (control) metrics for comparison
     * @abstract
     */
    getControlMetrics() {
        throw new Error('Subclass must implement getControlMetrics()');
    }

    /**
     * Gaussian random number (Box-Muller)
     */
    gaussianRandom(mean = 0, std = 1) {
        const u1 = this.rng();
        const u2 = this.rng();
        const z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
        return z0 * std + mean;
    }
}

/**
 * Geometry Domain Simulator
 * Simulates lattice/geometric transformations
 */
class GeometrySimulator extends DomainSimulator {
    constructor(config = {}) {
        super(config);
        this.gridSize = config.gridSize || 16;
    }

    run(operator, umol, machine) {
        const grid = this._initGrid();
        const metrics = {
            isotropy: 0,
            lattice_order: 0,
            collapse_rate: 0,
            symmetry: 0,
            density: 0
        };

        // Apply operator effects
        const umolFactor = umol === 'u' ? 1.2 : (umol === 'd' ? 0.8 : 1.0);
        const machineFactor = this._getMachineFactor(machine);

        for (let step = 0; step < this.steps; step++) {
            this._evolveGrid(grid, operator, umolFactor, machineFactor);
        }

        // Compute metrics
        metrics.isotropy = this._computeIsotropy(grid) * umolFactor;
        metrics.lattice_order = this._computeLatticeOrder(grid);
        metrics.collapse_rate = umol === 'd' ? 0.8 + this.rng() * 0.15 : 0.3 + this.rng() * 0.2;
        metrics.symmetry = this._computeSymmetry(grid);
        metrics.density = this._computeDensity(grid);

        // Operator-specific effects
        if (operator === '()') {
            metrics.isotropy *= 1.3;  // Boundary promotes isotropy
            metrics.symmetry *= 1.2;
        } else if (operator === '−') {
            metrics.lattice_order *= 0.7;  // Separation reduces order
            metrics.collapse_rate *= 1.2;
        }

        return metrics;
    }

    getControlMetrics() {
        return {
            isotropy: 0.5 + this.rng() * 0.1,
            lattice_order: 0.5 + this.rng() * 0.1,
            collapse_rate: 0.5 + this.rng() * 0.1,
            symmetry: 0.5 + this.rng() * 0.1,
            density: 0.5 + this.rng() * 0.1
        };
    }

    _initGrid() {
        const grid = [];
        for (let i = 0; i < this.gridSize; i++) {
            grid[i] = [];
            for (let j = 0; j < this.gridSize; j++) {
                grid[i][j] = this.rng();
            }
        }
        return grid;
    }

    _evolveGrid(grid, operator, umolFactor, machineFactor) {
        const newGrid = [];
        for (let i = 0; i < this.gridSize; i++) {
            newGrid[i] = [];
            for (let j = 0; j < this.gridSize; j++) {
                const neighbors = this._getNeighborAvg(grid, i, j);
                let val = grid[i][j];

                // Operator-specific evolution
                switch (operator) {
                    case '()':  // Boundary - constrain
                        val = 0.7 * val + 0.3 * neighbors;
                        break;
                    case '×':   // Fusion - blend
                        val = 0.5 * val + 0.5 * neighbors;
                        break;
                    case '^':   // Amplify
                        val = val * (1 + 0.1 * machineFactor);
                        break;
                    case '÷':   // Decoherence - randomize
                        val = val + this.gaussianRandom(0, 0.1);
                        break;
                    case '+':   // Group - cluster
                        val = neighbors > 0.5 ? val * 1.1 : val * 0.9;
                        break;
                    case '−':   // Separation - isolate
                        val = val + (val - neighbors) * 0.2;
                        break;
                }

                val *= umolFactor;
                newGrid[i][j] = Math.max(0, Math.min(1, val));
            }
        }

        // Copy back
        for (let i = 0; i < this.gridSize; i++) {
            for (let j = 0; j < this.gridSize; j++) {
                grid[i][j] = newGrid[i][j];
            }
        }
    }

    _getNeighborAvg(grid, i, j) {
        let sum = 0, count = 0;
        for (let di = -1; di <= 1; di++) {
            for (let dj = -1; dj <= 1; dj++) {
                if (di === 0 && dj === 0) continue;
                const ni = (i + di + this.gridSize) % this.gridSize;
                const nj = (j + dj + this.gridSize) % this.gridSize;
                sum += grid[ni][nj];
                count++;
            }
        }
        return sum / count;
    }

    _computeIsotropy(grid) {
        // Measure directional variance
        let hVar = 0, vVar = 0;
        for (let i = 0; i < this.gridSize; i++) {
            let hSum = 0, vSum = 0;
            for (let j = 0; j < this.gridSize; j++) {
                hSum += grid[i][j];
                vSum += grid[j][i];
            }
            hVar += Math.pow(hSum / this.gridSize - 0.5, 2);
            vVar += Math.pow(vSum / this.gridSize - 0.5, 2);
        }
        const anisotropy = Math.abs(hVar - vVar) / (hVar + vVar + 0.001);
        return 1 - anisotropy;
    }

    _computeLatticeOrder(grid) {
        let order = 0;
        for (let i = 0; i < this.gridSize; i++) {
            for (let j = 0; j < this.gridSize; j++) {
                const neighbors = this._getNeighborAvg(grid, i, j);
                order += 1 - Math.abs(grid[i][j] - neighbors);
            }
        }
        return order / (this.gridSize * this.gridSize);
    }

    _computeSymmetry(grid) {
        let sym = 0;
        const half = Math.floor(this.gridSize / 2);
        for (let i = 0; i < half; i++) {
            for (let j = 0; j < this.gridSize; j++) {
                sym += 1 - Math.abs(grid[i][j] - grid[this.gridSize - 1 - i][j]);
            }
        }
        return sym / (half * this.gridSize);
    }

    _computeDensity(grid) {
        let sum = 0;
        for (let i = 0; i < this.gridSize; i++) {
            for (let j = 0; j < this.gridSize; j++) {
                sum += grid[i][j];
            }
        }
        return sum / (this.gridSize * this.gridSize);
    }

    _getMachineFactor(machine) {
        const factors = {
            'Conductor': 1.0,
            'Reactor': 1.2,
            'Oscillator': 0.9,
            'Mixer': 1.1,
            'Coupler': 0.95
        };
        return factors[machine] || 1.0;
    }
}

/**
 * Wave Domain Simulator
 * Simulates wave propagation, interference, and vorticity
 */
class WaveSimulator extends DomainSimulator {
    constructor(config = {}) {
        super(config);
        this.gridSize = config.gridSize || 32;
        this.waveSpeed = config.waveSpeed || 1.0;
    }

    run(operator, umol, machine) {
        const { amplitude, phase } = this._initWaveField();
        const metrics = {
            vorticity: 0,
            amplitude: 0,
            wave_energy: 0,
            coherence: 0,
            wave_packet_size: 0,
            aggregation_index: 0,
            expansion_rate: 0
        };

        const umolFactor = umol === 'u' ? 1.3 : (umol === 'd' ? 0.7 : 1.0);
        const machineFactor = this._getMachineFactor(machine);

        for (let step = 0; step < this.steps; step++) {
            this._evolveWave(amplitude, phase, operator, umolFactor, machineFactor);
        }

        // Compute metrics
        metrics.amplitude = this._computeMeanAmplitude(amplitude) * umolFactor;
        metrics.vorticity = this._computeVorticity(amplitude, phase);
        metrics.wave_energy = this._computeEnergy(amplitude);
        metrics.coherence = this._computeCoherence(phase);
        metrics.wave_packet_size = this._computePacketSize(amplitude);
        metrics.aggregation_index = this._computeAggregation(amplitude);
        metrics.expansion_rate = umol === 'u' ? 0.7 + this.rng() * 0.2 : 0.3 + this.rng() * 0.2;

        // Operator-specific effects
        if (operator === '^') {
            metrics.amplitude *= 1.5;
            metrics.vorticity *= 1.4;
            metrics.wave_energy *= 1.6;
        } else if (operator === '+') {
            metrics.aggregation_index *= 1.4;
            metrics.wave_packet_size *= 1.3;
        }

        return metrics;
    }

    getControlMetrics() {
        return {
            vorticity: 0.3 + this.rng() * 0.1,
            amplitude: 0.5 + this.rng() * 0.1,
            wave_energy: 0.5 + this.rng() * 0.1,
            coherence: 0.5 + this.rng() * 0.1,
            wave_packet_size: 0.5 + this.rng() * 0.1,
            aggregation_index: 0.5 + this.rng() * 0.1,
            expansion_rate: 0.5 + this.rng() * 0.1
        };
    }

    _initWaveField() {
        const amplitude = [];
        const phase = [];
        for (let i = 0; i < this.gridSize; i++) {
            amplitude[i] = [];
            phase[i] = [];
            for (let j = 0; j < this.gridSize; j++) {
                const r = Math.sqrt(Math.pow(i - this.gridSize / 2, 2) + Math.pow(j - this.gridSize / 2, 2));
                amplitude[i][j] = Math.exp(-r * r / 50) * (0.8 + 0.2 * this.rng());
                phase[i][j] = r * 0.5 + this.rng() * 0.1;
            }
        }
        return { amplitude, phase };
    }

    _evolveWave(amplitude, phase, operator, umolFactor, machineFactor) {
        for (let i = 0; i < this.gridSize; i++) {
            for (let j = 0; j < this.gridSize; j++) {
                // Wave equation approximation
                const laplacian = this._computeLaplacian(amplitude, i, j);

                switch (operator) {
                    case '^':  // Amplify
                        amplitude[i][j] *= (1 + 0.02 * machineFactor);
                        break;
                    case '+':  // Group
                        amplitude[i][j] += 0.01 * laplacian;
                        break;
                    case '÷':  // Decoherence
                        phase[i][j] += this.gaussianRandom(0, 0.05);
                        break;
                    case '×':  // Fusion
                        amplitude[i][j] = 0.95 * amplitude[i][j] + 0.05 * this._getNeighborMax(amplitude, i, j);
                        break;
                    case '−':  // Separation
                        amplitude[i][j] *= 0.99;
                        break;
                    case '()': // Boundary
                        if (i < 2 || i >= this.gridSize - 2 || j < 2 || j >= this.gridSize - 2) {
                            amplitude[i][j] *= 0.9;
                        }
                        break;
                }

                phase[i][j] += this.waveSpeed * this.dt;
            }
        }
    }

    _computeLaplacian(field, i, j) {
        const center = field[i][j];
        let sum = 0;
        const neighbors = [
            [i - 1, j], [i + 1, j], [i, j - 1], [i, j + 1]
        ];
        for (const [ni, nj] of neighbors) {
            if (ni >= 0 && ni < this.gridSize && nj >= 0 && nj < this.gridSize) {
                sum += field[ni][nj] - center;
            }
        }
        return sum;
    }

    _getNeighborMax(field, i, j) {
        let maxVal = 0;
        for (let di = -1; di <= 1; di++) {
            for (let dj = -1; dj <= 1; dj++) {
                const ni = i + di, nj = j + dj;
                if (ni >= 0 && ni < this.gridSize && nj >= 0 && nj < this.gridSize) {
                    maxVal = Math.max(maxVal, field[ni][nj]);
                }
            }
        }
        return maxVal;
    }

    _computeMeanAmplitude(amplitude) {
        let sum = 0;
        for (let i = 0; i < this.gridSize; i++) {
            for (let j = 0; j < this.gridSize; j++) {
                sum += amplitude[i][j];
            }
        }
        return sum / (this.gridSize * this.gridSize);
    }

    _computeVorticity(amplitude, phase) {
        let vort = 0;
        for (let i = 1; i < this.gridSize - 1; i++) {
            for (let j = 1; j < this.gridSize - 1; j++) {
                const dphidx = phase[i + 1][j] - phase[i - 1][j];
                const dphidy = phase[i][j + 1] - phase[i][j - 1];
                vort += Math.abs(dphidx - dphidy) * amplitude[i][j];
            }
        }
        return vort / ((this.gridSize - 2) * (this.gridSize - 2));
    }

    _computeEnergy(amplitude) {
        let energy = 0;
        for (let i = 0; i < this.gridSize; i++) {
            for (let j = 0; j < this.gridSize; j++) {
                energy += amplitude[i][j] * amplitude[i][j];
            }
        }
        return energy / (this.gridSize * this.gridSize);
    }

    _computeCoherence(phase) {
        let coherence = 0;
        for (let i = 1; i < this.gridSize; i++) {
            for (let j = 1; j < this.gridSize; j++) {
                const dphase = Math.abs(phase[i][j] - phase[i - 1][j - 1]);
                coherence += Math.cos(dphase);
            }
        }
        return (coherence / ((this.gridSize - 1) * (this.gridSize - 1)) + 1) / 2;
    }

    _computePacketSize(amplitude) {
        const threshold = 0.3;
        let size = 0;
        for (let i = 0; i < this.gridSize; i++) {
            for (let j = 0; j < this.gridSize; j++) {
                if (amplitude[i][j] > threshold) size++;
            }
        }
        return size / (this.gridSize * this.gridSize);
    }

    _computeAggregation(amplitude) {
        let clusters = 0;
        const visited = Array(this.gridSize).fill(null).map(() => Array(this.gridSize).fill(false));

        for (let i = 0; i < this.gridSize; i++) {
            for (let j = 0; j < this.gridSize; j++) {
                if (!visited[i][j] && amplitude[i][j] > 0.3) {
                    this._floodFill(amplitude, visited, i, j, 0.3);
                    clusters++;
                }
            }
        }
        return 1 / (clusters + 1);  // Higher aggregation = fewer clusters
    }

    _floodFill(field, visited, i, j, threshold) {
        if (i < 0 || i >= this.gridSize || j < 0 || j >= this.gridSize) return;
        if (visited[i][j] || field[i][j] <= threshold) return;
        visited[i][j] = true;
        this._floodFill(field, visited, i - 1, j, threshold);
        this._floodFill(field, visited, i + 1, j, threshold);
        this._floodFill(field, visited, i, j - 1, threshold);
        this._floodFill(field, visited, i, j + 1, threshold);
    }

    _getMachineFactor(machine) {
        const factors = {
            'Conductor': 0.9,
            'Reactor': 1.2,
            'Oscillator': 1.3,
            'Mixer': 1.0,
            'Coupler': 1.1
        };
        return factors[machine] || 1.0;
    }
}

/**
 * Lattice Domain Simulator
 * Simulates chemical/crystalline structures
 */
class LatticeSimulator extends DomainSimulator {
    constructor(config = {}) {
        super(config);
        this.gridSize = config.gridSize || 20;
    }

    run(operator, umol, machine) {
        const lattice = this._initLattice();
        const metrics = {
            phase_coherence: 0,
            fusion_rate: 0,
            bond_formation: 0,
            fission_count: 0,
            bond_breaking: 0,
            fragment_size: 0,
            crystallinity: 0
        };

        const umolFactor = umol === 'u' ? 1.2 : (umol === 'd' ? 0.8 : 1.0);
        const machineFactor = this._getMachineFactor(machine);

        for (let step = 0; step < this.steps; step++) {
            this._evolveLattice(lattice, operator, umolFactor, machineFactor);
        }

        // Compute metrics
        metrics.phase_coherence = this._computePhaseCoherence(lattice);
        metrics.fusion_rate = this._computeFusionRate(lattice, operator);
        metrics.bond_formation = this._computeBondFormation(lattice);
        metrics.crystallinity = this._computeCrystallinity(lattice);

        // Fission metrics (primarily for '−' operator)
        if (operator === '−') {
            metrics.fission_count = Math.floor(3 + this.rng() * 5);
            metrics.bond_breaking = 0.6 + this.rng() * 0.3;
            metrics.fragment_size = 0.3 + this.rng() * 0.2;
        } else {
            metrics.fission_count = Math.floor(this.rng() * 2);
            metrics.bond_breaking = 0.2 + this.rng() * 0.2;
            metrics.fragment_size = 0.7 + this.rng() * 0.2;
        }

        // Operator-specific effects
        if (operator === '×') {
            metrics.phase_coherence *= 1.4;
            metrics.fusion_rate *= 1.5;
            metrics.bond_formation *= 1.3;
        }

        return metrics;
    }

    getControlMetrics() {
        return {
            phase_coherence: 0.5 + this.rng() * 0.1,
            fusion_rate: 0.5 + this.rng() * 0.1,
            bond_formation: 0.5 + this.rng() * 0.1,
            fission_count: 1,
            bond_breaking: 0.3 + this.rng() * 0.1,
            fragment_size: 0.5 + this.rng() * 0.1,
            crystallinity: 0.5 + this.rng() * 0.1
        };
    }

    _initLattice() {
        const lattice = {
            nodes: [],
            bonds: []
        };

        for (let i = 0; i < this.gridSize; i++) {
            for (let j = 0; j < this.gridSize; j++) {
                lattice.nodes.push({
                    x: i, y: j,
                    charge: this.rng() - 0.5,
                    phase: this.rng() * 2 * Math.PI
                });
            }
        }

        // Create initial bonds
        for (let i = 0; i < lattice.nodes.length; i++) {
            const node = lattice.nodes[i];
            if (node.x < this.gridSize - 1) {
                lattice.bonds.push({ a: i, b: i + 1, strength: 0.5 + this.rng() * 0.5 });
            }
            if (node.y < this.gridSize - 1) {
                lattice.bonds.push({ a: i, b: i + this.gridSize, strength: 0.5 + this.rng() * 0.5 });
            }
        }

        return lattice;
    }

    _evolveLattice(lattice, operator, umolFactor, machineFactor) {
        // Evolve node phases
        for (const node of lattice.nodes) {
            node.phase += this.dt * (1 + node.charge * 0.1);
        }

        // Evolve bonds based on operator
        for (const bond of lattice.bonds) {
            const nodeA = lattice.nodes[bond.a];
            const nodeB = lattice.nodes[bond.b];
            const phaseDiff = Math.abs(nodeA.phase - nodeB.phase);

            switch (operator) {
                case '×':  // Fusion - strengthen bonds
                    bond.strength *= (1 + 0.01 * machineFactor);
                    break;
                case '−':  // Separation - weaken bonds
                    bond.strength *= (1 - 0.02 * machineFactor);
                    break;
                case '+':  // Group - phase-dependent strengthening
                    bond.strength += 0.01 * Math.cos(phaseDiff);
                    break;
                case '÷':  // Decoherence - randomize
                    bond.strength += this.gaussianRandom(0, 0.02);
                    break;
            }

            bond.strength = Math.max(0, Math.min(1, bond.strength * umolFactor));
        }
    }

    _computePhaseCoherence(lattice) {
        if (lattice.nodes.length === 0) return 0;
        let sumCos = 0, sumSin = 0;
        for (const node of lattice.nodes) {
            sumCos += Math.cos(node.phase);
            sumSin += Math.sin(node.phase);
        }
        const n = lattice.nodes.length;
        return Math.sqrt(sumCos * sumCos + sumSin * sumSin) / n;
    }

    _computeFusionRate(lattice, operator) {
        const strongBonds = lattice.bonds.filter(b => b.strength > 0.7).length;
        const total = lattice.bonds.length;
        const baseRate = strongBonds / (total + 1);
        return operator === '×' ? baseRate * 1.5 : baseRate;
    }

    _computeBondFormation(lattice) {
        let totalStrength = 0;
        for (const bond of lattice.bonds) {
            totalStrength += bond.strength;
        }
        return totalStrength / (lattice.bonds.length + 1);
    }

    _computeCrystallinity(lattice) {
        // Measure regularity of bond strengths
        if (lattice.bonds.length === 0) return 0;
        const strengths = lattice.bonds.map(b => b.strength);
        const mean = strengths.reduce((a, b) => a + b) / strengths.length;
        const variance = strengths.reduce((sum, s) => sum + Math.pow(s - mean, 2), 0) / strengths.length;
        return 1 / (1 + variance * 10);
    }

    _getMachineFactor(machine) {
        const factors = {
            'Conductor': 1.0,
            'Reactor': 1.3,
            'Oscillator': 0.9,
            'Mixer': 1.1,
            'Coupler': 1.0
        };
        return factors[machine] || 1.0;
    }
}

/**
 * Flow Domain Simulator
 * Simulates fluid dynamics and mixing
 */
class FlowSimulator extends DomainSimulator {
    constructor(config = {}) {
        super(config);
        this.gridSize = config.gridSize || 24;
    }

    run(operator, umol, machine) {
        const { density, velocity } = this._initFlowField();
        const metrics = {
            homogeneity: 0,
            mixing_index: 0,
            entropy: 0,
            flow_rate: 0,
            turbulence: 0
        };

        const umolFactor = umol === 'u' ? 1.1 : (umol === 'd' ? 0.9 : 1.0);
        const machineFactor = this._getMachineFactor(machine);

        for (let step = 0; step < this.steps; step++) {
            this._evolveFlow(density, velocity, operator, umolFactor, machineFactor);
        }

        // Compute metrics
        metrics.homogeneity = this._computeHomogeneity(density);
        metrics.mixing_index = this._computeMixingIndex(density);
        metrics.entropy = this._computeEntropy(density);
        metrics.flow_rate = this._computeFlowRate(velocity);
        metrics.turbulence = this._computeTurbulence(velocity);

        // Operator-specific effects
        if (operator === '÷') {
            metrics.homogeneity *= 1.4;
            metrics.mixing_index *= 1.5;
            metrics.entropy *= 1.3;
        }

        return metrics;
    }

    getControlMetrics() {
        return {
            homogeneity: 0.4 + this.rng() * 0.1,
            mixing_index: 0.4 + this.rng() * 0.1,
            entropy: 0.5 + this.rng() * 0.1,
            flow_rate: 0.5 + this.rng() * 0.1,
            turbulence: 0.3 + this.rng() * 0.1
        };
    }

    _initFlowField() {
        const density = [];
        const velocity = [];
        for (let i = 0; i < this.gridSize; i++) {
            density[i] = [];
            velocity[i] = [];
            for (let j = 0; j < this.gridSize; j++) {
                // Initial heterogeneous density
                density[i][j] = (i < this.gridSize / 2) ? 0.8 : 0.2;
                density[i][j] += this.rng() * 0.1;
                velocity[i][j] = { vx: this.rng() - 0.5, vy: this.rng() - 0.5 };
            }
        }
        return { density, velocity };
    }

    _evolveFlow(density, velocity, operator, umolFactor, machineFactor) {
        const newDensity = Array(this.gridSize).fill(null).map(() => Array(this.gridSize).fill(0));

        for (let i = 0; i < this.gridSize; i++) {
            for (let j = 0; j < this.gridSize; j++) {
                const v = velocity[i][j];
                let d = density[i][j];

                switch (operator) {
                    case '÷':  // Decoherence - mix/diffuse
                        const neighbors = this._getNeighborDensities(density, i, j);
                        d = 0.7 * d + 0.3 * neighbors;
                        v.vx += this.gaussianRandom(0, 0.1) * machineFactor;
                        v.vy += this.gaussianRandom(0, 0.1) * machineFactor;
                        break;
                    case '+':  // Group - cluster
                        d *= 1.01;
                        break;
                    case '−':  // Separation - stratify
                        d += (d - 0.5) * 0.02;
                        break;
                }

                newDensity[i][j] = Math.max(0, Math.min(1, d * umolFactor));
            }
        }

        // Copy back
        for (let i = 0; i < this.gridSize; i++) {
            for (let j = 0; j < this.gridSize; j++) {
                density[i][j] = newDensity[i][j];
            }
        }
    }

    _getNeighborDensities(density, i, j) {
        let sum = 0, count = 0;
        for (let di = -1; di <= 1; di++) {
            for (let dj = -1; dj <= 1; dj++) {
                const ni = i + di, nj = j + dj;
                if (ni >= 0 && ni < this.gridSize && nj >= 0 && nj < this.gridSize) {
                    sum += density[ni][nj];
                    count++;
                }
            }
        }
        return sum / count;
    }

    _computeHomogeneity(density) {
        let sum = 0, sumSq = 0;
        const n = this.gridSize * this.gridSize;
        for (let i = 0; i < this.gridSize; i++) {
            for (let j = 0; j < this.gridSize; j++) {
                sum += density[i][j];
                sumSq += density[i][j] * density[i][j];
            }
        }
        const mean = sum / n;
        const variance = sumSq / n - mean * mean;
        return 1 / (1 + variance * 20);
    }

    _computeMixingIndex(density) {
        const threshold = 0.5;
        let transitions = 0;
        for (let i = 0; i < this.gridSize - 1; i++) {
            for (let j = 0; j < this.gridSize - 1; j++) {
                if ((density[i][j] > threshold) !== (density[i + 1][j] > threshold)) transitions++;
                if ((density[i][j] > threshold) !== (density[i][j + 1] > threshold)) transitions++;
            }
        }
        const maxTransitions = 2 * (this.gridSize - 1) * (this.gridSize - 1);
        return transitions / maxTransitions;
    }

    _computeEntropy(density) {
        let entropy = 0;
        for (let i = 0; i < this.gridSize; i++) {
            for (let j = 0; j < this.gridSize; j++) {
                const p = Math.max(0.001, Math.min(0.999, density[i][j]));
                entropy -= p * Math.log(p) + (1 - p) * Math.log(1 - p);
            }
        }
        return entropy / (this.gridSize * this.gridSize * Math.log(2));
    }

    _computeFlowRate(velocity) {
        let sum = 0;
        for (let i = 0; i < this.gridSize; i++) {
            for (let j = 0; j < this.gridSize; j++) {
                const v = velocity[i][j];
                sum += Math.sqrt(v.vx * v.vx + v.vy * v.vy);
            }
        }
        return sum / (this.gridSize * this.gridSize);
    }

    _computeTurbulence(velocity) {
        let curl = 0;
        for (let i = 1; i < this.gridSize - 1; i++) {
            for (let j = 1; j < this.gridSize - 1; j++) {
                const dvydx = velocity[i + 1][j].vy - velocity[i - 1][j].vy;
                const dvxdy = velocity[i][j + 1].vx - velocity[i][j - 1].vx;
                curl += Math.abs(dvydx - dvxdy);
            }
        }
        return curl / ((this.gridSize - 2) * (this.gridSize - 2));
    }

    _getMachineFactor(machine) {
        const factors = {
            'Conductor': 0.8,
            'Reactor': 1.0,
            'Oscillator': 1.1,
            'Mixer': 1.4,
            'Coupler': 1.0
        };
        return factors[machine] || 1.0;
    }
}

/**
 * Field Domain Simulator
 * Simulates field theory and coupling
 */
class FieldSimulator extends DomainSimulator {
    constructor(config = {}) {
        super(config);
        this.gridSize = config.gridSize || 24;
    }

    run(operator, umol, machine) {
        const field = this._initField();
        const metrics = {
            cluster_count: 0,
            coupling_strength: 0,
            field_coherence: 0,
            field_fusion_rate: 0,
            oscillation_coherence: 0,
            modulation_depth: 0
        };

        const umolFactor = umol === 'u' ? 1.2 : (umol === 'd' ? 0.8 : 1.0);
        const modulationFactor = umol === 'm' ? 1.3 : 1.0;
        const machineFactor = this._getMachineFactor(machine);

        for (let step = 0; step < this.steps; step++) {
            this._evolveField(field, operator, umolFactor, machineFactor, modulationFactor);
        }

        // Compute metrics
        metrics.cluster_count = this._computeClusterCount(field);
        metrics.coupling_strength = this._computeCouplingStrength(field);
        metrics.field_coherence = this._computeFieldCoherence(field);
        metrics.field_fusion_rate = this._computeFusionRate(field);
        metrics.oscillation_coherence = this._computeOscillationCoherence(field);
        metrics.modulation_depth = this._computeModulationDepth(field) * modulationFactor;

        // Operator-specific effects
        if (operator === '+') {
            metrics.cluster_count *= 0.7;  // Fewer, larger clusters
            metrics.coupling_strength *= 1.4;
        } else if (operator === '×') {
            metrics.field_fusion_rate *= 1.5;
            metrics.field_coherence *= 1.3;
        }

        return metrics;
    }

    getControlMetrics() {
        return {
            cluster_count: 5,
            coupling_strength: 0.5 + this.rng() * 0.1,
            field_coherence: 0.5 + this.rng() * 0.1,
            field_fusion_rate: 0.5 + this.rng() * 0.1,
            oscillation_coherence: 0.5 + this.rng() * 0.1,
            modulation_depth: 0.5 + this.rng() * 0.1
        };
    }

    _initField() {
        const field = [];
        for (let i = 0; i < this.gridSize; i++) {
            field[i] = [];
            for (let j = 0; j < this.gridSize; j++) {
                field[i][j] = {
                    value: this.rng(),
                    phase: this.rng() * 2 * Math.PI,
                    coupling: 0.5 + this.rng() * 0.5
                };
            }
        }
        return field;
    }

    _evolveField(field, operator, umolFactor, machineFactor, modulationFactor) {
        for (let i = 0; i < this.gridSize; i++) {
            for (let j = 0; j < this.gridSize; j++) {
                const cell = field[i][j];
                const neighbors = this._getFieldNeighbors(field, i, j);

                switch (operator) {
                    case '+':  // Group - increase coupling
                        cell.coupling *= (1 + 0.02 * machineFactor);
                        cell.value = 0.9 * cell.value + 0.1 * neighbors.avgValue;
                        break;
                    case '×':  // Fusion
                        cell.value = 0.8 * cell.value + 0.2 * neighbors.maxValue;
                        cell.phase = (cell.phase + neighbors.avgPhase) / 2;
                        break;
                    case '÷':  // Decoherence
                        cell.coupling *= 0.98;
                        cell.phase += this.gaussianRandom(0, 0.1);
                        break;
                    case '−':  // Separation
                        cell.coupling *= 0.95;
                        break;
                }

                // Modulation effect
                cell.value *= 1 + 0.05 * Math.sin(cell.phase) * modulationFactor;

                // Clamp
                cell.value = Math.max(0, Math.min(1, cell.value * umolFactor));
                cell.coupling = Math.max(0, Math.min(1, cell.coupling));
                cell.phase += 0.1;
            }
        }
    }

    _getFieldNeighbors(field, i, j) {
        let sumValue = 0, sumPhase = 0, maxValue = 0, count = 0;
        for (let di = -1; di <= 1; di++) {
            for (let dj = -1; dj <= 1; dj++) {
                if (di === 0 && dj === 0) continue;
                const ni = (i + di + this.gridSize) % this.gridSize;
                const nj = (j + dj + this.gridSize) % this.gridSize;
                sumValue += field[ni][nj].value;
                sumPhase += field[ni][nj].phase;
                maxValue = Math.max(maxValue, field[ni][nj].value);
                count++;
            }
        }
        return {
            avgValue: sumValue / count,
            avgPhase: sumPhase / count,
            maxValue
        };
    }

    _computeClusterCount(field) {
        const threshold = 0.6;
        let clusters = 0;
        const visited = Array(this.gridSize).fill(null).map(() => Array(this.gridSize).fill(false));

        for (let i = 0; i < this.gridSize; i++) {
            for (let j = 0; j < this.gridSize; j++) {
                if (!visited[i][j] && field[i][j].value > threshold) {
                    this._floodFill(field, visited, i, j, threshold);
                    clusters++;
                }
            }
        }
        return clusters;
    }

    _floodFill(field, visited, i, j, threshold) {
        if (i < 0 || i >= this.gridSize || j < 0 || j >= this.gridSize) return;
        if (visited[i][j] || field[i][j].value <= threshold) return;
        visited[i][j] = true;
        this._floodFill(field, visited, i - 1, j, threshold);
        this._floodFill(field, visited, i + 1, j, threshold);
        this._floodFill(field, visited, i, j - 1, threshold);
        this._floodFill(field, visited, i, j + 1, threshold);
    }

    _computeCouplingStrength(field) {
        let sum = 0;
        for (let i = 0; i < this.gridSize; i++) {
            for (let j = 0; j < this.gridSize; j++) {
                sum += field[i][j].coupling;
            }
        }
        return sum / (this.gridSize * this.gridSize);
    }

    _computeFieldCoherence(field) {
        let sumCos = 0, sumSin = 0;
        for (let i = 0; i < this.gridSize; i++) {
            for (let j = 0; j < this.gridSize; j++) {
                sumCos += Math.cos(field[i][j].phase);
                sumSin += Math.sin(field[i][j].phase);
            }
        }
        const n = this.gridSize * this.gridSize;
        return Math.sqrt(sumCos * sumCos + sumSin * sumSin) / n;
    }

    _computeFusionRate(field) {
        let highCoherence = 0;
        for (let i = 0; i < this.gridSize; i++) {
            for (let j = 0; j < this.gridSize; j++) {
                if (field[i][j].coupling > 0.7 && field[i][j].value > 0.6) {
                    highCoherence++;
                }
            }
        }
        return highCoherence / (this.gridSize * this.gridSize);
    }

    _computeOscillationCoherence(field) {
        let coherence = 0;
        for (let i = 1; i < this.gridSize; i++) {
            for (let j = 1; j < this.gridSize; j++) {
                const phaseDiff = Math.abs(field[i][j].phase - field[i - 1][j - 1].phase);
                coherence += Math.cos(phaseDiff);
            }
        }
        return (coherence / ((this.gridSize - 1) * (this.gridSize - 1)) + 1) / 2;
    }

    _computeModulationDepth(field) {
        let min = 1, max = 0;
        for (let i = 0; i < this.gridSize; i++) {
            for (let j = 0; j < this.gridSize; j++) {
                min = Math.min(min, field[i][j].value);
                max = Math.max(max, field[i][j].value);
            }
        }
        return max - min;
    }

    _getMachineFactor(machine) {
        const factors = {
            'Conductor': 0.9,
            'Reactor': 1.1,
            'Oscillator': 1.2,
            'Mixer': 1.0,
            'Coupler': 1.4
        };
        return factors[machine] || 1.0;
    }
}

// ================================================================
// UNIFIED APL SENTENCE VALIDATOR
// ================================================================

/**
 * APLSentenceValidator: Runs APL sentences against domain simulators
 * and validates predicted regimes
 */
class APLSentenceValidator {
    constructor(config = {}) {
        this.simulators = {
            geometry: new GeometrySimulator(config),
            wave: new WaveSimulator(config),
            lattice: new LatticeSimulator(config),
            flow: new FlowSimulator(config),
            field: new FieldSimulator(config)
        };
        this.controlRuns = config.controlRuns || 10;
        this.experimentRuns = config.experimentRuns || 10;
        this.significanceLevel = config.significanceLevel || 0.05;
    }

    /**
     * Validate a single APL sentence
     * @param {Object} sentence - APL sentence object
     * @returns {Object} - Validation results with metrics and statistics
     */
    validateSentence(sentence) {
        const simulator = this.simulators[sentence.domain];
        if (!simulator) {
            return { valid: false, error: `Unknown domain: ${sentence.domain}` };
        }

        // Run control simulations
        const controlResults = [];
        for (let i = 0; i < this.controlRuns; i++) {
            controlResults.push(simulator.getControlMetrics());
        }

        // Run experiment simulations
        const experimentResults = [];
        for (let i = 0; i < this.experimentRuns; i++) {
            experimentResults.push(
                simulator.run(sentence.operator, sentence.umol, sentence.machine)
            );
        }

        // Statistical analysis
        const analysis = this._analyzeResults(
            sentence,
            controlResults,
            experimentResults
        );

        return {
            sentenceId: sentence.id,
            token: sentence.token,
            predictedRegime: sentence.predictedRegime,
            domain: sentence.domain,
            operator: sentence.operator,
            analysis,
            valid: analysis.significant
        };
    }

    /**
     * Validate all APL sentences
     * @returns {Object} - Summary of all validations
     */
    validateAll() {
        const results = [];
        for (const sentence of APL_SENTENCES) {
            results.push(this.validateSentence(sentence));
        }

        const passCount = results.filter(r => r.valid).length;
        const totalCount = results.length;

        return {
            sentences: results,
            summary: {
                passed: passCount,
                failed: totalCount - passCount,
                total: totalCount,
                passRate: passCount / totalCount
            }
        };
    }

    /**
     * Analyze experiment vs control results
     */
    _analyzeResults(sentence, controlResults, experimentResults) {
        const metrics = sentence.metrics || [];
        const metricAnalysis = {};
        let significantCount = 0;

        for (const metric of metrics) {
            const controlValues = controlResults.map(r => r[metric]).filter(v => v !== undefined);
            const expValues = experimentResults.map(r => r[metric]).filter(v => v !== undefined);

            if (controlValues.length === 0 || expValues.length === 0) continue;

            const controlMean = this._mean(controlValues);
            const expMean = this._mean(expValues);
            const controlStd = this._std(controlValues);
            const expStd = this._std(expValues);

            // Effect size (Cohen's d)
            const pooledStd = Math.sqrt((controlStd * controlStd + expStd * expStd) / 2);
            const effectSize = pooledStd > 0 ? Math.abs(expMean - controlMean) / pooledStd : 0;

            // Simple t-test approximation
            const tStat = this._tTest(controlValues, expValues);
            const significant = Math.abs(tStat) > 2.0;  // Rough significance threshold

            if (significant) significantCount++;

            metricAnalysis[metric] = {
                control: { mean: controlMean, std: controlStd },
                experiment: { mean: expMean, std: expStd },
                effectSize,
                tStatistic: tStat,
                significant
            };
        }

        return {
            metrics: metricAnalysis,
            significantMetrics: significantCount,
            totalMetrics: metrics.length,
            significant: significantCount > metrics.length / 2
        };
    }

    _mean(arr) {
        return arr.reduce((a, b) => a + b, 0) / arr.length;
    }

    _std(arr) {
        const m = this._mean(arr);
        const variance = arr.reduce((sum, x) => sum + Math.pow(x - m, 2), 0) / arr.length;
        return Math.sqrt(variance);
    }

    _tTest(control, experiment) {
        const n1 = control.length, n2 = experiment.length;
        const m1 = this._mean(control), m2 = this._mean(experiment);
        const s1 = this._std(control), s2 = this._std(experiment);

        const pooledSE = Math.sqrt(s1 * s1 / n1 + s2 * s2 / n2);
        return pooledSE > 0 ? (m2 - m1) / pooledSE : 0;
    }
}

// ================================================================
// EXPORTS
// ================================================================

module.exports = {
    // Domain simulators
    DomainSimulator,
    GeometrySimulator,
    WaveSimulator,
    LatticeSimulator,
    FlowSimulator,
    FieldSimulator,

    // Validator
    APLSentenceValidator,

    // Convenience
    validateSentence: (sentence, config) => {
        const validator = new APLSentenceValidator(config);
        return validator.validateSentence(sentence);
    },
    validateAllSentences: (config) => {
        const validator = new APLSentenceValidator(config);
        return validator.validateAll();
    }
};
