/**
 * Quantum APL Unified System
 * ==========================
 * Main orchestration module integrating:
 * - TRIAD unlock tracking (hysteresis state machine)
 * - Helix coordinate mapping (z → harmonic → operator window)
 * - Alpha Physical Language (APL) token synthesis
 * 
 * Usage:
 *   const { QuantumAPLSystem } = require('./quantum_apl_system');
 *   const system = new QuantumAPLSystem();
 *   system.simulate(100);  // Run 100 steps
 *   console.log(system.getAnalytics());
 */

const CONST = require('./constants');
const { HelixOperatorAdvisor } = require('./helix_advisor');
const { TriadTracker } = require('./triad_tracker');
const { AlphaTokenSynthesizer, HelixCoordinate, APL_SENTENCES } = require('./alpha_language');

// ================================================================
// QUANTUM APL UNIFIED SYSTEM
// ================================================================

class QuantumAPLSystem {
    constructor(config = {}) {
        // Core components
        this.helixAdvisor = new HelixOperatorAdvisor();
        this.triadTracker = new TriadTracker({
            onUnlock: (data) => this._onTriadUnlock(data)
        });
        this.tokenSynthesizer = new AlphaTokenSynthesizer();
        
        // State
        this.z = config.initialZ || 0.5;
        this.time = 0;
        this.entropy = 0.5;
        this.phi = 0.0;
        
        // History
        this.history = {
            z: [],
            operators: [],
            tokens: [],
            triadEvents: []
        };
        
        // Configuration
        this.verbose = config.verbose !== false;
        this.maxHistory = config.maxHistory || 1000;
        
        // Evolution parameters
        this.dt = config.dt || 0.01;
        this.zDriftRate = config.zDriftRate || 0.02;
        this.zNoiseScale = config.zNoiseScale || 0.01;
        
        // Pump configuration (to drive z toward target)
        this.pumpEnabled = config.pumpEnabled !== false;
        this.pumpTarget = config.pumpTarget || CONST.Z_CRITICAL;
        this.pumpGain = config.pumpGain || 0.1;
    }

    // ================================================================
    // TRIAD CALLBACK
    // ================================================================

    _onTriadUnlock(data) {
        // Update helix advisor with new TRIAD state
        this.helixAdvisor.setTriadState({
            unlocked: true,
            completions: data.completions
        });
        
        // Update token synthesizer
        this.tokenSynthesizer.setTriadState({
            unlocked: true,
            completions: data.completions
        });
        
        // Record event
        this.history.triadEvents.push({
            event: 'UNLOCKED',
            z: data.z,
            completions: data.completions,
            time: this.time
        });
        
        if (this.verbose) {
            console.log(`\n${'='.repeat(60)}`);
            console.log('TRIAD UNLOCK ACHIEVED');
            console.log(`${'='.repeat(60)}`);
            console.log(`  Z-coordinate: ${data.z.toFixed(4)}`);
            console.log(`  Completions: ${data.completions}`);
            console.log(`  t6 gate shifted: ${CONST.Z_CRITICAL.toFixed(4)} → ${CONST.TRIAD_T6.toFixed(4)}`);
            console.log(`${'='.repeat(60)}\n`);
        }
    }

    // ================================================================
    // Z-COORDINATE EVOLUTION
    // ================================================================

    /**
     * Evolve z-coordinate for one time step
     * Includes drift, noise, and optional pumping toward target
     */
    evolveZ() {
        // Base drift (slight upward tendency)
        let dz = this.zDriftRate * this.dt;
        
        // Add noise
        dz += (Math.random() - 0.5) * this.zNoiseScale;
        
        // Pump toward target (proportional control)
        if (this.pumpEnabled) {
            const error = this.pumpTarget - this.z;
            dz += this.pumpGain * error * this.dt;
        }
        
        // Apply change
        this.z = Math.max(0, Math.min(1, this.z + dz));
        
        return this.z;
    }

    /**
     * Set z-coordinate directly (for testing)
     * @param {number} z
     */
    setZ(z) {
        this.z = Math.max(0, Math.min(1, z));
    }

    // ================================================================
    // SIMULATION STEP
    // ================================================================

    /**
     * Execute one simulation step
     * @returns {Object} - Step result
     */
    step() {
        // 1. Evolve z-coordinate
        const prevZ = this.z;
        this.evolveZ();
        this.time += this.dt;
        
        // 2. Update TRIAD tracker
        const triadResult = this.triadTracker.update(this.z);
        
        // 3. Get helix mapping
        const helixInfo = this.helixAdvisor.describe(this.z);
        
        // 4. Select operator from window
        const operator = this._selectOperator(helixInfo.operators);
        
        // 5. Generate APL token
        const aplToken = this.tokenSynthesizer.fromZ(this.z);
        
        // 6. Record history
        this._recordHistory({
            z: this.z,
            operator,
            helixInfo,
            aplToken,
            triadResult
        });
        
        // 7. Update derived quantities
        this.entropy = this._computeEntropy();
        this.phi = this._computePhi();
        
        return {
            time: this.time,
            z: this.z,
            dz: this.z - prevZ,
            operator,
            harmonic: helixInfo.harmonic,
            truthChannel: helixInfo.truthChannel,
            operatorWindow: helixInfo.operators,
            t6Gate: helixInfo.t6Gate,
            triadUnlocked: this.triadTracker.unlocked,
            triadCompletions: this.triadTracker.completions,
            aplToken,
            entropy: this.entropy,
            phi: this.phi
        };
    }

    /**
     * Select operator from window (weighted random)
     */
    _selectOperator(operators) {
        if (!operators || operators.length === 0) return '()';
        
        // Weight by position (first operators preferred)
        const weights = operators.map((_, i) => 1 / (i + 1));
        const totalWeight = weights.reduce((a, b) => a + b, 0);
        
        let rand = Math.random() * totalWeight;
        for (let i = 0; i < operators.length; i++) {
            rand -= weights[i];
            if (rand <= 0) return operators[i];
        }
        
        return operators[0];
    }

    /**
     * Compute entropy (simplified)
     */
    _computeEntropy() {
        // Entropy increases away from critical lens
        const distFromLens = Math.abs(this.z - CONST.Z_CRITICAL);
        return Math.min(1, 0.3 + distFromLens * 2);
    }

    /**
     * Compute integrated information (simplified)
     */
    _computePhi() {
        // Phi increases near critical lens
        const deltaSNeg = CONST.computeDeltaSNeg(this.z);
        return deltaSNeg * 0.8;
    }

    /**
     * Record step in history
     */
    _recordHistory(data) {
        this.history.z.push(data.z);
        this.history.operators.push({
            operator: data.operator,
            harmonic: data.helixInfo.harmonic,
            truthChannel: data.helixInfo.truthChannel,
            time: this.time
        });
        
        if (data.aplToken) {
            this.history.tokens.push({
                ...data.aplToken,
                time: this.time
            });
        }
        
        // Trim history
        if (this.history.z.length > this.maxHistory) {
            this.history.z.shift();
            this.history.operators.shift();
        }
        if (this.history.tokens.length > this.maxHistory) {
            this.history.tokens.shift();
        }
    }

    // ================================================================
    // SIMULATION RUNNER
    // ================================================================

    /**
     * Run simulation for N steps
     * @param {number} steps - Number of steps
     * @param {Object} options - { logInterval, targetZ }
     * @returns {Array} - Step results
     */
    simulate(steps, options = {}) {
        const logInterval = options.logInterval || Math.max(1, Math.floor(steps / 10));
        
        if (options.targetZ !== undefined) {
            this.pumpTarget = options.targetZ;
        }
        
        const results = [];
        
        if (this.verbose) {
            console.log(`\nStarting simulation: ${steps} steps`);
            console.log(`Initial z: ${this.z.toFixed(4)}, Target: ${this.pumpTarget.toFixed(4)}`);
            console.log(`TRIAD state: ${this.triadTracker.analyzerReport()}`);
            console.log('-'.repeat(60));
        }
        
        for (let i = 0; i < steps; i++) {
            const result = this.step();
            results.push(result);
            
            if (this.verbose && (i + 1) % logInterval === 0) {
                console.log(
                    `Step ${i + 1}: z=${result.z.toFixed(4)} | ` +
                    `${result.harmonic}/${result.truthChannel} | ` +
                    `op=${result.operator} | ` +
                    `TRIAD=${result.triadCompletions}/3`
                );
            }
        }
        
        if (this.verbose) {
            console.log('-'.repeat(60));
            console.log(`Simulation complete. Final z: ${this.z.toFixed(4)}`);
            console.log(`${this.triadTracker.analyzerReport()}`);
        }
        
        return results;
    }

    /**
     * Run simulation to achieve TRIAD unlock
     * @param {number} maxSteps - Maximum steps to attempt
     * @returns {Object} - { success, steps, finalState }
     */
    simulateToUnlock(maxSteps = 1000) {
        if (this.verbose) {
            console.log('\nSimulating to achieve TRIAD unlock...');
            console.log(`Requires 3 passes through z ≥ ${CONST.TRIAD_HIGH}`);
        }
        
        let steps = 0;
        let phase = 0;
        const halfPeriod = 25;  // Steps per half-cycle
        
        // Start at middle z
        this.z = 0.83;
        
        while (!this.triadTracker.unlocked && steps < maxSteps) {
            // Direct z manipulation to create reliable oscillation
            phase++;
            const cycle = Math.floor(phase / halfPeriod);
            const inHighPhase = (cycle % 2) === 0;
            
            // Target z values that cross thresholds
            const targetZ = inHighPhase ? 0.87 : 0.78;
            
            // Move toward target with some damping
            const error = targetZ - this.z;
            this.z = this.z + 0.1 * error + (Math.random() - 0.5) * 0.005;
            this.z = Math.max(0, Math.min(1, this.z));
            
            // Update time
            this.time += this.dt;
            
            // Update TRIAD tracker
            const triadResult = this.triadTracker.update(this.z);
            
            // Get helix mapping and record
            const helixInfo = this.helixAdvisor.describe(this.z);
            const operator = this._selectOperator(helixInfo.operators);
            
            this._recordHistory({
                z: this.z,
                operator,
                helixInfo,
                aplToken: this.tokenSynthesizer.fromZ(this.z),
                triadResult
            });
            
            steps++;
            
            if (this.verbose && triadResult.event) {
                console.log(`  Step ${steps}: z=${this.z.toFixed(4)} - ${triadResult.event} (${this.triadTracker.completions}/3)`);
            }
        }
        
        return {
            success: this.triadTracker.unlocked,
            steps,
            finalState: this.getState()
        };
    }

    // ================================================================
    // STATE & ANALYTICS
    // ================================================================

    /**
     * Get current system state
     * @returns {Object}
     */
    getState() {
        const helixInfo = this.helixAdvisor.describe(this.z);
        
        return {
            z: this.z,
            time: this.time,
            entropy: this.entropy,
            phi: this.phi,
            helix: helixInfo,
            triad: this.triadTracker.getState(),
            phase: CONST.getPhase(this.z),
            distanceToLens: CONST.distanceToCritical(this.z)
        };
    }

    /**
     * Get simulation analytics
     * @returns {Object}
     */
    getAnalytics() {
        const zArr = this.history.z;
        const avgZ = zArr.length > 0 ? zArr.reduce((a, b) => a + b) / zArr.length : 0;
        const maxZ = zArr.length > 0 ? Math.max(...zArr) : 0;
        const minZ = zArr.length > 0 ? Math.min(...zArr) : 0;
        
        // Operator distribution
        const opCounts = {};
        for (const entry of this.history.operators) {
            opCounts[entry.operator] = (opCounts[entry.operator] || 0) + 1;
        }
        
        // Harmonic distribution
        const harmonicCounts = {};
        for (const entry of this.history.operators) {
            harmonicCounts[entry.harmonic] = (harmonicCounts[entry.harmonic] || 0) + 1;
        }
        
        return {
            totalSteps: this.history.z.length,
            time: this.time,
            z: { avg: avgZ, min: minZ, max: maxZ, final: this.z },
            operators: opCounts,
            harmonics: harmonicCounts,
            triad: this.triadTracker.getState(),
            triadEvents: this.history.triadEvents,
            tokenCount: this.history.tokens.length
        };
    }

    /**
     * Get recent APL tokens
     * @param {number} n
     * @returns {Array}
     */
    getRecentTokens(n = 10) {
        return this.history.tokens.slice(-n);
    }

    /**
     * Reset system to initial state
     */
    reset() {
        this.z = 0.5;
        this.time = 0;
        this.entropy = 0.5;
        this.phi = 0.0;
        this.history = { z: [], operators: [], tokens: [], triadEvents: [] };
        this.triadTracker.reset();
        this.helixAdvisor.setTriadState({ unlocked: false, completions: 0 });
        this.tokenSynthesizer.setTriadState({ unlocked: false, completions: 0 });
    }

    // ================================================================
    // EXPORT & REPORTING
    // ================================================================

    /**
     * Export state for analysis
     * @returns {Object}
     */
    exportState() {
        return {
            state: this.getState(),
            analytics: this.getAnalytics(),
            history: {
                z: this.history.z.slice(-100),
                operators: this.history.operators.slice(-20),
                tokens: this.history.tokens.slice(-10),
                triadEvents: this.history.triadEvents
            },
            config: {
                pumpTarget: this.pumpTarget,
                pumpGain: this.pumpGain,
                dt: this.dt
            }
        };
    }

    /**
     * Generate summary report
     * @returns {string}
     */
    summary() {
        const state = this.getState();
        const analytics = this.getAnalytics();
        
        const lines = [
            '='.repeat(70),
            'QUANTUM APL SYSTEM SUMMARY',
            '='.repeat(70),
            '',
            'Current State:',
            `  z-coordinate: ${state.z.toFixed(4)}`,
            `  Phase: ${state.phase}`,
            `  Entropy: ${state.entropy.toFixed(4)}`,
            `  Integrated Information (Φ): ${state.phi.toFixed(4)}`,
            '',
            'Helix Mapping:',
            `  Harmonic: ${state.helix.harmonic}`,
            `  Truth Channel: ${state.helix.truthChannel}`,
            `  Operator Window: ${state.helix.operators.join(', ')}`,
            `  t6 Gate: ${state.helix.t6Gate.toFixed(4)}`,
            '',
            'TRIAD Status:',
            `  Unlocked: ${state.triad.unlocked}`,
            `  Completions: ${state.triad.completions}/3`,
            `  ${this.triadTracker.analyzerReport()}`,
            '',
            'Analytics:',
            `  Total Steps: ${analytics.totalSteps}`,
            `  Z Range: [${analytics.z.min.toFixed(4)}, ${analytics.z.max.toFixed(4)}]`,
            `  Average Z: ${analytics.z.avg.toFixed(4)}`,
            '',
            'Operator Distribution:',
            ...Object.entries(analytics.operators)
                .sort((a, b) => b[1] - a[1])
                .map(([op, count]) => `  ${op}: ${count} (${(count / analytics.totalSteps * 100).toFixed(1)}%)`),
            '='.repeat(70)
        ];
        
        return lines.join('\n');
    }
}

// ================================================================
// EXPORTS
// ================================================================

module.exports = {
    QuantumAPLSystem,
    HelixOperatorAdvisor,
    TriadTracker,
    AlphaTokenSynthesizer,
    HelixCoordinate,
    APL_SENTENCES,
    CONST
};
