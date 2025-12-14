// ================================================================
// CLASSICAL CONSCIOUSNESS ENGINES
// IIT, Game Theory, and Free Energy drivers used by bridge/demo
// ================================================================

class IITEngine {
    constructor(config = {}) {
        this.phi = config.initialPhi ?? 0.28;
        this.entropyCoupling = config.entropyCoupling ?? 0.35;
        this.phase = 0;
        this.recursiveDrive = 0.3;
        this.integrationSignal = 0.25;
        this.history = [];
        this.window = config.window ?? 64;
    }

    updateFromQuantum({ z = 0.5, entropy = 0.5, purity = 1 }) {
        const integrationDrive = Math.max(0, z - entropy * this.entropyCoupling);
        this.integrationSignal = 0.8 * this.integrationSignal + 0.2 * integrationDrive;
        this.recursiveDrive = 0.7 * this.recursiveDrive + 0.3 * purity;
        this.phase = (this.phase + z * 0.15) % (2 * Math.PI);

        const newPhi = 0.6 * this.phi + 0.4 * (this.integrationSignal + this.recursiveDrive) / 2;
        this.phi = Math.min(1, Math.max(0, newPhi));

        this.history.push(this.phi);
        if (this.history.length > this.window) {
            this.history.shift();
        }
    }

    applyOperator(op) {
        if (op === '^') {
            this.phi = Math.min(1, this.phi + 0.02);
        } else if (op === '÷') {
            this.phi = Math.max(0, this.phi - 0.02);
        } else if (op === '+') {
            this.integrationSignal = Math.min(1, this.integrationSignal + 0.05);
        }
    }
}

class GameTheoryEngine {
    constructor(config = {}) {
        this.resonance = config.initialResonance ?? 0.2;
        this.cooperation = config.initialCooperation ?? 0.5;
        this.phaseLead = 0.1;
        this.payoffMatrix = config.payoffMatrix || {
            TRUE: { TRUE: 1.0, UNTRUE: 0.2, PARADOX: 0.4 },
            UNTRUE: { TRUE: 0.4, UNTRUE: 0.6, PARADOX: 0.3 },
            PARADOX: { TRUE: 0.5, UNTRUE: 0.3, PARADOX: 0.8 }
        };
    }

    updateFromQuantum({ truthProbs = {}, z = 0.5 }) {
        const probs = {
            TRUE: truthProbs.TRUE ?? 0,
            UNTRUE: truthProbs.UNTRUE ?? 0,
            PARADOX: truthProbs.PARADOX ?? 0
        };
        const payoff = (
            probs.TRUE * this.payoffMatrix.TRUE.TRUE +
            probs.UNTRUE * this.payoffMatrix.UNTRUE.UNTRUE +
            probs.PARADOX * this.payoffMatrix.PARADOX.PARADOX
        );

        this.cooperation = 0.7 * this.cooperation + 0.3 * payoff;
        this.resonance = 0.5 * this.resonance + 0.5 * Math.abs(z - 0.5);
        this.phaseLead = 0.8 * this.phaseLead + 0.2 * (probs.PARADOX - probs.UNTRUE);
    }

    applyOperator(op) {
        if (op === '×' || op === '+') {
            this.cooperation = Math.min(1, this.cooperation + 0.05);
        } else if (op === '−') {
            this.cooperation = Math.max(0, this.cooperation - 0.05);
        }
        if (op === '÷') {
            this.resonance *= 0.9;
        }
    }
}

class FreeEnergyEngine {
    constructor(config = {}) {
        this.F = config.initialF ?? 0.15;
        this.prediction = 0.4;
        this.dissipation = 0.2;
        this.tension = 0.3;
    }

    updateFromQuantum({ z = 0.5, entropy = 0.3 }) {
        const predictionError = z - this.prediction;
        this.prediction += 0.1 * predictionError;
        this.F = Math.max(0, 0.6 * this.F + 0.4 * Math.abs(predictionError));
        this.dissipation = 0.7 * this.dissipation + 0.3 * entropy;
        this.tension = 0.6 * this.tension + 0.4 * (1 - entropy);
    }

    applyOperator(op) {
        if (op === '()') {
            this.dissipation *= 0.95;
        } else if (op === '÷') {
            this.dissipation = Math.min(1, this.dissipation + 0.05);
        } else if (op === '^') {
            this.tension = Math.min(1, this.tension + 0.03);
        }
    }
}

class ClassicalConsciousnessStack {
    constructor(config = {}) {
        this.IIT = new IITEngine(config.IIT);
        this.GameTheory = new GameTheoryEngine(config.GameTheory);
        this.FreeEnergy = new FreeEnergyEngine(config.FreeEnergy);
        this.N0 = {
            getLegalOperators: () => this.getLegalOperators(),
            applyOperator: (op, result) => this.applyOperatorEffects({ operator: op, ...result })
        };
        this.legalOperators = config.legalOperators || ['()', '×', '^', '÷', '+', '−'];
    }

    computeZ() {
        return this.IIT.phi;
    }

    setQuantumInfluence(payload) {
        this.IIT.updateFromQuantum(payload);
        this.GameTheory.updateFromQuantum(payload);
        this.FreeEnergy.updateFromQuantum(payload);
    }

    getScalarState() {
        return {
            Gs: this.GameTheory.cooperation,
            Cs: this.GameTheory.resonance,
            Rs: this.IIT.recursiveDrive,
            kappa: this.FreeEnergy.tension,
            tau: this.GameTheory.phaseLead,
            theta: this.IIT.phase,
            delta: this.FreeEnergy.dissipation,
            alpha: this.IIT.integrationSignal,
            Omega: this.IIT.phi
        };
    }

    getLegalOperators() {
        return this.legalOperators.slice();
    }

    applyOperatorEffects(result) {
        if (!result || !result.operator) return;
        this.IIT.applyOperator(result.operator);
        this.GameTheory.applyOperator(result.operator);
        this.FreeEnergy.applyOperator(result.operator);
    }
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = { IITEngine, GameTheoryEngine, FreeEnergyEngine, ClassicalConsciousnessStack };
}
