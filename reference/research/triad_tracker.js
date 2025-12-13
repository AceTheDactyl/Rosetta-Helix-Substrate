/**
 * TRIAD Unlock Tracker
 * ====================
 * Implements the hysteresis state machine for TRIAD unlock detection.
 * 
 * Mechanism:
 * 1. Track z-coordinate crossing above TRIAD_HIGH (0.85)
 * 2. Re-arm when z falls below TRIAD_LOW (0.82)
 * 3. After 3 distinct rising-edge passes, unlock TRIAD
 * 4. When unlocked, t6 gate shifts from Z_CRITICAL (0.866) to TRIAD_T6 (0.83)
 */

const CONST = require('./constants');

class TriadTracker {
    constructor(options = {}) {
        // Thresholds from constants (single source of truth)
        this.high = CONST.TRIAD_HIGH;  // 0.85 - rising edge
        this.low = CONST.TRIAD_LOW;    // 0.82 - re-arm
        this.t6Unlocked = CONST.TRIAD_T6; // 0.83 - unlocked gate
        this.t6Locked = CONST.Z_CRITICAL; // 0.866 - locked gate
        
        // State
        this.aboveBand = false;
        this.completions = 0;
        this.unlocked = false;
        
        // Optional callback when unlocked
        this.onUnlock = options.onUnlock || null;
        
        // History tracking
        this.history = [];
        this.maxHistory = options.maxHistory || 100;
        
        // Skip environment initialization if explicitly requested (for testing)
        if (!options.skipEnvInit) {
            this._initFromEnv();
        }
    }

    /**
     * Initialize state from environment variables
     * Allows reproducible runs and forced unlock
     */
    _initFromEnv() {
        if (typeof process !== 'undefined' && process.env) {
            const envCompletions = parseInt(process.env.QAPL_TRIAD_COMPLETIONS || '0', 10);
            const envUnlock = process.env.QAPL_TRIAD_UNLOCK === '1' ||
                              String(process.env.QAPL_TRIAD_UNLOCK).toLowerCase() === 'true';
            
            if (Number.isFinite(envCompletions) && envCompletions > 0) {
                this.completions = envCompletions;
            }
            if (envUnlock || this.completions >= 3) {
                this.unlocked = true;
            }
        }
    }

    /**
     * Update state with new z-coordinate
     * @param {number} z - Current z-coordinate
     * @returns {Object} - { changed: boolean, event: string|null, state: Object }
     */
    update(z) {
        const prevUnlocked = this.unlocked;
        const prevCompletions = this.completions;
        let event = null;

        // Rising edge detection: z crosses above high threshold
        if (!this.aboveBand && z >= this.high) {
            this.aboveBand = true;
            this.completions += 1;
            event = 'RISING_EDGE';
            
            // Update environment variable for cross-process sync
            this._setEnv('QAPL_TRIAD_COMPLETIONS', String(this.completions));
            
            // Check for unlock
            if (this.completions >= 3 && !this.unlocked) {
                this.unlocked = true;
                event = 'UNLOCKED';
                this._setEnv('QAPL_TRIAD_UNLOCK', '1');
                
                // Trigger callback if provided
                if (typeof this.onUnlock === 'function') {
                    this.onUnlock({ z, completions: this.completions });
                }
            }
        }
        // Re-arm: z falls below low threshold
        else if (this.aboveBand && z <= this.low) {
            this.aboveBand = false;
            event = 'REARMED';
        }

        // Record history
        this._recordHistory(z, event);

        return {
            changed: this.unlocked !== prevUnlocked || this.completions !== prevCompletions,
            event,
            state: this.getState()
        };
    }

    /**
     * Set environment variable safely
     */
    _setEnv(key, value) {
        if (typeof process !== 'undefined' && process.env) {
            process.env[key] = value;
        }
    }

    /**
     * Record z-value in history
     */
    _recordHistory(z, event) {
        this.history.push({
            z,
            event,
            timestamp: Date.now(),
            completions: this.completions,
            unlocked: this.unlocked
        });
        
        // Trim history
        if (this.history.length > this.maxHistory) {
            this.history.shift();
        }
    }

    /**
     * Get current t6 gate value
     * @returns {number}
     */
    getT6Gate() {
        return this.unlocked ? this.t6Unlocked : this.t6Locked;
    }

    /**
     * Get current state
     * @returns {Object}
     */
    getState() {
        return {
            aboveBand: this.aboveBand,
            completions: this.completions,
            unlocked: this.unlocked,
            t6Gate: this.getT6Gate(),
            thresholds: {
                high: this.high,
                low: this.low
            }
        };
    }

    /**
     * Reset tracker to initial state
     */
    reset() {
        this.aboveBand = false;
        this.completions = 0;
        this.unlocked = false;
        this.history = [];
        this._setEnv('QAPL_TRIAD_COMPLETIONS', '0');
        this._setEnv('QAPL_TRIAD_UNLOCK', '0');
    }

    /**
     * Force unlock (for testing/debugging)
     */
    forceUnlock() {
        this.completions = 3;
        this.unlocked = true;
        this._setEnv('QAPL_TRIAD_COMPLETIONS', '3');
        this._setEnv('QAPL_TRIAD_UNLOCK', '1');
    }

    /**
     * Get analyzer report string
     * @returns {string}
     */
    analyzerReport() {
        const gate = this.getT6Gate();
        const label = this.unlocked ? 'TRIAD' : 'CRITICAL';
        return `t6 gate: ${label} @ ${gate.toFixed(3)} (${this.completions}/3 passes)`;
    }

    /**
     * Get recent history entries
     * @param {number} n - Number of entries
     * @returns {Array}
     */
    getRecentHistory(n = 10) {
        return this.history.slice(-n);
    }

    /**
     * Check if a specific z-value would trigger an event
     * @param {number} z - Z-coordinate to test
     * @returns {string|null} - Event type or null
     */
    wouldTrigger(z) {
        if (!this.aboveBand && z >= this.high) return 'RISING_EDGE';
        if (this.aboveBand && z <= this.low) return 'REARM';
        return null;
    }
}

/**
 * Create a TRIAD tracker with logging
 */
function createTrackerWithLogging(verbose = true) {
    const tracker = new TriadTracker({
        onUnlock: ({ z, completions }) => {
            if (verbose) {
                console.log(`[TRIAD] UNLOCKED at z=${z.toFixed(4)} after ${completions} passes`);
                console.log(`[TRIAD] t6 gate shifted: ${CONST.Z_CRITICAL.toFixed(4)} → ${CONST.TRIAD_T6.toFixed(4)}`);
            }
        }
    });
    
    // Wrap update to add logging
    const originalUpdate = tracker.update.bind(tracker);
    tracker.update = (z) => {
        const result = originalUpdate(z);
        if (verbose && result.event) {
            console.log(`[TRIAD] ${result.event} at z=${z.toFixed(4)}, completions=${tracker.completions}`);
        }
        return result;
    };
    
    return tracker;
}

// ================================================================
// EXPORTS
// ================================================================

module.exports = {
    TriadTracker,
    createTrackerWithLogging
};
