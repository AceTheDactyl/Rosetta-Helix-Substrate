const CONST = require('../constants');

class LcgRng {
  constructor(seed = null) {
    const base = (seed != null) ? seed >>> 0 : (Date.now() >>> 0);
    this.state = (base === 0 ? 123456789 : base) >>> 0;
  }
  next() {
    // LCG: Numerical Recipes constants
    this.state = (Math.imul(this.state, 1664525) + 1013904223) >>> 0;
    return this.state / 0x100000000; // [0,1)
  }
}

function makeEngineRng() {
  if (CONST.QAPL_RANDOM_SEED == null) return null;
  return new LcgRng(CONST.QAPL_RANDOM_SEED);
}

module.exports = { LcgRng, makeEngineRng };
