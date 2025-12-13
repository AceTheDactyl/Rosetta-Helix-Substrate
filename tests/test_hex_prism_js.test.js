const assert = (c, m) => { if (!c) throw new Error(m || 'assert'); };
const hexPrism = require('../src/hex_prism');
const CONST = require('../src/constants');

// Computes 63 vertices and sequential IDs
(() => {
  const vertices = hexPrism.computeHexPrismVertices(0.866);
  assert(vertices.length === 63, 'vertex count');
  vertices.forEach((v, i) => assert(v.id === i, 'sequential ids'));
})();

// Ring structure (top)
(() => {
  const vertices = hexPrism.computeHexPrismVertices(0.866);
  assert(vertices[0].ring === 0, 'top center ring');
  assert(vertices[1].ring === 1, 'top ring1 start');
  assert(vertices[7].ring === 2, 'top ring2 start');
  assert(vertices[19].ring === 3, 'top ring3 start');
})();

// R increases as |z - z_c| increases (ΔS_neg decreases)
(() => {
  const z1 = CONST.Z_CRITICAL;
  const z2 = CONST.Z_CRITICAL + 0.05;
  const z3 = CONST.Z_CRITICAL + 0.10;
  const R1 = CONST.hexPrismRadius(CONST.computeDeltaSNeg(z1));
  const R2 = CONST.hexPrismRadius(CONST.computeDeltaSNeg(z2));
  const R3 = CONST.hexPrismRadius(CONST.computeDeltaSNeg(z3));
  assert(R2 > R1, 'R2 > R1');
  assert(R3 > R2, 'R3 > R2');
})();

// Exports geometry sidecar shape
(() => {
  const sidecar = hexPrism.exportGeometrySidecar(0.866);
  assert(typeof sidecar.version === 'string', 'version');
  assert(typeof sidecar.z === 'number', 'z');
  assert(typeof sidecar.delta_S_neg === 'number', 'delta');
  assert(Array.isArray(sidecar.vertices), 'vertices array');
  assert(sidecar.vertices.length === 63, 'vertices length');
  assert(typeof sidecar.geometry.R === 'number', 'R');
  assert(typeof sidecar.geometry.H === 'number', 'H');
  assert(typeof sidecar.geometry.phi === 'number', 'phi');
  console.log('Hex prism JS tests passed');
})();
