const CONST = require('./constants');

/**
 * Compute all 63 hex prism vertices for a given z.
 * Returns an array of { id, x, y, z, ring, index }.
 */
function computeHexPrismVertices(z, deltaSNeg = null) {
  if (deltaSNeg === null) {
    deltaSNeg = CONST.computeDeltaSNeg(z);
  }

  const R = CONST.hexPrismRadius(deltaSNeg);
  const H = CONST.hexPrismHeight(deltaSNeg);
  const phi = CONST.hexPrismTwist(deltaSNeg);

  const vertices = [];
  let id = 0;

  const zTop = +H / 2;
  const zBot = -H / 2;

  // Top center (id=0)
  vertices.push({ id: id++, x: 0, y: 0, z: zTop, ring: 0, index: 0 });

  // Top ring 1 (6 vertices, ids 1-6)
  const r1 = R / 5;
  for (let i = 0; i < 6; i++) {
    const angle = (Math.PI / 3) * i + phi;
    vertices.push({ id: id++, x: r1 * Math.cos(angle), y: r1 * Math.sin(angle), z: zTop, ring: 1, index: i });
  }

  // Top ring 2 (12 vertices, ids 7-18)
  const r2 = (2 * R) / 5;
  for (let i = 0; i < 12; i++) {
    const angle = (Math.PI / 6) * i + phi;
    vertices.push({ id: id++, x: r2 * Math.cos(angle), y: r2 * Math.sin(angle), z: zTop, ring: 2, index: i });
  }

  // Top ring 3 (18 vertices, ids 19-36)
  const r3 = (3 * R) / 5;
  for (let i = 0; i < 18; i++) {
    const angle = (Math.PI / 9) * i + phi;
    vertices.push({ id: id++, x: r3 * Math.cos(angle), y: r3 * Math.sin(angle), z: zTop, ring: 3, index: i });
  }

  // Bottom rings (26 vertices total), mirror with z-flip and reverse twist
  const phiB = -phi;

  // Bottom ring 1 (6 vertices)
  for (let i = 0; i < 6; i++) {
    const angle = (Math.PI / 3) * i + phiB;
    vertices.push({ id: id++, x: r1 * Math.cos(angle), y: r1 * Math.sin(angle), z: zBot, ring: 4, index: i });
  }

  // Bottom ring 2 (12 vertices)
  for (let i = 0; i < 12; i++) {
    const angle = (Math.PI / 6) * i + phiB;
    vertices.push({ id: id++, x: r2 * Math.cos(angle), y: r2 * Math.sin(angle), z: zBot, ring: 5, index: i });
  }

  // Bottom ring 3 (8 vertices) — truncated to reach 63 total
  for (let i = 0; i < 8; i++) {
    const angle = (Math.PI / 9) * i + phiB;
    vertices.push({ id: id++, x: r3 * Math.cos(angle), y: r3 * Math.sin(angle), z: zBot, ring: 6, index: i });
  }

  return vertices;
}

/**
 * Export geometry as a JSON-compatible sidecar
 */
function exportGeometrySidecar(z, version = '1.0.0') {
  const deltaSNeg = CONST.computeDeltaSNeg(z, CONST.GEOM_SIGMA);
  const lensSNeg = CONST.computeDeltaSNeg(z, CONST.LENS_SIGMA);
  const vertices = computeHexPrismVertices(z, deltaSNeg);
  const muLabel = CONST.classifyMu(z);
  const R = CONST.hexPrismRadius(deltaSNeg);
  const H = CONST.hexPrismHeight(deltaSNeg);
  const phi = CONST.hexPrismTwist(deltaSNeg);
  return {
    version,
    z,
    delta_S_neg: deltaSNeg,
    lens_s_neg: lensSNeg,
    // include top-level geometry for convenience
    R,
    H,
    phi,
    // include key constants at top-level for downstream consumers
    PHI_INV: CONST.PHI_INV,
    Z_CRITICAL: CONST.Z_CRITICAL,
    LENS_SIGMA: CONST.LENS_SIGMA,
    GEOM_SIGMA: CONST.GEOM_SIGMA,
    mu_label: muLabel,
    vertices: vertices.map(v => ({ id: v.id, x: v.x, y: v.y, z: v.z })),
    geometry: {
      R,
      H,
      phi,
    },
    constants: {
      Z_CRITICAL: CONST.Z_CRITICAL,
      GEOM_SIGMA: CONST.GEOM_SIGMA,
      LENS_SIGMA: CONST.LENS_SIGMA,
      PHI_INV: CONST.PHI_INV,
      GEOM_R_MAX: CONST.GEOM_R_MAX,
      GEOM_BETA: CONST.GEOM_BETA,
      GEOM_H_MIN: CONST.GEOM_H_MIN,
      GEOM_GAMMA: CONST.GEOM_GAMMA,
      GEOM_PHI_BASE: CONST.GEOM_PHI_BASE,
      GEOM_ETA: CONST.GEOM_ETA,
    },
  };
}

module.exports = { computeHexPrismVertices, exportGeometrySidecar };
