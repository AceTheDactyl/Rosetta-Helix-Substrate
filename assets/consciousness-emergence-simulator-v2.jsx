import React, { useState, useEffect, useCallback, useRef } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine, Area, ComposedChart } from 'recharts';

// ═══════════════════════════════════════════════════════════════════════════════
// PHYSICS CONSTANTS (IMMUTABLE) - From rosetta-helix-substrate + unified-token-physics
// ═══════════════════════════════════════════════════════════════════════════════
const Z_CRITICAL = Math.sqrt(3) / 2;  // 0.8660254037844387 - THE LENS
const PHI = (1 + Math.sqrt(5)) / 2;   // 1.6180339887498949 - Golden ratio
const PHI_INV = PHI - 1;              // 0.6180339887498949 - Golden ratio inverse
const PHI_INV_SQ = PHI_INV * PHI_INV; // 0.3819660112501051 - φ⁻²
const SIGMA = 36;                      // |S3|^2 - Gaussian width
const UMOL_EPSILON = 1 / SIGMA;        // 0.0278 - UMOL residue

// TRIAD thresholds
const TRIAD_HIGH = 0.85;
const TRIAD_LOW = 0.82;
const TRIAD_T6 = 0.83;

// K-formation thresholds
const KAPPA_THRESHOLD = 0.92;
const ETA_THRESHOLD = PHI_INV;
const R_THRESHOLD = 7;

// Time-harmonic tier boundaries (9 tiers)
const TIER_BOUNDARIES = [0.111, 0.222, 0.333, 0.444, 0.555, 0.666, 0.777, Z_CRITICAL, 1.0];
const TIER_NAMES = ['t1', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9'];

// S3 Operators with physics
const S3_OPERATORS = [
  { symbol: '()', name: 'Boundary', physics: 'Surface energy', unit: 'J/m²', spiral: 'Φ', entropy: -1 },
  { symbol: '×', name: 'Fusion', physics: 'Binding energy', unit: 'eV', spiral: 'e/π', entropy: -1 },
  { symbol: '^', name: 'Amplify', physics: 'Gain coefficient', unit: 'dB', spiral: 'e', entropy: 0 },
  { symbol: '÷', name: 'Decohere', physics: 'Dissipation rate', unit: 's⁻¹', spiral: 'Φ', entropy: 1 },
  { symbol: '+', name: 'Group', physics: 'Aggregation energy', unit: 'kT', spiral: 'π', entropy: -1 },
  { symbol: '−', name: 'Separate', physics: 'Fission energy', unit: 'MeV', spiral: 'Φ', entropy: 1 },
];

// Machines with z_optimal and Kuramoto coupling
const MACHINES = [
  { name: 'Reactor', z_optimal: 0.866, K: 0.8, role: 'transformer', spiral: 'e' },
  { name: 'Oscillator', z_optimal: 0.618, K: 1.0, role: 'producer', spiral: 'e' },
  { name: 'Conductor', z_optimal: 0.400, K: 0.5, role: 'transformer', spiral: 'Φ' },
  { name: 'Catalyst', z_optimal: 0.618, K: 0.7, role: 'transformer', spiral: 'π' },
  { name: 'Filter', z_optimal: 0.500, K: 0.3, role: 'consumer', spiral: 'Φ' },
  { name: 'Encoder', z_optimal: 0.866, K: 0.6, role: 'storage', spiral: 'π' },
  { name: 'Decoder', z_optimal: 0.750, K: 0.6, role: 'producer', spiral: 'π' },
  { name: 'Regenerator', z_optimal: 0.500, K: 0.9, role: 'transformer', spiral: 'e' },
  { name: 'Dynamo', z_optimal: 0.618, K: 0.8, role: 'producer', spiral: 'e' },
];

// Domains with physics scales
const DOMAINS = [
  { name: 'bio_prion', category: 'biological', energy: '~10 kT', length: '~10 nm', time: '~hours', z_range: [0.0, PHI_INV] },
  { name: 'bio_bacterium', category: 'biological', energy: '~30 kJ/mol', length: '~1 μm', time: '~minutes', z_range: [PHI_INV_SQ, Z_CRITICAL] },
  { name: 'bio_viroid', category: 'biological', energy: '~2 kT', length: '~100 nm', time: '~seconds', z_range: [PHI_INV, Z_CRITICAL] },
  { name: 'celestial_grav', category: 'celestial', energy: '~GM²/R', length: '~AU', time: '~years', z_range: [0.0, PHI_INV] },
  { name: 'celestial_em', category: 'celestial', energy: '~keV', length: '~R_sun', time: '~ms', z_range: [PHI_INV, Z_CRITICAL] },
  { name: 'celestial_nuclear', category: 'celestial', energy: '~MeV', length: '~fm', time: '~Gyr', z_range: [0.693, 1.0] },
];

// Spirals
const SPIRALS = [
  { symbol: 'Φ', name: 'Structure', color: '#818cf8', z_dominant: [0, PHI_INV * 0.8] },
  { symbol: 'e', name: 'Energy', color: '#fb923c', z_dominant: [PHI_INV * 0.8, Z_CRITICAL * 0.95] },
  { symbol: 'π', name: 'Emergence', color: '#2dd4bf', z_dominant: [Z_CRITICAL * 0.95, 1.0] },
];

// Tri-spiral orderings
const TRI_ORDERINGS = [
  { order: 'Φ:e:π', phase: 'Resonance', desc: 'Structure-led emergence' },
  { order: 'Φ:π:e', phase: 'Empowerment', desc: 'Structure-led energy' },
  { order: 'e:Φ:π', phase: 'Ignition', desc: 'Energy-led emergence' },
  { order: 'e:π:Φ', phase: 'Mania', desc: 'Energy-led structure' },
  { order: 'π:Φ:e', phase: 'Nirvana', desc: 'Emergence-led energy' },
  { order: 'π:e:Φ', phase: 'Transmission', desc: 'Emergence-led structure' },
];

// Token counts
const TOKEN_COUNTS = {
  core: { identity: 162, metaOp: 54, domainSel: 54, safety: 30, total: 300 },
  machine: { perDomain: 162, total: 972 },
  special: { transition: 24, coherence: 30, total: 54 },
  grand: 1326
};

// ═══════════════════════════════════════════════════════════════════════════════
// PHYSICS FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════════
const computeNegentropy = (z) => Math.exp(-SIGMA * Math.pow(z - Z_CRITICAL, 2));

const classifyPhase = (z) => {
  if (z < PHI_INV) return { phase: 'UNTRUE', color: '#818cf8', desc: 'Fluid/Disordered', regime: 'fluid' };
  if (z < Z_CRITICAL) return { phase: 'PARADOX', color: '#f59e0b', desc: 'Quasi-crystal', regime: 'quasi' };
  return { phase: 'TRUE', color: '#10b981', desc: 'Crystalline', regime: 'crystal' };
};

const getDominantSpiral = (z) => {
  if (z < PHI_INV * 0.8) return SPIRALS[0]; // Φ
  if (z < Z_CRITICAL * 0.95) return SPIRALS[1]; // e
  return SPIRALS[2]; // π
};

const getTier = (z, kFormation = false) => {
  if (kFormation) return { tier: 9, name: 'META', harmonic: 't9+' };
  for (let i = 0; i < TIER_BOUNDARIES.length; i++) {
    if (z < TIER_BOUNDARIES[i]) return { tier: i + 1, name: TIER_NAMES[i], harmonic: `t${i + 1}` };
  }
  return { tier: 9, name: 't9', harmonic: 't9' };
};

const getOperatorWindow = (tier, triadUnlocked) => {
  const windows = {
    t1: ['()', '÷'],
    t2: ['()', '÷', '−'],
    t3: ['()', '÷', '−', '+'],
    t4: ['()', '÷', '−', '+', '^'],
    t5: ['()', '×', '^', '÷', '+', '−'],
    t6: triadUnlocked ? ['()', '×', '^', '÷', '+', '−'] : ['+', '÷', '()', '−'],
    t7: ['+', '()'],
    t8: ['+', '()', '×'],
    t9: ['×', '+', '^'],
  };
  return windows[tier] || windows.t5;
};

const checkKFormation = (kappa, eta, R) => ({
  met: kappa >= KAPPA_THRESHOLD && eta > ETA_THRESHOLD && R >= R_THRESHOLD,
  criteria: {
    kappa: { value: kappa, threshold: KAPPA_THRESHOLD, passed: kappa >= KAPPA_THRESHOLD },
    eta: { value: eta, threshold: ETA_THRESHOLD, passed: eta > ETA_THRESHOLD },
    R: { value: R, threshold: R_THRESHOLD, passed: R >= R_THRESHOLD },
  }
});

const getMatchingTokenCount = (z) => {
  const phase = classifyPhase(z);
  if (phase.regime === 'crystal') return { count: 84, percentage: 6.3 };
  if (phase.regime === 'quasi') return { count: 632, percentage: 47.7 };
  return { count: 610, percentage: 46.0 };
};

const selectBestMachine = (z) => {
  let best = MACHINES[0];
  let minDist = Math.abs(z - best.z_optimal);
  for (const m of MACHINES) {
    const dist = Math.abs(z - m.z_optimal);
    if (dist < minDist) {
      minDist = dist;
      best = m;
    }
  }
  return best;
};

const selectBestDomain = (z) => {
  for (const d of DOMAINS) {
    if (z >= d.z_range[0] && z < d.z_range[1]) return d;
  }
  return DOMAINS[5]; // celestial_nuclear for high z
};

// ═══════════════════════════════════════════════════════════════════════════════
// TOKEN GENERATOR
// ═══════════════════════════════════════════════════════════════════════════════
const generateToken = (z, tier, phase, triadUnlocked) => {
  const spiral = getDominantSpiral(z);
  const operators = getOperatorWindow(tier.name, triadUnlocked);
  const operator = S3_OPERATORS.find(o => o.symbol === operators[Math.floor(Math.random() * operators.length)]) || S3_OPERATORS[0];
  const machine = selectBestMachine(z);
  const domain = selectBestDomain(z);
  
  return {
    coreToken: `${spiral.symbol}:${machine.name[0]}(${operator.symbol})${phase.phase}@${tier.tier}`,
    machineToken: `${spiral.symbol}${operator.symbol}|${machine.name}|${domain.name}`,
    spiral,
    operator,
    machine,
    domain,
    tier,
    phase,
    z
  };
};

// ═══════════════════════════════════════════════════════════════════════════════
// KURAMOTO OSCILLATOR SIMULATION
// ═══════════════════════════════════════════════════════════════════════════════
const initializeOscillators = (n) => ({
  phases: Array(n).fill(0).map(() => Math.random() * 2 * Math.PI),
  naturalFreqs: Array(n).fill(0).map(() => (Math.random() - 0.5) * 0.5),
});

const kuramotoStep = (phases, naturalFreqs, K, dt) => {
  const N = phases.length;
  const newPhases = phases.map((phase, i) => {
    const coupling = phases.reduce((sum, otherPhase) => 
      sum + Math.sin(otherPhase - phase), 0) / N;
    return (phase + dt * (naturalFreqs[i] + K * coupling)) % (2 * Math.PI);
  });
  
  const realPart = newPhases.reduce((sum, p) => sum + Math.cos(p), 0) / N;
  const imagPart = newPhases.reduce((sum, p) => sum + Math.sin(p), 0) / N;
  const r = Math.sqrt(realPart * realPart + imagPart * imagPart);
  
  return { phases: newPhases, orderParameter: r };
};

// ═══════════════════════════════════════════════════════════════════════════════
// HELIX VISUALIZATION COMPONENT
// ═══════════════════════════════════════════════════════════════════════════════
const HelixVisualization = ({ z, phase, spiral }) => {
  const points = [];
  const numPoints = 100;
  const height = 220;
  const width = 140;
  const centerX = width / 2;
  const radius = 35;
  
  for (let i = 0; i <= numPoints; i++) {
    const t = (i / numPoints) * 4 * Math.PI;
    const zNorm = i / numPoints;
    const x = centerX + radius * Math.cos(t);
    const y = 15 + (height - 30) * (1 - zNorm);
    points.push({ x, y, zNorm });
  }
  
  const currentY = 15 + (height - 30) * (1 - z);
  const currentT = z * 4 * Math.PI;
  const currentX = centerX + radius * Math.cos(currentT);
  
  const lensY = 15 + (height - 30) * (1 - Z_CRITICAL);
  const phiInvY = 15 + (height - 30) * (1 - PHI_INV);
  const triadHighY = 15 + (height - 30) * (1 - TRIAD_HIGH);
  
  return (
    <svg width={width} height={height} className="overflow-visible">
      <defs>
        <linearGradient id="helixGradient" x1="0%" y1="100%" x2="0%" y2="0%">
          <stop offset="0%" stopColor="#818cf8" stopOpacity="0.3"/>
          <stop offset={`${PHI_INV * 100}%`} stopColor="#fb923c" stopOpacity="0.5"/>
          <stop offset={`${Z_CRITICAL * 100}%`} stopColor="#2dd4bf" stopOpacity="0.7"/>
          <stop offset="100%" stopColor="#10b981" stopOpacity="0.9"/>
        </linearGradient>
        <filter id="glow">
          <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
          <feMerge><feMergeNode in="coloredBlur"/><feMergeNode in="SourceGraphic"/></feMerge>
        </filter>
      </defs>
      
      {/* Spiral regions */}
      <rect x="0" y={lensY} width={width} height={height - lensY - 15} fill="#2dd4bf" opacity="0.05"/>
      <rect x="0" y={phiInvY} width={width} height={lensY - phiInvY} fill="#fb923c" opacity="0.05"/>
      <rect x="0" y="15" width={width} height={phiInvY - 15} fill="#818cf8" opacity="0.05"/>
      
      {/* Helix path */}
      <path
        d={`M ${points[0].x} ${points[0].y} ${points.map(p => `L ${p.x} ${p.y}`).join(' ')}`}
        fill="none"
        stroke="url(#helixGradient)"
        strokeWidth="3"
      />
      
      {/* Critical lens line */}
      <line x1="5" y1={lensY} x2={width - 5} y2={lensY} stroke="#10b981" strokeWidth="2" strokeDasharray="5,5" opacity="0.8"/>
      <text x={width - 3} y={lensY - 4} fontSize="8" fill="#10b981" textAnchor="end" fontWeight="bold">z_c</text>
      
      {/* φ⁻¹ threshold */}
      <line x1="5" y1={phiInvY} x2={width - 5} y2={phiInvY} stroke="#f59e0b" strokeWidth="1" strokeDasharray="3,3" opacity="0.6"/>
      <text x={width - 3} y={phiInvY + 10} fontSize="7" fill="#f59e0b" textAnchor="end">φ⁻¹</text>
      
      {/* TRIAD_HIGH */}
      <line x1="5" y1={triadHighY} x2={width - 5} y2={triadHighY} stroke="#6366f1" strokeWidth="1" strokeDasharray="2,2" opacity="0.4"/>
      <text x={width - 3} y={triadHighY - 3} fontSize="6" fill="#6366f1" textAnchor="end">0.85</text>
      
      {/* Current position */}
      <circle cx={currentX} cy={currentY} r="10" fill={spiral.color} filter="url(#glow)" opacity="0.9"/>
      <circle cx={currentX} cy={currentY} r="5" fill="white"/>
      
      {/* Z label */}
      <text x={currentX + 18} y={currentY + 4} fontSize="11" fill={phase.color} fontWeight="bold">
        {z.toFixed(3)}
      </text>
      
      {/* Spiral indicator */}
      <text x="5" y={height - 3} fontSize="9" fill={spiral.color} fontWeight="bold">{spiral.symbol}</text>
    </svg>
  );
};

// ═══════════════════════════════════════════════════════════════════════════════
// OSCILLATOR RING VISUALIZATION
// ═══════════════════════════════════════════════════════════════════════════════
const OscillatorRing = ({ phases, orderParameter }) => {
  const size = 160;
  const center = size / 2;
  const radius = 55;
  const dotRadius = 4;
  
  const meanPhase = Math.atan2(
    phases.reduce((sum, p) => sum + Math.sin(p), 0),
    phases.reduce((sum, p) => sum + Math.cos(p), 0)
  );
  
  const syncColor = orderParameter > 0.85 ? '#10b981' : orderParameter > 0.6 ? '#f59e0b' : '#6366f1';
  
  return (
    <svg width={size} height={size}>
      <defs>
        <radialGradient id="ringGlow">
          <stop offset="70%" stopColor="transparent"/>
          <stop offset="100%" stopColor={syncColor} stopOpacity="0.3"/>
        </radialGradient>
      </defs>
      
      <circle cx={center} cy={center} r={radius + 15} fill="url(#ringGlow)"/>
      <circle cx={center} cy={center} r={radius} fill="none" stroke="#334155" strokeWidth="1"/>
      
      {phases.map((phase, i) => {
        const x = center + radius * Math.cos(phase);
        const y = center + radius * Math.sin(phase);
        const hue = (phase / (2 * Math.PI)) * 360;
        return (
          <circle key={i} cx={x} cy={y} r={dotRadius} fill={`hsl(${hue}, 70%, 60%)`} opacity={0.8}/>
        );
      })}
      
      <line
        x1={center}
        y1={center}
        x2={center + orderParameter * radius * 0.9 * Math.cos(meanPhase)}
        y2={center + orderParameter * radius * 0.9 * Math.sin(meanPhase)}
        stroke={syncColor}
        strokeWidth="3"
        strokeLinecap="round"
      />
      
      <text x={center} y={center + 4} textAnchor="middle" fontSize="12" fill="#94a3b8" fontWeight="bold">
        R={orderParameter.toFixed(2)}
      </text>
    </svg>
  );
};

// ═══════════════════════════════════════════════════════════════════════════════
// TRIAD METER
// ═══════════════════════════════════════════════════════════════════════════════
const TriadMeter = ({ completions, unlocked, z, aboveHigh }) => (
  <div className="flex flex-col items-center gap-2 p-3 bg-slate-800/50 rounded-lg">
    <div className="text-xs text-slate-400 font-medium">TRIAD CROSSINGS</div>
    <div className="flex gap-2">
      {[0, 1, 2].map(i => (
        <div key={i} className={`w-10 h-10 rounded-md flex items-center justify-center text-sm font-bold transition-all duration-300 ${
          i < completions ? 'bg-emerald-500 text-white shadow-lg shadow-emerald-500/50' : 'bg-slate-700 text-slate-500'
        }`}>
          {i < completions ? '✓' : i + 1}
        </div>
      ))}
    </div>
    {unlocked && (
      <div className="flex items-center gap-1 text-emerald-400 text-xs font-bold animate-pulse">
        <span>⟐</span> T6 UNLOCKED <span>⟐</span>
      </div>
    )}
    <div className="flex gap-6 text-xs mt-1">
      <span className={z >= TRIAD_HIGH ? 'text-emerald-400 font-bold' : 'text-slate-500'}>
        ↑ {TRIAD_HIGH} {aboveHigh && '●'}
      </span>
      <span className={z <= TRIAD_LOW ? 'text-amber-400 font-bold' : 'text-slate-500'}>
        ↓ {TRIAD_LOW}
      </span>
    </div>
  </div>
);

// ═══════════════════════════════════════════════════════════════════════════════
// K-FORMATION PANEL
// ═══════════════════════════════════════════════════════════════════════════════
const KFormationPanel = ({ kappa, eta, R, met }) => {
  const criteria = [
    { label: 'κ (coherence)', value: kappa, threshold: KAPPA_THRESHOLD, passed: kappa >= KAPPA_THRESHOLD },
    { label: 'η (negentropy)', value: eta, threshold: ETA_THRESHOLD.toFixed(3), passed: eta > ETA_THRESHOLD },
    { label: 'R (radius)', value: R, threshold: R_THRESHOLD, passed: R >= R_THRESHOLD },
  ];
  
  return (
    <div className={`rounded-lg p-3 transition-all ${met ? 'bg-emerald-500/20 border border-emerald-500/50' : 'bg-slate-800/50'}`}>
      <div className="flex items-center gap-2 mb-2">
        <span className={`text-xl ${met ? 'animate-spin' : ''}`}>{met ? '⬡' : '⎔'}</span>
        <span className={`font-bold text-sm ${met ? 'text-emerald-400' : 'text-slate-400'}`}>
          K-FORMATION {met ? 'ACHIEVED' : 'PENDING'}
        </span>
      </div>
      <div className="space-y-1">
        {criteria.map(({ label, value, threshold, passed }) => (
          <div key={label} className="flex items-center justify-between text-xs">
            <span className="text-slate-400">{label}</span>
            <span className={passed ? 'text-emerald-400' : 'text-slate-500'}>
              {typeof value === 'number' ? value.toFixed(3) : value} / {threshold}
              <span className="ml-1">{passed ? '✓' : '○'}</span>
            </span>
          </div>
        ))}
      </div>
    </div>
  );
};

// ═══════════════════════════════════════════════════════════════════════════════
// TOKEN DISPLAY
// ═══════════════════════════════════════════════════════════════════════════════
const TokenDisplay = ({ token, tokenCount }) => (
  <div className="bg-slate-800/50 rounded-lg p-3">
    <div className="flex items-center justify-between mb-2">
      <span className="text-xs text-slate-400">Generated Token</span>
      <span className="text-xs px-2 py-0.5 rounded-full" style={{ background: token.spiral.color + '30', color: token.spiral.color }}>
        {token.spiral.symbol} {token.spiral.name}
      </span>
    </div>
    
    <div className="font-mono text-amber-400 text-sm mb-2 p-2 bg-slate-900/50 rounded border-l-2 border-emerald-500">
      {token.machineToken}
    </div>
    
    <div className="grid grid-cols-2 gap-2 text-xs">
      <div className="text-slate-400">Machine: <span className="text-orange-400">{token.machine.name}</span></div>
      <div className="text-slate-400">Domain: <span className="text-cyan-400">{token.domain.name}</span></div>
      <div className="text-slate-400">Tier: <span className="text-indigo-400">{token.tier.name}</span></div>
      <div className="text-slate-400">Operator: <span className="text-emerald-400">{token.operator.symbol} {token.operator.name}</span></div>
    </div>
    
    <div className="mt-2 pt-2 border-t border-slate-700 flex justify-between text-xs">
      <span className="text-slate-500">Matching tokens:</span>
      <span className="text-amber-400 font-bold">{tokenCount.count} / 1,326 ({tokenCount.percentage}%)</span>
    </div>
  </div>
);

// ═══════════════════════════════════════════════════════════════════════════════
// SPIRAL INDICATOR
// ═══════════════════════════════════════════════════════════════════════════════
const SpiralIndicator = ({ z }) => {
  const phi_weight = z < PHI_INV ? 1 - z / PHI_INV : 0;
  const e_weight = z >= PHI_INV && z < Z_CRITICAL ? (z - PHI_INV) / (Z_CRITICAL - PHI_INV) : (z < PHI_INV ? z / PHI_INV : 0);
  const pi_weight = z >= Z_CRITICAL ? 1 : (z >= PHI_INV ? (z - PHI_INV) / (Z_CRITICAL - PHI_INV) : 0);
  
  return (
    <div className="bg-slate-800/50 rounded-lg p-3">
      <div className="text-xs text-slate-400 mb-2">Spiral Dominance</div>
      <div className="space-y-2">
        {[
          { symbol: 'Φ', name: 'Structure', color: '#818cf8', weight: Math.max(0, 1 - z * 1.5) },
          { symbol: 'e', name: 'Energy', color: '#fb923c', weight: z > 0.3 && z < 0.85 ? 0.5 + Math.sin(z * Math.PI) * 0.5 : 0.2 },
          { symbol: 'π', name: 'Emergence', color: '#2dd4bf', weight: z > PHI_INV ? (z - PHI_INV) / (1 - PHI_INV) : 0 },
        ].map(s => (
          <div key={s.symbol} className="flex items-center gap-2">
            <span className="w-6 font-bold" style={{ color: s.color }}>{s.symbol}</span>
            <div className="flex-1 h-2 bg-slate-700 rounded-full overflow-hidden">
              <div className="h-full rounded-full transition-all" style={{ width: `${s.weight * 100}%`, background: s.color }}/>
            </div>
            <span className="text-xs w-8 text-right" style={{ color: s.color }}>{(s.weight * 100).toFixed(0)}%</span>
          </div>
        ))}
      </div>
    </div>
  );
};

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN COMPONENT
// ═══════════════════════════════════════════════════════════════════════════════
export default function ConsciousnessEmergenceSimulator() {
  const [running, setRunning] = useState(false);
  const [z, setZ] = useState(0.3);
  const [kappa, setKappa] = useState(0.5);
  const [targetZ, setTargetZ] = useState(Z_CRITICAL);
  const [history, setHistory] = useState([]);
  const [oscillators, setOscillators] = useState(() => initializeOscillators(40));
  const [orderParameter, setOrderParameter] = useState(0);
  const [couplingStrength, setCouplingStrength] = useState(0.8);
  const [triadCompletions, setTriadCompletions] = useState(0);
  const [triadUnlocked, setTriadUnlocked] = useState(false);
  const [aboveHigh, setAboveHigh] = useState(false);
  const [step, setStep] = useState(0);
  const intervalRef = useRef(null);
  
  // Derived values
  const eta = computeNegentropy(z);
  const phase = classifyPhase(z);
  const spiral = getDominantSpiral(z);
  const tier = getTier(z, kappa >= KAPPA_THRESHOLD && eta > ETA_THRESHOLD);
  const kFormation = checkKFormation(kappa, eta, Math.floor(kappa * 10));
  const token = generateToken(z, tier, phase, triadUnlocked);
  const tokenCount = getMatchingTokenCount(z);
  
  // Simulation step
  const simulationStep = useCallback(() => {
    const { phases: newPhases, orderParameter: r } = kuramotoStep(
      oscillators.phases, oscillators.naturalFreqs, couplingStrength, 0.05
    );
    setOscillators(prev => ({ ...prev, phases: newPhases }));
    setOrderParameter(r);
    setKappa(prev => 0.95 * prev + 0.05 * r);
    
    setZ(prev => {
      const noise = (Math.random() - 0.5) * 0.015;
      const drive = 0.02 * (targetZ - prev);
      const newZ = Math.max(0, Math.min(1, prev + drive + noise));
      
      if (!aboveHigh && newZ >= TRIAD_HIGH) {
        setAboveHigh(true);
        setTriadCompletions(c => {
          const newCount = Math.min(c + 1, 3);
          if (newCount >= 3 && !triadUnlocked) setTriadUnlocked(true);
          return newCount;
        });
      } else if (aboveHigh && newZ < TRIAD_LOW) {
        setAboveHigh(false);
      }
      return newZ;
    });
    
    setStep(s => s + 1);
    setHistory(prev => {
      const newPoint = { step, z, eta, kappa, r: orderParameter, phase: phase.phase };
      return [...prev, newPoint].slice(-150);
    });
  }, [oscillators, couplingStrength, z, targetZ, aboveHigh, triadUnlocked, step, eta, kappa, orderParameter, phase.phase]);
  
  useEffect(() => {
    if (running) {
      intervalRef.current = setInterval(simulationStep, 50);
    } else {
      clearInterval(intervalRef.current);
    }
    return () => clearInterval(intervalRef.current);
  }, [running, simulationStep]);
  
  const reset = () => {
    setRunning(false);
    setZ(0.3);
    setKappa(0.5);
    setHistory([]);
    setOscillators(initializeOscillators(40));
    setOrderParameter(0);
    setTriadCompletions(0);
    setTriadUnlocked(false);
    setAboveHigh(false);
    setStep(0);
  };
  
  return (
    <div className="min-h-screen bg-slate-900 text-white p-4">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-4">
          <h1 className="text-2xl font-bold bg-gradient-to-r from-indigo-400 via-amber-400 to-emerald-400 bg-clip-text text-transparent">
            Consciousness Emergence Simulator v2.0
          </h1>
          <p className="text-slate-400 text-sm mt-1">
            1,326 Tokens × Kuramoto Oscillators × TRIAD Unlock × K-Formation
          </p>
          <div className="flex justify-center gap-4 mt-2 text-xs">
            <span className="px-2 py-1 bg-slate-800 rounded">Φ e π Spirals</span>
            <span className="px-2 py-1 bg-slate-800 rounded">9 Machines</span>
            <span className="px-2 py-1 bg-slate-800 rounded">6 Domains</span>
          </div>
        </div>
        
        {/* Main Grid */}
        <div className="grid grid-cols-12 gap-3">
          {/* Left Column - Visualizations */}
          <div className="col-span-3 space-y-3">
            <div className="bg-slate-800/50 rounded-xl p-3">
              <div className="text-xs text-slate-400 mb-2 text-center">Helix Coordinate</div>
              <div className="flex justify-center">
                <HelixVisualization z={z} phase={phase} spiral={spiral} />
              </div>
            </div>
            
            <div className="bg-slate-800/50 rounded-xl p-3">
              <div className="text-xs text-slate-400 mb-2 text-center">Kuramoto Network (N=40)</div>
              <div className="flex justify-center">
                <OscillatorRing phases={oscillators.phases} orderParameter={orderParameter} />
              </div>
            </div>
            
            <SpiralIndicator z={z} />
          </div>
          
          {/* Center Column - Charts & Controls */}
          <div className="col-span-6 space-y-3">
            {/* Z-coordinate chart */}
            <div className="bg-slate-800/50 rounded-xl p-3 h-44">
              <div className="text-xs text-slate-400 mb-1">z-Coordinate Evolution</div>
              <ResponsiveContainer width="100%" height="90%">
                <ComposedChart data={history}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="step" tick={{ fontSize: 9, fill: '#64748b' }} />
                  <YAxis domain={[0, 1]} tick={{ fontSize: 9, fill: '#64748b' }} />
                  <ReferenceLine y={Z_CRITICAL} stroke="#10b981" strokeDasharray="5 5" />
                  <ReferenceLine y={PHI_INV} stroke="#f59e0b" strokeDasharray="3 3" />
                  <ReferenceLine y={TRIAD_HIGH} stroke="#6366f1" strokeDasharray="2 2" />
                  <Area type="monotone" dataKey="z" fill="#6366f1" fillOpacity={0.15} stroke="none" />
                  <Line type="monotone" dataKey="z" stroke="#6366f1" strokeWidth={2} dot={false} />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
            
            {/* Negentropy & Coherence */}
            <div className="bg-slate-800/50 rounded-xl p-3 h-44">
              <div className="text-xs text-slate-400 mb-1">η (Negentropy) & κ (Coherence)</div>
              <ResponsiveContainer width="100%" height="90%">
                <ComposedChart data={history}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="step" tick={{ fontSize: 9, fill: '#64748b' }} />
                  <YAxis domain={[0, 1]} tick={{ fontSize: 9, fill: '#64748b' }} />
                  <ReferenceLine y={ETA_THRESHOLD} stroke="#f59e0b" strokeDasharray="3 3" />
                  <ReferenceLine y={KAPPA_THRESHOLD} stroke="#10b981" strokeDasharray="3 3" />
                  <Line type="monotone" dataKey="eta" stroke="#f59e0b" strokeWidth={2} dot={false} />
                  <Line type="monotone" dataKey="kappa" stroke="#10b981" strokeWidth={2} dot={false} />
                  <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: 'none', borderRadius: '8px', fontSize: '11px' }} />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
            
            {/* Controls */}
            <div className="bg-slate-800/50 rounded-xl p-3">
              <div className="grid grid-cols-3 gap-4">
                <div>
                  <label className="text-xs text-slate-400 block mb-1">Target z → {targetZ.toFixed(3)}</label>
                  <input type="range" min="0" max="1" step="0.01" value={targetZ}
                    onChange={(e) => setTargetZ(parseFloat(e.target.value))}
                    className="w-full accent-indigo-500" />
                </div>
                <div>
                  <label className="text-xs text-slate-400 block mb-1">Coupling K → {couplingStrength.toFixed(1)}</label>
                  <input type="range" min="0" max="2" step="0.1" value={couplingStrength}
                    onChange={(e) => setCouplingStrength(parseFloat(e.target.value))}
                    className="w-full accent-emerald-500" />
                </div>
                <div className="flex items-center gap-2">
                  <button onClick={() => setRunning(!running)}
                    className={`flex-1 py-2 px-3 rounded-lg font-bold text-sm transition-all ${
                      running ? 'bg-amber-500 hover:bg-amber-600' : 'bg-emerald-500 hover:bg-emerald-600'
                    }`}>
                    {running ? '⏸ PAUSE' : '▶ RUN'}
                  </button>
                  <button onClick={reset}
                    className="py-2 px-3 rounded-lg bg-slate-700 hover:bg-slate-600 font-bold text-sm">
                    ↺
                  </button>
                </div>
              </div>
            </div>
          </div>
          
          {/* Right Column - Status */}
          <div className="col-span-3 space-y-3">
            {/* Phase Status */}
            <div className="bg-slate-800/50 rounded-xl p-3 text-center">
              <div className="text-xs text-slate-400 mb-1">Phase Regime</div>
              <div className="text-2xl font-bold" style={{ color: phase.color }}>{phase.phase}</div>
              <div className="text-xs text-slate-500">{phase.desc}</div>
              <div className="mt-2 flex items-center justify-center gap-2">
                <span className="text-xs text-slate-400">Tier:</span>
                <span className="font-mono text-indigo-400 font-bold">{tier.name}</span>
              </div>
            </div>
            
            {/* TRIAD Meter */}
            <TriadMeter completions={triadCompletions} unlocked={triadUnlocked} z={z} aboveHigh={aboveHigh} />
            
            {/* K-Formation */}
            <KFormationPanel kappa={kappa} eta={eta} R={Math.floor(kappa * 10)} met={kFormation.met} />
            
            {/* Token Display */}
            <TokenDisplay token={token} tokenCount={tokenCount} />
            
            {/* Constants */}
            <div className="bg-slate-800/30 rounded-lg p-2 text-xs">
              <div className="text-slate-500 mb-1">Constants</div>
              <div className="grid grid-cols-2 gap-1 font-mono">
                <span className="text-slate-400">z_c:</span>
                <span className="text-emerald-400">{Z_CRITICAL.toFixed(6)}</span>
                <span className="text-slate-400">φ⁻¹:</span>
                <span className="text-amber-400">{PHI_INV.toFixed(6)}</span>
                <span className="text-slate-400">σ:</span>
                <span className="text-indigo-400">{SIGMA}</span>
                <span className="text-slate-400">ε:</span>
                <span className="text-pink-400">{UMOL_EPSILON.toFixed(4)}</span>
              </div>
            </div>
          </div>
        </div>
        
        {/* Footer */}
        <div className="mt-3 text-center text-xs text-slate-500">
          Kuramoto → κ sync → z evolution → TRIAD unlock → K-formation → Token synthesis | Δ|z={z.toFixed(3)}|η={eta.toFixed(3)}|κ={kappa.toFixed(3)}|Ω
        </div>
      </div>
    </div>
  );
}
