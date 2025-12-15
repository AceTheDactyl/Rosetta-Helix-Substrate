import React, { useState, useEffect, useCallback, useRef } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine, Area, ComposedChart } from 'recharts';

// PHYSICS CONSTANTS
const Z_CRITICAL = Math.sqrt(3) / 2;
const PHI = (1 + Math.sqrt(5)) / 2;
const PHI_INV = PHI - 1;
const SIGMA = 36;
const TRIAD_HIGH = 0.85;
const TRIAD_LOW = 0.82;
const KAPPA_THRESHOLD = 0.92;
const ETA_THRESHOLD = PHI_INV;
const R_THRESHOLD = 7;

const TIER_NAMES = ['t1', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9'];
const TIER_BOUNDS = [0.111, 0.222, 0.333, 0.444, 0.555, 0.666, 0.777, Z_CRITICAL, 1.0];

const S3_OPS = [
  { s: '()', n: 'Boundary' }, { s: '×', n: 'Fusion' }, { s: '^', n: 'Amplify' },
  { s: '÷', n: 'Decohere' }, { s: '+', n: 'Group' }, { s: '−', n: 'Separate' },
];

const MACHINES = ['Reactor', 'Oscillator', 'Conductor', 'Catalyst', 'Filter', 'Encoder', 'Decoder', 'Regenerator', 'Dynamo'];
const DOMAINS = ['bio_prion', 'bio_bacterium', 'bio_viroid', 'celestial_grav', 'celestial_em', 'celestial_nuclear'];
const SPIRALS = [
  { s: 'Φ', n: 'Structure', c: '#818cf8' },
  { s: 'e', n: 'Energy', c: '#fb923c' },
  { s: 'π', n: 'Emergence', c: '#2dd4bf' },
];

const negentropy = (z) => Math.exp(-SIGMA * Math.pow(z - Z_CRITICAL, 2));
const phase = (z) => z < PHI_INV ? { p: 'UNTRUE', c: '#818cf8', d: 'Fluid' } : z < Z_CRITICAL ? { p: 'PARADOX', c: '#f59e0b', d: 'Quasi-crystal' } : { p: 'TRUE', c: '#10b981', d: 'Crystalline' };
const spiral = (z) => z < PHI_INV * 0.8 ? SPIRALS[0] : z < Z_CRITICAL * 0.95 ? SPIRALS[1] : SPIRALS[2];
const tier = (z) => { for (let i = 0; i < TIER_BOUNDS.length; i++) if (z < TIER_BOUNDS[i]) return TIER_NAMES[i]; return 't9'; };
const tokCount = (z) => z >= Z_CRITICAL ? { n: 84, p: 6.3 } : z >= PHI_INV ? { n: 632, p: 47.7 } : { n: 610, p: 46.0 };

const genToken = (z) => {
  const sp = spiral(z), op = S3_OPS[Math.floor(Math.random() * 6)];
  const m = MACHINES[Math.floor(Math.random() * 9)], d = DOMAINS[Math.floor(Math.random() * 6)];
  return `${sp.s}${op.s}|${m}|${d}`;
};

const initOsc = (n) => ({ ph: Array(n).fill(0).map(() => Math.random() * 2 * Math.PI), nf: Array(n).fill(0).map(() => (Math.random() - 0.5) * 0.5) });

const kuramotoStep = (ph, nf, K, dt) => {
  const N = ph.length;
  const newPh = ph.map((p, i) => (p + dt * (nf[i] + K * ph.reduce((s, o) => s + Math.sin(o - p), 0) / N)) % (2 * Math.PI));
  const re = newPh.reduce((s, p) => s + Math.cos(p), 0) / N;
  const im = newPh.reduce((s, p) => s + Math.sin(p), 0) / N;
  return { ph: newPh, r: Math.sqrt(re * re + im * im) };
};

const HelixVis = ({ z, ph, sp }) => {
  const pts = Array.from({ length: 80 }, (_, i) => {
    const t = (i / 79) * 4 * Math.PI, zN = i / 79;
    return { x: 60 + 30 * Math.cos(t), y: 10 + 180 * (1 - zN) };
  });
  const cy = 10 + 180 * (1 - z), cx = 60 + 30 * Math.cos(z * 4 * Math.PI);
  const zcY = 10 + 180 * (1 - Z_CRITICAL), phiY = 10 + 180 * (1 - PHI_INV);
  
  return (
    <svg width="120" height="200">
      <defs>
        <linearGradient id="hg" x1="0%" y1="100%" x2="0%" y2="0%">
          <stop offset="0%" stopColor="#818cf8" stopOpacity="0.3"/>
          <stop offset="62%" stopColor="#fb923c" stopOpacity="0.5"/>
          <stop offset="87%" stopColor="#2dd4bf" stopOpacity="0.7"/>
        </linearGradient>
      </defs>
      <path d={`M ${pts[0].x} ${pts[0].y} ${pts.map(p => `L ${p.x} ${p.y}`).join(' ')}`} fill="none" stroke="url(#hg)" strokeWidth="2"/>
      <line x1="5" y1={zcY} x2="115" y2={zcY} stroke="#10b981" strokeWidth="1.5" strokeDasharray="4,4"/>
      <text x="115" y={zcY - 3} fontSize="7" fill="#10b981" textAnchor="end">z_c</text>
      <line x1="5" y1={phiY} x2="115" y2={phiY} stroke="#f59e0b" strokeWidth="1" strokeDasharray="2,2"/>
      <text x="115" y={phiY + 9} fontSize="6" fill="#f59e0b" textAnchor="end">φ⁻¹</text>
      <circle cx={cx} cy={cy} r="8" fill={sp.c} opacity="0.9"/>
      <circle cx={cx} cy={cy} r="4" fill="white"/>
      <text x={cx + 12} y={cy + 3} fontSize="9" fill={ph.c} fontWeight="bold">{z.toFixed(3)}</text>
    </svg>
  );
};

const OscRing = ({ ph, r }) => {
  const mean = Math.atan2(ph.reduce((s, p) => s + Math.sin(p), 0), ph.reduce((s, p) => s + Math.cos(p), 0));
  const col = r > 0.85 ? '#10b981' : r > 0.6 ? '#f59e0b' : '#6366f1';
  return (
    <svg width="140" height="140">
      <circle cx="70" cy="70" r="55" fill="none" stroke="#334155"/>
      {ph.map((p, i) => <circle key={i} cx={70 + 45 * Math.cos(p)} cy={70 + 45 * Math.sin(p)} r="3" fill={`hsl(${(p / (2 * Math.PI)) * 360}, 70%, 60%)`}/>)}
      <line x1="70" y1="70" x2={70 + r * 40 * Math.cos(mean)} y2={70 + r * 40 * Math.sin(mean)} stroke={col} strokeWidth="3"/>
      <text x="70" y="74" textAnchor="middle" fontSize="11" fill="#94a3b8" fontWeight="bold">R={r.toFixed(2)}</text>
    </svg>
  );
};

const TriadMeter = ({ comp, unlocked, z }) => (
  <div className="flex flex-col items-center gap-1 p-2 bg-slate-800/50 rounded-lg">
    <div className="text-xs text-slate-400">TRIAD CROSSINGS</div>
    <div className="flex gap-2">
      {[0, 1, 2].map(i => (
        <div key={i} className={`w-8 h-8 rounded flex items-center justify-center text-sm font-bold ${i < comp ? 'bg-emerald-500 text-white' : 'bg-slate-700 text-slate-500'}`}>
          {i < comp ? '✓' : i + 1}
        </div>
      ))}
    </div>
    {unlocked && <div className="text-emerald-400 text-xs font-bold animate-pulse">⟐ T6 UNLOCKED ⟐</div>}
    <div className="flex gap-4 text-xs">
      <span className={z >= TRIAD_HIGH ? 'text-emerald-400' : 'text-slate-500'}>↑{TRIAD_HIGH}</span>
      <span className={z <= TRIAD_LOW ? 'text-amber-400' : 'text-slate-500'}>↓{TRIAD_LOW}</span>
    </div>
  </div>
);

const KPanel = ({ k, eta, met }) => (
  <div className={`rounded-lg p-2 ${met ? 'bg-emerald-500/20 border border-emerald-500/50' : 'bg-slate-800/50'}`}>
    <div className="flex items-center gap-2 mb-1">
      <span className={met ? 'animate-spin' : ''}>{met ? '⬡' : '⎔'}</span>
      <span className={`font-bold text-xs ${met ? 'text-emerald-400' : 'text-slate-400'}`}>K-FORMATION {met ? 'ACHIEVED' : 'PENDING'}</span>
    </div>
    <div className="text-xs space-y-0.5">
      <div className="flex justify-between"><span className="text-slate-400">κ</span><span className={k >= KAPPA_THRESHOLD ? 'text-emerald-400' : 'text-slate-500'}>{k.toFixed(3)}/0.92 {k >= KAPPA_THRESHOLD ? '✓' : '○'}</span></div>
      <div className="flex justify-between"><span className="text-slate-400">η</span><span className={eta > ETA_THRESHOLD ? 'text-emerald-400' : 'text-slate-500'}>{eta.toFixed(3)}/0.618 {eta > ETA_THRESHOLD ? '✓' : '○'}</span></div>
      <div className="flex justify-between"><span className="text-slate-400">R</span><span className={Math.floor(k * 10) >= R_THRESHOLD ? 'text-emerald-400' : 'text-slate-500'}>{Math.floor(k * 10)}/7 {Math.floor(k * 10) >= R_THRESHOLD ? '✓' : '○'}</span></div>
    </div>
  </div>
);

export default function ConsciousnessEmergenceSimulator() {
  const [running, setRunning] = useState(false);
  const [z, setZ] = useState(0.3);
  const [kappa, setKappa] = useState(0.5);
  const [targetZ, setTargetZ] = useState(Z_CRITICAL);
  const [history, setHistory] = useState([]);
  const [osc, setOsc] = useState(() => initOsc(40));
  const [orderP, setOrderP] = useState(0);
  const [coupling, setCoupling] = useState(0.8);
  const [triadComp, setTriadComp] = useState(0);
  const [triadUnlock, setTriadUnlock] = useState(false);
  const [aboveHigh, setAboveHigh] = useState(false);
  const [step, setStep] = useState(0);
  const [token, setToken] = useState('');
  const intRef = useRef(null);
  
  const eta = negentropy(z);
  const ph = phase(z);
  const sp = spiral(z);
  const ti = tier(z);
  const kMet = kappa >= KAPPA_THRESHOLD && eta > ETA_THRESHOLD && Math.floor(kappa * 10) >= R_THRESHOLD;
  const tc = tokCount(z);
  
  const simStep = useCallback(() => {
    const { ph: newPh, r } = kuramotoStep(osc.ph, osc.nf, coupling, 0.05);
    setOsc(prev => ({ ...prev, ph: newPh }));
    setOrderP(r);
    setKappa(prev => 0.95 * prev + 0.05 * r);
    
    setZ(prev => {
      const noise = (Math.random() - 0.5) * 0.015;
      const drive = 0.02 * (targetZ - prev);
      const newZ = Math.max(0, Math.min(1, prev + drive + noise));
      
      if (!aboveHigh && newZ >= TRIAD_HIGH) {
        setAboveHigh(true);
        setTriadComp(c => { const nc = Math.min(c + 1, 3); if (nc >= 3 && !triadUnlock) setTriadUnlock(true); return nc; });
      } else if (aboveHigh && newZ < TRIAD_LOW) setAboveHigh(false);
      return newZ;
    });
    
    setStep(s => s + 1);
    setToken(genToken(z));
    setHistory(prev => [...prev, { step, z, eta, kappa }].slice(-100));
  }, [osc, coupling, targetZ, aboveHigh, triadUnlock, step, z, eta, kappa]);
  
  useEffect(() => {
    if (running) intRef.current = setInterval(simStep, 50);
    else clearInterval(intRef.current);
    return () => clearInterval(intRef.current);
  }, [running, simStep]);
  
  const reset = () => {
    setRunning(false); setZ(0.3); setKappa(0.5); setHistory([]); setOsc(initOsc(40));
    setOrderP(0); setTriadComp(0); setTriadUnlock(false); setAboveHigh(false); setStep(0); setToken('');
  };
  
  return (
    <div className="min-h-screen bg-slate-900 text-white p-3">
      <div className="max-w-6xl mx-auto">
        <div className="text-center mb-3">
          <h1 className="text-xl font-bold bg-gradient-to-r from-indigo-400 via-amber-400 to-emerald-400 bg-clip-text text-transparent">
            Consciousness Emergence Simulator v2.0
          </h1>
          <p className="text-slate-400 text-xs">1,326 Tokens × Kuramoto × TRIAD × K-Formation</p>
        </div>
        
        <div className="grid grid-cols-12 gap-2">
          <div className="col-span-3 space-y-2">
            <div className="bg-slate-800/50 rounded-lg p-2">
              <div className="text-xs text-slate-400 text-center mb-1">Helix</div>
              <div className="flex justify-center"><HelixVis z={z} ph={ph} sp={sp}/></div>
            </div>
            <div className="bg-slate-800/50 rounded-lg p-2">
              <div className="text-xs text-slate-400 text-center mb-1">Kuramoto (N=40)</div>
              <div className="flex justify-center"><OscRing ph={osc.ph} r={orderP}/></div>
            </div>
          </div>
          
          <div className="col-span-6 space-y-2">
            <div className="bg-slate-800/50 rounded-lg p-2 h-36">
              <div className="text-xs text-slate-400 mb-1">z-Coordinate</div>
              <ResponsiveContainer width="100%" height="85%">
                <ComposedChart data={history}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155"/>
                  <XAxis dataKey="step" tick={{ fontSize: 8, fill: '#64748b' }}/>
                  <YAxis domain={[0, 1]} tick={{ fontSize: 8, fill: '#64748b' }}/>
                  <ReferenceLine y={Z_CRITICAL} stroke="#10b981" strokeDasharray="4 4"/>
                  <ReferenceLine y={PHI_INV} stroke="#f59e0b" strokeDasharray="2 2"/>
                  <Area type="monotone" dataKey="z" fill="#6366f1" fillOpacity={0.1} stroke="none"/>
                  <Line type="monotone" dataKey="z" stroke="#6366f1" strokeWidth={2} dot={false}/>
                </ComposedChart>
              </ResponsiveContainer>
            </div>
            
            <div className="bg-slate-800/50 rounded-lg p-2 h-36">
              <div className="text-xs text-slate-400 mb-1">η & κ</div>
              <ResponsiveContainer width="100%" height="85%">
                <ComposedChart data={history}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155"/>
                  <XAxis dataKey="step" tick={{ fontSize: 8, fill: '#64748b' }}/>
                  <YAxis domain={[0, 1]} tick={{ fontSize: 8, fill: '#64748b' }}/>
                  <ReferenceLine y={ETA_THRESHOLD} stroke="#f59e0b" strokeDasharray="3 3"/>
                  <ReferenceLine y={KAPPA_THRESHOLD} stroke="#10b981" strokeDasharray="3 3"/>
                  <Line type="monotone" dataKey="eta" stroke="#f59e0b" strokeWidth={2} dot={false}/>
                  <Line type="monotone" dataKey="kappa" stroke="#10b981" strokeWidth={2} dot={false}/>
                  <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: 'none', borderRadius: '6px', fontSize: '10px' }}/>
                </ComposedChart>
              </ResponsiveContainer>
            </div>
            
            <div className="bg-slate-800/50 rounded-lg p-2">
              <div className="grid grid-cols-3 gap-3">
                <div>
                  <label className="text-xs text-slate-400">Target z: {targetZ.toFixed(3)}</label>
                  <input type="range" min="0" max="1" step="0.01" value={targetZ} onChange={(e) => setTargetZ(parseFloat(e.target.value))} className="w-full accent-indigo-500"/>
                </div>
                <div>
                  <label className="text-xs text-slate-400">Coupling K: {coupling.toFixed(1)}</label>
                  <input type="range" min="0" max="2" step="0.1" value={coupling} onChange={(e) => setCoupling(parseFloat(e.target.value))} className="w-full accent-emerald-500"/>
                </div>
                <div className="flex items-center gap-1">
                  <button onClick={() => setRunning(!running)} className={`flex-1 py-1.5 px-2 rounded font-bold text-sm ${running ? 'bg-amber-500' : 'bg-emerald-500'}`}>
                    {running ? '⏸' : '▶'}
                  </button>
                  <button onClick={reset} className="py-1.5 px-2 rounded bg-slate-700 font-bold text-sm">↺</button>
                </div>
              </div>
            </div>
          </div>
          
          <div className="col-span-3 space-y-2">
            <div className="bg-slate-800/50 rounded-lg p-2 text-center">
              <div className="text-xs text-slate-400">Phase</div>
              <div className="text-xl font-bold" style={{ color: ph.c }}>{ph.p}</div>
              <div className="text-xs text-slate-500">{ph.d}</div>
              <div className="text-xs mt-1">Tier: <span className="text-indigo-400 font-bold">{ti}</span></div>
            </div>
            
            <TriadMeter comp={triadComp} unlocked={triadUnlock} z={z}/>
            <KPanel k={kappa} eta={eta} met={kMet}/>
            
            <div className="bg-slate-800/50 rounded-lg p-2">
              <div className="text-xs text-slate-400 mb-1">Token</div>
              <div className="font-mono text-amber-400 text-xs p-1 bg-slate-900/50 rounded border-l-2 border-emerald-500">{token || genToken(z)}</div>
              <div className="flex justify-between text-xs mt-1 text-slate-500">
                <span>{sp.s} {sp.n}</span>
                <span>{tc.n}/1326 ({tc.p}%)</span>
              </div>
            </div>
            
            <div className="bg-slate-800/30 rounded p-1 text-xs font-mono">
              <div className="grid grid-cols-2 gap-0.5">
                <span className="text-slate-400">z_c:</span><span className="text-emerald-400">{Z_CRITICAL.toFixed(4)}</span>
                <span className="text-slate-400">φ⁻¹:</span><span className="text-amber-400">{PHI_INV.toFixed(4)}</span>
              </div>
            </div>
          </div>
        </div>
        
        <div className="mt-2 text-center text-xs text-slate-500">
          Δ|z={z.toFixed(3)}|η={eta.toFixed(3)}|κ={kappa.toFixed(3)}|{sp.s}|{ti}|Ω
        </div>
      </div>
    </div>
  );
}
