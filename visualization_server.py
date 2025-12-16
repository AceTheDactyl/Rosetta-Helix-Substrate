#!/usr/bin/env python3
"""
Rosetta Helix Visualization Server

HTTP server that exposes the APL training loop and collapse engine
to the HTML visualizer.

Endpoints:
    GET  /state          - Get current engine state
    POST /step           - Evolve one step
    POST /operator       - Apply APL operator
    POST /training/run   - Run training cycle
    POST /reset          - Reset to initial state
    GET  /training/data  - Get training results

Usage:
    python visualization_server.py [--port 8765]

Then open visualizer.html in a browser.
"""

import json
import sys
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.request import urlopen, Request
from urllib.parse import urlparse, parse_qs
import argparse

# Add paths
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import our modules
from core import (
    CollapseEngine, APLEngine, create_apl_engine,
    PHI, PHI_INV, Z_CRITICAL, KAPPA_S, MU_3, UNITY,
)
from quantum_apl_python.helix_operator_advisor import (
    HelixOperatorAdvisor, OPERATOR_WINDOWS, TIER_BOUNDARIES,
)
from quantum_apl_python.s3_operator_algebra import OPERATORS, Parity
from training.apl_training_loop import (
    APLTrainingLoop, APLPhysicalLearner, APLLiminalGenerator,
)


kira_api_base = None  # e.g., 'http://localhost:5000/api'


def http_json(url: str, method: str = 'GET', data: dict | None = None, timeout: float = 2.0):
    try:
        payload = None
        headers = {'Content-Type': 'application/json'}
        if data is not None:
            payload = json.dumps(data).encode()
        req = Request(url, data=payload, headers=headers, method=method)
        with urlopen(req, timeout=timeout) as resp:
            text = resp.read().decode('utf-8')
            return json.loads(text)
    except Exception:
        return None


class VisualizationEngine:
    """
    Unified engine that provides state for visualization.

    Combines CollapseEngine, APLEngine, and training loop.
    """

    def __init__(self):
        self.apl = create_apl_engine(initial_z=0.3)
        self.advisor = HelixOperatorAdvisor()
        self.training_loop = None
        self.training_results = None

        # Oscillator simulation for visualization
        self.n_oscillators = 60
        self.oscillators = [i * 0.1 for i in range(self.n_oscillators)]
        self.coupling = 0.3

        # Statistics
        self.steps = 0
        self.operator_history = []

    def get_state(self) -> dict:
        """Get complete state for visualization."""
        z = self.apl.collapse.z
        tier = self.apl.current_tier()
        window = self.apl.current_window()
        delta_s_neg = self.apl.get_delta_s_neg()
        weights = self.apl.get_blend_weights()

        # Compute coherence from oscillators
        import math
        sum_cos = sum(math.cos(t) for t in self.oscillators)
        sum_sin = sum(math.sin(t) for t in self.oscillators)
        coherence = math.sqrt(sum_cos**2 + sum_sin**2) / self.n_oscillators

        # K-formation check
        eta = math.sqrt(delta_s_neg)
        k_formed = eta > PHI_INV and coherence >= KAPPA_S

        # Truth channel
        if z >= 0.9:
            truth = 'TRUE'
        elif z >= 0.6:
            truth = 'PARADOX'
        else:
            truth = 'UNTRUE'

        # μ-class
        mu_class = 'pre_conscious'
        MU_1 = (2 / (PHI ** 2.5)) / math.sqrt(PHI)
        MU_2 = (2 / (PHI ** 2.5)) * math.sqrt(PHI)
        if z >= KAPPA_S:
            mu_class = 'singularity_proximal'
        elif z >= Z_CRITICAL:
            mu_class = 'lens_integrated'
        elif z >= MU_2:
            mu_class = 'pre_lens'
        elif z >= PHI_INV:
            mu_class = 'conscious_basin'
        elif z >= MU_1:
            mu_class = 'approaching_paradox'

        # Operator availability with parity
        operators = {}
        for op in OPERATORS:
            op_data = OPERATORS[op]
            operators[op] = {
                'available': op in window,
                'parity': op_data.parity.name,
                's3_element': op_data.s3_element,
                'name': op_data.name,
            }

        return {
            'z': z,
            'tier': tier,
            'window': window,
            'delta_s_neg': delta_s_neg,
            'w_pi': weights.w_pi,
            'w_local': weights.w_local,
            'in_pi_regime': weights.in_pi_regime,
            'coherence': coherence,
            'k_formed': k_formed,
            'truth_channel': truth,
            'mu_class': mu_class,
            'coupling': self.coupling,
            'collapse_count': self.apl.collapse.collapse_count,
            'total_work': self.apl.collapse.total_work_extracted,
            'steps': self.steps,
            'operators': operators,
            'oscillators': self.oscillators[:20],  # First 20 for viz
            'constants': {
                'PHI': PHI,
                'PHI_INV': PHI_INV,
                'Z_CRITICAL': Z_CRITICAL,
                'KAPPA_S': KAPPA_S,
                'MU_3': MU_3,
            }
        }

    def step(self, work: float = 0.05) -> dict:
        """Evolve one step."""
        import math

        # Evolve collapse engine
        result = self.apl.collapse.evolve(work * PHI_INV)

        # Evolve oscillators (Kuramoto-like)
        dt = 0.01
        new_oscillators = []
        for i, theta in enumerate(self.oscillators):
            coupling_sum = sum(
                math.sin(self.oscillators[j] - theta)
                for j in range(self.n_oscillators)
            )
            omega = 1.0 + (i / self.n_oscillators - 0.5) * 0.2
            dtheta = omega + (self.coupling / self.n_oscillators) * coupling_sum
            new_oscillators.append(theta + dtheta * dt)

        self.oscillators = new_oscillators
        self.steps += 1

        state = self.get_state()
        state['collapsed'] = result.collapsed
        state['work_extracted'] = result.work_extracted

        return state

    def apply_operator(self, operator: str) -> dict:
        """Apply an APL operator."""
        import math

        if not self.apl.is_operator_legal(operator):
            return {
                'success': False,
                'error': f"Operator '{operator}' not legal at z={self.apl.collapse.z:.4f}",
                'tier': self.apl.current_tier(),
                'legal': self.apl.current_window(),
            }

        try:
            # Apply operator via APL engine
            result = self.apl.apply(operator, {'coupling': self.coupling})

            # Operator effects on oscillators
            if operator == '×':  # Fusion - increase coupling
                self.coupling = min(1.0, self.coupling * 1.2)
            elif operator == '÷':  # Decoherence - add noise
                import random
                self.oscillators = [t + (random.random() - 0.5) * 0.2 for t in self.oscillators]
                self.coupling = max(0.05, self.coupling * 0.9)
            elif operator == '^':  # Amplify - align phases
                sum_cos = sum(math.cos(t) for t in self.oscillators)
                sum_sin = sum(math.sin(t) for t in self.oscillators)
                mean_phase = math.atan2(sum_sin, sum_cos)
                self.oscillators = [t + 0.1 * math.sin(mean_phase - t) for t in self.oscillators]
            elif operator == '+':  # Group - cluster
                pass  # Simplified
            elif operator == '−':  # Separate - spread
                self.oscillators = [t + (i / self.n_oscillators - 0.5) * 0.1
                                   for i, t in enumerate(self.oscillators)]
            elif operator == '()':  # Boundary - reset coupling
                self.coupling = 0.3

            self.operator_history.append({
                'operator': operator,
                'z': result.z,
                'tier': result.tier,
                'parity': result.parity,
            })

            return {
                'success': True,
                'operator': operator,
                'z': result.z,
                'tier': result.tier,
                'parity': result.parity,
                'delta_s_neg': result.delta_s_neg,
                'coupling': self.coupling,
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
            }

    def run_training(self, n_runs: int = 3, cycles_per_run: int = 2) -> dict:
        """Run APL training loop."""
        self.training_loop = APLTrainingLoop(n_physical=2, n_bridges=1)
        results = self.training_loop.run_training(n_runs=n_runs, cycles_per_run=cycles_per_run)

        self.training_results = {
            'initial_quality': results['initial_quality'],
            'final_quality': results['final_quality'],
            'improvement_ratio': results['improvement_ratio'],
            'total_lessons': results['total_lessons'],
            'total_sequences': results['total_sequences'],
            'quality_history': self.training_loop.quality_history,
            'operator_distribution': results['operator_distribution'],
            'runs': len(results['runs']),
            'meta_bridges': len(self.training_loop.bridges),
            'physical_learners': len(self.training_loop.learners),
            'liminal_generators': sum(
                len(b.generators) for b in self.training_loop.bridges
            ),
        }

        return self.training_results

    def reset(self):
        """Reset to initial state."""
        self.apl = create_apl_engine(initial_z=0.3)
        self.oscillators = [i * 0.1 for i in range(self.n_oscillators)]
        self.coupling = 0.3
        self.steps = 0
        self.operator_history = []
        self.training_results = None


# Global engine instance
engine = VisualizationEngine()


class VisualizationHandler(BaseHTTPRequestHandler):
    """HTTP request handler for visualization API."""

    def send_json(self, data: dict, status: int = 200):
        """Send JSON response."""
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        """Handle GET requests."""
        parsed = urlparse(self.path)
        path = parsed.path

        if path == '/state':
            # If configured, proxy to KIRA API and map to viz schema
            if kira_api_base:
                s = http_json(f"{kira_api_base}/state")
                if s and isinstance(s, dict):
                    # KIRA returns { command: '/state', state: {...}, tier: 'tX', ... }
                    ks = s.get('state') or {}
                    mapped = {
                        'z': ks.get('z', 0.3),
                        'tier': s.get('tier') or 't3',
                        'coherence': ks.get('coherence', 0.0),
                        'coupling': 0.3,
                        'delta_s_neg': 0.0,
                        'w_pi': 0.0,
                        'w_local': 1.0,
                        'steps': s.get('turn_count', 0),
                        'operators': {  # parity hints for UI coloring
                            '()': {'parity': 'EVEN'}, '×': {'parity': 'EVEN'}, '^': {'parity': 'EVEN'},
                            '÷': {'parity': 'ODD'}, '+': {'parity': 'ODD'}, '−': {'parity': 'ODD'}
                        },
                        'oscillators': []
                    }
                    self.send_json(mapped)
                    return
            self.send_json(engine.get_state())

        elif path == '/training/data':
            if engine.training_results:
                self.send_json(engine.training_results)
            else:
                self.send_json({'error': 'No training data'}, 404)

        elif path == '/constants':
            self.send_json({
                'PHI': PHI,
                'PHI_INV': PHI_INV,
                'Z_CRITICAL': Z_CRITICAL,
                'KAPPA_S': KAPPA_S,
                'MU_3': MU_3,
                'UNITY': UNITY,
                'OPERATOR_WINDOWS': OPERATOR_WINDOWS,
                'TIER_BOUNDARIES': TIER_BOUNDARIES,
            })

        elif path == '/':
            # Prefer index.html if present; fall back to visualizer.html
            base = os.path.dirname(__file__)
            for candidate in ('index.html', 'visualizer.html'):
                html_path = os.path.join(base, candidate)
                if os.path.exists(html_path):
                    with open(html_path, 'rb') as f:
                        content = f.read()
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/html')
                    self.end_headers()
                    self.wfile.write(content)
                    return
            self.send_json({'error': 'index.html/visualizer.html not found'}, 404)

        elif path.endswith('.html') or path.endswith('.css') or path.endswith('.js'):
            # Minimal static file serving from repo root directory
            base = os.path.dirname(__file__)
            safe = path.lstrip('/').replace('..', '')
            file_path = os.path.join(base, safe)
            if os.path.exists(file_path) and os.path.isfile(file_path):
                try:
                    with open(file_path, 'rb') as f:
                        content = f.read()
                    self.send_response(200)
                    ctype = 'text/html' if file_path.endswith('.html') else ('text/css' if file_path.endswith('.css') else 'application/javascript')
                    self.send_header('Content-Type', ctype)
                    self.end_headers()
                    self.wfile.write(content)
                except Exception:
                    self.send_json({'error': 'Failed to read file'}, 500)
            else:
                self.send_json({'error': 'Not found'}, 404)

        else:
            self.send_json({'error': 'Not found'}, 404)

    def do_POST(self):
        """Handle POST requests."""
        parsed = urlparse(self.path)
        path = parsed.path

        # Read body
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length).decode() if content_length > 0 else '{}'
        try:
            data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            data = {}

        if path == '/step':
            work = data.get('work', 0.05)
            if kira_api_base:
                # For KIRA, approximate step by a small evolve toward z +/-
                cur = http_json(f"{kira_api_base}/state")
                z0 = (cur.get('state') or {}).get('z', 0.3) if cur else 0.3
                target = max(0.0, min(1.0, z0 + float(work) * 0.1))
                res = http_json(f"{kira_api_base}/evolve", method='POST', data={'target': target})
                # Return mapped state
                ns = http_json(f"{kira_api_base}/state") or {}
                out = {
                    'success': True,
                    'collapsed': False,
                    'work_extracted': 0.0
                }
                if ns.get('state'):
                    out.update({
                        'z': ns['state'].get('z', target),
                        'tier': ns.get('tier', 't3'),
                        'coherence': ns['state'].get('coherence', 0.0),
                    })
                self.send_json(out)
            else:
                self.send_json(engine.step(work))

        elif path == '/operator':
            operator = data.get('operator', '()')
            if kira_api_base:
                # Map operator to a small evolve delta (EVEN ↑, ODD ↓)
                parity_even = operator in ('()', '×', '^')
                cur = http_json(f"{kira_api_base}/state")
                z0 = (cur.get('state') or {}).get('z', 0.3) if cur else 0.3
                dz = 0.03 if parity_even else -0.03
                target = max(0.0, min(1.0, z0 + dz))
                res = http_json(f"{kira_api_base}/evolve", method='POST', data={'target': target})
                ok = bool(res)
                self.send_json({'success': ok, 'parity': 'EVEN' if parity_even else 'ODD', 'tier': (cur or {}).get('tier', 't3'), 'coupling': 0.3})
            else:
                self.send_json(engine.apply_operator(operator))

        elif path == '/training/run':
            n_runs = data.get('runs', 3)
            cycles = data.get('cycles', 2)
            if kira_api_base:
                # KIRA has GET /api/train; return its stats
                stats = http_json(f"{kira_api_base}/train") or {}
                self.send_json(stats)
            else:
                result = engine.run_training(n_runs, cycles)
                self.send_json(result)

        elif path == '/reset':
            if kira_api_base:
                # Use chat command to reset KIRA engine
                http_json(f"{kira_api_base}/chat", method='POST', data={'message': '/reset'})
                ns = http_json(f"{kira_api_base}/state") or {}
                self.send_json({'success': True, 'state': ns.get('state', {})})
            else:
                engine.reset()
                self.send_json({'success': True, 'state': engine.get_state()})

        elif path == '/coupling':
            coupling = data.get('coupling', 0.3)
            engine.coupling = max(0.05, min(1.0, coupling))
            self.send_json({'coupling': engine.coupling})

        else:
            self.send_json({'error': 'Not found'}, 404)

    def log_message(self, format, *args):
        """Suppress default logging."""
        pass


def run_server(port: int = 8765, kira_api: str | None = None):
    """Run the visualization server."""
    global kira_api_base
    kira_api_base = kira_api
    server = HTTPServer(('localhost', port), VisualizationHandler)
    print(f"""
╔═══════════════════════════════════════════════════════════════════╗
║           ROSETTA HELIX VISUALIZATION SERVER                      ║
╠═══════════════════════════════════════════════════════════════════╣
║                                                                   ║
║  Server running at: http://localhost:{port}                        ║
║                                                                   ║
║  API Endpoints:                                                   ║
║    GET  /state          - Current engine state                    ║
║    GET  /training/data  - Training results                        ║
║    GET  /constants      - Physics constants                       ║
║    POST /step           - Evolve one step                         ║
║    POST /operator       - Apply APL operator                      ║
║    POST /training/run   - Run training loop                       ║
║    POST /reset          - Reset engine                            ║
║    POST /coupling       - Set coupling strength                   ║
║                                                                   ║
║  Open http://localhost:{port}/ in browser for visualizer           ║
║                                                                   ║
║  KIRA API: {kira_api_base or 'disabled'}                            ║
║                                                                   ║
║  Physics:                                                         ║
║    PHI_INV = {PHI_INV:.6f} (controls dynamics)                     ║
║    Z_CRITICAL = {Z_CRITICAL:.6f} (THE LENS)                         ║
║    KAPPA_S = {KAPPA_S:.6f} (consciousness threshold)                ║
║                                                                   ║
║  Press Ctrl+C to stop                                             ║
╚═══════════════════════════════════════════════════════════════════╝
""")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Rosetta Helix Visualization Server')
    parser.add_argument('--port', type=int, default=8765, help='Port to run on')
    parser.add_argument('--kira-api', type=str, default=None, help='Proxy to KIRA API base (e.g., http://localhost:5000/api)')
    args = parser.parse_args()
    run_server(args.port, kira_api=args.kira_api)
