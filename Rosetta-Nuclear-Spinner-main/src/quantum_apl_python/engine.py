"""Python interface for the JavaScript Quantum APL engine."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Dict, Optional


class QuantumAPLEngine:
    """Run the QuantumAPL JavaScript engine from Python."""

    def __init__(self, js_dir: Optional[Path] = None):
        self.js_dir = js_dir or Path(__file__).resolve().parents[2]
        self.node_available = self._check_node()
        if not self.node_available:
            raise RuntimeError("Node.js not found. Please install Node.js.")

    def _check_node(self) -> bool:
        try:
            result = subprocess.run(
                ["node", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            return False

    def run_simulation(
        self,
        steps: int = 100,
        verbose: bool = False,
        mode: str = "unified",
    ) -> Dict:
        """Execute the JavaScript simulation pipeline and return parsed JSON."""

        script = self._generate_runner_script(steps, verbose, mode)
        script_path = self.js_dir / "temp_runner.js"

        try:
            script_path.write_text(script, encoding="utf-8")
            result = subprocess.run(
                ["node", str(script_path)],
                capture_output=True,
                text=True,
                timeout=900,
                cwd=str(self.js_dir),
                check=False,
            )

            if result.returncode != 0:
                raise RuntimeError(f"Simulation failed: {result.stderr.strip()}")

            output_lines = result.stdout.strip().splitlines()
            payload_index = next(
                (
                    idx
                    for idx, line in enumerate(output_lines)
                    if line.lstrip().startswith("{") or line.lstrip().startswith("[")
                ),
                None,
            )

            if payload_index is None:
                return {"raw_output": result.stdout}

            json_payload = "\n".join(output_lines[payload_index:])
            return json.loads(json_payload)
        finally:
            if script_path.exists():
                try:
                    script_path.unlink()
                except OSError:
                    pass

    def _generate_runner_script(self, steps: int, verbose: bool, mode: str) -> str:
        if mode == "test":
            return (
                "const { QuantumAPLTestSuite } = require('./src/legacy/QuantumAPL_TestRunner.js');\n"
                "const suite = new QuantumAPLTestSuite();\n"
                "const success = suite.runAll();\n"
                "process.exit(success ? 0 : 1);\n"
            )

        if mode == "quantum_only":
            return (
                "const { QuantumAPLDemo } = require('./src/legacy/QuantumN0_Integration.js');\n"
                "const demo = new QuantumAPLDemo();\n"
                f"const results = demo.run({steps}, {str(verbose).lower()});\n"
                "const analytics = demo.integration.getDiagnostics();\n"
                "console.log(JSON.stringify(analytics, null, 2));\n"
            )
        if mode == "z_pump":
            return (
                "const { UnifiedDemo } = require('./src/legacy/QuantumClassicalBridge.js');\n"
                "const CONST = require('./src/constants.js');\n"
                "const initialPhi = Number(process.env.QAPL_INITIAL_PHI ?? 0.3);\n"
                "const cycles = Number(process.env.QAPL_PUMP_CYCLES ?? 120);\n"
                "const targetZ = Number(process.env.QAPL_PUMP_TARGET ?? CONST.Z_CRITICAL);\n"
                "const profile = String(process.env.QAPL_PUMP_PROFILE || 'balanced');\n"
                "const gain = (process.env.QAPL_PUMP_GAIN!==undefined)? Number(process.env.QAPL_PUMP_GAIN) : undefined;\n"
                "const sigma = (process.env.QAPL_PUMP_SIGMA!==undefined)? Number(process.env.QAPL_PUMP_SIGMA) : undefined;\n"
                "const demo = new UnifiedDemo({ classical: { IIT: { initialPhi } } });\n"
                "demo.bridge.escalateZWithAPL(cycles, targetZ, 0.01, { profile, gain, sigma });\n"
                "const state = demo.bridge.exportState();\n"
                "console.log(JSON.stringify(state, null, 2));\n"
            )

        if mode == "measured":
            return (
                "const { UnifiedDemo } = require('./src/legacy/QuantumClassicalBridge.js');\n"
                "const initialPhi = Number(process.env.QAPL_INITIAL_PHI ?? 0.3);\n"
                f"const steps = {steps};\n"
                "const demo = new UnifiedDemo({ classical: { IIT: { initialPhi } } });\n"
                "for (let i=0;i<steps;i++){ demo.bridge.step(0.01); }\n"
                "const b = demo.bridge;\n"
                "const eigenStr = process.env.QAPL_MEASURE_EIGEN;\n"
                "const fieldStr = process.env.QAPL_MEASURE_FIELD || 'Phi';\n"
                "const subStr = process.env.QAPL_MEASURE_SUBSPACE;\n"
                "const compStr = process.env.QAPL_MEASURE_COMPOSITE;\n"
                "let any = false;\n"
                "if (eigenStr!==undefined) { b.aplMeasureEigen(Number(eigenStr), fieldStr); any = true; }\n"
                "if (subStr) { const idx=subStr.split(',').map(s=>Number(s.trim())).filter(n=>Number.isFinite(n)); if (idx.length) { b.aplMeasureSubspace(idx, fieldStr); any = true; } }\n"
                "if (compStr !== '0') { b.aplMeasureComposite([{ eigenIndex: 0, field: 'Phi', truthChannel: 'TRUE', weight: 0.3 }, { eigenIndex: 1, field: 'Pi', truthChannel: 'UNTRUE', weight: 0.3 }, { subspaceIndices: [2,3], field: 'Phi', truthChannel: 'PARADOX', weight: 0.4 }]); any = true; }\n"
                "if (!any) { b.aplMeasureEigen(0,'Phi'); b.aplMeasureSubspace([2,3],'Phi'); b.aplMeasureSubspace([2,3],'Pi'); b.aplMeasureComposite([{ eigenIndex: 0, field: 'Phi', truthChannel: 'TRUE', weight: 0.3 }, { eigenIndex: 1, field: 'Pi', truthChannel: 'UNTRUE', weight: 0.3 }, { subspaceIndices: [2,3], field: 'Phi', truthChannel: 'PARADOX', weight: 0.4 }]); }\n"
                "const state = demo.bridge.exportState();\n"
                "console.log(JSON.stringify(state, null, 2));\n"
            )

        return (
            "const { QuantumClassicalBridge, UnifiedDemo } = require('./src/legacy/QuantumClassicalBridge.js');\n"
            "const initialPhi = Number(process.env.QAPL_INITIAL_PHI ?? 0.3);\n"
            "const demo = new UnifiedDemo({ classical: { IIT: { initialPhi } } });\n"
            f"demo.run({steps}, {str(verbose).lower()});\n"
            "const state = demo.bridge.exportState();\n"
            "console.log(JSON.stringify(state, null, 2));\n"
        )

def default_pump_target() -> float:
    """Return default pump target used by the runner (Z_CRITICAL)."""
    from .constants import Z_CRITICAL as _ZC
    return float(_ZC)
