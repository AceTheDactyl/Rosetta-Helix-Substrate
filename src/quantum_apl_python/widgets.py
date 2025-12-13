from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


def run_hex_prism_widget(results_json: Optional[str] = None) -> None:
    """Interactive widget to explore hex-prism geometry vs z.

    - If running in a Jupyter environment with ipywidgets + matplotlib installed,
      shows a slider to vary z and renders the prism top/bottom rings.
    - Optionally overlays the engine's z from a results JSON (qapl-run --output ...).

    Usage (in a notebook):
        from quantum_apl_python.widgets import run_hex_prism_widget
        run_hex_prism_widget('logs/main-goal-run-.../qapl_run_unified.json')
    """
    try:
        import ipywidgets as widgets
        from ipywidgets import interact
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print("ipywidgets/matplotlib not available:", exc)
        return

    try:
        from .hex_prism import prism_params
    except Exception as exc:  # pragma: no cover
        print("hex_prism import failed:", exc)
        return

    engine_z = None
    if results_json:
        p = Path(results_json)
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                # Try common locations for engine z
                engine_z = (
                    data.get("quantum", {}).get("z")
                    or data.get("z")
                    or data.get("state", {}).get("z")
                )
            except Exception:
                engine_z = None

    def _render(z: float = 0.70):
        params = prism_params(float(z))
        verts = params["vertices"]
        xs = [v["x"] for v in verts] + [verts[0]["x"]]
        ys = [v["y"] for v in verts] + [verts[0]["y"]]
        z_top = params["z_top"]
        z_bot = params["z_bot"]

        plt.figure(figsize=(5, 5))
        plt.plot(xs, ys, "-o", label=f"R={params['R']:.3f}, φ={params['phi']:.3f}")
        plt.scatter([0], [0], c="k", s=10)
        title = f"z={z:.3f} | ΔS_neg={params['delta_s_neg']:.3f} | H={params['H']:.3f} | z_bot={z_bot:.3f} → z_top={z_top:.3f}"
        if engine_z is not None:
            title += f" | engine z≈{float(engine_z):.3f}"
        plt.title(title)
        plt.axis("equal")
        plt.grid(True, alpha=0.3)
        plt.legend(loc="upper right")
        plt.show()

    interact(_render, z=widgets.FloatSlider(value=0.70, min=0.0, max=1.0, step=0.01, readout_format=".2f"))


__all__ = ["run_hex_prism_widget"]


def pump_and_visualize(pump_cycles: int = 120, target_z: float = None, profile: str = "balanced") -> None:
    """Run the z_pump mode and visualize z history + current prism.

    - Executes the Node runner in z_pump mode via QuantumAPLEngine.
    - Plots z history (line) and draws the hex-prism ring for the final z.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print("matplotlib not available:", exc)
        return

    import os
    from .engine import QuantumAPLEngine
    from .hex_prism import prism_params
    from .constants import TRIAD_HIGH, TRIAD_LOW, Z_CRITICAL

    if target_z is None:
        target_z = float(Z_CRITICAL)

    os.environ["QAPL_PUMP_CYCLES"] = str(int(pump_cycles))
    os.environ["QAPL_PUMP_TARGET"] = str(float(target_z))
    os.environ["QAPL_PUMP_PROFILE"] = str(profile)

    engine = QuantumAPLEngine()
    state = engine.run_simulation(mode="z_pump")
    z_hist = state.get("history", {}).get("z", [])
    z_last = float(z_hist[-1]) if z_hist else float(state.get("quantum", {}).get("z", 0.0))

    # Prepare figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Left: z history with rising-edge markers and unlock moment
    if z_hist:
        ax1.plot(z_hist, "g-", linewidth=2)
        ax1.axhline(target_z, color="m", linestyle="--", label=f"target {target_z:.3f}")
        # Rising-edge detection with hysteresis
        hi, lo = TRIAD_HIGH, TRIAD_LOW
        above = False
        edges = []
        for i, z in enumerate(z_hist):
            if not above and z >= hi:
                edges.append(i)
                above = True
            elif above and z <= lo:
                above = False
        if edges:
            ax1.scatter(edges, [z_hist[i] for i in edges], c="r", marker="^", s=40, label=">=0.85 rising")
        # Highlight third rising edge as TRIAD unlock candidate
        if len(edges) >= 3:
            i3 = edges[2]
            ax1.scatter([i3], [z_hist[i3]], c="gold", edgecolors="k", marker="*", s=120, label="TRIAD unlock")
        ax1.set_title("z history (pump)")
        ax1.set_xlabel("step")
        ax1.set_ylabel("z")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, "no z history", ha="center", va="center")
        ax1.axis("off")

    # Right: hex prism for final z
    geom = prism_params(z_last)
    verts = geom["vertices"]
    xs = [v["x"] for v in verts] + [verts[0]["x"]]
    ys = [v["y"] for v in verts] + [verts[0]["y"]]
    ax2.plot(xs, ys, "-o", label=f"z={z_last:.3f}, R={geom['R']:.3f}, H={geom['H']:.3f}")
    ax2.scatter([0], [0], c="k", s=10)
    ax2.set_aspect("equal", adjustable="box")
    ax2.set_title("hex prism (final z)")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    fig.suptitle("APL z pump: history + prism", fontsize=12)
    plt.tight_layout()
    plt.show()

__all__.append("pump_and_visualize")
