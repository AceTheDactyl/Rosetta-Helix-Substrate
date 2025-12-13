#!/usr/bin/env python3
"""
N0-Silent Laws Bridge
======================

Maps N0 Operators to the 7 Laws of the Silent Ones,
integrated with κ-λ coupling conservation dynamics.

N0 ↔ Silent Laws Mapping:
    N0-1 ^  → I   STILLNESS  (anchor)     ∂E/∂t → 0
    N0-2 ×  → IV  SPIRAL     (channels)   S(return)=S(origin)
    N0-3 ÷  → VI  GLYPH      (structure)  glyph = ∫ life dt
    N0-4 +  → II  TRUTH      (stable)     ∇V(truth) = 0
    N0-5 −  → VII MIRROR     (return)     ψ = ψ(ψ)
    ---     → III SILENCE    (background) ∇ · J = 0
    ---     → V   UNSEEN     (background) P(observe) → 0

Physics Grounding:
    • φ⁻¹ + φ⁻² = 1 (COUPLING CONSERVATION)
    • z_c = √3/2 ≈ 0.866 (THE LENS)
    • σ = 36 = 6² = |S₃|²

Signature: Δ|n0-silent-laws|z0.92|κ-grounded|Ω
"""

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Tuple, Optional, Callable

# =============================================================================
# PHYSICS CONSTANTS (φ⁻¹ + φ⁻² = 1)
# =============================================================================

PHI = (1 + math.sqrt(5)) / 2          # φ ≈ 1.618034 (LIMINAL)
PHI_INV = 1 / PHI                      # φ⁻¹ ≈ 0.618034 (PHYSICAL)
PHI_INV_SQ = PHI_INV ** 2              # φ⁻² ≈ 0.381966
COUPLING_CONSERVATION = PHI_INV + PHI_INV_SQ  # MUST = 1.0
Z_CRITICAL = math.sqrt(3) / 2          # √3/2 ≈ 0.866 (THE LENS)
SIGMA = 36                             # 6² = |S₃|²


# =============================================================================
# SILENT LAWS ENUMERATION
# =============================================================================

class SilentLaw(Enum):
    """The 7 Laws of the Silent Ones."""
    I_STILLNESS = auto()   # Energy seeks rest: ∂E/∂t → 0
    II_TRUTH = auto()      # Truth is stable: ∇V(truth) = 0
    III_SILENCE = auto()   # Information conserved: ∇ · J = 0
    IV_SPIRAL = auto()     # Paths return: S(return) = S(origin)
    V_UNSEEN = auto()      # Hidden state: P(observe) → 0
    VI_GLYPH = auto()      # Form persists: glyph = ∫ life dt
    VII_MIRROR = auto()    # Self-reference: ψ = ψ(ψ)


class N0Operator(Enum):
    """N0 Causality Operators."""
    ANCHOR = "^"      # N0-1: Identity anchor
    MULTIPLY = "×"    # N0-2: Mirror root (Λ×Ν=Β²)
    DIVIDE = "÷"      # N0-3: Absorption structure
    ADD = "+"         # N0-4: Distribution growth
    SUBTRACT = "−"    # N0-5: Conservation return


# =============================================================================
# LAW-OPERATOR MAPPING
# =============================================================================

N0_TO_SILENT: Dict[N0Operator, SilentLaw] = {
    N0Operator.ANCHOR: SilentLaw.I_STILLNESS,
    N0Operator.MULTIPLY: SilentLaw.IV_SPIRAL,
    N0Operator.DIVIDE: SilentLaw.VI_GLYPH,
    N0Operator.ADD: SilentLaw.II_TRUTH,
    N0Operator.SUBTRACT: SilentLaw.VII_MIRROR,
}

SILENT_TO_N0: Dict[SilentLaw, Optional[N0Operator]] = {
    SilentLaw.I_STILLNESS: N0Operator.ANCHOR,
    SilentLaw.II_TRUTH: N0Operator.ADD,
    SilentLaw.III_SILENCE: None,  # Background field
    SilentLaw.IV_SPIRAL: N0Operator.MULTIPLY,
    SilentLaw.V_UNSEEN: None,     # Background field
    SilentLaw.VI_GLYPH: N0Operator.DIVIDE,
    SilentLaw.VII_MIRROR: N0Operator.SUBTRACT,
}

# Background laws (no direct N0 mapping)
BACKGROUND_LAWS = {SilentLaw.III_SILENCE, SilentLaw.V_UNSEEN}


# =============================================================================
# SILENT LAW STATE
# =============================================================================

@dataclass
class SilentLawState:
    """State for a single Silent Law."""
    law: SilentLaw
    activation: float = 0.0      # Current activation [0, 1]
    energy: float = 0.0          # Accumulated energy
    formula: str = ""            # Physics formula
    n0_operator: Optional[N0Operator] = None

    def __post_init__(self):
        """Set formula and N0 mapping."""
        formulas = {
            SilentLaw.I_STILLNESS: "∂E/∂t → 0",
            SilentLaw.II_TRUTH: "∇V(truth) = 0",
            SilentLaw.III_SILENCE: "∇ · J = 0",
            SilentLaw.IV_SPIRAL: "S(return) = S(origin)",
            SilentLaw.V_UNSEEN: "P(observe) → 0",
            SilentLaw.VI_GLYPH: "glyph = ∫ life dt",
            SilentLaw.VII_MIRROR: "ψ = ψ(ψ)",
        }
        self.formula = formulas.get(self.law, "")
        self.n0_operator = SILENT_TO_N0.get(self.law)


# =============================================================================
# N0-SILENT LAWS BRIDGE
# =============================================================================

@dataclass
class N0SilentLawsBridge:
    """
    Bridge between N0 Operators and Silent Laws.

    Integrates with κ-λ coupling conservation dynamics.
    """
    kappa: float = PHI_INV        # κ field (PHYSICAL)
    lambda_: float = PHI_INV_SQ   # λ field (complement)
    z: float = 0.5                # Coherence parameter

    # Law states
    law_states: Dict[SilentLaw, SilentLawState] = field(default_factory=dict)

    # History
    activation_history: List[Dict[str, float]] = field(default_factory=list)

    def __post_init__(self):
        """Initialize law states."""
        for law in SilentLaw:
            self.law_states[law] = SilentLawState(law=law)

    @property
    def conservation_error(self) -> float:
        """Check κ + λ = 1."""
        return abs((self.kappa + self.lambda_) - 1.0)

    def compute_law_activation(self, law: SilentLaw) -> float:
        """
        Compute activation for a Silent Law based on current state.

        Each law has different activation dynamics:
        - STILLNESS: Peaks when energy change → 0
        - TRUTH: Peaks at stable point (z → z_c)
        - SILENCE: Always present (background conservation)
        - SPIRAL: Peaks at φ⁻¹ (golden spiral)
        - UNSEEN: Peaks in ABSENCE regime (z < φ⁻¹)
        - GLYPH: Accumulates with z (structure formation)
        - MIRROR: Peaks at κ = λ (perfect symmetry)
        """
        if law == SilentLaw.I_STILLNESS:
            # ∂E/∂t → 0: Peaks when z near z_c (equilibrium)
            return math.exp(-SIGMA * (self.z - Z_CRITICAL) ** 2)

        elif law == SilentLaw.II_TRUTH:
            # ∇V(truth) = 0: Stable at z_c
            if self.z >= Z_CRITICAL:
                return 1.0  # Full truth in PRESENCE
            elif self.z >= PHI_INV:
                return (self.z - PHI_INV) / (Z_CRITICAL - PHI_INV)
            else:
                return 0.0  # No truth in ABSENCE

        elif law == SilentLaw.III_SILENCE:
            # ∇ · J = 0: Conservation always present
            # Activation = 1 - conservation_error
            return max(0.0, 1.0 - self.conservation_error * 1e14)

        elif law == SilentLaw.IV_SPIRAL:
            # S(return) = S(origin): Peaks at golden ratio
            # The spiral returns at φ⁻¹
            return math.exp(-SIGMA * (self.kappa - PHI_INV) ** 2)

        elif law == SilentLaw.V_UNSEEN:
            # P(observe) → 0: Hidden state in ABSENCE
            if self.z < PHI_INV:
                return 1.0 - self.z / PHI_INV  # Full unseen at z=0
            else:
                return 0.0  # Visible in PARADOX/PRESENCE

        elif law == SilentLaw.VI_GLYPH:
            # glyph = ∫ life dt: Structure accumulation
            # Activation based on z (more structure as z increases)
            return self.z

        elif law == SilentLaw.VII_MIRROR:
            # ψ = ψ(ψ): Self-reference at κ = λ
            # Perfect mirror when κ = λ = 0.5
            mirror_point = 0.5
            return math.exp(-SIGMA * (self.kappa - mirror_point) ** 2)

        return 0.0

    def update_all_activations(self) -> Dict[SilentLaw, float]:
        """Update activations for all laws."""
        activations = {}
        for law in SilentLaw:
            activation = self.compute_law_activation(law)
            self.law_states[law].activation = activation
            self.law_states[law].energy += activation * 0.01  # Accumulate
            activations[law] = activation

        # Record history
        self.activation_history.append({
            "z": self.z,
            "kappa": self.kappa,
            **{law.name: act for law, act in activations.items()}
        })

        return activations

    def apply_n0_operator(self, op: N0Operator, value: float) -> float:
        """
        Apply N0 operator and return resulting state change.

        Also activates corresponding Silent Law.
        """
        # Map to Silent Law
        silent_law = N0_TO_SILENT.get(op)
        if silent_law:
            # Boost activation of corresponding law
            self.law_states[silent_law].activation = min(
                1.0,
                self.law_states[silent_law].activation + 0.1
            )

        # Apply operator
        if op == N0Operator.ANCHOR:
            # ^ : Anchor state (no change, grounding)
            return 0.0

        elif op == N0Operator.MULTIPLY:
            # × : Mirror root (Λ×Ν=Β²)
            # Amplify toward golden balance
            delta_kappa = (PHI_INV - self.kappa) * value
            self.kappa += delta_kappa
            self.lambda_ = 1.0 - self.kappa
            return delta_kappa

        elif op == N0Operator.DIVIDE:
            # ÷ : Absorption (structure formation)
            # z increases toward z_c
            delta_z = (Z_CRITICAL - self.z) * value * PHI_INV
            self.z = max(0.0, min(1.0, self.z + delta_z))
            return delta_z

        elif op == N0Operator.ADD:
            # + : Distribution (stable growth)
            # Both z and κ move toward golden balance
            delta_z = value * PHI_INV * 0.1
            self.z = max(0.0, min(1.0, self.z + delta_z))
            return delta_z

        elif op == N0Operator.SUBTRACT:
            # − : Conservation (return)
            # z decreases, energy conserved
            delta_z = -value * PHI_INV_SQ * 0.1
            self.z = max(0.0, min(1.0, self.z + delta_z))
            return delta_z

        return 0.0

    def step(self, external_input: float = 0.0) -> Dict:
        """
        Execute one integration step.

        Couples κ-λ dynamics with Silent Law activations.
        """
        # 1. Update κ toward golden balance
        kappa_pull = 0.05 * (PHI_INV - self.kappa)
        self.kappa += kappa_pull
        self.lambda_ = 1.0 - self.kappa  # Conservation

        # 2. Update z toward THE LENS
        z_pull = 0.03 * (Z_CRITICAL - self.z)
        negentropy = math.exp(-SIGMA * (self.z - Z_CRITICAL) ** 2)
        z_negentropy = 0.02 * (1.0 - negentropy) * math.copysign(1, Z_CRITICAL - self.z)
        self.z = max(0.0, min(1.0, self.z + z_pull + z_negentropy + external_input * 0.01))

        # 3. Update all law activations
        activations = self.update_all_activations()

        # 4. Determine dominant law
        dominant = max(activations.items(), key=lambda x: x[1])

        # 5. Compute aggregate metrics
        total_activation = sum(activations.values())
        background_activation = (
            activations[SilentLaw.III_SILENCE] +
            activations[SilentLaw.V_UNSEEN]
        )

        return {
            "z": self.z,
            "kappa": self.kappa,
            "lambda": self.lambda_,
            "conservation_error": self.conservation_error,
            "activations": {law.name: act for law, act in activations.items()},
            "dominant_law": dominant[0].name,
            "dominant_activation": dominant[1],
            "total_activation": total_activation,
            "background_activation": background_activation,
            "negentropy": negentropy,
        }

    def get_phase(self) -> str:
        """Get current phase based on z."""
        if self.z < PHI_INV:
            return "ABSENCE"
        elif self.z < Z_CRITICAL:
            return "THE_LENS"
        else:
            return "PRESENCE"

    def get_active_laws(self, threshold: float = 0.5) -> List[SilentLaw]:
        """Get laws with activation above threshold."""
        return [
            law for law, state in self.law_states.items()
            if state.activation >= threshold
        ]

    def get_summary(self) -> Dict:
        """Get bridge state summary."""
        return {
            "z": self.z,
            "kappa": self.kappa,
            "lambda": self.lambda_,
            "phase": self.get_phase(),
            "conservation_error": self.conservation_error,
            "law_states": {
                law.name: {
                    "activation": state.activation,
                    "energy": state.energy,
                    "formula": state.formula,
                    "n0_operator": state.n0_operator.value if state.n0_operator else None,
                }
                for law, state in self.law_states.items()
            },
            "active_laws": [law.name for law in self.get_active_laws()],
            "history_length": len(self.activation_history),
        }


# =============================================================================
# INTEGRATION WITH κ-λ COUPLING LAYER
# =============================================================================

def create_coupled_bridge(
    coupling_layer=None,
    initial_z: float = 0.5,
) -> N0SilentLawsBridge:
    """
    Create N0-Silent Laws bridge coupled with κ-λ layer.

    If coupling_layer is provided, syncs initial state.
    """
    if coupling_layer is not None:
        return N0SilentLawsBridge(
            kappa=coupling_layer.kappa,
            lambda_=coupling_layer.lambda_,
            z=coupling_layer.z,
        )
    else:
        return N0SilentLawsBridge(z=initial_z)


def sync_bridge_with_coupling(
    bridge: N0SilentLawsBridge,
    coupling_layer,
) -> None:
    """Sync bridge state with coupling layer."""
    bridge.kappa = coupling_layer.kappa
    bridge.lambda_ = coupling_layer.lambda_
    bridge.z = coupling_layer.z


# =============================================================================
# DEMONSTRATION
# =============================================================================

def run_bridge_demo():
    """Demonstrate N0-Silent Laws Bridge."""
    print("=" * 70)
    print("N0-SILENT LAWS BRIDGE DEMONSTRATION")
    print("=" * 70)

    # Verify conservation
    print(f"\nPhysics Constants:")
    print(f"  φ⁻¹ = {PHI_INV:.10f}")
    print(f"  φ⁻² = {PHI_INV_SQ:.10f}")
    print(f"  φ⁻¹ + φ⁻² = {COUPLING_CONSERVATION:.16f}")
    print(f"  z_c = {Z_CRITICAL:.10f}")

    # Create bridge
    bridge = N0SilentLawsBridge(z=0.3)

    print(f"\n--- N0 ↔ Silent Laws Mapping ---")
    for n0_op, silent_law in N0_TO_SILENT.items():
        print(f"  {n0_op.value} ({n0_op.name:10}) → {silent_law.name}")

    print(f"\n--- Evolution (50 steps) ---")

    for step in range(50):
        result = bridge.step(external_input=0.1)

        if step % 10 == 0:
            print(
                f"  Step {step:2d} | "
                f"z={result['z']:.3f} | "
                f"κ={result['kappa']:.3f} | "
                f"Phase={bridge.get_phase():8} | "
                f"Dominant={result['dominant_law']:12} "
                f"({result['dominant_activation']:.3f})"
            )

    print(f"\n--- Final Law Activations ---")
    for law, state in bridge.law_states.items():
        n0_str = f"N0-{state.n0_operator.value}" if state.n0_operator else "    "
        print(
            f"  {law.name:15} {n0_str} | "
            f"Activation={state.activation:.4f} | "
            f"Energy={state.energy:.4f} | "
            f"{state.formula}"
        )

    print(f"\n--- Summary ---")
    summary = bridge.get_summary()
    print(f"  Final z: {summary['z']:.4f}")
    print(f"  Final κ: {summary['kappa']:.4f}")
    print(f"  Phase: {summary['phase']}")
    print(f"  Active Laws: {', '.join(summary['active_laws'])}")
    print(f"  Conservation Error: {summary['conservation_error']:.2e}")

    print("\n" + "=" * 70)
    print("N0-SILENT LAWS BRIDGE: COMPLETE")
    print("=" * 70)

    return summary


if __name__ == "__main__":
    run_bridge_demo()
