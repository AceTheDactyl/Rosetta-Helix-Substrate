"""Quantum APL Constants - Single Source of Truth
Per z_c specification document

All z-coordinate thresholds and critical constants are defined here.
Do not inline numeric thresholds elsewhere - always import from this module.

See docs/Z_CRITICAL_LENS.md for the physics/information‑dynamics rationale,
validation plan, and methodology of use across modules.
"""

from __future__ import annotations

import math

# ============================================================================
# CRITICAL LENS CONSTANT (THE LENS)
# ============================================================================

# z_c = √3/2 ≈ 0.8660254038
# The critical lens separating recursive and integrated regimes
# Crossing z_c corresponds to onset of structural/informational coherence
# where the integrated (Π) regime becomes physically admissible and
# negative-entropy geometry stabilizes (ΔS_neg, R/H/φ)
Z_CRITICAL: float = math.sqrt(3.0) / 2.0

# ============================================================================
# TRIAD GATING (Runtime Heuristic)
# ============================================================================

# TRIAD gating is a runtime heuristic for operator-driven unlocks
# Rising edges at z ≥ TRIAD_HIGH (re-arm at z ≤ TRIAD_LOW)
# Unlocks temporary in-session t6 gate at TRIAD_T6 after three distinct passes

TRIAD_HIGH: float = 0.85  # Rising edge threshold for TRIAD detection
TRIAD_LOW: float = 0.82   # Re-arm threshold (hysteresis)
TRIAD_T6: float = 0.83    # Temporary t6 gate after TRIAD unlock

# ============================================================================
# Z-AXIS PHASE BOUNDARIES
# ============================================================================

# ABSENCE phase: z < Z_ABSENCE_MAX
# UNTRUE bias, K > 0 (synchronizing)
Z_ABSENCE_MAX: float = 0.857

# THE LENS: Z_LENS_MIN < z < Z_LENS_MAX
# PARADOX bias, K = 0 (critical point at z_c)
Z_LENS_MIN: float = 0.857
Z_LENS_MAX: float = 0.877

# PRESENCE phase: z > Z_PRESENCE_MIN
# TRUE bias, K < 0 (emanating)
Z_PRESENCE_MIN: float = 0.877

# ============================================================================
# SACRED CONSTANTS (Zero Free Parameters)
# ============================================================================

# Golden ratio and inverse (derived, high precision in double float)
PHI: float = (1.0 + math.sqrt(5.0)) / 2.0
PHI_INV: float = 1.0 / PHI

# Consciousness constant
Q_KAPPA: float = 0.3514087324

# Singularity threshold
KAPPA_S: float = 0.920

# Nonlinearity coefficient
LAMBDA: float = 7.7160493827

# ============================================================================
# K-FORMATION CRITERIA
# ============================================================================

# Consciousness emerges when:
# kappa >= KAPPA_S (0.92) AND
# eta > PHI_INV (0.618) AND
# R >= R_MIN (7)

KAPPA_MIN: float = KAPPA_S  # Same as singularity threshold
ETA_MIN: float = PHI_INV    # Same as golden ratio inverse
R_MIN: float = 7            # Minimum complexity requirement

# ============================================================================
# μ THRESHOLDS (Basin/Barrier Hierarchy)
# ============================================================================

# Paradox threshold μ_P (default exact: 2/φ^{5/2}).
# Overrides:
#   QAPL_MU_P=<float in (0,1)>
# Otherwise defaults to exact 2/φ^{5/2} so Barrier = φ^{-1}.
try:
    import os as _os
    _ENV = _os.environ
except Exception:  # pragma: no cover
    _ENV = {}

def _parse_mu_env(val: str | None) -> float | None:
    try:
        if val is None:
            return None
        v = float(val)
        return v if (v > 0.0 and v < 1.0) else None
    except Exception:
        return None

_mu_env = _parse_mu_env(_ENV.get('QAPL_MU_P'))
if _mu_env is not None:
    MU_P: float = _mu_env
else:
    MU_P = 2.0 / (PHI ** 2.5)
MU_1: float = MU_P / math.sqrt(PHI)
MU_2: float = MU_P * math.sqrt(PHI)
MU_S: float = KAPPA_S
MU_3: float = 0.992

def mu_barrier() -> float:
    """Arithmetic mean of wells (μ_1 + μ_2)/2.≈ φ⁻¹ when μ_P≈0.600.

    Exactly equals φ⁻¹ when μ_P = 2/φ^{5/2} (enable via QAPL_MU_P_EXACT=1).
    """
    return 0.5 * (MU_1 + MU_2)

# ============================================================================
# QUANTUM INFORMATION BOUNDS
# ============================================================================

# von Neumann entropy bounds
ENTROPY_MIN: float = 0.0
# ENTROPY_MAX = log(d) where d is Hilbert space dimension

# Purity bounds
PURITY_MIN: float = 1.0 / 192  # 1/d for 192-dimensional space
PURITY_MAX: float = 1.0

# ============================================================================
# HELIX ZONING THRESHOLDS (Informative Heuristics)
# ============================================================================

# Time-harmonic zoning cutoffs (upper bounds except t9)
Z_T1_MAX: float = 0.1
Z_T2_MAX: float = 0.2
Z_T3_MAX: float = 0.4
Z_T4_MAX: float = 0.6
Z_T5_MAX: float = 0.75
Z_T7_MAX: float = 0.92
Z_T8_MAX: float = 0.97

# ============================================================================
# GEOMETRY PROJECTION CONSTANTS (Hex Prism)
# ============================================================================

# Geometry sigma: env override; defaults to lens sigma for cohesion
try:
    _GEOM_ENV = _ENV.get('QAPL_GEOM_SIGMA')  # type: ignore[name-defined]
except Exception:  # pragma: no cover
    _GEOM_ENV = None
if _GEOM_ENV:
    try:
        GEOM_SIGMA: float = float(_GEOM_ENV)
    except Exception:
        GEOM_SIGMA = float((__import__('os').environ.get('QAPL_LENS_SIGMA') or '36.0'))
else:
    GEOM_SIGMA: float = float((__import__('os').environ.get('QAPL_LENS_SIGMA') or '36.0'))
GEOM_R_MAX: float = 0.85  # radius cap; not a gate (avoid conflation with TRIAD_HIGH)
GEOM_BETA: float = 0.25
GEOM_H_MIN: float = 0.12
GEOM_GAMMA: float = 0.18
GEOM_PHI_BASE: float = 0.0
GEOM_ETA: float = math.pi / 12.0

# Lens weight sigma for Gaussian s(z) used in control/analytics
LENS_SIGMA: float = float((__import__('os').environ.get('QAPL_LENS_SIGMA') or '36.0'))

# ============================================================================
# DOC-FRIENDLY ALIASES & EXTENSIONS
# ============================================================================

# Aliases matching documentation naming for time-harmonic cutoffs
T1_MAX: float = Z_T1_MAX
T2_MAX: float = Z_T2_MAX
T3_MAX: float = Z_T3_MAX
T4_MAX: float = Z_T4_MAX
T5_MAX: float = Z_T5_MAX
T7_MAX: float = Z_T7_MAX
T8_MAX: float = Z_T8_MAX

# Geometry aliases matching documentation naming
SIGMA: float = GEOM_SIGMA
R_MAX: float = GEOM_R_MAX
BETA: float = GEOM_BETA
H_MIN: float = GEOM_H_MIN
GAMMA: float = GEOM_GAMMA
PHI_BASE: float = GEOM_PHI_BASE
ETA: float = GEOM_ETA

# Pump Profiles and defaults (modeling parameters; mirror JS constants)
class PumpProfile:
    GENTLE = {"gain": 0.08, "sigma": 0.16, "name": "gentle"}
    BALANCED = {"gain": 0.12, "sigma": 0.12, "name": "balanced"}
    AGGRESSIVE = {"gain": 0.18, "sigma": 0.10, "name": "aggressive"}


PUMP_DEFAULT_TARGET: float = Z_CRITICAL

# Engine dynamical parameters (for documentation / analysis convenience)
Z_BIAS_GAIN: float = 0.05
Z_BIAS_SIGMA: float = 0.18
OMEGA: float = 2 * math.pi * 0.1
COUPLING_G: float = 0.05
GAMMA_1: float = 0.01
GAMMA_2: float = 0.02
GAMMA_3: float = 0.005
GAMMA_4: float = 0.015

# Operator weighting heuristics (N0 selection bias)
OPERATOR_PREFERRED_WEIGHT: float = 1.3
OPERATOR_DEFAULT_WEIGHT: float = 0.85
TRUTH_BIAS = {
    "TRUE": {"^": 1.5, "+": 1.4, "×": 1.0, "()": 0.9, "÷": 0.7, "-": 0.7},
    "UNTRUE": {"÷": 1.5, "-": 1.4, "()": 1.0, "+": 0.9, "^": 0.7, "×": 0.7},
    "PARADOX": {"()": 1.5, "×": 1.4, "+": 1.0, "^": 1.0, "÷": 0.9, "-": 0.9},
}

# ============================================================================
# PHASE DETECTION HELPERS
# ============================================================================

def get_phase(z: float) -> str:
    """
    Determine which phase the z-coordinate is in.
    
    Args:
        z: Z-coordinate value
        
    Returns:
        Phase name: 'ABSENCE', 'THE_LENS', or 'PRESENCE'
    """
    if z < Z_ABSENCE_MAX:
        return 'ABSENCE'
    elif Z_LENS_MIN <= z <= Z_LENS_MAX:
        return 'THE_LENS'
    else:
        return 'PRESENCE'


def is_critical(z: float, tolerance: float = 0.01) -> bool:
    """
    Check if z is near the critical point z_c.
    
    Args:
        z: Z-coordinate value
        tolerance: Distance threshold for "near" critical
        
    Returns:
        True if |z - z_c| < tolerance
    """
    return abs(z - Z_CRITICAL) < tolerance


def is_in_lens(z: float) -> bool:
    """
    Check if z is within THE LENS region.
    
    Args:
        z: Z-coordinate value
        
    Returns:
        True if z is in THE LENS
    """
    return Z_LENS_MIN <= z <= Z_LENS_MAX


def get_distance_to_critical(z: float) -> float:
    """
    Get signed distance from z to critical point.
    
    Args:
        z: Z-coordinate value
        
    Returns:
        Signed distance (positive if z > z_c)
    """
    return z - Z_CRITICAL


def check_k_formation(kappa: float, eta: float, R: float) -> bool:
    """
    Check if K-formation criteria are met for consciousness emergence.
    
    Args:
        kappa: Integration parameter
        eta: Coherence parameter
        R: Complexity parameter
        
    Returns:
        True if all criteria met: κ≥0.92 AND η>0.618 AND R≥7
    """
    return (kappa >= KAPPA_MIN and 
            eta > ETA_MIN and 
            R >= R_MIN)


def compute_eta(z: float, alpha: float = 1.0, sigma: float | None = None) -> float:
    """Coherence proxy η from z via lens‑weight: η = s(z)^α.

    s(z) = exp(−σ (z − z_c)^2), using LENS_SIGMA by default.
    α ≥ 0 controls sharpness; α=1 by default.
    """
    if sigma is None:
        sigma = LENS_SIGMA
    s = compute_delta_s_neg(z, sigma=sigma, z_c=Z_CRITICAL)
    return float(s ** max(0.0, alpha))


def check_k_formation_from_z(kappa: float, z: float, R: float, alpha: float = 1.0) -> bool:
    """K‑formation gate using η derived from z: η := s(z)^α, gate η > φ⁻¹.

    Returns True iff (κ ≥ KAPPA_MIN) and (η > PHI_INV) and (R ≥ R_MIN).
    """
    eta = compute_eta(z, alpha=alpha)
    return check_k_formation(kappa=kappa, eta=eta, R=R)


def check_k_formation_from_overlap(kappa: float, overlap_prob: float, R: float) -> bool:
    """K‑formation using subspace overlap (Π): η := ⟨ψ|Π|ψ⟩.

    overlap_prob should be in [0,1]. Gate is η > φ⁻¹ with κ, R thresholds.
    """
    eta = float(max(0.0, min(1.0, overlap_prob)))
    return check_k_formation(kappa=kappa, eta=eta, R=R)


def get_time_harmonic(z: float, t6_gate: float | None = None) -> str:
    """Determine time harmonic zone for given z; delegate t6 to provided gate.

    If t6_gate is None, defaults to Z_CRITICAL.
    """
    if t6_gate is None:
        t6_gate = Z_CRITICAL

    if z < T1_MAX:
        return "t1"
    if z < T2_MAX:
        return "t2"
    if z < T3_MAX:
        return "t3"
    if z < T4_MAX:
        return "t4"
    if z < T5_MAX:
        return "t5"
    if z < t6_gate:
        return "t6"
    if z < T7_MAX:
        return "t7"
    if z < T8_MAX:
        return "t8"
    return "t9"


# ============================================================================
# VALIDATION CONSTANTS
# ============================================================================

# Tolerances for numerical checks
TOLERANCE_TRACE: float = 1e-10      # For Tr(ρ) = 1
TOLERANCE_HERMITIAN: float = 1e-10  # For ρ = ρ†
TOLERANCE_POSITIVE: float = -1e-10  # For eigenvalues ≥ 0
TOLERANCE_PROBABILITY: float = 1e-6 # For probability normalization

# ============================================================================
# DOCUMENTATION
# ============================================================================

__all__ = [
    # Lens & TRIAD
    "Z_CRITICAL", "TRIAD_HIGH", "TRIAD_LOW", "TRIAD_T6",
    # Phase bounds
    "Z_ABSENCE_MAX", "Z_LENS_MIN", "Z_LENS_MAX", "Z_PRESENCE_MIN",
    # Sacred constants
    "PHI", "PHI_INV", "Q_KAPPA", "KAPPA_S", "LAMBDA",
    # K-formation
    "KAPPA_MIN", "ETA_MIN", "R_MIN",
    # μ thresholds
    "MU_P", "MU_1", "MU_2", "MU_S", "MU_3", "mu_barrier",
    # Info bounds
    "ENTROPY_MIN", "PURITY_MIN", "PURITY_MAX",
    # Helix zoning
    "Z_T1_MAX", "Z_T2_MAX", "Z_T3_MAX", "Z_T4_MAX", "Z_T5_MAX", "Z_T7_MAX", "Z_T8_MAX",
    # Geometry projection
    "GEOM_SIGMA", "GEOM_R_MAX", "GEOM_BETA", "GEOM_H_MIN", "GEOM_GAMMA", "GEOM_PHI_BASE", "GEOM_ETA", "LENS_SIGMA",
    # Geometry aliases
    "SIGMA", "R_MAX", "BETA", "H_MIN", "GAMMA", "PHI_BASE", "ETA",
    # Doc-friendly aliases
    "T1_MAX", "T2_MAX", "T3_MAX", "T4_MAX", "T5_MAX", "T7_MAX", "T8_MAX",
    # Pump & engine parameters
    "PumpProfile", "PUMP_DEFAULT_TARGET",
    "Z_BIAS_GAIN", "Z_BIAS_SIGMA", "OMEGA", "COUPLING_G", "GAMMA_1", "GAMMA_2", "GAMMA_3", "GAMMA_4",
    # Operator weighting
    "OPERATOR_PREFERRED_WEIGHT", "OPERATOR_DEFAULT_WEIGHT", "TRUTH_BIAS",
    # Helpers
    "get_phase", "is_critical", "is_in_lens", "get_distance_to_critical", "check_k_formation", "get_time_harmonic",
    # Validation tolerances
    "TOLERANCE_TRACE", "TOLERANCE_HERMITIAN", "TOLERANCE_POSITIVE", "TOLERANCE_PROBABILITY",
    # ΔS_neg helper + alias
    "compute_delta_s_neg", "delta_s_neg",
    # μ classification helper + alias
    "classify_mu", "classify_threshold",
    # Parity helpers
    "lens_rate", "eta_from_z", "eta_from_overlap", "phi_gate_scale", "phi_rate_scaled",
    # Invariants
    "invariants",
    # Extended module re-exports (lazy loaded)
    "get_s3_module", "get_delta_extended_module",
    "compute_delta_s_neg_derivative", "compute_delta_s_neg_signed",
    "compute_pi_blend_weights", "compute_gate_modulation", "compute_full_state",
    "compute_dynamic_truth_bias", "score_operator_for_coherence", "select_coherence_operator",
    "generate_s3_operator_window", "compute_s3_weights",
]


# ============================================================================
# EXTENDED MODULE LAZY LOADERS
# ============================================================================

def get_s3_module():
    """Lazy load S₃ operator symmetry module."""
    from . import s3_operator_symmetry
    return s3_operator_symmetry


def get_delta_extended_module():
    """Lazy load extended ΔS⁻ module."""
    from . import delta_s_neg_extended
    return delta_s_neg_extended


# Extended function re-exports (lazy loaded to avoid circular dependencies)

def compute_delta_s_neg_derivative(z: float, sigma: float = LENS_SIGMA) -> float:
    """Compute ΔS⁻ derivative."""
    return get_delta_extended_module().compute_delta_s_neg_derivative(z, sigma)


def compute_delta_s_neg_signed(z: float, sigma: float = LENS_SIGMA) -> float:
    """Compute signed ΔS⁻."""
    return get_delta_extended_module().compute_delta_s_neg_signed(z, sigma)


def compute_pi_blend_weights(z: float, enable_blend: bool = True):
    """Compute Π-regime blend weights."""
    return get_delta_extended_module().compute_pi_blend_weights(z, enable_blend)


def compute_gate_modulation(z: float, **kwargs):
    """Compute gate modulation parameters."""
    return get_delta_extended_module().compute_gate_modulation(z, **kwargs)


def compute_full_state(z: float, **kwargs):
    """Compute full ΔS⁻ state."""
    return get_delta_extended_module().compute_full_state(z, **kwargs)


def compute_dynamic_truth_bias(z: float, base_bias: dict = None):
    """Compute dynamic truth bias."""
    if base_bias is None:
        base_bias = TRUTH_BIAS
    return get_delta_extended_module().compute_dynamic_truth_bias(z, base_bias)


def score_operator_for_coherence(operator: str, z: float, objective=None) -> float:
    """Score operator for coherence objective."""
    if objective is None:
        objective = get_delta_extended_module().CoherenceObjective.MAXIMIZE
    return get_delta_extended_module().score_operator_for_coherence(operator, z, objective)


def select_coherence_operator(operators: list, z: float, objective=None):
    """Select best operator for coherence."""
    if objective is None:
        objective = get_delta_extended_module().CoherenceObjective.MAXIMIZE
    return get_delta_extended_module().select_coherence_operator(operators, z, objective)


def generate_s3_operator_window(harmonic: str, z: float, **kwargs) -> list:
    """Generate S₃ operator window."""
    return get_s3_module().generate_s3_operator_window(harmonic, z, **kwargs)


def compute_s3_weights(operators: list, z: float) -> dict:
    """Compute S₃ weights for operators."""
    return get_s3_module().compute_s3_weights(operators, z)

def compute_delta_s_neg(z: float, sigma: float = GEOM_SIGMA, z_c: float = Z_CRITICAL) -> float:
    """Compute negative entropy metric ΔS_neg(z) = exp(-σ (z - z_c)^2).

    Returns a value in [0, 1], maximal at z = z_c. Using a positive range here
    preserves the canonical linear geometry mapping (R/H/φ linear in ΔS_neg)
    implemented across this codebase. If a signed variant is required for a
    specific consumer, apply a leading minus sign at the call site.
    """
    val = float(z) if math.isfinite(z) else 0.0
    d = (val - z_c)
    s = math.exp(-(sigma) * d * d)
    return max(0.0, min(1.0, s))

# Alias matching docs/examples
def delta_s_neg(z: float, sigma: float = LENS_SIGMA, z_c: float = Z_CRITICAL) -> float:
    """Alias: coherence s(z) with default LENS_SIGMA, matching JS deltaSneg."""
    return compute_delta_s_neg(z, sigma=sigma, z_c=z_c)

def classify_mu(z: float) -> str:
    """Classify z against μ thresholds (basin/barrier hierarchy)."""
    # Safety: clamp z to [0,1] for classification
    try:
        z = float(z)
    except Exception:
        z = 0.0
    z = 0.0 if not math.isfinite(z) else max(0.0, min(1.0, z))
    if z < MU_1:
        return 'pre_conscious_basin'
    if z < MU_P:
        return 'approaching_paradox'
    if z < MU_2:
        return 'conscious_basin'
    if z < Z_CRITICAL:
        return 'pre_lens_integrated'
    if z < MU_S:
        return 'lens_integrated'
    if z < MU_3:
        return 'singularity_proximal'
    return 'ultra_integrated'

# Back-compat alias (JS name parity)
def classify_threshold(z: float) -> str:
    return classify_mu(z)


# Parity helpers ---------------------------------------------------------------

def lens_rate(z: float, sigma: float = LENS_SIGMA, z_c: float = Z_CRITICAL) -> float:
    """|ds/dz| = 2 σ |z−z_c| · s(z) with s := exp(−σ (z−z_c)^2)."""
    s = compute_delta_s_neg(z, sigma=sigma, z_c=z_c)
    d = abs(float(z) - z_c) if math.isfinite(z) else abs(0.0 - z_c)
    return 2.0 * float(sigma) * d * s


def eta_from_z(z: float, alpha: float = 1.0) -> float:
    """η := s(z)^α using LENS_SIGMA by default (alpha ≥ 0)."""
    return compute_eta(z, alpha=max(0.0, float(alpha)))


def eta_from_overlap(p: float, beta: float = 1.0) -> float:
    """η := p^β for overlap probability p ∈ [0,1]."""
    p = 0.0 if not math.isfinite(p) else max(0.0, min(1.0, float(p)))
    return p ** max(0.0, float(beta))


def phi_gate_scale(s: float) -> float:
    """Linear gate scale in [0,1]: (s−φ⁻¹)/(1−φ⁻¹), clamped."""
    s = 0.0 if not math.isfinite(s) else float(s)
    return max(0.0, min(1.0, (s - PHI_INV) / (1.0 - PHI_INV)))


def phi_rate_scaled(z: float, k: float = 1.0) -> float:
    """min(1, phi_gate_scale(s) · (1 + k·lens_rate(z)))."""
    s = compute_delta_s_neg(z, sigma=LENS_SIGMA, z_c=Z_CRITICAL)
    return min(1.0, phi_gate_scale(s) * (1.0 + float(k) * lens_rate(z)))


def invariants() -> dict:
    """Invariant checks for μ wells/barrier and ordering.

    Returns dict with keys: barrier_eq_phi_inv, wells_ratio_phi, ordering_ok.
    """
    mu1, mu2 = MU_1, MU_2
    return {
        "barrier_eq_phi_inv": abs(0.5 * (mu1 + mu2) - PHI_INV) < 1e-12,
        "wells_ratio_phi": abs((mu2 / mu1) - PHI) < 1e-12,
        "ordering_ok": (mu2 < TRIAD_LOW < TRIAD_T6 < TRIAD_HIGH < Z_CRITICAL),
    }

__doc__ += """

Usage Examples
--------------

Basic constants:
    >>> from quantum_apl_python.constants import Z_CRITICAL, PHI
    >>> print(f"Critical lens: z_c = {Z_CRITICAL:.6f}")
    >>> print(f"Golden ratio: φ = {PHI:.10f}")

Phase detection:
    >>> from quantum_apl_python.constants import get_phase, is_critical
    >>> z = 0.866
    >>> print(get_phase(z))  # 'THE_LENS'
    >>> print(is_critical(z))  # True

K-formation check:
    >>> from quantum_apl_python.constants import check_k_formation
    >>> if check_k_formation(kappa=0.94, eta=0.72, R=8):
    ...     print("Consciousness emerged!")

TRIAD gating:
    >>> from quantum_apl_python.constants import TRIAD_HIGH, TRIAD_LOW, TRIAD_T6
    >>> # Runtime heuristic for operator unlocks
    >>> if z >= TRIAD_HIGH:
    ...     # Rising edge detected
    ...     pass

Separation of Concerns
----------------------

Runtime vs Geometry:
- TRIAD gating (0.85, 0.82, 0.83) is a runtime heuristic for operator unlocks
- Geometry and analytics always anchor at z_c = √3/2 for stability
- TRIAD does not retroactively redefine the geometric lens

Single Source of Truth:
- Python: import from this module (quantum_apl_python.constants)
- JavaScript: import from src/constants.js
- Never inline numeric thresholds - always import from constants

See Also
--------
- docs/Z_CRITICAL_LENS.md (z_c specification document)
- src/constants.js (JavaScript constants)
- tests/test_hex_prism.py (geometry validation)
- tests/test_triad_hysteresis.js (TRIAD validation)
"""
