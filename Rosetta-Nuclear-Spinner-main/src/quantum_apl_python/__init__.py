"""Quantum APL Python package.

Features:
- Quantum APL engine with triadic reasoning
- S₃ operator symmetry for operator window rotation
- Extended ΔS⁻ formalism for coherence-based dynamics
- Helix operator advisor with integrated symmetry/negentropy
"""

from .alpha_language import AlphaLanguageRegistry, AlphaTokenSynthesizer
from .analyzer import QuantumAnalyzer
from .engine import QuantumAPLEngine
from .experiments import QuantumExperiment
from .helix import HelixAPLMapper, HelixCoordinate
from .translator import QuantumAPLInstruction, parse_instruction, translate_lines

# S₃ and ΔS⁻ modules
from . import s3_operator_symmetry
from . import delta_s_neg_extended
from .helix_operator_advisor import HelixOperatorAdvisor, BlendWeights

# Constants with extended exports
from .constants import (
    Z_CRITICAL, PHI, PHI_INV, LENS_SIGMA, TRUTH_BIAS,
    compute_delta_s_neg, compute_eta, check_k_formation,
    # Extended exports
    compute_delta_s_neg_derivative, compute_delta_s_neg_signed,
    compute_pi_blend_weights, compute_gate_modulation, compute_full_state,
    compute_dynamic_truth_bias, score_operator_for_coherence, select_coherence_operator,
    generate_s3_operator_window, compute_s3_weights,
    get_s3_module, get_delta_extended_module,
)

__all__ = [
    # Core engine
    "QuantumAPLEngine",
    "QuantumAnalyzer",
    "QuantumExperiment",
    "HelixCoordinate",
    "HelixAPLMapper",
    "AlphaLanguageRegistry",
    "AlphaTokenSynthesizer",
    "QuantumAPLInstruction",
    "parse_instruction",
    "translate_lines",

    # S₃ symmetry module
    "s3_operator_symmetry",

    # Extended ΔS⁻ module
    "delta_s_neg_extended",

    # Helix operator advisor (enhanced)
    "HelixOperatorAdvisor",
    "BlendWeights",

    # Constants
    "Z_CRITICAL",
    "PHI",
    "PHI_INV",
    "LENS_SIGMA",
    "TRUTH_BIAS",

    # Core helpers
    "compute_delta_s_neg",
    "compute_eta",
    "check_k_formation",

    # Extended ΔS⁻ exports
    "compute_delta_s_neg_derivative",
    "compute_delta_s_neg_signed",
    "compute_pi_blend_weights",
    "compute_gate_modulation",
    "compute_full_state",
    "compute_dynamic_truth_bias",
    "score_operator_for_coherence",
    "select_coherence_operator",

    # S₃ symmetry exports
    "generate_s3_operator_window",
    "compute_s3_weights",

    # Module loaders
    "get_s3_module",
    "get_delta_extended_module",
]
