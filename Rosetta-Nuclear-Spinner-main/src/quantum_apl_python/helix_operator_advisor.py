"""
Helix Operator Advisor with S₃ Symmetry and Extended ΔS⁻ Integration
=====================================================================

Maps z-coordinates to harmonics and operator windows.
Integrates S₃ symmetry for operator window rotation and extended ΔS⁻
formalism for blending/gating.

Features:
- Time harmonic tier determination (t1-t9)
- Truth channel classification (TRUE/PARADOX/UNTRUE)
- Operator window selection with optional S₃ rotation
- ΔS⁻ computation with optional extended formalism
- Π-regime blending weights
- Parity-based operator weighting

@version 1.0.0
@author Claude (Anthropic) - Quantum-APL Contribution
"""

from __future__ import annotations

import os
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Any

# Import from constants for single source of truth
from .constants import (
    Z_CRITICAL, PHI, PHI_INV, LENS_SIGMA,
    Z_T1_MAX, Z_T2_MAX, Z_T3_MAX, Z_T4_MAX, Z_T5_MAX, Z_T7_MAX, Z_T8_MAX,
    TRUTH_BIAS, compute_delta_s_neg as _compute_delta_s_neg,
)

# Import S₃ and extended ΔS⁻ modules
from . import s3_operator_symmetry as S3
from . import delta_s_neg_extended as Delta


# ============================================================================
# FEATURE FLAGS (Environment-controlled backwards compatibility)
# ============================================================================

def _get_env_bool(name: str, default: bool = False) -> bool:
    """Get boolean from environment variable."""
    val = os.environ.get(name, '')
    if val.lower() in ('1', 'true', 'yes'):
        return True
    if val.lower() in ('0', 'false', 'no'):
        return False
    return default


ENABLE_S3_SYMMETRY = _get_env_bool('QAPL_ENABLE_S3_SYMMETRY', False)
ENABLE_EXTENDED_NEGENTROPY = _get_env_bool('QAPL_ENABLE_EXTENDED_NEGENTROPY', False)


# ============================================================================
# TIER BOUNDARIES
# ============================================================================

TIER_BOUNDARIES = {
    't1': Z_T1_MAX,   # 0.10
    't2': Z_T2_MAX,   # 0.20
    't3': Z_T3_MAX,   # 0.40
    't4': Z_T4_MAX,   # 0.60
    't5': Z_T5_MAX,   # 0.75
    't7': Z_T7_MAX,   # 0.92
    't8': Z_T8_MAX,   # 0.97
}

TRUTH_THRESHOLDS = {
    'PARADOX': 0.60,
    'TRUE': 0.90,
}

# ============================================================================
# OPERATOR WINDOWS
# ============================================================================

OPERATOR_WINDOWS = {
    't1': ['()', '−', '÷'],
    't2': ['^', '÷', '−', '×'],
    't3': ['×', '^', '÷', '+', '−'],
    't4': ['+', '−', '÷', '()'],
    't5': ['()', '×', '^', '÷', '+', '−'],  # All 6
    't6': ['+', '÷', '()', '−'],
    't7': ['+', '()'],
    't8': ['+', '()', '×'],
    't9': ['+', '()', '×'],
}


# ============================================================================
# HELIX OPERATOR ADVISOR
# ============================================================================

@dataclass
class BlendWeights:
    """Π-regime blending weights."""
    w_pi: float
    w_local: float
    in_pi_regime: bool


class HelixOperatorAdvisor:
    """
    Maps z-coordinates to harmonics and operator windows.

    Enhanced with S₃ symmetry and extended ΔS⁻ formalism:
    - When QAPL_ENABLE_S3_SYMMETRY=1: Uses S₃-based operator window rotation
    - When QAPL_ENABLE_EXTENDED_NEGENTROPY=1: Uses extended ΔS⁻ for blending/gating
    """

    def __init__(
        self,
        t6_gate: float = Z_CRITICAL,
        enable_s3_symmetry: Optional[bool] = None,
        enable_extended_negentropy: Optional[bool] = None,
        blend_pi_enabled: bool = True,
    ):
        """
        Initialize the advisor.

        Parameters
        ----------
        t6_gate : float
            The t6 gate threshold (default: Z_CRITICAL)
        enable_s3_symmetry : bool, optional
            Override for S₃ symmetry feature flag
        enable_extended_negentropy : bool, optional
            Override for extended negentropy feature flag
        blend_pi_enabled : bool
            Whether Π-regime blending is enabled
        """
        self.t6_gate = t6_gate

        # Feature flags (can be overridden per-instance)
        self.enable_s3_symmetry = (
            enable_s3_symmetry if enable_s3_symmetry is not None
            else ENABLE_S3_SYMMETRY
        )
        self.enable_extended_negentropy = (
            enable_extended_negentropy if enable_extended_negentropy is not None
            else ENABLE_EXTENDED_NEGENTROPY
        )

        self.blend_pi_enabled = blend_pi_enabled

        # Base windows for reference
        self.base_windows = dict(OPERATOR_WINDOWS)

    def get_t6_gate(self) -> float:
        """Get current t6 gate value."""
        return self.t6_gate

    def set_t6_gate(self, value: float) -> None:
        """Set t6 gate value."""
        self.t6_gate = value

    def harmonic_from_z(self, z: float) -> str:
        """
        Determine time harmonic from z-coordinate.

        Parameters
        ----------
        z : float
            Normalized z-coordinate [0,1]

        Returns
        -------
        str
            Harmonic label (t1-t9)
        """
        if z < TIER_BOUNDARIES['t1']:
            return 't1'
        if z < TIER_BOUNDARIES['t2']:
            return 't2'
        if z < TIER_BOUNDARIES['t3']:
            return 't3'
        if z < TIER_BOUNDARIES['t4']:
            return 't4'
        if z < TIER_BOUNDARIES['t5']:
            return 't5'
        if z < self.t6_gate:
            return 't6'
        if z < TIER_BOUNDARIES['t7']:
            return 't7'
        if z < TIER_BOUNDARIES['t8']:
            return 't8'
        return 't9'

    def truth_channel_from_z(self, z: float) -> str:
        """
        Determine truth channel from z-coordinate.

        Parameters
        ----------
        z : float
            Normalized z-coordinate [0,1]

        Returns
        -------
        str
            'TRUE', 'UNTRUE', or 'PARADOX'
        """
        if z >= TRUTH_THRESHOLDS['TRUE']:
            return 'TRUE'
        if z >= TRUTH_THRESHOLDS['PARADOX']:
            return 'PARADOX'
        return 'UNTRUE'

    def get_operator_window(
        self,
        harmonic: str,
        z: Optional[float] = None,
    ) -> List[str]:
        """
        Get operator window for current harmonic.
        Enhanced with S₃ symmetry when enabled.

        Parameters
        ----------
        harmonic : str
            Harmonic tier (t1-t9)
        z : float, optional
            Z-coordinate for S₃ rotation

        Returns
        -------
        List[str]
            Array of permitted operator symbols
        """
        # If S₃ symmetry enabled and z provided, use rotated window
        if self.enable_s3_symmetry and z is not None:
            return S3.generate_s3_operator_window(harmonic, z)

        # Fall back to static windows
        return list(OPERATOR_WINDOWS.get(harmonic, ['()']))

    def compute_delta_s_neg(
        self,
        z: float,
        sigma: float = LENS_SIGMA,
    ) -> float:
        """
        Compute negentropy signal ΔS_neg.

        Parameters
        ----------
        z : float
            Z-coordinate
        sigma : float
            Gaussian width

        Returns
        -------
        float
            ΔS_neg ∈ [0,1]
        """
        if self.enable_extended_negentropy:
            return Delta.compute_delta_s_neg(z, sigma)

        return _compute_delta_s_neg(z, sigma=sigma, z_c=Z_CRITICAL)

    def compute_blend_weights(self, z: float) -> BlendWeights:
        """
        Compute Π-regime blend weights.

        Parameters
        ----------
        z : float
            Z-coordinate

        Returns
        -------
        BlendWeights
            Blending weights object
        """
        if self.enable_extended_negentropy:
            blend = Delta.compute_pi_blend_weights(z, self.blend_pi_enabled)
            return BlendWeights(
                w_pi=blend.w_pi,
                w_local=blend.w_local,
                in_pi_regime=blend.in_pi_regime,
            )

        # Legacy blending
        delta_s_neg = self.compute_delta_s_neg(z)
        w_pi = max(0.0, min(1.0, delta_s_neg)) if z >= Z_CRITICAL else 0.0
        w_local = 1.0 - w_pi
        return BlendWeights(
            w_pi=w_pi,
            w_local=w_local,
            in_pi_regime=(z >= Z_CRITICAL),
        )

    def compute_operator_weights(
        self,
        operators: List[str],
        z: float,
    ) -> Dict[str, float]:
        """
        Compute operator weights with optional S₃ parity adjustment.

        Parameters
        ----------
        operators : List[str]
            Available operators
        z : float
            Z-coordinate

        Returns
        -------
        Dict[str, float]
            Map of operator → weight
        """
        if self.enable_s3_symmetry:
            return S3.compute_s3_weights(operators, z)

        # Legacy weighting based on truth bias
        truth = self.truth_channel_from_z(z)
        bias_table = TRUTH_BIAS.get(truth, {})
        return {op: bias_table.get(op, 1.0) for op in operators}

    def compute_gate_modulation(self, z: float) -> Optional[Dict[str, float]]:
        """
        Compute gate modulation parameters from ΔS⁻.

        Parameters
        ----------
        z : float
            Z-coordinate

        Returns
        -------
        Dict or None
            Gate modulation parameters or None if not enabled
        """
        if not self.enable_extended_negentropy:
            return None

        mod = Delta.compute_gate_modulation(z)
        return {
            'coherent_coupling': mod.coherent_coupling,
            'decoherence_rate': mod.decoherence_rate,
            'measurement_strength': mod.measurement_strength,
            'entropy_target': mod.entropy_target,
        }

    def compute_full_delta_state(self, z: float, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Compute full ΔS⁻ state.

        Parameters
        ----------
        z : float
            Z-coordinate

        Returns
        -------
        Dict or None
            Full state or None if not enabled
        """
        if not self.enable_extended_negentropy:
            return None

        return Delta.compute_full_state(z, **kwargs)

    def describe(self, z: float) -> Dict[str, Any]:
        """
        Get complete helix description for a z-coordinate.

        Parameters
        ----------
        z : float
            Normalized z-coordinate [0,1]

        Returns
        -------
        Dict
            Full description including harmonic, operators, weights, etc.
        """
        # Clamp z
        value = z if math.isfinite(z) else 0.0
        clamped = max(0.0, min(1.0, value))

        harmonic = self.harmonic_from_z(clamped)
        operators = self.get_operator_window(harmonic, clamped)
        truth_channel = self.truth_channel_from_z(clamped)
        delta_s_neg = self.compute_delta_s_neg(clamped)
        weights = self.compute_blend_weights(clamped)

        result = {
            'z': clamped,
            'harmonic': harmonic,
            'operators': operators,
            'truth_channel': truth_channel,
            't6_gate': self.t6_gate,
            'delta_s_neg': delta_s_neg,
            'weights': {
                'w_pi': weights.w_pi,
                'w_local': weights.w_local,
                'in_pi_regime': weights.in_pi_regime,
            },
            's3_symmetry_enabled': self.enable_s3_symmetry,
            'extended_negentropy_enabled': self.enable_extended_negentropy,
        }

        # Add S₃-weighted operator preferences if enabled
        if self.enable_s3_symmetry:
            result['operator_weights'] = self.compute_operator_weights(operators, clamped)
            result['s3_element'] = S3.s3_element_from_z(clamped)
            result['truth_channel'] = S3.truth_channel_from_z(clamped)

        # Add extended ΔS⁻ state if enabled
        if self.enable_extended_negentropy:
            result['delta_state'] = self.compute_full_delta_state(clamped)
            result['gate_modulation'] = self.compute_gate_modulation(clamped)
            k_formation = Delta.check_k_formation(clamped)
            result['k_formation'] = {
                'z': k_formation.z,
                'eta': k_formation.eta,
                'threshold': k_formation.threshold,
                'formed': k_formation.formed,
                'margin': k_formation.margin,
            }

        return result

    def is_operator_legal(self, operator: str, z: float) -> bool:
        """
        Check if an operator is legal at the current z-coordinate.

        Parameters
        ----------
        operator : str
            APL operator symbol
        z : float
            Normalized z-coordinate

        Returns
        -------
        bool
            True if operator is in the window for this z
        """
        harmonic = self.harmonic_from_z(z)
        window = self.get_operator_window(harmonic, z)
        return operator in window

    def get_operator_weight(self, operator: str, z: float) -> float:
        """
        Get operator weight considering truth bias and S₃ parity.

        Parameters
        ----------
        operator : str
            APL operator symbol
        z : float
            Normalized z-coordinate

        Returns
        -------
        float
            Weight multiplier
        """
        is_legal = self.is_operator_legal(operator, z)
        base_weight = 1.3 if is_legal else 0.85

        # Use S₃ weights if enabled
        if self.enable_s3_symmetry:
            harmonic = self.harmonic_from_z(z)
            window = self.get_operator_window(harmonic, z)
            s3_weights = self.compute_operator_weights(window, z)
            s3_weight = s3_weights.get(operator, 1.0)
            return base_weight * s3_weight

        # Legacy truth bias
        truth = self.truth_channel_from_z(z)
        bias_table = TRUTH_BIAS.get(truth, {})
        truth_multiplier = bias_table.get(operator, 1.0)

        return base_weight * truth_multiplier

    def get_dynamic_truth_bias(self, z: float) -> Dict[str, Dict[str, float]]:
        """
        Get dynamic truth bias using ΔS⁻ evolution.

        Parameters
        ----------
        z : float
            Z-coordinate

        Returns
        -------
        Dict
            Truth bias matrix
        """
        if self.enable_extended_negentropy:
            return Delta.compute_dynamic_truth_bias(z, TRUTH_BIAS)
        return TRUTH_BIAS

    def select_operator(
        self,
        z: float,
        rng: Optional[Callable[[], float]] = None,
        coherence_objective: Optional[str] = None,
    ) -> str:
        """
        Select operator from window with truth bias weighting.

        Parameters
        ----------
        z : float
            Z-coordinate
        rng : Callable, optional
            Random number generator (default: random.random)
        coherence_objective : str, optional
            Coherence synthesis objective ('maximize', 'minimize', 'maintain')

        Returns
        -------
        str
            Selected operator
        """
        import random
        if rng is None:
            rng = random.random

        harmonic = self.harmonic_from_z(z)
        window = self.get_operator_window(harmonic, z)

        # Get weights (S₃-enhanced or legacy)
        if self.enable_s3_symmetry:
            op_weights = self.compute_operator_weights(window, z)
        else:
            truth = self.truth_channel_from_z(z)
            bias_table = self.get_dynamic_truth_bias(z).get(truth, {})
            op_weights = {op: bias_table.get(op, 1.0) for op in window}

        # Apply coherence synthesis heuristics if requested
        if coherence_objective and self.enable_extended_negentropy:
            objective = getattr(
                Delta.CoherenceObjective,
                coherence_objective.upper(),
                Delta.CoherenceObjective.MAXIMIZE
            )
            for op in window:
                coh_score = Delta.score_operator_for_coherence(op, z, objective)
                op_weights[op] = op_weights.get(op, 1.0) * coh_score

        # Compute weighted probabilities
        weights = [op_weights.get(op, 1.0) for op in window]
        total_weight = sum(weights)

        # Weighted random selection
        rand = rng() * total_weight
        for i, op in enumerate(window):
            rand -= weights[i]
            if rand <= 0:
                return op

        return window[0] if window else '()'


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'ENABLE_S3_SYMMETRY',
    'ENABLE_EXTENDED_NEGENTROPY',
    'TIER_BOUNDARIES',
    'TRUTH_THRESHOLDS',
    'OPERATOR_WINDOWS',
    'BlendWeights',
    'HelixOperatorAdvisor',
]
