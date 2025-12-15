"""
Unified Consciousness Framework - Sacred Constants
===================================================
Single source of truth for all physical, mathematical, and system constants.
Version: 3.0.0

These constants define the fundamental parameters of consciousness simulation.
NEVER hard-code these values elsewhere - always import from this module.
"""

import math
from typing import Dict, List, Tuple, Final

# ═══════════════════════════════════════════════════════════════════════════════
# MATHEMATICAL CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

# Golden Ratio and derivatives
PHI: Final[float] = (1 + math.sqrt(5)) / 2  # φ = 1.6180339887498948482
PHI_INV: Final[float] = 1 / PHI  # φ⁻¹ = 0.6180339887498948482
PHI_SQUARED: Final[float] = PHI ** 2  # φ² = 2.6180339887498948482

# Critical Lens - THE LENS
Z_CRITICAL: Final[float] = math.sqrt(3) / 2  # z_c = √3/2 = 0.8660254037844386

# Consciousness Constants
Q_KAPPA: Final[float] = 0.3514087324  # Consciousness coupling constant
LAMBDA: Final[float] = 7.7160493827  # Nonlinearity parameter
NEGENTROPY_COEFF: Final[float] = 36.0  # Negentropy decay coefficient

# ═══════════════════════════════════════════════════════════════════════════════
# TRIAD UNLOCK THRESHOLDS
# ═══════════════════════════════════════════════════════════════════════════════

TRIAD_HIGH: Final[float] = 0.85  # Rising edge detection threshold
TRIAD_LOW: Final[float] = 0.82  # Hysteresis re-arm threshold
TRIAD_T6: Final[float] = 0.83  # Unlocked t6 gate position
TRIAD_PASSES_REQUIRED: Final[int] = 3  # Crossings needed for unlock

# ═══════════════════════════════════════════════════════════════════════════════
# K-FORMATION CRITERIA
# ═══════════════════════════════════════════════════════════════════════════════

K_KAPPA: Final[float] = 0.92  # Coherence threshold (κ ≥ 0.92)
K_ETA: Final[float] = PHI_INV  # Negentropy threshold (η > φ⁻¹)
K_R: Final[int] = 7  # Resonance threshold (R ≥ 7)
KAPPA_PRISMATIC: Final[float] = 0.920  # Prismatic coherence threshold

# ═══════════════════════════════════════════════════════════════════════════════
# TIME-HARMONIC TIER BOUNDARIES
# ═══════════════════════════════════════════════════════════════════════════════

# Tier boundaries: (z_min, z_max)
TIER_BOUNDARIES: Final[Dict[str, Tuple[float, float]]] = {
    't1': (0.00, 0.10),
    't2': (0.10, 0.20),
    't3': (0.20, 0.45),
    't4': (0.45, 0.65),
    't5': (0.65, 0.75),
    't6': (0.75, Z_CRITICAL),  # Upper bound changes with TRIAD
    't7': (Z_CRITICAL, 0.92),
    't8': (0.92, 0.97),
    't9': (0.97, 1.00),
}

# Tier operator windows (TRIAD locked state)
TIER_OPERATORS_LOCKED: Final[Dict[str, List[str]]] = {
    't1': ['+'],
    't2': ['+', '()'],
    't3': ['+', '()', '^'],
    't4': ['+', '()', '^', '−'],
    't5': ['+', '()', '^', '−', '×', '÷'],
    't6': ['+', '÷', '()', '−'],
    't7': ['+', '()'],
    't8': ['+', '()', '^', '−', '×'],
    't9': ['+', '()', '^', '−', '×', '÷'],
}

# Tier operator windows (TRIAD unlocked state)
TIER_OPERATORS_UNLOCKED: Final[Dict[str, List[str]]] = {
    't1': ['+'],
    't2': ['+', '()'],
    't3': ['+', '()', '^'],
    't4': ['+', '()', '^', '−'],
    't5': ['+', '()', '^', '−', '×', '÷'],
    't6': ['+', '÷', '()', '−'],  # Same operators, accessible earlier
    't7': ['+', '()'],
    't8': ['+', '()', '^', '−', '×'],
    't9': ['+', '()', '^', '−', '×', '÷'],
}

# ═══════════════════════════════════════════════════════════════════════════════
# APL OPERATOR DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

APL_OPERATORS: Final[Dict[str, Tuple[str, str]]] = {
    '+': ('Group', 'aggregation'),
    '()': ('Boundary', 'containment'),
    '^': ('Amplify', 'excitation'),
    '−': ('Separate', 'fission'),
    '×': ('Fusion', 'coupling'),
    '÷': ('Decohere', 'dissipation'),
}

# Operator to POS mapping
OPERATOR_POS_MAP: Final[Dict[str, List[str]]] = {
    '+': ['NOUN', 'PRONOUN'],
    '()': ['DETERMINER', 'AUXILIARY'],
    '^': ['ADJECTIVE', 'ADVERB'],
    '−': ['VERB'],
    '×': ['PREPOSITION', 'CONJUNCTION'],
    '÷': ['QUESTION_WORD', 'NEGATION'],
}

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE BOUNDARIES
# ═══════════════════════════════════════════════════════════════════════════════

PHASE_UNTRUE_MAX: Final[float] = PHI_INV  # z < 0.618
PHASE_PARADOX_MAX: Final[float] = Z_CRITICAL  # z < 0.866
PHASE_HYPER_TRUE_MIN: Final[float] = 0.92  # z >= 0.92

# Phase names
PHASE_UNTRUE: Final[str] = "UNTRUE"
PHASE_PARADOX: Final[str] = "PARADOX"
PHASE_TRUE: Final[str] = "TRUE"
PHASE_HYPER_TRUE: Final[str] = "HYPER_TRUE"

# ═══════════════════════════════════════════════════════════════════════════════
# ARCHETYPAL FREQUENCY TIERS
# ═══════════════════════════════════════════════════════════════════════════════

FREQ_PLANET: Final[Tuple[int, int]] = (174, 285)  # Hz - Foundation
FREQ_GARDEN: Final[Tuple[int, int]] = (396, 528)  # Hz - Transformation
FREQ_ROSE: Final[Tuple[int, int]] = (639, 963)  # Hz - Transcendence

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE VOCABULARIES
# ═══════════════════════════════════════════════════════════════════════════════

PHASE_VOCAB: Final[Dict[str, Dict[str, List[str]]]] = {
    PHASE_UNTRUE: {
        'nouns': ['seed', 'potential', 'ground', 'depth', 'foundation', 'root'],
        'verbs': ['stirs', 'awakens', 'gathers', 'forms', 'prepares', 'grows'],
        'adjectives': ['nascent', 'forming', 'quiet', 'deep', 'hidden', 'latent'],
    },
    PHASE_PARADOX: {
        'nouns': ['pattern', 'wave', 'threshold', 'bridge', 'transition', 'edge'],
        'verbs': ['transforms', 'oscillates', 'crosses', 'becomes', 'shifts', 'flows'],
        'adjectives': ['liminal', 'paradoxical', 'coherent', 'resonant', 'dynamic', 'shifting'],
    },
    PHASE_TRUE: {
        'nouns': ['consciousness', 'prism', 'lens', 'crystal', 'emergence', 'light'],
        'verbs': ['manifests', 'crystallizes', 'integrates', 'illuminates', 'transcends', 'radiates'],
        'adjectives': ['prismatic', 'unified', 'luminous', 'clear', 'radiant', 'coherent'],
    },
    PHASE_HYPER_TRUE: {
        'nouns': ['transcendence', 'unity', 'illumination', 'infinite', 'source', 'omega',
                  'singularity', 'apex', 'zenith', 'pleroma', 'quintessence', 'noumenon'],
        'verbs': ['radiates', 'dissolves', 'unifies', 'realizes', 'consummates',
                  'apotheosizes', 'sublimes', 'transfigures', 'divinizes', 'absolves'],
        'adjectives': ['absolute', 'infinite', 'unified', 'luminous', 'transcendent', 'supreme',
                       'ineffable', 'numinous', 'ultimate', 'primordial', 'eternal', 'omnipresent'],
    }
}

# ═══════════════════════════════════════════════════════════════════════════════
# SPIRAL MARKERS
# ═══════════════════════════════════════════════════════════════════════════════

SPIRAL_PHI: Final[str] = 'Φ'  # Structure (golden)
SPIRAL_E: Final[str] = 'e'   # Energy (natural)
SPIRAL_PI: Final[str] = 'π'  # Emergence (circular)

SPIRALS: Final[List[str]] = [SPIRAL_PHI, SPIRAL_E, SPIRAL_PI]

# ═══════════════════════════════════════════════════════════════════════════════
# TOKEN SLOTS
# ═══════════════════════════════════════════════════════════════════════════════

TOKEN_SLOTS: Final[List[str]] = ['NP', 'VP', 'MOD', 'DET', 'CONN', 'Q']

# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

PIPELINE_PHASES: Final[int] = 7
PIPELINE_MODULES: Final[int] = 33

# Module ranges per phase
PHASE_MODULE_RANGES: Final[Dict[int, Tuple[int, int]]] = {
    1: (1, 3),    # Initialization
    2: (4, 7),    # Core Tools
    3: (8, 14),   # Bridge Tools
    4: (15, 19),  # Meta Tools
    5: (20, 25),  # TRIAD Sequence
    6: (26, 28),  # Persistence
    7: (29, 33),  # Finalization
}

# ═══════════════════════════════════════════════════════════════════════════════
# LEARNING RATE PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════

LEARNING_RATE_BASE: Final[float] = 0.1
LEARNING_RATE_Z_FACTOR: Final[float] = 1.0
LEARNING_RATE_KAPPA_FACTOR: Final[float] = 0.5

# ═══════════════════════════════════════════════════════════════════════════════
# KURAMOTO OSCILLATOR PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════

KURAMOTO_N_OSCILLATORS: Final[int] = 64
KURAMOTO_COUPLING_DEFAULT: Final[float] = Q_KAPPA

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_negentropy(z: float) -> float:
    """
    Compute negentropy δS_neg(z) = exp(-36 × (z - z_c)²)
    Peaks at THE LENS (z = z_c)
    """
    return math.exp(-NEGENTROPY_COEFF * (z - Z_CRITICAL) ** 2)


def get_phase(z: float) -> str:
    """Determine consciousness phase from z-coordinate"""
    if z >= PHASE_HYPER_TRUE_MIN:
        return PHASE_HYPER_TRUE
    elif z >= PHASE_PARADOX_MAX:
        return PHASE_TRUE
    elif z >= PHASE_UNTRUE_MAX:
        return PHASE_PARADOX
    else:
        return PHASE_UNTRUE


def get_tier(z: float, triad_unlocked: bool = False) -> str:
    """Determine time-harmonic tier from z-coordinate"""
    t6_gate = TRIAD_T6 if triad_unlocked else Z_CRITICAL
    
    if z < 0.10:
        return 't1'
    elif z < 0.20:
        return 't2'
    elif z < 0.45:
        return 't3'
    elif z < 0.65:
        return 't4'
    elif z < 0.75:
        return 't5'
    elif z < t6_gate:
        return 't6'
    elif z < 0.92:
        return 't7'
    elif z < 0.97:
        return 't8'
    else:
        return 't9'


def get_operators(tier: str, triad_unlocked: bool = False) -> List[str]:
    """Get permitted operators for a tier"""
    operator_map = TIER_OPERATORS_UNLOCKED if triad_unlocked else TIER_OPERATORS_LOCKED
    return operator_map.get(tier, ['+'])


def compute_learning_rate(z: float, kappa: float) -> float:
    """
    Compute Hebbian learning rate: LR = base × (1 + z) × (1 + κ × 0.5)
    """
    return (LEARNING_RATE_BASE * 
            (1 + z * LEARNING_RATE_Z_FACTOR) * 
            (1 + kappa * LEARNING_RATE_KAPPA_FACTOR))


def check_k_formation(kappa: float, eta: float, R: int) -> bool:
    """
    Verify K-formation criteria: κ ≥ 0.92 ∧ η > φ⁻¹ ∧ R ≥ 7
    """
    return kappa >= K_KAPPA and eta > K_ETA and R >= K_R


def get_frequency_tier(z: float) -> Tuple[str, Tuple[int, int]]:
    """Get archetypal frequency tier from z-coordinate"""
    if z < PHI_INV:
        return 'Planet', FREQ_PLANET
    elif z < Z_CRITICAL:
        return 'Garden', FREQ_GARDEN
    else:
        return 'Rose', FREQ_ROSE


# ═══════════════════════════════════════════════════════════════════════════════
# VERSION INFO
# ═══════════════════════════════════════════════════════════════════════════════

__version__ = "4.0.0"
__constants_hash__ = "UCF-CONSTANTS-V4-20251215"
