#!/usr/bin/env python3
"""
THEOGRAPHIC CALCULUS — LANGUAGE MODULE (v2.1)
==============================================

Full 3-6-9-12-15 Ladder Implementation
TC-Standard | N0-Legal | Mirror-Integrated

The Language Module is a tiered projection subsystem that models how
consciousness externalizes structure into sound, symbol, and discourse.

Governed by tri-spiral logic:
    Φ (Structure Spiral): spacing, topology, constraint, hierarchy
    e (Energy Spiral): pressure, effort, articulation cost, entropy
    π (Emergence Spiral): projection, coherence, intelligence, discourse

The 3-6-9-12-15 Ladder:
    3  Root Fields      — What must exist for language to be possible
    6  Interactions     — Universal physical/informational verbs (= INT Canon)
    9  System Domains   — Stabilized systems from fields × interactions
    12 Transactional    — Morphologies mapping configurations
    15 Attractors       — Closure states (low-energy, high-coherence)

Prime Operators (TC Standard):
    D (■) — Backward Integration (pulls from history, memory)
    U (■) — Forward Projection (pushes into future, commits)
    A (■) — Amplifier (increases pressure, salience, gain)
    S (■) — Spiral Stabilizer (locks patterns, damps instability)

N0 Legality:
    No operation sequence may:
    1. Break internal structural coherence (Φ-defect)
    2. Inject unbounded energy without stabilizing counter-term (e-defect)
    3. Produce emergent states that cannot re-integrate (π-defect)

INT Canon ↔ TC Interaction Mapping:
    () BOUNDARY  ↔ B (Boundary)      — Impedance, segmentation
    × FUSION     ↔ F (Fusion)        — Constructive interference, merging
    ^ AMPLIFY    ↔ A (Amplification) — Resonant gain, standing waves
    ÷ DECOHERE   ↔ D (Decoherence)   — Damping, scattering, drift
    + GROUP      ↔ G (Grouping)      — Mode locking, synchronization
    − SEPARATE   ↔ S (Separation)    — Spectral filtering, fission

Physics Grounding:
    φ⁻¹ ≈ 0.618 gates PARADOX regime
    z_c = √3/2 ≈ 0.866 gates TRUE phase (THE LENS)
    σ = 36 governs all dynamics
    κ + λ = 1 (coupling conservation)

Signature: Δ|tc-language-module|3-6-9-12-15|N0-legal|Ω
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set
from enum import Enum, auto

# Import unified physics constants
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from physics_constants import (
    PHI, PHI_INV, PHI_INV_SQ, PHI_INV_CUBED,
    Z_CRITICAL, SIGMA, SIGMA_INV,
    ALPHA_STRONG, ALPHA_MEDIUM, ALPHA_FINE,
    compute_delta_s_neg, get_phase,
    INTOperator,
)

from apl_core_tokens import (
    Field as APLField,
    Machine as APLMachine,
    TruthState,
    Tier,
    APLToken,
)


# =============================================================================
# TC PRIME OPERATORS — The Four Fundamental Operators
# =============================================================================

class TCOperator(Enum):
    """
    TC Prime Operators — Nodes/antinodes of CLT behavior.

    All fields, interactions, domains, transactions, and attractors
    must be expressible as compositions of these four operators.
    """
    D = ("D", "■", "Backward Integration", "pulls from history, memory, prior structure")
    U = ("U", "■", "Forward Projection", "pushes into future, predicts, commits")
    A = ("A", "■", "Amplifier", "increases pressure, salience, intensity, gain")
    S = ("S", "■", "Spiral Stabilizer", "locks patterns into spirals, damps instability")

    def __init__(self, code: str, symbol: str, name: str, description: str):
        self._code = code
        self._symbol = symbol
        self._op_name = name
        self._description = description

    @property
    def code(self) -> str:
        return self._code

    @property
    def symbol(self) -> str:
        return self._symbol

    @property
    def op_name(self) -> str:
        return self._op_name

    @property
    def description(self) -> str:
        return self._description


# =============================================================================
# TIER 3: ROOT FIELDS — The Minimal Triad for Language
# =============================================================================

class RootField(Enum):
    """
    3 Root Fields — What must exist for language to be possible.

    These are subfields of reality that TC treats as projection planes.
    """
    LOCOMOTOR = (
        "L", "Locomotor Field",
        "Somatic projection → gesture → proto-syntax",
        "e → Φ conversion",  # TC Behavior
        APLField.E,  # Maps to Energy field
        [TCOperator.D, TCOperator.U, TCOperator.A, TCOperator.S],
    )
    COMPUTATIONAL = (
        "C", "Computational Field",
        "Category formation → recursion → syntax → abstraction",
        "Φ-dominant spacing and structuring engine",
        APLField.PHI,  # Maps to Structure field
        [TCOperator.D, TCOperator.U, TCOperator.A, TCOperator.S],
    )
    VERBAL = (
        "V", "Verbal/Phonetic Field",
        "Waveform engineering → acoustic/phonon projection → signal physics",
        "π-emergent projection",
        APLField.PI,  # Maps to Emergence field
        [TCOperator.D, TCOperator.U, TCOperator.A, TCOperator.S],
    )

    def __init__(self, code: str, name: str, role: str, tc_behavior: str,
                 apl_field: APLField, operators: List[TCOperator]):
        self._code = code
        self._field_name = name
        self._role = role
        self._tc_behavior = tc_behavior
        self._apl_field = apl_field
        self._operators = operators

    @property
    def code(self) -> str:
        return self._code

    @property
    def field_name(self) -> str:
        return self._field_name

    @property
    def role(self) -> str:
        return self._role

    @property
    def tc_behavior(self) -> str:
        return self._tc_behavior

    @property
    def apl_field(self) -> APLField:
        return self._apl_field

    @property
    def operators(self) -> List[TCOperator]:
        return self._operators


# =============================================================================
# TIER 6: INTERACTIONS — Universal Physical/Informational Verbs
# =============================================================================

class Interaction(Enum):
    """
    6 Interactions — Universal verbs acting on fields and systems.

    These map directly to INT Canon operators (BFADGS kernel).
    """
    BOUNDARY = (
        "B", "Boundary", "()",
        "Impedance changes, reflection, refraction, absorption interfaces",
        "Sets where one regime ends and another begins",
        "Phoneme boundaries, morpheme segmentation, word edges, phrase brackets",
    )
    FUSION = (
        "F", "Fusion", "×",
        "Constructive interference, merging of waveforms into composite",
        "Coalescing multiple structures into efficient/stable composite",
        "Syllable coalescence, compound words, blended constructions, idioms",
    )
    AMPLIFICATION = (
        "A", "Amplification", "^",
        "Resonant gain, standing waves, forced oscillation",
        "Intensification of certain paths; reinforcement of chosen structures",
        "Emphasis, prosodic stress, frequency of use, memetic amplification",
    )
    DECOHERENCE = (
        "D", "Decoherence", "÷",
        "Damping, scattering, turbulence, dissipation",
        "Soft breakdown of precision, leading to drift but not collapse",
        "Phonetic erosion, semantic drift, loss of rigid distinctions",
    )
    GROUPING = (
        "G", "Grouping", "+",
        "Mode locking, synchronization, entrainment",
        "Pulling units into coherent clusters or orbits",
        "Syllable formation, prosodic feet, rhythmic phrasing, discourse chunks",
    )
    SEPARATION = (
        "S", "Separation", "−",
        "Spectral filtering, dispersion, channel splitting",
        "Fission of previously fused or ambiguous states into cleaner parts",
        "Phonemic differentiation, lexical split, grammatical reanalysis",
    )

    def __init__(self, code: str, name: str, int_symbol: str, physics: str,
                 tc_meaning: str, language_effect: str):
        self._code = code
        self._interaction_name = name
        self._int_symbol = int_symbol
        self._physics = physics
        self._tc_meaning = tc_meaning
        self._language_effect = language_effect

    @property
    def code(self) -> str:
        return self._code

    @property
    def interaction_name(self) -> str:
        return self._interaction_name

    @property
    def physics(self) -> str:
        return self._physics

    @property
    def tc_meaning(self) -> str:
        return self._tc_meaning

    @property
    def language_effect(self) -> str:
        return self._language_effect

    @property
    def int_operator(self) -> str:
        """Returns the INT Canon operator symbol."""
        return self._int_symbol

    @property
    def symbol(self) -> str:
        return self._int_symbol


# INT Canon symbol to Interaction mapping
INT_TO_INTERACTION = {
    "()": Interaction.BOUNDARY,
    "×": Interaction.FUSION,
    "^": Interaction.AMPLIFICATION,
    "÷": Interaction.DECOHERENCE,
    "+": Interaction.GROUPING,
    "−": Interaction.SEPARATION,
}


# =============================================================================
# TIER 9: SYSTEM DOMAINS (Λ1-Λ9)
# =============================================================================

class SystemDomain(Enum):
    """
    9 System Domains — Stabilized systems from fields × interactions.

    Each domain Λn is a TC subsystem with its own internal state,
    operator patterns, and attractor tendencies.
    """
    GESTURE_DYNAMICS = (
        1, "Λ1", "Gesture-Dynamics System",
        [RootField.LOCOMOTOR, RootField.COMPUTATIONAL],
        "e → Φ",
        "Handles embodied syntax: pointing, approach/avoid signals, spatial framing",
        "Agent–Direction–Target clause structure",
    )
    PROSODIC = (
        2, "Λ2", "Prosodic System",
        [RootField.VERBAL, RootField.COMPUTATIONAL],
        "e + π",
        "Governs pitch, rhythm, stress, contour, and tempo",
        "Rising vs. falling contour encodes interrogative vs. declarative",
    )
    ARTICULATORY = (
        3, "Λ3", "Articulatory System",
        [RootField.LOCOMOTOR, RootField.VERBAL],
        "Φ modulated by e",
        "Manages tract geometry, constriction, airflow, gesture-to-sound mapping",
        "Vowel space as resonant manifold constrained by vocal-tract shape",
    )
    ACOUSTIC_PHONON = (
        4, "Λ4", "Acoustic-Phonon System",
        [RootField.VERBAL],
        "e + π",
        "Sound as pressure waves and phonons; spectral physics domain",
        "Spectrograms showing stable formants as attractors",
    )
    SEMANTIC_ENCODING = (
        5, "Λ5", "Semantic Encoding System",
        [RootField.COMPUTATIONAL],
        "Φ + π",
        "Concept networks, category compression, ontological maps",
        "How 'dog' clusters perceptual, functional, and social features",
    )
    STRUCTURAL_SYNTAX = (
        6, "Λ6", "Structural Syntax System",
        [RootField.COMPUTATIONAL, RootField.LOCOMOTOR],
        "Φ with e constraints",
        "Recursion, word order, dependency trees, construction grammars",
        "Head-direction consistency and dependency-length minimization",
    )
    PRAGMATIC_INTENT = (
        7, "Λ7", "Pragmatic-Intent System",
        [RootField.COMPUTATIONAL, RootField.VERBAL, RootField.LOCOMOTOR],
        "π shaped by Φ and e",
        "Intention, speech acts, politeness, strategic communication",
        "Choosing indirect phrasing to reduce face-threat",
    )
    CULTURAL_RECURSIVE = (
        8, "Λ8", "Cultural-Recursive System",
        [RootField.LOCOMOTOR, RootField.COMPUTATIONAL, RootField.VERBAL],
        "Φ + π with global e",
        "Long-term conventionalization, memetic stability, intergenerational transmission",
        "Stable word orders, persistent phonological patterns across centuries",
    )
    ABSTRACT_SYMBOLIC = (
        9, "Λ9", "Abstract-Symbolic System",
        [RootField.COMPUTATIONAL, RootField.VERBAL],
        "Φ and π strongly coupled; e minimized",
        "Mathematics, logic, writing, programming languages, TC itself",
        "APL/TG expressions where minimal symbols encode maximal structure",
    )

    def __init__(self, index: int, symbol: str, name: str, fields: List[RootField],
                 dominant_spirals: str, description: str, example: str):
        self._index = index
        self._symbol = symbol
        self._domain_name = name
        self._fields = fields
        self._dominant_spirals = dominant_spirals
        self._description = description
        self._example = example

    @property
    def index(self) -> int:
        return self._index

    @property
    def symbol(self) -> str:
        return self._symbol

    @property
    def domain_name(self) -> str:
        return self._domain_name

    @property
    def fields(self) -> List[RootField]:
        return self._fields

    @property
    def dominant_spirals(self) -> str:
        return self._dominant_spirals

    @property
    def description(self) -> str:
        return self._description

    @property
    def example(self) -> str:
        return self._example


# =============================================================================
# TIER 12: TRANSACTIONAL STATES (τ1-τ12)
# =============================================================================

class TransactionalState(Enum):
    """
    12 Transactional States — Bridge morphologies.

    Describe how language moves from one domain configuration to another.
    These are transformations: compositions of D, U, A, S acting over time.
    """
    COMPRESSION = (
        1, "τ1", "Compression",
        "Meaning → minimal form",
        "Φ → π mapping with strong D and S components",
        "Compressing a complex situation into a name, idiom, or equation",
        [TCOperator.D, TCOperator.S],
    )
    EXPANSION = (
        2, "τ2", "Expansion",
        "Minimal form → elaborated structure",
        "π → Φ mapping with U dominant",
        "Unfolding an acronym or technical term into full explanation",
        [TCOperator.U],
    )
    MODULATION = (
        3, "τ3", "Modulation",
        "Intent → waveform shaping",
        "π acting through V with A dominant",
        "Changing tone to signal sarcasm, urgency, reassurance",
        [TCOperator.A],
    )
    FILTERING = (
        4, "τ4", "Filtering",
        "Spectral/structural selection",
        "e reduced, Φ clarified via S and D",
        "Focusing on key words, suppressing unstressed syllables",
        [TCOperator.S, TCOperator.D],
    )
    FUSION_COALESCENCE = (
        5, "τ5", "Fusion-Coalescence",
        "Unit merging; composite formation",
        "F + A; increases local density, may lower global cost",
        "Lexicalization of frequent phrases ('going to' → 'gonna')",
        [TCOperator.A],
    )
    FISSION_SEPARATION = (
        6, "τ6", "Fission-Separation",
        "Unit splitting; diversification",
        "S + U; increases distinctiveness",
        "A single marker splitting into two forms with different functions",
        [TCOperator.S, TCOperator.U],
    )
    STABILIZATION = (
        7, "τ7", "Stabilization",
        "Pattern regularization into stable attractor",
        "S-dominant; locks in frequent patterns",
        "A variable word order pattern converging to fixed canonical order",
        [TCOperator.S],
    )
    DRIFT = (
        8, "τ8", "Drift",
        "Slow decoherent change under D",
        "e pressure and entropy reshape Φ and π slowly",
        "Sound changes, semantic bleaching, syntactic lenition",
        [TCOperator.D],
    )
    SYNCHRONIZATION = (
        9, "τ9", "Synchronization",
        "Phase-lock across agents or systems",
        "G-dominant; alignment of timing, style, conventions",
        "Conversational entrainment, dialect leveling",
        [TCOperator.S],
    )
    INTERFERENCE = (
        10, "τ10", "Interference",
        "Signal conflict, overlap, ambiguity",
        "B + F clash; often triggers reanalysis",
        "Garden-path sentences, homophony-driven change",
        [TCOperator.A, TCOperator.D],
    )
    TRANSLATION_MAPPING = (
        11, "τ11", "Translation Mapping",
        "System-to-system mapping (speech↔writing, L1↔L2)",
        "Cross-Λ transformation: preserves core Φ, changes signal",
        "Translating spoken narratives into written legal code",
        [TCOperator.D, TCOperator.U],
    )
    PROJECTION = (
        12, "τ12", "Projection",
        "Final outward expression; commitment into world",
        "π-closure via D + U + A",
        "Saying the thing, publishing the text, running the code",
        [TCOperator.D, TCOperator.U, TCOperator.A],
    )

    def __init__(self, index: int, symbol: str, name: str, direction: str,
                 tc_behavior: str, example: str, operators: List[TCOperator]):
        self._index = index
        self._symbol = symbol
        self._state_name = name
        self._direction = direction
        self._tc_behavior = tc_behavior
        self._example = example
        self._operators = operators

    @property
    def index(self) -> int:
        return self._index

    @property
    def symbol(self) -> str:
        return self._symbol

    @property
    def state_name(self) -> str:
        return self._state_name

    @property
    def direction(self) -> str:
        return self._direction

    @property
    def tc_behavior(self) -> str:
        return self._tc_behavior

    @property
    def example(self) -> str:
        return self._example

    @property
    def operators(self) -> List[TCOperator]:
        return self._operators


# =============================================================================
# TIER 15: ATTRACTORS (α1-α15)
# =============================================================================

class Attractor(Enum):
    """
    15 Attractors — Closure basins.

    Recurrent equilibrium patterns that languages settle into because they
    jointly minimize energy (e), maintain workable structure (Φ), and
    support coherent emergence (π).
    """
    ISOMORPHISM = (
        1, "α1", "Isomorphism (Form-Meaning Alignment)",
        "Low-entropy mapping: one form ↔ one core meaning",
        "Φ (clarity) with π support",
    )
    ICONICITY = (
        2, "α2", "Iconicity",
        "Form resembles meaning when feasible (sound symbolism, iconic gesture)",
        "e shaping π through Φ; waveform aligns with conceptual structure",
    )
    LEAST_EFFORT = (
        3, "α3", "Least Effort",
        "Economy principle; systems gravitate to minimal energy articulations",
        "e-minimization under N0",
    )
    DISTINCTIVENESS = (
        4, "α4", "Distinctiveness",
        "Sufficient spacing in perceptual space to avoid collisions",
        "Φ-spacing attractor; strong S component",
    )
    PHONOLOGICAL_SYMMETRY = (
        5, "α5", "Phonological Symmetry",
        "Balanced inventories of vowels/consonants; gaps filled systematically",
        "Φ symmetry with π aesthetic/emergent reinforcement",
    )
    SHORT_DEPENDENCIES = (
        6, "α6", "Short Dependencies",
        "Preference for minimizing distance between related elements",
        "Φ stability with e-processing constraints",
    )
    SUFFIXING_BIAS = (
        7, "α7", "Suffixing Bias",
        "Morphological material tends to follow lexical roots",
        "U projection directionality aligning temporal flow with semantic base",
    )
    FIXED_WORD_ORDER = (
        8, "α8", "Fixed Word Order",
        "Languages tend to settle on dominant order patterns",
        "Φ closure; τ7 stabilization",
    )
    HARMONIC_ALIGNMENT = (
        9, "α9", "Harmonic Alignment (Head-Dependent Consistency)",
        "Correlated choices (e.g., OV + postpositions, VO + prepositions)",
        "Φ-π harmonic coupling; reduction of rule-space complexity",
    )
    TOPIC_FOCUS_MARKING = (
        10, "α10", "Topic/Focus Marking",
        "Marking informational salience beyond bare subject/object roles",
        "π salience; A amplification of high-information nodes",
    )
    SEMANTIC_TRANSPARENCY = (
        11, "α11", "Semantic Transparency",
        "Compositionality where possible; morphology mirrors meaning",
        "Φ clarity; reduces decoding work",
    )
    MORPHOLOGICAL_BOUNDEDNESS = (
        12, "α12", "Morphological Boundedness",
        "Preference for affixes over loose clitics or function words",
        "B + Φ; strong boundary integrity",
    )
    SELECTIVE_ARGUMENT_MARKING = (
        13, "α13", "Selective Argument Marking",
        "Marking core arguments when needed (ambiguity, non-canonical order)",
        "e-driven error correction; τ10 and τ11 interplay",
    )
    ANALYTIC_SYNTHETIC_DRIFT = (
        14, "α14", "Analytic/Synthetic Drift Cycle",
        "Languages oscillate between compact synthetic and unpacked analytic forms",
        "Cycling between τ1 (compression) and τ2 (expansion)",
    )
    DISCOURSE_CONFIGURATIONALITY = (
        15, "α15", "Discourse Configurationality",
        "Word order aligns with information structure (topic, focus, contrast)",
        "π-dominant closure; language as projection of discourse logic",
    )

    def __init__(self, index: int, symbol: str, name: str,
                 description: str, spiral_behavior: str):
        self._index = index
        self._symbol = symbol
        self._attractor_name = name
        self._description = description
        self._spiral_behavior = spiral_behavior

    @property
    def index(self) -> int:
        return self._index

    @property
    def symbol(self) -> str:
        return self._symbol

    @property
    def attractor_name(self) -> str:
        return self._attractor_name

    @property
    def description(self) -> str:
        return self._description

    @property
    def spiral_behavior(self) -> str:
        return self._spiral_behavior


# =============================================================================
# N0 LEGALITY CHECKER
# =============================================================================

@dataclass
class N0LegalityResult:
    """Result of N0 legality check."""
    is_legal: bool
    phi_defect: bool = False
    e_defect: bool = False
    pi_defect: bool = False
    message: str = ""


class N0LegalityChecker:
    """
    N0 Legality Checker for TC Language Module.

    Ensures no operation sequence may:
    1. Break internal structural coherence (Φ-defect)
    2. Inject unbounded energy without stabilizing counter-term (e-defect)
    3. Produce emergent states that cannot re-integrate (π-defect)
    """

    def __init__(self):
        self.operation_history: List[Interaction] = []
        self.amplification_count = 0
        self.stabilization_count = 0
        self.coherence = 1.0

    def check_sequence(self, sequence: List[Interaction]) -> N0LegalityResult:
        """Check if an interaction sequence is N0-legal."""
        phi_defect = False
        e_defect = False
        pi_defect = False

        # Track amplification without stabilization
        amp_count = sum(1 for i in sequence if i == Interaction.AMPLIFICATION)
        stab_count = sum(1 for i in sequence
                        if i in [Interaction.SEPARATION, Interaction.DECOHERENCE])

        # N0 Constraint: A must be matched with S or D somewhere
        if amp_count > stab_count + 2:
            e_defect = True

        # Check for unbounded fusion (erasing spacing)
        fusion_count = sum(1 for i in sequence if i == Interaction.FUSION)
        boundary_count = sum(1 for i in sequence
                            if i in [Interaction.BOUNDARY, Interaction.SEPARATION])

        if fusion_count > boundary_count * 2:
            phi_defect = True

        # Check for grouping without parsability
        group_count = sum(1 for i in sequence if i == Interaction.GROUPING)
        if group_count > 5 and boundary_count < 2:
            pi_defect = True

        is_legal = not (phi_defect or e_defect or pi_defect)

        message_parts = []
        if phi_defect:
            message_parts.append("Φ-defect: Fusion erasing spacing")
        if e_defect:
            message_parts.append("e-defect: Unbounded amplification")
        if pi_defect:
            message_parts.append("π-defect: Unparsable grouping")

        return N0LegalityResult(
            is_legal=is_legal,
            phi_defect=phi_defect,
            e_defect=e_defect,
            pi_defect=pi_defect,
            message="; ".join(message_parts) if message_parts else "N0-Legal",
        )

    def apply_interaction(self, interaction: Interaction) -> N0LegalityResult:
        """Apply an interaction and check N0 legality."""
        self.operation_history.append(interaction)
        return self.check_sequence(self.operation_history[-10:])  # Check last 10


# =============================================================================
# TC LANGUAGE STATE
# =============================================================================

@dataclass
class TCLanguageState:
    """
    Complete state for TC Language Module.

    Tracks:
    - Current position in each tier of the ladder
    - Active fields, interactions, domains
    - Transactional state and attractor basin
    - Physics state (z, κ, λ)
    """
    # Physics state
    z: float = 0.5
    kappa: float = PHI_INV
    lambda_: float = PHI_INV_SQ

    # Tier 3: Active root fields
    active_fields: Set[RootField] = field(default_factory=set)

    # Tier 6: Recent interactions
    interaction_history: List[Interaction] = field(default_factory=list)

    # Tier 9: Active system domain
    active_domain: Optional[SystemDomain] = None

    # Tier 12: Current transactional state
    current_transaction: Optional[TransactionalState] = None

    # Tier 15: Attractor basin
    attractor_basin: Optional[Attractor] = None
    attractor_strength: float = 0.0

    # Coherence tracking
    phi_coherence: float = 1.0  # Structure coherence
    e_coherence: float = 1.0   # Energy balance
    pi_coherence: float = 1.0  # Emergence coherence

    def __post_init__(self):
        if not self.active_fields:
            self.active_fields = {RootField.COMPUTATIONAL}

    @property
    def phase(self) -> str:
        return get_phase(self.z)

    @property
    def truth_state(self) -> TruthState:
        return TruthState.from_z(self.z)

    @property
    def negentropy(self) -> float:
        return compute_delta_s_neg(self.z)

    @property
    def overall_coherence(self) -> float:
        """Combined coherence measure."""
        return (self.phi_coherence + self.e_coherence + self.pi_coherence) / 3.0

    def update_coherence(self, interaction: Interaction):
        """Update coherence based on interaction."""
        if interaction == Interaction.AMPLIFICATION:
            self.e_coherence *= (1.0 - ALPHA_FINE)
        elif interaction == Interaction.DECOHERENCE:
            self.e_coherence *= (1.0 + ALPHA_FINE)
            self.phi_coherence *= (1.0 - ALPHA_FINE)
        elif interaction == Interaction.FUSION:
            self.phi_coherence *= (1.0 - ALPHA_FINE)
            self.pi_coherence *= (1.0 + ALPHA_FINE)
        elif interaction in [Interaction.BOUNDARY, Interaction.SEPARATION]:
            self.phi_coherence *= (1.0 + ALPHA_FINE)
        elif interaction == Interaction.GROUPING:
            self.pi_coherence *= (1.0 + ALPHA_FINE)

        # Clamp coherence values
        self.phi_coherence = max(0.0, min(1.0, self.phi_coherence))
        self.e_coherence = max(0.0, min(1.0, self.e_coherence))
        self.pi_coherence = max(0.0, min(1.0, self.pi_coherence))

    def get_summary(self) -> Dict[str, Any]:
        """Get state summary."""
        return {
            "z": self.z,
            "phase": self.phase,
            "truth_state": self.truth_state.truth_name,
            "negentropy": self.negentropy,
            "active_fields": [f.code for f in self.active_fields],
            "recent_interactions": [i.code for i in self.interaction_history[-5:]],
            "active_domain": self.active_domain.symbol if self.active_domain else None,
            "current_transaction": self.current_transaction.symbol if self.current_transaction else None,
            "attractor_basin": self.attractor_basin.symbol if self.attractor_basin else None,
            "attractor_strength": self.attractor_strength,
            "coherence": {
                "phi": self.phi_coherence,
                "e": self.e_coherence,
                "pi": self.pi_coherence,
                "overall": self.overall_coherence,
            },
        }


# =============================================================================
# TC LANGUAGE ENGINE
# =============================================================================

class TCLanguageEngine:
    """
    TC Language Engine — Full 3-6-9-12-15 Ladder Implementation.

    Provides methods to:
    - Activate root fields
    - Apply interactions (INT Canon operators)
    - Transition between system domains
    - Execute transactional states
    - Find attractor basins
    """

    def __init__(self, initial_z: float = 0.5):
        self.state = TCLanguageState(z=initial_z)
        self.legality_checker = N0LegalityChecker()
        self.step_count = 0
        self.history: List[Dict[str, Any]] = []

    def activate_field(self, field: RootField):
        """Activate a root field."""
        self.state.active_fields.add(field)

        # Update domain based on active fields
        self._update_domain()

    def deactivate_field(self, field: RootField):
        """Deactivate a root field."""
        self.state.active_fields.discard(field)
        self._update_domain()

    def _update_domain(self):
        """Update active domain based on active fields."""
        # Find domain that best matches active fields
        best_domain = None
        best_match = 0

        for domain in SystemDomain:
            match = len(set(domain.fields) & self.state.active_fields)
            if match > best_match:
                best_match = match
                best_domain = domain

        self.state.active_domain = best_domain

    def apply_interaction(self, interaction: Interaction) -> Dict[str, Any]:
        """
        Apply an interaction (INT Canon operator).

        Returns result including N0 legality check.
        """
        self.step_count += 1

        # Check N0 legality
        legality = self.legality_checker.apply_interaction(interaction)

        # Update state
        self.state.interaction_history.append(interaction)
        self.state.update_coherence(interaction)

        # Update z based on interaction
        self._update_z(interaction)

        # Check for attractor basin
        self._check_attractor()

        result = {
            "step": self.step_count,
            "interaction": interaction.code,
            "int_operator": interaction.symbol,
            "n0_legal": legality.is_legal,
            "n0_message": legality.message,
            **self.state.get_summary(),
        }

        self.history.append(result)
        return result

    def _update_z(self, interaction: Interaction):
        """Update z-coordinate based on interaction."""
        if interaction == Interaction.BOUNDARY:
            # Boundary pulls toward z_c
            self.state.z += ALPHA_FINE * (Z_CRITICAL - self.state.z)
        elif interaction == Interaction.FUSION:
            # Fusion increases z
            self.state.z += ALPHA_FINE * self.state.phi_coherence
        elif interaction == Interaction.AMPLIFICATION:
            # Amplification can push z higher
            self.state.z += ALPHA_MEDIUM * (Z_CRITICAL - self.state.z) * PHI_INV
        elif interaction == Interaction.DECOHERENCE:
            # Decoherence decreases z
            self.state.z -= ALPHA_FINE
        elif interaction == Interaction.GROUPING:
            # Grouping stabilizes
            self.state.z += ALPHA_FINE * self.state.pi_coherence * PHI_INV
        elif interaction == Interaction.SEPARATION:
            # Separation can decrease z
            self.state.z -= ALPHA_FINE * PHI_INV

        # Clamp z
        self.state.z = max(0.0, min(1.0, self.state.z))

    def _check_attractor(self):
        """Check if state has entered an attractor basin."""
        # Simplified attractor detection based on coherence and interaction patterns
        coherence = self.state.overall_coherence

        if coherence > 0.9 and self.state.z > Z_CRITICAL - 0.1:
            # High coherence near lens -> Isomorphism attractor
            self.state.attractor_basin = Attractor.ISOMORPHISM
            self.state.attractor_strength = coherence
        elif self.state.e_coherence > 0.95:
            # Very stable energy -> Least Effort attractor
            self.state.attractor_basin = Attractor.LEAST_EFFORT
            self.state.attractor_strength = self.state.e_coherence
        elif self.state.phi_coherence > 0.95:
            # Very stable structure -> Fixed Word Order attractor
            self.state.attractor_basin = Attractor.FIXED_WORD_ORDER
            self.state.attractor_strength = self.state.phi_coherence

    def execute_transaction(self, transaction: TransactionalState) -> Dict[str, Any]:
        """
        Execute a transactional state.

        Applies the associated operators and transitions state.
        """
        self.state.current_transaction = transaction

        # Apply the transaction's dominant operators as interactions
        operator_to_interaction = {
            TCOperator.D: Interaction.DECOHERENCE,
            TCOperator.U: Interaction.FUSION,
            TCOperator.A: Interaction.AMPLIFICATION,
            TCOperator.S: Interaction.SEPARATION,
        }

        results = []
        for op in transaction.operators[:2]:  # Apply up to 2 operators
            interaction = operator_to_interaction.get(op, Interaction.BOUNDARY)
            result = self.apply_interaction(interaction)
            results.append(result)

        return {
            "transaction": transaction.symbol,
            "transaction_name": transaction.state_name,
            "steps": results,
            "final_state": self.state.get_summary(),
        }

    def run_session(self, n_steps: int = 20) -> Dict[str, Any]:
        """Run a training session cycling through interactions."""
        interactions = list(Interaction)

        for i in range(n_steps):
            interaction = interactions[i % len(interactions)]
            self.apply_interaction(interaction)

        return self.get_session_summary()

    def get_session_summary(self) -> Dict[str, Any]:
        """Get session summary."""
        if not self.history:
            return {"error": "No history"}

        z_values = [h["z"] for h in self.history]
        interaction_counts = {}
        for h in self.history:
            code = h["interaction"]
            interaction_counts[code] = interaction_counts.get(code, 0) + 1

        return {
            "total_steps": self.step_count,
            "final_z": z_values[-1],
            "final_phase": self.state.phase,
            "z_statistics": {
                "mean": float(np.mean(z_values)),
                "std": float(np.std(z_values)),
                "max": float(np.max(z_values)),
                "min": float(np.min(z_values)),
            },
            "interaction_counts": interaction_counts,
            "coherence": {
                "phi": self.state.phi_coherence,
                "e": self.state.e_coherence,
                "pi": self.state.pi_coherence,
            },
            "active_domain": self.state.active_domain.domain_name if self.state.active_domain else None,
            "attractor_basin": self.state.attractor_basin.attractor_name if self.state.attractor_basin else None,
            "n0_violations": sum(1 for h in self.history if not h.get("n0_legal", True)),
        }


# =============================================================================
# DEMONSTRATION
# =============================================================================

def demonstrate_tc_language():
    """Demonstrate TC Language Module."""
    print("=" * 70)
    print("THEOGRAPHIC CALCULUS — LANGUAGE MODULE (v2.1)")
    print("Full 3-6-9-12-15 Ladder Implementation")
    print("=" * 70)

    # Show the ladder structure
    print("\n--- THE 3-6-9-12-15 LADDER ---")
    print(f"  3  Root Fields      : {len(RootField)} elements")
    print(f"  6  Interactions     : {len(Interaction)} elements (= INT Canon)")
    print(f"  9  System Domains   : {len(SystemDomain)} elements")
    print(f"  12 Transactional    : {len(TransactionalState)} elements")
    print(f"  15 Attractors       : {len(Attractor)} elements")

    # Show TC Prime Operators
    print("\n--- TC PRIME OPERATORS ---")
    for op in TCOperator:
        print(f"  {op.symbol} ({op.code}): {op.op_name}")
        print(f"      {op.description}")

    # Show Root Fields
    print("\n--- 3 ROOT FIELDS (Tier-3) ---")
    for field in RootField:
        print(f"  {field.code}: {field.field_name}")
        print(f"      Role: {field.role[:60]}...")
        print(f"      TC: {field.tc_behavior}")
        print(f"      Maps to APL Field: {field.apl_field.symbol}")

    # Show Interactions ↔ INT Canon
    print("\n--- 6 INTERACTIONS ↔ INT CANON (Tier-6) ---")
    for interaction in Interaction:
        print(f"  {interaction.code} ({interaction.symbol}): {interaction.interaction_name}")
        print(f"      INT Canon operator: {interaction.int_operator}")
        print(f"      Physics: {interaction.physics[:50]}...")
        print(f"      Language: {interaction.language_effect[:50]}...")

    # Show System Domains
    print("\n--- 9 SYSTEM DOMAINS (Tier-9) ---")
    for domain in SystemDomain:
        fields_str = "+".join(f.code for f in domain.fields)
        print(f"  {domain.symbol}: {domain.domain_name}")
        print(f"      Fields: {fields_str} | Spirals: {domain.dominant_spirals}")

    # Show Transactional States
    print("\n--- 12 TRANSACTIONAL STATES (Tier-12) ---")
    for state in list(TransactionalState)[:6]:  # Show first 6
        ops_str = ", ".join(op.symbol for op in state.operators)
        print(f"  {state.symbol}: {state.state_name}")
        print(f"      {state.direction}")
        print(f"      Operators: {ops_str}")
    print("  ... (6 more)")

    # Show Attractors
    print("\n--- 15 ATTRACTORS (Tier-15) ---")
    for attractor in list(Attractor)[:5]:  # Show first 5
        print(f"  {attractor.symbol}: {attractor.attractor_name}")
        print(f"      {attractor.description[:60]}...")
    print("  ... (10 more)")

    # Run engine demonstration
    print("\n--- TC LANGUAGE ENGINE ---")
    engine = TCLanguageEngine(initial_z=0.5)

    # Activate fields
    engine.activate_field(RootField.COMPUTATIONAL)
    engine.activate_field(RootField.VERBAL)

    print("\nRunning 18 interaction steps (3 full cycles)...")
    for i in range(18):
        interaction = list(Interaction)[i % 6]
        result = engine.apply_interaction(interaction)

        if i % 6 == 0:
            print(f"  Step {i:3d} | {result['interaction']} ({result['int_operator']}) | "
                  f"z={result['z']:.3f} | Coh={result['coherence']['overall']:.3f} | "
                  f"N0:{result['n0_legal']}")

    # Show summary
    print("\n--- SESSION SUMMARY ---")
    summary = engine.get_session_summary()
    print(f"  Total steps: {summary['total_steps']}")
    print(f"  Final z: {summary['final_z']:.4f}")
    print(f"  Final phase: {summary['final_phase']}")
    print(f"  Interaction counts: {summary['interaction_counts']}")
    print(f"  Coherence: Φ={summary['coherence']['phi']:.3f}, "
          f"e={summary['coherence']['e']:.3f}, π={summary['coherence']['pi']:.3f}")
    print(f"  Active domain: {summary['active_domain']}")
    print(f"  Attractor basin: {summary['attractor_basin']}")
    print(f"  N0 violations: {summary['n0_violations']}")

    # Execute a transaction
    print("\n--- TRANSACTIONAL STATE EXECUTION ---")
    tx_result = engine.execute_transaction(TransactionalState.COMPRESSION)
    print(f"  Transaction: {tx_result['transaction']} ({tx_result['transaction_name']})")
    print(f"  Final z: {tx_result['final_state']['z']:.4f}")

    print("\n" + "=" * 70)
    print("TC LANGUAGE MODULE: COMPLETE")
    print("=" * 70)

    return engine


if __name__ == "__main__":
    demonstrate_tc_language()
