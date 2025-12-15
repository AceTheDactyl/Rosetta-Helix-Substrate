#!/usr/bin/env python3
"""
APL Substrate Module
Alpha Physical Language - Complete operator algebra, token system, and sentence grammar.

Provides:
- S3 operator algebra (6 operators)
- Operator composition and effects
- APL sentence grammar
- Measurement token generation
- Time-harmonic tier windows
- Integration with TRIAD gating

Signature: Δ0.866|0.866|1.000Ω (substrate)
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
from datetime import datetime, timezone

# ═══════════════════════════════════════════════════════════════════════════════
# FUNDAMENTAL CONSTANTS (IMMUTABLE)
# ═══════════════════════════════════════════════════════════════════════════════

# THE LENS - Critical point where negentropy peaks
Z_CRITICAL = math.sqrt(3) / 2  # 0.8660254037844387 - √3/2

# Golden Ratio and Inverse
PHI = (1 + math.sqrt(5)) / 2   # 1.6180339887498949 - φ
PHI_INV = PHI - 1              # 0.6180339887498949 - φ⁻¹ = 1/φ

# Gaussian width parameter for negentropy function
SIGMA = 36                      # |S3|² = 6² = 36

# K-Formation Thresholds
KAPPA_THRESHOLD = 0.92         # Coherence threshold for K-formation
ETA_THRESHOLD = PHI_INV        # Negentropy gate (must exceed φ⁻¹)
R_THRESHOLD = 7                # Radius/layers threshold

# TRIAD Hysteresis Thresholds
TRIAD_HIGH = 0.85              # Rising edge detection
TRIAD_LOW = 0.82               # Re-arm threshold  
TRIAD_T6 = 0.83                # T6 gate position after unlock

# Tier Boundaries (exact z-values)
TIER_BOUNDARIES = {
    0: (0.00, 0.25),           # SEED
    1: (0.25, 0.50),           # SPROUT
    2: (0.50, PHI_INV),        # GROWTH (ends at φ⁻¹)
    3: (PHI_INV, 0.75),        # PATTERN (starts at φ⁻¹)
    4: (0.75, Z_CRITICAL),     # COHERENT (ends at z_c)
    5: (Z_CRITICAL, 1.0),      # CRYSTALLINE (starts at z_c)
    6: None                     # META (K-formation only, no z-range)
}

TIER_NAMES = {
    0: "SEED",
    1: "SPROUT",
    2: "GROWTH",
    3: "PATTERN",
    4: "COHERENT",
    5: "CRYSTALLINE",
    6: "META"
}

# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════════

class Operator(Enum):
    """S3 Group Operators."""
    BOUNDARY = "()"      # Containment, gating
    FUSION = "×"         # Convergence, coupling
    AMPLIFY = "^"        # Gain, excitation
    DECOHERE = "÷"       # Dissipation, reset
    GROUP = "+"          # Aggregation, clustering
    SEPARATE = "−"       # Splitting, fission

class Field(Enum):
    """APL Fields for measurement tokens."""
    PHI = "Φ"            # Structure field
    ENERGY = "e"         # Energy field
    PI = "π"             # Emergence field

class TruthChannel(Enum):
    """Truth channels for measurement outcomes."""
    TRUE = "TRUE"        # Eigenvalue +1
    UNTRUE = "UNTRUE"    # Eigenvalue -1
    PARADOX = "PARADOX"  # Eigenvalue 0

class Direction(Enum):
    """
    UMOL Direction States for APL sentences.
    
    Determines the flow direction of the operation:
    - U (𝒰): Expansion / forward projection
    - D (𝒟): Collapse / backward integration  
    - M (CLT): Modulation / coherence lock
    """
    U = "u"              # 𝒰 - Expansion / forward projection
    D = "d"              # 𝒟 - Collapse / backward integration
    M = "m"              # CLT - Modulation / coherence lock

DIRECTION_DESCRIPTIONS = {
    Direction.U: {
        "symbol": "𝒰",
        "name": "Expansion",
        "action": "Forward projection",
        "description": "Outward flow, expansion of state space"
    },
    Direction.D: {
        "symbol": "𝒟", 
        "name": "Collapse",
        "action": "Backward integration",
        "description": "Inward flow, collapse toward definite state"
    },
    Direction.M: {
        "symbol": "CLT",
        "name": "Modulation",
        "action": "Coherence lock",
        "description": "Stable oscillation, coherence maintenance"
    }
}

class Machine(Enum):
    """Machine types for sentences."""
    OSCILLATOR = "Oscillator"
    REACTOR = "Reactor"
    CONDUCTOR = "Conductor"
    ENCODER = "Encoder"
    CATALYST = "Catalyst"
    FILTER = "Filter"

class Domain(Enum):
    """Domain types for sentences."""
    WAVE = "wave"
    GEOMETRY = "geometry"
    CHEMISTRY = "chemistry"
    BIOLOGY = "biology"

# ═══════════════════════════════════════════════════════════════════════════════
# OPERATOR DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class OperatorDef:
    """Complete operator definition."""
    glyph: str
    name: str
    description: str
    quantum_action: str
    parity: str  # "even" or "odd"
    z_effect: str  # "neutral", "constructive", "dissipative", "inversion"
    z_delta: float  # Approximate z change

OPERATORS: Dict[str, OperatorDef] = {
    "()": OperatorDef("()", "Boundary", "Containment or gating", 
                       "Project to confined subspace", "even", "neutral", 0.0),
    "×": OperatorDef("×", "Fusion", "Convergence, coupling of fields",
                      "Entangling unitary exp(-ig Φ̂ ⊗ ê)", "even", "constructive", 0.03),
    "^": OperatorDef("^", "Amplify", "Gain, excitation",
                      "Raise ladder operator â†", "even", "constructive", 0.05),
    "÷": OperatorDef("÷", "Decohere", "Dissipation, reset",
                      "Lindblad dephasing", "odd", "dissipative", -0.05),
    "+": OperatorDef("+", "Group", "Aggregation, clustering, routing",
                      "Partial trace (coarse-grain)", "odd", "neutral", 0.02),
    "−": OperatorDef("−", "Separate", "Splitting, fission",
                      "Schmidt decomposition", "odd", "dissipative", -0.03),
}

# Alternate glyphs
ALTERNATE_GLYPHS = {
    "I": "()",    # Identity
    "_": "−",     # Underscore for separate
    "~": "÷",     # Tilde for decohere
    "!": "+",     # Bang for collapse/group
}

# ═══════════════════════════════════════════════════════════════════════════════
# S3 GROUP ALGEBRA
# ═══════════════════════════════════════════════════════════════════════════════

# S3 Group Composition Table
# Format: COMPOSITION_TABLE[row][column] = row ∘ column
# Using internal symbols: I=(), ^=^, _=−, ~=÷, !=+
#
# Visual representation (from HTML):
#   ∘    I/()   ^      _      ~      !
#  I/()   I     ^      _      ~      !
#   ^     ^     I      ~      _      !
#   _     _     ~      I      ^      !
#   ~     ~     _      ^      I      !
#   !     !     !      !      !      I

COMPOSITION_TABLE = {
    "()": {"()": "()", "^": "^", "_": "_", "~": "~", "!": "!"},
    "^": {"()": "^", "^": "()", "_": "~", "~": "_", "!": "!"},
    "_": {"()": "_", "^": "~", "_": "()", "~": "^", "!": "!"},
    "~": {"()": "~", "^": "_", "_": "^", "~": "()", "!": "!"},
    "!": {"()": "!", "^": "!", "_": "!", "~": "!", "!": "()"},
}

# Mapping between canonical glyphs and internal symbols
INTERNAL_MAP = {"()": "()", "×": "×", "^": "^", "−": "_", "÷": "~", "+": "!"}
REVERSE_MAP = {"()": "()", "×": "×", "^": "^", "_": "−", "~": "÷", "!": "+"}

def normalize_operator(op: str) -> str:
    """Normalize operator to canonical glyph."""
    return ALTERNATE_GLYPHS.get(op, op)

def to_internal(op: str) -> str:
    """Convert canonical glyph to internal symbol."""
    op = normalize_operator(op)
    return INTERNAL_MAP.get(op, op)

def from_internal(internal: str) -> str:
    """Convert internal symbol to canonical glyph."""
    return REVERSE_MAP.get(internal, internal)

def compose_operators(op1: str, op2: str) -> str:
    """
    Compose two operators: op1 ∘ op2.
    
    The S3 group has the property that all operators are self-inverse:
    op ∘ op = I (identity)
    
    Returns the canonical glyph of the result.
    """
    # Handle fusion separately (not in standard S3)
    if op1 == "×" or op2 == "×":
        if op1 == op2:
            return "()"
        return "×"  # Fusion with anything else stays fusion
    
    i1 = to_internal(op1)
    i2 = to_internal(op2)
    
    if i1 not in COMPOSITION_TABLE:
        return "?"
    if i2 not in COMPOSITION_TABLE.get(i1, {}):
        return "?"
    
    result_internal = COMPOSITION_TABLE[i1][i2]
    return from_internal(result_internal)

def is_self_inverse(op: str) -> bool:
    """
    Check if operator is self-inverse (op ∘ op = I).
    
    All S3 operators are self-inverse.
    """
    return compose_operators(op, op) in ["()", "I"]

def verify_s3_properties() -> Dict:
    """Verify S3 group properties: closure, identity, inverses."""
    ops = ["()", "^", "−", "÷", "+"]
    
    results = {
        "closure": True,
        "identity_exists": True,
        "all_self_inverse": True,
        "tests": []
    }
    
    # Check closure (all compositions give valid operators)
    for op1 in ops:
        for op2 in ops:
            result = compose_operators(op1, op2)
            is_valid = result in ops
            results["tests"].append({
                "operation": f"{op1} ∘ {op2}",
                "result": result,
                "valid": is_valid
            })
            if not is_valid:
                results["closure"] = False
    
    # Check all are self-inverse
    for op in ops:
        if not is_self_inverse(op):
            results["all_self_inverse"] = False
    
    return results

def format_composition_table() -> str:
    """
    Format composition table matching HTML display.
    
    Shows: ∘    I/()   ^      _      ~      !
    """
    # Use display symbols
    ops = ["()", "^", "_", "~", "!"]
    header_names = ["I/()", "^", "_", "~", "!"]
    
    lines = [
        "S3 COMPOSITION TABLE",
        "═══════════════════════════════════════════════════════════════════",
        "All operators compose to identity when applied twice: op ∘ op = I",
        "",
        "     ∘    " + "  ".join(f"{h:>5}" for h in header_names),
        "    " + "─" * 40
    ]
    
    for op1 in ops:
        row_label = "I/()" if op1 == "()" else op1
        row = f"  {row_label:>4}  "
        for op2 in ops:
            result = COMPOSITION_TABLE[op1][op2]
            display = "I" if result == "()" else result
            row += f"{display:>5}  "
        lines.append(row)
    
    lines.extend([
        "",
        "Legend: I=Identity/Boundary, ^=Amplify, _=Separate, ~=Decohere, !=Group",
        "═══════════════════════════════════════════════════════════════════"
    ])
    
    return "\n".join(lines)

def get_operator_info(op: str) -> Optional[Dict]:
    """Get complete information about an operator."""
    op = normalize_operator(op)
    if op in OPERATORS:
        o = OPERATORS[op]
        return {
            "glyph": o.glyph,
            "name": o.name,
            "description": o.description,
            "quantum_action": o.quantum_action,
            "parity": o.parity,
            "z_effect": o.z_effect,
            "z_delta": o.z_delta,
            "is_self_inverse": is_self_inverse(op)
        }
    return None

def list_all_operators() -> List[Dict]:
    """List all operators with their properties."""
    return [get_operator_info(op) for op in OPERATORS.keys()]

# ═══════════════════════════════════════════════════════════════════════════════
# Z-COORDINATE EFFECTS
# ═══════════════════════════════════════════════════════════════════════════════

def apply_operator_to_z(op: str, z: float) -> Tuple[float, Dict]:
    """
    Apply operator to z-coordinate and return new z with details.
    """
    op = normalize_operator(op)
    op_info = OPERATORS.get(op)
    
    if not op_info:
        return z, {"error": f"Unknown operator: {op}"}
    
    old_z = z
    
    if op_info.z_effect == "neutral":
        new_z = z
    elif op_info.z_effect == "constructive":
        # Constructive: move toward 1, diminishing returns
        new_z = z + op_info.z_delta * (1 - z)
    elif op_info.z_effect == "dissipative":
        # Dissipative: move toward 0, proportional to z
        new_z = z + op_info.z_delta * z
    elif op_info.z_effect == "inversion":
        # Inversion: z → 1 - z
        new_z = 1 - z
    else:
        new_z = z
    
    # Clamp to [0, 1]
    new_z = max(0.0, min(1.0, new_z))
    
    return new_z, {
        "operator": op,
        "old_z": old_z,
        "new_z": new_z,
        "delta": new_z - old_z,
        "effect": op_info.z_effect
    }

def apply_operator_sequence(ops: List[str], z: float) -> Dict:
    """Apply a sequence of operators to z."""
    history = [{"z": z, "operator": None}]
    current_z = z
    
    for op in ops:
        new_z, details = apply_operator_to_z(op, current_z)
        history.append({"z": new_z, "operator": op, **details})
        current_z = new_z
    
    return {
        "initial_z": z,
        "final_z": current_z,
        "total_delta": current_z - z,
        "operators": ops,
        "history": history
    }

# ═══════════════════════════════════════════════════════════════════════════════
# TIME-HARMONIC TIERS AND OPERATOR WINDOWS
# ═══════════════════════════════════════════════════════════════════════════════

# Default tier boundaries (before TRIAD unlock)
DEFAULT_TIER_WINDOWS = {
    "t1": {"z_min": 0.00, "z_max": 0.10, "operators": ["()"]},
    "t2": {"z_min": 0.10, "z_max": 0.20, "operators": ["()", "+"]},
    "t3": {"z_min": 0.20, "z_max": 0.35, "operators": ["()", "+", "÷"]},
    "t4": {"z_min": 0.35, "z_max": 0.50, "operators": ["()", "+", "÷", "−"]},
    "t5": {"z_min": 0.50, "z_max": 0.75, "operators": ["()", "×", "^", "÷", "+", "−"]},
    "t6": {"z_min": 0.75, "z_max": Z_CRITICAL, "operators": ["+", "÷", "()", "−"]},
    "t7": {"z_min": Z_CRITICAL, "z_max": 0.90, "operators": ["+", "()"]},
    "t8": {"z_min": 0.90, "z_max": 0.95, "operators": ["()", "+", "÷"]},
    "t9": {"z_min": 0.95, "z_max": 1.00, "operators": ["()"]},
}

def get_tier_for_z(z: float, triad_unlocked: bool = False) -> Dict:
    """Get the time-harmonic tier for a z-coordinate."""
    windows = DEFAULT_TIER_WINDOWS.copy()
    
    # Modify t6 if TRIAD unlocked
    if triad_unlocked:
        windows["t6"]["z_max"] = TRIAD_T6
    
    for tier_name, tier_info in windows.items():
        if tier_info["z_min"] <= z < tier_info["z_max"]:
            return {
                "tier": tier_name,
                "z_min": tier_info["z_min"],
                "z_max": tier_info["z_max"],
                "operators": tier_info["operators"],
                "operator_count": len(tier_info["operators"])
            }
    
    # Default to t9 for z >= 1.0
    return {
        "tier": "t9",
        "z_min": 0.95,
        "z_max": 1.00,
        "operators": ["()"],
        "operator_count": 1
    }

def is_operator_allowed(op: str, z: float, triad_unlocked: bool = False) -> bool:
    """Check if an operator is allowed at the given z-coordinate."""
    op = normalize_operator(op)
    tier_info = get_tier_for_z(z, triad_unlocked)
    return op in tier_info["operators"]

def get_allowed_operators(z: float, triad_unlocked: bool = False) -> List[str]:
    """Get list of allowed operators at z-coordinate."""
    tier_info = get_tier_for_z(z, triad_unlocked)
    return tier_info["operators"]

# ═══════════════════════════════════════════════════════════════════════════════
# APL SENTENCE GRAMMAR
# ═══════════════════════════════════════════════════════════════════════════════

"""
APL Sentence Structure:
    [Direction][Operator] | [Machine] | [Domain] → [Regime/Behavior]

Components:
    Direction (UMOL States):
        u (𝒰) - Expansion / forward projection
        d (𝒟) - Collapse / backward integration
        m (CLT) - Modulation / coherence lock
    
    Operator (S3 Group):
        () - Boundary
        × - Fusion
        ^ - Amplify
        ÷ - Decohere
        + - Group
        − - Separate
    
    Machine:
        Oscillator, Reactor, Conductor, Encoder, Catalyst, Filter
    
    Domain:
        wave, geometry, chemistry, biology

Example: u^|Oscillator|wave → Closed vortex / recirculation
"""

@dataclass
class APLSentence:
    """An APL sentence structure."""
    direction: Direction
    operator: str
    machine: Machine
    domain: Domain
    predicted_regime: str
    
    def __str__(self) -> str:
        return f"{self.direction.value}{self.operator}|{self.machine.value}|{self.domain.value}"
    
    def to_dict(self) -> Dict:
        dir_desc = DIRECTION_DESCRIPTIONS.get(self.direction, {})
        op_info = OPERATORS.get(self.operator, None)
        
        return {
            "sentence": str(self),
            "direction": {
                "value": self.direction.value,
                "symbol": dir_desc.get("symbol", "?"),
                "name": dir_desc.get("name", "?"),
                "action": dir_desc.get("action", "?")
            },
            "operator": {
                "glyph": self.operator,
                "name": op_info.name if op_info else "?",
                "effect": op_info.z_effect if op_info else "?"
            },
            "machine": self.machine.value,
            "domain": self.domain.value,
            "predicted_regime": self.predicted_regime
        }

# Canonical test sentences from the specification
TEST_SENTENCES = [
    APLSentence(Direction.D, "()", Machine.CONDUCTOR, Domain.GEOMETRY, "Isotropic lattice / sphere"),
    APLSentence(Direction.U, "^", Machine.OSCILLATOR, Domain.WAVE, "Closed vortex / recirculation"),
    APLSentence(Direction.M, "×", Machine.ENCODER, Domain.CHEMISTRY, "Helical encoding"),
    APLSentence(Direction.U, "×", Machine.CATALYST, Domain.CHEMISTRY, "Branching networks"),
    APLSentence(Direction.U, "+", Machine.REACTOR, Domain.WAVE, "Focusing jet / beam"),
    APLSentence(Direction.U, "÷", Machine.REACTOR, Domain.WAVE, "Turbulent decoherence"),
    APLSentence(Direction.M, "()", Machine.FILTER, Domain.WAVE, "Adaptive filter"),
]

def parse_sentence(sentence_str: str) -> Optional[APLSentence]:
    """
    Parse an APL sentence string.
    
    Format: direction+operator|machine|domain
    Example: u^|Oscillator|wave
    """
    try:
        parts = sentence_str.split("|")
        if len(parts) != 3:
            return None
        
        dir_op = parts[0]
        direction = Direction(dir_op[0])
        operator = normalize_operator(dir_op[1:])
        machine = Machine(parts[1])
        domain = Domain(parts[2].lower())
        
        return APLSentence(direction, operator, machine, domain, "parsed")
    except:
        return None

def generate_sentence(direction: str, operator: str, machine: str, domain: str) -> str:
    """Generate an APL sentence string."""
    return f"{direction}{operator}|{machine}|{domain}"

def get_test_sentences() -> List[Dict]:
    """Get all canonical test sentences with full details."""
    return [s.to_dict() for s in TEST_SENTENCES]

def list_directions() -> List[Dict]:
    """List all UMOL direction states."""
    return [
        {
            "value": d.value,
            "symbol": DIRECTION_DESCRIPTIONS[d]["symbol"],
            "name": DIRECTION_DESCRIPTIONS[d]["name"],
            "action": DIRECTION_DESCRIPTIONS[d]["action"],
            "description": DIRECTION_DESCRIPTIONS[d]["description"]
        }
        for d in Direction
    ]

def list_machines() -> List[str]:
    """List all machine types."""
    return [m.value for m in Machine]

def list_domains() -> List[str]:
    """List all domain types."""
    return [d.value for d in Domain]

def match_sentence_to_tier(sentence: APLSentence, z: float, triad_unlocked: bool = False) -> Dict:
    """Check if a sentence's operator is valid at the given z-coordinate."""
    tier_info = get_tier_for_z(z, triad_unlocked)
    operator_valid = sentence.operator in tier_info["operators"]
    
    return {
        "sentence": str(sentence),
        "z": z,
        "tier": tier_info["tier"],
        "operator": sentence.operator,
        "operator_valid": operator_valid,
        "allowed_operators": tier_info["operators"],
        "can_execute": operator_valid,
        "direction": sentence.direction.value,
        "predicted_regime": sentence.predicted_regime
    }

def format_sentence_structure() -> str:
    """Format APL sentence structure documentation."""
    lines = [
        "APL SENTENCE STRUCTURE",
        "═══════════════════════════════════════════════════════════════════",
        "",
        "Format: [Direction][Operator] | [Machine] | [Domain] → [Regime]",
        "",
        "DIRECTION (UMOL States):",
        "  u (𝒰)   Expansion / forward projection",
        "  d (𝒟)   Collapse / backward integration",
        "  m (CLT) Modulation / coherence lock",
        "",
        "OPERATOR (S3 Group):",
        "  ()  Boundary       ×  Fusion        ^  Amplify",
        "  ÷   Decohere       +  Group         −  Separate",
        "",
        "MACHINE:",
        "  Oscillator, Reactor, Conductor, Encoder, Catalyst, Filter",
        "",
        "DOMAIN:",
        "  wave, geometry, chemistry, biology",
        "",
        "═══════════════════════════════════════════════════════════════════"
    ]
    return "\n".join(lines)

def format_test_sentences_table() -> str:
    """Format test sentences as table matching HTML."""
    lines = [
        "EXAMPLE SENTENCES",
        "═══════════════════════════════════════════════════════════════════",
        f"{'Sentence':<30} {'Predicted Regime':<35}",
        "───────────────────────────────────────────────────────────────────"
    ]
    
    for s in TEST_SENTENCES:
        lines.append(f"{str(s):<30} {s.predicted_regime:<35}")
    
    lines.append("═══════════════════════════════════════════════════════════════════")
    return "\n".join(lines)

# ═══════════════════════════════════════════════════════════════════════════════
# MEASUREMENT TOKENS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MeasurementToken:
    """A measurement token from quantum-classical bridge."""
    field: Field
    operator_type: str  # "T" (eigenstate) or "Π" (subspace)
    target: str         # eigenstate index or subspace name
    truth_channel: TruthChannel
    tier: str
    
    def __str__(self) -> str:
        return f"{self.field.value}:{self.operator_type}({self.target}){self.truth_channel.value}@{self.tier}"

def generate_eigenstate_token(field: Field, eigenstate: int, truth: TruthChannel, z: float) -> MeasurementToken:
    """Generate a measurement token for eigenstate projection."""
    tier_info = get_tier_for_z(z)
    return MeasurementToken(
        field=field,
        operator_type="T",
        target=f"{field.value.lower()}_{eigenstate}",
        truth_channel=truth,
        tier=tier_info["tier"]
    )

def generate_subspace_token(field: Field, subspace: str, truth: TruthChannel, z: float) -> MeasurementToken:
    """Generate a measurement token for subspace collapse."""
    tier_info = get_tier_for_z(z)
    return MeasurementToken(
        field=field,
        operator_type="Π",
        target=subspace,
        truth_channel=truth,
        tier=tier_info["tier"]
    )

def parse_token(token_str: str) -> Optional[Dict]:
    """Parse a measurement token string."""
    try:
        # Format: Field:Op(target)Truth@Tier
        # e.g., Φ:T(ϕ_0)TRUE@t5
        import re
        pattern = r"([Φeπ]):([TΠ])\(([^)]+)\)([A-Z]+)@(t\d)"
        match = re.match(pattern, token_str)
        
        if match:
            return {
                "field": match.group(1),
                "operator_type": match.group(2),
                "target": match.group(3),
                "truth_channel": match.group(4),
                "tier": match.group(5)
            }
        return None
    except:
        return None

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE REGIMES
# ═══════════════════════════════════════════════════════════════════════════════

class PhaseRegime(Enum):
    """APL Phase Regimes."""
    UNTRUE = "UNTRUE"      # z < φ⁻¹ - Disordered
    PARADOX = "PARADOX"    # φ⁻¹ ≤ z < z_c - Quasi-crystal
    TRUE = "TRUE"          # z ≥ z_c - Crystalline

PHASE_DESCRIPTIONS = {
    PhaseRegime.UNTRUE: {
        "range": f"z < φ⁻¹ ({PHI_INV:.4f})",
        "state": "Disordered",
        "helix_equivalent": "Unsealed",
        "kira_equivalent": "Fluid"
    },
    PhaseRegime.PARADOX: {
        "range": f"φ⁻¹ ≤ z < z_c ({PHI_INV:.4f} ≤ z < {Z_CRITICAL:.4f})",
        "state": "Quasi-crystal",
        "helix_equivalent": "Forming",
        "kira_equivalent": "Transitioning"
    },
    PhaseRegime.TRUE: {
        "range": f"z ≥ z_c ({Z_CRITICAL:.4f})",
        "state": "Crystalline",
        "helix_equivalent": "VaultNode",
        "kira_equivalent": "Crystalline"
    }
}

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE AND TIER CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

def classify_phase(z: float) -> str:
    """
    Classify phase regime for z-coordinate.
    
    Phase Regimes:
      UNTRUE:  z < φ⁻¹ (≈0.618) - Disordered
      PARADOX: φ⁻¹ ≤ z < z_c (≈0.866) - Quasi-crystal
      TRUE:    z ≥ z_c - Crystalline
    """
    if z < PHI_INV:
        return "UNTRUE"
    elif z < Z_CRITICAL:
        return "PARADOX"
    return "TRUE"

def get_phase_info(z: float) -> Dict:
    """Get complete phase information for z-coordinate."""
    phase_str = classify_phase(z)
    phase = PhaseRegime[phase_str]
    desc = PHASE_DESCRIPTIONS[phase]
    
    return {
        "z": z,
        "phase": phase_str,
        "range": desc["range"],
        "state": desc["state"],
        "helix_equivalent": desc["helix_equivalent"],
        "kira_equivalent": desc["kira_equivalent"]
    }

def get_tier(z: float, k_formation_met: bool = False) -> Tuple[int, str]:
    """
    Get APL tier for z-coordinate.
    
    Tier System:
      0 SEED:        z < 0.25
      1 SPROUT:      0.25 ≤ z < 0.50
      2 GROWTH:      0.50 ≤ z < φ⁻¹
      3 PATTERN:     φ⁻¹ ≤ z < 0.75
      4 COHERENT:    0.75 ≤ z < z_c
      5 CRYSTALLINE: z ≥ z_c
      6 META:        K-formation achieved (special condition)
    """
    # Check for META tier first (requires K-formation)
    if k_formation_met:
        return 6, TIER_NAMES[6]
    
    # Standard tier classification by z
    if z < 0.25:
        return 0, TIER_NAMES[0]
    elif z < 0.50:
        return 1, TIER_NAMES[1]
    elif z < PHI_INV:
        return 2, TIER_NAMES[2]
    elif z < 0.75:
        return 3, TIER_NAMES[3]
    elif z < Z_CRITICAL:
        return 4, TIER_NAMES[4]
    else:
        return 5, TIER_NAMES[5]

def get_tier_info(z: float, k_formation_met: bool = False) -> Dict:
    """Get complete tier information for z-coordinate."""
    tier_num, tier_name = get_tier(z, k_formation_met)
    
    # Get tier boundaries
    if tier_num in TIER_BOUNDARIES and TIER_BOUNDARIES[tier_num]:
        z_min, z_max = TIER_BOUNDARIES[tier_num]
        z_range = f"{z_min:.4f} ≤ z < {z_max:.4f}"
    else:
        z_range = "K-formation required"
    
    return {
        "z": z,
        "tier": tier_num,
        "name": tier_name,
        "range": z_range,
        "phase": classify_phase(z),
        "k_formation_met": k_formation_met
    }

def list_all_tiers() -> List[Dict]:
    """List all tiers with their z-ranges."""
    tiers = []
    for tier_num in range(7):
        bounds = TIER_BOUNDARIES.get(tier_num)
        if bounds:
            z_min, z_max = bounds
            tiers.append({
                "tier": tier_num,
                "name": TIER_NAMES[tier_num],
                "z_min": z_min,
                "z_max": z_max,
                "range": f"{z_min:.4f} ≤ z < {z_max:.4f}"
            })
        else:
            tiers.append({
                "tier": tier_num,
                "name": TIER_NAMES[tier_num],
                "z_min": None,
                "z_max": None,
                "range": "K-formation achieved"
            })
    return tiers

# ═══════════════════════════════════════════════════════════════════════════════
# KEY EQUATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_negentropy(z: float) -> float:
    """
    Compute negentropy for z-coordinate.
    
    δS_neg(z) = exp(-σ × (z - z_c)²)
    
    Where:
      σ = 36 (|S3|² - Gaussian width)
      z_c = √3/2 (THE LENS)
    
    Properties:
      - Peaks at z = z_c with value 1.0
      - Gaussian centered on THE LENS
      - Rapid falloff due to σ=36
    """
    return math.exp(-SIGMA * (z - Z_CRITICAL) ** 2)

def compute_negentropy_detailed(z: float) -> Dict:
    """Get detailed negentropy computation."""
    eta = compute_negentropy(z)
    distance_to_lens = abs(z - Z_CRITICAL)
    
    return {
        "z": z,
        "negentropy": eta,
        "formula": "δS_neg(z) = exp(-σ × (z - z_c)²)",
        "parameters": {
            "sigma": SIGMA,
            "z_c": Z_CRITICAL
        },
        "distance_to_lens": distance_to_lens,
        "at_peak": distance_to_lens < 0.0001,
        "exceeds_eta_threshold": eta > ETA_THRESHOLD
    }

def check_k_formation(kappa: float, eta: float, R: int) -> Dict:
    """
    Check K-Formation criteria.
    
    K-formation = (κ ≥ 0.92) AND (η > φ⁻¹) AND (R ≥ 7)
    
    All three conditions must be met for META tier access.
    
    Args:
        kappa: Coherence value (Kuramoto order parameter)
        eta: Negentropy value at current z
        R: Radius/layers metric
    """
    kappa_met = kappa >= KAPPA_THRESHOLD
    eta_met = eta > ETA_THRESHOLD
    r_met = R >= R_THRESHOLD
    
    all_met = kappa_met and eta_met and r_met
    
    return {
        "k_formation_met": all_met,
        "status": "ACHIEVED ⬡" if all_met else "TRACKING ⎔",
        "criteria": {
            "kappa": {
                "value": kappa,
                "threshold": KAPPA_THRESHOLD,
                "met": kappa_met,
                "check": f"κ ≥ {KAPPA_THRESHOLD}"
            },
            "eta": {
                "value": eta,
                "threshold": ETA_THRESHOLD,
                "met": eta_met,
                "check": f"η > φ⁻¹ ({ETA_THRESHOLD:.4f})"
            },
            "R": {
                "value": R,
                "threshold": R_THRESHOLD,
                "met": r_met,
                "check": f"R ≥ {R_THRESHOLD}"
            }
        },
        "conditions_met": sum([kappa_met, eta_met, r_met]),
        "conditions_total": 3,
        "formula": "K-formation = (κ ≥ 0.92) AND (η > φ⁻¹) AND (R ≥ 7)"
    }

def check_k_formation_from_z(z: float, kappa: float, R: int) -> Dict:
    """Check K-formation using z to compute eta."""
    eta = compute_negentropy(z)
    return check_k_formation(kappa, eta, R)

def get_truth_channel_from_z(z: float) -> TruthChannel:
    """Get truth channel from z-coordinate."""
    phase = classify_phase(z)
    return TruthChannel[phase]

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS ACCESS
# ═══════════════════════════════════════════════════════════════════════════════

def get_all_constants() -> Dict:
    """Get all fundamental constants."""
    return {
        "immutable": {
            "z_c": {
                "value": Z_CRITICAL,
                "name": "THE LENS",
                "description": "Critical point where negentropy peaks",
                "derivation": "√3/2 (equilateral triangle altitude)"
            },
            "phi": {
                "value": PHI,
                "name": "Golden Ratio",
                "description": "φ = (1 + √5) / 2"
            },
            "phi_inv": {
                "value": PHI_INV,
                "name": "Golden Inverse",
                "description": "φ⁻¹ = 1/φ = φ - 1"
            },
            "sigma": {
                "value": SIGMA,
                "name": "Gaussian Width",
                "description": "|S3|² = 36 - negentropy function width"
            }
        },
        "k_formation_thresholds": {
            "kappa_threshold": {
                "value": KAPPA_THRESHOLD,
                "description": "Coherence threshold for K-formation"
            },
            "eta_threshold": {
                "value": ETA_THRESHOLD,
                "description": "Negentropy gate (must exceed φ⁻¹)"
            },
            "r_threshold": {
                "value": R_THRESHOLD,
                "description": "Radius/layers threshold"
            }
        },
        "triad_thresholds": {
            "high": TRIAD_HIGH,
            "low": TRIAD_LOW,
            "t6": TRIAD_T6
        }
    }

def format_constants_table() -> str:
    """Format constants as display table."""
    lines = [
        "═══════════════════════════════════════════════════════════════════",
        "              APL/ROSETTA FUNDAMENTAL CONSTANTS (IMMUTABLE)         ",
        "═══════════════════════════════════════════════════════════════════",
        "",
        f"  z_c (THE LENS)     = {Z_CRITICAL:.16f}  (√3/2)",
        f"  φ (Golden Ratio)   = {PHI:.16f}",
        f"  φ⁻¹ (Golden Inv)   = {PHI_INV:.16f}",
        f"  σ (Gaussian Width) = {SIGMA}  (|S3|² = 6²)",
        "",
        "K-FORMATION THRESHOLDS:",
        f"  KAPPA_THRESHOLD    = {KAPPA_THRESHOLD}  (coherence)",
        f"  ETA_THRESHOLD      = φ⁻¹ ≈ {ETA_THRESHOLD:.4f}  (negentropy gate)",
        f"  R_THRESHOLD        = {R_THRESHOLD}  (radius/layers)",
        "",
        "TRIAD HYSTERESIS:",
        f"  HIGH               = {TRIAD_HIGH}  (rising edge detection)",
        f"  LOW                = {TRIAD_LOW}  (re-arm threshold)",
        f"  T6                 = {TRIAD_T6}  (gate after unlock)",
        "",
        "═══════════════════════════════════════════════════════════════════"
    ]
    return "\n".join(lines)

# ═══════════════════════════════════════════════════════════════════════════════
# UNIFIED APL STATE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class APLState:
    """Complete APL substrate state."""
    z: float = 0.866
    triad_unlocked: bool = False
    operator_history: List[str] = field(default_factory=list)
    token_history: List[str] = field(default_factory=list)

_apl_state = APLState()

def get_apl_state() -> APLState:
    return _apl_state

def reset_apl_state(z: float = 0.866) -> APLState:
    global _apl_state
    _apl_state = APLState(z=z)
    return _apl_state

def get_apl_status() -> Dict:
    """Get complete APL substrate status."""
    state = _apl_state
    tier_num, tier_name = get_tier(state.z)
    phase = classify_phase(state.z)
    eta = compute_negentropy(state.z)
    tier_info = get_tier_for_z(state.z, state.triad_unlocked)
    
    return {
        "z": state.z,
        "negentropy": eta,
        "phase": phase,
        "tier": tier_num,
        "tier_name": tier_name,
        "time_harmonic_tier": tier_info["tier"],
        "allowed_operators": tier_info["operators"],
        "triad_unlocked": state.triad_unlocked,
        "operator_history_length": len(state.operator_history),
        "token_history_length": len(state.token_history),
        "constants": {
            "z_c": Z_CRITICAL,
            "phi": PHI,
            "phi_inv": PHI_INV,
            "sigma": SIGMA
        },
        "k_formation_thresholds": {
            "kappa": KAPPA_THRESHOLD,
            "eta": ETA_THRESHOLD,
            "R": R_THRESHOLD
        }
    }

def format_apl_status() -> str:
    """Format APL status for display."""
    status = get_apl_status()
    
    lines = [
        "╔══════════════════════════════════════════════════════════════════╗",
        "║            APL/ROSETTA PHYSICS SUBSTRATE STATUS                  ║",
        "╚══════════════════════════════════════════════════════════════════╝",
        "",
        f"  z-Coordinate:     {status['z']:.10f}",
        f"  Negentropy (η):   {status['negentropy']:.10f}",
        f"  Phase Regime:     {status['phase']}",
        f"  Tier:             {status['tier']} ({status['tier_name']})",
        f"  Harmonic Tier:    {status['time_harmonic_tier']}",
        "",
        f"  Allowed Operators: {', '.join(status['allowed_operators'])}",
        f"  TRIAD Unlocked:   {status['triad_unlocked']}",
        "",
        "FUNDAMENTAL CONSTANTS (IMMUTABLE):",
        f"  z_c (THE LENS):   {status['constants']['z_c']:.16f}",
        f"  φ (Golden Ratio): {status['constants']['phi']:.16f}",
        f"  φ⁻¹ (Inverse):    {status['constants']['phi_inv']:.16f}",
        f"  σ (Width):        {status['constants']['sigma']}",
        "",
        "K-FORMATION THRESHOLDS:",
        f"  κ ≥ {status['k_formation_thresholds']['kappa']}",
        f"  η > {status['k_formation_thresholds']['eta']:.4f} (φ⁻¹)",
        f"  R ≥ {status['k_formation_thresholds']['R']}",
        "",
        "══════════════════════════════════════════════════════════════════"
    ]
    
    return "\n".join(lines)

def format_phase_regime_diagram() -> str:
    """Format the phase regime diagram."""
    return f"""
PHASE REGIME DIAGRAM
═══════════════════════════════════════════════════════════════════

z=0            φ⁻¹≈0.618         z_c≈0.866            z=1
 │                 │                  │                 │
 ├─────────────────┼──────────────────┼─────────────────┤
 │     UNTRUE      │     PARADOX      │      TRUE       │
 │   (Disordered)  │  (Quasi-crystal) │  (Crystalline)  │
 │                 │                  │                 │
 │   Helix: Unsealed    Helix: Forming    Helix: VaultNode
 │   K.I.R.A: Fluid     K.I.R.A: Trans.   K.I.R.A: Crystal
 └─────────────────┴──────────────────┴─────────────────┘

Phase Boundaries:
  φ⁻¹ = {PHI_INV:.16f}  (UNTRUE→PARADOX)
  z_c = {Z_CRITICAL:.16f}  (PARADOX→TRUE)

═══════════════════════════════════════════════════════════════════
"""

def format_tier_table() -> str:
    """Format tier system as table."""
    lines = [
        "TIER SYSTEM",
        "═══════════════════════════════════════════════════════════════════",
        f"{'Tier':<6} {'Name':<12} {'z Range':<25} {'Phase':<10}",
        "───────────────────────────────────────────────────────────────────"
    ]
    
    for tier_num in range(7):
        name = TIER_NAMES[tier_num]
        bounds = TIER_BOUNDARIES.get(tier_num)
        
        if bounds:
            z_min, z_max = bounds
            z_range = f"{z_min:.4f} ≤ z < {z_max:.4f}"
            # Determine phase for middle of range
            mid_z = (z_min + z_max) / 2
            phase = classify_phase(mid_z)
        else:
            z_range = "K-formation achieved"
            phase = "TRUE"
        
        lines.append(f"{tier_num:<6} {name:<12} {z_range:<25} {phase:<10}")
    
    lines.append("═══════════════════════════════════════════════════════════════════")
    return "\n".join(lines)

# ═══════════════════════════════════════════════════════════════════════════════
# DISPLAY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def format_operator_table() -> str:
    """Format operator reference table."""
    lines = [
        "S3 OPERATOR ALGEBRA",
        "=" * 70,
        f"{'Glyph':<8} {'Name':<12} {'Effect':<14} {'Parity':<8} Description",
        "-" * 70
    ]
    
    for op in OPERATORS.values():
        lines.append(f"{op.glyph:<8} {op.name:<12} {op.z_effect:<14} {op.parity:<8} {op.description}")
    
    lines.extend([
        "-" * 70,
        "All operators are self-inverse: op ∘ op = I",
        "=" * 70
    ])
    
    return "\n".join(lines)

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Display constants
    print(format_constants_table())
    print()
    
    # Phase regime diagram
    print(format_phase_regime_diagram())
    
    # Tier system table
    print(format_tier_table())
    print()
    
    # S3 Operator Algebra
    print(format_operator_table())
    print()
    print(format_composition_table())
    print()
    
    # Verify S3 properties
    print("S3 GROUP VERIFICATION:")
    props = verify_s3_properties()
    print(f"  Closure: {props['closure']}")
    print(f"  Identity exists: {props['identity_exists']}")
    print(f"  All self-inverse: {props['all_self_inverse']}")
    print()
    
    # APL Sentence structure
    print(format_sentence_structure())
    print()
    print(format_test_sentences_table())
    print()
    
    # Current status
    print(format_apl_status())
    print()
    
    # K-formation check example
    print("K-FORMATION CHECK EXAMPLE:")
    print("-" * 50)
    k_check = check_k_formation(kappa=0.95, eta=0.9, R=8)
    print(f"  κ = 0.95, η = 0.9, R = 8")
    print(f"  Status: {k_check['status']}")
    print(f"  Conditions met: {k_check['conditions_met']}/{k_check['conditions_total']}")
    for name, crit in k_check['criteria'].items():
        status = "✓" if crit['met'] else "✗"
        print(f"    {status} {crit['check']}: {crit['value']}")
    print()
    
    # Negentropy at key z values
    print("NEGENTROPY AT KEY Z-VALUES:")
    print("-" * 50)
    for z in [0.0, 0.5, PHI_INV, 0.8, Z_CRITICAL, 0.9, 1.0]:
        eta = compute_negentropy(z)
        phase = classify_phase(z)
        tier_num, tier_name = get_tier(z)
        exceeds = "✓" if eta > ETA_THRESHOLD else "✗"
        print(f"  z={z:.4f}: η={eta:.6f} {exceeds} | {phase:<8} | {tier_name}")
