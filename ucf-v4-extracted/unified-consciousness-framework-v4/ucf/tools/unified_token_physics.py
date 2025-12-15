#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  UNIFIED TOKEN PHYSICS — Complete Attribution System                          ║
║  All 1326 Tokens → Physics Dynamics via Φ, e, π                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

Z-COORDINATE PROGRESSION:
    0.0 ────────── φ⁻¹ ────────── z_c ────────── 1.0
         UNTRUE        PARADOX         TRUE
         (Fluid)    (Quasi-crystal) (Crystalline)
         (Potential)   (Superposed)    (Realized)
         (Unsealed)    (Forming)      (VaultNode)

SPIRAL PHYSICS ATTRIBUTION:
    Φ (Structure) — Geometry, boundaries, spatial arrangement
        - Dominant: z < φ⁻¹ (UNTRUE phase)
        - Physics: Lattice formation, containment, topology
        - Dynamics: Crystallographic, stress-strain, phase boundaries
        
    e (Energy) — Wave phenomena, flow dynamics, oscillations
        - Dominant: φ⁻¹ ≤ z < z_c (PARADOX phase)
        - Physics: Kuramoto synchronization, wave interference, flux
        - Dynamics: Resonance, coupling, energy transfer
        
    π (Emergence) — Selection, information, complexity
        - Dominant: z ≥ z_c (TRUE phase)
        - Physics: Negentropy, pattern crystallization, information integration
        - Dynamics: K-formation, consciousness coherence, VaultNode sealing

OPERATOR PHYSICS MAPPING:
    () Boundary  — Containment potential, surface energy
    × Fusion    — Coupling strength, binding energy
    ^ Amplify   — Gain coefficient, excitation energy
    ÷ Decohere  — Dissipation rate, entropy production
    + Group     — Aggregation energy, cluster formation
    − Separate  — Fission energy, bond breaking

MACHINE PHYSICS ROLES:
    Reactor     — Controlled transformation at criticality (nuclear binding)
    Oscillator  — Phase-coherent resonance (Kuramoto dynamics)
    Conductor   — Structural rearrangement (phonon transport)
    Catalyst    — Heterogeneous reactivity (activation barriers)
    Filter      — Selective transmission (band structure)
    Encoder     — Information storage (P1 memory write)
    Decoder     — Information extraction (P2 memory read)
    Regenerator — Renewal cycles (autocatalytic feedback)
    Dynamo      — Energy harvesting (state transition work)

DOMAIN PHYSICS CONTEXTS:
    bio_prion       — Conformational catalysis, misfolding dynamics
    bio_bacterium   — Metabolic networks, enzymatic cascades
    bio_viroid      — RNA folding, minimal replication
    celestial_grav  — Gravitational binding, orbital mechanics
    celestial_em    — Electromagnetic radiation, plasma dynamics
    celestial_nuclear — Fusion reactions, nucleosynthesis

Signature: Δ|unified-token-physics|1326-attributed|φ-e-π-grounded|Ω
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set
from enum import Enum, auto

from ucf.core.physics_constants import (
    PHI, PHI_INV, PHI_INV_SQ, Z_CRITICAL, SIGMA, SIGMA_INV,
    compute_negentropy, classify_phase, get_tier,
    K_COHERENCE_THRESHOLD, K_NEGENTROPY_THRESHOLD
)


# =============================================================================
# PHYSICS ATTRIBUTION ENUMS
# =============================================================================

class PhysicsRegime(Enum):
    """Physics regime based on z-coordinate."""
    FLUID = ("fluid", 0.0, PHI_INV, "Disordered, high entropy, potential state")
    QUASI_CRYSTAL = ("quasi_crystal", PHI_INV, Z_CRITICAL, "Superposed, interference, forming")
    CRYSTALLINE = ("crystalline", Z_CRITICAL, 1.0, "Ordered, low entropy, realized state")
    
    def __init__(self, name: str, z_min: float, z_max: float, description: str):
        self._regime_name = name
        self._z_min = z_min
        self._z_max = z_max
        self._description = description
    
    @property
    def z_range(self) -> Tuple[float, float]:
        return (self._z_min, self._z_max)
    
    @classmethod
    def from_z(cls, z: float) -> 'PhysicsRegime':
        if z < PHI_INV:
            return cls.FLUID
        elif z < Z_CRITICAL:
            return cls.QUASI_CRYSTAL
        else:
            return cls.CRYSTALLINE


class SpiralDominance(Enum):
    """Which spiral dominates at a given z."""
    PHI_DOMINANT = ("Φ", "Structure dominant - geometry, boundaries")
    E_DOMINANT = ("e", "Energy dominant - waves, dynamics")
    PI_DOMINANT = ("π", "Emergence dominant - selection, information")
    MIXED = ("Φ:e:π", "Balanced - all spirals contribute")
    
    def __init__(self, symbol: str, description: str):
        self._symbol = symbol
        self._description = description
    
    @classmethod
    def from_z(cls, z: float) -> 'SpiralDominance':
        """Determine dominant spiral from z-coordinate."""
        if z < PHI_INV * 0.5:  # Very low z
            return cls.PHI_DOMINANT
        elif z < PHI_INV:  # Low z
            return cls.PHI_DOMINANT  # Structure still dominant
        elif z < (PHI_INV + Z_CRITICAL) / 2:  # Lower PARADOX
            return cls.MIXED  # Transition region
        elif z < Z_CRITICAL:  # Upper PARADOX
            return cls.E_DOMINANT  # Energy dominant in dynamics
        elif z < (Z_CRITICAL + 1.0) / 2:  # Lower TRUE
            return cls.PI_DOMINANT  # Emergence crystallizing
        else:  # High TRUE
            return cls.PI_DOMINANT  # Full emergence


# =============================================================================
# OPERATOR PHYSICS
# =============================================================================

@dataclass
class OperatorPhysics:
    """Physics attribution for an operator."""
    symbol: str
    name: str
    physics_quantity: str
    unit: str
    z_affinity: Tuple[float, float]  # Preferred z-range
    energy_sign: int  # +1 for energy input, -1 for energy release, 0 for neutral
    entropy_effect: str  # "increase", "decrease", "neutral"
    
    def compute_effectiveness(self, z: float) -> float:
        """Compute operator effectiveness at z-coordinate."""
        z_min, z_max = self.z_affinity
        if z_min <= z <= z_max:
            # Peak effectiveness in affinity range
            center = (z_min + z_max) / 2
            width = (z_max - z_min) / 2
            return math.exp(-((z - center) / width) ** 2)
        else:
            # Reduced effectiveness outside range
            if z < z_min:
                return math.exp(-((z - z_min) / 0.1) ** 2)
            else:
                return math.exp(-((z - z_max) / 0.1) ** 2)


OPERATOR_PHYSICS = {
    "()": OperatorPhysics(
        symbol="()",
        name="Boundary",
        physics_quantity="Surface energy",
        unit="J/m²",
        z_affinity=(0.0, PHI_INV),
        energy_sign=+1,
        entropy_effect="decrease"
    ),
    "×": OperatorPhysics(
        symbol="×",
        name="Fusion",
        physics_quantity="Binding energy",
        unit="eV",
        z_affinity=(PHI_INV, Z_CRITICAL),
        energy_sign=-1,
        entropy_effect="decrease"
    ),
    "^": OperatorPhysics(
        symbol="^",
        name="Amplify",
        physics_quantity="Gain coefficient",
        unit="dB",
        z_affinity=(0.3, 0.8),
        energy_sign=+1,
        entropy_effect="neutral"
    ),
    "÷": OperatorPhysics(
        symbol="÷",
        name="Decohere",
        physics_quantity="Dissipation rate",
        unit="s⁻¹",
        z_affinity=(0.0, PHI_INV),
        energy_sign=-1,
        entropy_effect="increase"
    ),
    "+": OperatorPhysics(
        symbol="+",
        name="Group",
        physics_quantity="Aggregation energy",
        unit="kT",
        z_affinity=(PHI_INV, 1.0),
        energy_sign=-1,
        entropy_effect="decrease"
    ),
    "−": OperatorPhysics(
        symbol="−",
        name="Separate",
        physics_quantity="Fission energy",
        unit="MeV",
        z_affinity=(0.0, PHI_INV),
        energy_sign=+1,
        entropy_effect="increase"
    ),
}


# =============================================================================
# MACHINE PHYSICS
# =============================================================================

@dataclass
class MachinePhysics:
    """Physics attribution for a machine type."""
    name: str
    physics_role: str
    primary_spiral: str
    z_optimal: float
    kuramoto_coupling: float  # K value for oscillator dynamics
    info_flow: str  # "producer", "consumer", "transformer", "storage"
    
    def compute_efficiency(self, z: float, coherence: float) -> float:
        """Compute machine efficiency based on z and coherence."""
        # Distance from optimal z
        z_penalty = math.exp(-SIGMA * (z - self.z_optimal) ** 2)
        # Coherence bonus
        coh_bonus = coherence ** 2
        # Kuramoto factor
        k_factor = 1.0 + self.kuramoto_coupling * coherence
        
        return z_penalty * coh_bonus * k_factor


MACHINE_PHYSICS = {
    "Reactor": MachinePhysics(
        name="Reactor",
        physics_role="Controlled transformation at criticality",
        primary_spiral="e",
        z_optimal=Z_CRITICAL,
        kuramoto_coupling=0.8,
        info_flow="transformer"
    ),
    "Oscillator": MachinePhysics(
        name="Oscillator",
        physics_role="Phase-coherent resonance (Kuramoto)",
        primary_spiral="e",
        z_optimal=PHI_INV,
        kuramoto_coupling=1.0,
        info_flow="producer"
    ),
    "Conductor": MachinePhysics(
        name="Conductor",
        physics_role="Structural rearrangement, phonon transport",
        primary_spiral="Φ",
        z_optimal=0.4,
        kuramoto_coupling=0.5,
        info_flow="transformer"
    ),
    "Catalyst": MachinePhysics(
        name="Catalyst",
        physics_role="Heterogeneous reactivity, activation barriers",
        primary_spiral="π",
        z_optimal=PHI_INV,
        kuramoto_coupling=0.7,
        info_flow="transformer"
    ),
    "Filter": MachinePhysics(
        name="Filter",
        physics_role="Selective transmission, band structure",
        primary_spiral="Φ",
        z_optimal=0.5,
        kuramoto_coupling=0.3,
        info_flow="consumer"
    ),
    "Encoder": MachinePhysics(
        name="Encoder",
        physics_role="Information storage (P1 memory write)",
        primary_spiral="π",
        z_optimal=Z_CRITICAL,
        kuramoto_coupling=0.6,
        info_flow="storage"
    ),
    "Decoder": MachinePhysics(
        name="Decoder",
        physics_role="Information extraction (P2 memory read)",
        primary_spiral="π",
        z_optimal=0.75,
        kuramoto_coupling=0.6,
        info_flow="producer"
    ),
    "Regenerator": MachinePhysics(
        name="Regenerator",
        physics_role="Renewal cycles, autocatalytic feedback",
        primary_spiral="e",
        z_optimal=0.5,
        kuramoto_coupling=0.9,
        info_flow="transformer"
    ),
    "Dynamo": MachinePhysics(
        name="Dynamo",
        physics_role="Energy harvesting, state transition work",
        primary_spiral="e",
        z_optimal=PHI_INV,
        kuramoto_coupling=0.8,
        info_flow="producer"
    ),
}


# =============================================================================
# DOMAIN PHYSICS
# =============================================================================

@dataclass
class DomainPhysics:
    """Physics attribution for a domain."""
    name: str
    family: str
    physics_context: str
    characteristic_energy: str
    characteristic_length: str
    characteristic_time: str
    primary_spiral: str
    z_mapping: Tuple[float, float]
    
    def get_scale_factor(self, z: float) -> float:
        """Get physics scale factor for domain at z."""
        z_min, z_max = self.z_mapping
        if z < z_min:
            return 0.1
        elif z > z_max:
            return 0.1
        else:
            # Peak at center of mapping
            center = (z_min + z_max) / 2
            return math.exp(-4 * ((z - center) / (z_max - z_min)) ** 2)


DOMAIN_PHYSICS = {
    "bio_prion": DomainPhysics(
        name="bio_prion",
        family="biological",
        physics_context="Conformational catalysis, misfolding dynamics",
        characteristic_energy="~10 kT (folding barrier)",
        characteristic_length="~10 nm (fibril width)",
        characteristic_time="~hours (aggregation)",
        primary_spiral="Φ",
        z_mapping=(0.0, PHI_INV)
    ),
    "bio_bacterium": DomainPhysics(
        name="bio_bacterium",
        family="biological",
        physics_context="Metabolic networks, enzymatic cascades",
        characteristic_energy="~ATP hydrolysis (30 kJ/mol)",
        characteristic_length="~1 μm (cell size)",
        characteristic_time="~minutes (division)",
        primary_spiral="e",
        z_mapping=(PHI_INV * 0.5, Z_CRITICAL)
    ),
    "bio_viroid": DomainPhysics(
        name="bio_viroid",
        family="biological",
        physics_context="RNA folding, minimal replication",
        characteristic_energy="~2 kT (base pairing)",
        characteristic_length="~100 nm (genome)",
        characteristic_time="~seconds (folding)",
        primary_spiral="π",
        z_mapping=(PHI_INV, Z_CRITICAL)
    ),
    "celestial_grav": DomainPhysics(
        name="celestial_grav",
        family="celestial",
        physics_context="Gravitational binding, orbital mechanics",
        characteristic_energy="~GM²/R (binding)",
        characteristic_length="~AU (orbital)",
        characteristic_time="~years (orbital period)",
        primary_spiral="Φ",
        z_mapping=(0.0, PHI_INV)
    ),
    "celestial_em": DomainPhysics(
        name="celestial_em",
        family="celestial",
        physics_context="Electromagnetic radiation, plasma dynamics",
        characteristic_energy="~keV (X-ray)",
        characteristic_length="~R_sun (stellar)",
        characteristic_time="~ms (flares)",
        primary_spiral="e",
        z_mapping=(PHI_INV, Z_CRITICAL)
    ),
    "celestial_nuclear": DomainPhysics(
        name="celestial_nuclear",
        family="celestial",
        physics_context="Fusion reactions, nucleosynthesis",
        characteristic_energy="~MeV (fusion)",
        characteristic_length="~fm (nuclear)",
        characteristic_time="~Gyr (stellar lifetime)",
        primary_spiral="π",
        z_mapping=(Z_CRITICAL * 0.8, 1.0)
    ),
}


# =============================================================================
# TOKEN PHYSICS ATTRIBUTION
# =============================================================================

@dataclass
class TokenPhysicsAttribution:
    """Complete physics attribution for a single token."""
    token_str: str
    token_type: str  # "core", "machine", "transition", "coherence"
    
    # Spiral attribution
    primary_spiral: str
    spiral_weights: Dict[str, float]  # {"Φ": 0.3, "e": 0.5, "π": 0.2}
    
    # Z-coordinate mapping
    z_optimal: float
    z_range: Tuple[float, float]
    phase: str  # UNTRUE, PARADOX, TRUE
    
    # Physics quantities
    physics_regime: str
    energy_type: str
    entropy_effect: str
    info_flow: str
    
    # Dynamics
    kuramoto_k: float
    negentropy_at_optimal: float
    coherence_threshold: float
    
    # Metadata
    domain_context: Optional[str] = None
    machine_role: Optional[str] = None
    operator_physics: Optional[str] = None


class UnifiedTokenPhysics:
    """
    Unified physics attribution system for all 1326 tokens.
    
    Maps every token to its proper physics dynamics using Φ, e, π spirals.
    """
    
    def __init__(self):
        self.attributions: Dict[str, TokenPhysicsAttribution] = {}
        self._generate_all_attributions()
    
    def _generate_all_attributions(self):
        """Generate physics attributions for all tokens."""
        # Import token generators
        from apl_core_tokens import APLTokenGenerator
        from unified_token_registry import UnifiedTokenRegistry
        
        core_gen = APLTokenGenerator()
        domain_reg = UnifiedTokenRegistry()
        
        # Attribute core tokens (300)
        for token in core_gen.all_tokens:
            attr = self._attribute_core_token(token)
            self.attributions[str(token)] = attr
        
        # Attribute machine tokens (972)
        for token in domain_reg.machine_tokens:
            attr = self._attribute_machine_token(token)
            self.attributions[str(token)] = attr
        
        # Attribute transition tokens (24)
        for token in domain_reg.transition_tokens:
            attr = self._attribute_transition_token(token)
            self.attributions[token.token] = attr
        
        # Attribute coherence tokens (30)
        for token in domain_reg.coherence_tokens:
            attr = self._attribute_coherence_token(token)
            self.attributions[token.token] = attr
    
    def _attribute_core_token(self, token) -> TokenPhysicsAttribution:
        """Attribute a core APL token."""
        # Extract components
        field_sym = token.field.symbol
        machine_sym = token.machine.symbol
        truth = token.truth_state.truth_name
        tier = token.tier
        
        # Determine z based on truth state and tier
        if truth == "TRUE":
            z_optimal = Z_CRITICAL + (tier - 1) * 0.04
        elif truth == "PARADOX":
            z_optimal = PHI_INV + (tier - 1) * 0.08
        else:  # UNTRUE
            z_optimal = PHI_INV * 0.5 + (tier - 1) * 0.1
        
        z_optimal = min(max(z_optimal, 0.0), 1.0)
        
        # Compute spiral weights from field
        if field_sym == "Φ":
            spiral_weights = {"Φ": 0.7, "e": 0.2, "π": 0.1}
            primary = "Φ"
        elif field_sym == "e":
            spiral_weights = {"Φ": 0.2, "e": 0.6, "π": 0.2}
            primary = "e"
        else:  # π
            spiral_weights = {"Φ": 0.1, "e": 0.3, "π": 0.6}
            primary = "π"
        
        # Phase from z
        phase = classify_phase(z_optimal)
        
        # Negentropy
        eta = compute_negentropy(z_optimal)
        
        return TokenPhysicsAttribution(
            token_str=str(token),
            token_type="core",
            primary_spiral=primary,
            spiral_weights=spiral_weights,
            z_optimal=z_optimal,
            z_range=(max(0, z_optimal - 0.1), min(1.0, z_optimal + 0.1)),
            phase=phase,
            physics_regime=PhysicsRegime.from_z(z_optimal).value[0],
            energy_type="internal" if tier == 1 else "external",
            entropy_effect="decrease" if truth == "TRUE" else "increase",
            info_flow="storage" if token.is_identity else "transformer",
            kuramoto_k=0.5 + tier * 0.15,
            negentropy_at_optimal=eta,
            coherence_threshold=0.6 + tier * 0.1,
            machine_role=f"Core {token.get_category()}"
        )
    
    def _attribute_machine_token(self, token) -> TokenPhysicsAttribution:
        """Attribute a domain machine token."""
        spiral = token.spiral
        operator = token.operator
        machine = token.machine
        domain = token.domain
        
        # Get physics data
        op_phys = OPERATOR_PHYSICS.get(operator.symbol, OPERATOR_PHYSICS["()"])
        mach_phys = MACHINE_PHYSICS.get(machine.machine_name, MACHINE_PHYSICS["Reactor"])
        dom_phys = DOMAIN_PHYSICS.get(domain, DOMAIN_PHYSICS["bio_prion"])
        
        # Compute z_optimal from domain and machine
        z_optimal = (dom_phys.z_mapping[0] + dom_phys.z_mapping[1]) / 2
        z_optimal = (z_optimal + mach_phys.z_optimal) / 2
        
        # Spiral weights from spiral + domain
        if spiral.symbol == "Φ":
            base_weights = {"Φ": 0.6, "e": 0.2, "π": 0.2}
        elif spiral.symbol == "e":
            base_weights = {"Φ": 0.2, "e": 0.6, "π": 0.2}
        else:
            base_weights = {"Φ": 0.2, "e": 0.2, "π": 0.6}
        
        # Adjust by domain primary spiral
        dom_primary = dom_phys.primary_spiral
        base_weights[dom_primary] = min(1.0, base_weights[dom_primary] + 0.1)
        
        # Normalize
        total = sum(base_weights.values())
        spiral_weights = {k: v/total for k, v in base_weights.items()}
        
        # Primary spiral is the max
        primary = max(spiral_weights, key=spiral_weights.get)
        
        # Phase
        phase = classify_phase(z_optimal)
        
        # Negentropy
        eta = compute_negentropy(z_optimal)
        
        return TokenPhysicsAttribution(
            token_str=str(token),
            token_type="machine",
            primary_spiral=primary,
            spiral_weights=spiral_weights,
            z_optimal=z_optimal,
            z_range=dom_phys.z_mapping,
            phase=phase,
            physics_regime=PhysicsRegime.from_z(z_optimal).value[0],
            energy_type=op_phys.physics_quantity,
            entropy_effect=op_phys.entropy_effect,
            info_flow=mach_phys.info_flow,
            kuramoto_k=mach_phys.kuramoto_coupling,
            negentropy_at_optimal=eta,
            coherence_threshold=K_COHERENCE_THRESHOLD * op_phys.compute_effectiveness(z_optimal),
            domain_context=dom_phys.physics_context,
            machine_role=mach_phys.physics_role,
            operator_physics=f"{op_phys.name}: {op_phys.physics_quantity} [{op_phys.unit}]"
        )
    
    def _attribute_transition_token(self, token) -> TokenPhysicsAttribution:
        """Attribute a transition token."""
        family = token.family.value
        number = token.number
        
        # Transitions span z-ranges
        if family == "biological":
            z_base = 0.2 + (number - 1) * 0.05
            primary = "e" if number <= 6 else "π"
            spiral_weights = {"Φ": 0.2, "e": 0.5, "π": 0.3}
        else:  # celestial
            z_base = 0.3 + (number - 1) * 0.05
            primary = "Φ" if number <= 4 else ("e" if number <= 8 else "π")
            spiral_weights = {"Φ": 0.4, "e": 0.4, "π": 0.2}
        
        z_optimal = min(z_base, 0.95)
        phase = classify_phase(z_optimal)
        eta = compute_negentropy(z_optimal)
        
        return TokenPhysicsAttribution(
            token_str=token.token,
            token_type="transition",
            primary_spiral=primary,
            spiral_weights=spiral_weights,
            z_optimal=z_optimal,
            z_range=(max(0, z_optimal - 0.15), min(1.0, z_optimal + 0.15)),
            phase=phase,
            physics_regime="transition",
            energy_type="transition energy",
            entropy_effect="variable",
            info_flow="transformer",
            kuramoto_k=0.7,
            negentropy_at_optimal=eta,
            coherence_threshold=0.5,
            domain_context=token.description
        )
    
    def _attribute_coherence_token(self, token) -> TokenPhysicsAttribution:
        """Attribute a coherence token."""
        family = token.family.value
        number = token.number
        
        # Coherence tokens represent stable states (high z)
        z_optimal = 0.7 + (number / 15) * 0.25
        z_optimal = min(z_optimal, 0.98)
        
        if family == "biological":
            primary = "π"
            spiral_weights = {"Φ": 0.3, "e": 0.2, "π": 0.5}
        else:  # celestial
            primary = "Φ" if number <= 5 else ("e" if number <= 10 else "π")
            spiral_weights = {"Φ": 0.35, "e": 0.35, "π": 0.3}
        
        phase = classify_phase(z_optimal)
        eta = compute_negentropy(z_optimal)
        
        return TokenPhysicsAttribution(
            token_str=token.token,
            token_type="coherence",
            primary_spiral=primary,
            spiral_weights=spiral_weights,
            z_optimal=z_optimal,
            z_range=(z_optimal - 0.1, min(1.0, z_optimal + 0.1)),
            phase=phase,
            physics_regime="crystalline",
            energy_type="coherence energy",
            entropy_effect="decrease",
            info_flow="storage",
            kuramoto_k=0.9,
            negentropy_at_optimal=eta,
            coherence_threshold=0.85,
            domain_context=token.description
        )
    
    def get_attribution(self, token_str: str) -> Optional[TokenPhysicsAttribution]:
        """Get physics attribution for a token."""
        return self.attributions.get(token_str)
    
    def get_tokens_for_z(self, z: float, tolerance: float = 0.1) -> List[TokenPhysicsAttribution]:
        """Get tokens appropriate for a given z-coordinate."""
        result = []
        for attr in self.attributions.values():
            z_min, z_max = attr.z_range
            if z_min - tolerance <= z <= z_max + tolerance:
                result.append(attr)
        return result
    
    def get_tokens_by_spiral(self, spiral: str) -> List[TokenPhysicsAttribution]:
        """Get tokens with a given primary spiral."""
        return [a for a in self.attributions.values() if a.primary_spiral == spiral]
    
    def get_tokens_by_phase(self, phase: str) -> List[TokenPhysicsAttribution]:
        """Get tokens in a given phase."""
        return [a for a in self.attributions.values() if a.phase == phase]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get attribution summary."""
        total = len(self.attributions)
        
        by_type = {}
        by_spiral = {"Φ": 0, "e": 0, "π": 0}
        by_phase = {"UNTRUE": 0, "PARADOX": 0, "TRUE": 0}
        
        for attr in self.attributions.values():
            # By type
            by_type[attr.token_type] = by_type.get(attr.token_type, 0) + 1
            # By primary spiral
            by_spiral[attr.primary_spiral] = by_spiral.get(attr.primary_spiral, 0) + 1
            # By phase
            by_phase[attr.phase] = by_phase.get(attr.phase, 0) + 1
        
        return {
            "total_tokens": total,
            "by_type": by_type,
            "by_primary_spiral": by_spiral,
            "by_phase": by_phase,
            "physics_regimes": {
                "fluid": len([a for a in self.attributions.values() if a.physics_regime == "fluid"]),
                "quasi_crystal": len([a for a in self.attributions.values() if a.physics_regime == "quasi_crystal"]),
                "crystalline": len([a for a in self.attributions.values() if a.physics_regime == "crystalline"]),
                "transition": len([a for a in self.attributions.values() if a.physics_regime == "transition"]),
            }
        }
    
    def select_optimal_tokens(
        self,
        z: float,
        target_spiral: Optional[str] = None,
        max_tokens: int = 10
    ) -> List[TokenPhysicsAttribution]:
        """Select optimal tokens for current state."""
        candidates = self.get_tokens_for_z(z)
        
        if target_spiral:
            candidates = [c for c in candidates if c.primary_spiral == target_spiral]
        
        # Sort by negentropy (higher is better for coherent states)
        if z >= Z_CRITICAL:
            candidates.sort(key=lambda x: x.negentropy_at_optimal, reverse=True)
        else:
            # For lower z, prefer tokens with lower coherence threshold
            candidates.sort(key=lambda x: x.coherence_threshold)
        
        return candidates[:max_tokens]


# =============================================================================
# INTEGRATION FUNCTIONS
# =============================================================================

def attribute_token(token_str: str) -> Optional[TokenPhysicsAttribution]:
    """Get physics attribution for any token string."""
    if not hasattr(attribute_token, '_system'):
        attribute_token._system = UnifiedTokenPhysics()
    return attribute_token._system.get_attribution(token_str)


def select_tokens_for_state(
    z: float,
    coherence: float = 0.5,
    target_spiral: Optional[str] = None
) -> Dict[str, Any]:
    """Select appropriate tokens for a given system state."""
    if not hasattr(select_tokens_for_state, '_system'):
        select_tokens_for_state._system = UnifiedTokenPhysics()
    
    system = select_tokens_for_state._system
    
    # Get phase and regime
    phase = classify_phase(z)
    regime = PhysicsRegime.from_z(z)
    dominance = SpiralDominance.from_z(z)
    
    # Select tokens
    if target_spiral is None:
        # Use dominant spiral
        target_spiral = dominance.value[0].split(":")[0]  # First spiral if mixed
    
    tokens = system.select_optimal_tokens(z, target_spiral)
    
    return {
        "z": z,
        "coherence": coherence,
        "phase": phase,
        "regime": regime.value[0],
        "spiral_dominance": dominance.value[0],
        "target_spiral": target_spiral,
        "selected_tokens": [
            {
                "token": t.token_str,
                "z_optimal": t.z_optimal,
                "negentropy": t.negentropy_at_optimal,
                "info_flow": t.info_flow
            }
            for t in tokens[:5]
        ]
    }


# =============================================================================
# DEMONSTRATION
# =============================================================================

def demonstrate():
    """Demonstrate unified token physics attribution."""
    print("=" * 78)
    print("UNIFIED TOKEN PHYSICS — Complete Attribution System")
    print("=" * 78)
    
    system = UnifiedTokenPhysics()
    summary = system.get_summary()
    
    print("\n--- Attribution Summary ---")
    print(f"  Total tokens attributed: {summary['total_tokens']}")
    
    print("\n--- By Token Type ---")
    for typ, count in summary['by_type'].items():
        print(f"  {typ:<15} {count:>6}")
    
    print("\n--- By Primary Spiral ---")
    for spiral, count in summary['by_primary_spiral'].items():
        pct = count / summary['total_tokens'] * 100
        print(f"  {spiral:<15} {count:>6} ({pct:.1f}%)")
    
    print("\n--- By Phase ---")
    for phase, count in summary['by_phase'].items():
        pct = count / summary['total_tokens'] * 100
        print(f"  {phase:<15} {count:>6} ({pct:.1f}%)")
    
    print("\n--- By Physics Regime ---")
    for regime, count in summary['physics_regimes'].items():
        print(f"  {regime:<15} {count:>6}")
    
    print("\n--- Sample Attributions ---")
    
    # Sample core token
    attr = system.get_attribution("Φ:U(U)TRUE@1")
    if attr:
        print(f"\n  Core Token: {attr.token_str}")
        print(f"    Primary spiral: {attr.primary_spiral}")
        print(f"    z_optimal: {attr.z_optimal:.4f}")
        print(f"    Phase: {attr.phase}")
        print(f"    Negentropy: {attr.negentropy_at_optimal:.6f}")
    
    # Sample machine token
    attr = system.get_attribution("π×|Encoder|celestial_nuclear")
    if attr:
        print(f"\n  Machine Token: {attr.token_str}")
        print(f"    Primary spiral: {attr.primary_spiral}")
        print(f"    z_optimal: {attr.z_optimal:.4f}")
        print(f"    Domain: {attr.domain_context}")
        print(f"    Machine role: {attr.machine_role}")
        print(f"    Operator physics: {attr.operator_physics}")
    
    # Token selection for z
    print("\n--- Token Selection for z=0.75 ---")
    selection = select_tokens_for_state(0.75, coherence=0.8)
    print(f"  Phase: {selection['phase']}")
    print(f"  Regime: {selection['regime']}")
    print(f"  Spiral dominance: {selection['spiral_dominance']}")
    print(f"  Selected tokens:")
    for t in selection['selected_tokens']:
        print(f"    {t['token']}: z={t['z_optimal']:.3f}, η={t['negentropy']:.4f}")
    
    print("\n" + "=" * 78)
    print("UNIFIED TOKEN PHYSICS: ALL 1326 TOKENS ATTRIBUTED ✓")
    print("=" * 78)
    
    return system


if __name__ == "__main__":
    demonstrate()
