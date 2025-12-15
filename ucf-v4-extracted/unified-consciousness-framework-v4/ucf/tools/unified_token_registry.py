#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  UNIFIED TOKEN REGISTRY                                                       ║
║  APL Core (300) + Domain Token Sets (1026) = 1326 Total Tokens                ║
║  Part of the Unified Consciousness Framework                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

TOKEN HIERARCHY:

┌─────────────────────────────────────────────────────────────────────────────┐
│  TIER 1: APL CORE SET (300 Tokens)                                          │
│  ─────────────────────────────────────────────────────────────────────────  │
│  162 Identity Tokens     Field:Machine(Machine)TruthState@Tier              │
│   54 Meta-Operators      Field:M(operator)TruthState@2                      │
│   54 Domain Selectors    Field:Machine(domain)UNTRUE@3                      │
│   30 Safety Tokens       Field:M(safety_level)PARADOX@Tier                  │
├─────────────────────────────────────────────────────────────────────────────┤
│  TIER 2: DOMAIN TOKEN SETS (1026 Tokens)                                    │
│  ─────────────────────────────────────────────────────────────────────────  │
│  972 Machine Tokens      [Spiral][Operator]|[Machine]|[Domain]              │
│   24 Transition Tokens   [family]_transition_[1-12]                         │
│   30 Coherence Tokens    [family]_coherence_[1-15]                          │
├─────────────────────────────────────────────────────────────────────────────┤
│  GRAND TOTAL: 1326 Tokens                                                   │
└─────────────────────────────────────────────────────────────────────────────┘

DOMAIN FAMILIES:
  Biological: bio_prion, bio_bacterium, bio_viroid
  Celestial:  celestial_grav, celestial_em, celestial_nuclear

Integration Points:
  - apl_core_tokens.py: 300-token foundational grammar
  - nuclear_spinner.py: 972 machine tokens (9 machines × 6 domains)
  - tool_shed.py: token_index tool for queries
  - emission_pipeline.py: token-driven language generation

Signature: Δ|unified-token-registry|1326-tokens|complete|Ω
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set, Iterator
from enum import Enum, auto

# Import from sibling modules
from ucf.core.physics_constants import (
    PHI, PHI_INV, Z_CRITICAL, SIGMA, SIGMA_INV,
    compute_negentropy, classify_phase, get_tier
)

# =============================================================================
# ENUMERATIONS
# =============================================================================

class TokenTier(Enum):
    """Token system tiers."""
    CORE = "core"           # APL Core 300
    DOMAIN = "domain"       # Domain Token Sets 1026

class Spiral(Enum):
    """The three field types (spirals)."""
    PHI = ("Φ", "Structure", "geometry")
    E = ("e", "Energy", "wave")
    PI = ("π", "Emergence", "emergence")
    
    def __init__(self, symbol: str, name: str, domain: str):
        self._symbol = symbol
        self._name = name
        self._domain = domain
    
    @property
    def symbol(self) -> str:
        return self._symbol

class Operator(Enum):
    """The six universal operators."""
    BOUNDARY = ("()", "Boundary", "Containment, gating")
    FUSION = ("×", "Fusion", "Coupling, convergence")
    AMPLIFY = ("^", "Amplify", "Gain, excitation")
    DECOHERE = ("÷", "Decohere", "Dissipation, reset")
    GROUP = ("+", "Group", "Aggregation, clustering")
    SEPARATE = ("−", "Separate", "Splitting, fission")
    
    def __init__(self, symbol: str, name: str, description: str):
        self._symbol = symbol
        self._name = name
        self._description = description
    
    @property
    def symbol(self) -> str:
        return self._symbol

class Machine(Enum):
    """The nine archetypal machines."""
    REACTOR = ("Reactor", "Controlled transformation at criticality")
    OSCILLATOR = ("Oscillator", "Phase-coherent resonance (Kuramoto)")
    CONDUCTOR = ("Conductor", "Structural rearrangement, relaxation")
    CATALYST = ("Catalyst", "Heterogeneous reactivity, emergence check")
    FILTER = ("Filter", "Selective information passing")
    ENCODER = ("Encoder", "Information storage (P1)")
    DECODER = ("Decoder", "Information extraction (P2)")
    REGENERATOR = ("Regenerator", "Renewal, autocatalytic cycles")
    DYNAMO = ("Dynamo", "Energy harvesting from state transitions")
    
    def __init__(self, name: str, description: str):
        self._machine_name = name
        self._description = description
    
    @property
    def machine_name(self) -> str:
        return self._machine_name

class DomainFamily(Enum):
    """Domain families."""
    BIOLOGICAL = "biological"
    CELESTIAL = "celestial"

class BioDomain(Enum):
    """Biological subdomains."""
    PRION = ("bio_prion", "Misfolded protein aggregates, conformational catalysis")
    BACTERIUM = ("bio_bacterium", "Single-cell organisms, metabolic networks")
    VIROID = ("bio_viroid", "RNA-only replicators, minimal genetic systems")
    
    def __init__(self, token: str, description: str):
        self._token = token
        self._description = description
    
    @property
    def token(self) -> str:
        return self._token

class CelestialDomain(Enum):
    """Celestial subdomains."""
    GRAV = ("celestial_grav", "Gravitational dynamics, orbital mechanics")
    EM = ("celestial_em", "Electromagnetic phenomena, radiation processes")
    NUCLEAR = ("celestial_nuclear", "Nuclear fusion, stellar nucleosynthesis")
    
    def __init__(self, token: str, description: str):
        self._token = token
        self._description = description
    
    @property
    def token(self) -> str:
        return self._token

# =============================================================================
# TRANSITION TOKENS (24 Total)
# =============================================================================

BIO_TRANSITIONS = {
    "bio_transition_1": "Conformational shift (prion-like template propagation)",
    "bio_transition_2": "Metabolic phase transition (quiescence ↔ active growth)",
    "bio_transition_3": "Replication initiation (dormant → copying)",
    "bio_transition_4": "Aggregation clustering (monomers → oligomers → fibrils)",
    "bio_transition_5": "Horizontal gene transfer analogue (sequence exchange)",
    "bio_transition_6": "Error threshold crossing (stable → error catastrophe)",
    "bio_transition_7": "Catalytic onset (passive → enzymatic)",
    "bio_transition_8": "Compartmentalization (free → membrane-bound)",
    "bio_transition_9": "Symbiotic coupling (independent → mutualistic)",
    "bio_transition_10": "Dormancy induction (stress response, spore formation)",
    "bio_transition_11": "Phenotypic switch (bistability, hysteresis in gene circuits)",
    "bio_transition_12": "Extinction/clearance (population collapse, immune clearance)",
}

CELESTIAL_TRANSITIONS = {
    "celestial_transition_1": "Gravitational collapse (cloud → protostar)",
    "celestial_transition_2": "Fusion ignition (protostar → main sequence star)",
    "celestial_transition_3": "Main sequence exit (hydrogen depletion)",
    "celestial_transition_4": "Red giant phase (shell burning, envelope expansion)",
    "celestial_transition_5": "Planetary nebula ejection (envelope loss)",
    "celestial_transition_6": "White dwarf cooling (degenerate remnant)",
    "celestial_transition_7": "Supernova explosion (core collapse or Type Ia)",
    "celestial_transition_8": "Neutron star formation (post-supernova collapse)",
    "celestial_transition_9": "Black hole formation (beyond neutron degeneracy)",
    "celestial_transition_10": "Accretion disk formation (matter infall, angular momentum)",
    "celestial_transition_11": "Magnetospheric coupling (field-dominated dynamics)",
    "celestial_transition_12": "Tidal disruption event (gravitational shearing)",
}

# =============================================================================
# COHERENCE TOKENS (30 Total)
# =============================================================================

BIO_COHERENCE = {
    "bio_coherence_1": "Amyloid fibril structure (cross-β stacking)",
    "bio_coherence_2": "Biofilm matrix (extracellular polymer networks)",
    "bio_coherence_3": "Quorum sensing synchrony (population-level coordination)",
    "bio_coherence_4": "Circadian oscillator (biochemical clock)",
    "bio_coherence_5": "Metabolic cycle (glycolysis, TCA, circadian metabolite rhythms)",
    "bio_coherence_6": "RNA secondary structure (stem-loops, pseudoknots)",
    "bio_coherence_7": "Ribozyme catalytic core (conserved tertiary motif)",
    "bio_coherence_8": "Viral capsid geometry (icosahedral symmetry)",
    "bio_coherence_9": "Quasispecies cloud (error-coupled replicator ensemble)",
    "bio_coherence_10": "Protein folding funnel (energy landscape convergence)",
    "bio_coherence_11": "Allosteric network (long-range coupling in proteins)",
    "bio_coherence_12": "Gene regulatory motif (feed-forward loop, toggle switch)",
    "bio_coherence_13": "Chemotaxis gradient sensing (spatial information processing)",
    "bio_coherence_14": "Autoinducer feedback loop (self-reinforcing signaling)",
    "bio_coherence_15": "Replication-transcription coupling (co-localized synthesis)",
}

CELESTIAL_COHERENCE = {
    "celestial_coherence_1": "Keplerian orbit (stable elliptical motion)",
    "celestial_coherence_2": "Lagrange point equilibrium (gravitational balance)",
    "celestial_coherence_3": "Tidal locking (synchronous rotation)",
    "celestial_coherence_4": "Roche lobe geometry (equipotential surface)",
    "celestial_coherence_5": "Magnetic dynamo (self-sustaining field generation)",
    "celestial_coherence_6": "Stellar convection cell (Bénard-like circulation)",
    "celestial_coherence_7": "Accretion disk structure (Keplerian shear flow)",
    "celestial_coherence_8": "Magnetospheric current sheet (field topology)",
    "celestial_coherence_9": "Radiation pressure equilibrium (photon-matter balance)",
    "celestial_coherence_10": "Nuclear burning shell (fusion layer stratification)",
    "celestial_coherence_11": "Pulsar beaming cone (lighthouse effect)",
    "celestial_coherence_12": "Gravitational wave chirp (inspiral signature)",
    "celestial_coherence_13": "Plasma oscillation mode (Langmuir, Alfvén waves)",
    "celestial_coherence_14": "Magnetic reconnection site (topology change, energy release)",
    "celestial_coherence_15": "Neutron star crust lattice (nuclear pasta phases)",
}

# =============================================================================
# TOKEN DATA STRUCTURES
# =============================================================================

@dataclass
class MachineToken:
    """A machine token from the Domain Token Sets (972 total)."""
    spiral: Spiral
    operator: Operator
    machine: Machine
    domain: str
    
    def __str__(self) -> str:
        return f"{self.spiral.symbol}{self.operator.symbol}|{self.machine.machine_name}|{self.domain}"
    
    def __hash__(self) -> int:
        return hash(str(self))
    
    @classmethod
    def parse(cls, token_str: str) -> Optional['MachineToken']:
        """Parse token string into MachineToken."""
        try:
            parts = token_str.split("|")
            if len(parts) != 3:
                return None
            
            spiral_op = parts[0]
            if len(spiral_op) < 2:
                return None
            
            # Parse spiral
            spiral_sym = spiral_op[0]
            spiral = None
            for s in Spiral:
                if s.symbol == spiral_sym:
                    spiral = s
                    break
            if not spiral:
                return None
            
            # Parse operator
            op_sym = spiral_op[1:]
            operator = None
            for o in Operator:
                if o.symbol == op_sym:
                    operator = o
                    break
            if not operator:
                return None
            
            # Parse machine
            machine_name = parts[1]
            machine = None
            for m in Machine:
                if m.machine_name == machine_name:
                    machine = m
                    break
            if not machine:
                return None
            
            domain = parts[2]
            
            return cls(spiral=spiral, operator=operator, machine=machine, domain=domain)
        except Exception:
            return None

@dataclass
class TransitionToken:
    """A transition token (24 total)."""
    token: str
    family: DomainFamily
    number: int
    description: str
    
    def __str__(self) -> str:
        return self.token
    
    def __hash__(self) -> int:
        return hash(self.token)

@dataclass
class CoherenceToken:
    """A coherence token (30 total)."""
    token: str
    family: DomainFamily
    number: int
    description: str
    
    def __str__(self) -> str:
        return self.token
    
    def __hash__(self) -> int:
        return hash(self.token)

# =============================================================================
# UNIFIED TOKEN REGISTRY
# =============================================================================

class UnifiedTokenRegistry:
    """
    Unified registry for all 1326 APL tokens.
    
    Combines:
    - APL Core Set (300 tokens)
    - Domain Token Sets (1026 tokens)
    """
    
    def __init__(self):
        self.machine_tokens: List[MachineToken] = []
        self.transition_tokens: List[TransitionToken] = []
        self.coherence_tokens: List[CoherenceToken] = []
        
        self._generate_all()
    
    def _generate_all(self):
        """Generate all domain tokens."""
        self._generate_machine_tokens()
        self._generate_transition_tokens()
        self._generate_coherence_tokens()
    
    def _generate_machine_tokens(self):
        """Generate all 972 machine tokens."""
        domains = [
            BioDomain.PRION.token,
            BioDomain.BACTERIUM.token,
            BioDomain.VIROID.token,
            CelestialDomain.GRAV.token,
            CelestialDomain.EM.token,
            CelestialDomain.NUCLEAR.token,
        ]
        
        for spiral in Spiral:
            for operator in Operator:
                for machine in Machine:
                    for domain in domains:
                        token = MachineToken(
                            spiral=spiral,
                            operator=operator,
                            machine=machine,
                            domain=domain
                        )
                        self.machine_tokens.append(token)
    
    def _generate_transition_tokens(self):
        """Generate all 24 transition tokens."""
        for token, desc in BIO_TRANSITIONS.items():
            num = int(token.split("_")[-1])
            self.transition_tokens.append(TransitionToken(
                token=token,
                family=DomainFamily.BIOLOGICAL,
                number=num,
                description=desc
            ))
        
        for token, desc in CELESTIAL_TRANSITIONS.items():
            num = int(token.split("_")[-1])
            self.transition_tokens.append(TransitionToken(
                token=token,
                family=DomainFamily.CELESTIAL,
                number=num,
                description=desc
            ))
    
    def _generate_coherence_tokens(self):
        """Generate all 30 coherence tokens."""
        for token, desc in BIO_COHERENCE.items():
            num = int(token.split("_")[-1])
            self.coherence_tokens.append(CoherenceToken(
                token=token,
                family=DomainFamily.BIOLOGICAL,
                number=num,
                description=desc
            ))
        
        for token, desc in CELESTIAL_COHERENCE.items():
            num = int(token.split("_")[-1])
            self.coherence_tokens.append(CoherenceToken(
                token=token,
                family=DomainFamily.CELESTIAL,
                number=num,
                description=desc
            ))
    
    def get_summary(self) -> Dict[str, Any]:
        """Get registry summary."""
        return {
            "machine_tokens": len(self.machine_tokens),
            "transition_tokens": len(self.transition_tokens),
            "coherence_tokens": len(self.coherence_tokens),
            "domain_total": len(self.machine_tokens) + len(self.transition_tokens) + len(self.coherence_tokens),
            "core_tokens": 300,  # From apl_core_tokens.py
            "grand_total": 300 + len(self.machine_tokens) + len(self.transition_tokens) + len(self.coherence_tokens),
            "by_domain": {
                "bio_prion": len([t for t in self.machine_tokens if t.domain == "bio_prion"]),
                "bio_bacterium": len([t for t in self.machine_tokens if t.domain == "bio_bacterium"]),
                "bio_viroid": len([t for t in self.machine_tokens if t.domain == "bio_viroid"]),
                "celestial_grav": len([t for t in self.machine_tokens if t.domain == "celestial_grav"]),
                "celestial_em": len([t for t in self.machine_tokens if t.domain == "celestial_em"]),
                "celestial_nuclear": len([t for t in self.machine_tokens if t.domain == "celestial_nuclear"]),
            },
            "by_spiral": {
                s.symbol: len([t for t in self.machine_tokens if t.spiral == s])
                for s in Spiral
            },
            "by_machine": {
                m.machine_name: len([t for t in self.machine_tokens if t.machine == m])
                for m in Machine
            },
        }
    
    def get_tokens_by_domain(self, domain: str) -> List[MachineToken]:
        """Get all machine tokens for a domain."""
        return [t for t in self.machine_tokens if t.domain == domain]
    
    def get_tokens_by_spiral(self, spiral: Spiral) -> List[MachineToken]:
        """Get all machine tokens for a spiral."""
        return [t for t in self.machine_tokens if t.spiral == spiral]
    
    def get_tokens_by_machine(self, machine: Machine) -> List[MachineToken]:
        """Get all machine tokens for a machine type."""
        return [t for t in self.machine_tokens if t.machine == machine]
    
    def get_tokens_by_operator(self, operator: Operator) -> List[MachineToken]:
        """Get all machine tokens for an operator."""
        return [t for t in self.machine_tokens if t.operator == operator]
    
    def get_transitions_by_family(self, family: DomainFamily) -> List[TransitionToken]:
        """Get transition tokens for a domain family."""
        return [t for t in self.transition_tokens if t.family == family]
    
    def get_coherence_by_family(self, family: DomainFamily) -> List[CoherenceToken]:
        """Get coherence tokens for a domain family."""
        return [t for t in self.coherence_tokens if t.family == family]
    
    def lookup_machine(self, token_str: str) -> Optional[MachineToken]:
        """Look up a machine token by string."""
        for t in self.machine_tokens:
            if str(t) == token_str:
                return t
        return None
    
    def lookup_transition(self, token_str: str) -> Optional[TransitionToken]:
        """Look up a transition token by string."""
        for t in self.transition_tokens:
            if t.token == token_str:
                return t
        return None
    
    def lookup_coherence(self, token_str: str) -> Optional[CoherenceToken]:
        """Look up a coherence token by string."""
        for t in self.coherence_tokens:
            if t.token == token_str:
                return t
        return None
    
    def select_for_z(self, z: float) -> Dict[str, Any]:
        """Select appropriate tokens for a given z-coordinate."""
        phase = classify_phase(z)
        tier = get_tier(z)
        
        # Select spiral based on z
        if z < 0.33:
            primary_spiral = Spiral.PHI
        elif z < 0.66:
            primary_spiral = Spiral.E
        else:
            primary_spiral = Spiral.PI
        
        # Select operators based on phase
        if phase == "TRUE":
            operators = [Operator.FUSION, Operator.AMPLIFY, Operator.GROUP]
        elif phase == "PARADOX":
            operators = [Operator.BOUNDARY, Operator.FUSION, Operator.GROUP, Operator.SEPARATE]
        else:  # UNTRUE
            operators = [Operator.BOUNDARY, Operator.DECOHERE, Operator.SEPARATE]
        
        # Get matching tokens
        matching = [
            t for t in self.machine_tokens
            if t.spiral == primary_spiral and t.operator in operators
        ]
        
        return {
            "z": z,
            "phase": phase,
            "tier": tier,
            "primary_spiral": primary_spiral.symbol,
            "operators": [o.symbol for o in operators],
            "matching_tokens": len(matching),
            "sample": [str(t) for t in matching[:10]]
        }


# =============================================================================
# DEMONSTRATION
# =============================================================================

def demonstrate_registry():
    """Demonstrate the unified token registry."""
    print("=" * 70)
    print("UNIFIED TOKEN REGISTRY — APL Core + Domain Token Sets")
    print("=" * 70)
    
    registry = UnifiedTokenRegistry()
    summary = registry.get_summary()
    
    print("\n--- Token Count Summary ---")
    print(f"  APL Core Set:     {summary['core_tokens']:>6}")
    print(f"  Machine Tokens:   {summary['machine_tokens']:>6}")
    print(f"  Transition Tokens:{summary['transition_tokens']:>6}")
    print(f"  Coherence Tokens: {summary['coherence_tokens']:>6}")
    print(f"  ─────────────────────────")
    print(f"  GRAND TOTAL:      {summary['grand_total']:>6}")
    
    print("\n--- By Domain ---")
    for domain, count in summary['by_domain'].items():
        print(f"  {domain:<20} {count:>4}")
    
    print("\n--- By Spiral ---")
    for spiral, count in summary['by_spiral'].items():
        print(f"  {spiral:<20} {count:>4}")
    
    print("\n--- By Machine ---")
    for machine, count in summary['by_machine'].items():
        print(f"  {machine:<20} {count:>4}")
    
    print("\n--- Sample Machine Tokens ---")
    for token in registry.machine_tokens[:6]:
        print(f"  {token}")
    
    print("\n--- Sample Transition Tokens ---")
    for token in registry.transition_tokens[:6]:
        print(f"  {token.token}: {token.description[:50]}...")
    
    print("\n--- Sample Coherence Tokens ---")
    for token in registry.coherence_tokens[:6]:
        print(f"  {token.token}: {token.description[:50]}...")
    
    print("\n--- Token Selection for z=0.75 ---")
    selection = registry.select_for_z(0.75)
    print(f"  Phase: {selection['phase']}")
    print(f"  Spiral: {selection['primary_spiral']}")
    print(f"  Operators: {selection['operators']}")
    print(f"  Matching: {selection['matching_tokens']} tokens")
    print(f"  Sample: {selection['sample'][:3]}")
    
    print("\n" + "=" * 70)
    print("UNIFIED TOKEN REGISTRY: COMPLETE")
    print("=" * 70)
    
    return registry


if __name__ == "__main__":
    demonstrate_registry()
