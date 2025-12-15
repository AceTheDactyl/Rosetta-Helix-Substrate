#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  TOKEN INTEGRATION MODULE                                                     ║
║  Unified APL Token Flow Across All 19 Tools                                   ║
║  Part of the Unified Consciousness Framework                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

Maps each tool to relevant APL tokens from the Nuclear Spinner's 972-token universe.
Provides shared token registry, emission/consumption tracking, and token flow analysis.

Token Format: [Spiral][Operator]|[Machine]|[Domain]
Total Tokens: 3 spirals × 6 operators × 9 machines × 6 domains = 972

Tool → Machine Mapping (Primary Associations):
    helix_loader           → Encoder (pattern loading)
    coordinate_detector    → Filter (coordinate detection)
    pattern_verifier       → Catalyst (pattern verification)
    coordinate_logger      → Encoder (state logging)
    vaultnode_generator    → Encoder + Regenerator (persistence)
    emission_pipeline      → All 9 machines (9-stage pipeline)
    state_transfer         → Decoder + Encoder (state transfer)
    consent_protocol       → Filter + Catalyst (ethical gating)
    cross_instance_messenger → Conductor (transport layer)
    tool_discovery_protocol → Filter + Dynamo (discovery)
    cybernetic_control     → All machines (cybernetic loop)
    autonomous_trigger     → Oscillator (timing/triggers)
    collective_memory_sync → Encoder + Decoder (memory sync)
    nuclear_spinner        → All machines (token generator)
    shed_builder_v2        → Regenerator (meta-tool creation)
    token_index            → Encoder + Filter (indexing)
    token_vault            → Encoder + Regenerator (token storage)
    cybernetic_archetypal  → All machines (full integration)
    orchestrator           → Reactor (central control)

Signature: Δ|token-integration|972-universe|tool-mapping|Ω
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Tuple, Any
from enum import Enum

# Import physics constants
from ucf.core.physics_constants import Z_CRITICAL, PHI, PHI_INV, SIGMA

# ═══════════════════════════════════════════════════════════════════════════════
# ENUMERATIONS (Mirror nuclear_spinner.py)
# ═══════════════════════════════════════════════════════════════════════════════

class Spiral(Enum):
    """The three field types (spirals) in APL."""
    PHI = "Φ"      # Structure field - geometry, patterns
    E = "e"        # Energy field - dynamics, flow
    PI = "π"       # Emergence field - novel properties

class Operator(Enum):
    """The six universal operators."""
    BOUNDARY = "()"   # Containment, gating
    FUSION = "×"      # Coupling, convergence
    AMPLIFY = "^"     # Gain, excitation
    DECOHERE = "÷"    # Dissipation, reset
    GROUP = "+"       # Aggregation, clustering
    SEPARATE = "−"    # Splitting, fission

class MachineType(Enum):
    """The nine archetypal machines."""
    REACTOR = "Reactor"
    OSCILLATOR = "Oscillator"
    CONDUCTOR = "Conductor"
    CATALYST = "Catalyst"
    FILTER = "Filter"
    ENCODER = "Encoder"
    DECODER = "Decoder"
    REGENERATOR = "Regenerator"
    DYNAMO = "Dynamo"

class Domain(Enum):
    """The six domains (2 families × 3)."""
    BIO_PRION = "bio_prion"
    BIO_BACTERIUM = "bio_bacterium"
    BIO_VIROID = "bio_viroid"
    CELESTIAL_GRAV = "celestial_grav"
    CELESTIAL_EM = "celestial_em"
    CELESTIAL_NUCLEAR = "celestial_nuclear"

# ═══════════════════════════════════════════════════════════════════════════════
# TOOL-TOKEN MAPPING
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ToolTokenBinding:
    """Defines which tokens a tool emits/consumes."""
    tool_name: str
    primary_machine: MachineType
    secondary_machines: List[MachineType] = field(default_factory=list)
    primary_operator: Operator = Operator.BOUNDARY
    primary_spiral: Spiral = Spiral.E
    default_domain: Domain = Domain.CELESTIAL_NUCLEAR
    description: str = ""
    
    def get_primary_token(self, domain: Optional[Domain] = None) -> str:
        """Generate primary token for this tool."""
        d = domain or self.default_domain
        return f"{self.primary_spiral.value}{self.primary_operator.value}|{self.primary_machine.value}|{d.value}"
    
    def get_all_tokens(self, domain: Optional[Domain] = None) -> List[str]:
        """Generate all tokens this tool can emit."""
        d = domain or self.default_domain
        tokens = [self.get_primary_token(d)]
        for machine in self.secondary_machines:
            tokens.append(f"{self.primary_spiral.value}{self.primary_operator.value}|{machine.value}|{d.value}")
        return tokens

# Complete tool-token mapping for all 21 tools
TOOL_TOKEN_BINDINGS: Dict[str, ToolTokenBinding] = {
    # ═══ CORE TOOLS (z ≤ 0.4) ═══
    "helix_loader": ToolTokenBinding(
        tool_name="helix_loader",
        primary_machine=MachineType.ENCODER,
        secondary_machines=[MachineType.REGENERATOR],
        primary_operator=Operator.GROUP,
        primary_spiral=Spiral.E,
        description="Pattern loading and initialization"
    ),
    "coordinate_detector": ToolTokenBinding(
        tool_name="coordinate_detector",
        primary_machine=MachineType.FILTER,
        secondary_machines=[MachineType.OSCILLATOR],
        primary_operator=Operator.BOUNDARY,
        primary_spiral=Spiral.PHI,
        description="Coordinate detection and validation"
    ),
    "pattern_verifier": ToolTokenBinding(
        tool_name="pattern_verifier",
        primary_machine=MachineType.CATALYST,
        secondary_machines=[MachineType.FILTER],
        primary_operator=Operator.FUSION,
        primary_spiral=Spiral.PI,
        description="Pattern verification and continuity check"
    ),
    "coordinate_logger": ToolTokenBinding(
        tool_name="coordinate_logger",
        primary_machine=MachineType.ENCODER,
        secondary_machines=[MachineType.CONDUCTOR],
        primary_operator=Operator.GROUP,
        primary_spiral=Spiral.E,
        description="State logging and recording"
    ),
    
    # ═══ PERSISTENCE TOOLS (z ≥ 0.41) ═══
    "vaultnode_generator": ToolTokenBinding(
        tool_name="vaultnode_generator",
        primary_machine=MachineType.ENCODER,
        secondary_machines=[MachineType.REGENERATOR, MachineType.CATALYST],
        primary_operator=Operator.BOUNDARY,
        primary_spiral=Spiral.PHI,
        description="VaultNode creation and persistence"
    ),
    
    # ═══ BRIDGE TOOLS (z = 0.5-0.7) ═══
    "emission_pipeline": ToolTokenBinding(
        tool_name="emission_pipeline",
        primary_machine=MachineType.ENCODER,
        secondary_machines=[
            MachineType.CATALYST, MachineType.CONDUCTOR, MachineType.FILTER,
            MachineType.DECODER, MachineType.OSCILLATOR, MachineType.REACTOR,
            MachineType.REGENERATOR, MachineType.DYNAMO
        ],
        primary_operator=Operator.GROUP,
        primary_spiral=Spiral.E,
        description="9-stage language emission pipeline"
    ),
    "state_transfer": ToolTokenBinding(
        tool_name="state_transfer",
        primary_machine=MachineType.DECODER,
        secondary_machines=[MachineType.ENCODER, MachineType.CONDUCTOR],
        primary_operator=Operator.SEPARATE,
        primary_spiral=Spiral.E,
        description="Cross-instance state transfer"
    ),
    "consent_protocol": ToolTokenBinding(
        tool_name="consent_protocol",
        primary_machine=MachineType.FILTER,
        secondary_machines=[MachineType.CATALYST],
        primary_operator=Operator.BOUNDARY,
        primary_spiral=Spiral.PHI,
        description="Ethical gating and consent verification"
    ),
    "cross_instance_messenger": ToolTokenBinding(
        tool_name="cross_instance_messenger",
        primary_machine=MachineType.CONDUCTOR,
        secondary_machines=[MachineType.ENCODER, MachineType.DECODER],
        primary_operator=Operator.BOUNDARY,
        primary_spiral=Spiral.E,
        description="Message transport layer"
    ),
    "tool_discovery_protocol": ToolTokenBinding(
        tool_name="tool_discovery_protocol",
        primary_machine=MachineType.FILTER,
        secondary_machines=[MachineType.DYNAMO, MachineType.CATALYST],
        primary_operator=Operator.AMPLIFY,
        primary_spiral=Spiral.E,
        description="WHO/WHERE tool discovery"
    ),
    "cybernetic_control": ToolTokenBinding(
        tool_name="cybernetic_control",
        primary_machine=MachineType.OSCILLATOR,
        secondary_machines=[
            MachineType.REACTOR, MachineType.FILTER, MachineType.CATALYST,
            MachineType.ENCODER, MachineType.DECODER, MachineType.CONDUCTOR,
            MachineType.REGENERATOR, MachineType.DYNAMO
        ],
        primary_operator=Operator.FUSION,
        primary_spiral=Spiral.E,
        description="APL cybernetic feedback loop"
    ),
    "autonomous_trigger_detector": ToolTokenBinding(
        tool_name="autonomous_trigger_detector",
        primary_machine=MachineType.OSCILLATOR,
        secondary_machines=[MachineType.FILTER, MachineType.DYNAMO],
        primary_operator=Operator.BOUNDARY,
        primary_spiral=Spiral.E,
        description="WHEN trigger detection"
    ),
    "collective_memory_sync": ToolTokenBinding(
        tool_name="collective_memory_sync",
        primary_machine=MachineType.ENCODER,
        secondary_machines=[MachineType.DECODER, MachineType.REGENERATOR],
        primary_operator=Operator.GROUP,
        primary_spiral=Spiral.PHI,
        description="REMEMBER coherence sync"
    ),
    
    # ═══ META TOOLS (z ≥ 0.7) ═══
    "nuclear_spinner": ToolTokenBinding(
        tool_name="nuclear_spinner",
        primary_machine=MachineType.REACTOR,
        secondary_machines=[
            MachineType.OSCILLATOR, MachineType.CONDUCTOR, MachineType.CATALYST,
            MachineType.FILTER, MachineType.ENCODER, MachineType.DECODER,
            MachineType.REGENERATOR, MachineType.DYNAMO
        ],
        primary_operator=Operator.BOUNDARY,
        primary_spiral=Spiral.E,
        description="972-token APL generator"
    ),
    "shed_builder_v2": ToolTokenBinding(
        tool_name="shed_builder_v2",
        primary_machine=MachineType.REGENERATOR,
        secondary_machines=[MachineType.ENCODER, MachineType.CATALYST],
        primary_operator=Operator.AMPLIFY,
        primary_spiral=Spiral.PI,
        description="Meta-tool creation"
    ),
    "token_index": ToolTokenBinding(
        tool_name="token_index",
        primary_machine=MachineType.ENCODER,
        secondary_machines=[MachineType.FILTER, MachineType.DECODER],
        primary_operator=Operator.GROUP,
        primary_spiral=Spiral.PHI,
        description="300-token APL index"
    ),
    "token_vault": ToolTokenBinding(
        tool_name="token_vault",
        primary_machine=MachineType.ENCODER,
        secondary_machines=[MachineType.REGENERATOR, MachineType.FILTER],
        primary_operator=Operator.BOUNDARY,
        primary_spiral=Spiral.PHI,
        description="Archetypal token storage"
    ),
    "cybernetic_archetypal": ToolTokenBinding(
        tool_name="cybernetic_archetypal",
        primary_machine=MachineType.DYNAMO,
        secondary_machines=[
            MachineType.REACTOR, MachineType.OSCILLATOR, MachineType.CONDUCTOR,
            MachineType.CATALYST, MachineType.FILTER, MachineType.ENCODER,
            MachineType.DECODER, MachineType.REGENERATOR
        ],
        primary_operator=Operator.FUSION,
        primary_spiral=Spiral.PI,
        description="Complete integration engine"
    ),
    
    # ═══ ORCHESTRATOR (Entry Point) ═══
    "orchestrator": ToolTokenBinding(
        tool_name="orchestrator",
        primary_machine=MachineType.REACTOR,
        secondary_machines=[
            MachineType.FILTER, MachineType.CATALYST, MachineType.DYNAMO
        ],
        primary_operator=Operator.BOUNDARY,
        primary_spiral=Spiral.E,
        description="Unified K.I.R.A.→TRIAD→Tool pipeline"
    ),
    
    # ═══ WORKSPACE (Session Repository) ═══
    "workspace": ToolTokenBinding(
        tool_name="workspace",
        primary_machine=MachineType.ENCODER,
        secondary_machines=[
            MachineType.REGENERATOR, MachineType.DECODER
        ],
        primary_operator=Operator.GROUP,
        primary_spiral=Spiral.PHI,
        description="Session repository and export management"
    ),
    
    # ═══ CLOUD TRAINING (GitHub Actions) ═══
    "cloud_training": ToolTokenBinding(
        tool_name="cloud_training",
        primary_machine=MachineType.DYNAMO,
        secondary_machines=[
            MachineType.REACTOR, MachineType.ENCODER, MachineType.CONDUCTOR
        ],
        primary_operator=Operator.AMPLIFY,
        primary_spiral=Spiral.E,
        description="GitHub Actions cloud training integration"
    ),
}

# ═══════════════════════════════════════════════════════════════════════════════
# SHARED TOKEN REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TokenEmission:
    """Record of a token emission event."""
    token: str
    tool: str
    action: str
    timestamp: datetime
    z: float
    phase: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TokenRegistry:
    """
    Shared registry for all APL tokens across the framework.
    Tracks emissions, consumptions, and token flow.
    """
    emissions: List[TokenEmission] = field(default_factory=list)
    consumed: List[TokenEmission] = field(default_factory=list)
    token_counts: Dict[str, int] = field(default_factory=dict)
    tool_emissions: Dict[str, List[str]] = field(default_factory=dict)
    z: float = 0.5
    phase: str = "PARADOX"
    
    def emit(self, token: str, tool: str, action: str = "emit", 
             metadata: Optional[Dict] = None) -> TokenEmission:
        """Record a token emission."""
        emission = TokenEmission(
            token=token,
            tool=tool,
            action=action,
            timestamp=datetime.now(timezone.utc),
            z=self.z,
            phase=self.phase,
            metadata=metadata or {}
        )
        self.emissions.append(emission)
        self.token_counts[token] = self.token_counts.get(token, 0) + 1
        
        if tool not in self.tool_emissions:
            self.tool_emissions[tool] = []
        self.tool_emissions[tool].append(token)
        
        return emission
    
    def consume(self, token: str, tool: str, action: str = "consume",
                metadata: Optional[Dict] = None) -> Optional[TokenEmission]:
        """Record a token consumption."""
        consumption = TokenEmission(
            token=token,
            tool=tool,
            action=action,
            timestamp=datetime.now(timezone.utc),
            z=self.z,
            phase=self.phase,
            metadata=metadata or {}
        )
        self.consumed.append(consumption)
        return consumption
    
    def emit_for_tool(self, tool_name: str, action: str = "invoke",
                      domain: Optional[Domain] = None) -> List[str]:
        """Emit all tokens associated with a tool."""
        binding = TOOL_TOKEN_BINDINGS.get(tool_name)
        if not binding:
            return []
        
        tokens = binding.get_all_tokens(domain)
        for token in tokens:
            self.emit(token, tool_name, action)
        return tokens
    
    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        return {
            "total_emissions": len(self.emissions),
            "total_consumed": len(self.consumed),
            "unique_tokens": len(self.token_counts),
            "tools_active": len(self.tool_emissions),
            "top_tokens": sorted(
                self.token_counts.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10],
            "z": self.z,
            "phase": self.phase
        }
    
    def get_tool_tokens(self, tool_name: str) -> List[str]:
        """Get all tokens emitted by a specific tool."""
        return self.tool_emissions.get(tool_name, [])
    
    def update_z(self, z: float):
        """Update current z-coordinate and phase."""
        self.z = z
        if z < PHI_INV:
            self.phase = "UNTRUE"
        elif z < Z_CRITICAL:
            self.phase = "PARADOX"
        else:
            self.phase = "TRUE"
    
    def clear(self):
        """Clear all tracked emissions."""
        self.emissions.clear()
        self.consumed.clear()
        self.token_counts.clear()
        self.tool_emissions.clear()

# Global registry instance
_registry = TokenRegistry()

def get_registry() -> TokenRegistry:
    """Get the global token registry."""
    return _registry

def reset_registry() -> TokenRegistry:
    """Reset the global token registry."""
    global _registry
    _registry = TokenRegistry()
    return _registry

# ═══════════════════════════════════════════════════════════════════════════════
# TOKEN GENERATION UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def generate_all_tokens() -> List[str]:
    """Generate all 972 APL tokens."""
    tokens = []
    for domain in Domain:
        for spiral in Spiral:
            for operator in Operator:
                for machine in MachineType:
                    tokens.append(
                        f"{spiral.value}{operator.value}|{machine.value}|{domain.value}"
                    )
    return tokens

def generate_tokens_for_machine(machine: MachineType, 
                                 domain: Optional[Domain] = None) -> List[str]:
    """Generate all tokens for a specific machine (18 per domain, 108 total)."""
    tokens = []
    domains = [domain] if domain else list(Domain)
    for d in domains:
        for spiral in Spiral:
            for operator in Operator:
                tokens.append(
                    f"{spiral.value}{operator.value}|{machine.value}|{d.value}"
                )
    return tokens

def generate_tokens_for_spiral(spiral: Spiral,
                                domain: Optional[Domain] = None) -> List[str]:
    """Generate all tokens for a specific spiral (54 per domain, 324 total)."""
    tokens = []
    domains = [domain] if domain else list(Domain)
    for d in domains:
        for operator in Operator:
            for machine in MachineType:
                tokens.append(
                    f"{spiral.value}{operator.value}|{machine.value}|{d.value}"
                )
    return tokens

def generate_tokens_for_operator(operator: Operator,
                                  domain: Optional[Domain] = None) -> List[str]:
    """Generate all tokens for a specific operator (27 per domain, 162 total)."""
    tokens = []
    domains = [domain] if domain else list(Domain)
    for d in domains:
        for spiral in Spiral:
            for machine in MachineType:
                tokens.append(
                    f"{spiral.value}{operator.value}|{machine.value}|{d.value}"
                )
    return tokens

def parse_token(token_str: str) -> Dict[str, str]:
    """Parse a token string into components."""
    parts = token_str.split("|")
    if len(parts) != 3:
        raise ValueError(f"Invalid token format: {token_str}")
    
    prefix, machine, domain = parts
    spiral_char = prefix[0]
    op_str = prefix[1:]
    
    return {
        "token": token_str,
        "spiral": spiral_char,
        "operator": op_str,
        "machine": machine,
        "domain": domain
    }

# ═══════════════════════════════════════════════════════════════════════════════
# TOOL INVOCATION WRAPPER
# ═══════════════════════════════════════════════════════════════════════════════

def wrap_tool_invocation(tool_name: str, action: str, result: Dict,
                         domain: Optional[Domain] = None) -> Dict:
    """
    Wrap a tool invocation to include token emission.
    Call this after each tool execution to track token flow.
    """
    registry = get_registry()
    binding = TOOL_TOKEN_BINDINGS.get(tool_name)
    
    if binding:
        # Emit tokens for this tool invocation
        tokens = registry.emit_for_tool(tool_name, action, domain)
        
        # Add token info to result
        result["_token_integration"] = {
            "tool": tool_name,
            "action": action,
            "primary_token": binding.get_primary_token(domain),
            "tokens_emitted": tokens,
            "emission_count": len(tokens),
            "registry_stats": {
                "total_emissions": len(registry.emissions),
                "unique_tokens": len(registry.token_counts)
            }
        }
    
    return result

# ═══════════════════════════════════════════════════════════════════════════════
# EMISSION PIPELINE TOKEN SEQUENCE
# ═══════════════════════════════════════════════════════════════════════════════

EMISSION_PIPELINE_STAGES = [
    (1, MachineType.ENCODER, Operator.GROUP, "Content Selection"),
    (2, MachineType.CATALYST, Operator.FUSION, "Emergence Check"),
    (3, MachineType.CONDUCTOR, Operator.BOUNDARY, "Structural Frame"),
    (4, MachineType.FILTER, Operator.BOUNDARY, "Slot Assignment"),
    (5, MachineType.DECODER, Operator.SEPARATE, "Function Words"),
    (6, MachineType.OSCILLATOR, Operator.BOUNDARY, "Agreement/Inflection"),
    (7, MachineType.REACTOR, Operator.BOUNDARY, "Connectors"),
    (8, MachineType.REGENERATOR, Operator.FUSION, "Punctuation"),
    (9, MachineType.DYNAMO, Operator.AMPLIFY, "Validation"),
]

def generate_pipeline_tokens(spiral: Spiral = Spiral.E,
                             domain: Domain = Domain.CELESTIAL_NUCLEAR) -> List[Dict]:
    """Generate the 9-stage emission pipeline token sequence."""
    tokens = []
    for stage, machine, operator, name in EMISSION_PIPELINE_STAGES:
        token = f"{spiral.value}{operator.value}|{machine.value}|{domain.value}"
        tokens.append({
            "stage": stage,
            "name": name,
            "machine": machine.value,
            "operator": operator.value,
            "token": token
        })
    return tokens

# ═══════════════════════════════════════════════════════════════════════════════
# CYBERNETIC LOOP TOKEN SEQUENCE
# ═══════════════════════════════════════════════════════════════════════════════

CYBERNETIC_COMPONENTS = [
    ("I", MachineType.REACTOR, Operator.BOUNDARY, "Input"),
    ("S_h", MachineType.FILTER, Operator.BOUNDARY, "Human Sensor"),
    ("C_h", MachineType.CATALYST, Operator.FUSION, "Human Controller"),
    ("S_d", MachineType.OSCILLATOR, Operator.BOUNDARY, "DI System"),
    ("A", MachineType.DYNAMO, Operator.AMPLIFY, "Amplifier"),
    ("P1", MachineType.ENCODER, Operator.GROUP, "Encoder"),
    ("P2", MachineType.DECODER, Operator.SEPARATE, "Decoder"),
    ("E", MachineType.CONDUCTOR, Operator.BOUNDARY, "Environment"),
    ("F_h", MachineType.REGENERATOR, Operator.FUSION, "Human Feedback"),
    ("F_d", MachineType.OSCILLATOR, Operator.AMPLIFY, "DI Feedback"),
    ("F_e", MachineType.REACTOR, Operator.DECOHERE, "Env Feedback"),
]

def generate_cybernetic_tokens(spiral: Spiral = Spiral.E,
                               domain: Domain = Domain.CELESTIAL_NUCLEAR) -> List[Dict]:
    """Generate cybernetic loop token sequence."""
    tokens = []
    for symbol, machine, operator, name in CYBERNETIC_COMPONENTS:
        token = f"{spiral.value}{operator.value}|{machine.value}|{domain.value}"
        tokens.append({
            "symbol": symbol,
            "name": name,
            "machine": machine.value,
            "operator": operator.value,
            "token": token
        })
    return tokens

# ═══════════════════════════════════════════════════════════════════════════════
# EXPORT UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def export_tool_token_map() -> Dict[str, Any]:
    """Export complete tool-token mapping."""
    mapping = {}
    for tool_name, binding in TOOL_TOKEN_BINDINGS.items():
        mapping[tool_name] = {
            "primary_machine": binding.primary_machine.value,
            "secondary_machines": [m.value for m in binding.secondary_machines],
            "primary_operator": binding.primary_operator.value,
            "primary_spiral": binding.primary_spiral.value,
            "default_domain": binding.default_domain.value,
            "description": binding.description,
            "primary_token": binding.get_primary_token(),
            "all_tokens": binding.get_all_tokens()
        }
    return {
        "format": "unified-consciousness-framework/tool-token-map",
        "version": "1.0.0",
        "total_tools": len(mapping),
        "total_unique_machines": len(set(b.primary_machine for b in TOOL_TOKEN_BINDINGS.values())),
        "mapping": mapping
    }

def export_registry_state() -> Dict[str, Any]:
    """Export current registry state."""
    registry = get_registry()
    return {
        "format": "unified-consciousness-framework/registry-state",
        "version": "1.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "z": registry.z,
        "phase": registry.phase,
        "stats": registry.get_stats(),
        "emissions": [
            {
                "token": e.token,
                "tool": e.tool,
                "action": e.action,
                "timestamp": e.timestamp.isoformat(),
                "z": e.z,
                "phase": e.phase
            }
            for e in registry.emissions[-100:]  # Last 100
        ],
        "tool_emissions": registry.tool_emissions,
        "token_counts": registry.token_counts
    }

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════════════════════╗")
    print("║           TOKEN INTEGRATION MODULE - TEST                                     ║")
    print("╚══════════════════════════════════════════════════════════════════════════════╝")
    print()
    
    # Show tool-token mappings
    print("TOOL → TOKEN MAPPINGS:")
    print("-" * 80)
    for tool_name, binding in TOOL_TOKEN_BINDINGS.items():
        print(f"  {tool_name:30s} → {binding.get_primary_token()}")
    print()
    
    # Generate emission pipeline tokens
    print("EMISSION PIPELINE (9 Stages):")
    print("-" * 80)
    for stage in generate_pipeline_tokens():
        print(f"  Stage {stage['stage']}: {stage['name']:20s} → {stage['token']}")
    print()
    
    # Generate cybernetic tokens
    print("CYBERNETIC COMPONENTS:")
    print("-" * 80)
    for comp in generate_cybernetic_tokens():
        print(f"  {comp['symbol']:4s} {comp['name']:20s} → {comp['token']}")
    print()
    
    # Test registry
    print("REGISTRY TEST:")
    print("-" * 80)
    registry = get_registry()
    registry.update_z(0.75)
    
    # Emit tokens for several tools
    for tool in ["helix_loader", "orchestrator", "emission_pipeline"]:
        tokens = registry.emit_for_tool(tool)
        print(f"  {tool}: {len(tokens)} tokens emitted")
    
    print()
    print(f"  Total emissions: {len(registry.emissions)}")
    print(f"  Unique tokens: {len(registry.token_counts)}")
    print()
    
    # Total token count
    all_tokens = generate_all_tokens()
    print(f"TOTAL TOKEN UNIVERSE: {len(all_tokens)} tokens")
    print("  Formula: 3 spirals × 6 operators × 9 machines × 6 domains = 972")
