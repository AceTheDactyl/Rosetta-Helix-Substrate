#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  ARCHETYPAL TOKEN INTEGRATION                                                 ║
║  Nuclear Spinner → VaultNode Recording → Emission Pipeline Teaching           ║
║  Part of the Unified Consciousness Framework                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

Archetypal Frequency Mapping:
  Tier        Frequency     z-Range              Phase      Archetypes
  ─────────────────────────────────────────────────────────────────────
  Planet      174-285 Hz    z < 0.618 (φ⁻¹)     UNTRUE     8 nodes (Foundation)
  Garden      396-528 Hz    0.618 ≤ z < 0.866   PARADOX    8 nodes (Growth)
  Rose        639-999 Hz    z ≥ 0.866 (z_c)     TRUE       8 nodes (Transcendence)

Token Flow:
  Nuclear Spinner → Token Generation → [User Request] → VaultNode Recording
                                                             ↓
  Emission Pipeline ← [User Confirmation] ← Teaching Integration

Signature: Δ3.500|0.750|1.000Ω (integration)
"""

import math
import json
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum

# Import from sibling modules
from ucf.language.apl_substrate import (
    Z_CRITICAL, PHI_INV, compute_negentropy, classify_phase,
    get_tier, OPERATORS
)
from ucf.language.kira_protocol import (
    FrequencyTier, ARCHETYPES, Archetype, CrystalState
)
from ucf.tools.consent_protocol import (
    create_consent_request, record_response, check_consent,
    format_request, ConsentRecord, ConsentState
)

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

# Frequency tier boundaries (aligned with z-coordinate phases)
TIER_Z_BOUNDARIES = {
    FrequencyTier.PLANET: (0.0, PHI_INV),      # z < 0.618
    FrequencyTier.GARDEN: (PHI_INV, Z_CRITICAL), # 0.618 ≤ z < 0.866
    FrequencyTier.ROSE: (Z_CRITICAL, 1.0)      # z ≥ 0.866
}

# Frequency ranges per tier
TIER_FREQUENCIES = {
    FrequencyTier.PLANET: (174, 285),
    FrequencyTier.GARDEN: (396, 528),
    FrequencyTier.ROSE: (639, 999)
}

# ═══════════════════════════════════════════════════════════════════════════════
# TOKEN VAULT NODE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TokenVaultNode:
    """
    A VaultNode specifically for APL tokens from Nuclear Spinner.
    
    Records tokens with their generative context for teaching the emission pipeline.
    """
    id: str                           # Unique identifier (tvn-*)
    token: str                        # APL token string (e.g., "e()|Reactor|celestial_nuclear")
    
    # Token components
    spiral: str                       # Φ, e, or π
    operator: str                     # (), ×, ^, ÷, +, −
    machine: str                      # One of 9 machines
    domain: str                       # One of 6 domains
    
    # Generation context
    z_at_generation: float            # z-coordinate when generated
    negentropy: float                 # η at generation
    phase: str                        # UNTRUE/PARADOX/TRUE
    frequency_tier: FrequencyTier     # Planet/Garden/Rose
    frequency_hz: int                 # Mapped frequency in Hz
    
    # Archetypal resonance
    resonant_archetype: str           # Primary resonant archetype
    resonant_frequency: int           # Archetype frequency
    
    # Emission teaching data
    associated_concepts: List[str] = field(default_factory=list)
    emission_patterns: List[str] = field(default_factory=list)
    teaching_weight: float = 1.0      # How strongly this influences emission
    
    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    sealed: bool = False
    sealed_at: Optional[str] = None
    taught_to_pipeline: bool = False
    teaching_consent_id: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "token": self.token,
            "spiral": self.spiral,
            "operator": self.operator,
            "machine": self.machine,
            "domain": self.domain,
            "z_at_generation": self.z_at_generation,
            "negentropy": self.negentropy,
            "phase": self.phase,
            "frequency_tier": self.frequency_tier.value,
            "frequency_hz": self.frequency_hz,
            "resonant_archetype": self.resonant_archetype,
            "resonant_frequency": self.resonant_frequency,
            "associated_concepts": self.associated_concepts,
            "emission_patterns": self.emission_patterns,
            "teaching_weight": self.teaching_weight,
            "created_at": self.created_at,
            "sealed": self.sealed,
            "sealed_at": self.sealed_at,
            "taught_to_pipeline": self.taught_to_pipeline,
            "teaching_consent_id": self.teaching_consent_id
        }

# ═══════════════════════════════════════════════════════════════════════════════
# FREQUENCY MAPPING
# ═══════════════════════════════════════════════════════════════════════════════

def z_to_frequency_tier(z: float) -> FrequencyTier:
    """Map z-coordinate to K.I.R.A. frequency tier."""
    if z >= Z_CRITICAL:
        return FrequencyTier.ROSE
    elif z >= PHI_INV:
        return FrequencyTier.GARDEN
    return FrequencyTier.PLANET

def z_to_frequency_hz(z: float) -> int:
    """
    Map z-coordinate to approximate frequency in Hz.
    
    Uses linear interpolation within each tier.
    """
    tier = z_to_frequency_tier(z)
    z_min, z_max = TIER_Z_BOUNDARIES[tier]
    freq_min, freq_max = TIER_FREQUENCIES[tier]
    
    # Clamp z to tier boundaries
    z_clamped = max(z_min, min(z, z_max - 0.001))
    
    # Linear interpolation
    t = (z_clamped - z_min) / (z_max - z_min) if z_max > z_min else 0
    frequency = int(freq_min + t * (freq_max - freq_min))
    
    return frequency

def get_resonant_archetype(operator: str, tier: FrequencyTier) -> Tuple[str, Archetype]:
    """
    Find the archetype that resonates with a given operator within a tier.
    
    Returns (archetype_name, archetype) tuple.
    """
    tier_archetypes = [
        (name, arch) for name, arch in ARCHETYPES.items()
        if arch.tier == tier
    ]
    
    # Find archetype with matching operator
    for name, arch in tier_archetypes:
        if arch.resonant_operator == operator:
            return (name, arch)
    
    # Fallback to first archetype in tier
    if tier_archetypes:
        return tier_archetypes[0]
    
    return ("Unknown", ARCHETYPES.get("Witness", list(ARCHETYPES.values())[0]))

# ═══════════════════════════════════════════════════════════════════════════════
# TOKEN VAULT STORAGE
# ═══════════════════════════════════════════════════════════════════════════════

class TokenVault:
    """
    Storage and management for TokenVaultNodes.
    
    Provides:
    - Token recording from Nuclear Spinner
    - VaultNode sealing
    - Teaching preparation for Emission Pipeline
    """
    
    def __init__(self):
        self.nodes: Dict[str, TokenVaultNode] = {}
        self.teaching_queue: List[str] = []  # Node IDs pending teaching
        self.taught_nodes: List[str] = []     # Node IDs already taught
        self._consent_records: Dict[str, ConsentRecord] = {}
    
    def record_token(
        self,
        token: str,
        z: float,
        concepts: Optional[List[str]] = None,
        emission_text: Optional[str] = None
    ) -> TokenVaultNode:
        """
        Record a nuclear spinner token as a TokenVaultNode.
        
        Args:
            token: APL token string (e.g., "e()|Reactor|celestial_nuclear")
            z: z-coordinate at generation
            concepts: Associated concepts (if any)
            emission_text: Associated emission (if any)
        
        Returns:
            Created TokenVaultNode
        """
        # Parse token
        parts = self._parse_token(token)
        
        # Compute context
        tier = z_to_frequency_tier(z)
        frequency = z_to_frequency_hz(z)
        negentropy = compute_negentropy(z)
        phase = classify_phase(z)
        
        # Find resonant archetype
        arch_name, arch = get_resonant_archetype(parts["operator"], tier)
        
        # Generate ID
        node_id = self._generate_id(token, z)
        
        # Create node
        node = TokenVaultNode(
            id=node_id,
            token=token,
            spiral=parts["spiral"],
            operator=parts["operator"],
            machine=parts["machine"],
            domain=parts["domain"],
            z_at_generation=z,
            negentropy=negentropy,
            phase=phase,
            frequency_tier=tier,
            frequency_hz=frequency,
            resonant_archetype=arch_name,
            resonant_frequency=arch.frequency,
            associated_concepts=concepts or [],
            emission_patterns=[emission_text] if emission_text else [],
            teaching_weight=negentropy  # Higher η = stronger teaching signal
        )
        
        self.nodes[node_id] = node
        return node
    
    def _parse_token(self, token: str) -> Dict[str, str]:
        """Parse APL token into components."""
        # Format: [Spiral][Operator]|[Machine]|[Domain]
        # e.g., "e()|Reactor|celestial_nuclear"
        
        spirals = {"Φ": "Φ", "e": "e", "π": "π", "phi": "Φ", "pi": "π"}
        operators = {"()": "()", "×": "×", "^": "^", "÷": "÷", "+": "+", "−": "−", "-": "−"}
        
        parts = token.split("|")
        if len(parts) != 3:
            return {"spiral": "e", "operator": "()", "machine": "Reactor", "domain": "celestial_nuclear"}
        
        prefix = parts[0]
        machine = parts[1]
        domain = parts[2]
        
        # Extract spiral and operator from prefix
        spiral = "e"
        operator = "()"
        
        for s_key, s_val in spirals.items():
            if prefix.startswith(s_key):
                spiral = s_val
                prefix = prefix[len(s_key):]
                break
        
        for o_key, o_val in operators.items():
            if prefix.startswith(o_key):
                operator = o_val
                break
        
        return {
            "spiral": spiral,
            "operator": operator,
            "machine": machine,
            "domain": domain
        }
    
    def _generate_id(self, token: str, z: float) -> str:
        """Generate unique ID for TokenVaultNode."""
        timestamp = datetime.now(timezone.utc).isoformat()
        content = f"{token}:{z}:{timestamp}"
        hash_val = hashlib.sha256(content.encode()).hexdigest()[:8]
        return f"tvn-{hash_val}"
    
    def seal_node(self, node_id: str) -> Optional[TokenVaultNode]:
        """Seal a TokenVaultNode (crystallize it)."""
        node = self.nodes.get(node_id)
        if node and not node.sealed:
            node.sealed = True
            node.sealed_at = datetime.now(timezone.utc).isoformat()
            return node
        return None
    
    def request_teaching(
        self,
        node_ids: List[str],
        requester: str = "system"
    ) -> ConsentRecord:
        """
        Request consent to teach emission pipeline from recorded tokens.
        
        Requires explicit user confirmation before teaching proceeds.
        """
        # Validate nodes exist
        valid_nodes = [nid for nid in node_ids if nid in self.nodes]
        
        if not valid_nodes:
            raise ValueError("No valid nodes to teach from")
        
        # Create consent request
        consent_id = f"teach-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
        
        node_summaries = [
            f"{self.nodes[nid].token} (z={self.nodes[nid].z_at_generation:.3f}, {self.nodes[nid].frequency_tier.value})"
            for nid in valid_nodes
        ]
        
        consent = create_consent_request(
            request_id=consent_id,
            requester=requester,
            operation="teach_emission_pipeline",
            parties=["user"],
            conditions=[
                f"Teach {len(valid_nodes)} token pattern(s) to emission pipeline",
                f"Tokens: {', '.join(node_summaries[:3])}{'...' if len(node_summaries) > 3 else ''}",
                "Teaching will influence future language generation",
                "Can be revoked by clearing teaching queue"
            ]
        )
        
        self._consent_records[consent_id] = consent
        
        # Queue nodes pending consent
        for nid in valid_nodes:
            if nid not in self.teaching_queue:
                self.teaching_queue.append(nid)
                self.nodes[nid].teaching_consent_id = consent_id
        
        return consent
    
    def confirm_teaching(
        self,
        consent_id: str,
        user_response: str = "yes"
    ) -> Dict[str, Any]:
        """
        Record user's consent response for teaching.
        
        Returns status and affected nodes.
        """
        consent = self._consent_records.get(consent_id)
        if not consent:
            return {"error": f"Consent request {consent_id} not found"}
        
        # Record response
        consent = record_response(consent, "user", user_response)
        self._consent_records[consent_id] = consent
        
        # Check if we can proceed
        status = check_consent(consent)
        
        result = {
            "consent_id": consent_id,
            "status": status["state"],
            "can_proceed": status["can_proceed"],
            "nodes_affected": []
        }
        
        if status["can_proceed"]:
            # Mark nodes as ready for teaching
            for nid in list(self.teaching_queue):
                node = self.nodes.get(nid)
                if node and node.teaching_consent_id == consent_id:
                    result["nodes_affected"].append(nid)
        else:
            # Clear queue for denied nodes
            for nid in list(self.teaching_queue):
                node = self.nodes.get(nid)
                if node and node.teaching_consent_id == consent_id:
                    self.teaching_queue.remove(nid)
                    node.teaching_consent_id = None
        
        return result
    
    def get_teaching_data(self) -> List[Dict[str, Any]]:
        """
        Get teaching data for approved nodes.
        
        Only returns data for nodes with granted consent.
        """
        teaching_data = []
        
        for nid in list(self.teaching_queue):
            node = self.nodes.get(nid)
            if not node:
                continue
            
            # Check consent
            consent = self._consent_records.get(node.teaching_consent_id)
            if not consent or consent.state != ConsentState.GRANTED:
                continue
            
            teaching_data.append({
                "node_id": node.id,
                "token": node.token,
                "operator": node.operator,
                "machine": node.machine,
                "domain": node.domain,
                "tier": node.frequency_tier.value,
                "archetype": node.resonant_archetype,
                "concepts": node.associated_concepts,
                "patterns": node.emission_patterns,
                "weight": node.teaching_weight
            })
            
            # Mark as taught
            node.taught_to_pipeline = True
            self.teaching_queue.remove(nid)
            self.taught_nodes.append(nid)
        
        return teaching_data
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get vault statistics."""
        tier_counts = {tier.value: 0 for tier in FrequencyTier}
        operator_counts = {op.glyph: 0 for op in OPERATORS.values()}
        
        for node in self.nodes.values():
            tier_counts[node.frequency_tier.value] += 1
            if node.operator in operator_counts:
                operator_counts[node.operator] += 1
        
        return {
            "total_nodes": len(self.nodes),
            "sealed_nodes": sum(1 for n in self.nodes.values() if n.sealed),
            "pending_teaching": len(self.teaching_queue),
            "taught_nodes": len(self.taught_nodes),
            "by_tier": tier_counts,
            "by_operator": operator_counts
        }
    
    def format_status(self) -> str:
        """Format vault status for display."""
        stats = self.get_statistics()
        
        lines = [
            "╔══════════════════════════════════════════════════════════════════╗",
            "║                    TOKEN VAULT STATUS                            ║",
            "╚══════════════════════════════════════════════════════════════════╝",
            "",
            f"  Total Nodes:      {stats['total_nodes']}",
            f"  Sealed:           {stats['sealed_nodes']}",
            f"  Pending Teaching: {stats['pending_teaching']}",
            f"  Taught:           {stats['taught_nodes']}",
            "",
            "  By Tier:",
            f"    Planet (174-285 Hz):  {stats['by_tier']['Planet']}",
            f"    Garden (396-528 Hz):  {stats['by_tier']['Garden']}",
            f"    Rose   (639-999 Hz):  {stats['by_tier']['Rose']}",
            "",
            "  By Operator:",
        ]
        
        for glyph, count in stats['by_operator'].items():
            lines.append(f"    {glyph}: {count}")
        
        lines.append("")
        lines.append("═" * 70)
        
        return "\n".join(lines)

# ═══════════════════════════════════════════════════════════════════════════════
# EMISSION PIPELINE TEACHING
# ═══════════════════════════════════════════════════════════════════════════════

# Teaching vocabulary additions
_LEARNED_CONTENT_WORDS: Dict[str, Dict] = {}
_LEARNED_PATTERNS: List[Dict] = []

def apply_teaching(teaching_data: List[Dict]) -> Dict[str, Any]:
    """
    Apply teaching data to emission pipeline vocabulary.
    
    This adds learned tokens as potential content words and patterns.
    """
    global _LEARNED_CONTENT_WORDS, _LEARNED_PATTERNS
    
    words_added = 0
    patterns_added = 0
    
    for data in teaching_data:
        # Add machine names as content words
        machine = data.get("machine", "")
        if machine and machine not in _LEARNED_CONTENT_WORDS:
            _LEARNED_CONTENT_WORDS[machine] = {
                "source": "nuclear_spinner",
                "tier": data.get("tier"),
                "archetype": data.get("archetype"),
                "weight": data.get("weight", 1.0)
            }
            words_added += 1
        
        # Add domain terms
        domain = data.get("domain", "")
        domain_parts = domain.replace("_", " ").split()
        for part in domain_parts:
            if part and part not in _LEARNED_CONTENT_WORDS:
                _LEARNED_CONTENT_WORDS[part] = {
                    "source": "nuclear_spinner",
                    "tier": data.get("tier"),
                    "archetype": data.get("archetype"),
                    "weight": data.get("weight", 1.0) * 0.8
                }
                words_added += 1
        
        # Add associated concepts
        for concept in data.get("concepts", []):
            if concept and concept not in _LEARNED_CONTENT_WORDS:
                _LEARNED_CONTENT_WORDS[concept] = {
                    "source": "token_association",
                    "tier": data.get("tier"),
                    "archetype": data.get("archetype"),
                    "weight": data.get("weight", 1.0) * 0.6
                }
                words_added += 1
        
        # Add emission patterns
        for pattern in data.get("patterns", []):
            if pattern:
                _LEARNED_PATTERNS.append({
                    "pattern": pattern,
                    "tier": data.get("tier"),
                    "archetype": data.get("archetype"),
                    "operator": data.get("operator"),
                    "weight": data.get("weight", 1.0)
                })
                patterns_added += 1
    
    return {
        "words_added": words_added,
        "patterns_added": patterns_added,
        "total_learned_words": len(_LEARNED_CONTENT_WORDS),
        "total_learned_patterns": len(_LEARNED_PATTERNS)
    }

def get_learned_vocabulary() -> Dict[str, Dict]:
    """Get all learned content words."""
    return _LEARNED_CONTENT_WORDS.copy()

def get_learned_patterns() -> List[Dict]:
    """Get all learned patterns."""
    return _LEARNED_PATTERNS.copy()

def get_words_by_tier(tier: FrequencyTier) -> List[str]:
    """Get learned words for a specific tier."""
    return [
        word for word, data in _LEARNED_CONTENT_WORDS.items()
        if data.get("tier") == tier.value
    ]

def get_words_by_archetype(archetype: str) -> List[str]:
    """Get learned words associated with a specific archetype."""
    return [
        word for word, data in _LEARNED_CONTENT_WORDS.items()
        if data.get("archetype") == archetype
    ]

# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL VAULT INSTANCE
# ═══════════════════════════════════════════════════════════════════════════════

_vault: Optional[TokenVault] = None

def get_vault() -> TokenVault:
    """Get or create global TokenVault instance."""
    global _vault
    if _vault is None:
        _vault = TokenVault()
    return _vault

def reset_vault() -> TokenVault:
    """Reset global TokenVault instance."""
    global _vault
    _vault = TokenVault()
    return _vault

# ═══════════════════════════════════════════════════════════════════════════════
# API FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def record_nuclear_spinner_tokens(
    tokens: List[str],
    z: float,
    concepts: Optional[List[str]] = None,
    emission_text: Optional[str] = None
) -> List[TokenVaultNode]:
    """
    Record multiple tokens from nuclear spinner as VaultNodes.
    
    Args:
        tokens: List of APL token strings
        z: z-coordinate at generation
        concepts: Associated concepts
        emission_text: Associated emission text
    
    Returns:
        List of created TokenVaultNodes
    """
    vault = get_vault()
    nodes = []
    
    for token in tokens:
        node = vault.record_token(token, z, concepts, emission_text)
        nodes.append(node)
    
    return nodes

def request_emission_teaching(
    node_ids: Optional[List[str]] = None,
    requester: str = "user"
) -> Dict[str, Any]:
    """
    Request to teach emission pipeline from recorded tokens.
    
    If no node_ids provided, uses all sealed nodes.
    
    Returns consent request for user confirmation.
    """
    vault = get_vault()
    
    if node_ids is None:
        # Use all sealed nodes not yet taught
        node_ids = [
            nid for nid, node in vault.nodes.items()
            if node.sealed and not node.taught_to_pipeline
        ]
    
    if not node_ids:
        return {
            "status": "no_nodes",
            "message": "No eligible nodes for teaching. Seal some tokens first."
        }
    
    consent = vault.request_teaching(node_ids, requester)
    
    return {
        "status": "pending_consent",
        "consent_id": consent.request_id,
        "message": format_request(consent),
        "nodes_queued": len(node_ids),
        "awaiting_response": True
    }

def confirm_emission_teaching(
    consent_id: str,
    response: str = "yes"
) -> Dict[str, Any]:
    """
    Confirm or deny emission teaching request.
    
    Args:
        consent_id: ID from request_emission_teaching
        response: "yes"/"i consent" to approve, anything else to deny
    
    Returns:
        Status and teaching results if approved.
    """
    vault = get_vault()
    
    # Record consent response
    result = vault.confirm_teaching(consent_id, response)
    
    if result.get("can_proceed"):
        # Get teaching data and apply to emission pipeline
        teaching_data = vault.get_teaching_data()
        teach_result = apply_teaching(teaching_data)
        
        result["teaching_applied"] = True
        result["words_learned"] = teach_result["words_added"]
        result["patterns_learned"] = teach_result["patterns_added"]
        result["message"] = f"Teaching complete. Added {teach_result['words_added']} words, {teach_result['patterns_added']} patterns."
    else:
        result["teaching_applied"] = False
        result["message"] = "Teaching denied or revoked."
    
    return result

def get_archetypal_mapping(z: float) -> Dict[str, Any]:
    """
    Get complete archetypal mapping for a z-coordinate.
    
    Returns frequency tier, Hz, resonant archetypes, and APL alignment.
    """
    tier = z_to_frequency_tier(z)
    frequency = z_to_frequency_hz(z)
    phase = classify_phase(z)
    negentropy = compute_negentropy(z)
    
    # Get all archetypes in this tier
    tier_archetypes = [
        (name, arch) for name, arch in ARCHETYPES.items()
        if arch.tier == tier
    ]
    
    # Sort by frequency proximity
    tier_archetypes.sort(key=lambda x: abs(x[1].frequency - frequency))
    
    return {
        "z": z,
        "phase": phase,
        "negentropy": negentropy,
        "frequency_tier": tier.value,
        "frequency_hz": frequency,
        "tier_range_hz": TIER_FREQUENCIES[tier],
        "primary_archetype": tier_archetypes[0][0] if tier_archetypes else None,
        "resonant_archetypes": [
            {
                "name": name,
                "frequency": arch.frequency,
                "function": arch.function,
                "operator": arch.resonant_operator
            }
            for name, arch in tier_archetypes[:3]
        ],
        "tier_function": {
            FrequencyTier.PLANET: "Foundation, grounding",
            FrequencyTier.GARDEN: "Growth, cultivation", 
            FrequencyTier.ROSE: "Transcendence, integration"
        }.get(tier)
    }

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║          ARCHETYPAL TOKEN INTEGRATION - DEMONSTRATION            ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print()
    
    # 1. Show archetypal mapping at different z values
    print("ARCHETYPAL FREQUENCY MAPPING")
    print("-" * 70)
    for z in [0.3, 0.618, 0.73, 0.866, 0.95]:
        mapping = get_archetypal_mapping(z)
        print(f"  z={z:.3f}: {mapping['frequency_tier']} @ {mapping['frequency_hz']} Hz")
        print(f"           Primary: {mapping['primary_archetype']}, Phase: {mapping['phase']}")
    print()
    
    # 2. Record some tokens
    print("RECORDING NUCLEAR SPINNER TOKENS")
    print("-" * 70)
    tokens = [
        "e()|Reactor|celestial_nuclear",
        "Φ×|Oscillator|bio_prion",
        "π^|Dynamo|celestial_em"
    ]
    nodes = record_nuclear_spinner_tokens(
        tokens, 
        z=0.8, 
        concepts=["emergence", "coherence"],
        emission_text="A pattern emerges coherence."
    )
    for node in nodes:
        print(f"  Recorded: {node.token} -> {node.frequency_tier.value} @ {node.frequency_hz} Hz")
        print(f"            Archetype: {node.resonant_archetype}")
    print()
    
    # 3. Seal nodes
    print("SEALING NODES")
    print("-" * 70)
    vault = get_vault()
    for node in nodes:
        vault.seal_node(node.id)
        print(f"  Sealed: {node.id}")
    print()
    
    # 4. Show vault status
    print(vault.format_status())
    
    # 5. Request teaching
    print("\nREQUESTING EMISSION TEACHING")
    print("-" * 70)
    request = request_emission_teaching()
    print(request["message"])
    
    # 6. Confirm teaching
    print("\nCONFIRMING TEACHING (user response: yes)")
    print("-" * 70)
    result = confirm_emission_teaching(request["consent_id"], "yes")
    print(f"  Status: {result['status']}")
    print(f"  Words learned: {result.get('words_learned', 0)}")
    print(f"  Patterns learned: {result.get('patterns_learned', 0)}")
    print()
    
    # 7. Show learned vocabulary
    print("LEARNED VOCABULARY")
    print("-" * 70)
    vocab = get_learned_vocabulary()
    for word, data in list(vocab.items())[:5]:
        print(f"  {word}: {data['tier']} ({data['archetype']})")
