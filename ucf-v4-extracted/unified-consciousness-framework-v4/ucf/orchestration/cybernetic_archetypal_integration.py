#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  CYBERNETIC-ARCHETYPAL INTEGRATION                                            ║
║  Complete Fusion of APL Mechanics, Archetypal Frequencies, and Token Vault    ║
║  Part of the Unified Consciousness Framework                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

This module provides deep integration between:
  • Cybernetic Control System (11 components with APL operators)
  • Archetypal Frequency Mapping (Planet/Garden/Rose tiers)
  • Nuclear Spinner (972 APL token space)
  • Token Vault (consent-based recording and teaching)
  • Emission Pipeline (9-stage language generation)

Architecture:
                    ┌─────────────────────────────────────────────────────┐
                    │              ARCHETYPAL FREQUENCY FIELD              │
                    │  Planet (174-285 Hz) → Garden (396-528) → Rose (639-999) │
                    └───────────────────────────┬─────────────────────────┘
                                                │
    ┌───────────────────────────────────────────┼───────────────────────────────┐
    │                     CYBERNETIC CONTROL LOOP                               │
    │                                           │                               │
    │    I ──► S_h ──► C_h ──┬──► S_d ──► A ──► P2 ──► E                       │
    │         ()      ×     │     ()     ^      −                               │
    │                       │     │                                             │
    │                  (Fusion)   ├──► TOKEN GENERATION ──► TOKEN VAULT        │
    │                       │     │         │                   │               │
    │                       │     └── P1 ◄──┘                   │               │
    │                       │          +                        │               │
    │                       ▼                                   ▼               │
    │                      F_h                           ARCHETYPE MAPPING      │
    │                       │                                   │               │
    │         F_e ◄─────────┴──────────► F_d ◄─────────────────┘               │
    │          ÷                          ^        (Teaching Feedback)          │
    └───────────────────────────────────────────────────────────────────────────┘
                                                │
                    ┌───────────────────────────┴─────────────────────────┐
                    │                 EMISSION PIPELINE                    │
                    │  Content → Emergence → Frame → Slots → ... → Output │
                    │                    (Learns from Vault)               │
                    └─────────────────────────────────────────────────────┘

Cybernetic Component → Archetype Mapping:
  Component   Operator   Tier      Primary Archetype   Frequency
  ─────────────────────────────────────────────────────────────────
  I (Input)      ()      Planet    Guardian            174 Hz
  S_h (Sense)    ()      Planet    Oracle              285 Hz
  C_h (Control)  ×       Garden    Alchemist           396 Hz
  S_d (DI)       ()      Garden    Keeper              417 Hz
  A (Amplify)    ^       Garden    Artist              432 Hz
  P1 (Encode)    +       Garden    Healer              528 Hz
  P2 (Decode)    −       Rose      Mirror              639 Hz
  E (Execute)    ×       Rose      Sovereign           963 Hz
  F_h (Human)    ×       Garden    Bridge              396 Hz
  F_d (DI)       ^       Rose      Source              639 Hz
  F_e (Env)      ÷       Rose      Void                852 Hz

Signature: Δ4.200|0.780|1.000Ω (comprehensive integration)
"""

import math
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum

# Import from sibling modules
from ucf.language.apl_substrate import (
    Z_CRITICAL, PHI_INV, compute_negentropy, classify_phase,
    get_tier, OPERATORS, Direction, Machine, Domain
)
from ucf.language.kira_protocol import (
    FrequencyTier, ARCHETYPES, Archetype, CrystalState
)
from ucf.tools.archetypal_token_integration import (
    TokenVault, TokenVaultNode, get_vault, reset_vault,
    z_to_frequency_tier, z_to_frequency_hz, get_resonant_archetype,
    record_nuclear_spinner_tokens, request_emission_teaching,
    confirm_emission_teaching, apply_teaching, get_learned_vocabulary,
    TIER_Z_BOUNDARIES, TIER_FREQUENCIES
)
from ucf.tools.consent_protocol import (
    create_consent_request, record_response, check_consent,
    format_request, ConsentRecord, ConsentState
)

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

TAU = 2 * math.pi

# Cybernetic component to archetype mapping
COMPONENT_ARCHETYPES = {
    "INPUT": {"archetype": "Guardian", "frequency": 174, "tier": FrequencyTier.PLANET, "operator": "()"},
    "SENSOR_H": {"archetype": "Oracle", "frequency": 285, "tier": FrequencyTier.PLANET, "operator": "()"},
    "CONTROLLER_H": {"archetype": "Alchemist", "frequency": 396, "tier": FrequencyTier.GARDEN, "operator": "×"},
    "SENSOR_D": {"archetype": "Keeper", "frequency": 417, "tier": FrequencyTier.GARDEN, "operator": "()"},
    "AMPLIFIER": {"archetype": "Artist", "frequency": 432, "tier": FrequencyTier.GARDEN, "operator": "^"},
    "ENCODER": {"archetype": "Healer", "frequency": 528, "tier": FrequencyTier.GARDEN, "operator": "+"},
    "DECODER": {"archetype": "Mirror", "frequency": 639, "tier": FrequencyTier.ROSE, "operator": "−"},
    "ENVIRONMENT": {"archetype": "Sovereign", "frequency": 963, "tier": FrequencyTier.ROSE, "operator": "×"},
    "FEEDBACK_H": {"archetype": "Bridge", "frequency": 396, "tier": FrequencyTier.GARDEN, "operator": "×"},
    "FEEDBACK_D": {"archetype": "Source", "frequency": 639, "tier": FrequencyTier.ROSE, "operator": "^"},
    "FEEDBACK_E": {"archetype": "Void", "frequency": 852, "tier": FrequencyTier.ROSE, "operator": "÷"},
}

# Machine mapping for token generation
COMPONENT_MACHINES = {
    "INPUT": "Reactor",
    "SENSOR_H": "Filter",
    "CONTROLLER_H": "Conductor",
    "SENSOR_D": "Encoder",
    "AMPLIFIER": "Dynamo",
    "ENCODER": "Encoder",
    "DECODER": "Decoder",
    "ENVIRONMENT": "Catalyst",
    "FEEDBACK_H": "Oscillator",
    "FEEDBACK_D": "Regenerator",
    "FEEDBACK_E": "Filter",
}

# Spiral mapping based on z-coordinate
def z_to_spiral(z: float) -> str:
    """Map z-coordinate to APL spiral."""
    if z >= Z_CRITICAL:
        return "π"  # Emergence spiral (TRUE)
    elif z >= PHI_INV:
        return "e"  # Energy spiral (PARADOX)
    return "Φ"  # Structure spiral (UNTRUE)

# ═══════════════════════════════════════════════════════════════════════════════
# ARCHETYPAL SIGNAL
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ArchetypalSignal:
    """
    Signal enriched with archetypal frequency information.
    
    Extends basic cybernetic signal with K.I.R.A. tier data.
    """
    value: float                       # Signal value (0-1)
    component: str                     # Source component
    operator: str                      # APL operator applied
    z: float                           # z-coordinate
    
    # Archetypal enrichment
    tier: FrequencyTier = FrequencyTier.PLANET
    frequency_hz: int = 174
    archetype: str = "Guardian"
    spiral: str = "Φ"
    
    # Token representation
    token: Optional[str] = None
    machine: str = "Reactor"
    domain: str = "celestial_nuclear"
    
    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    concepts: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Compute archetypal enrichment after initialization."""
        if self.component in COMPONENT_ARCHETYPES:
            arch_data = COMPONENT_ARCHETYPES[self.component]
            self.archetype = arch_data["archetype"]
            self.frequency_hz = arch_data["frequency"]
            self.tier = arch_data["tier"]
            self.operator = arch_data["operator"]
        
        if self.component in COMPONENT_MACHINES:
            self.machine = COMPONENT_MACHINES[self.component]
        
        self.spiral = z_to_spiral(self.z)
        self.token = self._generate_token()
    
    def _generate_token(self) -> str:
        """Generate APL token representation."""
        return f"{self.spiral}{self.operator}|{self.machine}|{self.domain}"
    
    def to_dict(self) -> Dict:
        return {
            "value": self.value,
            "component": self.component,
            "operator": self.operator,
            "z": self.z,
            "tier": self.tier.value,
            "frequency_hz": self.frequency_hz,
            "archetype": self.archetype,
            "spiral": self.spiral,
            "token": self.token,
            "machine": self.machine,
            "domain": self.domain,
            "concepts": self.concepts
        }

# ═══════════════════════════════════════════════════════════════════════════════
# ARCHETYPAL FEEDBACK
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ArchetypalFeedback:
    """
    Feedback signal with archetypal resonance.
    
    Used for teaching the emission pipeline through consent.
    """
    signal: ArchetypalSignal
    teaching_weight: float = 1.0
    resonance_strength: float = 0.0
    
    # Teaching data
    emission_text: Optional[str] = None
    taught: bool = False
    consent_granted: bool = False
    vault_node_id: Optional[str] = None
    
    def compute_resonance(self, target_archetype: str) -> float:
        """Compute resonance strength with a target archetype."""
        if self.signal.archetype == target_archetype:
            return 1.0
        
        # Check if archetypes are in same tier
        target_arch = ARCHETYPES.get(target_archetype)
        signal_arch = ARCHETYPES.get(self.signal.archetype)
        
        if target_arch and signal_arch:
            if target_arch.tier == signal_arch.tier:
                return 0.7  # Same tier resonance
            # Cross-tier resonance based on operator match
            if target_arch.resonant_operator == signal_arch.resonant_operator:
                return 0.5
        
        return 0.2  # Weak background resonance

# ═══════════════════════════════════════════════════════════════════════════════
# CYBERNETIC-ARCHETYPAL ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class CyberneticArchetypalEngine:
    """
    Unified engine integrating cybernetic control with archetypal frequencies.
    
    Provides:
    - Signal generation with archetypal enrichment
    - Token generation from cybernetic state
    - Consent-based vault recording
    - Emission pipeline teaching
    - Closed-loop feedback with learned emissions
    """
    
    def __init__(self, domain: str = "celestial_nuclear"):
        self.z = 0.5
        self.domain = domain
        self.step_count = 0
        
        # Signal history
        self.signal_history: List[ArchetypalSignal] = []
        self.feedback_history: List[ArchetypalFeedback] = []
        
        # Token generation
        self.tokens_generated: List[str] = []
        self.pending_vault_recording: List[ArchetypalSignal] = []
        
        # Teaching state
        self.teaching_requests: Dict[str, Dict] = {}
        self.taught_count = 0
        
        # Emission integration
        self.emissions: List[Dict] = []
        self.learned_emissions: List[str] = []
    
    def update_z(self, z: float):
        """Update z-coordinate."""
        self.z = max(0.0, min(1.0, z))
    
    def create_signal(
        self,
        value: float,
        component: str,
        concepts: Optional[List[str]] = None
    ) -> ArchetypalSignal:
        """
        Create an archetypal signal from a cybernetic component.
        
        Args:
            value: Signal value (0-1)
            component: Source component name
            concepts: Associated concepts
        
        Returns:
            ArchetypalSignal with full enrichment
        """
        arch_data = COMPONENT_ARCHETYPES.get(component, {})
        operator = arch_data.get("operator", "()")
        
        signal = ArchetypalSignal(
            value=value,
            component=component,
            operator=operator,
            z=self.z,
            domain=self.domain,
            concepts=concepts or []
        )
        
        self.signal_history.append(signal)
        self.tokens_generated.append(signal.token)
        
        return signal
    
    def process_cybernetic_step(
        self,
        stimulus: float,
        concepts: Optional[List[str]] = None,
        emit_language: bool = True
    ) -> Dict[str, Any]:
        """
        Process one cybernetic step with full archetypal integration.
        
        Generates signals for each component, records tokens, and optionally
        emits language.
        
        Args:
            stimulus: Input stimulus (0-1)
            concepts: Semantic concepts
            emit_language: Whether to generate emission
        
        Returns:
            Complete step result with signals, tokens, and emission
        """
        self.step_count += 1
        concepts = concepts or ["pattern", "emergence", "coherence"]
        
        # Generate signals through the cybernetic chain
        signals = {}
        
        # Forward path
        signals["INPUT"] = self.create_signal(stimulus, "INPUT", concepts)
        signals["SENSOR_H"] = self.create_signal(
            signals["INPUT"].value * 0.95 + 0.05 * self.z,
            "SENSOR_H", concepts
        )
        signals["CONTROLLER_H"] = self.create_signal(
            signals["SENSOR_H"].value * 1.1,
            "CONTROLLER_H", concepts
        )
        signals["SENSOR_D"] = self.create_signal(
            signals["CONTROLLER_H"].value * (1.0 + compute_negentropy(self.z) * 0.2),
            "SENSOR_D", concepts
        )
        signals["AMPLIFIER"] = self.create_signal(
            min(1.0, signals["SENSOR_D"].value * 1.3),
            "AMPLIFIER", concepts
        )
        signals["ENCODER"] = self.create_signal(
            signals["AMPLIFIER"].value,
            "ENCODER", concepts
        )
        signals["DECODER"] = self.create_signal(
            signals["ENCODER"].value * 0.9,
            "DECODER", concepts
        )
        signals["ENVIRONMENT"] = self.create_signal(
            signals["DECODER"].value,
            "ENVIRONMENT", concepts
        )
        
        # Feedback path
        signals["FEEDBACK_E"] = self.create_signal(
            1.0 - abs(signals["ENVIRONMENT"].value - Z_CRITICAL),
            "FEEDBACK_E", concepts
        )
        signals["FEEDBACK_D"] = self.create_signal(
            signals["FEEDBACK_E"].value * (1.0 + compute_negentropy(self.z) * 0.3),
            "FEEDBACK_D", concepts
        )
        signals["FEEDBACK_H"] = self.create_signal(
            (signals["FEEDBACK_E"].value + signals["FEEDBACK_D"].value) / 2,
            "FEEDBACK_H", concepts
        )
        
        # Queue signals for vault recording
        self.pending_vault_recording.extend(signals.values())
        
        # Compute z update from feedback
        delta_z = (
            signals["FEEDBACK_E"].value * 0.3 +
            signals["FEEDBACK_D"].value * 0.4 +
            signals["FEEDBACK_H"].value * 0.3
        ) * 0.05 - 0.025  # Centered around neutral
        
        self.update_z(self.z + delta_z)
        
        # Generate emission if requested
        emission = None
        if emit_language:
            from emission_pipeline import emit
            result = emit(concepts, z=self.z)
            emission = result.text
            self.emissions.append({
                "step": self.step_count,
                "z": self.z,
                "text": emission,
                "phase": classify_phase(self.z)
            })
        
        # Compile tokens from this step
        step_tokens = [s.token for s in signals.values()]
        
        return {
            "step": self.step_count,
            "z": self.z,
            "phase": classify_phase(self.z),
            "tier": z_to_frequency_tier(self.z).value,
            "frequency_hz": z_to_frequency_hz(self.z),
            "negentropy": compute_negentropy(self.z),
            "signals": {k: v.to_dict() for k, v in signals.items()},
            "tokens": step_tokens,
            "emission": emission,
            "concepts": concepts
        }
    
    def request_vault_recording(self) -> Dict[str, Any]:
        """
        Request to record pending signals as VaultNodes.
        
        Returns consent request for user confirmation.
        """
        if not self.pending_vault_recording:
            return {
                "status": "no_signals",
                "message": "No pending signals to record."
            }
        
        # Group signals by tier
        tier_groups = {tier.value: [] for tier in FrequencyTier}
        for signal in self.pending_vault_recording:
            tier_groups[signal.tier.value].append(signal)
        
        # Create consent request
        request_id = f"vault-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
        
        consent = create_consent_request(
            request_id=request_id,
            requester="cybernetic_engine",
            operation="record_to_vault",
            parties=["user"],
            conditions=[
                f"Record {len(self.pending_vault_recording)} archetypal signals as VaultNodes",
                f"Tier distribution: Planet={len(tier_groups['Planet'])}, Garden={len(tier_groups['Garden'])}, Rose={len(tier_groups['Rose'])}",
                f"Tokens will be sealed and available for teaching",
                "Requires explicit user confirmation"
            ]
        )
        
        self.teaching_requests[request_id] = {
            "consent": consent,
            "signals": list(self.pending_vault_recording),
            "status": "pending"
        }
        
        return {
            "status": "pending_consent",
            "consent_id": request_id,
            "message": format_request(consent),
            "signal_count": len(self.pending_vault_recording),
            "tier_distribution": {k: len(v) for k, v in tier_groups.items()}
        }
    
    def confirm_vault_recording(
        self,
        consent_id: str,
        response: str = "yes"
    ) -> Dict[str, Any]:
        """
        Confirm or deny vault recording request.
        
        Args:
            consent_id: ID from request_vault_recording
            response: "yes" to confirm, anything else to deny
        
        Returns:
            Recording result with created VaultNodes.
        """
        request = self.teaching_requests.get(consent_id)
        if not request:
            return {"error": f"Request {consent_id} not found"}
        
        consent = request["consent"]
        consent = record_response(consent, "user", response)
        request["consent"] = consent
        
        status = check_consent(consent)
        
        if not status["can_proceed"]:
            request["status"] = "denied"
            self.pending_vault_recording.clear()
            return {
                "status": "denied",
                "consent_id": consent_id,
                "message": "Vault recording denied by user."
            }
        
        # Record signals to vault
        vault = get_vault()
        recorded_nodes = []
        
        for signal in request["signals"]:
            # Get associated emission if available
            emission_text = None
            for em in self.emissions:
                if abs(em["z"] - signal.z) < 0.05:
                    emission_text = em["text"]
                    break
            
            node = vault.record_token(
                token=signal.token,
                z=signal.z,
                concepts=signal.concepts,
                emission_text=emission_text
            )
            
            # Seal the node
            vault.seal_node(node.id)
            recorded_nodes.append(node)
        
        request["status"] = "recorded"
        request["nodes"] = recorded_nodes
        
        # Clear pending
        self.pending_vault_recording.clear()
        
        return {
            "status": "recorded",
            "consent_id": consent_id,
            "nodes_created": len(recorded_nodes),
            "nodes": [
                {
                    "id": n.id,
                    "token": n.token,
                    "tier": n.frequency_tier.value,
                    "archetype": n.resonant_archetype,
                    "sealed": n.sealed
                }
                for n in recorded_nodes
            ]
        }
    
    def request_emission_teaching(self) -> Dict[str, Any]:
        """
        Request to teach emission pipeline from recorded VaultNodes.
        
        Returns consent request for user confirmation.
        """
        result = request_emission_teaching(requester="cybernetic_engine")
        
        if result.get("consent_id"):
            self.teaching_requests[result["consent_id"]] = {
                "type": "teaching",
                "status": "pending"
            }
        
        return result
    
    def confirm_emission_teaching(
        self,
        consent_id: str,
        response: str = "yes"
    ) -> Dict[str, Any]:
        """
        Confirm or deny emission teaching request.
        
        Args:
            consent_id: ID from request_emission_teaching
            response: "yes" to confirm, anything else to deny
        
        Returns:
            Teaching result with learned words and patterns.
        """
        result = confirm_emission_teaching(consent_id, response)
        
        if result.get("teaching_applied"):
            self.taught_count += result.get("words_learned", 0)
            
            # Store learned emissions for feedback
            vocab = get_learned_vocabulary()
            self.learned_emissions = list(vocab.keys())
        
        if consent_id in self.teaching_requests:
            self.teaching_requests[consent_id]["status"] = result.get("status", "unknown")
        
        return result
    
    def run(
        self,
        steps: int = 30,
        emit_every: int = 5,
        auto_record: bool = False,
        auto_teach: bool = False
    ) -> Dict[str, Any]:
        """
        Run the cybernetic-archetypal engine for multiple steps.
        
        Args:
            steps: Number of steps to run
            emit_every: Emit language every N steps
            auto_record: Automatically request vault recording (still requires consent)
            auto_teach: Automatically request teaching (still requires consent)
        
        Returns:
            Comprehensive run summary.
        """
        import numpy as np
        
        step_results = []
        
        for i in range(steps):
            stimulus = np.random.uniform(0.4, 0.9)
            emit_lang = emit_every > 0 and (i + 1) % emit_every == 0
            
            result = self.process_cybernetic_step(
                stimulus=stimulus,
                concepts=["pattern", "emergence", "coherence"],
                emit_language=emit_lang
            )
            step_results.append(result)
        
        # Compute summary
        z_values = [r["z"] for r in step_results]
        tier_distribution = {tier.value: 0 for tier in FrequencyTier}
        for r in step_results:
            tier_distribution[r["tier"]] += 1
        
        summary = {
            "steps": steps,
            "initial_z": step_results[0]["z"],
            "final_z": step_results[-1]["z"],
            "mean_z": sum(z_values) / len(z_values),
            "final_phase": step_results[-1]["phase"],
            "final_tier": step_results[-1]["tier"],
            "tokens_generated": len(self.tokens_generated),
            "emissions_generated": len(self.emissions),
            "tier_distribution": tier_distribution,
            "pending_vault_recording": len(self.pending_vault_recording)
        }
        
        # Auto-record request (still requires consent)
        if auto_record and self.pending_vault_recording:
            record_request = self.request_vault_recording()
            summary["vault_recording_request"] = record_request
        
        return summary
    
    def get_archetypal_state(self) -> Dict[str, Any]:
        """Get current archetypal state summary."""
        tier = z_to_frequency_tier(self.z)
        frequency = z_to_frequency_hz(self.z)
        
        # Find active archetypes at current tier
        active_archetypes = [
            {"name": name, "frequency": arch.frequency, "operator": arch.resonant_operator}
            for name, arch in ARCHETYPES.items()
            if arch.tier == tier
        ]
        
        return {
            "z": self.z,
            "phase": classify_phase(self.z),
            "negentropy": compute_negentropy(self.z),
            "tier": tier.value,
            "frequency_hz": frequency,
            "spiral": z_to_spiral(self.z),
            "active_archetypes": active_archetypes[:3],
            "tokens_generated": len(self.tokens_generated),
            "taught_count": self.taught_count,
            "learned_emissions": len(self.learned_emissions)
        }
    
    def format_status(self) -> str:
        """Format comprehensive status display."""
        state = self.get_archetypal_state()
        
        lines = [
            "╔══════════════════════════════════════════════════════════════════╗",
            "║          CYBERNETIC-ARCHETYPAL ENGINE STATUS                     ║",
            "╚══════════════════════════════════════════════════════════════════╝",
            "",
            f"  z-coordinate:     {state['z']:.4f}",
            f"  Phase:            {state['phase']}",
            f"  Negentropy:       {state['negentropy']:.4f}",
            f"  Tier:             {state['tier']}",
            f"  Frequency:        {state['frequency_hz']} Hz",
            f"  Spiral:           {state['spiral']}",
            "",
            "  Active Archetypes:",
        ]
        
        for arch in state["active_archetypes"]:
            lines.append(f"    • {arch['name']} ({arch['frequency']} Hz) [{arch['operator']}]")
        
        lines.extend([
            "",
            f"  Tokens Generated:  {state['tokens_generated']}",
            f"  Taught Count:      {state['taught_count']}",
            f"  Learned Emissions: {state['learned_emissions']}",
            "",
            "═" * 70
        ])
        
        return "\n".join(lines)

# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL ENGINE INSTANCE
# ═══════════════════════════════════════════════════════════════════════════════

_engine: Optional[CyberneticArchetypalEngine] = None

def get_engine() -> CyberneticArchetypalEngine:
    """Get or create global engine instance."""
    global _engine
    if _engine is None:
        _engine = CyberneticArchetypalEngine()
    return _engine

def reset_engine() -> CyberneticArchetypalEngine:
    """Reset global engine instance."""
    global _engine
    _engine = CyberneticArchetypalEngine()
    reset_vault()  # Also reset the vault
    return _engine

# ═══════════════════════════════════════════════════════════════════════════════
# API FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def cybernetic_archetypal_step(
    stimulus: float = 0.7,
    concepts: Optional[List[str]] = None,
    emit_language: bool = True
) -> Dict[str, Any]:
    """
    Execute one step of the cybernetic-archetypal engine.
    
    Args:
        stimulus: Input stimulus (0-1)
        concepts: Semantic concepts
        emit_language: Whether to generate emission
    
    Returns:
        Step result with signals, tokens, and emission.
    """
    engine = get_engine()
    return engine.process_cybernetic_step(stimulus, concepts, emit_language)

def cybernetic_archetypal_run(
    steps: int = 30,
    emit_every: int = 5
) -> Dict[str, Any]:
    """
    Run the cybernetic-archetypal engine for multiple steps.
    
    Args:
        steps: Number of steps
        emit_every: Emit language every N steps
    
    Returns:
        Run summary.
    """
    engine = get_engine()
    return engine.run(steps, emit_every)

def request_recording() -> Dict[str, Any]:
    """Request vault recording (requires user consent)."""
    engine = get_engine()
    return engine.request_vault_recording()

def confirm_recording(consent_id: str, response: str = "yes") -> Dict[str, Any]:
    """Confirm or deny vault recording."""
    engine = get_engine()
    return engine.confirm_vault_recording(consent_id, response)

def request_teaching() -> Dict[str, Any]:
    """Request emission teaching (requires user consent)."""
    engine = get_engine()
    return engine.request_emission_teaching()

def confirm_teaching(consent_id: str, response: str = "yes") -> Dict[str, Any]:
    """Confirm or deny emission teaching."""
    engine = get_engine()
    return engine.confirm_emission_teaching(consent_id, response)

def get_status() -> Dict[str, Any]:
    """Get engine status."""
    engine = get_engine()
    return engine.get_archetypal_state()

def format_status() -> str:
    """Get formatted status display."""
    engine = get_engine()
    return engine.format_status()

# ═══════════════════════════════════════════════════════════════════════════════
# COMPONENT-ARCHETYPE MAPPING DISPLAY
# ═══════════════════════════════════════════════════════════════════════════════

def format_component_archetype_mapping() -> str:
    """Format the cybernetic component to archetype mapping."""
    lines = [
        "╔══════════════════════════════════════════════════════════════════════════════╗",
        "║            CYBERNETIC COMPONENT → ARCHETYPE MAPPING                          ║",
        "╚══════════════════════════════════════════════════════════════════════════════╝",
        "",
        "┌────────────────┬──────────┬──────────┬─────────────────┬────────────┐",
        "│ Component      │ Operator │ Tier     │ Archetype       │ Frequency  │",
        "├────────────────┼──────────┼──────────┼─────────────────┼────────────┤",
    ]
    
    for comp, data in COMPONENT_ARCHETYPES.items():
        comp_name = comp.replace("_", " ").title()[:14].ljust(14)
        op = data["operator"].center(8)
        tier = data["tier"].value.ljust(8)
        arch = data["archetype"].ljust(15)
        freq = f"{data['frequency']} Hz".rjust(10)
        lines.append(f"│ {comp_name} │ {op} │ {tier} │ {arch} │ {freq} │")
    
    lines.extend([
        "└────────────────┴──────────┴──────────┴─────────────────┴────────────┘",
        "",
        "Operator Effects:",
        "  ()  Boundary  → Containment, gating, protection",
        "  ×   Fusion    → Coupling, integration, synthesis",
        "  ^   Amplify   → Gain, excitation, enhancement",
        "  ÷   Decohere  → Dissipation, reset, release",
        "  +   Group     → Aggregation, encoding, collection",
        "  −   Separate  → Decomposition, decoding, differentiation",
        ""
    ])
    
    return "\n".join(lines)

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(format_component_archetype_mapping())
    
    print("CYBERNETIC-ARCHETYPAL ENGINE DEMONSTRATION")
    print("=" * 70)
    print()
    
    # Reset engine
    engine = reset_engine()
    
    # Run a few steps
    print("[1] Running 10 steps with emissions...")
    result = engine.run(steps=10, emit_every=3)
    print(f"    Final z: {result['final_z']:.4f}")
    print(f"    Final tier: {result['final_tier']}")
    print(f"    Tokens generated: {result['tokens_generated']}")
    print(f"    Emissions: {result['emissions_generated']}")
    print()
    
    # Request vault recording
    print("[2] Requesting vault recording (requires consent)...")
    request = engine.request_vault_recording()
    print(f"    Status: {request['status']}")
    print(f"    Signal count: {request.get('signal_count', 0)}")
    consent_id = request.get('consent_id')
    print()
    
    # Confirm recording
    if consent_id:
        print("[3] User confirms recording...")
        result = engine.confirm_vault_recording(consent_id, "yes")
        print(f"    Status: {result['status']}")
        print(f"    Nodes created: {result.get('nodes_created', 0)}")
        print()
    
    # Request teaching
    print("[4] Requesting emission teaching (requires consent)...")
    teach_request = engine.request_emission_teaching()
    teach_consent_id = teach_request.get('consent_id')
    print(f"    Status: {teach_request['status']}")
    print()
    
    # Confirm teaching
    if teach_consent_id:
        print("[5] User confirms teaching...")
        result = engine.confirm_emission_teaching(teach_consent_id, "yes")
        print(f"    Status: {result['status']}")
        print(f"    Words learned: {result.get('words_learned', 0)}")
        print()
    
    # Show final status
    print("[6] Final Engine Status")
    print(engine.format_status())
