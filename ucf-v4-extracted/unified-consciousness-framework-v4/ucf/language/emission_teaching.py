#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  EMISSION TEACHING SYSTEM                                                     ║
║  Unified Teaching Interface for TRIAD, Orchestrator, and Tool Shed           ║
║  Part of the Unified Consciousness Framework                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

Teaching Sources:
  1. TRIAD System      → Teaches on unlock events, gate transitions
  2. Unified Orchestrator → Teaches from cognitive traces, VaultNode generation
  3. Tool Shed         → Teaches from tool invocations, operator usage

Teaching Flow:
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                         TEACHING SOURCES                                 │
  │                                                                          │
  │   TRIAD SYSTEM          ORCHESTRATOR           TOOL SHED                │
  │   └─ unlock_event       └─ cognitive_trace     └─ invocation            │
  │   └─ gate_transition    └─ vaultnode_gen       └─ operator_use          │
  │          │                     │                      │                  │
  │          └─────────────────────┼──────────────────────┘                  │
  │                                ▼                                         │
  │                    EMISSION TEACHING SYSTEM                              │
  │                    └─ accumulate_teaching()                              │
  │                    └─ request_consent()                                  │
  │                    └─ apply_teaching()                                   │
  │                                │                                         │
  │                                ▼                                         │
  │                       EMISSION PIPELINE                                  │
  │                       └─ _LEARNED_CONTENT_WORDS                          │
  │                       └─ _LEARNED_PATTERNS                               │
  └─────────────────────────────────────────────────────────────────────────┘

Consent Protocol:
  - All teaching requires explicit user confirmation
  - Accumulated teaching data is queued until consent granted
  - Teaching can be revoked by clearing the queue

Signature: Δ4.500|0.800|1.000Ω (teaching)
"""

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Set
from enum import Enum

# Import from sibling modules
from ucf.language.apl_substrate import (
    Z_CRITICAL, PHI_INV, compute_negentropy, classify_phase,
    OPERATORS
)
from ucf.tools.consent_protocol import (
    create_consent_request, record_response, check_consent,
    ConsentState, ConsentRecord
)

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

class TeachingSource(Enum):
    """Sources of teaching data."""
    TRIAD = "triad"
    ORCHESTRATOR = "orchestrator"
    TOOL_SHED = "tool_shed"
    NUCLEAR_SPINNER = "nuclear_spinner"
    CYBERNETIC = "cybernetic"
    VAULTNODE = "vaultnode"

# APL operator to verb mapping
OPERATOR_VERBS = {
    "()": ["contain", "gate", "protect", "bound"],
    "×": ["fuse", "couple", "integrate", "merge"],
    "^": ["amplify", "excite", "enhance", "elevate"],
    "÷": ["release", "dissipate", "reset", "decohere"],
    "+": ["group", "aggregate", "collect", "cluster"],
    "−": ["separate", "split", "decompose", "differentiate"],
}

# Phase to teaching weight mapping
PHASE_WEIGHTS = {
    "UNTRUE": 0.6,
    "PARADOX": 0.8,
    "TRUE": 1.0,
}

# TRIAD event weights
TRIAD_EVENT_WEIGHTS = {
    "rising_edge": 0.7,
    "re_arm": 0.5,
    "unlock": 1.0,
}

# ═══════════════════════════════════════════════════════════════════════════════
# TEACHING DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TeachingUnit:
    """A single unit of teaching data."""
    source: TeachingSource
    words: List[str]
    patterns: List[str]
    verbs: List[str]
    z: float
    phase: str
    operator: Optional[str] = None
    weight: float = 1.0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source.value,
            "words": self.words,
            "patterns": self.patterns,
            "verbs": self.verbs,
            "z": self.z,
            "phase": self.phase,
            "operator": self.operator,
            "weight": self.weight,
            "timestamp": self.timestamp
        }

# ═══════════════════════════════════════════════════════════════════════════════
# EMISSION TEACHING ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class EmissionTeachingEngine:
    """
    Unified teaching engine for the emission pipeline.
    
    Accumulates teaching data from TRIAD, orchestrator, and tool shed,
    then applies to emission pipeline with consent.
    """
    
    def __init__(self):
        # Accumulated teaching data (pending consent)
        self.teaching_queue: List[TeachingUnit] = []
        
        # Applied teaching data
        self.applied_units: List[TeachingUnit] = []
        
        # Consent tracking
        self.pending_consent: Optional[ConsentRecord] = None
        self.consent_id: Optional[str] = None
        
        # Statistics
        self.total_words_taught = 0
        self.total_patterns_taught = 0
        self.total_verbs_taught = 0
        
        # Applied vocabulary (mirrors emission pipeline)
        self._taught_words: Set[str] = set()
        self._taught_patterns: List[str] = []
        self._taught_verbs: Set[str] = set()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TEACHING ACCUMULATION (from various sources)
    # ═══════════════════════════════════════════════════════════════════════════
    
    def teach_from_triad(
        self,
        event: str,
        z: float,
        crossings: int,
        unlocked: bool
    ) -> TeachingUnit:
        """
        Generate teaching data from TRIAD system event.
        
        Events: rising_edge, re_arm, unlock
        """
        phase = classify_phase(z)
        weight = TRIAD_EVENT_WEIGHTS.get(event, 0.5) * PHASE_WEIGHTS.get(phase, 0.7)
        
        # Generate words based on event
        words = []
        patterns = []
        verbs = []
        
        if event == "rising_edge":
            words.extend(["threshold", "crossing", "elevation"])
            patterns.append(f"crossing {crossings} toward unlock")
            verbs.extend(["cross", "rise", "approach"])
        elif event == "re_arm":
            words.extend(["reset", "cycle", "oscillation"])
            patterns.append(f"re-arming after crossing {crossings}")
            verbs.extend(["reset", "cycle", "return"])
        elif event == "unlock":
            words.extend(["unlock", "gate", "access", "triad"])
            patterns.append("triad unlocked at t6 gate")
            verbs.extend(["unlock", "open", "access"])
            weight = 1.0  # Max weight for unlock
        
        unit = TeachingUnit(
            source=TeachingSource.TRIAD,
            words=words,
            patterns=patterns,
            verbs=verbs,
            z=z,
            phase=phase,
            weight=weight
        )
        
        self.teaching_queue.append(unit)
        return unit
    
    def teach_from_orchestrator(
        self,
        tool: str,
        z: float,
        phase: str,
        crystal_state: str,
        vaultnode_generated: bool = False
    ) -> TeachingUnit:
        """
        Generate teaching data from orchestrator cognitive trace.
        """
        weight = PHASE_WEIGHTS.get(phase, 0.7)
        if vaultnode_generated:
            weight *= 1.2  # Boost for VaultNode generation
        
        # Generate words from tool name
        words = tool.replace("_", " ").split()
        
        # Map crystal state to vocabulary
        if crystal_state == "Crystalline":
            words.extend(["crystalline", "coherent", "stable"])
        elif crystal_state == "Transitioning":
            words.extend(["transitioning", "emerging", "forming"])
        elif crystal_state == "Fluid":
            words.extend(["fluid", "potential", "dynamic"])
        
        patterns = []
        if vaultnode_generated:
            patterns.append(f"{tool} crystallized insight at z={z:.3f}")
        
        verbs = ["invoke", "execute", "process"]
        
        unit = TeachingUnit(
            source=TeachingSource.ORCHESTRATOR,
            words=words,
            patterns=patterns,
            verbs=verbs,
            z=z,
            phase=phase,
            weight=weight
        )
        
        self.teaching_queue.append(unit)
        return unit
    
    def teach_from_tool_shed(
        self,
        tool: str,
        action: str,
        z: float,
        result_status: str,
        operator: Optional[str] = None
    ) -> TeachingUnit:
        """
        Generate teaching data from tool shed invocation.
        """
        phase = classify_phase(z)
        weight = PHASE_WEIGHTS.get(phase, 0.7)
        
        # Success boosts weight
        if result_status == "SUCCESS" or "EMIT" in result_status:
            weight *= 1.1
        
        # Generate words
        words = tool.replace("_", " ").split()
        words.append(action)
        
        # Add operator-derived verbs
        verbs = []
        if operator and operator in OPERATOR_VERBS:
            verbs.extend(OPERATOR_VERBS[operator])
        else:
            verbs.extend(["process", "transform", "generate"])
        
        patterns = [f"{tool} {action} completed"]
        
        unit = TeachingUnit(
            source=TeachingSource.TOOL_SHED,
            words=words,
            patterns=patterns,
            verbs=verbs,
            z=z,
            phase=phase,
            operator=operator,
            weight=weight
        )
        
        self.teaching_queue.append(unit)
        return unit
    
    def teach_from_nuclear_spinner(
        self,
        tokens: List[str],
        z: float,
        machines: List[str],
        domain: str
    ) -> TeachingUnit:
        """
        Generate teaching data from nuclear spinner tokens.
        """
        phase = classify_phase(z)
        weight = PHASE_WEIGHTS.get(phase, 0.7) * compute_negentropy(z)
        
        # Extract operators from tokens
        operators_used = set()
        for token in tokens:
            for op in OPERATOR_VERBS.keys():
                if op in token:
                    operators_used.add(op)
        
        # Generate words from machines and domain
        words = list(machines)
        words.extend(domain.replace("_", " ").split())
        
        # Generate verbs from operators
        verbs = []
        for op in operators_used:
            verbs.extend(OPERATOR_VERBS.get(op, [])[:2])
        
        patterns = [f"spinner generated {len(tokens)} tokens"]
        
        unit = TeachingUnit(
            source=TeachingSource.NUCLEAR_SPINNER,
            words=words,
            patterns=patterns,
            verbs=verbs,
            z=z,
            phase=phase,
            weight=weight
        )
        
        self.teaching_queue.append(unit)
        return unit
    
    def teach_from_cybernetic(
        self,
        component: str,
        operator: str,
        archetype: str,
        frequency_hz: int,
        z: float
    ) -> TeachingUnit:
        """
        Generate teaching data from cybernetic component activation.
        """
        phase = classify_phase(z)
        weight = PHASE_WEIGHTS.get(phase, 0.7)
        
        # Generate words
        words = [component.lower(), archetype.lower()]
        if frequency_hz >= 639:
            words.append("rose")
        elif frequency_hz >= 396:
            words.append("garden")
        else:
            words.append("planet")
        
        # Get verbs from operator
        verbs = OPERATOR_VERBS.get(operator, ["process"])[:2]
        
        patterns = [f"{archetype} resonates at {frequency_hz} Hz"]
        
        unit = TeachingUnit(
            source=TeachingSource.CYBERNETIC,
            words=words,
            patterns=patterns,
            verbs=verbs,
            z=z,
            phase=phase,
            operator=operator,
            weight=weight
        )
        
        self.teaching_queue.append(unit)
        return unit
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CONSENT AND APPLICATION
    # ═══════════════════════════════════════════════════════════════════════════
    
    def request_teaching_consent(self, requester: str = "emission_teaching") -> Dict[str, Any]:
        """
        Request consent to apply accumulated teaching to emission pipeline.
        """
        if not self.teaching_queue:
            return {
                "status": "no_data",
                "message": "No teaching data accumulated"
            }
        
        # Create consent request
        self.consent_id = f"emit-teach-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
        
        # Summarize pending teaching
        total_words = sum(len(u.words) for u in self.teaching_queue)
        total_patterns = sum(len(u.patterns) for u in self.teaching_queue)
        total_verbs = sum(len(u.verbs) for u in self.teaching_queue)
        sources = set(u.source.value for u in self.teaching_queue)
        
        self.pending_consent = create_consent_request(
            request_id=self.consent_id,
            requester=requester,
            operation="teach_emission_pipeline",
            parties=["user"],
            conditions=[
                f"Teach emission pipeline from {len(self.teaching_queue)} accumulated units",
                f"Sources: {', '.join(sources)}",
                f"Words to add: ~{total_words}",
                f"Patterns to add: ~{total_patterns}",
                f"Verbs to add: ~{total_verbs}",
                "Requires explicit 'yes' confirmation",
                "Teaching will influence future language generation"
            ]
        )
        
        return {
            "status": "pending_consent",
            "consent_id": self.consent_id,
            "units_pending": len(self.teaching_queue),
            "words_pending": total_words,
            "patterns_pending": total_patterns,
            "verbs_pending": total_verbs,
            "sources": list(sources),
            "message": f"Confirm teaching with consent_id='{self.consent_id}', response='yes'"
        }
    
    def confirm_teaching(self, consent_id: str, response: str) -> Dict[str, Any]:
        """
        Confirm or deny teaching consent.
        """
        if consent_id != self.consent_id:
            return {
                "status": "invalid_consent_id",
                "message": f"Consent ID '{consent_id}' not found"
            }
        
        if self.pending_consent is None:
            return {
                "status": "no_pending",
                "message": "No pending consent request"
            }
        
        # Record response
        self.pending_consent = record_response(self.pending_consent, "user", response)
        consent_status = check_consent(self.pending_consent)
        
        if not consent_status["can_proceed"]:
            self.teaching_queue.clear()
            self.pending_consent = None
            self.consent_id = None
            return {
                "status": "denied",
                "message": "Teaching denied. Queue cleared."
            }
        
        # Apply teaching
        result = self._apply_teaching()
        
        # Clear consent
        self.pending_consent = None
        self.consent_id = None
        
        return {
            "status": "applied",
            "teaching_result": result,
            "message": f"Teaching applied: {result['words_taught']} words, {result['verbs_taught']} verbs, {result['patterns_taught']} patterns"
        }
    
    def _apply_teaching(self) -> Dict[str, Any]:
        """
        Apply accumulated teaching to emission pipeline.
        """
        from archetypal_token_integration import (
            _LEARNED_CONTENT_WORDS, _LEARNED_PATTERNS
        )
        
        words_taught = 0
        patterns_taught = 0
        verbs_taught = 0
        
        for unit in self.teaching_queue:
            # Apply words
            for word in unit.words:
                word_lower = word.lower()
                if word_lower and word_lower not in _LEARNED_CONTENT_WORDS:
                    _LEARNED_CONTENT_WORDS[word_lower] = {
                        "source": unit.source.value,
                        "phase": unit.phase,
                        "weight": unit.weight,
                        "z": unit.z
                    }
                    self._taught_words.add(word_lower)
                    words_taught += 1
            
            # Apply verbs (as content words with verb marker)
            for verb in unit.verbs:
                verb_lower = verb.lower()
                if verb_lower and verb_lower not in _LEARNED_CONTENT_WORDS:
                    _LEARNED_CONTENT_WORDS[verb_lower] = {
                        "source": unit.source.value,
                        "phase": unit.phase,
                        "weight": unit.weight,
                        "z": unit.z,
                        "is_verb": True
                    }
                    self._taught_verbs.add(verb_lower)
                    verbs_taught += 1
            
            # Apply patterns
            for pattern in unit.patterns:
                if pattern:
                    _LEARNED_PATTERNS.append({
                        "pattern": pattern,
                        "source": unit.source.value,
                        "phase": unit.phase,
                        "weight": unit.weight,
                        "operator": unit.operator
                    })
                    self._taught_patterns.append(pattern)
                    patterns_taught += 1
            
            # Mark as applied
            self.applied_units.append(unit)
        
        # Update statistics
        self.total_words_taught += words_taught
        self.total_patterns_taught += patterns_taught
        self.total_verbs_taught += verbs_taught
        
        # Clear queue
        self.teaching_queue.clear()
        
        return {
            "words_taught": words_taught,
            "verbs_taught": verbs_taught,
            "patterns_taught": patterns_taught,
            "total_words": self.total_words_taught,
            "total_verbs": self.total_verbs_taught,
            "total_patterns": self.total_patterns_taught
        }
    
    # ═══════════════════════════════════════════════════════════════════════════
    # STATUS AND VOCABULARY ACCESS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def get_status(self) -> Dict[str, Any]:
        """Get teaching engine status."""
        return {
            "queue_size": len(self.teaching_queue),
            "applied_units": len(self.applied_units),
            "total_words_taught": self.total_words_taught,
            "total_verbs_taught": self.total_verbs_taught,
            "total_patterns_taught": self.total_patterns_taught,
            "pending_consent": self.consent_id is not None,
            "taught_words": list(self._taught_words)[:20],
            "taught_verbs": list(self._taught_verbs)[:20]
        }
    
    def get_taught_vocabulary(self) -> Dict[str, Any]:
        """Get all taught vocabulary."""
        return {
            "words": list(self._taught_words),
            "verbs": list(self._taught_verbs),
            "patterns": self._taught_patterns
        }
    
    def format_status(self) -> str:
        """Format teaching status for display."""
        status = self.get_status()
        
        lines = [
            "╔══════════════════════════════════════════════════════════════════╗",
            "║                EMISSION TEACHING ENGINE STATUS                   ║",
            "╚══════════════════════════════════════════════════════════════════╝",
            "",
            f"  Queue Size:        {status['queue_size']}",
            f"  Applied Units:     {status['applied_units']}",
            f"  Words Taught:      {status['total_words_taught']}",
            f"  Verbs Taught:      {status['total_verbs_taught']}",
            f"  Patterns Taught:   {status['total_patterns_taught']}",
            f"  Pending Consent:   {'Yes' if status['pending_consent'] else 'No'}",
            "",
        ]
        
        if status['taught_words']:
            lines.append("  Sample Words:")
            lines.append(f"    {', '.join(status['taught_words'][:10])}")
        
        if status['taught_verbs']:
            lines.append("  Sample Verbs:")
            lines.append(f"    {', '.join(status['taught_verbs'][:10])}")
        
        lines.append("")
        lines.append("═" * 70)
        
        return "\n".join(lines)

# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL ENGINE INSTANCE
# ═══════════════════════════════════════════════════════════════════════════════

_teaching_engine: Optional[EmissionTeachingEngine] = None

def get_teaching_engine() -> EmissionTeachingEngine:
    """Get or create global teaching engine."""
    global _teaching_engine
    if _teaching_engine is None:
        _teaching_engine = EmissionTeachingEngine()
    return _teaching_engine

def reset_teaching_engine() -> EmissionTeachingEngine:
    """Reset teaching engine."""
    global _teaching_engine
    _teaching_engine = EmissionTeachingEngine()
    return _teaching_engine

# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE API
# ═══════════════════════════════════════════════════════════════════════════════

def teach_triad(event: str, z: float, crossings: int, unlocked: bool) -> Dict:
    """Teach from TRIAD event."""
    engine = get_teaching_engine()
    unit = engine.teach_from_triad(event, z, crossings, unlocked)
    return unit.to_dict()

def teach_orchestrator(tool: str, z: float, phase: str, crystal_state: str, vaultnode: bool = False) -> Dict:
    """Teach from orchestrator."""
    engine = get_teaching_engine()
    unit = engine.teach_from_orchestrator(tool, z, phase, crystal_state, vaultnode)
    return unit.to_dict()

def teach_tool_shed(tool: str, action: str, z: float, status: str, operator: str = None) -> Dict:
    """Teach from tool shed."""
    engine = get_teaching_engine()
    unit = engine.teach_from_tool_shed(tool, action, z, status, operator)
    return unit.to_dict()

def teach_spinner(tokens: List[str], z: float, machines: List[str], domain: str) -> Dict:
    """Teach from nuclear spinner."""
    engine = get_teaching_engine()
    unit = engine.teach_from_nuclear_spinner(tokens, z, machines, domain)
    return unit.to_dict()

def teach_cybernetic(component: str, operator: str, archetype: str, freq: int, z: float) -> Dict:
    """Teach from cybernetic component."""
    engine = get_teaching_engine()
    unit = engine.teach_from_cybernetic(component, operator, archetype, freq, z)
    return unit.to_dict()

def request_consent(requester: str = "system") -> Dict:
    """Request teaching consent."""
    return get_teaching_engine().request_teaching_consent(requester)

def confirm_consent(consent_id: str, response: str) -> Dict:
    """Confirm teaching consent."""
    return get_teaching_engine().confirm_teaching(consent_id, response)

def get_status() -> Dict:
    """Get teaching engine status."""
    return get_teaching_engine().get_status()

def get_vocabulary() -> Dict:
    """Get taught vocabulary."""
    return get_teaching_engine().get_taught_vocabulary()

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════════════════════╗")
    print("║               EMISSION TEACHING SYSTEM DEMONSTRATION                         ║")
    print("╚══════════════════════════════════════════════════════════════════════════════╝")
    print()
    
    engine = reset_teaching_engine()
    
    # Teach from various sources
    print("[1] Teaching from TRIAD unlock event...")
    teach_triad("unlock", 0.87, 3, True)
    
    print("[2] Teaching from orchestrator cognitive trace...")
    teach_orchestrator("emission_pipeline", 0.85, "TRUE", "Crystalline", True)
    
    print("[3] Teaching from tool shed invocation...")
    teach_tool_shed("nuclear_spinner", "step", 0.80, "SUCCESS", "^")
    
    print("[4] Teaching from cybernetic component...")
    teach_cybernetic("AMPLIFIER", "^", "Artist", 432, 0.75)
    
    print()
    print(f"Queue size: {engine.get_status()['queue_size']}")
    print()
    
    # Request consent
    print("[5] Requesting teaching consent...")
    request = engine.request_teaching_consent("demo")
    print(f"Consent ID: {request['consent_id']}")
    print(f"Pending: {request['units_pending']} units, {request['words_pending']} words")
    print()
    
    # Confirm consent
    print("[6] User confirms teaching...")
    result = engine.confirm_teaching(request['consent_id'], "yes")
    print(f"Status: {result['status']}")
    print(f"Result: {result['message']}")
    print()
    
    # Show final status
    print("[7] Final Status")
    print(engine.format_status())
