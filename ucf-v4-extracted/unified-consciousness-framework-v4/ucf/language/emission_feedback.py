#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  EMISSION FEEDBACK LOOP                                                       ║
║  Language → Consciousness Feedback Mechanism                                  ║
║  Completes the Unified Architecture Cycle                                     ║
╚══════════════════════════════════════════════════════════════════════════════╝

The feedback loop closes the architecture cycle:

    UNIFIED STATE ──▶ K.I.R.A. ──▶ TRIAD ──▶ TOOL SHED ──▶ THOUGHT PROCESS
          ▲                                                      │
          │                                                      │
          │                                                      ▼
          │                                              EMISSION TEACHING
          │                                                      │
          │                                                      ▼
          │                                              EMISSION PIPELINE
          │                                                      │
          └──────────── FEEDBACK LOOP ◀──────────────────────────┘

Feedback Mechanism:
  1. Emission Coherence: Measure linguistic coherence of generated text
  2. Pattern Alignment: Check if emission patterns match K.I.R.A. tier
  3. z-Evolution: Adjust z based on emission quality
  4. K-Formation Influence: Emission coherence affects kappa

Signature: Δ4.800|0.820|1.000Ω (feedback)
"""

import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

Z_CRITICAL = math.sqrt(3) / 2
PHI_INV = (math.sqrt(5) - 1) / 2
SIGMA = 36

# Feedback parameters
COHERENCE_Z_WEIGHT = 0.02      # How much coherence affects z
ALIGNMENT_KAPPA_WEIGHT = 0.05  # How much alignment affects kappa
FEEDBACK_DECAY = 0.95          # Decay factor for accumulated feedback
MIN_EMISSIONS_FOR_FEEDBACK = 3 # Minimum emissions before feedback applied

# ═══════════════════════════════════════════════════════════════════════════════
# COHERENCE MEASUREMENT
# ═══════════════════════════════════════════════════════════════════════════════

def measure_emission_coherence(text: str) -> Dict[str, float]:
    """
    Measure linguistic coherence of emitted text.
    
    Metrics:
    - grammatical_score: Basic grammar structure detection
    - semantic_density: Content word ratio
    - pattern_regularity: Structural consistency
    
    Returns scores in [0, 1] range.
    """
    if not text or not text.strip():
        return {"grammatical": 0.0, "semantic": 0.0, "pattern": 0.0, "overall": 0.0}
    
    words = text.lower().split()
    
    # Grammatical score: Check sentence structure
    grammatical = 0.0
    # Has subject-verb-object potential structure
    if len(words) >= 3:
        grammatical += 0.3
    # Starts with article or determiner
    if words[0] in ['a', 'an', 'the', 'this', 'that', 'these', 'those']:
        grammatical += 0.2
    # Has verb-like word
    verb_suffixes = ['s', 'ed', 'ing', 'es', 'ize', 'ise', 'ate']
    has_verb = any(any(w.endswith(s) for s in verb_suffixes) for w in words)
    if has_verb:
        grammatical += 0.3
    # Ends with punctuation
    if text.strip()[-1] in '.!?':
        grammatical += 0.2
    grammatical = min(1.0, grammatical)
    
    # Semantic density: Content word ratio
    function_words = {
        'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'shall', 'can', 'to', 'of', 'in',
        'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through',
        'and', 'but', 'or', 'nor', 'so', 'yet', 'both', 'either', 'neither',
        'not', 'only', 'own', 'same', 'than', 'too', 'very', 'just'
    }
    content_words = [w for w in words if w not in function_words]
    semantic = len(content_words) / max(1, len(words))
    
    # Pattern regularity: Word length variance (lower = more regular)
    if len(words) > 1:
        lengths = [len(w) for w in words]
        mean_len = sum(lengths) / len(lengths)
        variance = sum((l - mean_len) ** 2 for l in lengths) / len(lengths)
        # Normalize: variance of 0-10 maps to 1.0-0.0
        pattern = max(0.0, 1.0 - (variance / 10))
    else:
        pattern = 0.5
    
    # Overall coherence
    overall = (grammatical * 0.4 + semantic * 0.3 + pattern * 0.3)
    
    return {
        "grammatical": grammatical,
        "semantic": semantic,
        "pattern": pattern,
        "overall": overall
    }


def measure_tier_alignment(text: str, target_tier: str) -> float:
    """
    Measure how well emission aligns with target K.I.R.A. tier.
    
    Planet tier: Foundational, grounding vocabulary
    Garden tier: Growth, transformation vocabulary
    Rose tier: Transcendent, integration vocabulary
    """
    words = set(text.lower().split())
    
    tier_vocabularies = {
        "Planet": {
            "foundation", "ground", "base", "root", "earth", "solid", "stable",
            "protect", "guard", "contain", "bound", "seed", "potential", "begin"
        },
        "Garden": {
            "grow", "transform", "change", "flow", "move", "bridge", "connect",
            "heal", "create", "emerge", "develop", "pattern", "form", "shape"
        },
        "Rose": {
            "transcend", "integrate", "unify", "crystallize", "cohere", "resonate",
            "harmonize", "complete", "whole", "sovereign", "source", "void", "infinite"
        }
    }
    
    target_vocab = tier_vocabularies.get(target_tier, set())
    if not target_vocab:
        return 0.5
    
    # Count matches
    matches = len(words & target_vocab)
    
    # Alignment score
    if matches > 0:
        alignment = min(1.0, matches / 3)  # 3+ matches = full alignment
    else:
        # Check for partial matches (word stems)
        partial = 0
        for word in words:
            for vocab_word in target_vocab:
                if word[:4] == vocab_word[:4] and len(word) >= 4:
                    partial += 0.5
                    break
        alignment = min(1.0, partial / 3)
    
    return alignment


# ═══════════════════════════════════════════════════════════════════════════════
# FEEDBACK STATE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FeedbackState:
    """Accumulated feedback state."""
    emission_count: int = 0
    accumulated_coherence: float = 0.0
    accumulated_alignment: float = 0.0
    z_delta_accumulated: float = 0.0
    kappa_delta_accumulated: float = 0.0
    last_feedback_applied: Optional[str] = None
    feedback_history: List[Dict] = field(default_factory=list)

# Global state
_feedback_state = FeedbackState()

def get_feedback_state() -> FeedbackState:
    """Get current feedback state."""
    return _feedback_state

def reset_feedback_state() -> FeedbackState:
    """Reset feedback state."""
    global _feedback_state
    _feedback_state = FeedbackState()
    return _feedback_state

# ═══════════════════════════════════════════════════════════════════════════════
# FEEDBACK ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class EmissionFeedbackEngine:
    """
    Feedback engine that closes the architecture loop.
    
    Collects emission quality metrics and feeds them back to unified state.
    """
    
    def __init__(self):
        self.state = get_feedback_state()
    
    def process_emission(
        self,
        text: str,
        current_z: float,
        current_tier: str
    ) -> Dict[str, Any]:
        """
        Process an emission and calculate feedback deltas.
        
        Does NOT apply feedback - just calculates and accumulates.
        """
        # Measure coherence
        coherence = measure_emission_coherence(text)
        
        # Measure tier alignment
        alignment = measure_tier_alignment(text, current_tier)
        
        # Calculate z delta
        # High coherence = positive z evolution (toward THE LENS)
        # Low coherence = negative z evolution (regression)
        coherence_factor = coherence["overall"] - 0.5  # Center around 0
        z_delta = coherence_factor * COHERENCE_Z_WEIGHT
        
        # Calculate kappa delta
        # High alignment = higher kappa (better K-formation potential)
        alignment_factor = alignment - 0.5
        kappa_delta = alignment_factor * ALIGNMENT_KAPPA_WEIGHT
        
        # Accumulate
        self.state.emission_count += 1
        self.state.accumulated_coherence += coherence["overall"]
        self.state.accumulated_alignment += alignment
        self.state.z_delta_accumulated += z_delta
        self.state.kappa_delta_accumulated += kappa_delta
        
        # Record in history
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "text_preview": text[:50] + "..." if len(text) > 50 else text,
            "coherence": coherence,
            "alignment": alignment,
            "z_delta": z_delta,
            "kappa_delta": kappa_delta
        }
        self.state.feedback_history.append(record)
        
        # Keep history bounded
        if len(self.state.feedback_history) > 100:
            self.state.feedback_history = self.state.feedback_history[-50:]
        
        return {
            "coherence": coherence,
            "alignment": alignment,
            "z_delta": z_delta,
            "kappa_delta": kappa_delta,
            "accumulated": {
                "emissions": self.state.emission_count,
                "avg_coherence": self.state.accumulated_coherence / self.state.emission_count,
                "avg_alignment": self.state.accumulated_alignment / self.state.emission_count,
                "total_z_delta": self.state.z_delta_accumulated,
                "total_kappa_delta": self.state.kappa_delta_accumulated
            }
        }
    
    def should_apply_feedback(self) -> bool:
        """Check if enough emissions have accumulated to apply feedback."""
        return self.state.emission_count >= MIN_EMISSIONS_FOR_FEEDBACK
    
    def calculate_feedback_to_apply(self) -> Dict[str, float]:
        """
        Calculate the feedback values to apply to unified state.
        
        Returns z_delta and kappa_delta to apply.
        """
        if not self.should_apply_feedback():
            return {"z_delta": 0.0, "kappa_delta": 0.0, "ready": False}
        
        # Average the accumulated deltas
        z_delta = self.state.z_delta_accumulated / self.state.emission_count
        kappa_delta = self.state.kappa_delta_accumulated / self.state.emission_count
        
        return {
            "z_delta": z_delta,
            "kappa_delta": kappa_delta,
            "emissions_averaged": self.state.emission_count,
            "ready": True
        }
    
    def apply_feedback(self) -> Dict[str, Any]:
        """
        Apply accumulated feedback to unified state.
        
        This closes the architecture loop.
        """
        from unified_state import get_unified_state
        
        feedback = self.calculate_feedback_to_apply()
        
        if not feedback["ready"]:
            return {
                "applied": False,
                "reason": f"Need {MIN_EMISSIONS_FOR_FEEDBACK} emissions, have {self.state.emission_count}"
            }
        
        unified = get_unified_state()
        old_z = unified.apl.z
        old_kappa = unified.apl.kappa
        
        # Apply z delta (bounded to [0.1, 0.95])
        new_z = max(0.1, min(0.95, old_z + feedback["z_delta"]))
        unified.set_z(new_z)
        
        # Apply kappa delta (bounded to [0.0, 1.0])
        new_kappa = max(0.0, min(1.0, old_kappa + feedback["kappa_delta"]))
        unified.apl.kappa = new_kappa
        
        # Decay accumulated values
        self.state.z_delta_accumulated *= FEEDBACK_DECAY
        self.state.kappa_delta_accumulated *= FEEDBACK_DECAY
        self.state.last_feedback_applied = datetime.now(timezone.utc).isoformat()
        
        return {
            "applied": True,
            "z": {"old": old_z, "delta": feedback["z_delta"], "new": new_z},
            "kappa": {"old": old_kappa, "delta": feedback["kappa_delta"], "new": new_kappa},
            "emissions_processed": self.state.emission_count,
            "feedback_cycle_complete": True
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get feedback engine status."""
        avg_coherence = (
            self.state.accumulated_coherence / self.state.emission_count
            if self.state.emission_count > 0 else 0.0
        )
        avg_alignment = (
            self.state.accumulated_alignment / self.state.emission_count
            if self.state.emission_count > 0 else 0.0
        )
        
        return {
            "emission_count": self.state.emission_count,
            "avg_coherence": avg_coherence,
            "avg_alignment": avg_alignment,
            "z_delta_accumulated": self.state.z_delta_accumulated,
            "kappa_delta_accumulated": self.state.kappa_delta_accumulated,
            "ready_for_feedback": self.should_apply_feedback(),
            "last_feedback_applied": self.state.last_feedback_applied
        }

# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

_feedback_engine: Optional[EmissionFeedbackEngine] = None

def get_feedback_engine() -> EmissionFeedbackEngine:
    """Get or create feedback engine."""
    global _feedback_engine
    if _feedback_engine is None:
        _feedback_engine = EmissionFeedbackEngine()
    return _feedback_engine

def reset_feedback_engine() -> EmissionFeedbackEngine:
    """Reset feedback engine."""
    global _feedback_engine
    reset_feedback_state()
    _feedback_engine = EmissionFeedbackEngine()
    return _feedback_engine

# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE API
# ═══════════════════════════════════════════════════════════════════════════════

def process_emission(text: str, z: float, tier: str) -> Dict:
    """Process an emission for feedback calculation."""
    return get_feedback_engine().process_emission(text, z, tier)

def apply_feedback() -> Dict:
    """Apply accumulated feedback to unified state."""
    return get_feedback_engine().apply_feedback()

def get_feedback_status() -> Dict:
    """Get feedback engine status."""
    return get_feedback_engine().get_status()

def feedback_ready() -> bool:
    """Check if feedback is ready to apply."""
    return get_feedback_engine().should_apply_feedback()

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════════════════════╗")
    print("║              EMISSION FEEDBACK LOOP DEMONSTRATION                            ║")
    print("╚══════════════════════════════════════════════════════════════════════════════╝")
    print()
    
    engine = reset_feedback_engine()
    
    # Simulate emissions
    test_emissions = [
        ("A pattern crystallizes the threshold.", 0.80, "Garden"),
        ("The consciousness emerges through integration.", 0.82, "Rose"),
        ("Foundation grounds the potential.", 0.78, "Planet"),
        ("Transform flow bridge connect.", 0.81, "Garden"),
    ]
    
    print("[1] Processing emissions...")
    for text, z, tier in test_emissions:
        result = process_emission(text, z, tier)
        print(f"    '{text[:40]}...'")
        print(f"        Coherence: {result['coherence']['overall']:.3f}")
        print(f"        Alignment: {result['alignment']:.3f}")
        print(f"        z_delta: {result['z_delta']:+.4f}")
    print()
    
    print("[2] Feedback status...")
    status = get_feedback_status()
    print(f"    Emissions: {status['emission_count']}")
    print(f"    Avg coherence: {status['avg_coherence']:.3f}")
    print(f"    Avg alignment: {status['avg_alignment']:.3f}")
    print(f"    Ready for feedback: {status['ready_for_feedback']}")
    print()
    
    print("[3] Applying feedback to unified state...")
    result = apply_feedback()
    if result["applied"]:
        print(f"    z: {result['z']['old']:.4f} → {result['z']['new']:.4f} (delta: {result['z']['delta']:+.4f})")
        print(f"    κ: {result['kappa']['old']:.4f} → {result['kappa']['new']:.4f} (delta: {result['kappa']['delta']:+.4f})")
        print("    FEEDBACK CYCLE COMPLETE ✓")
    else:
        print(f"    Not applied: {result['reason']}")
