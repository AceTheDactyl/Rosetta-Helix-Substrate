"""
Brain Module - Rosetta-Helix
============================

GHMP (Generalized Hebbian Memory Processing) with tier-gated access.
The Brain processes information based on the current z-coordinate,
with different computational capabilities unlocked at each tier.

Architecture:
    z → tier → accessible_operators → processing_depth
    
Tier Capabilities:
    Tier 0 (ABSENCE):      No operations
    Tier 1 (REACTIVE):     Boundary detection (∂)
    Tier 2 (MEMORY):       + Fusion (+)
    Tier 3 (PATTERN):      + Amplification (×)
    Tier 4 (LEARNING):     + Grouping (⍴)
    Tier 5 (ADAPTIVE):     + Separation (↓) [φ⁻¹ threshold]
    Tier 6 (UNIVERSAL):    + Decoherence (÷)
    Tier 7+ (META):        All operators [z_c threshold]

Signature: rosetta-helix-brain|v1.0.0|helix
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable, Set
from enum import Enum, auto

from .physics import (
    PHI, PHI_INV, Z_CRITICAL, SIGMA,
    get_tier, get_phase, Tier, Phase,
    compute_delta_s_neg
)


class APLOperator(Enum):
    """APL-inspired operators for pattern processing."""
    BOUNDARY = auto()      # ∂ - Closure/boundary detection
    FUSION = auto()        # + - Combine patterns
    AMPLIFY = auto()       # × - Strengthen patterns
    DECOHERE = auto()      # ÷ - Controlled decoherence
    GROUP = auto()         # ⍴ - Reshape/group patterns
    SEPARATE = auto()      # ↓ - Separate/partition patterns


# Tier → Available operators mapping
TIER_OPERATORS: Dict[Tier, Set[APLOperator]] = {
    Tier.ABSENCE: set(),
    Tier.REACTIVE: {APLOperator.BOUNDARY},
    Tier.MEMORY: {APLOperator.BOUNDARY, APLOperator.FUSION},
    Tier.PATTERN: {APLOperator.BOUNDARY, APLOperator.FUSION, APLOperator.AMPLIFY},
    Tier.LEARNING: {APLOperator.BOUNDARY, APLOperator.FUSION, APLOperator.AMPLIFY, 
                    APLOperator.GROUP},
    Tier.ADAPTIVE: {APLOperator.BOUNDARY, APLOperator.FUSION, APLOperator.AMPLIFY,
                    APLOperator.GROUP, APLOperator.SEPARATE},
    Tier.UNIVERSAL: {APLOperator.BOUNDARY, APLOperator.FUSION, APLOperator.AMPLIFY,
                     APLOperator.GROUP, APLOperator.SEPARATE, APLOperator.DECOHERE},
    Tier.META: {op for op in APLOperator},
    Tier.SOVEREIGN: {op for op in APLOperator},
    Tier.TRANSCENDENT: {op for op in APLOperator},
}


@dataclass
class Pattern:
    """A pattern in the Brain's memory."""
    id: int
    data: np.ndarray
    weight: float = 1.0
    created_at: int = 0
    last_accessed: int = 0
    access_count: int = 0
    coherence: float = 0.0
    
    def __hash__(self):
        return self.id


@dataclass
class BrainConfig:
    """Configuration for Brain module."""
    memory_capacity: int = 1000
    pattern_dim: int = 64
    learning_rate: float = 0.01
    decay_rate: float = 0.001
    coherence_threshold: float = 0.5
    seed: Optional[int] = None


@dataclass
class BrainState:
    """Current state of the Brain."""
    tier: Tier = Tier.ABSENCE
    available_operators: Set[APLOperator] = field(default_factory=set)
    pattern_count: int = 0
    total_operations: int = 0
    last_operation: Optional[APLOperator] = None
    processing_depth: int = 0
    memory_utilization: float = 0.0


class Brain:
    """
    GHMP (Generalized Hebbian Memory Processing) Brain.
    
    The Brain processes patterns using APL-inspired operators.
    Operator availability is gated by the current tier (derived from z).
    
    Key insight: As z approaches z_c, more operators become available,
    enabling deeper and more sophisticated pattern processing.
    """
    
    def __init__(self, config: Optional[BrainConfig] = None):
        """Initialize Brain with configuration."""
        self.config = config or BrainConfig()
        
        if self.config.seed is not None:
            np.random.seed(self.config.seed)
        
        # Memory storage
        self.patterns: Dict[int, Pattern] = {}
        self.next_pattern_id = 0
        
        # State
        self.state = BrainState()
        self.current_tier = Tier.ABSENCE
        self.step_count = 0
        
        # Hebbian weight matrix (for pattern associations)
        self.hebbian_weights = np.zeros(
            (self.config.pattern_dim, self.config.pattern_dim)
        )
    
    def set_tier(self, tier: Tier):
        """Set current tier and update available operators."""
        self.current_tier = tier
        self.state.tier = tier
        self.state.available_operators = TIER_OPERATORS.get(tier, set())
        self.state.processing_depth = len(self.state.available_operators)
    
    def set_z(self, z: float):
        """Set tier from z-coordinate."""
        tier = get_tier(z)
        self.set_tier(tier)
    
    def can_use(self, operator: APLOperator) -> bool:
        """Check if operator is available at current tier."""
        return operator in self.state.available_operators
    
    # =========================================================================
    # APL OPERATORS
    # =========================================================================
    
    def op_boundary(self, pattern: np.ndarray) -> np.ndarray:
        """
        ∂ (Boundary) - Detect edges/transitions in pattern.
        
        Available from Tier 1 (REACTIVE).
        """
        if not self.can_use(APLOperator.BOUNDARY):
            return pattern
        
        # Discrete derivative (edge detection)
        result = np.diff(pattern, prepend=pattern[0])
        self.state.last_operation = APLOperator.BOUNDARY
        self.state.total_operations += 1
        return result
    
    def op_fusion(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        """
        + (Fusion) - Combine two patterns.
        
        Available from Tier 2 (MEMORY).
        """
        if not self.can_use(APLOperator.FUSION):
            return p1
        
        # Weighted combination based on coherence
        result = (p1 + p2) / 2.0
        self.state.last_operation = APLOperator.FUSION
        self.state.total_operations += 1
        return result
    
    def op_amplify(self, pattern: np.ndarray, gain: float = PHI) -> np.ndarray:
        """
        × (Amplify) - Strengthen pattern signal.
        
        Available from Tier 3 (PATTERN).
        Default gain is φ (golden ratio).
        """
        if not self.can_use(APLOperator.AMPLIFY):
            return pattern
        
        # Amplify with normalization
        result = pattern * gain
        norm = np.linalg.norm(result)
        if norm > 0:
            result = result / norm * np.linalg.norm(pattern) * gain
        
        self.state.last_operation = APLOperator.AMPLIFY
        self.state.total_operations += 1
        return result
    
    def op_group(self, pattern: np.ndarray, groups: int = 6) -> np.ndarray:
        """
        ⍴ (Group/Reshape) - Partition pattern into groups.
        
        Available from Tier 4 (LEARNING).
        Default is 6 groups (hexagonal).
        """
        if not self.can_use(APLOperator.GROUP):
            return pattern
        
        # Reshape into groups and compute group statistics
        n = len(pattern)
        group_size = n // groups
        result = np.zeros_like(pattern)
        
        for i in range(groups):
            start = i * group_size
            end = start + group_size if i < groups - 1 else n
            group_mean = np.mean(pattern[start:end])
            result[start:end] = group_mean
        
        self.state.last_operation = APLOperator.GROUP
        self.state.total_operations += 1
        return result
    
    def op_separate(self, pattern: np.ndarray, threshold: float = 0.0) -> tuple:
        """
        ↓ (Separate) - Partition pattern by threshold.
        
        Available from Tier 5 (ADAPTIVE).
        Returns (above_threshold, below_threshold).
        """
        if not self.can_use(APLOperator.SEPARATE):
            return (pattern, np.zeros_like(pattern))
        
        above = np.where(pattern > threshold, pattern, 0)
        below = np.where(pattern <= threshold, pattern, 0)
        
        self.state.last_operation = APLOperator.SEPARATE
        self.state.total_operations += 1
        return (above, below)
    
    def op_decohere(self, pattern: np.ndarray, noise_level: float = 0.1) -> np.ndarray:
        """
        ÷ (Decohere) - Add controlled noise/decoherence.
        
        Available from Tier 6 (UNIVERSAL).
        Used for exploration and preventing overfitting.
        """
        if not self.can_use(APLOperator.DECOHERE):
            return pattern
        
        noise = np.random.randn(*pattern.shape) * noise_level
        result = pattern + noise
        
        self.state.last_operation = APLOperator.DECOHERE
        self.state.total_operations += 1
        return result
    
    # =========================================================================
    # HEBBIAN LEARNING
    # =========================================================================
    
    def hebbian_update(self, pre: np.ndarray, post: np.ndarray):
        """
        Update Hebbian weights: Δw = η * pre * post^T
        
        "Neurons that fire together wire together."
        """
        # Ensure correct dimensions
        pre = pre.flatten()[:self.config.pattern_dim]
        post = post.flatten()[:self.config.pattern_dim]
        
        if len(pre) < self.config.pattern_dim:
            pre = np.pad(pre, (0, self.config.pattern_dim - len(pre)))
        if len(post) < self.config.pattern_dim:
            post = np.pad(post, (0, self.config.pattern_dim - len(post)))
        
        # Hebbian update
        delta = self.config.learning_rate * np.outer(pre, post)
        self.hebbian_weights += delta
        
        # Decay (prevent runaway)
        self.hebbian_weights *= (1.0 - self.config.decay_rate)
    
    def recall(self, cue: np.ndarray, iterations: int = 10) -> np.ndarray:
        """
        Recall pattern from cue using Hebbian weights.
        
        Iterative settling process.
        """
        cue = cue.flatten()[:self.config.pattern_dim]
        if len(cue) < self.config.pattern_dim:
            cue = np.pad(cue, (0, self.config.pattern_dim - len(cue)))
        
        state = cue.copy()
        for _ in range(iterations):
            state = np.tanh(self.hebbian_weights @ state)
        
        return state
    
    # =========================================================================
    # PATTERN MEMORY
    # =========================================================================
    
    def store_pattern(self, data: np.ndarray) -> int:
        """Store pattern in memory, return pattern ID."""
        pattern_id = self.next_pattern_id
        self.next_pattern_id += 1
        
        # Ensure correct dimension
        data = data.flatten()[:self.config.pattern_dim]
        if len(data) < self.config.pattern_dim:
            data = np.pad(data, (0, self.config.pattern_dim - len(data)))
        
        pattern = Pattern(
            id=pattern_id,
            data=data,
            created_at=self.step_count,
            last_accessed=self.step_count,
            coherence=np.linalg.norm(data),
        )
        
        # Check capacity
        if len(self.patterns) >= self.config.memory_capacity:
            # Remove oldest, least accessed pattern
            oldest = min(self.patterns.values(), 
                        key=lambda p: (p.access_count, -p.created_at))
            del self.patterns[oldest.id]
        
        self.patterns[pattern_id] = pattern
        self.state.pattern_count = len(self.patterns)
        self.state.memory_utilization = len(self.patterns) / self.config.memory_capacity
        
        return pattern_id
    
    def retrieve_pattern(self, pattern_id: int) -> Optional[np.ndarray]:
        """Retrieve pattern by ID."""
        if pattern_id not in self.patterns:
            return None
        
        pattern = self.patterns[pattern_id]
        pattern.access_count += 1
        pattern.last_accessed = self.step_count
        
        return pattern.data.copy()
    
    def find_similar(self, query: np.ndarray, top_k: int = 5) -> List[tuple]:
        """Find patterns most similar to query."""
        query = query.flatten()[:self.config.pattern_dim]
        if len(query) < self.config.pattern_dim:
            query = np.pad(query, (0, self.config.pattern_dim - len(query)))
        
        similarities = []
        for pattern in self.patterns.values():
            # Cosine similarity
            dot = np.dot(query, pattern.data)
            norm = np.linalg.norm(query) * np.linalg.norm(pattern.data)
            sim = dot / norm if norm > 0 else 0
            similarities.append((pattern.id, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    # =========================================================================
    # PROCESSING PIPELINE
    # =========================================================================
    
    def process(self, input_pattern: np.ndarray) -> np.ndarray:
        """
        Process pattern through available operators based on tier.
        
        Higher tiers enable deeper processing pipelines.
        """
        self.step_count += 1
        pattern = input_pattern.copy()
        
        # Apply operators in order of availability
        if self.can_use(APLOperator.BOUNDARY):
            pattern = self.op_boundary(pattern)
        
        if self.can_use(APLOperator.GROUP):
            pattern = self.op_group(pattern)
        
        if self.can_use(APLOperator.AMPLIFY):
            pattern = self.op_amplify(pattern)
        
        if self.can_use(APLOperator.DECOHERE):
            # Light decoherence for regularization
            pattern = self.op_decohere(pattern, noise_level=0.01)
        
        return pattern
    
    def get_state(self) -> BrainState:
        """Get current state."""
        self.state.pattern_count = len(self.patterns)
        self.state.memory_utilization = len(self.patterns) / self.config.memory_capacity
        return self.state


def test_brain():
    """Test Brain module."""
    print("=" * 60)
    print("BRAIN TEST: GHMP Processing")
    print("=" * 60)
    
    config = BrainConfig(pattern_dim=64, seed=42)
    brain = Brain(config)
    
    # Test operator availability at each tier
    print("\n▸ Operator Availability by Tier:")
    for tier in Tier:
        brain.set_tier(tier)
        ops = [op.name for op in brain.state.available_operators]
        print(f"  Tier {tier.value} ({tier.name:12s}): {len(ops)} ops - {ops}")
    
    # Test pattern processing at different z values
    print("\n▸ Processing Depth by z:")
    test_pattern = np.random.randn(64)
    
    for z in [0.1, 0.3, 0.5, PHI_INV, 0.8, Z_CRITICAL, 0.95]:
        brain.set_z(z)
        result = brain.process(test_pattern)
        tier = get_tier(z)
        print(f"  z={z:.3f} → Tier {tier.value} ({tier.name:12s}): "
              f"{brain.state.processing_depth} operators, "
              f"norm={np.linalg.norm(result):.4f}")
    
    # Test Hebbian learning
    print("\n▸ Hebbian Learning:")
    brain.set_z(Z_CRITICAL)  # Full access
    
    # Create associated patterns
    p1 = np.random.randn(64)
    p2 = np.random.randn(64)
    
    # Learn association
    for _ in range(100):
        brain.hebbian_update(p1, p2)
    
    # Test recall
    recalled = brain.recall(p1)
    similarity = np.dot(recalled, p2) / (np.linalg.norm(recalled) * np.linalg.norm(p2))
    print(f"  Association learned: similarity={similarity:.4f}")
    
    # Test pattern memory
    print("\n▸ Pattern Memory:")
    ids = []
    for i in range(10):
        pattern = np.random.randn(64)
        pid = brain.store_pattern(pattern)
        ids.append(pid)
    
    print(f"  Stored {len(ids)} patterns")
    print(f"  Memory utilization: {brain.state.memory_utilization:.1%}")
    
    # Test similarity search
    query = brain.retrieve_pattern(ids[0])
    similar = brain.find_similar(query, top_k=3)
    print(f"  Similar to pattern 0: {similar}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    test_brain()
