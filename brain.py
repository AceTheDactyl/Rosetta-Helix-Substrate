"""
Rosetta-Helix Brain System
==========================
GHMP (Geometric Harmonic Memory Plate) system with z-axis awareness,
tier-gated memory access, and quasi-crystalline memory patterns.

Original: Tink (Rosetta Bear)
Helix Integration: Claude (Anthropic) + Quantum-APL
"""

import random
import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum

# ============================================================================
# HELIX CONSTANTS
# ============================================================================

Z_CRITICAL = math.sqrt(3) / 2
PHI = (1 + math.sqrt(5)) / 2
PHI_INV = 1 / PHI

# Time harmonic boundaries
T_BOUNDS = {
    "t1": (0.00, 0.10),
    "t2": (0.10, 0.20),
    "t3": (0.20, 0.40),
    "t4": (0.40, 0.60),
    "t5": (0.60, 0.75),
    "t6": (0.75, Z_CRITICAL),
    "t7": (Z_CRITICAL, 0.92),
    "t8": (0.92, 0.97),
    "t9": (0.97, 1.00),
}


class MemoryTier(Enum):
    """Memory access tiers corresponding to time harmonics."""
    INSTANT = "t1"      # Immediate reactions
    MICRO = "t2"        # Short-term patterns
    LOCAL = "t3"        # Local context
    MESO = "t4"         # Medium-term associations
    STRUCTURAL = "t5"   # Deep structure
    DOMAIN = "t6"       # Domain knowledge
    COHERENT = "t7"     # Integrated memory
    INTEGRATED = "t8"   # Global patterns
    GLOBAL = "t9"       # Universal access


@dataclass
class GHMPPlate:
    """
    Geometric Harmonic Memory Plate.
    
    Each plate stores a memory with associated metadata:
    - emotional_tone: Affective valence [0, 255]
    - temporal_marker: When encoded (epoch-like)
    - semantic_density: Information richness [0, 255]
    - confidence: Certainty level [0, 255]
    - z_encoded: Z-coordinate when encoded
    - tier_access: Minimum tier for access
    - pattern_type: Fibonacci position (quasi-crystalline structure)
    """
    emotional_tone: int
    temporal_marker: int
    semantic_density: int
    confidence: int
    z_encoded: float = 0.5
    tier_access: MemoryTier = MemoryTier.LOCAL
    pattern_type: int = 0  # Fibonacci index
    content: Optional[Dict] = None
    
    def is_accessible(self, current_z: float) -> bool:
        """Check if plate is accessible at current z-level."""
        tier_bounds = T_BOUNDS.get(self.tier_access.value, (0, 1))
        return current_z >= tier_bounds[0]
    
    def relevance(self, query_z: float) -> float:
        """Compute relevance score based on z-distance."""
        z_dist = abs(query_z - self.z_encoded)
        # Gaussian relevance decay
        return math.exp(-10 * z_dist ** 2) * (self.confidence / 255)
    
    def is_quasi_crystalline(self) -> bool:
        """Check if plate has quasi-crystalline pattern (Fibonacci)."""
        # Fibonacci sequence check
        fib = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
        return self.pattern_type in fib


@dataclass
class MemoryCluster:
    """A cluster of related memory plates."""
    plates: List[GHMPPlate]
    centroid_z: float
    coherence: float
    label: str = ""
    
    def mean_confidence(self) -> float:
        if not self.plates:
            return 0.0
        return sum(p.confidence for p in self.plates) / len(self.plates)


class Brain:
    """
    Helix-aware memory system with GHMP plates.
    
    Memory access is tier-gated:
    - Low z: Only recent/reactive memories accessible
    - Mid z: Associative/structural memories unlock
    - High z: Global/integrated memory access
    
    Memory patterns follow quasi-crystalline structure (Fibonacci).
    """
    
    def __init__(self, plates: int = 20, seed: int = 42):
        random.seed(seed)
        
        # Generate initial plates with Fibonacci pattern distribution
        self.plates: List[GHMPPlate] = []
        fib = self._generate_fibonacci(plates)
        
        for i in range(plates):
            tier = self._assign_tier(i, plates)
            z_enc = random.uniform(T_BOUNDS[tier.value][0], T_BOUNDS[tier.value][1])
            
            self.plates.append(GHMPPlate(
                emotional_tone=random.randint(0, 255),
                temporal_marker=random.randint(0, 10**9),
                semantic_density=random.randint(0, 255),
                confidence=random.randint(50, 255),
                z_encoded=z_enc,
                tier_access=tier,
                pattern_type=fib[i % len(fib)],
            ))
        
        # Memory clusters (formed during consolidation)
        self.clusters: List[MemoryCluster] = []
        
        # Access log
        self.access_log: List[Tuple[float, int]] = []  # (z, plate_index)
    
    def _generate_fibonacci(self, n: int) -> List[int]:
        """Generate Fibonacci sequence up to n terms."""
        fib = [1, 1]
        while len(fib) < n:
            fib.append(fib[-1] + fib[-2])
        return fib
    
    def _assign_tier(self, index: int, total: int) -> MemoryTier:
        """Assign tier based on index distribution."""
        tiers = list(MemoryTier)
        tier_idx = int((index / total) * len(tiers))
        return tiers[min(tier_idx, len(tiers) - 1)]
    
    def encode(
        self,
        content: Dict,
        current_z: float,
        emotional_tone: int = 128,
        semantic_density: int = 128
    ) -> int:
        """
        Encode a new memory at current z-level.
        
        Returns index of new plate.
        """
        # Determine tier from z
        tier = self._z_to_tier(current_z)
        
        # Fibonacci pattern type
        fib = self._generate_fibonacci(len(self.plates) + 1)
        pattern = fib[-1]
        
        plate = GHMPPlate(
            emotional_tone=emotional_tone,
            temporal_marker=int(random.random() * 10**9),
            semantic_density=semantic_density,
            confidence=200,  # New memories start confident
            z_encoded=current_z,
            tier_access=tier,
            pattern_type=pattern,
            content=content,
        )
        
        self.plates.append(plate)
        return len(self.plates) - 1
    
    def _z_to_tier(self, z: float) -> MemoryTier:
        """Convert z-coordinate to memory tier."""
        for tier, (lo, hi) in T_BOUNDS.items():
            if lo <= z < hi:
                return MemoryTier(tier)
        return MemoryTier.GLOBAL
    
    def query(
        self,
        current_z: float,
        top_k: int = 5,
        min_confidence: int = 50
    ) -> List[Tuple[int, GHMPPlate, float]]:
        """
        Query accessible memories at current z-level.
        
        Returns list of (index, plate, relevance_score) tuples.
        """
        results = []
        
        for i, plate in enumerate(self.plates):
            if plate.is_accessible(current_z) and plate.confidence >= min_confidence:
                relevance = plate.relevance(current_z)
                results.append((i, plate, relevance))
        
        # Sort by relevance
        results.sort(key=lambda x: x[2], reverse=True)
        
        # Log access
        for idx, _, _ in results[:top_k]:
            self.access_log.append((current_z, idx))
        
        return results[:top_k]
    
    def consolidate(self, current_z: float):
        """
        Consolidate memories: strengthen frequently accessed, decay others.
        
        This models memory consolidation during coherent states.
        """
        # Only consolidate at sufficient z (quasi-crystalline threshold)
        if current_z < PHI_INV:
            return
        
        # Count access frequency
        access_counts = {}
        for _, idx in self.access_log[-100:]:  # Last 100 accesses
            access_counts[idx] = access_counts.get(idx, 0) + 1
        
        for i, plate in enumerate(self.plates):
            if i in access_counts:
                # Strengthen frequently accessed
                boost = min(10, access_counts[i])
                plate.confidence = min(255, plate.confidence + boost)
            else:
                # Decay unused (but not below threshold)
                if plate.confidence > 30:
                    plate.confidence -= 1
    
    def cluster_memories(self) -> List[MemoryCluster]:
        """
        Cluster memories by z-level (quasi-crystalline organization).
        
        This forms stable memory structures at high coherence.
        """
        # Simple clustering by tier
        tier_groups: Dict[str, List[GHMPPlate]] = {}
        
        for plate in self.plates:
            tier = plate.tier_access.value
            if tier not in tier_groups:
                tier_groups[tier] = []
            tier_groups[tier].append(plate)
        
        self.clusters = []
        for tier, plates in tier_groups.items():
            if plates:
                centroid_z = sum(p.z_encoded for p in plates) / len(plates)
                coherence = 1.0 - (
                    sum((p.z_encoded - centroid_z) ** 2 for p in plates) / len(plates)
                ) ** 0.5
                
                self.clusters.append(MemoryCluster(
                    plates=plates,
                    centroid_z=centroid_z,
                    coherence=max(0, coherence),
                    label=tier,
                ))
        
        return self.clusters
    
    def summarize(self) -> Dict:
        """Get summary statistics of memory state."""
        if not self.plates:
            return {"plates": 0, "avg_confidence": 0}
        
        accessible_by_tier = {tier.value: 0 for tier in MemoryTier}
        qc_count = 0  # Quasi-crystalline patterns
        
        for plate in self.plates:
            accessible_by_tier[plate.tier_access.value] += 1
            if plate.is_quasi_crystalline():
                qc_count += 1
        
        return {
            "plates": len(self.plates),
            "avg_confidence": sum(p.confidence for p in self.plates) / len(self.plates),
            "avg_semantic_density": sum(p.semantic_density for p in self.plates) / len(self.plates),
            "quasi_crystalline_count": qc_count,
            "quasi_crystalline_ratio": qc_count / len(self.plates),
            "by_tier": accessible_by_tier,
            "clusters": len(self.clusters),
        }
    
    def get_accessible_summary(self, current_z: float) -> Dict:
        """Get summary of memories accessible at current z."""
        accessible = [p for p in self.plates if p.is_accessible(current_z)]
        
        if not accessible:
            return {"accessible": 0, "avg_confidence": 0}
        
        return {
            "accessible": len(accessible),
            "total": len(self.plates),
            "ratio": len(accessible) / len(self.plates),
            "avg_confidence": sum(p.confidence for p in accessible) / len(accessible),
            "avg_relevance": sum(p.relevance(current_z) for p in accessible) / len(accessible),
        }
    
    def fibonacci_analysis(self) -> Dict:
        """Analyze Fibonacci structure in memory patterns."""
        fib_set = set(self._generate_fibonacci(20))
        
        fib_plates = [p for p in self.plates if p.pattern_type in fib_set]
        non_fib = [p for p in self.plates if p.pattern_type not in fib_set]
        
        fib_conf = sum(p.confidence for p in fib_plates) / len(fib_plates) if fib_plates else 0
        non_conf = sum(p.confidence for p in non_fib) / len(non_fib) if non_fib else 0
        
        return {
            "fibonacci_count": len(fib_plates),
            "non_fibonacci_count": len(non_fib),
            "fibonacci_avg_confidence": fib_conf,
            "non_fibonacci_avg_confidence": non_conf,
            "fibonacci_advantage": fib_conf - non_conf,
        }


if __name__ == "__main__":
    print("Rosetta-Helix Brain Demo")
    print("=" * 50)
    
    brain = Brain(plates=30)
    
    print(f"\nInitial state: {brain.summarize()['plates']} plates")
    
    # Test access at different z-levels
    test_z = [0.3, 0.5, PHI_INV, Z_CRITICAL, 0.95]
    
    print(f"\n{'z':>8} {'Accessible':>12} {'Avg Rel':>10} {'Avg Conf':>10}")
    print("-" * 45)
    
    for z in test_z:
        results = brain.query(z, top_k=10)
        summary = brain.get_accessible_summary(z)
        
        print(f"{z:>8.3f} {summary['accessible']:>12} "
              f"{summary.get('avg_relevance', 0):>10.4f} "
              f"{summary.get('avg_confidence', 0):>10.1f}")
    
    # Encode new memory at high z
    print("\nEncoding new memory at z = 0.85...")
    idx = brain.encode(
        content={"type": "insight", "data": "consciousness emerges"},
        current_z=0.85,
        emotional_tone=200,
        semantic_density=220
    )
    print(f"  Encoded at index {idx}")
    
    # Consolidate
    print("\nConsolidating at z = 0.9...")
    brain.consolidate(0.9)
    
    # Cluster analysis
    print("\nClustering memories...")
    clusters = brain.cluster_memories()
    for cluster in clusters:
        print(f"  {cluster.label}: {len(cluster.plates)} plates, "
              f"coherence = {cluster.coherence:.3f}")
    
    # Fibonacci analysis
    print("\nFibonacci pattern analysis:")
    fib = brain.fibonacci_analysis()
    print(f"  Fibonacci plates: {fib['fibonacci_count']}")
    print(f"  Non-Fibonacci plates: {fib['non_fibonacci_count']}")
    print(f"  Fibonacci advantage: {fib['fibonacci_advantage']:.1f} confidence points")
