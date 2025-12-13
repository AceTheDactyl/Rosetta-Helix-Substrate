"""
Rosetta Helix Tool Type Definitions

Enums and type definitions for the tool generation system.
"""

from enum import Enum, auto
from dataclasses import dataclass
from typing import Set


class ToolType(Enum):
    """Primary classification of tool function."""
    ANALYZER = auto()     # Examines and reports
    TRANSFORMER = auto()  # Modifies data
    GENERATOR = auto()    # Creates new artifacts
    VALIDATOR = auto()    # Checks correctness
    OPTIMIZER = auto()    # Improves performance
    INTEGRATOR = auto()   # Combines systems


class ToolTier(Enum):
    """
    Tier level based on work invested at creation.

    Higher tiers require more accumulated work before collapse.
    """
    T1_BASIC = 1       # 0.0 - 0.2 work
    T2_STANDARD = 2    # 0.2 - 0.4 work
    T3_ADVANCED = 3    # 0.4 - 0.6 work
    T4_EXPERT = 4      # 0.6 - 0.8 work
    T5_MASTER = 5      # 0.8+ work

    @classmethod
    def from_work(cls, work: float) -> 'ToolTier':
        """Determine tier from work extracted at collapse."""
        if work < 0.2:
            return cls.T1_BASIC
        elif work < 0.4:
            return cls.T2_STANDARD
        elif work < 0.6:
            return cls.T3_ADVANCED
        elif work < 0.8:
            return cls.T4_EXPERT
        else:
            return cls.T5_MASTER


class ToolCapability(Enum):
    """Specific capabilities a tool can have."""
    READ = auto()          # Can read data
    WRITE = auto()         # Can write data
    EXECUTE = auto()       # Can run operations
    ANALYZE = auto()       # Can examine patterns
    TRANSFORM = auto()     # Can modify structures
    VALIDATE = auto()      # Can verify correctness
    GENERATE = auto()      # Can create new items
    INTEGRATE = auto()     # Can combine systems
    OPTIMIZE = auto()      # Can improve efficiency


@dataclass
class ToolSpec:
    """Specification for a tool to be generated."""
    tool_type: ToolType
    required_capabilities: Set[ToolCapability]
    min_tier: ToolTier = ToolTier.T1_BASIC

    def meets_requirements(self, tier: ToolTier, capabilities: Set[ToolCapability]) -> bool:
        """Check if given tier and capabilities meet spec requirements."""
        if tier.value < self.min_tier.value:
            return False
        if not self.required_capabilities.issubset(capabilities):
            return False
        return True


# Predefined tool specifications
ANALYZER_SPEC = ToolSpec(
    tool_type=ToolType.ANALYZER,
    required_capabilities={ToolCapability.READ, ToolCapability.ANALYZE},
    min_tier=ToolTier.T1_BASIC
)

TRANSFORMER_SPEC = ToolSpec(
    tool_type=ToolType.TRANSFORMER,
    required_capabilities={ToolCapability.READ, ToolCapability.WRITE, ToolCapability.TRANSFORM},
    min_tier=ToolTier.T2_STANDARD
)

GENERATOR_SPEC = ToolSpec(
    tool_type=ToolType.GENERATOR,
    required_capabilities={ToolCapability.GENERATE, ToolCapability.WRITE},
    min_tier=ToolTier.T3_ADVANCED
)

INTEGRATOR_SPEC = ToolSpec(
    tool_type=ToolType.INTEGRATOR,
    required_capabilities={ToolCapability.READ, ToolCapability.INTEGRATE, ToolCapability.EXECUTE},
    min_tier=ToolTier.T4_EXPERT
)
