"""
Rosetta Helix Tool Generation Framework

Meta-tools that produce child tools using collapse physics.

Architecture:
    MetaTool (uses CollapseEngine)
        │
        ├── pumps work into mini-collapse
        ├── at collapse: extracts work
        └── work converts to ChildTool

CRITICAL: PHI_INV controls all dynamics. PHI only at collapse.
"""

from .tool_types import ToolType, ToolTier, ToolCapability
from .child_tool import ChildTool
from .meta_tool import MetaTool
from .prismatic_tools import PrismaticLayer, PrismaticTool, PrismaticToolGenerator

__all__ = [
    'ToolType',
    'ToolTier',
    'ToolCapability',
    'ChildTool',
    'MetaTool',
    'PrismaticLayer',
    'PrismaticTool',
    'PrismaticToolGenerator',
]
