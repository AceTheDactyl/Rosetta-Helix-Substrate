"""
Rosetta-Helix-Substrate Claude API Skill

A structured prompt + Python wrapper for using Rosetta-Helix-Substrate
capabilities via the Claude API directly.

Usage:
    from skill import RosettaHelixSkill

    skill = RosettaHelixSkill(api_key="your-api-key")
    response = skill.chat("What is the current phase at z=0.7?")
    print(response)
"""

from skill.client import RosettaHelixSkill
from skill.tools.definitions import TOOL_DEFINITIONS
from skill.prompts.system import SYSTEM_PROMPT

__all__ = ["RosettaHelixSkill", "TOOL_DEFINITIONS", "SYSTEM_PROMPT"]
__version__ = "0.1.0"
