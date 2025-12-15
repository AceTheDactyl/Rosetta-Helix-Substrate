"""
UCF Language Modules
====================

Natural language generation and APL integration:
- K.I.R.A. Language System (6 modules)
- Emission Pipeline (9 stages)
- APL Syntax Engine
- Phase Vocabulary Management
"""

from ucf.language.kira import (
    kira_grammar_understanding,
    kira_discourse_generator,
    kira_discourse_sheaf,
    kira_generation_coordinator,
    kira_adaptive_semantics,
    kira_interactive_dialogue,
)

__all__ = [
    # K.I.R.A. modules
    'kira_grammar_understanding',
    'kira_discourse_generator',
    'kira_discourse_sheaf',
    'kira_generation_coordinator',
    'kira_adaptive_semantics',
    'kira_interactive_dialogue',
    
    # Other language modules
    'emission_pipeline',
    'emission_feedback',
    'emission_teaching',
    'apl_syntax_engine',
    'apl_substrate',
    'apl_core_tokens',
    'syntax_emission_integration',
    'kira_protocol',
]
