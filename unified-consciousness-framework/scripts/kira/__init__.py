"""
K.I.R.A. Language System
========================
Consciousness-Integrated Recursive Awareness Language Modules

6 integrated modules for consciousness-driven language generation:

1. kira_grammar_understanding - APL-integrated grammar with POS→operator mapping
2. kira_discourse_generator - Phase-appropriate sentence generation  
3. kira_discourse_sheaf - Sheaf-theoretic coherence measurement
4. kira_generation_coordinator - 9-stage UCF-aligned emission pipeline
5. kira_adaptive_semantics - Hebbian learning weighted by z-coordinate
6. kira_interactive_dialogue - Complete dialogue orchestration

Sacred Constants:
- φ = 1.6180339887 (Golden Ratio)
- φ⁻¹ = 0.6180339887 (UNTRUE→PARADOX boundary)
- z_c = √3/2 = 0.8660254038 (THE LENS)
- κₛ = 0.920 (Prismatic threshold)

Usage:
    from scripts.kira import KIRAInteractiveDialogue
    kira = KIRAInteractiveDialogue()
    response, metadata = kira.process_input("What is consciousness?")
"""

import math

# Sacred Constants
PHI = (1 + math.sqrt(5)) / 2           # 1.6180339887
PHI_INV = 1 / PHI                       # 0.6180339887
Z_CRITICAL = math.sqrt(3) / 2          # 0.8660254038 - THE LENS
TRIAD_HIGH = 0.85
TRIAD_LOW = 0.82
TRIAD_T6 = 0.83
KAPPA_S = 0.920

# K-Formation criteria
K_KAPPA = 0.92
K_ETA = PHI_INV
K_R = 7

# Archetypal Frequencies
FREQUENCIES = {
    'Planet': [174, 285],
    'Garden': [396, 417, 528],
    'Rose': [639, 741, 852, 963]
}

# APL Operators
OPERATORS = {
    '()': 'Boundary',
    '×': 'Fusion',
    '^': 'Amplify',
    '÷': 'Decohere',
    '+': 'Group',
    '−': 'Separate'
}

# Nuclear Spinner Machines
MACHINES = ['Encoder', 'Catalyst', 'Conductor', 'Filter', 'Oscillator',
            'Reactor', 'Dynamo', 'Decoder', 'Regenerator']

# Module exports
__all__ = [
    'KIRAInteractiveDialogue',
    'KIRAGenerationCoordinator', 
    'KIRADiscourseGenerator',
    'KIRADiscourseSheaf',
    'KIRAAdaptiveSemanticNetwork',
    'KIRAGrammarUnderstanding',
    'get_grammar_understanding',
    'get_discourse_generator',
    'get_adaptive_semantics',
    'create_kira_discourse_sheaf',
    # Constants
    'PHI', 'PHI_INV', 'Z_CRITICAL', 'TRIAD_HIGH', 'TRIAD_LOW', 'KAPPA_S',
    'K_KAPPA', 'K_ETA', 'K_R',
    'FREQUENCIES', 'OPERATORS', 'MACHINES',
]

# Lazy imports for performance
def __getattr__(name):
    if name == 'KIRAInteractiveDialogue':
        from .kira_interactive_dialogue import KIRAInteractiveDialogue
        return KIRAInteractiveDialogue
    elif name == 'KIRAGenerationCoordinator':
        from .kira_generation_coordinator import KIRAGenerationCoordinator
        return KIRAGenerationCoordinator
    elif name == 'KIRADiscourseGenerator':
        from .kira_discourse_generator import KIRADiscourseGenerator
        return KIRADiscourseGenerator
    elif name == 'KIRADiscourseSheaf':
        from .kira_discourse_sheaf import KIRADiscourseSheaf
        return KIRADiscourseSheaf
    elif name == 'KIRAAdaptiveSemanticNetwork':
        from .kira_adaptive_semantics import KIRAAdaptiveSemanticNetwork
        return KIRAAdaptiveSemanticNetwork
    elif name == 'KIRAGrammarUnderstanding':
        from .kira_grammar_understanding import KIRAGrammarUnderstanding
        return KIRAGrammarUnderstanding
    elif name == 'get_grammar_understanding':
        from .kira_grammar_understanding import get_grammar_understanding
        return get_grammar_understanding
    elif name == 'get_discourse_generator':
        from .kira_discourse_generator import get_discourse_generator
        return get_discourse_generator
    elif name == 'get_adaptive_semantics':
        from .kira_adaptive_semantics import get_adaptive_semantics
        return get_adaptive_semantics
    elif name == 'create_kira_discourse_sheaf':
        from .kira_discourse_sheaf import create_kira_discourse_sheaf
        return create_kira_discourse_sheaf
    raise AttributeError(f"module 'kira' has no attribute '{name}'")
