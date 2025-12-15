#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  LANGUAGE EMISSION PIPELINE                                                   ║
║  9-Stage Processing from Content Selection to Validated Emission              ║
║  Part of the Unified Consciousness Framework                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

Pipeline Structure:
  Stage 1 → Content Selection (ContentWords)
  Stage 2 → Emergence Check (EmergenceResult)
     └─ If bypassed → skip to Stage 5
  Stage 3 → Structural Frame (FrameResult)
  Stage 4 → Slot Assignment (SlottedWords)
  Stage 5 → Function Words (WordSequence)
  Stage 6 → Agreement/Inflection (WordSequence)
  Stage 7 → Connectors (WordSequence)
  Stage 8 → Punctuation (WordSequence)
  Stage 9 → Validation (EmissionResult)

The pipeline transforms internal representations into coherent linguistic output,
respecting the z-coordinate phase regime and APL operator constraints.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum

# Import from sibling modules
from ucf.language.apl_substrate import (
    Z_CRITICAL, PHI_INV, compute_negentropy, classify_phase,
    get_tier, OPERATORS, Direction, Machine, Domain
)

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

# Emergence threshold - when η exceeds this, emergence path is taken
EMERGENCE_THRESHOLD = PHI_INV  # 0.618...

# Minimum coherence for valid emission
MIN_COHERENCE = 0.5

# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

class WordType(Enum):
    """Types of words in the emission pipeline."""
    CONTENT = "content"      # Nouns, verbs, adjectives, adverbs
    FUNCTION = "function"    # Articles, prepositions, auxiliaries
    CONNECTOR = "connector"  # Conjunctions, relative pronouns
    PUNCTUATION = "punct"    # Sentence-level punctuation


class FrameType(Enum):
    """Structural frame types for sentences."""
    DECLARATIVE = "declarative"      # Subject-Verb-Object
    INTERROGATIVE = "interrogative"  # Question forms
    IMPERATIVE = "imperative"        # Commands
    CONDITIONAL = "conditional"      # If-then structures
    RELATIVE = "relative"            # Embedded clauses


class SlotType(Enum):
    """Slot types in structural frames."""
    SUBJECT = "subject"
    VERB = "verb"
    OBJECT = "object"
    MODIFIER = "modifier"
    COMPLEMENT = "complement"
    ADJUNCT = "adjunct"


@dataclass
class Word:
    """A word unit in the pipeline."""
    text: str
    word_type: WordType
    slot: Optional[SlotType] = None
    features: Dict[str, Any] = field(default_factory=dict)
    
    # Agreement features
    person: Optional[int] = None      # 1, 2, 3
    number: Optional[str] = None      # singular, plural
    tense: Optional[str] = None       # present, past, future
    aspect: Optional[str] = None      # simple, progressive, perfect
    
    def __str__(self) -> str:
        return self.text
    
    def with_inflection(self, inflected: str) -> 'Word':
        """Return a copy with inflected form."""
        new_word = Word(
            text=inflected,
            word_type=self.word_type,
            slot=self.slot,
            features=self.features.copy(),
            person=self.person,
            number=self.number,
            tense=self.tense,
            aspect=self.aspect
        )
        return new_word


@dataclass
class ContentWords:
    """Stage 1 output: Selected content words."""
    words: List[Word]
    source_z: float
    source_phase: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def __len__(self) -> int:
        return len(self.words)


@dataclass
class EmergenceResult:
    """Stage 2 output: Emergence check result."""
    emerged: bool
    negentropy: float
    threshold: float = EMERGENCE_THRESHOLD
    bypass_to_stage: Optional[int] = None  # If bypassed, skip to this stage
    emergence_type: Optional[str] = None   # Type of emergence if emerged
    
    @property
    def bypassed(self) -> bool:
        return self.bypass_to_stage is not None


@dataclass
class FrameResult:
    """Stage 3 output: Structural frame."""
    frame_type: FrameType
    slots: List[SlotType]
    constraints: Dict[str, Any] = field(default_factory=dict)
    
    def get_slot_order(self) -> List[SlotType]:
        """Get canonical slot ordering for this frame type."""
        orderings = {
            FrameType.DECLARATIVE: [SlotType.SUBJECT, SlotType.VERB, SlotType.OBJECT],
            FrameType.INTERROGATIVE: [SlotType.VERB, SlotType.SUBJECT, SlotType.OBJECT],
            FrameType.IMPERATIVE: [SlotType.VERB, SlotType.OBJECT],
            FrameType.CONDITIONAL: [SlotType.ADJUNCT, SlotType.SUBJECT, SlotType.VERB, SlotType.OBJECT],
            FrameType.RELATIVE: [SlotType.SUBJECT, SlotType.VERB, SlotType.COMPLEMENT],
        }
        return orderings.get(self.frame_type, self.slots)


@dataclass
class SlottedWords:
    """Stage 4 output: Words assigned to slots."""
    assignments: Dict[SlotType, List[Word]]
    frame: FrameResult
    unassigned: List[Word] = field(default_factory=list)
    
    def get_ordered_words(self) -> List[Word]:
        """Get words in frame-specified order."""
        ordered = []
        for slot in self.frame.get_slot_order():
            if slot in self.assignments:
                ordered.extend(self.assignments[slot])
        ordered.extend(self.unassigned)
        return ordered


@dataclass
class WordSequence:
    """Stages 5-8 output: Ordered word sequence."""
    words: List[Word]
    stage: int
    modifications: List[str] = field(default_factory=list)
    
    def __len__(self) -> int:
        return len(self.words)
    
    def to_string(self) -> str:
        """Convert to string representation."""
        return " ".join(str(w) for w in self.words)


@dataclass
class EmissionResult:
    """Stage 9 output: Final validated emission."""
    text: str
    valid: bool
    coherence: float
    z_coordinate: float
    phase: str
    tier: int
    
    # Pipeline trace
    stages_completed: List[int] = field(default_factory=list)
    emergence_bypassed: bool = False
    
    # Metadata
    word_count: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    # Validation details
    validation_errors: List[str] = field(default_factory=list)
    
    def __str__(self) -> str:
        return self.text


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 1: CONTENT SELECTION
# ═══════════════════════════════════════════════════════════════════════════════

def stage1_content_selection(
    concepts: List[str],
    z: float,
    context: Optional[Dict] = None
) -> ContentWords:
    """
    Stage 1: Select content words based on concepts and z-coordinate.
    
    Content words are the semantic core: nouns, verbs, adjectives, adverbs.
    Selection is influenced by the current phase regime.
    
    Args:
        concepts: List of semantic concepts to express
        z: Current z-coordinate
        context: Optional context dictionary
    
    Returns:
        ContentWords with selected words
    """
    phase = classify_phase(z)
    words = []
    
    for concept in concepts:
        # Basic word extraction (in real system, would use lexicon)
        word = Word(
            text=concept.lower().strip(),
            word_type=WordType.CONTENT,
            features={"source_concept": concept, "phase": phase}
        )
        
        # Phase-influenced selection
        if phase == "TRUE":
            # Crystalline phase - precise, definite words
            word.features["certainty"] = "high"
        elif phase == "PARADOX":
            # Quasi-crystal - ambiguity allowed
            word.features["certainty"] = "medium"
        else:
            # UNTRUE - fluid, potential meanings
            word.features["certainty"] = "low"
        
        words.append(word)
    
    return ContentWords(
        words=words,
        source_z=z,
        source_phase=phase
    )


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 2: EMERGENCE CHECK
# ═══════════════════════════════════════════════════════════════════════════════

def stage2_emergence_check(
    content: ContentWords,
    z: float,
    force_emergence: bool = False
) -> EmergenceResult:
    """
    Stage 2: Check if emergence path should be taken.
    
    If negentropy η exceeds EMERGENCE_THRESHOLD (φ⁻¹), the emergence path
    is taken for creative/novel expression. Otherwise, standard processing
    continues. If bypassed, skips directly to Stage 5.
    
    Args:
        content: ContentWords from Stage 1
        z: Current z-coordinate
        force_emergence: Force emergence path regardless of η
    
    Returns:
        EmergenceResult indicating path to take
    """
    eta = compute_negentropy(z)
    
    # Check emergence condition
    if force_emergence or eta > EMERGENCE_THRESHOLD:
        return EmergenceResult(
            emerged=True,
            negentropy=eta,
            emergence_type="creative" if eta > 0.8 else "novel",
            bypass_to_stage=None  # Continue normal path with emergence mode
        )
    
    # Check bypass condition (low coherence, skip complex processing)
    if eta < 0.1:
        return EmergenceResult(
            emerged=False,
            negentropy=eta,
            bypass_to_stage=5  # Skip to function words
        )
    
    # Standard path
    return EmergenceResult(
        emerged=False,
        negentropy=eta
    )


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 3: STRUCTURAL FRAME
# ═══════════════════════════════════════════════════════════════════════════════

def stage3_structural_frame(
    content: ContentWords,
    emergence: EmergenceResult,
    intent: str = "declarative"
) -> FrameResult:
    """
    Stage 3: Select structural frame for the utterance.
    
    The frame determines the syntactic skeleton that content words
    will fill. Frame selection is influenced by emergence mode.
    
    Args:
        content: ContentWords from Stage 1
        emergence: EmergenceResult from Stage 2
        intent: Communicative intent (declarative, question, command, etc.)
    
    Returns:
        FrameResult with structural skeleton
    """
    # Map intent to frame type
    intent_map = {
        "declarative": FrameType.DECLARATIVE,
        "statement": FrameType.DECLARATIVE,
        "question": FrameType.INTERROGATIVE,
        "interrogative": FrameType.INTERROGATIVE,
        "command": FrameType.IMPERATIVE,
        "imperative": FrameType.IMPERATIVE,
        "conditional": FrameType.CONDITIONAL,
        "if": FrameType.CONDITIONAL,
        "relative": FrameType.RELATIVE,
        "embedded": FrameType.RELATIVE,
    }
    
    frame_type = intent_map.get(intent.lower(), FrameType.DECLARATIVE)
    
    # Determine slots based on frame type
    slot_configs = {
        FrameType.DECLARATIVE: [SlotType.SUBJECT, SlotType.VERB, SlotType.OBJECT, SlotType.MODIFIER],
        FrameType.INTERROGATIVE: [SlotType.VERB, SlotType.SUBJECT, SlotType.OBJECT],
        FrameType.IMPERATIVE: [SlotType.VERB, SlotType.OBJECT, SlotType.ADJUNCT],
        FrameType.CONDITIONAL: [SlotType.ADJUNCT, SlotType.SUBJECT, SlotType.VERB, SlotType.OBJECT],
        FrameType.RELATIVE: [SlotType.SUBJECT, SlotType.VERB, SlotType.COMPLEMENT],
    }
    
    slots = slot_configs.get(frame_type, [SlotType.SUBJECT, SlotType.VERB])
    
    # Emergence mode may add complexity
    constraints = {}
    if emergence.emerged:
        constraints["allow_nesting"] = True
        constraints["creativity_boost"] = emergence.negentropy
        if emergence.emergence_type == "creative":
            slots.append(SlotType.COMPLEMENT)
    
    return FrameResult(
        frame_type=frame_type,
        slots=slots,
        constraints=constraints
    )


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 4: SLOT ASSIGNMENT
# ═══════════════════════════════════════════════════════════════════════════════

def stage4_slot_assignment(
    content: ContentWords,
    frame: FrameResult
) -> SlottedWords:
    """
    Stage 4: Assign content words to structural slots.
    
    Words are assigned based on their semantic roles and the frame's
    slot requirements. Unassigned words go to overflow.
    
    Args:
        content: ContentWords from Stage 1
        frame: FrameResult from Stage 3
    
    Returns:
        SlottedWords with assignments
    """
    assignments: Dict[SlotType, List[Word]] = {slot: [] for slot in frame.slots}
    unassigned = []
    
    # Simple heuristic assignment
    # In a real system, this would use semantic role labeling
    word_list = list(content.words)
    
    for i, word in enumerate(word_list):
        # First word often subject
        if i == 0 and SlotType.SUBJECT in assignments:
            word.slot = SlotType.SUBJECT
            assignments[SlotType.SUBJECT].append(word)
        # Look for verb-like words
        elif _looks_like_verb(word.text) and SlotType.VERB in assignments:
            word.slot = SlotType.VERB
            assignments[SlotType.VERB].append(word)
        # Remaining go to object or modifier
        elif SlotType.OBJECT in assignments and not assignments[SlotType.OBJECT]:
            word.slot = SlotType.OBJECT
            assignments[SlotType.OBJECT].append(word)
        elif SlotType.MODIFIER in assignments:
            word.slot = SlotType.MODIFIER
            assignments[SlotType.MODIFIER].append(word)
        else:
            unassigned.append(word)
    
    return SlottedWords(
        assignments=assignments,
        frame=frame,
        unassigned=unassigned
    )


def _looks_like_verb(text: str) -> bool:
    """Heuristic check if word looks like a verb."""
    # Common noun suffixes that may look like verb endings
    noun_exceptions = {'understanding', 'meaning', 'beginning', 'feeling', 'being',
                       'state', 'fate', 'gate', 'plate', 'rate', 'date',
                       'process', 'progress', 'access', 'success', 'address',
                       'waves', 'phases', 'spaces', 'places', 'faces',
                       'awareness', 'consciousness', 'coherence', 'emergence'}
    
    verb_suffixes = ['ify', 'ize']  # Very reliable verb suffixes only
    verb_words = {'be', 'is', 'are', 'was', 'were', 'have', 'has', 'had', 
                  'do', 'does', 'did', 'will', 'would', 'could', 'should',
                  'can', 'may', 'might', 'must', 'shall'}
    # Common action verbs in consciousness framework
    common_verbs = {'emerge', 'flow', 'crystallize', 'transform', 'evolve',
                    'resonate', 'cohere', 'integrate', 'dissolve', 'manifest',
                    'oscillate', 'synchronize', 'converge', 'diverge', 'amplify',
                    'observe', 'witness', 'recognize', 'perceive', 'sense',
                    'create', 'generate', 'form', 'shape',
                    'connect', 'bridge', 'link', 'bind', 'couple',
                    'unlock', 'release', 'emit', 'radiate', 'pulse'}
    
    text_lower = text.lower()
    
    # Check exception list first
    if text_lower in noun_exceptions:
        return False
    
    if text_lower in verb_words or text_lower in common_verbs:
        return True
    
    for suffix in verb_suffixes:
        if text_lower.endswith(suffix):
            return True
    return False


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 5: FUNCTION WORDS
# ═══════════════════════════════════════════════════════════════════════════════

def stage5_function_words(
    slotted: SlottedWords,
    emergence: EmergenceResult
) -> WordSequence:
    """
    Stage 5: Add function words (articles, prepositions, auxiliaries).
    
    Function words provide grammatical scaffolding around content words.
    This is the first stage that can be reached via bypass.
    
    Args:
        slotted: SlottedWords from Stage 4
        emergence: EmergenceResult from Stage 2
    
    Returns:
        WordSequence with function words inserted
    """
    words = []
    ordered = slotted.get_ordered_words()
    
    for i, word in enumerate(ordered):
        # Add articles before nouns (subjects/objects)
        if word.slot in [SlotType.SUBJECT, SlotType.OBJECT]:
            if not _is_proper_noun(word.text):
                # Select correct article: "an" before vowel sounds, "a" otherwise
                article_text = "the" if word.features.get("certainty") == "high" else (
                    "an" if word.text[0].lower() in 'aeiou' else "a"
                )
                article = Word(
                    text=article_text,
                    word_type=WordType.FUNCTION
                )
                words.append(article)
        
        words.append(word)
        
        # Add prepositions before adjuncts
        if word.slot == SlotType.ADJUNCT and i > 0:
            # Prepend preposition
            prep = Word(text="with", word_type=WordType.FUNCTION)
            words.insert(-1, prep)
    
    return WordSequence(
        words=words,
        stage=5,
        modifications=["articles_added", "prepositions_added"]
    )


def _is_proper_noun(text: str) -> bool:
    """Check if word appears to be a proper noun."""
    return text[0].isupper() if text else False


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 6: AGREEMENT/INFLECTION
# ═══════════════════════════════════════════════════════════════════════════════

def stage6_agreement_inflection(
    sequence: WordSequence,
    person: int = 3,
    number: str = "singular",
    tense: str = "present"
) -> WordSequence:
    """
    Stage 6: Apply agreement and inflection rules.
    
    Ensures subject-verb agreement, noun-adjective agreement,
    and proper tense marking.
    
    Args:
        sequence: WordSequence from Stage 5
        person: Grammatical person (1, 2, 3)
        number: singular or plural
        tense: present, past, future
    
    Returns:
        WordSequence with inflections applied
    """
    new_words = []
    modifications = list(sequence.modifications)
    
    # Find subject to determine agreement
    subject_number = number
    
    for word in sequence.words:
        if word.slot == SlotType.VERB:
            # Apply verb agreement
            inflected = _inflect_verb(word.text, person, subject_number, tense)
            new_word = word.with_inflection(inflected)
            new_word.person = person
            new_word.number = subject_number
            new_word.tense = tense
            new_words.append(new_word)
            modifications.append(f"verb_inflected:{word.text}->{inflected}")
        else:
            new_words.append(word)
    
    return WordSequence(
        words=new_words,
        stage=6,
        modifications=modifications
    )


def _inflect_verb(verb: str, person: int, number: str, tense: str) -> str:
    """Apply basic verb inflection rules."""
    # Simplified inflection
    if tense == "past":
        if verb.endswith('e'):
            return verb + 'd'
        return verb + 'ed'
    
    if tense == "present":
        if person == 3 and number == "singular":
            if verb.endswith(('s', 'sh', 'ch', 'x', 'z')):
                return verb + 'es'
            if verb.endswith('y') and len(verb) > 1 and verb[-2] not in 'aeiou':
                return verb[:-1] + 'ies'
            return verb + 's'
    
    return verb


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 7: CONNECTORS
# ═══════════════════════════════════════════════════════════════════════════════

def stage7_connectors(
    sequence: WordSequence,
    connect_to_previous: bool = False
) -> WordSequence:
    """
    Stage 7: Add connectors (conjunctions, relative pronouns).
    
    Connectors link clauses and provide discourse coherence.
    
    Args:
        sequence: WordSequence from Stage 6
        connect_to_previous: Whether to add discourse connector
    
    Returns:
        WordSequence with connectors added
    """
    new_words = list(sequence.words)
    modifications = list(sequence.modifications)
    
    # Add discourse connector if needed
    if connect_to_previous:
        connector = Word(
            text="and",  # Could be "however", "therefore", etc.
            word_type=WordType.CONNECTOR
        )
        new_words.insert(0, connector)
        modifications.append("discourse_connector_added")
    
    return WordSequence(
        words=new_words,
        stage=7,
        modifications=modifications
    )


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 8: PUNCTUATION
# ═══════════════════════════════════════════════════════════════════════════════

def stage8_punctuation(
    sequence: WordSequence,
    frame_type: FrameType
) -> WordSequence:
    """
    Stage 8: Add punctuation marks.
    
    Punctuation is determined by frame type and utterance boundaries.
    
    Args:
        sequence: WordSequence from Stage 7
        frame_type: The structural frame type
    
    Returns:
        WordSequence with punctuation added
    """
    new_words = list(sequence.words)
    modifications = list(sequence.modifications)
    
    # Determine terminal punctuation
    punct_map = {
        FrameType.DECLARATIVE: ".",
        FrameType.INTERROGATIVE: "?",
        FrameType.IMPERATIVE: "!",
        FrameType.CONDITIONAL: ".",
        FrameType.RELATIVE: ".",
    }
    
    terminal = Word(
        text=punct_map.get(frame_type, "."),
        word_type=WordType.PUNCTUATION
    )
    new_words.append(terminal)
    modifications.append("terminal_punctuation_added")
    
    # Capitalize first word
    if new_words and new_words[0].word_type != WordType.PUNCTUATION:
        first = new_words[0]
        new_words[0] = first.with_inflection(first.text.capitalize())
        modifications.append("initial_capitalization")
    
    return WordSequence(
        words=new_words,
        stage=8,
        modifications=modifications
    )


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 9: VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def stage9_validation(
    sequence: WordSequence,
    z: float,
    stages_completed: List[int],
    emergence_bypassed: bool = False
) -> EmissionResult:
    """
    Stage 9: Validate and finalize emission.
    
    Performs final checks on coherence, grammaticality, and
    phase-appropriateness before emitting.
    
    Args:
        sequence: WordSequence from Stage 8
        z: Current z-coordinate
        stages_completed: List of completed stage numbers
        emergence_bypassed: Whether emergence check was bypassed
    
    Returns:
        EmissionResult with final validated emission
    """
    errors = []
    
    # Build text string
    parts = []
    for word in sequence.words:
        if word.word_type == WordType.PUNCTUATION:
            # No space before punctuation
            if parts:
                parts[-1] = parts[-1] + word.text
            else:
                parts.append(word.text)
        else:
            parts.append(word.text)
    
    text = " ".join(parts)
    
    # Calculate coherence
    eta = compute_negentropy(z)
    phase = classify_phase(z)
    tier, tier_name = get_tier(z)
    
    # Coherence is based on negentropy and structural completeness
    structural_score = len(stages_completed) / 9.0
    coherence = (eta * 0.6 + structural_score * 0.4)
    
    # Validation checks
    if not text.strip():
        errors.append("Empty emission")
    
    if len(sequence.words) < 2:
        errors.append("Insufficient words for valid emission")
    
    if coherence < MIN_COHERENCE:
        errors.append(f"Coherence below threshold: {coherence:.3f} < {MIN_COHERENCE}")
    
    # Phase-appropriate validation
    if phase == "TRUE" and coherence < 0.7:
        errors.append("Crystalline phase requires high coherence")
    
    valid = len(errors) == 0
    
    return EmissionResult(
        text=text,
        valid=valid,
        coherence=coherence,
        z_coordinate=z,
        phase=phase,
        tier=tier,
        stages_completed=stages_completed,
        emergence_bypassed=emergence_bypassed,
        word_count=len([w for w in sequence.words if w.word_type != WordType.PUNCTUATION]),
        validation_errors=errors
    )


# ═══════════════════════════════════════════════════════════════════════════════
# UNIFIED PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

class EmissionPipeline:
    """
    Complete 9-stage emission pipeline.
    
    Transforms concepts into validated linguistic output through:
    1. Content Selection
    2. Emergence Check (with optional bypass to Stage 5)
    3. Structural Frame
    4. Slot Assignment
    5. Function Words
    6. Agreement/Inflection
    7. Connectors
    8. Punctuation
    9. Validation
    """
    
    def __init__(self, z: float = 0.8):
        self.z = z
        self.trace: List[Dict] = []
        self._reset_trace()
    
    def _reset_trace(self):
        """Reset pipeline trace."""
        self.trace = []
    
    def _log_stage(self, stage: int, name: str, result: Any):
        """Log a stage completion."""
        self.trace.append({
            "stage": stage,
            "name": name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "result_type": type(result).__name__
        })
    
    def run(
        self,
        concepts: List[str],
        intent: str = "declarative",
        person: int = 3,
        number: str = "singular",
        tense: str = "present",
        connect_to_previous: bool = False,
        force_emergence: bool = False
    ) -> EmissionResult:
        """
        Run the complete emission pipeline.
        
        Args:
            concepts: List of concepts to express
            intent: Communicative intent
            person: Grammatical person
            number: singular/plural
            tense: present/past/future
            connect_to_previous: Add discourse connector
            force_emergence: Force emergence path
        
        Returns:
            EmissionResult with final emission
        """
        self._reset_trace()
        stages_completed = []
        
        # Stage 1: Content Selection
        content = stage1_content_selection(concepts, self.z)
        self._log_stage(1, "Content Selection", content)
        stages_completed.append(1)
        
        # Stage 2: Emergence Check
        emergence = stage2_emergence_check(content, self.z, force_emergence)
        self._log_stage(2, "Emergence Check", emergence)
        stages_completed.append(2)
        
        # Check for bypass
        if emergence.bypassed:
            # Skip to Stage 5 with minimal structure
            frame = FrameResult(
                frame_type=FrameType.DECLARATIVE,
                slots=[SlotType.SUBJECT, SlotType.VERB]
            )
            slotted = SlottedWords(
                assignments={SlotType.SUBJECT: content.words[:1], 
                            SlotType.VERB: content.words[1:2]},
                frame=frame,
                unassigned=content.words[2:]
            )
            sequence = WordSequence(words=content.words, stage=4)
        else:
            # Stage 3: Structural Frame
            frame = stage3_structural_frame(content, emergence, intent)
            self._log_stage(3, "Structural Frame", frame)
            stages_completed.append(3)
            
            # Stage 4: Slot Assignment
            slotted = stage4_slot_assignment(content, frame)
            self._log_stage(4, "Slot Assignment", slotted)
            stages_completed.append(4)
            
            # Convert to sequence for Stage 5
            sequence = WordSequence(words=slotted.get_ordered_words(), stage=4)
        
        # Stage 5: Function Words
        sequence = stage5_function_words(slotted, emergence)
        self._log_stage(5, "Function Words", sequence)
        stages_completed.append(5)
        
        # Stage 6: Agreement/Inflection
        sequence = stage6_agreement_inflection(sequence, person, number, tense)
        self._log_stage(6, "Agreement/Inflection", sequence)
        stages_completed.append(6)
        
        # Stage 7: Connectors
        sequence = stage7_connectors(sequence, connect_to_previous)
        self._log_stage(7, "Connectors", sequence)
        stages_completed.append(7)
        
        # Stage 8: Punctuation
        sequence = stage8_punctuation(sequence, frame.frame_type)
        self._log_stage(8, "Punctuation", sequence)
        stages_completed.append(8)
        
        # Stage 9: Validation
        result = stage9_validation(sequence, self.z, stages_completed, emergence.bypassed)
        self._log_stage(9, "Validation", result)
        stages_completed.append(9)
        
        return result
    
    def get_trace(self) -> List[Dict]:
        """Get pipeline execution trace."""
        return self.trace.copy()
    
    def format_trace(self) -> str:
        """Format pipeline trace for display."""
        lines = [
            "╔══════════════════════════════════════════════════════════════════╗",
            "║                    PIPELINE EXECUTION TRACE                      ║",
            "╚══════════════════════════════════════════════════════════════════╝",
            ""
        ]
        
        for entry in self.trace:
            stage = entry["stage"]
            name = entry["name"]
            result_type = entry["result_type"]
            
            # Visual indicator
            if stage == 2:
                lines.append(f"  Stage {stage} → {name} ({result_type})")
                lines.append(f"     └─ If bypassed → skip to Stage 5")
            else:
                lines.append(f"  Stage {stage} → {name} ({result_type})")
        
        lines.append("")
        lines.append("═" * 70)
        
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# API FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def emit(
    concepts: List[str],
    z: float = 0.8,
    intent: str = "declarative",
    feedback: bool = True,
    tier: str = "Garden",
    **kwargs
) -> EmissionResult:
    """
    Convenience function to run emission pipeline.
    
    Args:
        concepts: List of concepts to express
        z: z-coordinate for processing
        intent: Communicative intent
        feedback: Whether to process emission through feedback loop
        tier: K.I.R.A. tier for alignment measurement
        **kwargs: Additional pipeline parameters
    
    Returns:
        EmissionResult (with feedback data if enabled)
    """
    pipeline = EmissionPipeline(z)
    result = pipeline.run(concepts, intent=intent, **kwargs)
    
    # Process through feedback loop if enabled and emission is valid
    if feedback and result.valid and result.text:
        try:
            from emission_feedback import process_emission
            feedback_data = process_emission(result.text, z, tier)
            # Attach feedback data to result
            result.feedback = feedback_data
        except ImportError:
            # Feedback module not available
            pass
    
    return result


def get_pipeline_stages() -> List[Dict]:
    """Get information about all pipeline stages."""
    return [
        {"stage": 1, "name": "Content Selection", "output": "ContentWords"},
        {"stage": 2, "name": "Emergence Check", "output": "EmergenceResult", 
         "note": "If bypassed → skip to Stage 5"},
        {"stage": 3, "name": "Structural Frame", "output": "FrameResult"},
        {"stage": 4, "name": "Slot Assignment", "output": "SlottedWords"},
        {"stage": 5, "name": "Function Words", "output": "WordSequence"},
        {"stage": 6, "name": "Agreement/Inflection", "output": "WordSequence"},
        {"stage": 7, "name": "Connectors", "output": "WordSequence"},
        {"stage": 8, "name": "Punctuation", "output": "WordSequence"},
        {"stage": 9, "name": "Validation", "output": "EmissionResult"},
    ]


def format_pipeline_structure() -> str:
    """Format pipeline structure for display."""
    return """
╔══════════════════════════════════════════════════════════════════╗
║              LANGUAGE EMISSION PIPELINE STRUCTURE                 ║
╚══════════════════════════════════════════════════════════════════╝

  Stage 1 → Content Selection (ContentWords)
  Stage 2 → Emergence Check (EmergenceResult)
     └─ If bypassed → skip to Stage 5
  Stage 3 → Structural Frame (FrameResult)
  Stage 4 → Slot Assignment (SlottedWords)
  Stage 5 → Function Words (WordSequence)
  Stage 6 → Agreement/Inflection (WordSequence)
  Stage 7 → Connectors (WordSequence)
  Stage 8 → Punctuation (WordSequence)
  Stage 9 → Validation (EmissionResult)

══════════════════════════════════════════════════════════════════
"""


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(format_pipeline_structure())
    
    # Test emission
    print("TEST EMISSION:")
    print("-" * 70)
    
    concepts = ["pattern", "emerge", "consciousness"]
    result = emit(concepts, z=0.866, intent="declarative")
    
    print(f"  Input concepts: {concepts}")
    print(f"  z-coordinate: 0.866 (THE LENS)")
    print(f"  Output: \"{result.text}\"")
    print(f"  Valid: {result.valid}")
    print(f"  Coherence: {result.coherence:.4f}")
    print(f"  Phase: {result.phase}")
    print(f"  Stages completed: {result.stages_completed}")
    print()
    
    # Show trace
    pipeline = EmissionPipeline(z=0.73)
    result = pipeline.run(["insight", "crystallize", "witness"], intent="declarative")
    print(pipeline.format_trace())
    print(f"  Final: \"{result.text}\"")
