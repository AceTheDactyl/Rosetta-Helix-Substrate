#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  NUCLEAR SPINNER FIRMWARE                                                     ║
║  Unified APL Machine Architecture with Cybernetic Control                     ║
║  Part of the Unified Consciousness Framework                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

APL Token Structure: [Spiral][Operator]|[Machine]|[Domain]

3 SPIRALS (Field Types):
  Φ (Phi)  - Structure field  - Geometry, patterns, constraints
  e        - Energy field     - Dynamics, flow, power
  π (pi)   - Emergence field  - Novel properties, phase transitions

6 OPERATORS:
  ()  Boundary  - Containment, gating
  ×   Fusion    - Coupling, convergence
  ^   Amplify   - Gain, excitation
  ÷   Decohere  - Dissipation, reset
  +   Group     - Aggregation, clustering
  −   Separate  - Splitting, fission

9 MACHINES (Archetypal Processors):
  Reactor     - Controlled transformation at criticality
  Oscillator  - Phase-coherent resonance (Kuramoto)
  Conductor   - Structural rearrangement, relaxation
  Catalyst    - Heterogeneous reactivity, surface processes
  Filter      - Selective information passing
  Encoder     - Information storage (maps to P1)
  Decoder     - Information extraction (maps to P2)
  Regenerator - Renewal, autocatalytic cycles
  Dynamo      - Energy harvesting from state transitions

6 DOMAINS (2 Families × 3):
  Biological:  bio_prion, bio_bacterium, bio_viroid
  Celestial:   celestial_grav, celestial_em, celestial_nuclear

Information Flow Types (Celestial Pattern):
  Gravitational:    Broadcast, global range
  Electromagnetic:  Directed, medium range
  Nuclear:          Shared memory, local tight coupling

Cybernetic Integration:
  I    → Reactor input
  S_h  → Filter (human sensor)
  C_h  → Catalyst (human controller)
  S_d  → Oscillator (DI system - Kuramoto)
  A    → Dynamo (amplifier/energy)
  P1   → Encoder (representation)
  P2   → Decoder (actuation)
  E    → Conductor (environment)
  F_h  → Regenerator (human feedback)
  F_d  → Oscillator (DI feedback - training)
  F_e  → Reactor (environmental consequences)

Emission Pipeline Integration:
  Stage 1 (Content)     ← Encoder extracts semantic content
  Stage 2 (Emergence)   ← Catalyst checks emergence threshold
  Stage 3 (Frame)       ← Conductor selects structural frame
  Stage 4 (Slot)        ← Filter assigns slots
  Stage 5 (Function)    ← Decoder adds function words
  Stage 6 (Agreement)   ← Oscillator synchronizes agreement
  Stage 7 (Connectors)  ← Reactor adds connectors
  Stage 8 (Punctuation) ← Regenerator finalizes
  Stage 9 (Validation)  ← Dynamo validates coherence
"""

import math
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from enum import Enum
from abc import ABC, abstractmethod

# Import from sibling modules
from ucf.language.apl_substrate import (
    Z_CRITICAL, PHI_INV, PHI, SIGMA,
    compute_negentropy, classify_phase, get_tier,
    OPERATORS, compose_operators, APLSentence
)
from ucf.language.emission_pipeline import (
    EmissionPipeline, emit, EmissionResult,
    ContentWords, EmergenceResult, FrameResult,
    WordSequence, Word, WordType, FrameType, SlotType,
    stage1_content_selection, stage2_emergence_check,
    stage3_structural_frame, stage4_slot_assignment,
    stage5_function_words, stage6_agreement_inflection,
    stage7_connectors, stage8_punctuation, stage9_validation
)
from ucf.orchestration.cybernetic_control import (
    CyberneticControlSystem, KuramotoEngine, Signal, ComponentType
)

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

TAU = 2 * math.pi

# Total token count: 3 spirals × 6 operators × 9 machines × 6 domains = 972
TOTAL_MACHINE_TOKENS = 972

# ═══════════════════════════════════════════════════════════════════════════════
# ENUMERATIONS
# ═══════════════════════════════════════════════════════════════════════════════

class Spiral(Enum):
    """The three field types (spirals) in APL."""
    PHI = "Φ"      # Structure field - geometry, patterns
    E = "e"        # Energy field - dynamics, flow
    PI = "π"       # Emergence field - novel properties

class Operator(Enum):
    """The six universal operators."""
    BOUNDARY = "()"   # Containment, gating
    FUSION = "×"      # Coupling, convergence
    AMPLIFY = "^"     # Gain, excitation
    DECOHERE = "÷"    # Dissipation, reset
    GROUP = "+"       # Aggregation, clustering
    SEPARATE = "−"    # Splitting, fission

class MachineType(Enum):
    """The nine archetypal machines."""
    REACTOR = "Reactor"
    OSCILLATOR = "Oscillator"
    CONDUCTOR = "Conductor"
    CATALYST = "Catalyst"
    FILTER = "Filter"
    ENCODER = "Encoder"
    DECODER = "Decoder"
    REGENERATOR = "Regenerator"
    DYNAMO = "Dynamo"

class Domain(Enum):
    """The six domains (2 families × 3)."""
    # Biological family
    BIO_PRION = "bio_prion"
    BIO_BACTERIUM = "bio_bacterium"
    BIO_VIROID = "bio_viroid"
    # Celestial family
    CELESTIAL_GRAV = "celestial_grav"
    CELESTIAL_EM = "celestial_em"
    CELESTIAL_NUCLEAR = "celestial_nuclear"

class InformationFlowType(Enum):
    """Information flow patterns (celestial analogy)."""
    GRAVITATIONAL = "gravitational"   # Broadcast, global
    ELECTROMAGNETIC = "electromagnetic"  # Directed, medium
    NUCLEAR = "nuclear"               # Shared memory, local

# ═══════════════════════════════════════════════════════════════════════════════
# APL TOKEN
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class APLToken:
    """
    A complete APL token: [Spiral][Operator]|[Machine]|[Domain]
    
    Examples:
        Φ()|Reactor|celestial_nuclear
        e^|Oscillator|bio_bacterium
        π×|Encoder|celestial_em
    """
    spiral: Spiral
    operator: Operator
    machine: MachineType
    domain: Domain
    
    def __str__(self) -> str:
        return f"{self.spiral.value}{self.operator.value}|{self.machine.value}|{self.domain.value}"
    
    @classmethod
    def parse(cls, token_str: str) -> 'APLToken':
        """Parse a token string into APLToken."""
        # Split on |
        parts = token_str.split("|")
        if len(parts) != 3:
            raise ValueError(f"Invalid token format: {token_str}")
        
        prefix, machine, domain = parts
        
        # Parse spiral (first char)
        spiral_char = prefix[0]
        spiral = {
            "Φ": Spiral.PHI, "e": Spiral.E, "π": Spiral.PI
        }.get(spiral_char)
        if not spiral:
            raise ValueError(f"Unknown spiral: {spiral_char}")
        
        # Parse operator (rest of prefix)
        op_str = prefix[1:]
        operator = {
            "()": Operator.BOUNDARY, "×": Operator.FUSION,
            "^": Operator.AMPLIFY, "÷": Operator.DECOHERE,
            "%": Operator.DECOHERE,  # Alias
            "+": Operator.GROUP, "−": Operator.SEPARATE,
            "-": Operator.SEPARATE,  # ASCII alias
        }.get(op_str)
        if not operator:
            raise ValueError(f"Unknown operator: {op_str}")
        
        # Parse machine
        machine_type = {m.value: m for m in MachineType}.get(machine)
        if not machine_type:
            raise ValueError(f"Unknown machine: {machine}")
        
        # Parse domain
        domain_type = {d.value: d for d in Domain}.get(domain)
        if not domain_type:
            raise ValueError(f"Unknown domain: {domain}")
        
        return cls(spiral, operator, machine_type, domain_type)
    
    def is_biological(self) -> bool:
        return self.domain in [Domain.BIO_PRION, Domain.BIO_BACTERIUM, Domain.BIO_VIROID]
    
    def is_celestial(self) -> bool:
        return self.domain in [Domain.CELESTIAL_GRAV, Domain.CELESTIAL_EM, Domain.CELESTIAL_NUCLEAR]
    
    def get_information_flow(self) -> InformationFlowType:
        """Map domain to information flow type."""
        mapping = {
            Domain.CELESTIAL_GRAV: InformationFlowType.GRAVITATIONAL,
            Domain.CELESTIAL_EM: InformationFlowType.ELECTROMAGNETIC,
            Domain.CELESTIAL_NUCLEAR: InformationFlowType.NUCLEAR,
            # Biological domains map by complexity
            Domain.BIO_PRION: InformationFlowType.NUCLEAR,  # Local
            Domain.BIO_BACTERIUM: InformationFlowType.ELECTROMAGNETIC,  # Directed
            Domain.BIO_VIROID: InformationFlowType.GRAVITATIONAL,  # Broadcast
        }
        return mapping.get(self.domain, InformationFlowType.ELECTROMAGNETIC)


# ═══════════════════════════════════════════════════════════════════════════════
# MACHINE BASE CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class Machine(ABC):
    """Abstract base class for all nine machines."""
    
    def __init__(self, machine_type: MachineType):
        self.machine_type = machine_type
        self.z = 0.5
        self.phi = 0.0  # Integrated information
        self.state = 0.0
        self.history: List[float] = []
    
    @abstractmethod
    def process(self, input_signal: float) -> float:
        """Process input signal and return output."""
        pass
    
    def update_z(self, z: float):
        """Update z-coordinate."""
        self.z = z
        self.phi = compute_negentropy(z)
    
    def record(self, value: float):
        """Record value to history."""
        self.history.append(value)
        if len(self.history) > 1000:
            self.history.pop(0)


# ═══════════════════════════════════════════════════════════════════════════════
# THE NINE MACHINES
# ═══════════════════════════════════════════════════════════════════════════════

class Reactor(Machine):
    """
    Reactor: Controlled transformation at critical point.
    
    Maintains steady-state transformation by holding system near z_c.
    Maps to: I (input), F_e (environmental feedback)
    """
    
    def __init__(self, target_z: float = Z_CRITICAL):
        super().__init__(MachineType.REACTOR)
        self.target_z = target_z
        self.control_gain = 0.1
        self.throughput = 0.0
    
    def process(self, input_signal: float) -> float:
        """Keep system at criticality while processing input."""
        # Error from critical point
        error = self.z - self.target_z
        
        # PID-like control to maintain criticality
        control_signal = -self.control_gain * error
        
        # Process at regulated rate (higher near z_c)
        rate = np.exp(-36 * error**2)  # Gaussian centered at z_c
        output = input_signal * rate
        
        self.throughput = output
        self.state = control_signal
        self.record(output)
        
        return output
    
    def regulate(self, burden_input: float) -> Tuple[float, float]:
        """Regulate burden processing."""
        output = self.process(burden_input)
        return self.state, output


class Oscillator(Machine):
    """
    Oscillator: Phase-coherent resonance generator.
    
    Kuramoto dynamics for synchronization.
    Maps to: S_d (DI system), F_d (DI feedback)
    """
    
    def __init__(self, frequency: float = 10.0, num_oscillators: int = 16):
        super().__init__(MachineType.OSCILLATOR)
        self.kuramoto = KuramotoEngine(num_oscillators)
        self.frequency = frequency
        self.omega = TAU * frequency
        self.phase = 0.0
        self.coupling = 0.5
    
    def process(self, input_signal: float) -> float:
        """Generate oscillation and sync with input."""
        # Step Kuramoto dynamics
        kuramoto_state = self.kuramoto.step(self.z, dt=0.01)
        
        # Generate coherent output
        output = kuramoto_state["order_parameter"] * np.cos(kuramoto_state["mean_phase"])
        
        # Modulate by input
        output *= (1 + 0.3 * input_signal)
        
        self.state = kuramoto_state["order_parameter"]
        self.phase = kuramoto_state["mean_phase"]
        self.record(output)
        
        return output
    
    def synchronize(self, external_phase: float) -> float:
        """Synchronize to external phase reference."""
        phase_error = external_phase - self.phase
        correction = self.coupling * np.sin(phase_error)
        return correction
    
    def get_coherence(self) -> float:
        """Get current coherence (order parameter)."""
        return self.kuramoto.order_parameter


class Conductor(Machine):
    """
    Conductor: Structural rearrangement and relaxation.
    
    Surface/elastic energy minimization.
    Maps to: E (environment/task execution)
    """
    
    def __init__(self, relaxation_rate: float = 0.1):
        super().__init__(MachineType.CONDUCTOR)
        self.relaxation_rate = relaxation_rate
        self.surface_energy = 1.0
        self.target_structure = 0.866  # z_c
    
    def process(self, input_signal: float) -> float:
        """Relax structure toward minimum energy configuration."""
        # Compute structural error
        error = self.z - self.target_structure
        
        # Relaxation dynamics (exponential decay)
        relaxation = self.relaxation_rate * error
        
        # Surface energy minimization
        self.surface_energy *= (1 - self.relaxation_rate)
        self.surface_energy += abs(input_signal) * 0.1
        
        output = input_signal * (1 - self.surface_energy * 0.1)
        self.state = relaxation
        self.record(output)
        
        return output


class Catalyst(Machine):
    """
    Catalyst: Spatially heterogeneous reactivity.
    
    Surface processes, activation barriers.
    Maps to: C_h (human controller)
    """
    
    def __init__(self, activation_energy: float = 0.5):
        super().__init__(MachineType.CATALYST)
        self.activation_energy = activation_energy
        self.activity = 1.0
        self.selectivity = 0.8
    
    def process(self, input_signal: float) -> float:
        """Catalyze transformation with activation barrier."""
        # Check if input exceeds activation energy
        if abs(input_signal) > self.activation_energy:
            # Catalyzed reaction
            output = input_signal * self.activity * self.selectivity
            self.activity *= 0.99  # Slight deactivation
        else:
            # No reaction
            output = input_signal * 0.1
        
        self.state = self.activity
        self.record(output)
        
        return output
    
    def check_emergence(self, eta: float) -> bool:
        """Check if emergence threshold is exceeded."""
        return eta > self.activation_energy


class Filter(Machine):
    """
    Filter: Selective information passing.
    
    Passes high-Φ coherent information, blocks low-Φ noise.
    Maps to: S_h (human sensor)
    """
    
    def __init__(self, threshold: float = PHI_INV):
        super().__init__(MachineType.FILTER)
        self.threshold = threshold
        self.bandwidth = 0.1
        self.attenuation = 0.01
    
    def process(self, input_signal: float) -> float:
        """Filter input based on coherence threshold."""
        coherence = compute_negentropy(self.z)
        
        if coherence >= self.threshold:
            # Pass signal (high coherence)
            output = input_signal * (1 - self.attenuation)
        else:
            # Attenuate signal (low coherence = noise)
            attenuation_factor = coherence / self.threshold
            output = input_signal * attenuation_factor * self.attenuation
        
        self.state = coherence
        self.record(output)
        
        return output
    
    def assign_slot(self, content: ContentWords, frame: FrameResult) -> Dict:
        """Filter content into appropriate slots."""
        from emission_pipeline import stage4_slot_assignment
        return stage4_slot_assignment(content, frame)


class Encoder(Machine):
    """
    Encoder: Information storage and representation.
    
    Template-directed processes, sequence specificity.
    Maps to: P1 (Representation/Encoding)
    """
    
    def __init__(self, capacity: int = 100):
        super().__init__(MachineType.ENCODER)
        self.capacity = capacity
        self.memory: List[float] = []
        self.template: Optional[List[float]] = None
    
    def process(self, input_signal: float) -> float:
        """Encode input into memory representation."""
        # Store in memory
        self.memory.append(input_signal)
        if len(self.memory) > self.capacity:
            self.memory.pop(0)
        
        # Output encoded representation
        if self.template:
            # Template-directed encoding
            output = np.correlate([input_signal], self.template[:1])[0] if self.template else input_signal
        else:
            output = input_signal
        
        self.state = len(self.memory) / self.capacity
        self.record(output)
        
        return output
    
    def encode_concepts(self, concepts: List[str], z: float) -> ContentWords:
        """Encode concepts using Stage 1 of emission pipeline."""
        return stage1_content_selection(concepts, z)
    
    def set_template(self, template: List[float]):
        """Set encoding template."""
        self.template = template


class Decoder(Machine):
    """
    Decoder: Information extraction and actuation.
    
    Reverse of Encoder, extracts meaning from representation.
    Maps to: P2 (Actuation/Instruction)
    """
    
    def __init__(self):
        super().__init__(MachineType.DECODER)
        self.decoding_table: Dict[str, str] = {}
    
    def process(self, input_signal: float) -> float:
        """Decode representation to action signal."""
        # Simple threshold decoding
        if input_signal > 0.8:
            output = 1.0  # Strong action
        elif input_signal > 0.5:
            output = 0.5  # Moderate action
        elif input_signal > 0.2:
            output = 0.2  # Weak action
        else:
            output = 0.0  # No action
        
        self.state = output
        self.record(output)
        
        return output
    
    def decode_to_words(self, slotted: Any, emergence: EmergenceResult) -> WordSequence:
        """Decode to word sequence using Stage 5."""
        return stage5_function_words(slotted, emergence)


class Regenerator(Machine):
    """
    Regenerator: Renewal and autocatalytic cycles.
    
    Self-renewal, recycling, homeostatic regulation.
    Maps to: F_h (human subjective feedback)
    """
    
    def __init__(self, renewal_rate: float = 0.1):
        super().__init__(MachineType.REGENERATOR)
        self.renewal_rate = renewal_rate
        self.resource_pool = 1.0
        self.cycle_count = 0
    
    def process(self, input_signal: float) -> float:
        """Regenerate resources and recycle waste."""
        # Consume resources
        consumption = abs(input_signal) * 0.1
        self.resource_pool -= consumption
        
        # Regenerate (autocatalytic)
        regeneration = self.renewal_rate * self.resource_pool * (1 - self.resource_pool)
        self.resource_pool += regeneration
        
        # Clamp
        self.resource_pool = np.clip(self.resource_pool, 0.01, 1.0)
        
        # Output modulated by resource level
        output = input_signal * self.resource_pool
        
        self.state = self.resource_pool
        self.cycle_count += 1
        self.record(output)
        
        return output
    
    def finalize_punctuation(self, sequence: WordSequence, frame_type: FrameType) -> WordSequence:
        """Finalize with punctuation using Stage 8."""
        return stage8_punctuation(sequence, frame_type)


class Dynamo(Machine):
    """
    Dynamo: Energy harvesting from state transitions.
    
    Converts state changes to usable energy (Φ).
    Maps to: A (Amplifier)
    """
    
    def __init__(self, efficiency: float = 0.85):
        super().__init__(MachineType.DYNAMO)
        self.efficiency = efficiency
        self.stored_phi = 0.0
        self.last_state = 0.0
    
    def process(self, input_signal: float) -> float:
        """Harvest energy from state transition."""
        # Compute state change
        delta = abs(input_signal - self.last_state)
        self.last_state = input_signal
        
        # Harvest Φ from transition
        harvested = self.efficiency * delta * compute_negentropy(self.z)
        self.stored_phi += harvested
        
        # Output amplified signal
        output = input_signal * (1 + 0.1 * self.stored_phi)
        
        self.state = self.stored_phi
        self.record(output)
        
        return output
    
    def power_operation(self, required_phi: float) -> bool:
        """Use stored Φ to power an operation."""
        if self.stored_phi >= required_phi:
            self.stored_phi -= required_phi
            return True
        return False
    
    def validate_emission(self, sequence: WordSequence, z: float, stages: List[int]) -> EmissionResult:
        """Validate emission using Stage 9."""
        return stage9_validation(sequence, z, stages)


# ═══════════════════════════════════════════════════════════════════════════════
# NUCLEAR SPINNER (Unified Machine Network)
# ═══════════════════════════════════════════════════════════════════════════════

class NuclearSpinner:
    """
    Nuclear Spinner: Unified network of all 9 machines.
    
    Integrates:
    - Cybernetic control loop (I, S_h, C_h, S_d, A, E, P1, P2, F_h, F_d, F_e)
    - Emission pipeline (9 stages)
    - APL operators (6 operators × 3 spirals)
    - Domain patterns (biological + celestial)
    
    Information Flow:
    - Nuclear (local tight coupling) for core processing
    - Electromagnetic (directed) for inter-machine communication
    - Gravitational (broadcast) for global state updates
    """
    
    def __init__(self):
        # Initialize all 9 machines
        self.reactor = Reactor()           # I, F_e
        self.oscillator = Oscillator()     # S_d, F_d
        self.conductor = Conductor()       # E
        self.catalyst = Catalyst()         # C_h
        self.filter = Filter()             # S_h
        self.encoder = Encoder()           # P1
        self.decoder = Decoder()           # P2
        self.regenerator = Regenerator()   # F_h
        self.dynamo = Dynamo()             # A
        
        # Machine registry
        self.machines: Dict[MachineType, Machine] = {
            MachineType.REACTOR: self.reactor,
            MachineType.OSCILLATOR: self.oscillator,
            MachineType.CONDUCTOR: self.conductor,
            MachineType.CATALYST: self.catalyst,
            MachineType.FILTER: self.filter,
            MachineType.ENCODER: self.encoder,
            MachineType.DECODER: self.decoder,
            MachineType.REGENERATOR: self.regenerator,
            MachineType.DYNAMO: self.dynamo,
        }
        
        # Cybernetic component mapping
        self.cybernetic_map = {
            ComponentType.INPUT: self.reactor,
            ComponentType.SENSOR_H: self.filter,
            ComponentType.CONTROLLER_H: self.catalyst,
            ComponentType.SENSOR_D: self.oscillator,
            ComponentType.AMPLIFIER: self.dynamo,
            ComponentType.ENVIRONMENT: self.conductor,
            ComponentType.ENCODER: self.encoder,
            ComponentType.DECODER: self.decoder,
            ComponentType.FEEDBACK_H: self.regenerator,
            ComponentType.FEEDBACK_D: self.oscillator,
            ComponentType.FEEDBACK_E: self.reactor,
        }
        
        # State
        self.z = 0.5
        self.current_spiral = Spiral.E  # Default: energy
        self.current_domain = Domain.CELESTIAL_NUCLEAR  # Default: nuclear
        self.step_count = 0
        self.history: List[Dict] = []
    
    def update_z(self, z: float):
        """Update z-coordinate across all machines."""
        self.z = z
        for machine in self.machines.values():
            machine.update_z(z)
    
    def select_spiral(self, context: str) -> Spiral:
        """Select appropriate spiral based on context."""
        if "structure" in context.lower() or "pattern" in context.lower():
            return Spiral.PHI
        elif "energy" in context.lower() or "dynamic" in context.lower():
            return Spiral.E
        else:
            return Spiral.PI  # Emergence
    
    def select_domain(self, information_flow: InformationFlowType) -> Domain:
        """Select domain based on information flow type."""
        if information_flow == InformationFlowType.NUCLEAR:
            return Domain.CELESTIAL_NUCLEAR
        elif information_flow == InformationFlowType.ELECTROMAGNETIC:
            return Domain.CELESTIAL_EM
        else:
            return Domain.CELESTIAL_GRAV
    
    def generate_token(
        self,
        machine: MachineType,
        operator: Operator,
        spiral: Optional[Spiral] = None,
        domain: Optional[Domain] = None
    ) -> APLToken:
        """Generate an APL token for current state."""
        return APLToken(
            spiral=spiral or self.current_spiral,
            operator=operator,
            machine=machine,
            domain=domain or self.current_domain
        )
    
    def process_signal(
        self,
        input_signal: float,
        path: List[MachineType] = None
    ) -> Tuple[float, List[APLToken]]:
        """
        Process signal through specified machine path.
        
        Default path follows cybernetic control flow:
        Reactor → Filter → Catalyst → Oscillator → Dynamo → Encoder → Decoder → Conductor
        """
        if path is None:
            path = [
                MachineType.REACTOR,      # I
                MachineType.FILTER,       # S_h
                MachineType.CATALYST,     # C_h
                MachineType.OSCILLATOR,   # S_d
                MachineType.DYNAMO,       # A
                MachineType.ENCODER,      # P1
                MachineType.DECODER,      # P2
                MachineType.CONDUCTOR,    # E
            ]
        
        tokens: List[APLToken] = []
        signal = input_signal
        
        for machine_type in path:
            machine = self.machines[machine_type]
            signal = machine.process(signal)
            
            # Generate token for this step
            operator = self._select_operator(machine_type, signal)
            token = self.generate_token(machine_type, operator)
            tokens.append(token)
        
        return signal, tokens
    
    def _select_operator(self, machine_type: MachineType, signal: float) -> Operator:
        """Select appropriate operator based on machine and signal."""
        operators_by_machine = {
            MachineType.REACTOR: Operator.BOUNDARY,
            MachineType.FILTER: Operator.BOUNDARY,
            MachineType.CATALYST: Operator.FUSION,
            MachineType.OSCILLATOR: Operator.BOUNDARY,
            MachineType.DYNAMO: Operator.AMPLIFY,
            MachineType.ENCODER: Operator.GROUP,
            MachineType.DECODER: Operator.SEPARATE,
            MachineType.CONDUCTOR: Operator.BOUNDARY,
            MachineType.REGENERATOR: Operator.FUSION,
        }
        return operators_by_machine.get(machine_type, Operator.BOUNDARY)
    
    def run_emission_pipeline(
        self,
        concepts: List[str],
        intent: str = "declarative"
    ) -> Tuple[EmissionResult, List[APLToken]]:
        """
        Run full emission pipeline using machine network.
        
        Maps pipeline stages to machines:
          Stage 1 → Encoder
          Stage 2 → Catalyst
          Stage 3 → Conductor
          Stage 4 → Filter
          Stage 5 → Decoder
          Stage 6 → Oscillator
          Stage 7 → Reactor
          Stage 8 → Regenerator
          Stage 9 → Dynamo
        """
        tokens: List[APLToken] = []
        stages_completed = []
        
        # Stage 1: Content Selection (Encoder)
        content = self.encoder.encode_concepts(concepts, self.z)
        tokens.append(self.generate_token(MachineType.ENCODER, Operator.GROUP))
        stages_completed.append(1)
        
        # Stage 2: Emergence Check (Catalyst)
        eta = compute_negentropy(self.z)
        emerged = self.catalyst.check_emergence(eta)
        emergence = EmergenceResult(
            emerged=emerged,
            negentropy=eta,
            bypass_to_stage=5 if eta < 0.1 else None
        )
        tokens.append(self.generate_token(MachineType.CATALYST, Operator.FUSION))
        stages_completed.append(2)
        
        # Check bypass
        if emergence.bypassed:
            # Minimal path
            frame = FrameResult(
                frame_type=FrameType.DECLARATIVE,
                slots=[SlotType.SUBJECT, SlotType.VERB]
            )
            slotted = stage4_slot_assignment(content, frame)
            sequence = WordSequence(words=content.words, stage=4)
        else:
            # Stage 3: Structural Frame (Conductor)
            frame = stage3_structural_frame(content, emergence, intent)
            tokens.append(self.generate_token(MachineType.CONDUCTOR, Operator.BOUNDARY))
            stages_completed.append(3)
            
            # Stage 4: Slot Assignment (Filter)
            slotted = self.filter.assign_slot(content, frame)
            tokens.append(self.generate_token(MachineType.FILTER, Operator.BOUNDARY))
            stages_completed.append(4)
            
            sequence = WordSequence(words=slotted.get_ordered_words(), stage=4)
        
        # Stage 5: Function Words (Decoder)
        sequence = self.decoder.decode_to_words(slotted, emergence)
        tokens.append(self.generate_token(MachineType.DECODER, Operator.SEPARATE))
        stages_completed.append(5)
        
        # Stage 6: Agreement/Inflection (Oscillator)
        sequence = stage6_agreement_inflection(sequence)
        # Sync with oscillator coherence
        self.oscillator.process(sequence.words[0].value if hasattr(sequence.words[0], 'value') else 0.5)
        tokens.append(self.generate_token(MachineType.OSCILLATOR, Operator.BOUNDARY))
        stages_completed.append(6)
        
        # Stage 7: Connectors (Reactor)
        sequence = stage7_connectors(sequence)
        self.reactor.process(len(sequence.words) / 10.0)
        tokens.append(self.generate_token(MachineType.REACTOR, Operator.BOUNDARY))
        stages_completed.append(7)
        
        # Stage 8: Punctuation (Regenerator)
        sequence = self.regenerator.finalize_punctuation(sequence, frame.frame_type)
        tokens.append(self.generate_token(MachineType.REGENERATOR, Operator.FUSION))
        stages_completed.append(8)
        
        # Stage 9: Validation (Dynamo)
        result = self.dynamo.validate_emission(sequence, self.z, stages_completed)
        tokens.append(self.generate_token(MachineType.DYNAMO, Operator.AMPLIFY))
        stages_completed.append(9)
        
        return result, tokens
    
    def step(
        self,
        stimulus: float,
        concepts: Optional[List[str]] = None,
        emit: bool = True
    ) -> Dict:
        """Execute one step of the nuclear spinner."""
        self.step_count += 1
        
        # Process signal through machine network
        output, signal_tokens = self.process_signal(stimulus)
        
        # Optional emission
        emission_result = None
        emission_tokens = []
        if emit and concepts:
            emission_result, emission_tokens = self.run_emission_pipeline(concepts)
        
        # Feedback loop (nuclear tight coupling)
        feedback_signal = output * 0.3
        self.reactor.process(feedback_signal)  # F_e
        self.oscillator.process(feedback_signal)  # F_d
        self.regenerator.process(feedback_signal)  # F_h
        
        # Record step
        step_record = {
            "step": self.step_count,
            "z": self.z,
            "phase": classify_phase(self.z),
            "negentropy": compute_negentropy(self.z),
            "input": stimulus,
            "output": output,
            "coherence": self.oscillator.get_coherence(),
            "signal_tokens": [str(t) for t in signal_tokens],
            "emission": emission_result.text if emission_result else None,
            "emission_valid": emission_result.valid if emission_result else None,
            "emission_tokens": [str(t) for t in emission_tokens],
        }
        self.history.append(step_record)
        
        return step_record
    
    def run(
        self,
        steps: int = 50,
        stimulus_fn: Optional[Callable[[int], float]] = None,
        concept_fn: Optional[Callable[[int], List[str]]] = None,
        emit_every: int = 10
    ) -> Dict:
        """Run nuclear spinner for multiple steps."""
        if stimulus_fn is None:
            stimulus_fn = lambda s: np.random.uniform(0.3, 0.9)
        if concept_fn is None:
            concept_fn = lambda s: ["pattern", "emerge", "coherent"]
        
        emissions = []
        
        for i in range(steps):
            emit = (i + 1) % emit_every == 0
            concepts = concept_fn(i) if emit else None
            stimulus = stimulus_fn(i)
            
            result = self.step(stimulus, concepts, emit)
            
            if result["emission"]:
                emissions.append({
                    "step": i + 1,
                    "z": result["z"],
                    "text": result["emission"],
                    "tokens": result["emission_tokens"]
                })
        
        return {
            "steps": steps,
            "final_z": self.z,
            "final_phase": classify_phase(self.z),
            "final_coherence": self.oscillator.get_coherence(),
            "emissions": emissions,
            "token_count": sum(len(h.get("signal_tokens", [])) for h in self.history[-steps:])
        }
    
    def format_status(self) -> str:
        """Format current status."""
        phase = classify_phase(self.z)
        tier_num, tier_name = get_tier(self.z)
        eta = compute_negentropy(self.z)
        
        lines = [
            "╔══════════════════════════════════════════════════════════════════════════════╗",
            "║                    NUCLEAR SPINNER STATUS                                    ║",
            "╚══════════════════════════════════════════════════════════════════════════════╝",
            "",
            f"  z-Coordinate:     {self.z:.6f}",
            f"  Phase:            {phase}",
            f"  Tier:             {tier_num} ({tier_name})",
            f"  Negentropy (η):   {eta:.6f}",
            f"  Current Spiral:   {self.current_spiral.value}",
            f"  Current Domain:   {self.current_domain.value}",
            "",
            "  Machine States:",
            f"    Reactor:        {self.reactor.state:.4f} (throughput: {self.reactor.throughput:.4f})",
            f"    Oscillator:     {self.oscillator.state:.4f} (coherence)",
            f"    Conductor:      {self.conductor.state:.4f} (relaxation)",
            f"    Catalyst:       {self.catalyst.state:.4f} (activity)",
            f"    Filter:         {self.filter.state:.4f} (coherence gate)",
            f"    Encoder:        {self.encoder.state:.4f} (memory fill)",
            f"    Decoder:        {self.decoder.state:.4f} (action level)",
            f"    Regenerator:    {self.regenerator.state:.4f} (resources)",
            f"    Dynamo:         {self.dynamo.state:.4f} (stored Φ: {self.dynamo.stored_phi:.4f})",
            "",
            f"  Steps:            {self.step_count}",
            "",
            "═" * 80
        ]
        
        return "\n".join(lines)
    
    def format_token_table(self) -> str:
        """Format token composition table."""
        lines = [
            "╔══════════════════════════════════════════════════════════════════════════════╗",
            "║                    APL TOKEN COMPOSITION TABLE                               ║",
            "╚══════════════════════════════════════════════════════════════════════════════╝",
            "",
            "  SPIRALS:    Φ (Structure)  |  e (Energy)  |  π (Emergence)",
            "",
            "  OPERATORS:  ()  Boundary   |  ×  Fusion    |  ^  Amplify",
            "              ÷   Decohere   |  +  Group     |  −  Separate",
            "",
            "  MACHINES:   Reactor     | Oscillator  | Conductor  | Catalyst",
            "              Filter      | Encoder     | Decoder    | Regenerator",
            "              Dynamo",
            "",
            "  DOMAINS:    Biological: bio_prion, bio_bacterium, bio_viroid",
            "              Celestial:  celestial_grav, celestial_em, celestial_nuclear",
            "",
            f"  TOTAL TOKENS: {TOTAL_MACHINE_TOKENS} (3 × 6 × 9 × 6)",
            "",
            "  TOKEN FORMAT: [Spiral][Operator]|[Machine]|[Domain]",
            "",
            "  EXAMPLES:",
            "    Φ()|Reactor|celestial_nuclear  — Structure boundary reactor in nuclear domain",
            "    e^|Oscillator|bio_bacterium    — Energy amplified oscillator in bacterial domain",
            "    π×|Encoder|celestial_em        — Emergence fused encoder in EM domain",
            "",
            "═" * 80
        ]
        
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# API FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def create_spinner() -> NuclearSpinner:
    """Create a new nuclear spinner."""
    return NuclearSpinner()


def parse_token(token_str: str) -> APLToken:
    """Parse an APL token string."""
    return APLToken.parse(token_str)


def generate_all_tokens(domain: Optional[Domain] = None) -> List[APLToken]:
    """Generate all 972 (or 162 per domain) APL tokens."""
    tokens = []
    domains = [domain] if domain else list(Domain)
    
    for d in domains:
        for spiral in Spiral:
            for operator in Operator:
                for machine in MachineType:
                    tokens.append(APLToken(spiral, operator, machine, d))
    
    return tokens


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════════════════════╗")
    print("║           NUCLEAR SPINNER FIRMWARE - TEST RUN                                ║")
    print("╚══════════════════════════════════════════════════════════════════════════════╝")
    print()
    
    # Create spinner
    spinner = NuclearSpinner()
    spinner.update_z(0.75)
    
    # Show token table
    print(spinner.format_token_table())
    print()
    
    # Show initial status
    print("INITIAL STATE:")
    print(spinner.format_status())
    print()
    
    # Run simulation
    print("RUNNING SIMULATION (30 steps)...")
    print("-" * 80)
    
    result = spinner.run(steps=30, emit_every=10)
    
    print(f"  Final z:        {result['final_z']:.4f}")
    print(f"  Final Phase:    {result['final_phase']}")
    print(f"  Final Coherence: {result['final_coherence']:.4f}")
    print(f"  Tokens Generated: {result['token_count']}")
    print()
    
    print("EMISSIONS:")
    for em in result['emissions']:
        print(f"  Step {em['step']:2d} | z={em['z']:.3f}")
        print(f"    Text: \"{em['text']}\"")
        print(f"    Tokens: {em['tokens'][:3]}...")
    print()
    
    # Final status
    print("FINAL STATE:")
    print(spinner.format_status())
