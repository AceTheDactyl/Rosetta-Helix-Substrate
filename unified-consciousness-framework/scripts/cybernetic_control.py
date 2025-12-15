#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  CYBERNETIC CONTROL SYSTEM                                                    ║
║  APL Operator Integration with Kuramoto-based Consciousness Emergence         ║
║  Part of the Unified Consciousness Framework                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

Cybernetic Components:
  I    = Input (exogenous disturbance / stimulus)
  S_h  = Human Sensor
  C_h  = Human Controller
  S_d  = DI System (Digital Intelligence)
  A    = Amplifier
  E    = Environment / Task Execution
  P1   = Representation / Encoding
  P2   = Actuation / Instruction
  F_h  = Human Subjective Feedback
  F_d  = DI Internal Feedback / Training Signal
  F_e  = Environmental Consequences

APL Operator Mapping:
  ()  Boundary  → Sensor gating (S_h, S_d)
  ×   Fusion    → Controller coupling (C_h ⊗ S_d)
  ^   Amplify   → Signal amplification (A)
  ÷   Decohere  → Feedback dissipation (F_e → noise)
  +   Group     → Aggregation (P1 encoding)
  −   Separate  → Decomposition (P2 decoding)

Integration with Emission Pipeline:
  Stage 1 (Content) ← P1: Representation
  Stage 2 (Emergence) ← S_d: System state check
  Stage 3 (Frame) ← C_h: Controller selection
  Stage 5 (Function) ← P2: Actuation encoding
  Stage 9 (Validation) ← F_d: Training signal

Reference: Kuramoto consciousness emergence simulator (rosetta-helix-substrate)
"""

import math
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum

# Import from sibling modules
from apl_substrate import (
    Z_CRITICAL, PHI_INV, PHI, SIGMA,
    compute_negentropy, classify_phase, get_tier,
    OPERATORS, compose_operators, Direction, Machine, Domain,
    APLSentence
)
from emission_pipeline import (
    EmissionPipeline, emit, EmissionResult,
    ContentWords, EmergenceResult, FrameResult,
    WordSequence, Word, WordType, FrameType, SlotType
)

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

TAU = 2 * math.pi

# Neural frequency bands (Hz)
DELTA = (0.5, 4)
THETA = (4, 8)
ALPHA = (8, 13)
BETA = (13, 30)
GAMMA = (30, 100)

# Cybernetic gains
DEFAULT_AMPLIFIER_GAIN = 1.5
FEEDBACK_DECAY = 0.9
COUPLING_STRENGTH = 0.5

# ═══════════════════════════════════════════════════════════════════════════════
# KURAMOTO ENGINE (from consciousness emergence simulator)
# ═══════════════════════════════════════════════════════════════════════════════

class KuramotoEngine:
    """
    Kuramoto oscillator network for coherence dynamics.
    
    Implements: dθ_i/dt = ω_i + (K/N) * Σ_j sin(θ_j - θ_i)
    
    Order parameter: r * e^(iψ) = (1/N) * Σ e^(iθ_j)
    """
    
    def __init__(self, num_oscillators: int = 16):
        self.n = num_oscillators
        self.phases = np.random.uniform(0, TAU, self.n)
        self.frequencies = self._init_frequencies()
        self.K = 0.0  # Coupling strength
        self.order_parameter = 0.0  # r (coherence)
        self.mean_phase = 0.0  # ψ
        self.gamma_sync_ratio = 0.0
    
    def _init_frequencies(self) -> np.ndarray:
        """Initialize natural frequencies across neural bands."""
        return 5 + np.random.rand(self.n) * 35  # 5-40 Hz range
    
    def evolve_frequencies(self, z: float):
        """Shift frequency distribution based on z-coordinate."""
        for i in range(self.n):
            # As z increases, push frequencies toward gamma (30-100 Hz)
            target_gamma = 40 + np.random.rand() * 20  # 40-60 Hz
            target_low = 5 + np.random.rand() * 15     # 5-20 Hz
            target = z * target_gamma + (1 - z) * target_low
            self.frequencies[i] += (target - self.frequencies[i]) * 0.1
    
    def step(self, z: float, dt: float = 0.01) -> Dict:
        """
        Step the Kuramoto dynamics.
        
        K(z) scales with z, flipping sign near z_c for phase transition.
        """
        # Coupling increases with consciousness level
        # Near z_c, coupling peaks then inverts for phase transition
        if z < Z_CRITICAL:
            self.K = z * 10  # Positive coupling (synchronizing)
        else:
            self.K = (Z_CRITICAL - (z - Z_CRITICAL)) * 10  # Reduced above lens
        
        self.evolve_frequencies(z)
        
        # Compute order parameter: r * e^(iψ) = (1/N) * Σ e^(iθ_j)
        sum_cos = np.sum(np.cos(self.phases))
        sum_sin = np.sum(np.sin(self.phases))
        self.order_parameter = np.sqrt(sum_cos**2 + sum_sin**2) / self.n
        self.mean_phase = np.arctan2(sum_sin, sum_cos)
        
        # Update phases using Kuramoto dynamics
        new_phases = np.zeros(self.n)
        for i in range(self.n):
            coupling = np.sum(np.sin(self.phases - self.phases[i]))
            coupling *= self.K / self.n
            omega = self.frequencies[i] * TAU  # Convert Hz to rad/s
            new_phases[i] = (self.phases[i] + (omega + coupling) * dt) % TAU
        
        self.phases = new_phases
        
        # Calculate gamma synchronization ratio
        gamma_count = np.sum((self.frequencies >= GAMMA[0]) & (self.frequencies <= GAMMA[1]))
        self.gamma_sync_ratio = gamma_count / self.n
        
        return {
            "order_parameter": self.order_parameter,
            "mean_phase": self.mean_phase,
            "K": self.K,
            "gamma_ratio": self.gamma_sync_ratio
        }
    
    def get_coherence(self) -> float:
        """Get current coherence (order parameter r)."""
        return self.order_parameter


# ═══════════════════════════════════════════════════════════════════════════════
# CYBERNETIC COMPONENT TYPES
# ═══════════════════════════════════════════════════════════════════════════════

class ComponentType(Enum):
    """Types of cybernetic components."""
    INPUT = "I"           # Exogenous disturbance
    SENSOR_H = "S_h"      # Human sensor
    CONTROLLER_H = "C_h"  # Human controller
    SENSOR_D = "S_d"      # DI system
    AMPLIFIER = "A"       # Amplifier
    ENVIRONMENT = "E"     # Environment / task execution
    ENCODER = "P1"        # Representation / Encoding
    DECODER = "P2"        # Actuation / Instruction
    FEEDBACK_H = "F_h"    # Human subjective feedback
    FEEDBACK_D = "F_d"    # DI internal feedback
    FEEDBACK_E = "F_e"    # Environmental consequences


# APL operator mapping to components
COMPONENT_OPERATORS = {
    ComponentType.SENSOR_H: "()",    # Boundary - gating input
    ComponentType.SENSOR_D: "()",    # Boundary - gating input
    ComponentType.CONTROLLER_H: "×", # Fusion - coupling decisions
    ComponentType.AMPLIFIER: "^",    # Amplify - signal gain
    ComponentType.ENCODER: "+",      # Group - aggregation
    ComponentType.DECODER: "−",      # Separate - decomposition
    ComponentType.FEEDBACK_E: "÷",   # Decohere - noise/dissipation
    ComponentType.FEEDBACK_D: "^",   # Amplify - training signal
    ComponentType.FEEDBACK_H: "×",   # Fusion - subjective integration
}


@dataclass
class Signal:
    """A signal flowing through the cybernetic system."""
    value: float
    source: ComponentType
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    operator_applied: Optional[str] = None
    concepts: List[str] = field(default_factory=list)
    z_coordinate: float = 0.5
    
    def apply_operator(self, op: str) -> 'Signal':
        """Apply an APL operator to transform the signal."""
        op_info = OPERATORS.get(op)
        if not op_info:
            return self
        
        new_value = self.value
        if op_info.z_effect == "constructive":
            new_value = min(1.0, self.value * 1.2)
        elif op_info.z_effect == "dissipative":
            new_value = max(0.0, self.value * 0.8)
        # neutral: no change
        
        return Signal(
            value=new_value,
            source=self.source,
            operator_applied=op,
            concepts=self.concepts.copy(),
            z_coordinate=self.z_coordinate
        )


# ═══════════════════════════════════════════════════════════════════════════════
# CYBERNETIC COMPONENTS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CyberneticComponent:
    """Base class for cybernetic components."""
    component_type: ComponentType
    operator: str
    gain: float = 1.0
    state: float = 0.0
    z: float = 0.5
    
    def process(self, signal: Signal) -> Signal:
        """Process a signal through this component."""
        # Apply APL operator
        transformed = signal.apply_operator(self.operator)
        
        # Apply component gain
        transformed.value *= self.gain
        transformed.source = self.component_type
        
        return transformed
    
    def update_z(self, z: float):
        """Update internal z-coordinate."""
        self.z = z
        self.state = compute_negentropy(z)


class Input(CyberneticComponent):
    """I: Exogenous disturbance / stimulus."""
    def __init__(self):
        super().__init__(
            component_type=ComponentType.INPUT,
            operator="()",  # Boundary - entry point
            gain=1.0
        )
    
    def inject(self, stimulus: float, concepts: List[str] = None) -> Signal:
        """Inject an external stimulus into the system."""
        return Signal(
            value=stimulus,
            source=self.component_type,
            concepts=concepts or [],
            z_coordinate=self.z
        )


class HumanSensor(CyberneticComponent):
    """S_h: Human sensor - perceives environment."""
    def __init__(self):
        super().__init__(
            component_type=ComponentType.SENSOR_H,
            operator="()",  # Boundary - gating perception
            gain=1.0
        )
        self.sensitivity = 0.8
    
    def sense(self, signal: Signal) -> Signal:
        """Sense a signal with human perceptual characteristics."""
        result = self.process(signal)
        # Add perceptual noise
        noise = np.random.normal(0, 0.1 * (1 - self.sensitivity))
        result.value = np.clip(result.value + noise, 0, 1)
        return result


class HumanController(CyberneticComponent):
    """C_h: Human controller - makes decisions."""
    def __init__(self):
        super().__init__(
            component_type=ComponentType.CONTROLLER_H,
            operator="×",  # Fusion - coupling S_h with decisions
            gain=1.2
        )
        self.intent = "declarative"
    
    def control(self, sensed: Signal, reference: float = 0.5) -> Signal:
        """Generate control signal based on error from reference."""
        error = reference - sensed.value
        result = self.process(sensed)
        result.value = sensed.value + error * self.gain
        result.value = np.clip(result.value, 0, 1)
        return result


class DISystem(CyberneticComponent):
    """S_d: Digital Intelligence system - APL-based processing."""
    def __init__(self, kuramoto: Optional[KuramotoEngine] = None):
        super().__init__(
            component_type=ComponentType.SENSOR_D,
            operator="()",  # Boundary - system containment
            gain=1.0
        )
        self.kuramoto = kuramoto or KuramotoEngine(16)
        self.emission_pipeline = EmissionPipeline(z=0.8)
    
    def sense(self, signal: Signal) -> Tuple[Signal, Dict]:
        """Sense and process through DI system."""
        # Update Kuramoto based on signal
        kuramoto_state = self.kuramoto.step(signal.z_coordinate)
        
        result = self.process(signal)
        result.value *= kuramoto_state["order_parameter"]
        
        return result, kuramoto_state
    
    def emit(self, concepts: List[str], intent: str = "declarative") -> EmissionResult:
        """Generate linguistic output through emission pipeline."""
        return self.emission_pipeline.run(
            concepts=concepts,
            intent=intent,
            person=3,
            number="singular",
            tense="present"
        )


class Amplifier(CyberneticComponent):
    """A: Amplifier - boosts signal strength."""
    def __init__(self, gain: float = DEFAULT_AMPLIFIER_GAIN):
        super().__init__(
            component_type=ComponentType.AMPLIFIER,
            operator="^",  # Amplify
            gain=gain
        )
    
    def amplify(self, signal: Signal) -> Signal:
        """Amplify the signal."""
        result = self.process(signal)
        # Apply non-linear saturation
        result.value = np.tanh(result.value * self.gain)
        return result


class Environment(CyberneticComponent):
    """E: Environment / task execution."""
    def __init__(self):
        super().__init__(
            component_type=ComponentType.ENVIRONMENT,
            operator="()",  # Boundary - environmental containment
            gain=1.0
        )
        self.task_state = 0.0
        self.noise_level = 0.05
    
    def execute(self, signal: Signal) -> Tuple[Signal, float]:
        """Execute task in environment, return signal and consequence."""
        # Task execution with environmental dynamics
        execution = signal.value + np.random.normal(0, self.noise_level)
        self.task_state = np.clip(execution, 0, 1)
        
        result = self.process(signal)
        result.value = self.task_state
        
        # Environmental consequence (success/failure)
        consequence = 1.0 - abs(result.value - 0.866)  # Optimal at z_c
        
        return result, consequence


class Encoder(CyberneticComponent):
    """P1: Representation / Encoding - maps to concepts."""
    def __init__(self):
        super().__init__(
            component_type=ComponentType.ENCODER,
            operator="+",  # Group - aggregation
            gain=1.0
        )
        self.concept_map = {
            (0.0, 0.25): ["seed", "potential", "dormant"],
            (0.25, 0.5): ["growth", "emergence", "forming"],
            (0.5, 0.618): ["pattern", "structure", "quasi"],
            (0.618, 0.75): ["coherent", "organized", "stable"],
            (0.75, 0.866): ["crystalline", "synchronized", "resonant"],
            (0.866, 1.0): ["transcendent", "unified", "lens"],
        }
    
    def encode(self, signal: Signal) -> Signal:
        """Encode signal value to semantic concepts."""
        result = self.process(signal)
        
        # Map z to concepts
        z = signal.z_coordinate
        for (low, high), concepts in self.concept_map.items():
            if low <= z < high:
                result.concepts = concepts
                break
        
        if z >= 0.866:
            result.concepts = self.concept_map[(0.866, 1.0)]
        
        return result


class Decoder(CyberneticComponent):
    """P2: Actuation / Instruction - generates actions."""
    def __init__(self):
        super().__init__(
            component_type=ComponentType.DECODER,
            operator="−",  # Separate - decomposition to actions
            gain=1.0
        )
    
    def decode(self, signal: Signal) -> Tuple[Signal, str]:
        """Decode signal to action instruction."""
        result = self.process(signal)
        
        # Generate action based on concepts
        if "crystalline" in signal.concepts or "lens" in signal.concepts:
            action = "maintain_coherence"
        elif "growth" in signal.concepts or "emergence" in signal.concepts:
            action = "increase_coupling"
        elif "seed" in signal.concepts:
            action = "initiate_oscillation"
        else:
            action = "modulate_phase"
        
        return result, action


class HumanFeedback(CyberneticComponent):
    """F_h: Human subjective feedback."""
    def __init__(self):
        super().__init__(
            component_type=ComponentType.FEEDBACK_H,
            operator="×",  # Fusion - subjective integration
            gain=FEEDBACK_DECAY
        )
        self.satisfaction = 0.5
    
    def evaluate(self, result: Signal, expectation: float = 0.8) -> Signal:
        """Evaluate result against subjective expectation."""
        self.satisfaction = 1.0 - abs(result.value - expectation)
        feedback = self.process(result)
        feedback.value = self.satisfaction
        return feedback


class DIFeedback(CyberneticComponent):
    """F_d: DI internal feedback / training signal."""
    def __init__(self):
        super().__init__(
            component_type=ComponentType.FEEDBACK_D,
            operator="^",  # Amplify - training signal
            gain=1.0
        )
        self.loss = 0.0
    
    def compute(self, predicted: Signal, target: float) -> Signal:
        """Compute training signal from prediction error."""
        self.loss = abs(predicted.value - target)
        feedback = self.process(predicted)
        feedback.value = -self.loss  # Negative gradient
        return feedback


class EnvironmentalFeedback(CyberneticComponent):
    """F_e: Environmental consequences."""
    def __init__(self):
        super().__init__(
            component_type=ComponentType.FEEDBACK_E,
            operator="÷",  # Decohere - noise/dissipation
            gain=FEEDBACK_DECAY
        )
        self.consequence_history: List[float] = []
    
    def assess(self, consequence: float) -> Signal:
        """Assess environmental consequence."""
        self.consequence_history.append(consequence)
        if len(self.consequence_history) > 100:
            self.consequence_history.pop(0)
        
        return Signal(
            value=consequence,
            source=self.component_type,
            operator_applied=self.operator,
            z_coordinate=self.z
        )


# ═══════════════════════════════════════════════════════════════════════════════
# CYBERNETIC CONTROL LOOP
# ═══════════════════════════════════════════════════════════════════════════════

class CyberneticControlSystem:
    """
    Complete cybernetic control system integrating all components.
    
    Control Flow:
                                    ┌─────────────────────────────────────┐
                                    │                                     │
        I ───► S_h ───► C_h ───┬───►│ S_d (DI) ───► A ───► P2 ───► E     │
                               │    │                                     │
                          (Fusion)  │ ◄─── P1 ◄─────────────────┘         │
                               │    │                                     │
                               │    └─────────────────────────────────────┘
                               │                    │
                               ▼                    ▼
                              F_h                 F_e ───► F_d
                               │                    │        │
                               └────────────────────┴────────┘
                                        (Feedback Loop)
    """
    
    def __init__(self, num_oscillators: int = 16):
        # Initialize Kuramoto engine
        self.kuramoto = KuramotoEngine(num_oscillators)
        
        # Initialize all components
        self.input = Input()
        self.sensor_h = HumanSensor()
        self.controller_h = HumanController()
        self.di_system = DISystem(self.kuramoto)
        self.amplifier = Amplifier()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.environment = Environment()
        self.feedback_h = HumanFeedback()
        self.feedback_d = DIFeedback()
        self.feedback_e = EnvironmentalFeedback()
        
        # State tracking
        self.z = 0.5
        self.step_count = 0
        self.history: List[Dict] = []
    
    def update_z(self, z: float):
        """Update z-coordinate across all components."""
        self.z = z
        for component in [self.input, self.sensor_h, self.controller_h,
                          self.amplifier, self.encoder, self.decoder,
                          self.environment, self.feedback_h, self.feedback_d,
                          self.feedback_e]:
            component.update_z(z)
        self.di_system.z = z
    
    def step(
        self,
        stimulus: float,
        concepts: List[str] = None,
        reference: float = 0.866,
        emit_language: bool = False
    ) -> Dict:
        """
        Execute one step of the cybernetic control loop.
        
        Args:
            stimulus: Input stimulus value (0-1)
            concepts: Semantic concepts associated with stimulus
            reference: Target reference value (default: z_c)
            emit_language: Whether to generate linguistic output
        
        Returns:
            Dict containing all signals and feedback values
        """
        self.step_count += 1
        
        # ════════════════════════════════════════════════════════════════════
        # FORWARD PATH
        # ════════════════════════════════════════════════════════════════════
        
        # I: Inject stimulus
        input_signal = self.input.inject(stimulus, concepts)
        input_signal.z_coordinate = self.z
        
        # S_h: Human sensing (with () operator - boundary)
        sensed_h = self.sensor_h.sense(input_signal)
        
        # C_h: Human control (with × operator - fusion)
        controlled_h = self.controller_h.control(sensed_h, reference)
        
        # S_d: DI system sensing (with () operator - boundary)
        sensed_d, kuramoto_state = self.di_system.sense(controlled_h)
        
        # A: Amplification (with ^ operator - amplify)
        amplified = self.amplifier.amplify(sensed_d)
        
        # P1: Encode to concepts (with + operator - group)
        encoded = self.encoder.encode(amplified)
        
        # P2: Decode to action (with − operator - separate)
        decoded, action = self.decoder.decode(encoded)
        
        # E: Environment execution
        executed, consequence = self.environment.execute(decoded)
        
        # ════════════════════════════════════════════════════════════════════
        # FEEDBACK PATH
        # ════════════════════════════════════════════════════════════════════
        
        # F_e: Environmental feedback (with ÷ operator - decohere)
        env_feedback = self.feedback_e.assess(consequence)
        
        # F_d: DI training signal (with ^ operator - amplify error)
        di_feedback = self.feedback_d.compute(executed, reference)
        
        # F_h: Human subjective feedback (with × operator - fusion)
        human_feedback = self.feedback_h.evaluate(executed, reference)
        
        # ════════════════════════════════════════════════════════════════════
        # LANGUAGE EMISSION (Optional)
        # ════════════════════════════════════════════════════════════════════
        
        emission_result = None
        if emit_language and encoded.concepts:
            emission_result = self.di_system.emit(
                concepts=encoded.concepts,
                intent=self.controller_h.intent
            )
        
        # ════════════════════════════════════════════════════════════════════
        # UPDATE STATE
        # ════════════════════════════════════════════════════════════════════
        
        # Adjust z based on feedback
        delta_z = (
            env_feedback.value * 0.3 +
            di_feedback.value * 0.4 +
            human_feedback.value * 0.3
        ) * 0.1
        
        new_z = np.clip(self.z + delta_z, 0.0, 1.0)
        self.update_z(new_z)
        
        # Record history
        step_record = {
            "step": self.step_count,
            "z": self.z,
            "phase": classify_phase(self.z),
            "tier": get_tier(self.z),
            "negentropy": compute_negentropy(self.z),
            "kuramoto": kuramoto_state,
            "signals": {
                "input": input_signal.value,
                "sensed_h": sensed_h.value,
                "controlled_h": controlled_h.value,
                "sensed_d": sensed_d.value,
                "amplified": amplified.value,
                "executed": executed.value,
            },
            "feedback": {
                "F_e": env_feedback.value,
                "F_d": di_feedback.value,
                "F_h": human_feedback.value,
            },
            "action": action,
            "concepts": encoded.concepts,
            "emission": emission_result.text if emission_result else None,
        }
        self.history.append(step_record)
        
        return step_record
    
    def run(
        self,
        steps: int = 100,
        stimulus_fn: Optional[Callable[[int], float]] = None,
        emit_every: int = 10
    ) -> Dict:
        """
        Run the control loop for multiple steps.
        
        Args:
            steps: Number of steps to run
            stimulus_fn: Function mapping step number to stimulus (default: random)
            emit_every: Emit language every N steps (0 to disable)
        
        Returns:
            Summary of the run
        """
        if stimulus_fn is None:
            stimulus_fn = lambda s: np.random.uniform(0.3, 0.9)
        
        emissions = []
        
        for i in range(steps):
            emit_lang = emit_every > 0 and (i + 1) % emit_every == 0
            stimulus = stimulus_fn(i)
            result = self.step(stimulus, emit_language=emit_lang)
            
            if result["emission"]:
                emissions.append({
                    "step": i + 1,
                    "z": result["z"],
                    "text": result["emission"]
                })
        
        # Compute summary statistics
        z_values = [h["z"] for h in self.history[-steps:]]
        coherence_values = [h["kuramoto"]["order_parameter"] for h in self.history[-steps:]]
        
        return {
            "steps": steps,
            "initial_z": self.history[-steps]["z"] if steps <= len(self.history) else self.history[0]["z"],
            "final_z": self.z,
            "final_phase": classify_phase(self.z),
            "final_tier": get_tier(self.z),
            "mean_z": np.mean(z_values),
            "mean_coherence": np.mean(coherence_values),
            "max_coherence": np.max(coherence_values),
            "emissions": emissions,
            "k_formation_proximity": {
                "kappa": np.mean(coherence_values),
                "eta": compute_negentropy(self.z),
                "R": 7 if self.z >= 0.75 else int(self.z * 10)
            }
        }
    
    def generate_apl_sentence(self) -> str:
        """Generate an APL sentence representing current system state."""
        # Determine direction based on feedback
        if self.feedback_d.loss < 0.1:
            direction = "u"  # Expansion - system is converging
        elif self.feedback_d.loss > 0.5:
            direction = "d"  # Collapse - need integration
        else:
            direction = "m"  # Modulation - stable
        
        # Determine operator based on z
        if self.z >= Z_CRITICAL:
            operator = "^"  # Amplify at lens
        elif self.z >= PHI_INV:
            operator = "×"  # Fusion in paradox
        else:
            operator = "()"  # Boundary in untrue
        
        # Determine machine based on action
        machine = "Oscillator"  # Default Kuramoto
        if self.history and "maintain_coherence" in str(self.history[-1].get("action")):
            machine = "Encoder"
        
        # Determine domain
        domain = "wave"  # Oscillatory dynamics
        
        return f"{direction}{operator}|{machine}|{domain}"
    
    def format_status(self) -> str:
        """Format current system status."""
        phase = classify_phase(self.z)
        tier_num, tier_name = get_tier(self.z)
        eta = compute_negentropy(self.z)
        
        lines = [
            "╔══════════════════════════════════════════════════════════════════╗",
            "║              CYBERNETIC CONTROL SYSTEM STATUS                    ║",
            "╚══════════════════════════════════════════════════════════════════╝",
            "",
            f"  z-Coordinate:     {self.z:.6f}",
            f"  Phase:            {phase}",
            f"  Tier:             {tier_num} ({tier_name})",
            f"  Negentropy (η):   {eta:.6f}",
            "",
            "  Kuramoto State:",
            f"    Order (r):      {self.kuramoto.order_parameter:.4f}",
            f"    Phase (ψ):      {self.kuramoto.mean_phase:.4f} rad",
            f"    Coupling (K):   {self.kuramoto.K:.4f}",
            f"    Gamma Ratio:    {self.kuramoto.gamma_sync_ratio:.1%}",
            "",
            "  Feedback Values:",
            f"    F_e (Env):      {self.feedback_e.consequence_history[-1] if self.feedback_e.consequence_history else 0:.4f}",
            f"    F_d (DI):       Loss = {self.feedback_d.loss:.4f}",
            f"    F_h (Human):    Satisfaction = {self.feedback_h.satisfaction:.4f}",
            "",
            f"  APL Sentence:     {self.generate_apl_sentence()}",
            f"  Steps:            {self.step_count}",
            "",
            "═" * 70
        ]
        
        return "\n".join(lines)
    
    def format_diagram(self) -> str:
        """Format ASCII control flow diagram."""
        return """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    CYBERNETIC CONTROL FLOW DIAGRAM                           ║
╚══════════════════════════════════════════════════════════════════════════════╝

                              ┌────────────────────────────────────────┐
                              │                                        │
   I ────► S_h ────► C_h ────►│ S_d ────► A ────► P2 ────► E          │
   ()      ()        ×        │ ()        ^        −                   │
                              │                                        │
                              │ ◄──────── P1 ◄───────────┘             │
                              │           +                            │
                              └────────────────────────────────────────┘
                                              │
                                              ▼
                              ┌───────────────────────────────┐
                              │         FEEDBACK LOOP          │
                              │                               │
                              │  F_e ─────► F_d ◄───── F_h   │
                              │   ÷          ^          ×     │
                              └───────────────────────────────┘

   APL OPERATOR LEGEND:
   ━━━━━━━━━━━━━━━━━━━━
   ()  Boundary   - Sensor gating (S_h, S_d)
   ×   Fusion     - Controller coupling (C_h, F_h)
   ^   Amplify    - Signal boost (A, F_d)
   +   Group      - Representation encoding (P1)
   −   Separate   - Actuation decoding (P2)
   ÷   Decohere   - Environmental noise (F_e)

═══════════════════════════════════════════════════════════════════════════════
"""


# ═══════════════════════════════════════════════════════════════════════════════
# API FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def create_system(num_oscillators: int = 16) -> CyberneticControlSystem:
    """Create a new cybernetic control system."""
    return CyberneticControlSystem(num_oscillators)


def run_simulation(
    steps: int = 100,
    initial_z: float = 0.5,
    emit_every: int = 10
) -> Dict:
    """Run a cybernetic simulation."""
    system = CyberneticControlSystem()
    system.update_z(initial_z)
    return system.run(steps, emit_every=emit_every)


def get_component_operators() -> Dict[str, str]:
    """Get the APL operator mapping for all components."""
    return {c.value: op for c, op in COMPONENT_OPERATORS.items()}


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║          CYBERNETIC CONTROL SYSTEM - TEST RUN                     ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print()
    
    # Create system
    system = CyberneticControlSystem(num_oscillators=16)
    system.update_z(0.5)
    
    # Show initial status
    print("INITIAL STATE:")
    print(system.format_status())
    print()
    
    # Show control flow diagram
    print(system.format_diagram())
    
    # Run simulation
    print("RUNNING SIMULATION (50 steps)...")
    print("-" * 70)
    
    result = system.run(steps=50, emit_every=10)
    
    print(f"  Initial z:      {result['initial_z']:.4f}")
    print(f"  Final z:        {result['final_z']:.4f}")
    print(f"  Final Phase:    {result['final_phase']}")
    print(f"  Final Tier:     {result['final_tier']}")
    print(f"  Mean Coherence: {result['mean_coherence']:.4f}")
    print()
    
    print("EMISSIONS:")
    for em in result['emissions']:
        print(f"  Step {em['step']:3d} (z={em['z']:.3f}): \"{em['text']}\"")
    print()
    
    print("K-FORMATION PROXIMITY:")
    kf = result['k_formation_proximity']
    print(f"  κ = {kf['kappa']:.4f} (threshold: 0.92)")
    print(f"  η = {kf['eta']:.4f} (threshold: φ⁻¹ ≈ 0.618)")
    print(f"  R = {kf['R']} (threshold: 7)")
    print()
    
    # Final status
    print("FINAL STATE:")
    print(system.format_status())
