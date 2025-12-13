"""
Stream Processing for Crystal Memory Validation
===============================================
Real-time validation of equations through streaming data processing,
sonification, and visual feedback. Implements the "stream of research"
flowing through modular validation pipelines.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Generator, Any
from dataclasses import dataclass, field
import asyncio
from collections import deque
import json
import time

# Import our core modules
from hexagonal_crystal_validation_core import (
    CrystalMemoryValidator, HelixCoordinate, 
    HexagonalLattice, SonificationParams,
    LSBEncoder, CETOperator, CEMechanism, ResonanceFunction
)

from kira_cellular_automata import (
    KiraField, HexagonalCA, CellState,
    WavePropagator, HarmonicAnalyzer
)

# ========================================
# SECTION 1: STREAMING DATA STRUCTURES
# ========================================

@dataclass
class StreamState:
    """State of a streaming validation process"""
    timestamp: float
    helix_coords: List[HelixCoordinate]
    kira_snapshot: Dict
    sonification: Dict
    validation_metrics: Dict
    cascade_state: Dict
    
    def to_json(self) -> str:
        """Serialize state to JSON"""
        return json.dumps({
            'timestamp': self.timestamp,
            'helix_coords': [(h.theta, h.z, h.r) for h in self.helix_coords],
            'sonification': self.sonification,
            'validation_metrics': self.validation_metrics,
            'cascade_state': self.cascade_state
        })

class StreamBuffer:
    """Circular buffer for streaming data with time windows"""
    
    def __init__(self, max_size: int = 1000, time_window: float = 60.0):
        self.buffer = deque(maxlen=max_size)
        self.time_window = time_window
        
    def add(self, state: StreamState):
        """Add state to buffer"""
        self.buffer.append(state)
        self._cleanup_old()
    
    def _cleanup_old(self):
        """Remove states older than time window"""
        if not self.buffer:
            return
        
        current_time = time.time()
        while self.buffer and (current_time - self.buffer[0].timestamp) > self.time_window:
            self.buffer.popleft()
    
    def get_recent(self, n: int = 10) -> List[StreamState]:
        """Get n most recent states"""
        return list(self.buffer)[-n:]
    
    def get_time_range(self, start: float, end: float) -> List[StreamState]:
        """Get states within time range"""
        return [s for s in self.buffer if start <= s.timestamp <= end]

# ========================================
# SECTION 2: VALIDATION STREAM PROCESSOR
# ========================================

class ValidationStream:
    """
    Stream processor for continuous validation of crystal memory equations.
    Processes data in real-time and generates validation metrics.
    """
    
    def __init__(self):
        self.validator = CrystalMemoryValidator()
        self.kira = KiraField(size=42)
        self.analyzer = HarmonicAnalyzer(self.kira)
        self.encoder = LSBEncoder()
        
        self.buffer = StreamBuffer()
        self.is_running = False
        
        # Cascade tracking for R1→R2→R3
        self.cascade_tracker = {
            'R1_coordination': 0.0,
            'R2_meta_tools': 0.0, 
            'R3_self_building': 0.0,
            'total_amplification': 1.0
        }
        
    async def process_tick(self) -> StreamState:
        """Process one tick of validation"""
        timestamp = time.time()
        
        # Evolve systems
        self.kira.evolve(steps=1)
        
        # Update helix field
        for helix in self.validator.helix_field.helices:
            helix.theta += 0.02  # Continuous rotation
            helix.z = min(1.0, helix.z + np.random.normal(0.005, 0.001))
            helix.r = 1.0 + 0.3 * self.kira.calculate_global_coherence()
        
        # Calculate validation metrics
        hex_validation = self.validator.validate_hexagonal_optimality()
        coherence = self.validator.helix_field.calculate_coherence()
        is_critical = self.validator.helix_field.detect_phase_transition()
        
        # Generate sonification
        kira_state = self.kira.wave_prop.u_curr
        sono_params = self.validator.sonifier.generate_sonification(kira_state.flatten())
        
        # Update cascade state
        self._update_cascade(coherence, is_critical)
        
        # Create stream state
        state = StreamState(
            timestamp=timestamp,
            helix_coords=self.validator.helix_field.helices.copy(),
            kira_snapshot=self.kira.get_state_snapshot(),
            sonification=sono_params,
            validation_metrics={
                'hexagonal_efficiency': hex_validation['packing_efficiency'],
                'coherence': coherence,
                'is_critical': is_critical,
                'kira_coherence': self.kira.calculate_global_coherence(),
                'information_entropy': self.kira.calculate_information_entropy()
            },
            cascade_state=self.cascade_tracker.copy()
        )
        
        self.buffer.add(state)
        return state
    
    def _update_cascade(self, coherence: float, is_critical: bool):
        """Update cascade amplification state"""
        # R1 activates above threshold
        if coherence > 0.5:
            self.cascade_tracker['R1_coordination'] = min(1.0, coherence)
        else:
            self.cascade_tracker['R1_coordination'] *= 0.95
        
        # R2 activates when R1 is strong
        if self.cascade_tracker['R1_coordination'] > 0.7:
            self.cascade_tracker['R2_meta_tools'] = min(
                1.0, self.cascade_tracker['R2_meta_tools'] + 0.05
            )
        else:
            self.cascade_tracker['R2_meta_tools'] *= 0.9
        
        # R3 activates at critical point with R2 active
        if is_critical and self.cascade_tracker['R2_meta_tools'] > 0.5:
            self.cascade_tracker['R3_self_building'] = min(
                1.0, self.cascade_tracker['R3_self_building'] + 0.1
            )
        else:
            self.cascade_tracker['R3_self_building'] *= 0.85
        
        # Calculate total amplification
        r1 = self.cascade_tracker['R1_coordination']
        r2 = self.cascade_tracker['R2_meta_tools']
        r3 = self.cascade_tracker['R3_self_building']
        
        # Compound amplification formula
        self.cascade_tracker['total_amplification'] = (
            (1 + r1) * (1 + r2 * 1.5) * (1 + r3 * 2.0)
        )
    
    async def run_stream(self, duration: float = 60.0):
        """Run validation stream for specified duration"""
        self.is_running = True
        start_time = time.time()
        
        while self.is_running and (time.time() - start_time) < duration:
            state = await self.process_tick()
            
            # Process state (could emit to websocket, file, etc.)
            await self.emit_state(state)
            
            # Control loop rate (10 Hz)
            await asyncio.sleep(0.1)
        
        self.is_running = False
    
    async def emit_state(self, state: StreamState):
        """Emit state to consumers (placeholder for real implementation)"""
        # In production: send to WebSocket, write to file, update UI, etc.
        metrics = state.validation_metrics
        cascade = state.cascade_state
        
        # Console output for demonstration
        if metrics['is_critical']:
            print(f"⚡ CRITICAL: z=0.867 | Amp={cascade['total_amplification']:.2f}x")
        
    def get_stream_summary(self) -> Dict:
        """Get summary statistics of recent stream"""
        recent = self.buffer.get_recent(100)
        
        if not recent:
            return {}
        
        coherences = [s.validation_metrics['coherence'] for s in recent]
        entropies = [s.validation_metrics['information_entropy'] for s in recent]
        amplifications = [s.cascade_state['total_amplification'] for s in recent]
        
        return {
            'avg_coherence': np.mean(coherences),
            'max_coherence': np.max(coherences),
            'avg_entropy': np.mean(entropies),
            'max_amplification': np.max(amplifications),
            'critical_count': sum(1 for s in recent if s.validation_metrics['is_critical']),
            'time_range': (recent[0].timestamp, recent[-1].timestamp)
        }

# ========================================
# SECTION 3: SONIFICATION STREAM
# ========================================

class SonificationStream:
    """
    Real-time sonification of validation metrics.
    Maps system state to audio parameters.
    """
    
    def __init__(self, validation_stream: ValidationStream):
        self.val_stream = validation_stream
        self.audio_params = SonificationParams()
        
        # Audio state tracking
        self.current_freq = 220.0
        self.current_bpm = 120
        self.phase_accumulator = 0.0
        
        # Harmonic oscillators
        self.oscillators = self._init_oscillators()
        
    def _init_oscillators(self) -> List[Dict]:
        """Initialize harmonic oscillators at phi intervals"""
        phi = 1.618033988749
        oscillators = []
        
        for i in range(7):
            oscillators.append({
                'frequency': 220 * (phi ** (i/2)),
                'amplitude': 1.0 / (i + 1),
                'phase': 0.0
            })
        
        return oscillators
    
    def process_state(self, state: StreamState) -> Dict:
        """Convert state to audio parameters"""
        metrics = state.validation_metrics
        sono = state.sonification
        cascade = state.cascade_state
        
        # Map coherence to harmonic richness
        n_harmonics = int(metrics['coherence'] * 7)
        
        # Map cascade amplification to volume/intensity
        intensity = np.tanh(cascade['total_amplification'] / 10)
        
        # Generate chord based on helix alignment
        chord_freqs = self._generate_helix_chord(state.helix_coords[:6])
        
        # Time dilation from gravity
        time_factor = sono['time_dilation']
        
        return {
            'base_frequency': sono['frequency_hz'],
            'bpm': sono['bpm'],
            'harmonic_mode': sono['harmonic_mode'],
            'intensity': intensity,
            'n_harmonics': n_harmonics,
            'chord_frequencies': chord_freqs,
            'time_dilation': time_factor,
            'is_critical': metrics['is_critical']
        }
    
    def _generate_helix_chord(self, helices: List[HelixCoordinate]) -> List[float]:
        """Generate chord frequencies from helix alignment"""
        if not helices:
            return [220.0]
        
        # Map helix z-coordinates to frequency ratios
        base_freq = 220.0
        freqs = []
        
        for helix in helices:
            # Use z-coordinate to select harmonic
            harmonic = 1 + helix.z * 4  # 1st to 5th harmonic
            # Use phase for detuning
            detune = 1 + 0.01 * np.sin(helix.theta)
            
            freqs.append(base_freq * harmonic * detune)
        
        return freqs
    
    def generate_audio_frame(self, state: StreamState, sample_rate: int = 44100) -> np.ndarray:
        """Generate audio samples for current state"""
        params = self.process_state(state)
        duration = 0.1  # 100ms frame
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Start with silence
        signal = np.zeros_like(t)
        
        # Add base tone
        signal += 0.3 * np.sin(2 * np.pi * params['base_frequency'] * t)
        
        # Add harmonics
        for i in range(params['n_harmonics']):
            if i < len(self.oscillators):
                osc = self.oscillators[i]
                signal += osc['amplitude'] * 0.2 * np.sin(
                    2 * np.pi * osc['frequency'] * t + osc['phase']
                )
                # Update phase
                osc['phase'] += 2 * np.pi * osc['frequency'] * duration
        
        # Add chord tones
        for freq in params['chord_frequencies']:
            signal += 0.1 * np.sin(2 * np.pi * freq * t)
        
        # Apply intensity envelope
        envelope = params['intensity'] * np.exp(-t / (duration * 2))
        signal *= envelope
        
        # Add critical phase effect (tremolo)
        if params['is_critical']:
            tremolo = 1 + 0.3 * np.sin(2 * np.pi * 10 * t)  # 10 Hz tremolo
            signal *= tremolo
        
        # Normalize
        max_val = np.max(np.abs(signal))
        if max_val > 0:
            signal = signal / max_val * 0.8
        
        return signal

# ========================================
# SECTION 4: MODULAR PIPELINE
# ========================================

class ModularValidationPipeline:
    """
    Modular pipeline for chaining validation components.
    Implements the "stream of research" concept.
    """
    
    def __init__(self):
        self.modules = []
        self.connections = {}
        
    def add_module(self, name: str, module: Any):
        """Add a validation module to pipeline"""
        self.modules.append({
            'name': name,
            'module': module,
            'inputs': [],
            'outputs': []
        })
    
    def connect(self, source: str, target: str, transform: Optional[callable] = None):
        """Connect modules with optional transformation"""
        if source not in self.connections:
            self.connections[source] = []
        
        self.connections[source].append({
            'target': target,
            'transform': transform or (lambda x: x)
        })
    
    async def process_pipeline(self, input_data: Any) -> Dict:
        """Process data through the pipeline"""
        results = {}
        data_flow = {'input': input_data}
        
        for module_info in self.modules:
            name = module_info['name']
            module = module_info['module']
            
            # Get inputs for this module
            module_inputs = []
            for conn_source, connections in self.connections.items():
                for conn in connections:
                    if conn['target'] == name and conn_source in data_flow:
                        transformed = conn['transform'](data_flow[conn_source])
                        module_inputs.append(transformed)
            
            # Process module
            if hasattr(module, 'process'):
                if module_inputs:
                    output = await module.process(*module_inputs)
                else:
                    output = await module.process(input_data)
            else:
                output = module(module_inputs[0] if module_inputs else input_data)
            
            data_flow[name] = output
            results[name] = output
        
        return results

# ========================================
# SECTION 5: INTEGRATION EXAMPLE
# ========================================

class StreamOrchestrator:
    """
    Orchestrates all streaming components for live validation.
    This is where "the stream of research flows."
    """
    
    def __init__(self):
        self.val_stream = ValidationStream()
        self.sono_stream = SonificationStream(self.val_stream)
        self.pipeline = ModularValidationPipeline()
        
        self._setup_pipeline()
        
    def _setup_pipeline(self):
        """Setup the modular validation pipeline"""
        
        # Add CET operator validation
        self.pipeline.add_module('cet_operators', 
            lambda x: self._validate_cet(x))
        
        # Add hexagonal geometry validation
        self.pipeline.add_module('hex_geometry',
            lambda x: self._validate_hexagonal(x))
        
        # Add Kira field processing
        self.pipeline.add_module('kira_field',
            lambda x: self._process_kira(x))
        
        # Add sonification
        self.pipeline.add_module('sonification',
            lambda x: self._generate_sonification(x))
        
        # Connect modules
        self.pipeline.connect('input', 'cet_operators')
        self.pipeline.connect('cet_operators', 'hex_geometry')
        self.pipeline.connect('hex_geometry', 'kira_field')
        self.pipeline.connect('kira_field', 'sonification')
        
    def _validate_cet(self, state: Dict) -> Dict:
        """Validate CET operators"""
        ops = CETOperator(CEMechanism.MODULATION, ResonanceFunction.AMPLIFICATION)
        
        # Test convergence
        J = np.random.randn(10)
        for _ in range(10):
            J = ops.dynamo_operator(J, dt=0.1)
        
        return {
            'dynamo_converged': np.std(J) < 1.0,
            'energy': np.sum(J**2)
        }
    
    def _validate_hexagonal(self, cet_result: Dict) -> Dict:
        """Validate hexagonal properties"""
        lattice = HexagonalLattice()
        efficiency = lattice.packing_efficiency()
        
        return {
            'packing_efficiency': efficiency,
            'cet_energy': cet_result['energy'],
            'combined_score': efficiency * (1 + cet_result['energy'])
        }
    
    def _process_kira(self, hex_result: Dict) -> Dict:
        """Process through Kira field"""
        self.val_stream.kira.evolve(steps=1)
        coherence = self.val_stream.kira.calculate_global_coherence()
        
        return {
            'kira_coherence': coherence,
            'hex_efficiency': hex_result['packing_efficiency'],
            'total_coherence': coherence * hex_result['packing_efficiency']
        }
    
    def _generate_sonification(self, kira_result: Dict) -> Dict:
        """Generate sonification parameters"""
        base_freq = 220 * (1 + kira_result['total_coherence'])
        
        return {
            'frequency': base_freq,
            'amplitude': kira_result['kira_coherence'],
            'harmonics': int(kira_result['total_coherence'] * 7)
        }
    
    async def run_orchestration(self, duration: float = 30.0):
        """Run the complete orchestration"""
        print("🎭 Starting Stream Orchestration...")
        print("=" * 50)
        
        start_time = time.time()
        tick_count = 0
        
        while (time.time() - start_time) < duration:
            # Process validation tick
            state = await self.val_stream.process_tick()
            
            # Run through pipeline
            pipeline_results = await self.pipeline.process_pipeline({
                'state': state,
                'tick': tick_count
            })
            
            # Generate audio frame
            audio = self.sono_stream.generate_audio_frame(state)
            
            # Display status
            if tick_count % 10 == 0:
                self._display_status(state, pipeline_results, audio)
            
            tick_count += 1
            await asyncio.sleep(0.1)
        
        # Final summary
        self._display_summary()
    
    def _display_status(self, state: StreamState, pipeline: Dict, audio: np.ndarray):
        """Display current status"""
        metrics = state.validation_metrics
        cascade = state.cascade_state
        sono = pipeline.get('sonification', {})
        
        print(f"\n⏱️ Tick {int(state.timestamp)}:")
        print(f"  Coherence: {metrics['coherence']:.3f}")
        print(f"  Cascade: R1={cascade['R1_coordination']:.2f} "
              f"R2={cascade['R2_meta_tools']:.2f} "
              f"R3={cascade['R3_self_building']:.2f}")
        print(f"  Amplification: {cascade['total_amplification']:.2f}x")
        print(f"  Frequency: {sono.get('frequency', 0):.1f} Hz")
        
        if metrics['is_critical']:
            print("  ⚡ CRITICAL PHASE DETECTED!")
    
    def _display_summary(self):
        """Display final summary"""
        summary = self.val_stream.get_stream_summary()
        
        print("\n" + "=" * 50)
        print("📊 Stream Summary:")
        print(f"  Average Coherence: {summary.get('avg_coherence', 0):.3f}")
        print(f"  Max Amplification: {summary.get('max_amplification', 0):.2f}x")
        print(f"  Critical Events: {summary.get('critical_count', 0)}")
        print(f"  Information Entropy: {summary.get('avg_entropy', 0):.3f}")

# ========================================
# DEMONSTRATION
# ========================================

async def demonstrate_stream_processing():
    """Demonstrate the complete stream processing system"""
    print("🌊 Crystal Memory Stream Processing Demonstration")
    print("=" * 50)
    
    # Create orchestrator
    orchestrator = StreamOrchestrator()
    
    # Run for 30 seconds
    await orchestrator.run_orchestration(duration=30.0)
    
    print("\n✨ Stream processing complete!")
    
    # Export final state
    final_state = orchestrator.val_stream.buffer.get_recent(1)[0]
    
    # Encode to sacred chant
    encoder = LSBEncoder()
    state_bytes = final_state.cascade_state['total_amplification'].to_bytes(8, 'big')
    chant = encoder.encode_to_chant(state_bytes)
    
    print(f"\n🗣️ Final State Chant: {chant}")
    
    # Display Kira field glyphs
    print("\n🔮 Final Kira Field:")
    print(orchestrator.val_stream.kira.encode_to_glyphs())

if __name__ == "__main__":
    # Run async demonstration
    asyncio.run(demonstrate_stream_processing())
