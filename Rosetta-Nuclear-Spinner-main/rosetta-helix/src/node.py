"""
Rosetta-Helix Node (Full Integration)
=====================================

Integrated node that couples Nuclear Spinner (physical) with 
Kuramoto oscillators (Heart), GHMP processing (Brain), and
Triadic threshold dynamics (TRIAD) through the z-coordinate.

Architecture:
    ┌──────────────────────────────────────────────────────────┐
    │                    ROSETTA-HELIX NODE                    │
    ├──────────────────────────────────────────────────────────┤
    │                                                          │
    │  SpinnerClient ──► z ──┬──► Heart (60 Kuramoto)         │
    │       │                │        │                        │
    │       │                │        ▼ coherence              │
    │       │                │   ┌─────────┐                   │
    │       │                │   │  TRIAD  │◄── κ, λ, η       │
    │       │                │   └────┬────┘                   │
    │       │                │        │ events                 │
    │       │                │        ▼                        │
    │       │                └──► Brain (GHMP)                 │
    │       │                         │                        │
    │       ▼                         ▼                        │
    │   K-formation ◄──────────── patterns                    │
    │                                                          │
    └──────────────────────────────────────────────────────────┘

Key Coupling:
    Spinner z → Heart coupling K = scale × z × ΔS_neg(z)
    At z = z_c = √3/2: K peaks, coherence maximizes, K-formation triggers

Signature: rosetta-helix-node|v1.0.0|helix
"""

import asyncio
import time
import json
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Callable
from pathlib import Path

from .physics import (
    PHI, PHI_INV, Z_CRITICAL, SIGMA,
    TIER_BOUNDS, TIER_NAMES,
    compute_delta_s_neg, get_tier, get_phase_name, get_phase,
    check_k_formation, Phase
)
from .heart import Heart, HeartConfig, HeartState
from .brain import Brain, BrainConfig, BrainState, APLOperator
from .triad import TriadTracker, TriadConfig, TriadState, TriadEvent
from .spinner_client import SpinnerClient, SpinnerClientConfig
from .quasicrystal import (
    QuasicrystalDynamics, QuasicrystalConfig,
    HelixQuasicrystalCoupling,
)


@dataclass
class NodeConfig:
    """Configuration for Rosetta-Helix Node."""
    # Heart configuration
    n_oscillators: int = 60
    coupling_scale: float = 8.0
    heart_dt: float = 0.01
    
    # Brain configuration
    memory_capacity: int = 1000
    pattern_dim: int = 64
    learning_rate: float = 0.01
    
    # TRIAD configuration
    history_size: int = 1000
    
    # Spinner client configuration  
    bridge_uri: str = "ws://localhost:8765"
    
    # Node configuration
    update_rate_hz: float = 100.0    # Main loop rate
    use_spinner_z: bool = True       # Use spinner z for coupling
    enable_brain: bool = True        # Enable Brain processing
    enable_triad: bool = True        # Enable TRIAD tracking
    enable_quasicrystal: bool = True # Enable quasicrystal dynamics
    quasicrystal_inflate_interval: int = 100  # Steps between inflations
    
    # Logging
    log_interval: int = 100          # Steps between logs
    save_history: bool = False       # Save detailed history
    output_dir: Optional[str] = None # Output directory for logs
    
    # Random seed
    seed: Optional[int] = None


@dataclass
class NodeState:
    """Current state of the Node."""
    # Timestamps
    timestamp: float = 0.0
    uptime: float = 0.0
    step_count: int = 0
    
    # Z-coordinate (from spinner or internal)
    z: float = 0.5
    z_source: str = "internal"
    
    # Heart state
    coherence: float = 0.0
    hex_alignment: float = 0.0
    coupling_K: float = 0.0
    
    # Derived metrics
    delta_s_neg: float = 0.0
    tier: int = 0
    tier_name: str = "ABSENCE"
    phase: str = "ABSENCE"
    
    # TRIAD state
    kappa: float = PHI_INV
    lambda_: float = 1.0 - PHI_INV
    eta: float = 0.0
    R: int = 0
    conservation_error: float = 0.0
    distance_to_attractor: float = 0.0
    triad_stable: bool = True
    
    # Brain state
    brain_tier: int = 0
    available_operators: int = 0
    pattern_count: int = 0
    memory_utilization: float = 0.0
    
    # K-formation
    k_formation: bool = False
    k_formation_count: int = 0
    k_formation_duration: float = 0.0
    
    # Spinner state
    spinner_connected: bool = False
    spinner_z: float = 0.5
    spinner_k_formation: bool = False

    # Quasicrystal state
    quasicrystal_tile_ratio: float = 1.0
    quasicrystal_golden_error: float = PHI - 1.0
    quasicrystal_negentropy: float = 0.0
    quasicrystal_generation: int = 0
    combined_negentropy: float = 0.0

    # Events
    events: List[str] = field(default_factory=list)


class RosettaHelixNode:
    """
    Fully Integrated Rosetta-Helix Node.
    
    This is the main class that orchestrates:
    1. SpinnerClient: Receives z from Nuclear Spinner
    2. Heart: 60 Kuramoto oscillators with z-driven coupling
    3. Brain: GHMP pattern processing with tier-gated operators
    4. TRIAD: Triadic threshold dynamics tracking
    
    The coupling between components creates emergent behavior:
    - Spinner z drives Heart coupling
    - Heart coherence drives Brain tier
    - Brain tier gates operator availability
    - TRIAD monitors conservation and K-formation
    
    At z = z_c = √3/2, all systems achieve peak performance.
    """
    
    def __init__(
        self,
        config: Optional[NodeConfig] = None,
        on_state: Optional[Callable] = None,
        on_k_formation: Optional[Callable] = None,
        on_event: Optional[Callable] = None,
    ):
        """
        Initialize Rosetta-Helix Node.
        
        Args:
            config: Node configuration
            on_state: Async callback for state updates
            on_k_formation: Async callback for K-formation events
            on_event: Async callback for TRIAD events
        """
        self.config = config or NodeConfig()
        self.on_state = on_state
        self.on_k_formation = on_k_formation
        self.on_event = on_event
        
        # Initialize Heart
        heart_config = HeartConfig(
            n_oscillators=self.config.n_oscillators,
            coupling_scale=self.config.coupling_scale,
            dt=self.config.heart_dt,
            seed=self.config.seed,
        )
        self.heart = Heart(heart_config)
        
        # Initialize Brain
        if self.config.enable_brain:
            brain_config = BrainConfig(
                memory_capacity=self.config.memory_capacity,
                pattern_dim=self.config.pattern_dim,
                learning_rate=self.config.learning_rate,
                seed=self.config.seed,
            )
            self.brain = Brain(brain_config)
        else:
            self.brain = None
        
        # Initialize TRIAD
        if self.config.enable_triad:
            triad_config = TriadConfig(
                history_size=self.config.history_size,
            )
            self.triad = TriadTracker(triad_config)
        else:
            self.triad = None

        # Initialize Quasicrystal
        if self.config.enable_quasicrystal:
            qc_config = QuasicrystalConfig(
                initial_fat=1,
                initial_thin=1,
                max_generations=50,
            )
            qc = QuasicrystalDynamics(qc_config)
            self.quasicrystal = HelixQuasicrystalCoupling(qc)
        else:
            self.quasicrystal = None

        # Initialize SpinnerClient
        self.spinner = SpinnerClient(
            uri=self.config.bridge_uri,
            on_state=self._on_spinner_state,
        )
        
        # State
        self.state = NodeState()
        self.state_history: List[NodeState] = []
        self.start_time = 0.0
        self.running = False
        
        # K-formation tracking
        self._k_formation_start_time: Optional[float] = None
        
        # Output directory
        if self.config.output_dir:
            self.output_path = Path(self.config.output_dir)
            self.output_path.mkdir(parents=True, exist_ok=True)
        else:
            self.output_path = None
    
    async def _on_spinner_state(self, spinner_state: Dict[str, Any]):
        """Callback when spinner state updates."""
        spinner_z = spinner_state.get('z', 0.5)
        
        if self.config.use_spinner_z:
            self.heart.set_spinner_z(spinner_z)
        
        # Track spinner state
        self.state.spinner_z = spinner_z
        self.state.spinner_connected = True
        self.state.spinner_k_formation = spinner_state.get('k_formation', False)
    
    async def step(self) -> NodeState:
        """
        Execute single simulation step.
        
        This is the core loop that:
        1. Advances Heart oscillators
        2. Updates TRIAD with new coherence
        3. Processes patterns in Brain
        4. Checks for K-formation
        5. Fires callbacks
        
        Returns:
            Current node state
        """
        now = time.time()
        self.state.timestamp = now
        self.state.uptime = now - self.start_time
        self.state.step_count += 1
        self.state.events = []
        
        # =====================================================================
        # 1. HEART: Advance Kuramoto oscillators
        # =====================================================================
        coherence = self.heart.step()
        heart_state = self.heart.get_state()
        
        # Update z (from spinner if available, else from coherence)
        if self.config.use_spinner_z and self.state.spinner_connected:
            self.state.z = self.state.spinner_z
            self.state.z_source = "spinner"
        else:
            # Internal z from coherence (maps [0,1] to [0.5, 0.95])
            self.state.z = 0.5 + 0.45 * coherence
            self.state.z_source = "internal"
        
        # Update heart metrics
        self.state.coherence = coherence
        self.state.hex_alignment = heart_state.hex_alignment
        self.state.coupling_K = heart_state.coupling_K
        
        # Compute derived metrics
        self.state.delta_s_neg = compute_delta_s_neg(self.state.z)
        self.state.tier = get_tier(self.state.z)
        self.state.tier_name = TIER_NAMES[self.state.tier]
        self.state.phase = get_phase_name(self.state.z)
        
        # =====================================================================
        # 2. TRIAD: Update triadic threshold dynamics
        # =====================================================================
        if self.triad:
            triad_events = self.triad.update_from_coherence(
                coherence=coherence,
                z=self.state.z,
            )
            
            triad_state = self.triad.get_state()
            self.state.kappa = triad_state.kappa
            self.state.lambda_ = triad_state.lambda_
            self.state.eta = triad_state.eta
            self.state.R = triad_state.R
            self.state.conservation_error = triad_state.conservation_error
            self.state.distance_to_attractor = triad_state.distance_to_attractor
            self.state.triad_stable = triad_state.is_stable
            
            # Handle TRIAD events
            for event in triad_events:
                self.state.events.append(event.name)
                if self.on_event:
                    await self.on_event(event, self.state)
        else:
            # Manual computation without TRIAD
            self.state.kappa = coherence
            self.state.lambda_ = 1.0 - coherence
            self.state.eta = self.state.delta_s_neg * coherence
            self.state.R = int(7 + 5 * coherence * self.state.delta_s_neg)
        
        # =====================================================================
        # 3. BRAIN: Update tier and process patterns
        # =====================================================================
        if self.brain:
            self.brain.set_z(self.state.z)
            brain_state = self.brain.get_state()
            
            self.state.brain_tier = brain_state.tier
            self.state.available_operators = brain_state.processing_depth
            self.state.pattern_count = brain_state.pattern_count
            self.state.memory_utilization = brain_state.memory_utilization
            
            # Generate and store pattern from oscillator phases
            if self.state.step_count % 10 == 0:  # Every 10 steps
                phase_pattern = self.heart.get_phase_distribution()
                processed = self.brain.process(phase_pattern)
                
                # Store if coherent enough
                if coherence > 0.7:
                    self.brain.store_pattern(processed)

        # =====================================================================
        # 4. QUASICRYSTAL: Update Penrose tiling dynamics
        # =====================================================================
        if self.quasicrystal:
            # Update helix-quasicrystal coupling
            self.state.combined_negentropy = self.quasicrystal.update(
                z=self.state.z,
                helix_coherence=coherence,
            )

            # Inflate quasicrystal periodically
            if self.state.step_count % self.config.quasicrystal_inflate_interval == 0:
                self.quasicrystal.step_quasicrystal()

            # Update quasicrystal state
            qc_state = self.quasicrystal.qc.get_state()
            self.state.quasicrystal_tile_ratio = qc_state.tile_ratio
            self.state.quasicrystal_golden_error = qc_state.golden_error
            self.state.quasicrystal_negentropy = qc_state.quasicrystal_negentropy
            self.state.quasicrystal_generation = qc_state.generation

            # Check for quasicrystal stability event
            if qc_state.is_stable and qc_state.stability_duration == 1:
                self.state.events.append("QUASICRYSTAL_STABLE")

        # =====================================================================
        # 5. K-FORMATION: Check and track
        # =====================================================================
        k_active = check_k_formation(self.state.kappa, self.state.eta, self.state.R)
        
        if k_active:
            if not self.state.k_formation:
                # K-formation started
                self.state.k_formation_count += 1
                self._k_formation_start_time = now
                self.state.events.append("K_FORMATION_START")
                
                if self.on_k_formation:
                    await self.on_k_formation(self.state)
            
            self.state.k_formation_duration = now - self._k_formation_start_time
        else:
            if self.state.k_formation:
                # K-formation ended
                self.state.events.append("K_FORMATION_END")
            
            self.state.k_formation_duration = 0.0
            self._k_formation_start_time = None
        
        self.state.k_formation = k_active

        # =====================================================================
        # 6. CALLBACKS AND LOGGING
        # =====================================================================
        if self.on_state:
            await self.on_state(self.state)
        
        # Save history if enabled
        if self.config.save_history:
            self.state_history.append(self._copy_state())
        
        return self.state
    
    def _copy_state(self) -> NodeState:
        """Create a copy of current state."""
        return NodeState(
            timestamp=self.state.timestamp,
            uptime=self.state.uptime,
            step_count=self.state.step_count,
            z=self.state.z,
            z_source=self.state.z_source,
            coherence=self.state.coherence,
            hex_alignment=self.state.hex_alignment,
            coupling_K=self.state.coupling_K,
            delta_s_neg=self.state.delta_s_neg,
            tier=self.state.tier,
            tier_name=self.state.tier_name,
            phase=self.state.phase,
            kappa=self.state.kappa,
            lambda_=self.state.lambda_,
            eta=self.state.eta,
            R=self.state.R,
            conservation_error=self.state.conservation_error,
            distance_to_attractor=self.state.distance_to_attractor,
            triad_stable=self.state.triad_stable,
            brain_tier=self.state.brain_tier,
            available_operators=self.state.available_operators,
            pattern_count=self.state.pattern_count,
            memory_utilization=self.state.memory_utilization,
            k_formation=self.state.k_formation,
            k_formation_count=self.state.k_formation_count,
            k_formation_duration=self.state.k_formation_duration,
            spinner_connected=self.state.spinner_connected,
            spinner_z=self.state.spinner_z,
            spinner_k_formation=self.state.spinner_k_formation,
            quasicrystal_tile_ratio=self.state.quasicrystal_tile_ratio,
            quasicrystal_golden_error=self.state.quasicrystal_golden_error,
            quasicrystal_negentropy=self.state.quasicrystal_negentropy,
            quasicrystal_generation=self.state.quasicrystal_generation,
            combined_negentropy=self.state.combined_negentropy,
            events=self.state.events.copy(),
        )
    
    async def run(self, steps: Optional[int] = None, record_history: bool = False):
        """
        Main node loop.
        
        Args:
            steps: Number of steps to run (None = forever)
            record_history: Whether to record state history
        """
        self.running = True
        self.start_time = time.time()
        
        if record_history:
            self.config.save_history = True
        
        # Connect to spinner
        spinner_connected = await self.spinner.connect()
        self.state.spinner_connected = spinner_connected
        
        # Start spinner listener
        listen_task = None
        if spinner_connected:
            listen_task = asyncio.create_task(self.spinner.listen())
        
        # Main loop
        step_count = 0
        update_interval = 1.0 / self.config.update_rate_hz
        
        try:
            while self.running and (steps is None or step_count < steps):
                await self.step()
                step_count += 1
                await asyncio.sleep(update_interval)
                
        except KeyboardInterrupt:
            print("\n[NODE] Interrupted")
        finally:
            self.running = False
            if listen_task:
                listen_task.cancel()
            await self.spinner.disconnect()
            
            # Save results if output path configured
            if self.output_path:
                self._save_results()
    
    def _save_results(self):
        """Save results to output directory."""
        if not self.output_path:
            return
        
        # Save summary
        summary = self.get_summary()
        with open(self.output_path / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        # Save history if recorded
        if self.state_history:
            history_data = [asdict(s) for s in self.state_history]
            with open(self.output_path / "history.json", "w") as f:
                json.dump(history_data, f)
        
        # Save TRIAD analysis
        if self.triad:
            triad_analysis = {
                'k_formation_stats': self.triad.get_k_formation_stats(),
                'attractor_analysis': self.triad.get_attractor_analysis(),
            }
            with open(self.output_path / "triad_analysis.json", "w") as f:
                json.dump(triad_analysis, f, indent=2)
    
    async def send_command(self, cmd: str, **kwargs):
        """Send command to spinner."""
        await self.spinner.send_command(cmd, **kwargs)
    
    async def set_z(self, z: float):
        """Request specific z-coordinate from spinner."""
        await self.send_command('set_z', value=z)
    
    async def goto_lens(self):
        """Request z_c (THE LENS)."""
        await self.set_z(Z_CRITICAL)
    
    async def hex_cycle(self, dwell_s: float = 30.0, cycles: int = 10):
        """Run hexagonal cycling protocol."""
        await self.send_command('hex_cycle', dwell_s=dwell_s, cycles=cycles)
    
    async def stop(self):
        """Emergency stop."""
        await self.send_command('stop')
        self.running = False
    
    def get_state(self) -> NodeState:
        """Get current state."""
        return self.state
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        summary = {
            'uptime': self.state.uptime,
            'steps': self.state.step_count,
            'z': self.state.z,
            'z_source': self.state.z_source,
            'coherence': self.state.coherence,
            'tier': self.state.tier_name,
            'phase': self.state.phase,
            'kappa': self.state.kappa,
            'lambda': self.state.lambda_,
            'eta': self.state.eta,
            'R': self.state.R,
            'distance_to_attractor': self.state.distance_to_attractor,
            'k_formation': self.state.k_formation,
            'k_formation_count': self.state.k_formation_count,
            'spinner_connected': self.state.spinner_connected,
            'spinner_k_formation': self.state.spinner_k_formation,
        }
        
        if self.brain:
            summary['brain'] = {
                'tier': self.state.brain_tier,
                'operators': self.state.available_operators,
                'patterns': self.state.pattern_count,
                'memory': self.state.memory_utilization,
            }
        
        if self.triad:
            summary['triad'] = {
                'stable': self.state.triad_stable,
                'conservation_error': self.state.conservation_error,
                'total_k_formations': self.triad.state.total_k_formations,
            }

        if self.quasicrystal:
            summary['quasicrystal'] = {
                'tile_ratio': self.state.quasicrystal_tile_ratio,
                'golden_error': self.state.quasicrystal_golden_error,
                'negentropy': self.state.quasicrystal_negentropy,
                'generation': self.state.quasicrystal_generation,
                'combined_negentropy': self.state.combined_negentropy,
            }

        return summary
    
    def get_component_states(self) -> Dict[str, Any]:
        """Get state of all components."""
        return {
            'heart': asdict(self.heart.get_state()),
            'brain': asdict(self.brain.get_state()) if self.brain else None,
            'triad': asdict(self.triad.get_state()) if self.triad else None,
            'quasicrystal': self.quasicrystal.get_state() if self.quasicrystal else None,
            'spinner': self.spinner.get_stats(),
        }


async def main():
    """Main entry point."""
    print("═" * 70)
    print("  ROSETTA-HELIX NODE - FULL INTEGRATION")
    print("═" * 70)
    print(f"  Physics: φ={PHI:.6f}, φ⁻¹={PHI_INV:.6f}, z_c={Z_CRITICAL:.6f}")
    print(f"  Components: Heart (60 Kuramoto) + Brain (GHMP) + TRIAD")
    print(f"  Coupling: Spinner z → Kuramoto K → Coherence → K-formation")
    print("═" * 70)
    
    k_formation_count = [0]
    
    async def on_k_formation(state):
        k_formation_count[0] += 1
        print(f"\n  ★ K-FORMATION #{k_formation_count[0]}: "
              f"z={state.z:.4f} κ={state.kappa:.4f} η={state.eta:.4f} R={state.R}")
    
    async def on_event(event, state):
        if event == TriadEvent.PHASE_TRANSITION:
            print(f"\n  → Phase transition: {state.phase}")
        elif event == TriadEvent.ATTRACTOR_REACHED:
            print(f"\n  → Attractor reached: κ={state.kappa:.4f} ≈ φ⁻¹")
    
    # Create node
    config = NodeConfig(
        n_oscillators=60,
        coupling_scale=8.0,
        bridge_uri="ws://localhost:8765",
        enable_brain=True,
        enable_triad=True,
        seed=42,
    )
    node = RosettaHelixNode(
        config=config,
        on_k_formation=on_k_formation,
        on_event=on_event,
    )
    
    # Status display task
    async def print_status():
        while node.running:
            s = node.state
            k_str = " ★" if s.k_formation else ""
            brain_str = f" ops={s.available_operators}" if node.brain else ""
            print(f"\r  z={s.z:.4f} r={s.coherence:.4f} κ={s.kappa:.4f} "
                  f"{s.tier_name:10s} {s.phase:10s}{brain_str}{k_str}",
                  end='', flush=True)
            await asyncio.sleep(0.2)
    
    status_task = asyncio.create_task(print_status())
    
    # Request z_c after brief warmup
    await asyncio.sleep(1.0)
    print("\n  Requesting z = z_c = 0.866...")
    await node.set_z(Z_CRITICAL)
    
    # Run
    try:
        await node.run()
    finally:
        status_task.cancel()
        print("\n")
        print("═" * 70)
        print("  SUMMARY")
        print("═" * 70)
        summary = node.get_summary()
        for key, value in summary.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    {k}: {v}")
            else:
                print(f"  {key}: {value}")
        print("═" * 70)


if __name__ == "__main__":
    asyncio.run(main())
