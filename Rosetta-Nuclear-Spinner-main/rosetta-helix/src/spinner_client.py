"""
Spinner Client - Rosetta-Helix
==============================

WebSocket client for receiving Nuclear Spinner state from the bridge service.
Feeds z-coordinate to Kuramoto Heart for coupling.

Signature: rosetta-helix-spinner-client|v1.0.0|helix
"""

import asyncio
import json
from typing import Callable, Optional, Dict, Any
from dataclasses import dataclass

try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    print("[SPINNER_CLIENT] websockets not installed: pip install websockets")

from .physics import (
    Z_CRITICAL, PHI_INV,
    compute_delta_s_neg, check_k_formation
)


@dataclass
class SpinnerClientConfig:
    """Configuration for SpinnerClient."""
    uri: str = "ws://localhost:8765"
    reconnect_delay: float = 1.0
    max_reconnect_attempts: int = 5


class SpinnerClient:
    """
    WebSocket client for receiving Nuclear Spinner state.
    
    Connects to the bridge service (spinner_bridge.py) which relays
    state from the firmware (or simulation).
    
    Usage:
        client = SpinnerClient()
        await client.connect()
        
        # In loop:
        z = client.get_z()
        k = client.is_k_formation()
        
        # Send commands:
        await client.send_command('set_z', value=0.866)
    """
    
    def __init__(
        self, 
        uri: str = "ws://localhost:8765",
        on_state: Optional[Callable] = None,
        on_k_formation: Optional[Callable] = None,
        config: Optional[SpinnerClientConfig] = None,
    ):
        """
        Initialize SpinnerClient.
        
        Args:
            uri: WebSocket URI of bridge service
            on_state: Async callback for state updates
            on_k_formation: Async callback for K-formation events
            config: Full configuration object (overrides uri)
        """
        self.config = config or SpinnerClientConfig(uri=uri)
        self.uri = self.config.uri
        self.on_state = on_state
        self.on_k_formation = on_k_formation
        
        self.websocket = None
        self.connected = False
        self.latest_state: Optional[Dict[str, Any]] = None
        self.reconnect_attempts = 0
        
        # State tracking
        self.k_formation_active = False
        self.k_formation_count = 0
        self.state_count = 0
    
    async def connect(self) -> bool:
        """
        Connect to bridge service.
        
        Returns:
            True if connection successful
        """
        if not WEBSOCKETS_AVAILABLE:
            print("[CLIENT] websockets not available")
            return False
        
        try:
            self.websocket = await websockets.connect(self.uri)
            self.connected = True
            self.reconnect_attempts = 0
            print(f"[CLIENT] Connected to {self.uri}")
            return True
        except Exception as e:
            print(f"[CLIENT] Connection failed: {e}")
            self.connected = False
            return False
    
    async def disconnect(self):
        """Disconnect from bridge service."""
        if self.websocket:
            await self.websocket.close()
        self.connected = False
        print("[CLIENT] Disconnected")
    
    async def listen(self):
        """
        Listen for state updates from bridge.
        
        Runs until connection is closed.
        """
        while self.connected:
            try:
                message = await self.websocket.recv()
                data = json.loads(message)
                
                if data.get('type') == 'spinner_state':
                    self.latest_state = data
                    self.state_count += 1
                    
                    # Check for K-formation transition
                    new_k = data.get('k_formation', False)
                    if new_k and not self.k_formation_active:
                        self.k_formation_count += 1
                        if self.on_k_formation:
                            await self.on_k_formation(data)
                    self.k_formation_active = new_k
                    
                    # State callback
                    if self.on_state:
                        await self.on_state(data)
                        
            except websockets.ConnectionClosed:
                print("[CLIENT] Connection closed")
                self.connected = False
                break
            except Exception as e:
                print(f"[CLIENT] Error: {e}")
                await asyncio.sleep(0.1)
    
    async def listen_with_reconnect(self):
        """Listen with automatic reconnection."""
        while self.reconnect_attempts < self.config.max_reconnect_attempts:
            if not self.connected:
                success = await self.connect()
                if not success:
                    self.reconnect_attempts += 1
                    await asyncio.sleep(self.config.reconnect_delay)
                    continue
            
            await self.listen()
            
            if not self.connected:
                self.reconnect_attempts += 1
                print(f"[CLIENT] Reconnect attempt {self.reconnect_attempts}")
                await asyncio.sleep(self.config.reconnect_delay)
        
        print("[CLIENT] Max reconnect attempts reached")
    
    async def send_command(self, cmd: str, **kwargs):
        """
        Send command to spinner via bridge.
        
        Args:
            cmd: Command name ('set_z', 'stop', 'hex_cycle', etc.)
            **kwargs: Command parameters
        
        Commands:
            set_z: Set target z (value: float)
            set_rpm: Set target RPM (value: float)
            stop: Emergency stop
            hex_cycle: Hexagonal cycling (dwell_s: float, cycles: int)
            dwell_lens: Dwell at z_c (duration_s: float)
        """
        if not self.connected or not self.websocket:
            print("[CLIENT] Not connected, cannot send command")
            return
        
        message = json.dumps({"cmd": cmd, **kwargs})
        try:
            await self.websocket.send(message)
            print(f"[CLIENT] Sent: {cmd} {kwargs}")
        except Exception as e:
            print(f"[CLIENT] Send error: {e}")
    
    # =========================================================================
    # STATE ACCESSORS
    # =========================================================================
    
    def get_z(self) -> float:
        """Get current z-coordinate."""
        if self.latest_state:
            return self.latest_state.get('z', 0.5)
        return 0.5
    
    def get_rpm(self) -> int:
        """Get current RPM."""
        if self.latest_state:
            return self.latest_state.get('rpm', 0)
        return 0
    
    def get_delta_s_neg(self) -> float:
        """Get current negentropy signal."""
        if self.latest_state:
            return self.latest_state.get('delta_s_neg', 0.0)
        return 0.0
    
    def get_tier(self) -> int:
        """Get current tier."""
        if self.latest_state:
            return self.latest_state.get('tier', 0)
        return 0
    
    def get_tier_name(self) -> str:
        """Get current tier name."""
        if self.latest_state:
            return self.latest_state.get('tier_name', 'UNKNOWN')
        return 'UNKNOWN'
    
    def get_phase(self) -> str:
        """Get current phase."""
        if self.latest_state:
            return self.latest_state.get('phase', 'UNKNOWN')
        return 'UNKNOWN'
    
    def get_kappa(self) -> float:
        """Get current kappa (coherence)."""
        if self.latest_state:
            return self.latest_state.get('kappa', 0.0)
        return 0.0
    
    def get_eta(self) -> float:
        """Get current eta (efficiency)."""
        if self.latest_state:
            return self.latest_state.get('eta', 0.0)
        return 0.0
    
    def get_rank(self) -> int:
        """Get current complexity rank."""
        if self.latest_state:
            return self.latest_state.get('rank', 0)
        return 0
    
    def is_k_formation(self) -> bool:
        """Check if K-formation is active."""
        if self.latest_state:
            return self.latest_state.get('k_formation', False)
        return False
    
    def is_at_lens(self) -> bool:
        """Check if at THE LENS (z_c)."""
        z = self.get_z()
        return abs(z - Z_CRITICAL) < 0.02
    
    def get_state(self) -> Optional[Dict[str, Any]]:
        """Get complete latest state."""
        return self.latest_state
    
    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics."""
        return {
            'connected': self.connected,
            'state_count': self.state_count,
            'k_formation_count': self.k_formation_count,
            'current_z': self.get_z(),
            'current_tier': self.get_tier_name(),
            'k_formation_active': self.is_k_formation(),
        }


async def test_client():
    """Test SpinnerClient."""
    print("=" * 60)
    print("SPINNER CLIENT TEST")
    print("=" * 60)
    
    async def on_state(state):
        z = state.get('z', 0)
        k = state.get('k_formation', False)
        tier = state.get('tier_name', 'UNKNOWN')
        print(f"\r  z={z:.4f} tier={tier:10s} k={k}", end='', flush=True)
    
    async def on_k_formation(state):
        print(f"\n  ★ K-FORMATION at z={state.get('z', 0):.4f}")
    
    client = SpinnerClient(
        on_state=on_state,
        on_k_formation=on_k_formation,
    )
    
    print("\nConnecting to bridge...")
    success = await client.connect()
    
    if not success:
        print("Failed to connect. Make sure spinner_bridge.py is running.")
        return
    
    # Start listener task
    listen_task = asyncio.create_task(client.listen())
    
    # Request z_c
    print("\nRequesting z = z_c = 0.866...")
    await client.send_command('set_z', value=0.866)
    
    # Wait and observe
    try:
        await asyncio.sleep(10)
    except KeyboardInterrupt:
        pass
    
    # Cleanup
    listen_task.cancel()
    await client.disconnect()
    
    print("\n")
    print(f"Stats: {client.get_stats()}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_client())
