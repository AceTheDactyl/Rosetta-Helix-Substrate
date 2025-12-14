#!/usr/bin/env python3
"""
spinner_client.py
─────────────────
WebSocket client for receiving Nuclear Spinner state.
"""

import asyncio
import json
from typing import Callable, Optional

try:
    import websockets
except ImportError:
    print("Install websockets: pip install websockets")
    raise

class SpinnerClient:
    """WebSocket client for spinner state."""
    
    def __init__(
        self, 
        uri: str = "ws://localhost:8765",
        on_state: Optional[Callable] = None
    ):
        self.uri = uri
        self.on_state = on_state
        self.websocket = None
        self.connected = False
        self.latest_state = None
    
    async def connect(self):
        try:
            self.websocket = await websockets.connect(self.uri)
            self.connected = True
            print(f"[CLIENT] Connected to {self.uri}")
        except Exception as e:
            print(f"[CLIENT] Connection failed: {e}")
            self.connected = False
    
    async def listen(self):
        while self.connected:
            try:
                message = await self.websocket.recv()
                data = json.loads(message)
                if data.get('type') == 'spinner_state':
                    self.latest_state = data
                    if self.on_state:
                        await self.on_state(data)
            except Exception:
                self.connected = False
                break
    
    async def send_command(self, cmd: str, **kwargs):
        if self.connected:
            await self.websocket.send(json.dumps({"cmd": cmd, **kwargs}))
    
    def get_z(self) -> float:
        return self.latest_state.get('z', 0.5) if self.latest_state else 0.5
    
    def is_k_formation(self) -> bool:
        return self.latest_state.get('k_formation', False) if self.latest_state else False
