"""
Signal Handling
===============

Graceful shutdown handling for:
- SIGTERM (kubernetes termination)
- SIGINT (Ctrl+C)
- Custom termination conditions

Signature: signals|v0.1.0|helix
"""

from __future__ import annotations

import signal
import sys
import threading
from typing import Callable, Optional


class GracefulShutdown:
    """
    Context manager for graceful shutdown handling.

    Usage:
        with GracefulShutdown() as shutdown:
            while not shutdown.requested:
                # training loop
                train_step()
    """

    def __init__(self, callback: Optional[Callable[[], None]] = None):
        self.requested = False
        self.callback = callback
        self._original_handlers = {}
        self._lock = threading.Lock()

    def __enter__(self):
        self._register_handlers()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._restore_handlers()
        return False

    def request_shutdown(self) -> None:
        """Request a graceful shutdown."""
        with self._lock:
            if not self.requested:
                self.requested = True
                if self.callback:
                    self.callback()

    def _handler(self, signum, frame):
        """Handle termination signal."""
        signal_name = signal.Signals(signum).name
        print(f"\n[GracefulShutdown] Received {signal_name}, requesting shutdown...")
        self.request_shutdown()

    def _register_handlers(self):
        """Register signal handlers."""
        for sig in (signal.SIGTERM, signal.SIGINT):
            self._original_handlers[sig] = signal.signal(sig, self._handler)

    def _restore_handlers(self):
        """Restore original signal handlers."""
        for sig, handler in self._original_handlers.items():
            signal.signal(sig, handler)


class SignalHandler:
    """
    Global signal handler that can be used outside context manager.
    """

    _instance: Optional["SignalHandler"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.shutdown_requested = False
        self.callbacks = []
        self._lock = threading.Lock()

    def register(self) -> None:
        """Register signal handlers."""
        signal.signal(signal.SIGTERM, self._handler)
        signal.signal(signal.SIGINT, self._handler)

    def add_callback(self, callback: Callable[[], None]) -> None:
        """Add a callback to be called on shutdown."""
        self.callbacks.append(callback)

    def _handler(self, signum, frame):
        """Handle termination signal."""
        signal_name = signal.Signals(signum).name
        print(f"\n[SignalHandler] Received {signal_name}")

        with self._lock:
            if not self.shutdown_requested:
                self.shutdown_requested = True
                for callback in self.callbacks:
                    try:
                        callback()
                    except Exception as e:
                        print(f"[SignalHandler] Callback error: {e}")

    @classmethod
    def get_instance(cls) -> "SignalHandler":
        """Get the singleton instance."""
        return cls()

    @classmethod
    def is_shutdown_requested(cls) -> bool:
        """Check if shutdown has been requested."""
        return cls.get_instance().shutdown_requested
