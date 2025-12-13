"""
Structured Logging
==================

JSON-structured logging for training runs.

Signature: logger|v0.1.0|helix
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, Optional


class LogLevel(IntEnum):
    """Log levels matching Python logging."""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


class StructuredLogger:
    """
    Logger that outputs JSON-structured logs.

    Features:
    - JSON output for machine parsing
    - Human-readable console output
    - File output to run directory
    - Context enrichment (run_id, step, etc.)
    """

    def __init__(
        self,
        name: str = "helix",
        level: LogLevel = LogLevel.INFO,
        log_file: Optional[str] = None,
        json_output: bool = True,
        console_output: bool = True,
    ):
        self.name = name
        self.level = level
        self.json_output = json_output
        self.console_output = console_output

        # Context that gets added to every log
        self.context: Dict[str, Any] = {}

        # File handler
        self._file_handle = None
        if log_file:
            self.set_log_file(log_file)

    def set_context(self, **kwargs) -> None:
        """Set context that will be added to all log messages."""
        self.context.update(kwargs)

    def clear_context(self) -> None:
        """Clear all context."""
        self.context.clear()

    def set_log_file(self, path: str) -> None:
        """Set the log file path."""
        if self._file_handle:
            self._file_handle.close()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._file_handle = open(path, "a")

    def _format_message(self, level: str, message: str, **kwargs) -> str:
        """Format a log message as JSON."""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": level,
            "logger": self.name,
            "message": message,
            **self.context,
            **kwargs,
        }
        return json.dumps(entry)

    def _format_console(self, level: str, message: str, **kwargs) -> str:
        """Format a log message for console output."""
        timestamp = datetime.utcnow().strftime("%H:%M:%S")
        prefix = f"[{timestamp}] [{level:7}]"

        if "step" in self.context:
            prefix += f" [step {self.context['step']:>6}]"

        return f"{prefix} {message}"

    def _log(self, level: str, message: str, **kwargs) -> None:
        """Internal logging method."""
        level_value = getattr(LogLevel, level, LogLevel.INFO)
        if level_value < self.level:
            return

        # JSON output to file
        if self.json_output and self._file_handle:
            json_msg = self._format_message(level, message, **kwargs)
            self._file_handle.write(json_msg + "\n")
            self._file_handle.flush()

        # Console output
        if self.console_output:
            console_msg = self._format_console(level, message, **kwargs)
            stream = sys.stderr if level_value >= LogLevel.ERROR else sys.stdout
            print(console_msg, file=stream)

    def debug(self, message: str, **kwargs) -> None:
        """Log a debug message."""
        self._log("DEBUG", message, **kwargs)

    def info(self, message: str, **kwargs) -> None:
        """Log an info message."""
        self._log("INFO", message, **kwargs)

    def warning(self, message: str, **kwargs) -> None:
        """Log a warning message."""
        self._log("WARNING", message, **kwargs)

    def error(self, message: str, **kwargs) -> None:
        """Log an error message."""
        self._log("ERROR", message, **kwargs)

    def critical(self, message: str, **kwargs) -> None:
        """Log a critical message."""
        self._log("CRITICAL", message, **kwargs)

    def step(self, step: int, metrics: Dict[str, Any], **kwargs) -> None:
        """Log a training step with metrics."""
        # Update context
        self.context["step"] = step

        # Format metrics for console
        metrics_str = " | ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                  for k, v in metrics.items())

        self._log("INFO", f"Step {step}: {metrics_str}", metrics=metrics, **kwargs)

    def event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Log a structured event."""
        self._log("INFO", f"Event: {event_type}", event_type=event_type, event_data=data)

    def close(self) -> None:
        """Close the log file."""
        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
