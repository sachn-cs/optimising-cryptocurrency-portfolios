from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class MetricsRegistry:
    counters: dict[str, int] = field(default_factory=dict)
    timings_millis: dict[str, list[float]] = field(default_factory=dict)

    def increment(self, name: str, amount: int = 1) -> None:
        self.counters[name] = self.counters.get(name, 0) + amount

    def record_timing_millis(self, name: str, elapsed_millis: float) -> None:
        values = self.timings_millis.setdefault(name, [])
        values.append(float(elapsed_millis))


class StructuredLogger:
    def __init__(self, name: str, log_path: str | None = None) -> None:
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(message)s"))
        self.logger.addHandler(stream_handler)
        self.log_path = Path(log_path) if log_path else None

    def log_event(self, event: str, payload: dict[str, object]) -> None:
        message = {"event": event, **payload}
        line = json.dumps(message, default=str)
        self.logger.info(line)
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as file_handle:
                file_handle.write(line + "\n")


class Timer:
    def __init__(self) -> None:
        self.started = time.perf_counter()

    def elapsed_millis(self) -> float:
        return (time.perf_counter() - self.started) * 1000.0
