from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, TypeVar


ReturnType = TypeVar("ReturnType")


@dataclass(frozen=True)
class RetryConfig:
    max_attempts: int = 3
    initial_backoff_seconds: float = 0.1
    backoff_multiplier: float = 2.0


def execute_with_retry(callable_fn: Callable[[], ReturnType], config: RetryConfig) -> ReturnType:
    if config.max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    attempt = 0
    delay = config.initial_backoff_seconds
    while True:
        try:
            return callable_fn()
        except Exception:
            attempt += 1
            if attempt >= config.max_attempts:
                raise
            time.sleep(delay)
            delay *= config.backoff_multiplier
