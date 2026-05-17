"""Thread-safe token bucket rate limiter.

Configured per-API in config.yaml. The limiter is used for NIM and Gemini calls.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass


@dataclass
class _Bucket:
    capacity: float
    tokens: float
    refill_per_sec: float
    last_refill: float


class RateLimiter:
    """Simple token bucket. Acquire blocks until 1 token is available."""

    def __init__(self, rpm: float) -> None:
        self._lock = threading.Lock()
        self._bucket = _Bucket(
            capacity=max(1.0, rpm),
            tokens=max(1.0, rpm),
            refill_per_sec=rpm / 60.0,
            last_refill=time.monotonic(),
        )

    def _refill(self) -> None:
        now = time.monotonic()
        delta = now - self._bucket.last_refill
        self._bucket.tokens = min(
            self._bucket.capacity,
            self._bucket.tokens + delta * self._bucket.refill_per_sec,
        )
        self._bucket.last_refill = now

    def acquire(self) -> None:
        while True:
            with self._lock:
                self._refill()
                if self._bucket.tokens >= 1.0:
                    self._bucket.tokens -= 1.0
                    return
                needed = 1.0 - self._bucket.tokens
                wait = needed / self._bucket.refill_per_sec
            time.sleep(max(0.01, wait))
