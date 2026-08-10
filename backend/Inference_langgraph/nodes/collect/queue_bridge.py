
"""
app_simulator/pipeline/queue_bridge.py
=======================================
Thread-safe in-process deque that bridges the data generator (Terminal 1)
and the pipeline consumer (Terminal 2, or same process in single-process mode).

Design:
  - Uses collections.deque(maxlen=QUEUE_MAXLEN) protected by threading.Lock.
  - The generator pushes one item per tick (every 2 seconds).
  - The pipeline pops one item per tick.
  - If the pipeline is slower than the generator, older items are silently dropped
    (bounded queue). This is intentional — we never block the generator.

API:
    from nodes.collect.queue_bridge import TelemetryQueue
    TelemetryQueue.push(metric, log, spans)
    item = TelemetryQueue.pop()   # → (metric, log, spans) or None
    TelemetryQueue.size()         # → int
"""
from __future__ import annotations

import threading
from collections import deque
from typing import Optional

# Max items in queue. At 2s/tick, 5000 items = 166 minutes of backlog.
QUEUE_MAXLEN = 5_000


class _TelemetryQueueSingleton:
    """Singleton thread-safe bounded deque for telemetry items."""

    def __init__(self):
        self._lock  = threading.Lock()
        self._deque: deque = deque(maxlen=QUEUE_MAXLEN)

    def push(self, metric: dict, log: dict, spans: list[dict]) -> None:
        """Push one telemetry tick. Oldest item silently dropped if full."""
        with self._lock:
            self._deque.append((metric, log, spans))

    def pop(self) -> Optional[tuple[dict, dict, list[dict]]]:
        """Pop the oldest item, or None if queue is empty."""
        with self._lock:
            if self._deque:
                return self._deque.popleft()
            return None

    def size(self) -> int:
        """Return current queue length."""
        with self._lock:
            return len(self._deque)

    def is_empty(self) -> bool:
        """Return True if queue has no items."""
        with self._lock:
            return len(self._deque) == 0

    def clear(self) -> None:
        """Empty the queue (used in tests)."""
        with self._lock:
            self._deque.clear()


# Module-level singleton — imported by both generator and pipeline
TelemetryQueue = _TelemetryQueueSingleton()
