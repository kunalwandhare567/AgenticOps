"""
backend/live_feed_simulator/live_queue.py
==========================================
Thread-safe bounded deque bridging the live feed simulator (producer)
and the LangGraph pipeline (consumer) within the same process.

Isolated from nodes/collect/queue_bridge.py so historical and live
streams never interfere with each other.

API:
    from Simulator.live_feed_simulator.live_queue import LiveTelemetryQueue
    LiveTelemetryQueue.push(metric, log, spans)
    item = LiveTelemetryQueue.pop()   # → (metric, log, spans) or None
    LiveTelemetryQueue.size()         # → int
"""
from __future__ import annotations

import threading
from collections import deque
from typing import Optional

# At 0.5s/tick, 2000 items = ~16 minutes of backlog.
_QUEUE_MAXLEN = 2_000


class _LiveTelemetryQueueSingleton:
    """Singleton thread-safe bounded deque for live telemetry ticks."""

    def __init__(self) -> None:
        self._lock  = threading.Lock()
        self._deque: deque = deque(maxlen=_QUEUE_MAXLEN)

    def push(self, metric: dict, log: dict, spans: list) -> None:
        """Push one telemetry tick. Oldest item silently dropped if full."""
        with self._lock:
            self._deque.append((metric, log, spans))

    def pop(self) -> Optional[tuple]:
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
        """Empty the queue."""
        with self._lock:
            self._deque.clear()


# Module-level singleton — shared between live_feed_simulator and run_langgraph
LiveTelemetryQueue = _LiveTelemetryQueueSingleton()
