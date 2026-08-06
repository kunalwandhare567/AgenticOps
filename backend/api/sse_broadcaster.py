"""
backend/api/sse_broadcaster.py
================================
In-memory Server-Sent Events broadcaster for real-time pipeline state.

n10_db_writer calls SSEBroadcaster.update(state) after every completed
pipeline cycle. The FastAPI /api/live-stream endpoint calls
SSEBroadcaster.stream() to yield SSE events to connected clients.

Design:
  - Thread-safe: update() is called from the synchronous LangGraph
    pipeline thread; stream() is an async generator in FastAPI's
    async event loop.
  - Only the LATEST state is broadcast (no buffering of history).
  - Multiple SSE clients are supported simultaneously.
  - Falls back to empty dict if no state has been pushed yet.
"""
from __future__ import annotations

import asyncio
import json
import threading
from typing import AsyncGenerator


class SSEBroadcaster:
    """Thread-safe singleton that holds the latest pipeline state for SSE push."""

    _lock:         threading.Lock = threading.Lock()
    _latest_state: dict           = {}
    _updated_event: threading.Event = threading.Event()

    @classmethod
    def update(cls, state: dict) -> None:
        """
        Update the latest pipeline state (called from pipeline thread).

        Args:
            state: Full AIOpsLangState dict from the last completed cycle.
        """
        with cls._lock:
            # Store a safe serialisable copy — drop non-JSON-friendly values
            safe = {}
            for k, v in state.items():
                try:
                    json.dumps(v)   # quick serialisability check
                    safe[k] = v
                except (TypeError, ValueError):
                    safe[k] = str(v)
            cls._latest_state = safe
        cls._updated_event.set()

    @classmethod
    def get_latest(cls) -> dict:
        """Return the latest state snapshot (thread-safe)."""
        with cls._lock:
            return dict(cls._latest_state)

    @classmethod
    async def stream(cls, interval_s: float = 0.5) -> AsyncGenerator[str, None]:
        """
        Async generator for FastAPI StreamingResponse (SSE format).

        Yields:
            SSE-formatted string: "data: {json}\\n\\n"

        Fetches the complete enriched dashboard payload (with charts, history,
        forecasts, and Weibull parameters) from LiveFeedService.
        """
        from api.services import LiveFeedService
        while True:
            try:
                payload = LiveFeedService.get_live_feed_state()
                if not payload:
                    payload = cls.get_latest()
                data = json.dumps(payload) if payload else "{}"
                yield f"data: {data}\n\n"
            except Exception as e:
                yield "data: {}\n\n"
            await asyncio.sleep(interval_s)

