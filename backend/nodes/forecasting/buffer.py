"""
d:/Before_done/forecasting_node/buffer.py
==========================================
In-process episode history buffer.

Stores the accumulated sequence of per-cycle feature rows for each active
episode in a bounded in-memory deque. This is the DATA SOURCE for all
forecasting nodes — they read raw feature trajectories from here.

Mentor's spec:
  - Lives entirely in process memory (no disk I/O on the hot path).
  - Keyed by episode_id; automatically grows as cycles arrive.
  - Capped at MAX_LOOKBACK_CYCLES to prevent unbounded growth.
  - clear_episode() is called when the episode rolls over.

Usage:
    from nodes.forecasting.buffer import append_feature_row, get_metric_series, clear_episode

    # Every FE cycle — append the latest feature row:
    append_feature_row(episode_id="ep_MEMORY_LEAK_...", row=feature_dict)

    # In the forecasting node — read the full series for one metric:
    series = get_metric_series("ep_MEMORY_LEAK_...", "heap_mean")
    # → [512.1, 524.3, 536.8, 549.2, ...]
"""
from __future__ import annotations

from collections import deque
from typing import Any

from .thresholds import MAX_LOOKBACK_CYCLES

# ── Internal state ────────────────────────────────────────────────────────────
# { episode_id: { metric_name: deque[float] } }
_BUFFERS: dict[str, dict[str, deque]] = {}


def append_feature_row(episode_id: str, row: dict[str, Any]) -> None:
    """
    Append one feature row to the episode buffer.

    All numeric values in `row` are stored individually per metric.
    Non-numeric values (strings, booleans converted to int/float) are stored.
    Keys starting with '_' (evidence fields) are skipped.

    Args:
        episode_id: Current episode identifier string.
        row:        Dict of feature_name → value (from feature engineering output).
    """
    if episode_id not in _BUFFERS:
        _BUFFERS[episode_id] = {}

    buf = _BUFFERS[episode_id]
    for key, val in row.items():
        if key.startswith("_"):
            continue                                  # skip evidence fields
        try:
            float_val = float(val)
        except (TypeError, ValueError):
            continue                                  # skip non-numeric
        if key not in buf:
            buf[key] = deque(maxlen=MAX_LOOKBACK_CYCLES)
        buf[key].append(float_val)


def get_metric_series(episode_id: str, metric: str) -> list[float]:
    """
    Return the full accumulated series for one metric in one episode.

    Args:
        episode_id: Episode identifier.
        metric:     Feature name (e.g. 'heap_mean', 'cpu_mean').

    Returns:
        List of float values from oldest → newest. Empty list if not found.
    """
    if episode_id not in _BUFFERS:
        return []
    return list(_BUFFERS[episode_id].get(metric, []))


def get_episode_length(episode_id: str) -> int:
    """Return number of cycles accumulated for this episode."""
    if episode_id not in _BUFFERS:
        return 0
    # All metrics should have the same length — use the first one found
    for metric_deque in _BUFFERS[episode_id].values():
        return len(metric_deque)
    return 0


def get_all_metrics(episode_id: str) -> dict[str, list[float]]:
    """Return all metrics accumulated so far for this episode."""
    if episode_id not in _BUFFERS:
        return {}
    return {k: list(v) for k, v in _BUFFERS[episode_id].items()}


def clear_episode(episode_id: str) -> None:
    """
    Free memory for a completed episode.
    Call this when episode_id changes (episode rollover).
    """
    _BUFFERS.pop(episode_id, None)


def reset_all() -> None:
    """Clear all episode buffers. Used for testing / server restart."""
    _BUFFERS.clear()


def active_episodes() -> list[str]:
    """Return list of currently active episode_ids in the buffer."""
    return list(_BUFFERS.keys())
