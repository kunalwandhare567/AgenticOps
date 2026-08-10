"""
d:/Before_done/forecasting_node/router.py
==========================================
Routes dominant_state (from tumbling window) → correct forecasting function.

Responsibilities:
  1. Dispatch failure_mode → dedicated forecasting node.
  2. Manage episode lifecycle (register, heartbeat, close, auto-expire).
  3. Batch-route a list of feature rows in one call (for replay / backfill).
  4. Format the raw forecast result into the AgentState-compatible output dict.
  5. Provide a summary view of all active episodes and their TTF status.

Architecture:
  ┌───────────────────────────────────────────────────────┐
  │  Tumbling Window                                       │
  │  (decides dominant_state = WHAT is failing)           │
  └────────────────────────┬──────────────────────────────┘
                           │  dominant_state = "MEMORY_LEAK"
                           ▼
  ┌───────────────────────────────────────────────────────┐
  │  router.py → route_forecast()                         │
  │  Looks up _ROUTER[dominant_state]                     │
  │  Calls fn(episode_id, current_features)               │
  └────────────────────────┬──────────────────────────────┘
                           │  raw result dict
                           ▼
  ┌───────────────────────────────────────────────────────┐
  │  _format_result()                                     │
  │  Normalises keys for AgentState fields:               │
  │  • forecast            ← trajectory payload           │
  │  • time_to_failure     ← float or None                │
  │  • forecast_confidence ← 0.0–1.0                      │
  └───────────────────────────────────────────────────────┘

Usage (single cycle):
    from nodes.forecasting.router import route_forecast

    result = route_forecast(
        failure_mode     = "MEMORY_LEAK",
        episode_id       = "ep_MEMORY_LEAK_20260723_...",
        current_features = feature_row_dict,
    )
    print(result["time_to_failure"])      # minutes until breach or None
    print(result["forecast_confidence"]) # 0.0–1.0

Usage (batch replay):
    from nodes.forecasting.router import route_forecast_batch
    results = route_forecast_batch(
        failure_mode = "MEMORY_LEAK",
        episode_id   = "ep_...",
        feature_rows = [row_0, row_1, row_2, ...],   # chronological order
    )
    # results[-1] = forecast after consuming all rows

Usage (episode lifecycle):
    from nodes.forecasting.router import open_episode, close_episode, active_episode_summary

    open_episode("ep_MEMORY_LEAK_20260723_...")
    ...
    close_episode("ep_MEMORY_LEAK_20260723_...")
    print(active_episode_summary())
"""
from __future__ import annotations

import time
from typing import Any, Optional

from .buffer import clear_episode, active_episodes, get_episode_length
from .modes import (
    forecast_bad_deployment,
    forecast_cache_stampede,
    forecast_cascading_failure,
    forecast_cpu_saturation,
    forecast_db_slowdown,
    forecast_dependency_timeout,
    forecast_disk_io_saturation,
    forecast_error_storm,
    forecast_latency_spike,
    forecast_memory_leak,
    forecast_queue_backup,
    forecast_retry_storm,
)

# ── Routing table  ────────────────────────────────────────────────────────────
# Maps every known failure mode string → its dedicated forecast function.
# All functions share the signature: fn(episode_id, current_features) -> dict
# ─────────────────────────────────────────────────────────────────────────────
_ROUTER: dict[str, Any] = {
    "MEMORY_LEAK":        forecast_memory_leak,
    "CPU_SATURATION":     forecast_cpu_saturation,
    "LATENCY_SPIKE":      forecast_latency_spike,
    "DB_SLOWDOWN":        forecast_db_slowdown,
    "CACHE_STAMPEDE":     forecast_cache_stampede,
    "QUEUE_BACKUP":       forecast_queue_backup,
    "DEPENDENCY_TIMEOUT": forecast_dependency_timeout,
    "BAD_DEPLOYMENT":     forecast_bad_deployment,
    "BAD_DEPLOY":         forecast_bad_deployment,   # alias — simulator uses BAD_DEPLOY
    "ERROR_STORM":        forecast_error_storm,
    "RETRY_STORM":        forecast_retry_storm,
    "DISK_IO_SATURATION": forecast_disk_io_saturation,
    "CASCADING_FAILURE":  forecast_cascading_failure,
}

# ── Episode registry  ─────────────────────────────────────────────────────────
# Tracks open episodes with: open_time, last_seen_time, failure_mode.
# Used for lifecycle management and the active_episode_summary() view.
# ─────────────────────────────────────────────────────────────────────────────
_EPISODE_REGISTRY: dict[str, dict] = {}

# Auto-expire: if an episode hasn't been updated in 600 seconds it is stale.
_EPISODE_TTL_S: float = 600.0


# =============================================================================
# PUBLIC API — SINGLE CYCLE DISPATCH
# =============================================================================

def route_forecast(
    failure_mode:     str,
    episode_id:       str,
    current_features: dict[str, Any],
) -> dict:
    """
    Dispatch to the correct forecasting node for one cycle.

    This is the main entry-point called every FE cycle by the pipeline.

    Args:
        failure_mode:     Dominant failure mode string from the tumbling window.
                          e.g. "MEMORY_LEAK" | "CPU_SATURATION" | "NONE"
        episode_id:       Active episode identifier from the simulator/ingestion layer.
        current_features: Latest feature row dict (35 keys from Stage 1 FE).
                          This row is appended to the episode buffer INSIDE the
                          forecasting node before running the algorithm.

    Returns:
        Formatted result dict ready for AgentState update:
        {
          # AgentState.forecast (full trajectory payload)
          "forecast": {
              "failure_mode":        str,
              "primary_metric":      str,
              "algorithm":           str,
              "critical_threshold":  float,
              "direction":           str,
              "history_steps":       int,
              "forecast_steps":      int,
              "current_value":       float,
              "predictions":         list[float],
              "timestamps_min":      list[float],
              "secondary_check":     dict | None,
          },
          # AgentState.time_to_failure
          "time_to_failure":       float | None,   # minutes until breach
          # AgentState.forecast_confidence
          "forecast_confidence":   float,           # 0.0–1.0
          # Convenience flag
          "threshold_crossed":     bool,
        }
        Returns {} for NONE mode.
    """
    # NONE mode → no compute
    if not failure_mode or failure_mode == "NONE":
        return {}

    # Update episode registry heartbeat
    _heartbeat(episode_id, failure_mode)

    # Dispatch to the correct forecasting node
    fn  = _ROUTER.get(failure_mode, forecast_latency_spike)  # fallback: ARIMA/p99
    raw = fn(episode_id=episode_id, current_features=current_features)

    return _format_result(raw)


# =============================================================================
# PUBLIC API — BATCH REPLAY / BACKFILL
# =============================================================================

def route_forecast_batch(
    failure_mode: str,
    episode_id:   str,
    feature_rows: list[dict[str, Any]],
) -> list[dict]:
    """
    Process a chronological list of feature rows for one episode.

    Use this for:
    - Offline replay of historical episodes.
    - Backfilling TTF labels for training data.
    - Unit testing with synthetic trajectories.

    Each row is fed into route_forecast() in order. Only the final result
    is meaningful for TTF; intermediate results show the TTF evolution.

    Args:
        failure_mode:  Failure mode for the entire episode.
        episode_id:    Episode identifier.
        feature_rows:  List of feature row dicts, oldest → newest.

    Returns:
        List of formatted result dicts, one per input row.
        The last entry contains the TTF after consuming all rows.
    """
    results = []
    for row in feature_rows:
        result = route_forecast(failure_mode, episode_id, row)
        results.append(result)
    return results


# =============================================================================
# PUBLIC API — EPISODE LIFECYCLE
# =============================================================================

def open_episode(episode_id: str, failure_mode: str = "NONE") -> None:
    """
    Register a new episode with the router.

    Called when the ingestion layer detects a new episode_id.
    Optional — route_forecast() will register on first call anyway.

    Args:
        episode_id:   New episode identifier.
        failure_mode: Initial mode (usually "NONE" at episode start).
    """
    _EPISODE_REGISTRY[episode_id] = {
        "failure_mode": failure_mode,
        "open_time":    time.time(),
        "last_seen":    time.time(),
        "cycle_count":  0,
    }


def close_episode(episode_id: str) -> None:
    """
    Close an episode: free its buffer and remove it from the registry.

    Called when the episode_id changes (episode rollover) or when the
    episode ends normally.

    Args:
        episode_id: Episode to close.
    """
    clear_episode(episode_id)
    _EPISODE_REGISTRY.pop(episode_id, None)


def expire_stale_episodes() -> list[str]:
    """
    Scan registry for episodes not seen in _EPISODE_TTL_S seconds.
    Closes and returns the list of expired episode_ids.

    Call this periodically (e.g. every 60 seconds) to reclaim memory.
    """
    now     = time.time()
    expired = [
        ep_id for ep_id, meta in _EPISODE_REGISTRY.items()
        if (now - meta["last_seen"]) > _EPISODE_TTL_S
    ]
    for ep_id in expired:
        close_episode(ep_id)
    return expired


# =============================================================================
# PUBLIC API — OBSERVABILITY
# =============================================================================

def active_episode_summary() -> list[dict]:
    """
    Return a summary view of all currently active episodes.

    Useful for monitoring dashboards and health-check endpoints.

    Returns:
        List of dicts:
        [
          {
            "episode_id":    str,
            "failure_mode":  str,
            "cycle_count":   int,
            "buffer_length": int,
            "age_seconds":   float,
            "last_seen_s":   float,
          },
          ...
        ]
    """
    now = time.time()
    summary = []
    for ep_id, meta in _EPISODE_REGISTRY.items():
        summary.append({
            "episode_id":    ep_id,
            "failure_mode":  meta.get("failure_mode", "UNKNOWN"),
            "cycle_count":   meta.get("cycle_count", 0),
            "buffer_length": get_episode_length(ep_id),
            "age_seconds":   round(now - meta.get("open_time", now), 1),
            "last_seen_s":   round(now - meta.get("last_seen", now), 1),
        })
    return summary


def list_supported_modes() -> list[str]:
    """Return all failure mode strings that have a dedicated forecast node."""
    return list(_ROUTER.keys())


def is_mode_supported(failure_mode: str) -> bool:
    """Check if a failure mode string has a dedicated forecasting node."""
    return failure_mode in _ROUTER


# =============================================================================
# INTERNAL HELPERS
# =============================================================================

def _heartbeat(episode_id: str, failure_mode: str) -> None:
    """Update or create an episode registry entry."""
    if episode_id not in _EPISODE_REGISTRY:
        _EPISODE_REGISTRY[episode_id] = {
            "failure_mode": failure_mode,
            "open_time":    time.time(),
            "last_seen":    time.time(),
            "cycle_count":  0,
        }
    entry = _EPISODE_REGISTRY[episode_id]
    entry["failure_mode"] = failure_mode
    entry["last_seen"]    = time.time()
    entry["cycle_count"]  = entry.get("cycle_count", 0) + 1


def _format_result(raw: dict) -> dict:
    """
    Normalise the raw mode output into the AgentState-compatible output format.

    Supports two schemas:
      NEW (v2, multi-feature convergence schema from _mode_runner.py):
        Keys: failure_mode, algorithm_used, predictions (dict), feature_ttfs,
              time_to_failure, forecast_confidence, confidence_reason, ...
        → Passed through as-is under "forecast" with top-level convenience keys.

      LEGACY (v1, single-feature schema from base_forecaster.py):
        Keys: failure_mode, primary_metric, algorithm, predictions (list), ...
        → Normalised into the same output structure.

    Args:
        raw: Dict returned by a mode's forecast function.

    Returns:
        Formatted dict with:
          "forecast"            → full trajectory payload for AgentState.forecast
          "time_to_failure"     → float | None (seconds for v2, minutes for v1)
          "forecast_confidence" → float 0.0–1.0
          "confidence_reason"   → str explanation (v2 only, '' for v1)
          "threshold_crossed"   → bool
    """
    if not raw:
        return {}

    # ── Detect schema version ─────────────────────────────────────────────────
    is_v2 = "algorithm_used" in raw and "feature_ttfs" in raw

    if is_v2:
        # NEW schema — pass through entirely; router adds convenience top-level keys
        return {
            "forecast":            raw,
            "time_to_failure":     raw.get("time_to_failure"),
            "forecast_confidence": raw.get("forecast_confidence", 0.0),
            "confidence_reason":   raw.get("confidence_reason", ""),
            "threshold_crossed":   raw.get("threshold_crossed", False),
        }
    else:
        # LEGACY schema — extract known trajectory keys
        trajectory_keys = {
            "failure_mode", "primary_metric", "algorithm", "critical_threshold",
            "direction", "history_steps", "forecast_steps", "current_value",
            "predictions", "timestamps_min", "secondary_check",
        }
        forecast_payload = {k: raw[k] for k in trajectory_keys if k in raw}
        return {
            "forecast":            forecast_payload,
            "time_to_failure":     raw.get("time_to_failure"),
            "forecast_confidence": raw.get("forecast_confidence", 0.0),
            "confidence_reason":   "",
            "threshold_crossed":   raw.get("threshold_crossed", False),
        }
