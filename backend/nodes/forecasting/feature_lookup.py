"""
d:/Before_done/forecasting_node/feature_lookup.py
===================================================
Bidirectional feature name alias table.

The pipeline uses new names (e.g. 'heap_mb', 'db_p99', 'p99_latency') while
older test baselines and some buffer rows may use legacy names (e.g. 'heap_mean',
'db_latency_mean', 'p99_mean').

This module resolves either direction transparently so that mode files never
need to know which naming convention the incoming feature dict uses.

Public API:
    get_feature(row: dict, key: str, default: float = 0.0) -> float
        Read a single value from a feature dict, trying all alias names.

    get_feature_series(episode_id: str, key: str) -> list[float]
        Read the accumulated history series for a feature from the buffer,
        trying all alias names in order.
"""
from __future__ import annotations

from typing import Any

from .buffer import get_metric_series

# ---------------------------------------------------------------------------
# Alias table
# ---------------------------------------------------------------------------
# Each entry maps one canonical name to a list of alternative names.
# The lookup tries the canonical name first, then each alias in order.
# The table is bidirectional: if 'heap_mb' has alias 'heap_mean', then
# looking up 'heap_mean' will also try 'heap_mb' as a secondary name.
# ---------------------------------------------------------------------------
_ALIASES: dict[str, list[str]] = {
    # Memory
    "heap_mb":                  ["heap_mean", "heap_max"],
    "memory_utilization":       ["memory_mean", "memory_max", "memory_growth_rate"],

    # CPU
    "cpu_utilization":          ["cpu_mean", "cpu_max", "cpu_std"],
    "cpu_saturation":           ["cpu_saturation_mean"],

    # Latency
    "p99_latency":              ["p99_mean"],
    "p95_latency":              ["p95_mean"],
    "p50_latency":              ["p50_mean"],

    # Database
    "db_p99":                   ["db_latency_mean"],
    "active_connections":       ["db_connections_max"],
    "db_connection_wait":       ["slow_query_count", "lock_wait_time"],
    "db_connection_pool":       ["db_connections_max"],

    # Cache
    "cache_hit_rate":           ["cache_hit_ratio"],
    "cache_miss_rate":          ["cache_miss_ratio"],

    # Throughput / RPS
    "rps":                      ["throughput_mean", "throughput"],

    # Error rate
    "error_rate":               ["error_rate_mean", "error_rate_max"],

    # GC
    "gc_pause_p99":             ["gc_pause_mean", "gc_pause_max"],

    # Disk I/O
    "disk_read_latency":        ["disk_read_mean"],
    "disk_write_latency":       ["disk_write_mean"],
    "iops_utilization":         ["disk_utilization"],
    "queue_lag":                ["disk_queue_depth"],

    # Network / upstream
    "network_errors":           ["network_errors_mean"],
    "upstream_timeout_rate":    ["timeout_rate"],
    "retry_count_per_request":  ["retry_count"],

    # HTTP codes
    "http_5xx_rate":            ["error_rate_mean"],
    "http_4xx_rate":            [],
    "thread_pool_queue":        [],
    "circuit_breaker_state":    ["cb_open_ratio"],
}

# Build reverse lookup so old names also resolve to new ones
_REVERSE: dict[str, str] = {}
for _canonical, _aliases in _ALIASES.items():
    for _alias in _aliases:
        if _alias not in _REVERSE:
            _REVERSE[_alias] = _canonical


def _all_candidates(key: str) -> list[str]:
    """
    Return all candidate names to try for a given key.
    Order: key → canonical (if key is alias) → all aliases of canonical.
    """
    candidates = [key]
    # If key is itself an alias, try the canonical first
    canonical = _REVERSE.get(key)
    if canonical and canonical not in candidates:
        candidates.append(canonical)
        # Then all aliases of that canonical
        for a in _ALIASES.get(canonical, []):
            if a not in candidates:
                candidates.append(a)
    # If key is canonical, add its aliases
    for a in _ALIASES.get(key, []):
        if a not in candidates:
            candidates.append(a)
    return candidates


def get_feature(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    """
    Read a feature value from a dict, resolving aliases.

    Tries the exact key first, then all aliases in order.
    Returns `default` if none found or value is not numeric.

    Args:
        row:     Feature row dict (current_features from the pipeline).
        key:     Canonical or legacy feature name.
        default: Value to return if not found (default 0.0).

    Returns:
        float value of the feature.
    """
    for candidate in _all_candidates(key):
        val = row.get(candidate)
        if val is not None:
            try:
                f = float(val)
                import math
                if not math.isnan(f) and not math.isinf(f):
                    return f
            except (TypeError, ValueError):
                continue
    return default


def get_feature_series(episode_id: str, key: str) -> list[float]:
    """
    Read the accumulated history series for a feature from the buffer,
    trying the key and all its aliases.

    Args:
        episode_id: Active episode identifier.
        key:        Canonical or legacy feature name.

    Returns:
        List of float values (oldest → newest). Returns [] if not found.
    """
    for candidate in _all_candidates(key):
        series = get_metric_series(episode_id, candidate)
        if series:
            return series
    return []
