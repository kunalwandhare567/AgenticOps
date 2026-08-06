"""
app_simulator/feature_engineering/metrics_features.py
=======================================================
Raw metric passthrough — NO re-aggregation.

The classifier receives the exact values produced by the generator.
No rolling mean, no slope, no std, no derived ratios.

Why passthrough?
  Each 2-second tick already has the physics-modeled values.
  Re-averaging over 10 cycles would corrupt features that were already
  computed correctly (e.g., a re-averaged P99 is not a meaningful P99).

Feature columns returned:
  27 raw metric columns (from RAW_METRIC_FEATURE_COLS in config.py):
    cpu_utilization, memory_utilization, heap_mb,
    db_p99, disk_read_latency, disk_write_latency,
    error_rate, gc_pause_p99, cache_hit_rate, cache_miss_rate,
    active_connections, network_errors,
    p50_latency, p95_latency, p99_latency,
    queue_lag, retry_count_per_request, rps,
    upstream_timeout_rate, circuit_breaker_state,
    http_4xx_rate, http_5xx_rate, iops_utilization,
    thread_pool_queue, cpu_saturation,
    db_connection_pool, db_connection_wait

Note: 'circuit_breaker_state' is stored as string ("open"/"half-open"/"closed").
      The classifier node encodes it to int before inference.
"""
from __future__ import annotations

import sqlite3

from Simulator.app_data_generator_for_offline.config import RAW_METRIC_FEATURE_COLS

# Map circuit_breaker_state string → int for classifier
CB_ENCODE = {"closed": 0, "half-open": 1, "open": 2}

# Mapping from metrics CSV/DB column names → our RAW_METRIC_FEATURE_COLS names
# The generators write these exact field names to the DB.
_DB_TO_FEATURE = {
    "cpu_utilization":         "cpu_utilization",
    "memory_utilization":      "memory_utilization",
    "heap_mb":                 "heap_mb",
    "db_p99":                  "db_p99",
    "disk_read_latency":       "disk_read_latency",
    "disk_write_latency":      "disk_write_latency",
    "error_rate":              "error_rate",
    "gc_pause_p99":            "gc_pause_p99",
    "cache_hit_rate":          "cache_hit_rate",
    "cache_miss_rate":         "cache_miss_rate",
    "active_connections":      "active_connections",
    "network_errors":          "network_errors",
    "p50_latency":             "p50_latency",
    "p95_latency":             "p95_latency",
    "p99_latency":             "p99_latency",
    "queue_lag":               "queue_lag",
    "retry_count_per_request": "retry_count_per_request",
    "rps":                     "rps",
    "upstream_timeout_rate":   "upstream_timeout_rate",
    "circuit_breaker_state":   "circuit_breaker_state",
    "http_4xx_rate":           "http_4xx_rate",
    "http_5xx_rate":           "http_5xx_rate",
    "iops_utilization":        "iops_utilization",
    "thread_pool_queue":       "thread_pool_queue",
    "cpu_saturation":          "cpu_saturation",
    "db_connection_pool":      "db_connection_pool",
    "db_connection_wait":      "db_connection_wait",
}


def passthrough_metrics(metric_row: dict) -> dict:
    """
    Extract the 27 raw metric features from a generator output dict.
    Circuit breaker state is kept as its original string value.
    Missing keys default to 0.

    Args:
        metric_row: Dict produced by MetricsGenerator.generate()

    Returns:
        Dict with exactly the 27 keys in RAW_METRIC_FEATURE_COLS.
    """
    result: dict = {}
    for col in RAW_METRIC_FEATURE_COLS:
        # Try direct key, then generator-specific key aliases
        val = metric_row.get(col, None)
        if val is None:
            # Handle generator field name aliases
            val = _alias_lookup(metric_row, col)
        result[col] = val if val is not None else 0
    return result


def encode_circuit_breaker(features: dict) -> dict:
    """
    Encode circuit_breaker_state string → int for classifier input.
    Modifies the dict in-place and returns it.
    """
    cb = features.get("circuit_breaker_state", "closed")
    features["circuit_breaker_state"] = CB_ENCODE.get(str(cb).lower(), 0)
    return features


def _alias_lookup(row: dict, col: str):
    """
    Handle known generator field name differences vs config column names.
    The metrics_generator.py uses some different key names.
    """
    aliases = {
        "heap_mb":                 ["heap_mb", "heap"],
        "gc_pause_p99":            ["gc_pause_p99", "gc_pause_ms", "gc_pause"],
        "db_p99":                  ["db_p99", "db_latency_mean", "db_latency"],
        "disk_read_latency":       ["disk_read_latency", "disk_read_mean", "disk_read_ms"],
        "disk_write_latency":      ["disk_write_latency", "disk_write_mean", "disk_write_ms"],
        "cache_hit_rate":          ["cache_hit_rate", "cache_hit_ratio"],
        "cache_miss_rate":         ["cache_miss_rate", "cache_miss_ratio"],
        "retry_count_per_request": ["retry_count_per_request", "retry_count"],
        "upstream_timeout_rate":   ["upstream_timeout_rate", "upstream_timeout"],
        "circuit_breaker_state":   ["circuit_breaker_state", "cb_state"],
        "http_4xx_rate":           ["http_4xx_rate", "http_4xx"],
        "http_5xx_rate":           ["http_5xx_rate", "http_5xx"],
        "iops_utilization":        ["iops_utilization", "iops"],
        "thread_pool_queue":       ["thread_pool_queue", "thread_pool_q"],
        "cpu_saturation":          ["cpu_saturation", "cpu_sat"],
        "db_connection_pool":      ["db_connection_pool", "db_conn_pool"],
        "db_connection_wait":      ["db_connection_wait", "db_conn_wait"],
        "active_connections":      ["active_connections", "active_conn"],
        "network_errors":          ["network_errors", "net_errors"],
        "queue_lag":               ["queue_lag"],
    }
    for alias in aliases.get(col, [col]):
        if alias in row:
            return row[alias]
    return None


# =============================================================================
# SQLite wrapper — used by orchestrator when reading from DB
# =============================================================================

def compute_metrics_features(
    conn: sqlite3.Connection,
    episode_id: str,
    timestamp: float,
    window: int = 30,
) -> dict:
    """
    Query the LATEST metric row for (episode_id, timestamp) from SQLite.
    Returns the 27 raw feature values.

    Args:
        conn:       SQLite connection.
        episode_id: Current episode ID.
        timestamp:  Current tick timestamp.
        window:     Unused (kept for API compatibility with old orchestrator).

    Returns:
        Dict with 27 raw metric features.
    """
    try:
        cur = conn.execute(
            """
            SELECT *
            FROM   metrics
            WHERE  episode_id = ?
              AND  timestamp  = ?
            LIMIT  1
            """,
            (episode_id, timestamp),
        )
        cols = [d[0] for d in cur.description]
        row  = cur.fetchone()
        if row is None:
            return {col: 0 for col in RAW_METRIC_FEATURE_COLS}
        row_dict = dict(zip(cols, row))
        return passthrough_metrics(row_dict)
    except Exception as exc:
        print(f"[MetricFE] ERROR querying metrics: {exc}")
        return {col: 0 for col in RAW_METRIC_FEATURE_COLS}
