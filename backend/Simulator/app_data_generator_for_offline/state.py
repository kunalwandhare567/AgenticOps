"""
app_simulator/state.py
=======================
SimulatorState — shared dataclass passed to all generators each tick.

SYNCHRONIZATION GUARANTEE:
    state.timestamp is set ONCE at the top of each tick loop,
    BEFORE any generator (metrics/logs/traces) is called.
    All three generators read state.timestamp.
    It is structurally impossible for M/L/T to have different
    timestamps for the same simulation tick.
"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class SimulatorState:
    # ── Identity / sync fields (SET ONCE per tick) ───────────────────────────
    episode_id:       str   = ""
    failure_mode:     str   = "NONE"
    step:             int   = 0
    elapsed_s:        float = 0.0
    timestamp:        float = 0.0       # ← THE SYNCHRONIZATION LOCK
    service:          str   = "auth-service"
    source:           str   = "python-simulator"
    service_version:  str   = "v1.0.0"
    onset_delay_s:    float = 0.0       # ← Episode-level onset delay for realistic TTF spread

    # ── Metric fields (mutated by scenario.apply() each tick) ────────────────
    # All 27 metric columns correspond 1-to-1 with METRIC_FIELDS in config.py
    # (minus the 6 identity/sync fields listed above)

    active_connections:       int   = 60
    cache_hit_rate:           float = 0.95
    cache_miss_rate:          float = 0.05
    circuit_breaker_state:    str   = "closed"
    cpu_saturation:           float = 0.22
    cpu_utilization:          float = 22.0
    db_connection_pool:       float = 0.35
    db_connection_wait:       float = 5.0
    db_p99:                   float = 20.0
    disk_read_latency:        float = 4.5
    disk_write_latency:       float = 6.0
    error_rate:               float = 0.005
    gc_pause_p99:             float = 15.0
    heap_mb:                  float = 512.0
    http_4xx_rate:            float = 0.005
    http_5xx_rate:            float = 0.002
    iops_utilization:         float = 0.25
    memory_utilization:       float = 0.25
    network_errors:           int   = 1
    p50_latency:              float = 90.0
    p95_latency:              float = 105.0
    p99_latency:              float = 120.0
    queue_lag:                int   = 0
    retry_count_per_request:  float = 0.02
    rps:                      float = 200.0
    thread_pool_queue:        int   = 5
    upstream_timeout_rate:    float = 0.005


# Re-export PipelineState so all node files can use:
#     from Simulator.app_data_generator_for_offline.state import PipelineState
from nodes.collect.state import PipelineState  # noqa: F401

__all__ = ["SimulatorState", "PipelineState"]
