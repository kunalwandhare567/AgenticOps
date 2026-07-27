"""
app_simulator/generators/metrics_generator.py
===============================================
Produces a 33-column metric dict from SimulatorState.

Schema is IDENTICAL to generate_full_dataset.py METRIC_FIELDS — the same
column names, same ordering. Zero schema drift between the batch generator
and the live simulator.

SYNCHRONIZATION: reads state.timestamp which was set ONCE at tick start.
"""
from __future__ import annotations
from ..state import SimulatorState


class MetricsGenerator:
    """Stateless — call generate(state) every tick."""

    def generate(self, state: SimulatorState) -> dict:
        """
        Assemble the 33-column metric dict.
        All fields read directly from state — no computation here.
        Computation happened in scenario.apply().
        """
        return {
            # ── Identity / sync (6) ──────────────────────────────────────────
            "episode_id":             state.episode_id,
            "failure_mode":           state.failure_mode,
            "service":                state.service,
            "source":                 state.source,
            "elapsed_s":              round(state.elapsed_s, 1),
            "timestamp":              state.timestamp,        # ← SYNC FIELD

            # ── Metric fields (27) ───────────────────────────────────────────
            "active_connections":     state.active_connections,
            "cache_hit_rate":         round(state.cache_hit_rate, 4),
            "cache_miss_rate":        round(state.cache_miss_rate, 4),
            "circuit_breaker_state":  state.circuit_breaker_state,
            "cpu_saturation":         round(state.cpu_saturation, 4),
            "cpu_utilization":        round(state.cpu_utilization, 2),
            "db_connection_pool":     round(state.db_connection_pool, 4),
            "db_connection_wait":     round(state.db_connection_wait, 2),
            "db_p99":                 round(state.db_p99, 2),
            "disk_read_latency":      round(state.disk_read_latency, 2),
            "disk_write_latency":     round(state.disk_write_latency, 2),
            "error_rate":             round(state.error_rate, 4),
            "gc_pause_p99":           round(state.gc_pause_p99, 2),
            "heap_mb":                round(state.heap_mb, 1),
            "http_4xx_rate":          round(state.http_4xx_rate, 4),
            "http_5xx_rate":          round(state.http_5xx_rate, 4),
            "iops_utilization":       round(state.iops_utilization, 4),
            "memory_utilization":     round(state.memory_utilization, 4),
            "network_errors":         state.network_errors,
            "p50_latency":            round(state.p50_latency, 2),
            "p95_latency":            round(state.p95_latency, 2),
            "p99_latency":            round(state.p99_latency, 2),
            "queue_lag":              state.queue_lag,
            "retry_count_per_request":round(state.retry_count_per_request, 4),
            "rps":                    round(state.rps, 2),
            "thread_pool_queue":      state.thread_pool_queue,
            "upstream_timeout_rate":  round(state.upstream_timeout_rate, 4),
        }
