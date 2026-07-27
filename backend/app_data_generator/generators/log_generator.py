"""
app_simulator/generators/log_generator.py
==========================================
Produces one 8-column log dict per tick from SimulatorState.

Reuses build_log_msg() and LOG_LEVELS / EXCEPTION_MAP verbatim from
config.py (which copied them from generate_full_dataset.py lines 197-231).

SYNCHRONIZATION: reads state.timestamp which was set ONCE at tick start.
"""
from __future__ import annotations
from ..state import SimulatorState
from ..config import LOG_LEVELS, EXCEPTION_MAP, build_log_msg


class LogGenerator:
    """Stateless — call generate(state) every tick."""

    def generate(self, state: SimulatorState) -> dict:
        """
        Assemble the 8-column log dict.
        log_message content is mode-specific and driven by current metric values.
        """
        log_vals = {
            "cpu":  state.cpu_utilization,
            "p99":  state.p99_latency,
            "err":  state.error_rate * 100,
            "heap": state.heap_mb,
            "gc":   state.gc_pause_p99,
            "h5":   state.http_5xx_rate * 100,
            "db":   state.db_p99,
            "miss": state.cache_miss_rate * 100,
            "lag":  state.queue_lag,
            "tp":   state.thread_pool_queue,
            "ut":   state.upstream_timeout_rate * 100,
            "rc":   state.retry_count_per_request,
            "rps":  state.rps,
            "dr":   state.disk_read_latency,
            "iops": state.iops_utilization * 100,
        }

        return {
            "episode_id":    state.episode_id,
            "failure_mode":  state.failure_mode,
            "service":       state.service,
            "elapsed_s":     round(state.elapsed_s, 1),
            "timestamp":     state.timestamp,           # ← SYNC FIELD
            "log_level":     LOG_LEVELS.get(state.failure_mode, "INFO"),
            "exception_type":EXCEPTION_MAP.get(state.failure_mode, ""),
            "log_message":   build_log_msg(state.failure_mode, log_vals),
        }
