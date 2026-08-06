"""MEMORY_LEAK — Heap grows 12MB/min; GC pauses and p99 worsen over time."""
import numpy as np
from .base_scenario import BaseScenario
from .distributions import Dist
from ..state import SimulatorState


class MemoryLeakScenario(BaseScenario):
    failure_mode = "MEMORY_LEAK"

    def onset_delay_s(self, rng: np.random.Generator) -> float:
        # Cure fraction: 85% remain censored, 15% fail early
        if rng.random() < 0.15:
            return float(rng.uniform(0, 30))
        return 0.0

    def apply(self, state: SimulatorState, dist: Dist) -> SimulatorState:
        if state.onset_delay_s > 0 and state.elapsed_s < state.onset_delay_s:
            return self._apply_healthy(state, dist)

        effective_elapsed           = state.elapsed_s - state.onset_delay_s if state.onset_delay_s > 0 else state.elapsed_s
        state.heap_mb               = dist.heap_leak(effective_elapsed)
        state.gc_pause_p99          = dist.gc_leak(state.heap_mb)
        state.cpu_utilization       = float(np.clip(dist.cpu_util() + (state.heap_mb - 512) / 80, 15, 90))
        state.cpu_saturation        = state.cpu_utilization / 100.0
        state.p99_latency           = float(np.clip(dist.p99() + state.gc_pause_p99 * 0.4, 80, 2000))
        state.p50_latency           = dist.p50()
        state.p95_latency           = dist.p95(state.p50_latency, state.p99_latency)
        state.error_rate            = dist.err_rate()
        state.cache_miss_rate       = dist.cache_miss()
        state.cache_hit_rate        = 1.0 - state.cache_miss_rate
        state.db_p99                = dist.db_p99()
        state.memory_utilization    = dist.mem_util(state.heap_mb)
        state.db_connection_pool    = float(np.clip(dist.db_conn_pool() + state.heap_mb / 5000, 0.1, 0.9))
        state.db_connection_wait    = dist.db_conn_wait()
        state.disk_read_latency     = dist.disk_read_lat()
        state.disk_write_latency    = dist.disk_write_lat()
        state.iops_utilization      = dist.iops_util()
        state.network_errors        = dist.net_errors()
        state.http_4xx_rate         = dist.http_4xx()
        state.http_5xx_rate         = float(np.clip(dist.http_5xx() + state.gc_pause_p99 / 10000, 0, 0.05))
        state.thread_pool_queue     = int(np.clip(dist.thread_pool_q() + state.gc_pause_p99 / 50, 0, 200))
        state.upstream_timeout_rate = dist.upstream_timeout()
        state.circuit_breaker_state = dist.circuit_breaker(state.error_rate)
        state.rps                   = dist.rps()
        state.queue_lag             = dist.queue_lag()
        state.retry_count_per_request = dist.retry_cnt()
        state.active_connections    = dist.active_conn()
        return state
