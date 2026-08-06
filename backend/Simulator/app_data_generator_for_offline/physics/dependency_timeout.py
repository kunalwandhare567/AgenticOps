"""DEPENDENCY_TIMEOUT — External service takes 2-30s; upstream_timeout_rate high."""
import numpy as np
from .base_scenario import BaseScenario
from .distributions import Dist
from ..state import SimulatorState


class DependencyTimeoutScenario(BaseScenario):
    failure_mode = "DEPENDENCY_TIMEOUT"

    def onset_delay_s(self, rng: np.random.Generator) -> float:
        return float(rng.uniform(0, 8))

    def apply(self, state: SimulatorState, dist: Dist) -> SimulatorState:
        if state.elapsed_s < state.onset_delay_s:
            return self._apply_healthy(state, dist)

        dep_rate                    = dist.timeout_rate()
        state.upstream_timeout_rate = dep_rate
        state.error_rate            = float(np.clip(dep_rate * 0.8, 0, 0.65))
        state.p99_latency           = float(np.clip(dist.rng.normal(300 + dep_rate * 1000, 80), 200, 3000))
        state.p50_latency           = dist.p50()
        state.p95_latency           = dist.p95(state.p50_latency, state.p99_latency)
        state.heap_mb               = dist.heap()
        state.gc_pause_p99          = dist.gc_p99()
        state.cpu_utilization       = dist.cpu_util()
        state.cpu_saturation        = state.cpu_utilization / 100.0
        state.cache_miss_rate       = dist.cache_miss()
        state.cache_hit_rate        = 1.0 - state.cache_miss_rate
        state.db_p99                = dist.db_p99()
        state.queue_lag             = dist.queue_lag()
        state.retry_count_per_request = float(np.clip(dep_rate * 0.5, 0, 0.5))
        state.rps                   = dist.rps()
        state.active_connections    = dist.active_conn()
        state.db_connection_pool    = dist.db_conn_pool()
        state.db_connection_wait    = dist.db_conn_wait()
        state.disk_read_latency     = dist.disk_read_lat()
        state.disk_write_latency    = dist.disk_write_lat()
        state.iops_utilization      = dist.iops_util()
        state.memory_utilization    = dist.mem_util(state.heap_mb)
        state.network_errors        = int(np.clip(dist.net_errors() + dep_rate * 30, 0, 60))
        state.http_4xx_rate         = float(np.clip(dist.http_4xx() + dep_rate * 0.1, 0, 0.2))
        state.http_5xx_rate         = float(np.clip(dist.http_5xx() + dep_rate * 0.4, 0, 0.6))
        state.thread_pool_queue     = dist.thread_pool_q()
        state.circuit_breaker_state = dist.circuit_breaker(state.error_rate)
        return state
