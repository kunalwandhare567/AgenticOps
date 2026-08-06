"""QUEUE_BACKUP — Queue depth grows linearly ~1.8 items/sec; p99 follows."""
import numpy as np
from .base_scenario import BaseScenario
from .distributions import Dist
from ..state import SimulatorState


class QueueBackupScenario(BaseScenario):
    failure_mode = "QUEUE_BACKUP"

    def onset_delay_s(self, rng: np.random.Generator) -> float:
        return float(rng.uniform(0, 10))

    def apply(self, state: SimulatorState, dist: Dist) -> SimulatorState:
        if state.elapsed_s < state.onset_delay_s:
            return self._apply_healthy(state, dist)

        effective_elapsed = state.elapsed_s - state.onset_delay_s
        state.queue_lag             = dist.queue_lag_backup(effective_elapsed)
        state.p99_latency           = float(np.clip(dist.rng.normal(90 + state.queue_lag * 1.5, 20), 80, 2000))
        state.p50_latency           = float(np.clip(dist.p50() + state.queue_lag * 0.3, 80, 500))
        state.p95_latency           = dist.p95(state.p50_latency, state.p99_latency)
        state.heap_mb               = dist.heap()
        state.gc_pause_p99          = dist.gc_p99()
        state.cpu_utilization       = dist.cpu_util()
        state.cpu_saturation        = state.cpu_utilization / 100.0
        state.error_rate            = dist.err_rate()
        state.cache_miss_rate       = dist.cache_miss()
        state.cache_hit_rate        = 1.0 - state.cache_miss_rate
        state.db_p99                = dist.db_p99()
        state.retry_count_per_request = dist.retry_cnt()
        state.rps                   = dist.rps()
        state.active_connections    = int(np.clip(dist.active_conn() + state.queue_lag * 0.5, 20, 500))
        state.db_connection_pool    = dist.db_conn_pool()
        state.db_connection_wait    = float(np.clip(dist.db_conn_wait() + state.queue_lag * 0.2, 0.5, 200))
        state.disk_read_latency     = dist.disk_read_lat()
        state.disk_write_latency    = dist.disk_write_lat()
        state.iops_utilization      = dist.iops_util()
        state.memory_utilization    = dist.mem_util(state.heap_mb)
        state.network_errors        = dist.net_errors()
        state.http_4xx_rate         = dist.http_4xx()
        state.http_5xx_rate         = dist.http_5xx()
        state.thread_pool_queue     = int(np.clip(state.queue_lag * 2, 0, 500))
        state.upstream_timeout_rate = dist.upstream_timeout()
        state.circuit_breaker_state = dist.circuit_breaker(state.error_rate)
        return state
