"""DISK_IO_SATURATION — Disk wait 50-2000ms; IOPS at 80-100%; DB follows."""
import numpy as np
from .base_scenario import BaseScenario
from .distributions import Dist
from ..state import SimulatorState


class DiskIoSaturationScenario(BaseScenario):
    failure_mode = "DISK_IO_SATURATION"

    def onset_delay_s(self, rng: np.random.Generator) -> float:
        return float(rng.uniform(0, 4))

    def apply(self, state: SimulatorState, dist: Dist) -> SimulatorState:
        if state.elapsed_s < state.onset_delay_s:
            return self._apply_healthy(state, dist)

        effective_elapsed           = state.elapsed_s - state.onset_delay_s
        state.disk_read_latency     = dist.disk_lat_sat()
        state.disk_write_latency    = float(np.clip(state.disk_read_latency * dist.rng.uniform(0.9, 1.2), 50, 2500))
        state.iops_utilization      = dist.iops_sat()
        state.db_p99                = dist.db_slow_disk(effective_elapsed)
        state.p99_latency           = float(np.clip(state.db_p99 * 1.3 + state.disk_read_latency * 0.1, 80, 5000))
        state.p50_latency           = dist.p50()
        state.p95_latency           = dist.p95(state.p50_latency, state.p99_latency)
        state.heap_mb               = dist.heap()
        state.gc_pause_p99          = dist.gc_p99()
        state.cpu_utilization       = float(np.clip(dist.cpu_util() + state.iops_utilization * 10, 15, 80))
        state.cpu_saturation        = state.cpu_utilization / 100.0
        state.error_rate            = float(np.clip(dist.err_rate() + state.iops_utilization * 0.05, 0, 0.2))
        state.cache_miss_rate       = dist.cache_miss()
        state.cache_hit_rate        = 1.0 - state.cache_miss_rate
        state.queue_lag             = dist.queue_lag()
        state.retry_count_per_request = dist.retry_cnt()
        state.rps                   = dist.rps()
        state.active_connections    = dist.active_conn()
        state.db_connection_pool    = float(np.clip(dist.db_conn_pool() + state.iops_utilization * 0.3, 0.1, 0.99))
        state.db_connection_wait    = float(np.clip(state.db_p99 * 0.3, 1, 500))
        state.memory_utilization    = dist.mem_util(state.heap_mb)
        state.network_errors        = dist.net_errors()
        state.http_4xx_rate         = dist.http_4xx()
        state.http_5xx_rate         = float(np.clip(dist.http_5xx() + state.error_rate * 0.2, 0, 0.2))
        state.thread_pool_queue     = int(np.clip(dist.thread_pool_q() + state.iops_utilization * 20, 0, 200))
        state.upstream_timeout_rate = float(np.clip(dist.upstream_timeout() + state.error_rate * 0.1, 0, 0.2))
        state.circuit_breaker_state = dist.circuit_breaker(state.error_rate)
        return state
