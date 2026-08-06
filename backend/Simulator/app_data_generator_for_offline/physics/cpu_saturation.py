"""CPU_SATURATION — CPU 70-100%, thread pool fills, p99 climbs."""
import numpy as np
from .base_scenario import BaseScenario
from .distributions import Dist
from ..state import SimulatorState


class CpuSaturationScenario(BaseScenario):
    failure_mode = "CPU_SATURATION"

    def onset_delay_s(self, rng: np.random.Generator) -> float:
        return float(rng.uniform(0, 8))

    def apply(self, state: SimulatorState, dist: Dist) -> SimulatorState:
        if state.elapsed_s < state.onset_delay_s:
            return self._apply_healthy(state, dist)

        state.cpu_utilization       = dist.cpu_sat_util()
        state.cpu_saturation        = state.cpu_utilization / 100.0
        state.p99_latency           = dist.p99_cpu_sat(state.cpu_utilization)
        state.p50_latency           = float(np.clip(dist.p50() + (state.cpu_utilization - 35) * 2, 80, 500))
        state.p95_latency           = dist.p95(state.p50_latency, state.p99_latency)
        state.heap_mb               = dist.heap()
        state.gc_pause_p99          = dist.gc_p99()
        state.error_rate            = float(np.clip(dist.err_rate() + (state.cpu_utilization - 70) * 0.002, 0, 0.15))
        state.cache_miss_rate       = dist.cache_miss()
        state.cache_hit_rate        = 1.0 - state.cache_miss_rate
        state.db_p99                = dist.db_p99()
        state.queue_lag             = int(np.clip(dist.queue_lag() + state.cpu_utilization * 0.5, 0, 200))
        state.retry_count_per_request = dist.retry_cnt()
        state.rps                   = dist.rps()
        state.active_connections    = int(np.clip(dist.active_conn() + state.cpu_utilization * 0.5, 20, 500))
        state.db_connection_pool    = dist.db_conn_pool()
        state.db_connection_wait    = float(np.clip(dist.db_conn_wait() + state.cpu_utilization * 0.1, 0.5, 100))
        state.disk_read_latency     = dist.disk_read_lat()
        state.disk_write_latency    = dist.disk_write_lat()
        state.iops_utilization      = dist.iops_util()
        state.memory_utilization    = dist.mem_util(state.heap_mb)
        state.network_errors        = dist.net_errors()
        state.http_4xx_rate         = dist.http_4xx()
        state.http_5xx_rate         = float(np.clip(dist.http_5xx() + (state.cpu_utilization - 70) * 0.001, 0, 0.1))
        state.thread_pool_queue     = int(np.clip(dist.thread_pool_q() + state.cpu_utilization * 1.5, 0, 500))
        state.upstream_timeout_rate = float(np.clip(dist.upstream_timeout() + (state.cpu_utilization - 70) * 0.002, 0, 0.3))
        state.circuit_breaker_state = dist.circuit_breaker(state.error_rate)
        return state
