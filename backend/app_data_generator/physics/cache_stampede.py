"""CACHE_STAMPEDE — Cache miss 85-99%, DB overloaded via M/M/1 queueing model."""
import numpy as np
from .base_scenario import BaseScenario
from .distributions import Dist
from ..state import SimulatorState
from ..config import BASE_RPS, DB_CAPACITY_RPS


class CacheStampedeScenario(BaseScenario):
    failure_mode = "CACHE_STAMPEDE"

    def apply(self, state: SimulatorState, dist: Dist) -> SimulatorState:
        state.cache_miss_rate       = dist.cache_miss_stampede()
        state.cache_hit_rate        = 1.0 - state.cache_miss_rate
        state.db_p99                = dist.db_p99_mm1(state.cache_miss_rate, BASE_RPS, DB_CAPACITY_RPS)
        state.p99_latency           = float(np.clip(state.db_p99 + dist.rng.normal(40, 10), state.db_p99 * 0.9, state.db_p99 * 1.2))
        state.p50_latency           = dist.p50()
        state.p95_latency           = dist.p95(state.p50_latency, state.p99_latency)
        state.heap_mb               = dist.heap()
        state.gc_pause_p99          = dist.gc_p99()
        state.cpu_utilization       = float(np.clip(dist.cpu_util() + dist.rng.uniform(10, 25), 30, 85))
        state.cpu_saturation        = state.cpu_utilization / 100.0
        state.error_rate            = float(np.clip(dist.rng.beta(3, 12), 0.10, 0.35))
        state.queue_lag             = dist.queue_lag()
        state.retry_count_per_request = dist.retry_cnt()
        state.rps                   = dist.rps()
        state.active_connections    = int(np.clip(dist.active_conn() + state.cache_miss_rate * 200, 20, 500))
        state.db_connection_pool    = float(np.clip(dist.db_conn_pool() + state.cache_miss_rate * 0.5, 0.1, 0.99))
        state.db_connection_wait    = float(np.clip(state.db_p99 * 0.4, 1, 500))
        state.disk_read_latency     = dist.disk_read_lat()
        state.disk_write_latency    = dist.disk_write_lat()
        state.iops_utilization      = float(np.clip(dist.iops_util() + state.cache_miss_rate * 0.3, 0.05, 0.95))
        state.memory_utilization    = dist.mem_util(state.heap_mb)
        state.network_errors        = dist.net_errors()
        state.http_4xx_rate         = dist.http_4xx()
        state.http_5xx_rate         = float(np.clip(dist.http_5xx() + state.error_rate * 0.3, 0, 0.3))
        state.thread_pool_queue     = int(np.clip(dist.thread_pool_q() + state.cache_miss_rate * 50, 0, 200))
        state.upstream_timeout_rate = dist.upstream_timeout()
        state.circuit_breaker_state = dist.circuit_breaker(state.error_rate)
        return state
