"""RETRY_STORM — Retries amplify RPS 3-4x; thread pool + queue fill up."""
import numpy as np
from .base_scenario import BaseScenario
from .distributions import Dist
from ..state import SimulatorState
from ..config import BASE_RPS


class RetryStormScenario(BaseScenario):
    failure_mode = "RETRY_STORM"

    def onset_delay_s(self, rng: np.random.Generator) -> float:
        return float(rng.uniform(0, 10))

    def apply(self, state: SimulatorState, dist: Dist) -> SimulatorState:
        if state.elapsed_s < state.onset_delay_s:
            return self._apply_healthy(state, dist)

        retry                       = dist.retry_high()
        state.retry_count_per_request = retry
        state.rps                   = dist.rps_retry(retry, BASE_RPS)
        state.error_rate            = float(np.clip(dist.err_rate() + retry * 0.1, 0, 0.35))
        state.p99_latency           = float(np.clip(dist.rng.normal(80 + retry * 200, 30), 80, 800))
        state.p50_latency           = dist.p50()
        state.p95_latency           = dist.p95(state.p50_latency, state.p99_latency)
        state.heap_mb               = dist.heap()
        state.gc_pause_p99          = dist.gc_p99()
        state.cpu_utilization       = float(np.clip(dist.cpu_util() + retry * 30, 20, 95))
        state.cpu_saturation        = state.cpu_utilization / 100.0
        state.cache_miss_rate       = dist.cache_miss()
        state.cache_hit_rate        = 1.0 - state.cache_miss_rate
        state.db_p99                = float(np.clip(dist.db_p99() * (1 + retry), 20, 300))
        state.queue_lag             = int(np.clip(dist.queue_lag() + retry * 100, 0, 400))
        state.active_connections    = int(np.clip(dist.active_conn() * (1 + retry * 2), 20, 800))
        state.db_connection_pool    = float(np.clip(dist.db_conn_pool() + retry * 0.4, 0.1, 0.99))
        state.db_connection_wait    = float(np.clip(dist.db_conn_wait() + retry * 20, 0.5, 200))
        state.disk_read_latency     = dist.disk_read_lat()
        state.disk_write_latency    = dist.disk_write_lat()
        state.iops_utilization      = dist.iops_util()
        state.memory_utilization    = dist.mem_util(state.heap_mb)
        state.network_errors        = int(np.clip(dist.net_errors() + retry * 10, 0, 40))
        state.http_4xx_rate         = dist.http_4xx()
        state.http_5xx_rate         = float(np.clip(dist.http_5xx() + state.error_rate * 0.3, 0, 0.4))
        state.thread_pool_queue     = int(np.clip(state.rps / 10, 10, 500))
        state.upstream_timeout_rate = float(np.clip(dist.upstream_timeout() + retry * 0.05, 0, 0.3))
        state.circuit_breaker_state = dist.circuit_breaker(state.error_rate)
        return state
