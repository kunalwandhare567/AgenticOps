"""DB_SLOWDOWN — DB query latency grows exponentially, p99 follows."""
import numpy as np
from .base_scenario import BaseScenario
from .distributions import Dist
from ..state import SimulatorState


class DbSlowdownScenario(BaseScenario):
    failure_mode = "DB_SLOWDOWN"

    def onset_delay_s(self, rng: np.random.Generator) -> float:
        # Cure fraction: 85% remain censored, 15% fail early
        if rng.random() < 0.15:
            return float(rng.uniform(0, 30))
        return 0.0

    def apply(self, state: SimulatorState, dist: Dist) -> SimulatorState:
        if state.onset_delay_s > 0 and state.elapsed_s < state.onset_delay_s:
            return self._apply_healthy(state, dist)

        effective_elapsed           = state.elapsed_s - state.onset_delay_s if state.onset_delay_s > 0 else state.elapsed_s
        state.db_p99                = dist.db_slow(effective_elapsed)
        state.p99_latency           = float(np.clip(state.db_p99 * 1.2 + dist.rng.normal(30, 10), state.db_p99, state.db_p99 * 1.5))
        state.p50_latency           = dist.p50()
        state.p95_latency           = dist.p95(state.p50_latency, state.p99_latency)
        state.heap_mb               = dist.heap()
        state.gc_pause_p99          = dist.gc_p99()
        state.cpu_utilization       = dist.cpu_util()
        state.cpu_saturation        = state.cpu_utilization / 100.0
        state.error_rate            = float(np.clip(dist.err_rate() + (state.db_p99 - 80) * 0.0002, 0, 0.15))
        state.cache_miss_rate       = dist.cache_miss()
        state.cache_hit_rate        = 1.0 - state.cache_miss_rate
        state.queue_lag             = dist.queue_lag()
        state.retry_count_per_request = dist.retry_cnt()
        state.rps                   = dist.rps()
        state.active_connections    = int(np.clip(dist.active_conn() + state.db_p99 * 0.05, 20, 500))
        state.db_connection_pool    = float(np.clip(dist.db_conn_pool() + state.db_p99 / 2000, 0.1, 0.99))
        state.db_connection_wait    = float(np.clip(state.db_p99 * 0.3 + dist.rng.normal(0, 10), 1, 500))
        state.disk_read_latency     = dist.disk_read_lat()
        state.disk_write_latency    = float(np.clip(dist.disk_write_lat() + state.db_p99 * 0.01, 2, 100))
        state.iops_utilization      = float(np.clip(dist.iops_util() + state.db_p99 / 5000, 0.05, 0.95))
        state.memory_utilization    = dist.mem_util(state.heap_mb)
        state.network_errors        = dist.net_errors()
        state.http_4xx_rate         = dist.http_4xx()
        state.http_5xx_rate         = float(np.clip(dist.http_5xx() + state.error_rate * 0.5, 0, 0.2))
        state.thread_pool_queue     = int(np.clip(dist.thread_pool_q() + state.db_p99 * 0.01, 0, 100))
        state.upstream_timeout_rate = float(np.clip(dist.upstream_timeout() + state.db_p99 / 10000, 0, 0.3))
        state.circuit_breaker_state = dist.circuit_breaker(state.error_rate)
        return state
