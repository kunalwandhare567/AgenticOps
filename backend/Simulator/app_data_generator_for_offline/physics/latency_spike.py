"""LATENCY_SPIKE — 8% chance of GC event causing p99 spike each step."""
import numpy as np
from .base_scenario import BaseScenario
from .distributions import Dist
from ..state import SimulatorState


class LatencySpikeScenario(BaseScenario):
    failure_mode = "LATENCY_SPIKE"

    def onset_delay_s(self, rng: np.random.Generator) -> float:
        # Cure fraction: 85% remain censored, 15% fail early
        if rng.random() < 0.15:
            return float(rng.uniform(0, 30))
        return 0.0

    def apply(self, state: SimulatorState, dist: Dist) -> SimulatorState:
        if state.onset_delay_s > 0 and state.elapsed_s < state.onset_delay_s:
            return self._apply_healthy(state, dist)

        gc_fired = dist.gc_event(0.08)
        state.heap_mb            = dist.heap()
        state.cpu_utilization    = dist.cpu_util()
        state.cpu_saturation     = state.cpu_utilization / 100.0

        if gc_fired:
            state.gc_pause_p99   = dist.gc_spike()
            state.p99_latency    = dist.p99_spike(state.gc_pause_p99)
            state.p50_latency    = float(np.clip(dist.rng.normal(92, 4), 70, 120))
            state.error_rate     = float(np.clip(dist.err_rate() + dist.rng.uniform(0.01, 0.04), 0, 0.15))
        else:
            state.gc_pause_p99   = dist.gc_p99()
            state.p99_latency    = dist.p99()
            state.p50_latency    = dist.p50()
            state.error_rate     = dist.err_rate()

        state.p95_latency           = dist.p95(state.p50_latency, state.p99_latency)
        state.cache_miss_rate       = dist.cache_miss()
        state.cache_hit_rate        = 1.0 - state.cache_miss_rate
        state.db_p99                = dist.db_p99()
        state.queue_lag             = dist.queue_lag()
        state.retry_count_per_request = dist.retry_cnt()
        state.rps                   = dist.rps()
        state.active_connections    = dist.active_conn()
        state.db_connection_pool    = dist.db_conn_pool()
        state.db_connection_wait    = dist.db_conn_wait()
        state.disk_read_latency     = dist.disk_read_lat()
        state.disk_write_latency    = dist.disk_write_lat()
        state.iops_utilization      = dist.iops_util()
        state.memory_utilization    = dist.mem_util(state.heap_mb)
        state.network_errors        = dist.net_errors()
        state.http_4xx_rate         = dist.http_4xx()
        state.http_5xx_rate         = float(np.clip(dist.http_5xx() + state.error_rate * 0.05, 0, 0.1))
        state.thread_pool_queue     = dist.thread_pool_q()
        state.upstream_timeout_rate = dist.upstream_timeout()
        state.circuit_breaker_state = dist.circuit_breaker(state.error_rate)
        return state
