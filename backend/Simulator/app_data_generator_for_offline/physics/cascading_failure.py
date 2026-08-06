"""
CASCADING_FAILURE — 4-stage collapse driven by state.step.

Stage 1 (steps  0-25):  DB pressure starts; db_p99 grows 8%/step
Stage 2 (steps 26-50):  Cache stampede follows; miss_rate → 60-90%
Stage 3 (steps 51-70):  Error storm erupts; error_rate 35-85%
Stage 4 (steps 71-120): Full resource exhaustion; CPU 60-100%, CB open
"""
import numpy as np
from .base_scenario import BaseScenario
from .distributions import Dist
from ..state import SimulatorState


class CascadingFailureScenario(BaseScenario):
    failure_mode = "CASCADING_FAILURE"

    def onset_delay_s(self, rng: np.random.Generator) -> float:
        return float(rng.uniform(0, 8))

    def apply(self, state: SimulatorState, dist: Dist) -> SimulatorState:
        if state.elapsed_s < state.onset_delay_s:
            return self._apply_healthy(state, dist)

        s = int((state.elapsed_s - state.onset_delay_s) / 2.0)  # 2.0s per step

        # ── Stage 1 (DB pressure) ────────────────────────────────────────────
        state.db_p99 = float(np.clip(dist.db_p99() * (1 + s * 0.08), 20, 3000))

        # ── Stage 2 (Cache collapse after step 25) ───────────────────────────
        if s > 25:
            state.cache_miss_rate = float(np.clip(dist.cache_miss() + 0.6, 0.05, 0.90))
        else:
            state.cache_miss_rate = dist.cache_miss()
        state.cache_hit_rate = 1.0 - state.cache_miss_rate

        # ── Stage 3 (Error storm after step 50) ──────────────────────────────
        if s > 50:
            state.error_rate = dist.cascade_err()
        else:
            state.error_rate = float(np.clip(dist.err_rate() + s * 0.003, 0, 0.3))

        # ── Stage 4 (Full collapse after step 70) ────────────────────────────
        state.cpu_utilization       = dist.cascade_cpu()
        state.cpu_saturation        = state.cpu_utilization / 100.0
        state.p99_latency           = dist.cascade_p99(state.cpu_utilization, state.error_rate)
        state.p50_latency           = float(np.clip(dist.p50() + s * 2, 80, 800))
        state.p95_latency           = dist.p95(state.p50_latency, state.p99_latency)
        state.heap_mb               = float(np.clip(dist.heap() + state.cpu_utilization * 2, 400, 1200))
        state.gc_pause_p99          = float(np.clip(dist.gc_p99() + state.cpu_utilization * 2, 10, 400))
        state.queue_lag             = int(np.clip(state.cpu_utilization * 3, 50, 500))
        state.retry_count_per_request = float(np.clip(state.error_rate * 0.6, 0, 0.6))
        state.active_connections    = int(np.clip(dist.active_conn() + state.cpu_utilization * 3, 20, 1000))
        state.db_connection_pool    = float(np.clip(dist.db_conn_pool() + state.cache_miss_rate * 0.5, 0.1, 0.99))
        state.db_connection_wait    = float(np.clip(state.db_p99 * 0.4, 1, 500))
        state.disk_read_latency     = dist.disk_read_lat()
        state.disk_write_latency    = dist.disk_write_lat()
        state.iops_utilization      = dist.iops_util()
        state.memory_utilization    = dist.mem_util(state.heap_mb)
        state.network_errors        = int(np.clip(dist.net_errors() + state.error_rate * 40, 0, 100))
        state.http_4xx_rate         = float(np.clip(dist.http_4xx() + state.error_rate * 0.2, 0, 0.4))
        state.http_5xx_rate         = float(np.clip(dist.http_5xx() + state.error_rate * 0.5, 0, 0.6))
        state.thread_pool_queue     = int(np.clip(dist.thread_pool_q() + state.cpu_utilization * 5, 0, 1000))
        state.upstream_timeout_rate = float(np.clip(dist.upstream_timeout() + state.error_rate * 0.3, 0, 0.5))
        state.rps                   = dist.rps()
        state.circuit_breaker_state = dist.circuit_breaker(state.error_rate)
        return state
