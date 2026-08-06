"""app_simulator/physics/base_scenario.py — Abstract base for all 13 scenarios."""
from __future__ import annotations
from abc import ABC, abstractmethod
from .distributions import Dist
from ..state import SimulatorState


class BaseScenario(ABC):
    """
    Contract every failure-mode scenario must fulfil.

    apply() is called ONCE per simulation tick.
    It receives the current SimulatorState, mutates metric fields in-place,
    and returns the same state object.
    """
    failure_mode: str = "UNKNOWN"

    @abstractmethod
    def apply(self, state: SimulatorState, dist: Dist) -> SimulatorState:
        """Apply failure physics for one tick. Mutate state, return it."""
        ...

    def onset_delay_s(self, rng: np.random.Generator) -> float:
        """Draw random onset delay in seconds for realistic TTF spread. Default 0.0."""
        return 0.0

    def _apply_healthy(self, state: SimulatorState, dist: Dist) -> SimulatorState:
        """Apply healthy baseline metrics before onset_delay_s is reached."""
        state.cpu_utilization       = dist.cpu_util()
        state.cpu_saturation        = state.cpu_utilization / 100.0
        state.heap_mb               = dist.heap()
        state.gc_pause_p99          = dist.gc_p99()
        state.p50_latency           = dist.p50()
        state.p99_latency           = dist.p99()
        state.p95_latency           = dist.p95(state.p50_latency, state.p99_latency)
        state.error_rate            = dist.err_rate()
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
        state.http_5xx_rate         = dist.http_5xx()
        state.thread_pool_queue     = dist.thread_pool_q()
        state.upstream_timeout_rate = dist.upstream_timeout()
        state.circuit_breaker_state = dist.circuit_breaker(state.error_rate)
        return state

