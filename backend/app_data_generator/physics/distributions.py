"""
app_simulator/physics/distributions.py
========================================
Statistical distribution helpers for all failure mode scenarios.

Direct copy of the Dist class from generate_full_dataset.py (lines 96-190)
with one addition: jitter() method for applying Gaussian noise.
All methods use numpy for reproducible, seed-able randomness.
"""
from __future__ import annotations
import math
import numpy as np
from ..config import NOISE_LEVEL, BASE_RPS, DB_CAPACITY_RPS


class Dist:
    """Random sample helper — one instance shared across all scenario objects."""

    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    # ── Gaussian jitter ───────────────────────────────────────────────────────
    def jitter(self, value: float, noise: float = NOISE_LEVEL) -> float:
        """Apply ±noise% Gaussian jitter. Used by generators for realism."""
        return float(value * (1.0 + self.rng.normal(0, noise)))

    # ── Healthy baseline distributions ────────────────────────────────────────
    def p50(self):         return float(np.clip(self.rng.normal(90, 8), 50, 200))
    def p99(self):         return float(np.clip(self.rng.normal(120, 15), 70, 300))
    def p95(self, p50, p99):
        lo, hi = min(p50, p99), max(p50, p99)
        if hi <= lo: return float(lo)
        return float(np.clip(self.rng.uniform(lo, hi), lo, hi))
    def heap(self):        return float(np.clip(self.rng.normal(512, 20), 400, 700))
    def gc_p99(self):      return float(np.clip(self.rng.exponential(15), 2, 60))
    def err_rate(self):    return float(self.rng.beta(2, 198))
    def cache_miss(self):  return float(np.clip(self.rng.beta(1, 19), 0.01, 0.12))
    def db_p99(self):      return float(np.clip(self.rng.lognormal(3.0, 0.3), 5, 80))
    def cpu_util(self):    return float(np.clip(self.rng.normal(35, 5), 10, 65))
    def queue_lag(self):   return int(self.rng.integers(0, 20))
    def retry_cnt(self):   return float(np.clip(self.rng.beta(1, 49) * 2, 0, 0.1))
    def rps(self):         return float(np.clip(self.rng.normal(200, 20), 80, 400))
    def active_conn(self): return int(self.rng.integers(20, 120))
    def db_conn_pool(self):return float(np.clip(self.rng.normal(0.35, 0.08), 0.1, 0.7))
    def db_conn_wait(self):return float(np.clip(self.rng.exponential(5), 0.5, 30))
    def disk_read_lat(self):  return float(np.clip(self.rng.lognormal(1.5, 0.4), 1, 20))
    def disk_write_lat(self): return float(np.clip(self.rng.lognormal(1.8, 0.4), 2, 30))
    def iops_util(self):   return float(np.clip(self.rng.normal(0.25, 0.06), 0.05, 0.55))
    def mem_util(self, heap_mb):
        return float(np.clip(heap_mb / 2048.0 + self.rng.normal(0, 0.03), 0.1, 0.98))
    def net_errors(self):  return int(self.rng.integers(0, 5))
    def http_4xx(self):    return float(np.clip(self.rng.beta(1, 199), 0, 0.05))
    def http_5xx(self):    return float(np.clip(self.rng.beta(1, 499), 0, 0.02))
    def thread_pool_q(self): return int(self.rng.integers(0, 15))
    def upstream_timeout(self): return float(np.clip(self.rng.beta(1, 199), 0, 0.05))
    def circuit_breaker(self, err_rate: float) -> str:
        if err_rate > 0.40: return "open"
        if err_rate > 0.20: return "half-open"
        return "closed"

    # ── MEMORY_LEAK ───────────────────────────────────────────────────────────
    def heap_leak(self, elapsed_s: float, rate: float = 12.0) -> float:
        """Heap grows ~12MB/min from 512MB baseline."""
        return float(np.clip(512 + (elapsed_s / 60.0) * rate + self.rng.normal(0, 8), 400, 1800))

    def gc_leak(self, heap_mb: float) -> float:
        """GC pauses worsen as heap fills: 15ms at 512MB → 500ms at 1800MB."""
        ratio = max(0, (heap_mb - 512) / (1800 - 512))
        return float(np.clip(self.rng.lognormal(math.log(15 + ratio * 500), 0.3), 2, 900))

    # ── CPU_SATURATION ────────────────────────────────────────────────────────
    def cpu_sat_util(self) -> float:
        return float(np.clip(self.rng.normal(88, 4), 70, 100))

    def p99_cpu_sat(self, cpu: float) -> float:
        return float(np.clip(self.rng.normal(200 + (cpu - 70) * 8, 30), 150, 1500))

    # ── LATENCY_SPIKE ─────────────────────────────────────────────────────────
    def gc_spike(self) -> float:
        return float(np.clip(self.rng.lognormal(5.5, 0.4), 80, 650))

    def p99_spike(self, gc_ms: float) -> float:
        return float(np.clip(gc_ms * self.rng.uniform(0.8, 1.1) + 90 + self.rng.normal(0, 30), 200, 3000))

    def gc_event(self, lam: float = 0.08) -> bool:
        """Returns True with Poisson probability. Default 8% chance per step."""
        return bool(self.rng.poisson(lam) > 0)

    # ── ERROR_STORM ───────────────────────────────────────────────────────────
    def err_storm(self) -> float:
        return float(np.clip(self.rng.beta(8, 12), 0.25, 0.70))

    # ── DB_SLOWDOWN ───────────────────────────────────────────────────────────
    def db_slow(self, elapsed_s: float) -> float:
        """DB query latency grows exponentially from 20ms → 1500ms over 240s."""
        growth = (elapsed_s / 240.0) * 300
        return float(np.clip(self.rng.lognormal(math.log(max(20, 20 + growth)), 0.25), 20, 1500))

    # ── CACHE_STAMPEDE ────────────────────────────────────────────────────────
    def cache_miss_stampede(self) -> float:
        return float(np.clip(self.rng.beta(19, 1) * 0.97 + self.rng.normal(0, 0.025), 0.85, 0.99))

    def db_p99_mm1(self, miss_rate: float, base_rps: float = BASE_RPS,
                   cap_rps: float = DB_CAPACITY_RPS) -> float:
        """M/M/1 queueing model: db latency explodes as rho (utilization) → 1."""
        svc_ms  = 20.0 * self.rng.lognormal(0, 0.05)
        rho     = min(base_rps * miss_rate / cap_rps, 0.98)
        queue_ms = svc_ms * (rho / (1 - rho)) if rho < 0.99 else svc_ms * 50
        return float(np.clip(svc_ms + queue_ms + self.rng.normal(0, 8), 20, 1200))

    # ── QUEUE_BACKUP ─────────────────────────────────────────────────────────
    def queue_lag_backup(self, elapsed_s: float) -> int:
        """Queue depth grows linearly: ~1.8 items/second."""
        return int(min(500, int(elapsed_s * 1.8 + self.rng.integers(0, 30))))

    # ── DEPENDENCY_TIMEOUT ───────────────────────────────────────────────────
    def timeout_rate(self) -> float:
        return float(np.clip(self.rng.beta(5, 5), 0.15, 0.55))

    # ── BAD_DEPLOY ────────────────────────────────────────────────────────────
    def err_deploy(self) -> float:
        return float(np.clip(self.rng.beta(12, 8), 0.30, 0.80))

    # ── RETRY_STORM ──────────────────────────────────────────────────────────
    def retry_high(self) -> float:
        return float(np.clip(self.rng.beta(10, 5), 0.40, 0.90))

    def rps_retry(self, retry_rate: float, base_rps: float = BASE_RPS) -> float:
        """Retries amplify RPS 3-4x depending on retry rate."""
        return float(np.clip(base_rps * (1 + retry_rate * 3), base_rps, base_rps * 4))

    # ── DISK_IO_SATURATION ───────────────────────────────────────────────────
    def db_slow_disk(self, elapsed_s: float) -> float:
        growth = (elapsed_s / 240.0) * 500
        return float(np.clip(self.rng.lognormal(math.log(max(40, 40 + growth)), 0.3), 40, 2000))

    def disk_lat_sat(self) -> float:
        return float(np.clip(self.rng.lognormal(4.5, 0.4), 50, 2000))

    def iops_sat(self) -> float:
        return float(np.clip(self.rng.normal(0.92, 0.04), 0.80, 1.0))

    # ── CASCADING_FAILURE ────────────────────────────────────────────────────
    def cascade_cpu(self) -> float:
        return float(np.clip(self.rng.normal(82, 6), 60, 100))

    def cascade_err(self) -> float:
        return float(np.clip(self.rng.beta(10, 6), 0.35, 0.85))

    def cascade_p99(self, cpu: float, err: float) -> float:
        return float(np.clip(self.rng.normal(800 + cpu * 8 + err * 200, 100), 500, 5000))
