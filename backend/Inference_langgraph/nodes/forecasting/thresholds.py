"""
d:/Before_done/forecasting_node/thresholds.py
=============================================
Single source of truth for ALL failure-mode forecasting configuration.
No hardcoded values anywhere else in the forecasting package.

Features available from data_pipeline.py METRIC_FEATURES (28 metric + 7 log = 35 total):
  Metric features:
    cpu_mean, cpu_max, cpu_std, cpu_slope,
    memory_mean, memory_max, memory_growth_rate,
    heap_mean, heap_max,
    p50_mean, p95_mean, p99_mean, latency_std, latency_slope,
    throughput_mean, throughput_std,
    cache_hit_ratio, cache_miss_ratio,
    db_latency_mean, db_connections_max,
    network_errors_mean,
    error_rate_mean, error_rate_max,
    gc_pause_mean, gc_pause_max,
    disk_read_mean, disk_write_mean,
    cb_open_ratio
  Log features:
    log_count, log_max_severity, log_critical_count,
    log_has_exception, log_has_novel_template,
    log_exception_type_encoded, log_severity_ratio
"""
from __future__ import annotations

# ── Algorithm constants ───────────────────────────────────────────────────────
ALGO_LINEAR      = "linear_regression"
ALGO_EXP         = "exponential_regression"
ALGO_ARIMA       = "arima"
ALGO_CHANGE      = "change_point"

# ── ARIMA order for all ARIMA-mode forecasters ────────────────────────────────
ARIMA_ORDER      = (2, 1, 2)

# ── Rolling buffer sizes ──────────────────────────────────────────────────────
MAX_LOOKBACK_CYCLES = 120          # 120 cycles × 2s = 240s (one full episode)
ARIMA_MIN_HISTORY   = 15          # minimum rows before ARIMA is attempted
LINEAR_MIN_HISTORY  = 5           # minimum rows before linear fit is attempted

# ── Forecast horizon ──────────────────────────────────────────────────────────
FORECAST_STEPS   = 30             # steps ahead (30 × 2s = 60 seconds)
STEP_INTERVAL_S  = 2.0            # seconds per simulation step
STEP_INTERVAL_MIN = STEP_INTERVAL_S / 60.0

# ── Confidence penalty for short history ─────────────────────────────────────
SHORT_HISTORY_PENALTY_THRESHOLD = 10   # cycles below which confidence is penalized
SHORT_HISTORY_PENALTY_FACTOR    = 0.70

# =============================================================================
# PER-FAILURE-MODE CONFIG
# =============================================================================
# Each entry: (primary_metric, algorithm, critical_threshold, direction,
#              secondary_metric, secondary_threshold, secondary_direction)
# direction: "higher_worse" | "lower_worse"
# secondary_metric is used for compound-mode verification (can be None)
# =============================================================================

MODE_CONFIG: dict[str, dict] = {

    # ─ Healthy baseline ───────────────────────────────────────────────────────
    "NONE": {
        "primary_metric":     None,
        "algorithm":          None,
        "critical_threshold": None,
        "direction":          None,
        "secondary_metric":   None,
        "secondary_threshold":None,
    },

    # ─ Resource exhaustion ────────────────────────────────────────────────────
    "MEMORY_LEAK": {
        # Heap grows monotonically → deterministic countdown via linear slope.
        # Critical: OOM risk begins at 3500 MB (system limit ~4096 MB).
        # Secondary: memory_growth_rate confirms growth acceleration.
        "primary_metric":     "heap_mean",
        "algorithm":          ALGO_LINEAR,
        "critical_threshold": 3500.0,        # MB
        "direction":          "higher_worse",
        "secondary_metric":   "memory_growth_rate",
        "secondary_threshold": 0.30,         # slope per cycle > 0.30 = confirmed leak
    },

    "CPU_SATURATION": {
        # CPU climbs stochastically with trend → ARIMA best captures oscillating saturation.
        # Critical: OS scheduling degrades sharply at 90% utilization.
        # Secondary: throughput_std increases as thread contention grows.
        "primary_metric":     "cpu_mean",
        "algorithm":          ALGO_ARIMA,
        "critical_threshold": 90.0,          # %
        "direction":          "higher_worse",
        "secondary_metric":   "throughput_std",
        "secondary_threshold": 30.0,         # high std confirms unstable RPS
    },

    "LATENCY_SPIKE": {
        # P99 latency spikes stochastically via GC pauses → ARIMA captures intermittent behavior.
        # Critical: P99 > 700ms signals SLA breach for most services.
        # Secondary: gc_pause_max > 55ms confirms GC-induced pause is the cause.
        "primary_metric":     "p99_mean",
        "algorithm":          ALGO_ARIMA,
        "critical_threshold": 700.0,         # ms
        "direction":          "higher_worse",
        "secondary_metric":   "gc_pause_max",
        "secondary_threshold": 55.0,         # ms
    },

    # ─ Database & middleware ──────────────────────────────────────────────────
    "DB_SLOWDOWN": {
        # DB latency grows exponentially: each slow query adds lock contention
        # which slows the next query further (multiplicative degradation).
        # Exponential regression fits log-linear growth correctly.
        # Critical: 1500ms → connection pool exhaustion begins.
        # Secondary: db_connections_max rising confirms pool pressure.
        "primary_metric":     "db_latency_mean",
        "algorithm":          ALGO_EXP,
        "critical_threshold": 1500.0,        # ms
        "direction":          "higher_worse",
        "secondary_metric":   "db_connections_max",
        "secondary_threshold": 80.0,         # connections (pool_size typically 100)
    },

    "CACHE_STAMPEDE": {
        # Cache miss rate climbs linearly as more keys expire simultaneously.
        # Critical: 90% miss rate means virtually all requests hit the DB directly.
        # Direction: lower cache_hit_ratio is worse → use cache_miss_ratio "higher_worse".
        # Secondary: db_latency_mean rises as uncached load hits DB.
        "primary_metric":     "cache_miss_ratio",
        "algorithm":          ALGO_LINEAR,
        "critical_threshold": 0.90,          # fraction (90%)
        "direction":          "higher_worse",
        "secondary_metric":   "db_latency_mean",
        "secondary_threshold": 500.0,        # ms
    },

    "QUEUE_BACKUP": {
        # Queue lag grows at ~1.8 items/sec (linear from physics).
        # P99 latency climbs as queued requests wait → use p99_mean as primary.
        # Secondary: throughput_mean drops as backpressure throttles RPS.
        "primary_metric":     "p99_mean",
        "algorithm":          ALGO_ARIMA,
        "critical_threshold": 700.0,         # ms
        "direction":          "higher_worse",
        "secondary_metric":   "throughput_mean",
        "secondary_threshold": 100.0,        # rps below this = severe backpressure
    },

    "DEPENDENCY_TIMEOUT": {
        # Upstream service timeout rate rises linearly.
        # P99 climbs as requests wait 2–30s for upstream → P99 primary.
        # Secondary: network_errors_mean rising confirms upstream connectivity issue.
        "primary_metric":     "p99_mean",
        "algorithm":          ALGO_LINEAR,
        "critical_threshold": 700.0,         # ms
        "direction":          "higher_worse",
        "secondary_metric":   "network_errors_mean",
        "secondary_threshold": 10.0,         # errors per cycle
    },

    # ─ Application faults ────────────────────────────────────────────────────
    "BAD_DEPLOYMENT": {
        # Error rate jumps immediately (step function) when a bad canary deploys.
        # Change-point detection identifies the deployment moment.
        # Critical: 50% error rate = service effectively down.
        # Secondary: log_has_exception confirms NullPointerException is present.
        "primary_metric":     "error_rate_mean",
        "algorithm":          ALGO_CHANGE,
        "critical_threshold": 0.50,          # fraction (50%)
        "direction":          "higher_worse",
        "secondary_metric":   "log_has_exception",
        "secondary_threshold": 1.0,          # binary — exception present
    },

    "ERROR_STORM": {
        # Error rate climbs stochastically as cascading failures spread.
        # ARIMA captures the escalating pattern.
        # Secondary: log_critical_count > 5 per cycle confirms storm severity.
        "primary_metric":     "error_rate_mean",
        "algorithm":          ALGO_ARIMA,
        "critical_threshold": 0.50,          # fraction (50%)
        "direction":          "higher_worse",
        "secondary_metric":   "log_critical_count",
        "secondary_threshold": 5.0,          # critical log lines per cycle
    },

    "RETRY_STORM": {
        # Client retries amplify RPS by 3–4×, driving error rate up.
        # ARIMA captures the retry-amplification feedback loop.
        # Secondary: throughput_mean > 3× normal (~600 rps) confirms amplification.
        "primary_metric":     "error_rate_mean",
        "algorithm":          ALGO_ARIMA,
        "critical_threshold": 0.50,          # fraction (50%)
        "direction":          "higher_worse",
        "secondary_metric":   "throughput_mean",
        "secondary_threshold": 500.0,        # rps — 3× normal of ~175 rps
    },

    # ─ Infrastructure ─────────────────────────────────────────────────────────
    "DISK_IO_SATURATION": {
        # Disk write latency grows linearly as IOPS approaches physical limit.
        # Critical: 2000ms disk write wait → application timeouts occur.
        # Secondary: disk_read_mean also rising confirms both I/O paths saturated.
        "primary_metric":     "disk_write_mean",
        "algorithm":          ALGO_LINEAR,
        "critical_threshold": 2000.0,        # ms
        "direction":          "higher_worse",
        "secondary_metric":   "disk_read_mean",
        "secondary_threshold": 1500.0,       # ms
    },

    # ─ Compound failure ───────────────────────────────────────────────────────
    "CASCADING_FAILURE": {
        # Multi-stage compound: DB → Cache → Error Storm → CPU exhaustion.
        # P99 latency is the universal symptom that escalates through all stages.
        # ARIMA captures the multi-stage non-linear escalation pattern.
        # Secondary: cpu_mean > 85 AND error_rate_mean > 0.40 = Stage 4 confirmed.
        "primary_metric":     "p99_mean",
        "algorithm":          ALGO_ARIMA,
        "critical_threshold": 700.0,         # ms
        "direction":          "higher_worse",
        "secondary_metric":   "error_rate_mean",
        "secondary_threshold": 0.40,         # fraction — compound breach at 40%
    },
}

# =============================================================================
# CHANGE-POINT CONFIG (BAD_DEPLOYMENT)
# =============================================================================
CHANGE_POINT_CONFIG = {
    "history_window":  10,      # recent cycles to check for slope
    "slope_threshold": 0.05,    # error_rate slope per step that confirms step-change
}

# =============================================================================
# UTILITY LOOKUPS
# =============================================================================

def get_config(failure_mode: str) -> dict:
    """Returns the full config dict for a given failure mode."""
    return MODE_CONFIG.get(failure_mode, MODE_CONFIG["NONE"])

def get_primary_metric(failure_mode: str) -> str | None:
    return get_config(failure_mode)["primary_metric"]

def get_algorithm(failure_mode: str) -> str | None:
    return get_config(failure_mode)["algorithm"]

def get_critical_threshold(failure_mode: str) -> float | None:
    return get_config(failure_mode)["critical_threshold"]

def get_direction(failure_mode: str) -> str:
    return get_config(failure_mode).get("direction", "higher_worse")

def get_secondary(failure_mode: str) -> tuple[str | None, float | None]:
    cfg = get_config(failure_mode)
    return cfg.get("secondary_metric"), cfg.get("secondary_threshold")
