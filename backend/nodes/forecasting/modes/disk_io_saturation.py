"""
d:/Before_done/forecasting_node/modes/disk_io_saturation.py
============================================================
DISK_IO_SATURATION Forecasting Node.

Failure Physics:
    Disk I/O latency grows as IOPS approaches the physical drive limit.
    Both read and write paths degrade simultaneously.
    Queue depth (queue_lag) grows as I/O requests pile up waiting for
    the disk to service them.

Algorithm Tiers:
    Tier 1: Auto-ARIMA (pmdarima → statsmodels → linear)
    Tier 2: linear fallback (disk degradation is typically monotonic linear)

Critical Features for TTF:
    iops_utilization   → threshold 95.0  (% — physical drive limit)
    queue_lag          → threshold 50.0  — deep I/O queue = severe saturation
    disk_read_latency  → threshold 500.0 (ms) — application read timeouts
    disk_write_latency → threshold 500.0 (ms) — application write timeouts

Extra features forecasted (context only):
    rps, p99_latency
"""
from .._mode_runner import run_auto_arima_forecast

FAILURE_MODE = "DISK_IO_SATURATION"

_CRITICAL = {
    "iops_utilization":   95.0,    # % — physical drive limit
    "queue_lag":          50.0,    # I/O queue depth — severe saturation
    "disk_read_latency":  500.0,   # ms — read timeout threshold
    "disk_write_latency": 500.0,   # ms — write timeout threshold
}
_EXTRA = ["rps", "p99_latency"]


def forecast_disk_io_saturation(episode_id: str, current_features: dict) -> dict:
    """
    Run TTF forecast for DISK_IO_SATURATION failure mode.

    Args:
        episode_id:       Current episode identifier.
        current_features: Latest feature row dict from the pipeline.

    Returns:
        Convergence schema dict (see _mode_runner.run_auto_arima_forecast).
    """
    return run_auto_arima_forecast(
        episode_id        = episode_id,
        failure_mode      = FAILURE_MODE,
        current_features  = current_features,
        critical_features = _CRITICAL,
        extra_features    = _EXTRA,
        tier2_algorithm   = "linear",
    )
