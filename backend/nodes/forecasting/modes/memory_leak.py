"""
d:/Before_done/forecasting_node/modes/memory_leak.py
=====================================================
MEMORY_LEAK Forecasting Node.

Failure Physics:
    Heap grows monotonically as objects are allocated and never released.
    Growth rate is near-constant (MB per cycle) — ideal for linear regression.
    OOM risk zone begins at 3500 MB (system limit typically ~4096 MB).

Algorithm Tiers:
    Tier 1: Auto-ARIMA (pmdarima → statsmodels → linear)
    Tier 2: linear (default — MEMORY_LEAK is inherently linear)

Critical Features for TTF:
    heap_mb           → threshold 3500.0 MB
    memory_utilization → threshold 0.90 (90%)

Extra features forecasted (context only, not used for TTF):
    gc_pause_p99, error_rate, rps
"""
from .._mode_runner import run_auto_arima_forecast

FAILURE_MODE = "MEMORY_LEAK"

_CRITICAL = {
    "heap_mb":            3500.0,   # MB — OOM risk zone
    "memory_utilization": 0.90,     # fraction — 90% system memory
}
_EXTRA = ["gc_pause_p99", "error_rate", "rps"]


def forecast_memory_leak(episode_id: str, current_features: dict) -> dict:
    """
    Run TTF forecast for MEMORY_LEAK failure mode.

    Args:
        episode_id:       Current episode identifier.
        current_features: Latest feature row dict from the pipeline.

    Returns:
        Convergence schema dict (see _mode_runner.run_auto_arima_forecast).
    """
    return run_auto_arima_forecast(
        episode_id       = episode_id,
        failure_mode     = FAILURE_MODE,
        current_features = current_features,
        critical_features = _CRITICAL,
        extra_features   = _EXTRA,
        tier2_algorithm  = "linear",   # heap grows linearly — linear as strong tier-2
    )
