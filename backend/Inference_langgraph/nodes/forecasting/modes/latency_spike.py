"""
d:/Before_done/forecasting_node/modes/latency_spike.py
=======================================================
LATENCY_SPIKE Forecasting Node.

Failure Physics:
    P99 latency spikes intermittently via GC pressure or thread contention.
    Auto-ARIMA captures the stochastic spikes and underlying rising trend.
    GC pause P99 > 55ms is a reliable secondary indicator.

Algorithm Tiers:
    Tier 1: Auto-ARIMA (pmdarima → statsmodels → linear)
    Tier 2: linear fallback

Critical Features for TTF:
    p99_latency → threshold 700.0 (ms) — SLA breach for most services
    p95_latency → threshold 500.0 (ms) — early warning

Extra features forecasted (context only):
    gc_pause_p99, p50_latency, error_rate
"""
from .._mode_runner import run_auto_arima_forecast

FAILURE_MODE = "LATENCY_SPIKE"

_CRITICAL = {
    "p99_latency": 700.0,   # ms — SLA breach
    "p95_latency": 500.0,   # ms — early indicator
}
_EXTRA = ["gc_pause_p99", "p50_latency", "error_rate"]


def forecast_latency_spike(episode_id: str, current_features: dict) -> dict:
    """
    Run TTF forecast for LATENCY_SPIKE failure mode.

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
