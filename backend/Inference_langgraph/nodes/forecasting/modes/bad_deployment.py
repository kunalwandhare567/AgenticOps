"""
d:/Before_done/forecasting_node/modes/bad_deployment.py
========================================================
BAD_DEPLOYMENT Forecasting Node.

Failure Physics:
    A bad canary deploy causes an INSTANT step-change in error rate within
    1–2 cycles of deployment. The error rate jumps sharply and then continues
    rising as the bad code serves more traffic.

    Auto-ARIMA captures the initial spike and subsequent trajectory.
    The threshold is high (50%) because even a bad deploy needs significant
    error rate to be confirmed as truly failed (not just transient noise).

Algorithm Tiers:
    Tier 1: Auto-ARIMA (pmdarima → statsmodels → linear)
    Tier 2: linear fallback (step-change projects linearly after onset)

Critical Features for TTF:
    error_rate    → threshold 0.50  (50% error rate = service down)
    http_5xx_rate → threshold 0.40  (40% 5xx confirms server-side failure)

Extra features forecasted (context only):
    http_4xx_rate, network_errors, p99_latency
"""
from .._mode_runner import run_auto_arima_forecast

FAILURE_MODE = "BAD_DEPLOYMENT"

_CRITICAL = {
    "error_rate":    0.50,   # fraction — service effectively down
    "http_5xx_rate": 0.40,   # fraction — server-side failure confirmed
}
_EXTRA = ["http_4xx_rate", "network_errors", "p99_latency"]


def forecast_bad_deployment(episode_id: str, current_features: dict) -> dict:
    """
    Run TTF forecast for BAD_DEPLOYMENT failure mode.

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
