"""
d:/Before_done/forecasting_node/modes/dependency_timeout.py
============================================================
DEPENDENCY_TIMEOUT Forecasting Node.

Failure Physics:
    Upstream service timeouts arrive at a CONSTANT rate as the dependency
    degrades steadily (e.g. disk I/O saturation or memory pressure upstream).
    P99 latency climbs linearly as more requests hit the timeout ceiling.
    HTTP 504 (gateway timeout) rate also rises linearly.

    From design image: Linear Regression is the identified model for this mode
    because "timeouts generally increase gradually" and "risk increases at a
    steady rate" — not exponentially.

Algorithm Tiers:
    Tier 1: Auto-ARIMA (pmdarima → statsmodels → linear)
    Tier 2: linear (explicitly chosen — linear is the physics-correct model)

Critical Features for TTF:
    p99_latency          → threshold 1000.0 (ms)
    upstream_timeout_rate → threshold 0.50  (50% — service effectively unreachable)

Extra features forecasted (context only):
    network_errors, rps, error_rate
"""
from .._mode_runner import run_auto_arima_forecast

FAILURE_MODE = "DEPENDENCY_TIMEOUT"

_CRITICAL = {
    "p99_latency":           1000.0,   # ms — upstream SLA breach
    "upstream_timeout_rate": 0.50,     # fraction — 50% requests timing out
}
_EXTRA = ["network_errors", "rps", "error_rate"]


def forecast_dependency_timeout(episode_id: str, current_features: dict) -> dict:
    """
    Run TTF forecast for DEPENDENCY_TIMEOUT failure mode.

    Uses Auto-ARIMA as tier-1.
    Tier-2 fallback is explicitly linear (steady growth = linear physics).

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
        tier2_algorithm   = "linear",   # Linear is the physics-correct model
    )
