"""
app_simulator/pipeline/severity.py
====================================
Preliminary Severity Node — rule-based P1-P4.

Runs AFTER feature engineering, BEFORE classification.
No ML required. Pure threshold rules on raw metrics + log features.

Severity levels:
  P1 — CRITICAL: immediate response required
  P2 — HIGH:     response within 15 minutes
  P3 — MEDIUM:   response within 1 hour
  P4 — LOW:      baseline / healthy

Rules (from implementation plan, thresholds in config.py):
  P1 if ANY: cpu > 90%, memory > 90%, p99 > 1000ms, critical_log_count > 5, error_rate > 10%
  P2 if ANY: cpu > 80%, memory > 80%, p99 > 700ms, error_rate > 5%, gc > 200ms, exception in log
  P3 if ANY: cpu > 60%, p99 > 400ms, error_rate > 2%, cache_miss > 50%
  P4: otherwise (healthy baseline)
"""
from __future__ import annotations

from app_data_generator.state import PipelineState
from app_data_generator.config import SEVERITY_P1, SEVERITY_P2, SEVERITY_P3


def compute_preliminary_severity(state: PipelineState) -> PipelineState:
    """
    Evaluate severity thresholds on state.classifier_input.
    Writes state.preliminary_severity and state.severity_reasons.

    Args:
        state: PipelineState after feature engineering.

    Returns:
        Same state with preliminary_severity and severity_reasons filled.
    """
    feats   = state.classifier_input
    reasons = []

    def _f(key: str, default=0) -> float:
        """Safe float extraction from features dict."""
        val = feats.get(key, default)
        try:
            return float(val)
        except (TypeError, ValueError):
            return float(default)

    # ── P1 — CRITICAL ────────────────────────────────────────────────────────
    p1_checks = [
        (_f("cpu_utilization")   > SEVERITY_P1["cpu_utilization"],    f"cpu={_f('cpu_utilization'):.1f}%>90%"),
        (_f("memory_utilization") > SEVERITY_P1["memory_utilization"], f"mem={_f('memory_utilization'):.1f}%>90%"),
        (_f("p99_latency")        > SEVERITY_P1["p99_latency"],        f"p99={_f('p99_latency'):.0f}ms>1000ms"),
        (_f("log_critical_count") > SEVERITY_P1["log_critical_count"], f"critical_logs={int(_f('log_critical_count'))}>5"),
        (_f("error_rate")         > SEVERITY_P1["error_rate"],         f"err={_f('error_rate')*100:.1f}%>10%"),
    ]
    for triggered, reason in p1_checks:
        if triggered:
            reasons.append(reason)

    if reasons:
        state.preliminary_severity = "P1"
        state.severity_reasons     = reasons
        return state

    # ── P2 — HIGH ────────────────────────────────────────────────────────────
    p2_checks = [
        (_f("cpu_utilization")   > SEVERITY_P2["cpu_utilization"],    f"cpu={_f('cpu_utilization'):.1f}%>80%"),
        (_f("memory_utilization") > SEVERITY_P2["memory_utilization"], f"mem={_f('memory_utilization'):.1f}%>80%"),
        (_f("p99_latency")        > SEVERITY_P2["p99_latency"],        f"p99={_f('p99_latency'):.0f}ms>700ms"),
        (_f("error_rate")         > SEVERITY_P2["error_rate"],         f"err={_f('error_rate')*100:.1f}%>5%"),
        (_f("gc_pause_p99")       > SEVERITY_P2["gc_pause_p99"],       f"gc={_f('gc_pause_p99'):.0f}ms>200ms"),
        (int(_f("log_has_exception")) == 1,                            f"log_has_exception=1"),
    ]
    for triggered, reason in p2_checks:
        if triggered:
            reasons.append(reason)

    if reasons:
        state.preliminary_severity = "P2"
        state.severity_reasons     = reasons
        return state

    # ── P3 — MEDIUM ──────────────────────────────────────────────────────────
    p3_checks = [
        (_f("cpu_utilization") > SEVERITY_P3["cpu_utilization"],  f"cpu={_f('cpu_utilization'):.1f}%>60%"),
        (_f("p99_latency")      > SEVERITY_P3["p99_latency"],      f"p99={_f('p99_latency'):.0f}ms>400ms"),
        (_f("error_rate")       > SEVERITY_P3["error_rate"],       f"err={_f('error_rate')*100:.1f}%>2%"),
        (_f("cache_miss_rate")  > SEVERITY_P3["cache_miss_rate"],  f"cache_miss={_f('cache_miss_rate')*100:.0f}%>50%"),
    ]
    for triggered, reason in p3_checks:
        if triggered:
            reasons.append(reason)

    if reasons:
        state.preliminary_severity = "P3"
        state.severity_reasons     = reasons
        return state

    # ── P4 — LOW (baseline healthy) ───────────────────────────────────────────
    state.preliminary_severity = "P4"
    state.severity_reasons     = ["baseline — all metrics within normal range"]
    return state
