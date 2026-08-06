"""
backend/langgraph_pipeline/nodes/n10_db_writer.py
===================================================
LangGraph Node 10 — Centralised DB Writer.

The LAST node in the graph. Writes the fully-populated AIOpsLangState
to all pipeline DB tables in simulator_db.sqlite via the existing DbWriter.

Why centralise here?
  - All other nodes stay pure (input dict → output dict, no DB I/O).
  - Schema changes only touch this one file.
  - Retry logic, batching, or WAL flushing can be added here without
    touching any node logic.
  - Identical to what run_pipeline.py did per-cycle, but consolidated.

Tables written:
  1. node_feature_engineering       — 32 FE features
  2. node_preliminary_severity      — severity thresholds result
  3. node_classification            — LightGBM prediction
  4. node_tumbling_window           — majority-vote window
  5. node_forecasting               — TTF forecast
  6. node_severity_update           — revised severity
  7. node_human_gate                — gate decision (if triggered)
  8. pipeline_results               — combined snapshot (all fields)
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import sys
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent.parent))

from Simulator.app_data_generator_for_offline.config import DB_PATH
from Simulator.app_data_generator_for_offline.storage.db_writer import DbWriter
from Inference_langgraph.state import AIOpsLangState

# ── DbWriter singleton ────────────────────────────────────────────────────────
lock:     threading.Lock    = threading.Lock()
_db_writer: DbWriter | None = None

# ── Live DB path override ────────────────────────────────────────────────────────
# When set by run_langgraph.py --live, all writes target live_feed_db.sqlite.
_live_db_path: Path | None = None


def set_live_db_path(path: str) -> None:
    """Redirect DB writes to a live feed DB path. Called once at startup by run_langgraph.py."""
    global _live_db_path, _db_writer
    _live_db_path = Path(path)
    _db_writer = None   # force re-init against the new path
    print(f"[n10_db_writer] Live DB path set: {_live_db_path}")


def init() -> None:
    """Open and set up the DbWriter singleton. Called once at pipeline startup."""
    global _db_writer
    with lock:
        if _db_writer is None:
            target = _live_db_path if _live_db_path is not None else DB_PATH
            print(f"[n10_db_writer] Initialising DbWriter -> {target}")
            _db_writer = DbWriter(target)
            _db_writer.setup()


def _get_writer() -> DbWriter:
    global _db_writer
    if _db_writer is None:
        init()
    return _db_writer


# =============================================================================
# SSE Broadcaster helper
# =============================================================================

def _broadcast_state(state: dict) -> None:
    """Push latest pipeline state to SSE broadcaster (no-op if not available)."""
    try:
        from api.sse_broadcaster import SSEBroadcaster
        SSEBroadcaster.update(state)
    except Exception:
        pass  # SSE is optional; never fail the pipeline


# =============================================================================
# LangGraph Node Function
# =============================================================================

def run(state: AIOpsLangState) -> dict[str, Any]:
    """DB writer node — persists full pipeline state to all DB tables."""
    db  = _get_writer()
    cycle    = state.get("cycle", 0)
    ep_id    = state.get("episode_id", "")
    f_mode   = state.get("failure_mode", "NONE")
    ts       = state.get("timestamp", 0.0)
    elapsed  = state.get("elapsed_s", 0.0)
    fe_feats = state.get("classifier_input", {})

    # ── 1. Feature Engineering ────────────────────────────────────────────────
    try:
        db.write_feature_engineering({
            "cycle":        cycle,
            "episode_id":   ep_id,
            "failure_mode": f_mode,
            "timestamp":    ts,
            "elapsed_s":    elapsed,
            **fe_feats,
        })
    except Exception as exc:
        print(f"[n10_db_writer] FE write error: {exc}")

    # ── 2. Preliminary Severity ───────────────────────────────────────────────
    sev_r = state.get("severity_result", {})
    try:
        db.write_preliminary_severity({
            "cycle":                 cycle,
            "episode_id":            ep_id,
            "failure_mode":          f_mode,
            "timestamp":             ts,
            "elapsed_s":             elapsed,
            "preliminary_severity":  state.get("preliminary_severity", "P4"),
            "severity_raw":          sev_r.get("raw_severity") or sev_r.get("RawSeverity"),
            "weighted_score":        sev_r.get("weighted_score") or sev_r.get("WeightedScore"),
            "critical_count":        sev_r.get("critical_count") or sev_r.get("CriticalCount"),
            "warning_count":         sev_r.get("warning_count")  or sev_r.get("WarningCount"),
            "blast_size":            sev_r.get("blast_size")     or sev_r.get("BlastSize"),
            "high_risk_mode":        int(sev_r.get("high_risk_mode", 0)),
            "blast_radius_growing":  int(sev_r.get("blast_radius_growing", 0)),
            "reason":                sev_r.get("reason")         or sev_r.get("Reason"),
            "recommended_action":    sev_r.get("recommended_action") or sev_r.get("RecommendedAction"),
        })
    except Exception as exc:
        print(f"[n10_db_writer] PrelimSev write error: {exc}")

    # ── 3. Classification ─────────────────────────────────────────────────────
    try:
        db.write_classification({
            "cycle":                 cycle,
            "episode_id":            ep_id,
            "failure_mode":          f_mode,
            "timestamp":             ts,
            "elapsed_s":             elapsed,
            "predicted_failure":     state.get("predicted_failure", "NONE"),
            "prediction_probability": state.get("prediction_probability", 0.0),
        })
    except Exception as exc:
        print(f"[n10_db_writer] Classification write error: {exc}")

    # ── 4. Tumbling Window ────────────────────────────────────────────────────
    try:
        db.write_tumbling_window({
            "cycle":             cycle,
            "episode_id":        ep_id,
            "failure_mode":      f_mode,
            "timestamp":         ts,
            "elapsed_s":         elapsed,
            "dominant_state":    state.get("dominant_state", "NONE"),
            "vote_distribution": state.get("vote_distribution", {}),
            "window_margin":     state.get("window_margin", 0.0),
            "window_full":       int(state.get("window_full", False)),
            "window_size":       len(state.get("vote_distribution", {})),
        })
    except Exception as exc:
        print(f"[n10_db_writer] TumblingWindow write error: {exc}")

    # ── 5. Forecasting ────────────────────────────────────────────────────────
    fc_result = state.get("forecast_result", {})
    if fc_result:
        fc_payload = fc_result.get("forecast", {})
        try:
            db.write_forecasting({
                "cycle":               cycle,
                "episode_id":          ep_id,
                "failure_mode":        f_mode,
                "timestamp":           ts,
                "elapsed_s":           elapsed,
                "algorithm_used":      state.get("forecast_algorithm"),
                "time_to_failure":     state.get("time_to_failure"),
                "forecast_confidence": state.get("forecast_confidence"),
                "threshold_crossed":   state.get("threshold_crossed"),
                "earliest_ttf_feature": state.get("earliest_ttf_feature"),
                "feature_ttfs":        fc_payload.get("feature_ttfs", {}),
                "feature_slopes":      fc_payload.get("feature_slopes", {}),
                "predictions":         fc_payload.get("predictions", {}),
                "current_values":      fc_payload.get("current_values", {}),
            })
        except Exception as exc:
            print(f"[n10_db_writer] Forecasting write error: {exc}")

    # ── 6. Severity Update ────────────────────────────────────────────────────
    if state.get("revised_severity") is not None:
        try:
            db.write_severity_update({
                "cycle":                cycle,
                "episode_id":           ep_id,
                "failure_mode":         f_mode,
                "timestamp":            ts,
                "elapsed_s":            elapsed,
                "preliminary_severity": state.get("preliminary_severity", "P4"),
                "forecast_confidence":  state.get("forecast_confidence"),
                "time_to_failure":      state.get("time_to_failure"),
                "earliest_ttf_feature": state.get("earliest_ttf_feature"),
                "impact_band":          state.get("impact_band"),
                "urgency_band":         state.get("urgency_band"),
                "gate_passed":          state.get("gate_passed"),
                "candidate_severity":   state.get("candidate_severity"),
                "revised_severity":     state.get("revised_severity"),
                "is_escalated":         state.get("is_escalated"),
                "is_deescalated":       state.get("is_deescalated"),
                "dwell_count":          state.get("dwell_count"),
                "reason":               state.get("su_reason"),
            })
        except Exception as exc:
            print(f"[n10_db_writer] SeverityUpdate write error: {exc}")

    # ── 7. Human Gate ──────────────────────────────────────────────────────────
    if state.get("hg_needed") and state.get("hg_review_id"):
        from datetime import datetime, timezone
        try:
            db.write_human_gate({
                "review_id":         state.get("hg_review_id"),
                "incident_id":       f"inc_{ep_id}_{cycle}",
                "episode_id":        ep_id,
                "failure_mode":      f_mode,
                "failure_label":     f_mode.replace("_", " ").title(),
                "old_severity":      state.get("committed_severity") or state.get("preliminary_severity", "P4"),
                "new_severity":      state.get("revised_severity"),
                "final_severity":    state.get("hg_final_severity"),
                "decision":          state.get("hg_decision"),
                "operator":          state.get("hg_operator"),
                "reason":            "",
                "confidence":        state.get("forecast_confidence") or 0.0,
                "ttf_seconds":       state.get("time_to_failure") or -1.0,
                "impact_band":       state.get("impact_band"),
                "urgency_band":      state.get("urgency_band"),
                "is_large_jump":     int(state.get("hg_is_large_jump") or False),
                "escalation_summary": state.get("hg_escalation_summary", ""),
                "response_ms":       state.get("hg_response_ms", 0),
                "timeout_seconds":   2,
                "created_at":        datetime.now(timezone.utc).isoformat(),
                "decided_at":        datetime.now(timezone.utc).isoformat(),
                "recorded_at":       datetime.now(timezone.utc).isoformat(),
            })
        except Exception as exc:
            print(f"[n10_db_writer] HumanGate write error: {exc}")

    # ── 8. Combined pipeline_results ──────────────────────────────────────────
    try:
        db.write_pipeline_results({
            "cycle":              cycle,
            "episode_id":         ep_id,
            "failure_mode":       f_mode,
            "timestamp":          ts,
            "elapsed_s":          elapsed,
            # FE metrics (prefixed)
            "fe_cpu_utilization":         fe_feats.get("cpu_utilization"),
            "fe_memory_utilization":      fe_feats.get("memory_utilization"),
            "fe_heap_mb":                 fe_feats.get("heap_mb"),
            "fe_error_rate":              fe_feats.get("error_rate"),
            "fe_p99_latency":             fe_feats.get("p99_latency"),
            "fe_p95_latency":             fe_feats.get("p95_latency"),
            "fe_db_p99":                  fe_feats.get("db_p99"),
            "fe_queue_lag":               fe_feats.get("queue_lag"),
            "fe_log_count":               fe_feats.get("log_count"),
            "fe_log_critical_count":      fe_feats.get("log_critical_count"),
            "fe_log_has_exception":       fe_feats.get("log_has_exception"),
            "fe_log_has_novel_template":  fe_feats.get("log_has_novel_template"),
            # Severity
            "preliminary_severity":       state.get("preliminary_severity", "P4"),
            "severity_weighted_score":    sev_r.get("weighted_score") or sev_r.get("WeightedScore"),
            "severity_critical_count":    sev_r.get("critical_count") or sev_r.get("CriticalCount"),
            "severity_warning_count":     sev_r.get("warning_count")  or sev_r.get("WarningCount"),
            "severity_blast_size":        sev_r.get("blast_size")     or sev_r.get("BlastSize"),
            "severity_reason":            sev_r.get("reason")         or sev_r.get("Reason"),
            # Classification
            "predicted_failure":          state.get("predicted_failure", "NONE"),
            "prediction_probability":     state.get("prediction_probability", 0.0),
            # Tumbling Window
            "dominant_state":             state.get("dominant_state", "NONE"),
            "vote_distribution":          state.get("vote_distribution", {}),
            "window_margin":              state.get("window_margin", 0.0),
            "window_full":                state.get("window_full", False),
            "window_size":                len(state.get("vote_distribution", {})),
            # Forecasting
            "forecast_algorithm":         state.get("forecast_algorithm"),
            "time_to_failure":            state.get("time_to_failure"),
            "forecast_confidence":        state.get("forecast_confidence"),
            "threshold_crossed":          state.get("threshold_crossed"),
            "earliest_ttf_feature":       state.get("earliest_ttf_feature"),
            # Severity Update
            "revised_severity":           state.get("revised_severity"),
            "candidate_severity":         state.get("candidate_severity"),
            "impact_band":                state.get("impact_band"),
            "urgency_band":               state.get("urgency_band"),
            "gate_passed":                state.get("gate_passed"),
            "is_escalated":               state.get("is_escalated"),
            "is_deescalated":             state.get("is_deescalated"),
            "su_reason":                  state.get("su_reason"),
            # Human Gate
            "hg_review_id":               state.get("hg_review_id"),
            "hg_decision":                state.get("hg_decision"),
            "hg_final_severity":          state.get("hg_final_severity"),
            "hg_operator":                state.get("hg_operator"),
            "hg_response_ms":             state.get("hg_response_ms"),
        })
    except Exception as exc:
        print(f"[n10_db_writer] pipeline_results write error: {exc}")

    # ── Broadcast to SSE clients ─────────────────────────────────────────────────────
    _broadcast_state(dict(state))

    # Node returns empty dict — state is unchanged (we're last in the graph)
    return {}
