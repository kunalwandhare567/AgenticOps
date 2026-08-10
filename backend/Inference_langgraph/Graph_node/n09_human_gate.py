"""
backend/langgraph_pipeline/nodes/n09_human_gate.py
====================================================
LangGraph Node 9 — Human Gate.

Full escalation + de-escalation logic with proper state machine.

Escalation trigger (Human review REQUIRED):
  - revised_severity is numerically HIGHER priority than committed_severity
    (i.e., new P number < old P number: P4 → P1 = escalation).
  - Uses EscalationDetector.needs_review() — same as standalone human_gate node.

De-escalation (No human review needed):
  - When revised_severity is LOWER priority (P number increases), the pipeline
    commits the new severity immediately with decision = "DE_ESCALATED" and
    no operator interaction. The is_deescalated flag from severity_update is used.

Auto-approve on timeout:
  - HUMAN_GATE_TIMEOUT_SECONDS (from config, default 2s).
  - InterruptManager polls SQLite every 0.1s; on timeout it self-approves.

Writes one row to human_gate_output.csv and to the pending_reviews SQLite table.

Returns:
    committed_severity, hg_needed, hg_review_id, hg_decision,
    hg_final_severity, hg_operator, hg_response_ms,
    hg_escalation_summary, hg_is_large_jump
"""
from __future__ import annotations

import csv
import threading
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import sys
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent.parent))

from Simulator.app_data_generator_for_offline.config import (
    HUMAN_GATE_OUTPUT_CSV,
    HUMAN_GATE_TIMEOUT_SECONDS,
)
from Inference_langgraph.nodes.human_gate.escalation_detector import EscalationDetector
from Inference_langgraph.nodes.human_gate.review_builder import HumanReviewRequest, ReviewRequestBuilder
from Inference_langgraph.nodes.human_gate.approval_engine import ApprovalEngine
from Inference_langgraph.nodes.human_gate.interrupt_manager import get_interrupt_manager
from Inference_langgraph.state import AIOpsLangState

# ── Singletons ────────────────────────────────────────────────────────────────
_lock      = threading.Lock()
_detector  = EscalationDetector()
_engine    = ApprovalEngine()
_builder   = ReviewRequestBuilder()

# ── CSV ───────────────────────────────────────────────────────────────────────
_csv_fh     = None
_csv_writer = None
_csv_lock   = threading.Lock()


def _write_csv(row: dict) -> None:
    # Skip writing to historical batch CSV during live feed mode
    try:
        from Inference_langgraph.Graph_node.n01_collect import _LIVE_MODE
        if _LIVE_MODE:
            return
    except Exception:
        pass

    global _csv_writer, _csv_fh
    with _csv_lock:
        if _csv_fh is None:
            HUMAN_GATE_OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
            _csv_fh = HUMAN_GATE_OUTPUT_CSV.open("a", newline="", encoding="utf-8")
        if _csv_writer is None:
            write_header = _csv_fh.tell() == 0
            _csv_writer  = csv.DictWriter(
                _csv_fh, fieldnames=list(row.keys()), extrasaction="ignore"
            )
            if write_header:
                _csv_writer.writeheader()
        _csv_writer.writerow(row)
        _csv_fh.flush()



def _get_committed_severity(state: AIOpsLangState) -> str:
    """
    Return the currently committed severity for this episode.
    Falls back to hg_final_severity from a previous gate, then preliminary_severity.
    """
    return (
        state.get("committed_severity")
        or state.get("hg_final_severity")
        or state.get("preliminary_severity", "P4")
    )


# =============================================================================
# LangGraph Node Function
# =============================================================================

def run(state: AIOpsLangState) -> dict[str, Any]:
    """
    Human Gate node — escalation + de-escalation with SQLite-backed review queue.

    Escalation path:
      1. Detect if revised_severity > committed_severity (P number drops).
      2. Post a HumanReviewRequest to pending_reviews via InterruptManager.
      3. Poll for decision (auto-approves after HUMAN_GATE_TIMEOUT_SECONDS).
      4. Compute ApprovalResult → final_severity.
      5. Record to CSV + DB.

    De-escalation path:
      1. Detect if is_deescalated=True (severity improving, no review needed).
      2. Commit the new lower severity immediately.
      3. Record with decision = "DE_ESCALATED", operator = "system".

    No-op path:
      1. Severity unchanged (candidate == committed) → skip gate entirely.
    """
    revised_sev   = state.get("revised_severity") or state.get("preliminary_severity", "P4")
    committed_sev = _get_committed_severity(state)
    is_deescalated = state.get("is_deescalated", False)
    is_escalated   = state.get("is_escalated",   False)
    episode_id     = state.get("episode_id", "")
    cycle          = state.get("cycle", 0)
    dominant       = state.get("dominant_state", state.get("failure_mode", "NONE"))

    # ── Case A: No severity change ─────────────────────────────────────────────
    needs_review = _detector.needs_review(committed_sev, revised_sev)
    if not is_escalated and not is_deescalated and not needs_review:
        return {
            "hg_needed":       False,
            "committed_severity": committed_sev,
        }

    # ── Case B: De-escalation — no human review needed ────────────────────────
    if is_deescalated and not needs_review:
        review_id   = f"de_{episode_id}_{cycle}"
        now_str     = datetime.now(timezone.utc).isoformat()
        summary     = f"{committed_sev} → {revised_sev} (de-escalation, system auto-commit)"

        _write_csv({
            "cycle":                cycle,
            "episode_id":           episode_id,
            "failure_mode":         dominant,
            "timestamp":            state.get("timestamp", 0.0),
            "old_severity":         committed_sev,
            "new_severity":         revised_sev,
            "final_severity":       revised_sev,
            "decision":             "DE_ESCALATED",
            "operator":             "system",
            "reason":               "Severity improved — de-escalation committed immediately.",
            "is_large_jump":        0,
            "escalation_summary":   summary,
            "response_ms":          0,
            "decided_at":           now_str,
        })

        return {
            "hg_needed":            True,
            "hg_review_id":         review_id,
            "hg_decision":          "DE_ESCALATED",
            "hg_final_severity":    revised_sev,
            "hg_operator":          "system",
            "hg_response_ms":       0,
            "hg_escalation_summary": summary,
            "hg_is_large_jump":     False,
            "committed_severity":   revised_sev,
        }

    # ── Case C: Escalation — post review and wait for decision ────────────────
    if not needs_review:
        # Neither escalation nor de-escalation — nothing to do
        return {
            "hg_needed":          False,
            "committed_severity": committed_sev,
        }

    # Build the HumanReviewRequest
    now_dt     = datetime.now(timezone.utc)
    expires_dt = now_dt + timedelta(seconds=HUMAN_GATE_TIMEOUT_SECONDS)
    review_id  = str(uuid.uuid4())

    confidence   = state.get("forecast_confidence") or 0.0
    ttf_seconds  = state.get("time_to_failure") or -1.0
    impact_band  = state.get("impact_band")     or "None"
    urgency_band = state.get("urgency_band")    or "Distant"
    jump_size    = _detector.jump_size(committed_sev, revised_sev)
    is_large     = _detector.is_large_jump(committed_sev, revised_sev)
    summary_str  = _detector.escalation_summary(committed_sev, revised_sev)

    try:
        request = HumanReviewRequest(
            review_id          = review_id,
            incident_id        = f"inc_{episode_id}_{cycle}",
            episode_id         = episode_id,
            failure_mode       = dominant,
            failure_label      = dominant.replace("_", " ").title(),
            old_severity       = committed_sev,
            new_severity       = revised_sev,
            confidence         = float(confidence),
            ttf_seconds        = float(ttf_seconds),
            impact_band        = impact_band,
            urgency_band       = urgency_band,
            root_cause         = state.get("su_reason", ""),
            escalation_summary = summary_str,
            is_large_jump      = is_large,
            created_at         = now_dt.isoformat(),
            expires_at         = expires_dt.isoformat(),
            timeout_seconds    = HUMAN_GATE_TIMEOUT_SECONDS,
        )
    except Exception as exc:
        print(f"[n09_human_gate] ReviewRequest build error: {exc}")
        return {
            "hg_needed":          True,
            "hg_decision":        "AUTO_APPROVED",
            "hg_final_severity":  revised_sev,
            "committed_severity": revised_sev,
            "error":              f"HumanGate request error: {exc}",
        }

    # Post to SQLite + block until decision (or auto-approve)
    mgr = get_interrupt_manager()
    try:
        mgr.post_review(request)
        settled = mgr.poll_for_decision(review_id, timeout=HUMAN_GATE_TIMEOUT_SECONDS)
    except Exception as exc:
        print(f"[n09_human_gate] InterruptManager error: {exc}")
        settled = {
            "decision":     "AUTO_APPROVED",
            "operator":     "system",
            "reason":       f"Manager error: {exc}",
            "decided_at":   datetime.now(timezone.utc).isoformat(),
            "response_ms":  HUMAN_GATE_TIMEOUT_SECONDS * 1000,
        }

    # Compute final ApprovalResult
    settled["review_id"]    = review_id
    settled["incident_id"]  = f"inc_{episode_id}_{cycle}"
    settled["episode_id"]   = episode_id
    settled["old_severity"] = committed_sev
    settled["new_severity"] = revised_sev

    try:
        result = _engine.compute_result(settled)
    except Exception as exc:
        print(f"[n09_human_gate] ApprovalEngine error: {exc}")
        result_dict: dict[str, Any] = {
            "decision":        "AUTO_APPROVED",
            "final_severity":  revised_sev,
            "operator":        "system",
            "reason":          f"Engine error: {exc}",
            "response_ms":     0,
            "decided_at":      datetime.now(timezone.utc).isoformat(),
        }
        final_sev = revised_sev
    else:
        result_dict = result.to_dict()
        final_sev   = result.final_severity

    # Write to CSV
    try:
        _write_csv({
            "cycle":                cycle,
            "episode_id":           episode_id,
            "failure_mode":         dominant,
            "timestamp":            state.get("timestamp", 0.0),
            "old_severity":         committed_sev,
            "new_severity":         revised_sev,
            "final_severity":       final_sev,
            "decision":             result_dict.get("decision"),
            "operator":             result_dict.get("operator"),
            "reason":               result_dict.get("reason"),
            "is_large_jump":        int(is_large),
            "escalation_summary":   summary_str,
            "response_ms":          result_dict.get("response_ms", 0),
            "decided_at":           result_dict.get("decided_at"),
        })
    except Exception as exc:
        print(f"[n09_human_gate] CSV write error: {exc}")

    return {
        "hg_needed":            True,
        "hg_review_id":         review_id,
        "hg_decision":          result_dict.get("decision"),
        "hg_final_severity":    final_sev,
        "hg_operator":          result_dict.get("operator"),
        "hg_response_ms":       result_dict.get("response_ms", 0),
        "hg_escalation_summary": summary_str,
        "hg_is_large_jump":     is_large,
        "committed_severity":   final_sev,   # update committed severity for next cycle
    }
