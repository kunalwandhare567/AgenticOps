"""
backend/langgraph_pipeline/nodes/n07_severity_update.py
=========================================================
LangGraph Node 7 — Severity Update.

Wraps SeverityUpdater.process_cycle() from nodes/severity_update/updater.py.
Uses the forecast result (TTF, confidence) to revise the preliminary severity
upward or downward.

Escalation logic (managed by SeverityUpdater internally):
  - Impact Band   : derived from preliminary_severity string
  - Urgency Band  : derived from time_to_failure + confidence (with gate)
  - Matrix Lookup : (Impact × Urgency) → candidate_severity
  - Hysteresis    : candidate must persist K cycles before de-escalating
  - Escalation    : always immediate (no dwell required to go UP)
  - De-escalation : requires dwell_k cycles of sustained lower candidate

Writes one row to severity_update_output.csv.

Returns:
    revised_severity, candidate_severity, impact_band, urgency_band,
    gate_passed, is_escalated, is_deescalated, su_reason, dwell_count
"""
from __future__ import annotations

import csv
import threading
from pathlib import Path
from typing import Any

import sys
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent.parent))

from Simulator.app_data_generator_for_offline.config import SEVERITY_UPDATE_CSV
from Inference_langgraph.nodes.severity_update.updater import SeverityUpdater
from Inference_langgraph.state import AIOpsLangState

# ── SeverityUpdater singleton ─────────────────────────────────────────────────
_lock    = threading.Lock()
_updater: SeverityUpdater | None = None

# ── CSV ───────────────────────────────────────────────────────────────────────
_csv_fh     = None
_csv_writer = None
_csv_lock   = threading.Lock()


def init() -> None:
    """Initialise SeverityUpdater singleton. Called once at pipeline startup."""
    global _updater
    with _lock:
        if _updater is None:
            print("[n07_severity_update] Initialising SeverityUpdater...")
            _updater = SeverityUpdater(dwell_k=5, min_confidence=0.75)


def _write_csv(row: dict) -> None:
    global _csv_writer, _csv_fh
    with _csv_lock:
        if _csv_fh is None:
            SEVERITY_UPDATE_CSV.parent.mkdir(parents=True, exist_ok=True)
            _csv_fh = SEVERITY_UPDATE_CSV.open("a", newline="", encoding="utf-8")
        if _csv_writer is None:
            write_header = _csv_fh.tell() == 0
            _csv_writer  = csv.DictWriter(
                _csv_fh, fieldnames=list(row.keys()), extrasaction="ignore"
            )
            if write_header:
                _csv_writer.writeheader()
        _csv_writer.writerow(row)
        _csv_fh.flush()


# =============================================================================
# LangGraph Node Function
# =============================================================================

def run(state: AIOpsLangState) -> dict[str, Any]:
    """
    Severity update node — revises preliminary severity using forecast data.

    Escalation: always immediate when candidate_severity < preliminary_severity.
    De-escalation: requires dwell_k=5 consecutive cycles of lower candidate.
    """
    global _updater
    if _updater is None:
        init()

    cycle             = state.get("cycle", 0)
    episode_id        = state.get("episode_id", "")
    prelim_sev        = state.get("preliminary_severity", "P4")
    forecast_result   = state.get("forecast_result", {})

    try:
        su_result = _updater.process_cycle(
            episode_id           = episode_id,
            preliminary_severity = prelim_sev,
            forecast_result      = forecast_result,
        )
    except Exception as exc:
        print(f"[n07_severity_update] Error: {exc}")
        return {
            "revised_severity":   prelim_sev,
            "candidate_severity": prelim_sev,
            "impact_band":        None,
            "urgency_band":       None,
            "gate_passed":        False,
            "is_escalated":       False,
            "is_deescalated":     False,
            "su_reason":          f"Error: {exc}",
            "dwell_count":        0,
            "error":              f"SeverityUpdate error: {exc}",
        }

    # Write to CSV
    try:
        _write_csv({
            "cycle":                 cycle,
            "episode_id":            episode_id,
            "failure_mode":          state.get("failure_mode", "NONE"),
            "timestamp":             state.get("timestamp", 0.0),
            "elapsed_s":             state.get("elapsed_s", 0.0),
            "preliminary_severity":  prelim_sev,
            "forecast_confidence":   state.get("forecast_confidence"),
            "time_to_failure":       state.get("time_to_failure"),
            "revised_severity":      su_result["revised_severity"],
            "candidate_severity":    su_result["candidate_severity"],
            "impact_band":           su_result["impact_band"],
            "urgency_band":          su_result["urgency_band"],
            "gate_passed":           int(su_result["gate_passed"]),
            "is_escalated":          int(su_result["is_escalated"]),
            "is_deescalated":        int(su_result["is_deescalated"]),
            "dwell_count":           su_result.get("dwell_count", 0),
            "su_reason":             su_result.get("reason", ""),
        })
    except Exception as exc:
        print(f"[n07_severity_update] CSV write error: {exc}")

    return {
        "revised_severity":   su_result["revised_severity"],
        "candidate_severity": su_result["candidate_severity"],
        "impact_band":        su_result["impact_band"],
        "urgency_band":       su_result["urgency_band"],
        "gate_passed":        su_result["gate_passed"],
        "is_escalated":       su_result["is_escalated"],
        "is_deescalated":     su_result["is_deescalated"],
        "su_reason":          su_result.get("reason", ""),
        "dwell_count":        su_result.get("dwell_count", 0),
    }
