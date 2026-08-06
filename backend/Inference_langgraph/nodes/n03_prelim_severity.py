"""
backend/langgraph_pipeline/nodes/n03_prelim_severity.py
=========================================================
LangGraph Node 3 — Preliminary Severity.

Wraps SeverityNode.evaluate() from nodes/preliminary_severity/severity_node.py.
Converts the SeverityResult dataclass to a plain dict for JSON-serializable state.
Writes one row to preliminary_severity.csv.

Returns:
    preliminary_severity  — "P1" | "P2" | "P3" | "P4"
    severity_result       — full SeverityResult as dict
"""
from __future__ import annotations

import csv
import threading
from pathlib import Path
from typing import Any

import sys
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent.parent))

from Simulator.app_data_generator_for_offline.config import PRELIM_SEVERITY_CSV
from Simulator.app_data_generator_for_offline.state import PipelineState
from nodes.preliminary_severity.severity_node import SeverityNode
from Inference_langgraph.state import AIOpsLangState

# ── SeverityNode singleton ────────────────────────────────────────────────────
_lock         = threading.Lock()
_severity_node: SeverityNode | None = None

# ── CSV file handle ───────────────────────────────────────────────────────────
_csv_fh     = None
_csv_writer = None
_csv_lock   = threading.Lock()


def init() -> None:
    """Initialise SeverityNode singleton. Called once at pipeline startup."""
    global _severity_node, _csv_fh, _csv_writer
    with _lock:
        if _severity_node is None:
            print("[n03_prelim_severity] Initialising SeverityEngine...")
            _severity_node = SeverityNode(db_writer=None)
    with _csv_lock:
        if _csv_fh is None:
            PRELIM_SEVERITY_CSV.parent.mkdir(parents=True, exist_ok=True)
            _csv_fh = PRELIM_SEVERITY_CSV.open("a", newline="", encoding="utf-8")


def _write_csv(row: dict) -> None:
    global _csv_writer
    with _csv_lock:
        if _csv_writer is None:
            write_header = _csv_fh.tell() == 0
            _csv_writer  = csv.DictWriter(
                _csv_fh, fieldnames=list(row.keys()), extrasaction="ignore"
            )
            if write_header:
                _csv_writer.writeheader()
        _csv_writer.writerow(row)
        _csv_fh.flush()


def _sev_result_to_dict(sev_result: Any) -> dict:
    """Convert SeverityResult dataclass or dict to a plain dict."""
    if sev_result is None:
        return {}
    if isinstance(sev_result, dict):
        return sev_result
    # Dataclass → dict via to_dict() if available, else vars()
    if hasattr(sev_result, "to_dict"):
        return sev_result.to_dict()
    return vars(sev_result)


# =============================================================================
# LangGraph Node Function
# =============================================================================

def run(state: AIOpsLangState) -> dict[str, Any]:
    """Preliminary severity node — evaluates DEVOPS SeverityEngine thresholds."""
    global _severity_node
    if _severity_node is None:
        init()

    cycle = state.get("cycle", 0)

    # Build PipelineState that SeverityNode expects
    ps = PipelineState(
        raw_metric        = state.get("raw_metric", {}),
        raw_log           = state.get("raw_log", [{}])[0] if state.get("raw_log") else {},
        raw_traces        = state.get("raw_traces", []),
        episode_id        = state.get("episode_id", ""),
        failure_mode      = state.get("failure_mode", "NONE"),
        timestamp         = state.get("timestamp", 0.0),
        elapsed_s         = state.get("elapsed_s", 0.0),
        step              = cycle,
        service           = state.get("service", ""),
        classifier_input  = state.get("classifier_input", {}),
        evidence          = state.get("evidence", {}),
    )

    try:
        updated_ps = _severity_node.evaluate(ps, cycle)
    except Exception as exc:
        print(f"[n03_prelim_severity] Error: {exc}")
        return {
            "preliminary_severity": "P4",
            "severity_result":      {},
            "error":                f"PrelimSev error: {exc}",
        }

    sev_result_dict = _sev_result_to_dict(updated_ps.severity_result)
    preliminary_sev = str(updated_ps.preliminary_severity or "P4")

    # Write to CSV
    try:
        _write_csv({
            "cycle":                cycle,
            "episode_id":           state.get("episode_id", ""),
            "failure_mode":         state.get("failure_mode", "NONE"),
            "timestamp":            state.get("timestamp", 0.0),
            "elapsed_s":            state.get("elapsed_s", 0.0),
            "preliminary_severity": preliminary_sev,
            **{f"sev_{k}": v for k, v in sev_result_dict.items()},
        })
    except Exception as exc:
        print(f"[n03_prelim_severity] CSV write error: {exc}")

    return {
        "preliminary_severity": preliminary_sev,
        "severity_result":      sev_result_dict,
    }
