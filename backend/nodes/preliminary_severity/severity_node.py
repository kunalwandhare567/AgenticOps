"""
app_simulator/pipeline/severity_node.py
=========================================
Preliminary Severity Node — wraps the DEVOPS SeverityEngine.

Replaces the old simple 3-rule severity.py.

At startup:
  - Loads DEVOPS/severity_config/thresholds.yaml (all 270 calibrated thresholds).
  - Instantiates SeverityEngine (stateful: maintains EMA + hysteresis per episode).

Per cycle:
  - Calls engine.evaluate_row(feature_values, failure_mode, episode_id)
  - Stores the full SeverityResult in PipelineState.
  - Appends one row to preliminary_severity.csv.
  - Optionally writes to the SQLite severity table via DbWriter.

What uses the simulator ground-truth failure_mode, NOT the classifier output:
  - The DEVOPS SeverityEngine applies failure_mode_weights from thresholds.yaml.
  - Using the simulator label here gives the most accurate preliminary severity.
  - After classifier + tumbling window are stable, this can be switched to
    use dominant_state instead — a one-line change in run_pipeline.py.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from app_data_generator.config import (
    DEVOPS_DIR,
    PIPELINE_OUTPUT_DIR,
    PRELIM_SEVERITY_CSV,
    SEVERITY_THRESHOLDS,
)
from app_data_generator.state import PipelineState

if TYPE_CHECKING:
    from app_data_generator.storage.db_writer import DbWriter

# ── Inject DEVOPS into sys.path once ──────────────────────────────────────────
_DEVOPS_PARENT = str(DEVOPS_DIR.parent)
if _DEVOPS_PARENT not in sys.path:
    sys.path.insert(0, _DEVOPS_PARENT)

# Lazy import — fails gracefully if DEVOPS pkg is missing
try:
    from DEVOPS.severity_engine.severity_engine import SeverityEngine  # type: ignore
    _SEVERITY_ENGINE_AVAILABLE = True
except ImportError:
    _SEVERITY_ENGINE_AVAILABLE = False

# ── CSV columns (must match schema.sql severity table) ────────────────────────
_SEVERITY_CSV_COLS = [
    "cycle",
    "episode_id",
    "timestamp",
    "elapsed_s",
    "failure_mode",
    "Severity",
    "RawSeverity",
    "WeightedScore",
    "CriticalCount",
    "WarningCount",
    "BlastSize",
    "HighRiskMode",
    "BlastRadiusGrowing",
    "Reason",
    "RecommendedAction",
]


class SeverityNode:
    """
    Stateful severity evaluation node.

    One instance per pipeline session.
    Maintains per-episode temporal state (EMA + hysteresis) inside SeverityEngine.
    """

    def __init__(self, db_writer: "DbWriter | None" = None) -> None:
        """
        Initialise the DEVOPS SeverityEngine.

        Args:
            db_writer: Optional DbWriter instance. If provided, each severity
                       result is also written to the SQLite severity table.
        """
        self._db_writer = db_writer
        self._engine: "SeverityEngine | None" = None
        self._csv_fh = None
        self._csv_writer = None
        self._warn_shown = False

        if not _SEVERITY_ENGINE_AVAILABLE:
            print(
                "[SeverityNode] WARN: DEVOPS SeverityEngine could not be imported.\n"
                "              Check that d:\\AIOps_Incident_Management\\DEVOPS is on sys.path.\n"
                "              Falling back to P4 (unknown) for all cycles."
            )
            return

        yaml_path = str(SEVERITY_THRESHOLDS)
        try:
            self._engine = SeverityEngine(yaml_path=yaml_path)
            print(f"[SeverityNode] Loaded DEVOPS SeverityEngine from:\n"
                  f"              {yaml_path}")
        except Exception as exc:
            print(f"[SeverityNode] ERROR initialising SeverityEngine: {exc}")
            self._engine = None

        # Open preliminary_severity.csv for append
        PIPELINE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        write_header = not PRELIM_SEVERITY_CSV.exists()
        self._csv_fh = PRELIM_SEVERITY_CSV.open("a", newline="", encoding="utf-8")
        self._csv_writer = csv.DictWriter(
            self._csv_fh, fieldnames=_SEVERITY_CSV_COLS, extrasaction="ignore"
        )
        if write_header:
            self._csv_writer.writeheader()

    # ── Public API ─────────────────────────────────────────────────────────────

    def evaluate(self, state: PipelineState, cycle: int) -> PipelineState:
        """
        Run the DEVOPS SeverityEngine on the current cycle's feature values.

        Fills:
            state.preliminary_severity  (P1 / P2 / P3 / P4)
            state.severity_result       (full SeverityResult object, if available)
            state.severity_reasons      (human-readable reason string as list)

        Also appends one row to preliminary_severity.csv and SQLite.

        Args:
            state: PipelineState after Feature Engineering.
            cycle: Global pipeline cycle counter (for CSV row numbering).

        Returns:
            Same state with severity fields filled.
        """
        # Fallback if engine is unavailable
        if self._engine is None:
            if not self._warn_shown:
                print("[SeverityNode] WARN: running in fallback mode (P4 for all cycles).")
                self._warn_shown = True
            state.preliminary_severity = "P4"
            state.severity_reasons     = ["SeverityEngine unavailable"]
            return state

        features = dict(state.classifier_input)

        try:
            result = self._engine.evaluate_row(
                feature_values=features,
                failure_mode=state.failure_mode,
                episode_id=state.episode_id,
                timestamp=str(state.timestamp),
                elapsed_s=state.elapsed_s,
            )

            # Write to state
            state.preliminary_severity = result.severity
            state.severity_reasons     = [result.reason]

            # Attach the full result for downstream nodes that want it
            state.severity_result = result

            # ── Persist to CSV ────────────────────────────────────────────────
            row = {
                "cycle":             cycle,
                "episode_id":        result.episode_id,
                "timestamp":         result.timestamp,
                "elapsed_s":         result.elapsed_s,
                "failure_mode":      result.failure_mode,
                "Severity":          result.severity,
                "RawSeverity":       result.raw_severity,
                "WeightedScore":     result.weighted_score,
                "CriticalCount":     result.critical_count,
                "WarningCount":      result.warning_count,
                "BlastSize":         result.blast_size,
                "HighRiskMode":      int(result.high_risk_mode),
                "BlastRadiusGrowing":int(result.blast_radius_growing),
                "Reason":            result.reason,
                "RecommendedAction": result.recommended_action,
            }
            if self._csv_writer is not None:
                self._csv_writer.writerow(row)
                self._csv_fh.flush()

            # ── Persist to SQLite ─────────────────────────────────────────────
            if self._db_writer is not None:
                try:
                    self._db_writer.write_severity(row)
                except Exception as db_exc:
                    print(f"[SeverityNode] WARN: DB write failed: {db_exc}")

        except Exception as exc:
            print(f"[SeverityNode] ERROR during evaluate_row: {exc}")
            state.preliminary_severity = "P4"
            state.severity_reasons     = [f"Engine error: {exc}"]

        return state

    def close(self) -> None:
        """Flush and close the CSV file handle."""
        if self._csv_fh is not None:
            self._csv_fh.close()
            self._csv_fh = None
