"""
app_simulator/run_pipeline.py
==============================
AIOps Pipeline Runner — Terminal 2 process.

Pipeline node order (per design document):
  [1] Collect       -- dequeue from TelemetryQueue or poll SQLite
  [2] Feature Eng.  -- 27 raw metric features + 5 Drain3 log features
  [3] Prelim Sev.   -- DEVOPS SeverityEngine (thresholds.yaml, EMA+hysteresis)
  [4] Classify      -- LightGBM (auto-trains if model .pkl is missing)
  [5] Tumbling Win. -- 10-cycle majority vote on labels ONLY (NOT raw features)
                       Labels carry forward across episode boundaries.

Output files (pipeline/output/):
    engineered_features.csv       -- FE features per cycle (from orchestrator)
    preliminary_severity.csv      -- Full DEVOPS severity result per cycle
    pipeline_results.csv          -- Combined per-cycle row: FE + severity + classification + window
    tumbling_window_output.csv    -- Window state per cycle: dominant_state, votes, margin

Usage:
    # Terminal 1: python app_simulator/run_simulator.py
    # Terminal 2: python app_simulator/run_pipeline.py
"""
from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
import time
from pathlib import Path

# ── resolve package root ─────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE
sys.path.insert(0, str(PROJECT_ROOT))

from Simulator.app_data_generator_for_offline.config import (
    DB_PATH, DRAIN_INI, DRAIN_STATE, KNOWN_TEMPLATES_JSON,
    PIPELINE_OUTPUT_DIR, PIPELINE_RESULTS_CSV,
    PRELIM_SEVERITY_CSV, TUMBLING_WINDOW_CSV,
)
from Simulator.app_data_generator_for_offline.storage.db_writer import DbWriter
from nodes.collect.queue_bridge import TelemetryQueue
from Simulator.app_data_generator_for_offline.state import PipelineState
from nodes.preliminary_severity.severity_node import SeverityNode
from nodes.classification.classifier import load_classifier, classify
from nodes.tumbling_window.tumbling_window import TumblingWindow
from nodes.feature_engineering.orchestrator import run_feature_engineering_from_raw
from nodes.feature_engineering.log_features import (
    load_template_miner, load_known_template_ids,
)


# ── Result CSV schema ────────────────────────────────────────────────────────
RESULT_FIELDS = [
    "cycle",
    "episode_id",
    "failure_mode",
    "timestamp",
    "elapsed_s",
    # Severity
    "preliminary_severity",
    "severity_raw",
    "severity_weighted_score",
    "severity_critical_count",
    "severity_warning_count",
    "severity_blast_size",
    "severity_reason",
    # Classification
    "predicted_failure",
    "prediction_probability",
    # Tumbling window
    "dominant_state",
    "vote_distribution",
    "window_margin",
    "window_full",
    "window_size",
]


def _open_results_csv() -> tuple:
    """Open pipeline_results.csv for append. Return (file, writer)."""
    PIPELINE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not PIPELINE_RESULTS_CSV.exists()
    fh = PIPELINE_RESULTS_CSV.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=RESULT_FIELDS, extrasaction="ignore")
    if write_header:
        writer.writeheader()
    return fh, writer


def _print_banner(model_loaded: bool, db_mode: bool) -> None:
    sev_status = "DEVOPS SeverityEngine (thresholds.yaml)"
    cls_status = "Loaded" if model_loaded else "PENDING (auto-train on data)"
    mode_str   = "Database polling (simulator_db.sqlite)" if db_mode else "TelemetryQueue"
    print(f"\n{'='*65}")
    print(f"  AIOps Pipeline Runner")
    print(f"  Severity   : {sev_status}")
    print(f"  Classifier : {cls_status}")
    print(f"  Mode       : {mode_str}")
    print(f"  Results    : {PIPELINE_RESULTS_CSV}")
    print(f"  Severity   : {PRELIM_SEVERITY_CSV}")
    print(f"  Window     : {TUMBLING_WINDOW_CSV}")
    print(f"{'='*65}")
    print("  Press Ctrl+C to stop.\n")


# =============================================================================
# Database Polling Helper
# =============================================================================

def _get_next_db_row(
    conn: sqlite3.Connection, last_id: int
) -> tuple[int, dict, list[dict], list[dict]] | None:
    """Query the next row from SQLite. Returns (id, metric, logs, spans)."""
    try:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='metrics'"
        )
        if not cur.fetchone():
            return None

        cur = conn.execute(
            "SELECT * FROM metrics WHERE id > ? ORDER BY id ASC LIMIT 1", (last_id,)
        )
        row = cur.fetchone()
        if not row:
            return None

        cols   = [d[0] for d in cur.description]
        metric = dict(zip(cols, row))
        row_id = metric["id"]
        ep_id  = metric["episode_id"]
        ts     = metric["timestamp"]

        # Logs for this cycle
        cur  = conn.execute(
            "SELECT log_level, exception_type, log_message "
            "FROM logs WHERE episode_id = ? AND timestamp = ?",
            (ep_id, ts),
        )
        logs = [
            {"log_level": r[0], "exception_type": r[1], "log_message": r[2]}
            for r in cur.fetchall()
        ]

        # Spans (passed through for future trace-based features)
        cur   = conn.execute(
            "SELECT * FROM traces WHERE episode_id = ? AND timestamp = ?", (ep_id, ts)
        )
        s_cols = [d[0] for d in cur.description]
        spans  = [dict(zip(s_cols, r)) for r in cur.fetchall()]

        return row_id, metric, logs, spans

    except Exception as exc:
        print(f"[Pipeline] DB Query Error: {exc}")
        return None


# =============================================================================
# Main pipeline loop
# =============================================================================

def run_pipeline() -> None:
    """
    Main pipeline loop.

    Node order:
      Collect → Feature Engineering → Preliminary Severity
      → Classification (auto-train) → Tumbling Window (labels only)
    """
    # ── One-time startup ────────────────────────────────────────────────────
    print("[Pipeline] Loading Drain3 artifacts ...")
    template_miner = load_template_miner(str(DRAIN_STATE), str(DRAIN_INI))
    known_ids      = load_known_template_ids(str(KNOWN_TEMPLATES_JSON))

    # ── Initialise DbWriter (node outputs + combined pipeline_results) ──────
    print("[Pipeline] Initialising DbWriter (node output tables) ...")
    pipeline_db = DbWriter(DB_PATH)
    pipeline_db.setup()

    print("[Pipeline] Initialising DEVOPS SeverityEngine ...")
    severity_node = SeverityNode(db_writer=None)   # uses its own CSV writer

    print("[Pipeline] Loading LightGBM classifier ...")
    model_loaded = load_classifier()

    # Open SQLite connection (read-only polling from simulator_db)
    db_conn = None
    if DB_PATH.exists():
        db_conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)

    db_mode = db_conn is not None
    _print_banner(model_loaded, db_mode)

    window            = TumblingWindow()
    fh, results_writer = _open_results_csv()

    cycle             = 0
    last_processed_id = 0
    idle_warn         = 0

    try:
        while True:
            # ── 1. Collect ────────────────────────────────────────────────────
            item   = TelemetryQueue.pop()
            metric, log, spans = None, None, None

            if item is not None:
                metric, log, spans = item
            elif db_conn is not None:
                db_row = _get_next_db_row(db_conn, last_processed_id)
                if db_row is not None:
                    row_id, metric, log, spans = db_row
                    last_processed_id = row_id

            if metric is None:
                # Attempt to open DB if it appeared after startup
                if db_conn is None and DB_PATH.exists():
                    db_conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
                    db_mode = True
                    print("[Pipeline] Detected database. Switching to DB polling mode.")

                idle_warn += 1
                if idle_warn == 50:
                    print("[Pipeline] Waiting for data from simulator ...")
                    print("[Pipeline] Run: python app_simulator/run_simulator.py")
                time.sleep(0.1)
                continue

            idle_warn = 0
            cycle    += 1

            # ── Build base PipelineState ────────────────────────────────────
            state = PipelineState(
                raw_metric   = metric,
                raw_log      = log if isinstance(log, dict) else (log[0] if log else {}),
                raw_traces   = spans,
                episode_id   = metric.get("episode_id",   ""),
                failure_mode = metric.get("failure_mode", "NONE"),
                timestamp    = float(metric.get("timestamp", 0.0)),
                elapsed_s    = float(metric.get("elapsed_s",  0.0)),
                step         = cycle,
                service      = metric.get("service", ""),
            )

            # ── 2. Feature Engineering ───────────────────────────────────────
            # log param: orchestrator expects a single dict or list-of-dicts
            log_arg = log if log else {}
            fe_result = run_feature_engineering_from_raw(
                metric             = metric,
                log                = log_arg,
                episode_id         = state.episode_id,
                failure_mode       = state.failure_mode,
                timestamp          = state.timestamp,
                elapsed_s          = state.elapsed_s,
                template_miner     = template_miner,
                known_template_ids = known_ids,
            )
            state.classifier_input = fe_result.get("classifier_input", {})
            state.evidence         = fe_result.get("evidence", {})

            # ── DB: Feature Engineering node output ───────────────────────────
            fe_feats = state.classifier_input or {}
            try:
                pipeline_db.write_feature_engineering({
                    "cycle":       cycle,
                    "episode_id":  state.episode_id,
                    "failure_mode": state.failure_mode,
                    "timestamp":   state.timestamp,
                    "elapsed_s":   state.elapsed_s,
                    **fe_feats,
                })
            except Exception as _e:
                print(f"[Pipeline] WARN: FE DB write failed: {_e}")

            # ── 3. Preliminary Severity (DEVOPS SeverityEngine) ───────────────
            state = severity_node.evaluate(state, cycle)

            # ── DB: Preliminary Severity node output ──────────────────────────
            sev = state.severity_result  # full SeverityResult or None
            try:
                pipeline_db.write_preliminary_severity({
                    "cycle":                 cycle,
                    "episode_id":            state.episode_id,
                    "failure_mode":          state.failure_mode,
                    "timestamp":             state.timestamp,
                    "elapsed_s":             state.elapsed_s,
                    "preliminary_severity":  state.preliminary_severity,
                    "severity_raw":          getattr(sev, "raw_severity", None),
                    "weighted_score":        getattr(sev, "weighted_score", None),
                    "critical_count":        getattr(sev, "critical_count", None),
                    "warning_count":         getattr(sev, "warning_count", None),
                    "blast_size":            getattr(sev, "blast_size", None),
                    "high_risk_mode":        int(getattr(sev, "high_risk_mode", False)),
                    "blast_radius_growing":  int(getattr(sev, "blast_radius_growing", False)),
                    "reason":                getattr(sev, "reason", None),
                    "recommended_action":    getattr(sev, "recommended_action", None),
                })
            except Exception as _e:
                print(f"[Pipeline] WARN: PrelimSev DB write failed: {_e}")

            # ── 4. Classification (auto-train if needed) ──────────────────────
            state = classify(state)

            # ── DB: Classification node output ────────────────────────────────
            try:
                pipeline_db.write_classification({
                    "cycle":                   cycle,
                    "episode_id":              state.episode_id,
                    "failure_mode":            state.failure_mode,
                    "timestamp":               state.timestamp,
                    "elapsed_s":               state.elapsed_s,
                    "predicted_failure":       state.predicted_failure,
                    "prediction_probability":  state.prediction_probability,
                })
            except Exception as _e:
                print(f"[Pipeline] WARN: Classification DB write failed: {_e}")

            # ── 5. Tumbling Window (labels ONLY — raw features untouched) ─────
            state = window.update(state, cycle)

            # ── DB: Tumbling Window node output ───────────────────────────────
            try:
                pipeline_db.write_tumbling_window({
                    "cycle":             cycle,
                    "episode_id":        state.episode_id,
                    "failure_mode":      state.failure_mode,
                    "timestamp":         state.timestamp,
                    "elapsed_s":         state.elapsed_s,
                    "dominant_state":    state.summarized_failure,
                    "vote_distribution": state.vote_distribution,
                    "window_margin":     state.window_margin,
                    "window_full":       state.window_full,
                    "window_size":       len(state.window_predictions),
                })
            except Exception as _e:
                print(f"[Pipeline] WARN: TumblingWindow DB write failed: {_e}")

            # ── Write combined pipeline_results.csv ───────────────────────────
            sev = state.severity_result  # full SeverityResult or None
            results_writer.writerow({
                "cycle":                    cycle,
                "episode_id":               state.episode_id,
                "failure_mode":             state.failure_mode,
                "timestamp":                state.timestamp,
                "elapsed_s":                state.elapsed_s,
                # Severity
                "preliminary_severity":     state.preliminary_severity,
                "severity_raw":             getattr(sev, "raw_severity", ""),
                "severity_weighted_score":  getattr(sev, "weighted_score", ""),
                "severity_critical_count":  getattr(sev, "critical_count", ""),
                "severity_warning_count":   getattr(sev, "warning_count", ""),
                "severity_blast_size":      getattr(sev, "blast_size", ""),
                "severity_reason":          getattr(sev, "reason", ""),
                # Classification
                "predicted_failure":        state.predicted_failure,
                "prediction_probability":   state.prediction_probability,
                # Tumbling window
                "dominant_state":           state.summarized_failure,
                "vote_distribution":        str(state.vote_distribution),
                "window_margin":            state.window_margin,
                "window_full":              state.window_full,
                "window_size":              len(state.window_predictions),
            })
            fh.flush()

            # ── DB: Combined pipeline_results row ─────────────────────────────
            try:
                pipeline_db.write_pipeline_results({
                    "cycle":                      cycle,
                    "episode_id":                 state.episode_id,
                    "failure_mode":               state.failure_mode,
                    "timestamp":                  state.timestamp,
                    "elapsed_s":                  state.elapsed_s,
                    # Key FE metrics (prefixed fe_)
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
                    # Preliminary Severity
                    "preliminary_severity":       state.preliminary_severity,
                    "severity_weighted_score":    getattr(sev, "weighted_score", None),
                    "severity_critical_count":    getattr(sev, "critical_count", None),
                    "severity_warning_count":     getattr(sev, "warning_count", None),
                    "severity_blast_size":        getattr(sev, "blast_size", None),
                    "severity_reason":            getattr(sev, "reason", None),
                    # Classification
                    "predicted_failure":          state.predicted_failure,
                    "prediction_probability":     state.prediction_probability,
                    # Tumbling Window
                    "dominant_state":             state.summarized_failure,
                    "vote_distribution":          state.vote_distribution,
                    "window_margin":              state.window_margin,
                    "window_full":                state.window_full,
                    "window_size":                len(state.window_predictions),
                    # Forecasting — not computed in the live loop yet (see forecasting node)
                    # These are populated when forecasting is wired in
                    "forecast_algorithm":         None,
                    "time_to_failure":            None,
                    "forecast_confidence":        None,
                    "threshold_crossed":          None,
                    "earliest_ttf_feature":       None,
                    # Severity Update — populated by offline batch runner
                    "revised_severity":           None,
                    "candidate_severity":         None,
                    "impact_band":                None,
                    "urgency_band":               None,
                    "gate_passed":                None,
                    "is_escalated":               None,
                    "is_deescalated":             None,
                    "su_reason":                  None,
                    # Human Gate — NULL until settled
                    "hg_review_id":               None,
                    "hg_decision":                None,
                    "hg_final_severity":          None,
                    "hg_operator":                None,
                    "hg_response_ms":             None,
                })
            except Exception as _e:
                print(f"[Pipeline] WARN: pipeline_results DB write failed: {_e}")

            # ── Console summary ───────────────────────────────────────────────
            sev_str  = state.preliminary_severity
            pred_str = f"{state.predicted_failure}({state.prediction_probability:.2f})"
            win_str  = (
                f"{state.summarized_failure}"
                f"({state.window_predictions.count(state.summarized_failure)}/{len(state.window_predictions)})"
                if state.window_predictions else "cold-start"
            )
            print(
                f"[cycle {cycle:>5}] "
                f"mode={state.failure_mode:<22} "
                f"sev={sev_str}  "
                f"pred={pred_str:<32}  "
                f"win={win_str}"
            )

    except KeyboardInterrupt:
        print(f"\n\n  [Pipeline] Stopped by user after {cycle:,} cycles.")

    finally:
        fh.close()
        severity_node.close()
        window.close()
        pipeline_db.close()
        if db_conn:
            db_conn.close()
        print(f"\n  Results    : {PIPELINE_RESULTS_CSV}")
        print(f"  Severity   : {PRELIM_SEVERITY_CSV}")
        print(f"  Window     : {TUMBLING_WINDOW_CSV}")
        print(f"  Features   : {PIPELINE_OUTPUT_DIR}/engineered_features.csv")
        print(f"  Database   : {DB_PATH}\n")


# =============================================================================
# Entry point
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "AIOps Pipeline Runner — Collect → FE → Prelim Severity → "
            "Classification → Tumbling Window"
        )
    )
    parser.parse_args()
    run_pipeline()


if __name__ == "__main__":
    main()
