"""
app_simulator/offline/run_severity_update.py
=============================================
Offline batch runner — Severity Update Node (Stage 7).

Reads:
    pipeline/output/forecasting_output.csv   (Stage 6 output)
    pipeline/output/pipeline_results.csv     (Stage 3: preliminary_severity)

Writes:
    pipeline/output/severity_update_output.csv

Usage:
    python app_simulator/offline/run_severity_update.py
"""
from __future__ import annotations

import io
import json
import re
import shutil
import sys
from pathlib import Path

# Force UTF-8 stdout encoding on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from tqdm import tqdm

from Simulator.app_data_generator_for_offline.config import (
    PIPELINE_OUTPUT_DIR,
    FORECASTING_OUTPUT_CSV,
    PIPELINE_RESULTS_CSV,
    DB_PATH,
    SEVERITY_UPDATE_OUTPUT_DIR,
    SEVERITY_UPDATE_CSV,
)

from Simulator.app_data_generator_for_offline.storage.db_writer import DbWriter
from nodes.severity_update import SeverityUpdater


def _get_next_version(out_dir: Path) -> int:
    """Find next version N by scanning out_dir for severity_update_output_N.csv files."""
    pattern = re.compile(r"^severity_update_output_(\d+)\.csv$")
    max_v = 0
    if out_dir.exists():
        for f in out_dir.iterdir():
            m = pattern.match(f.name)
            if m:
                max_v = max(max_v, int(m.group(1)))
    return max_v + 1

from Simulator.app_data_generator_for_offline.storage.db_writer import DbWriter
from nodes.severity_update import SeverityUpdater

_OUTPUT_COLS = [
    "episode_id", "failure_mode", "elapsed_s",
    "preliminary_severity",
    "forecast_confidence", "time_to_failure", "earliest_ttf_feature",
    "impact_band", "urgency_band", "gate_passed",
    "candidate_severity", "revised_severity",
    "is_escalated", "is_deescalated", "dwell_count",
    "reason",
]


def _map_preliminary_severity(raw: str | float | None) -> str:
    """
    Normalize whatever string is in pipeline_results.csv
    into P1/P2/P3/P4 for the severity_update node.
    """
    if pd.isna(raw) or raw is None:
        return "P4"
    s = str(raw).strip().upper()
    mapping = {
        "CRITICAL": "P1",
        "P1": "P1",
        "HIGH": "P2",
        "P2": "P2",
        "WARNING": "P3",
        "MODERATE": "P3",
        "P3": "P3",
        "OK": "P4",
        "NONE": "P4",
        "LOW": "P4",
        "P4": "P4",
    }
    return mapping.get(s, "P4")


def main() -> None:
    # ── Load inputs ───────────────────────────────────────────────────────────
    print("[run_severity_update] Loading CSVs ...")
    if not FORECASTING_OUTPUT_CSV.exists():
        print(f"  ERROR: {FORECASTING_OUTPUT_CSV} not found.")
        print("  Run: python app_simulator/offline/run_forecasting.py --fast")
        sys.exit(1)
    if not PIPELINE_RESULTS_CSV.exists():
        print(f"  ERROR: {PIPELINE_RESULTS_CSV} not found.")
        print("  Run: python app_simulator/run_pipeline.py")
        sys.exit(1)

    fc_df  = pd.read_csv(FORECASTING_OUTPUT_CSV)
    pip_df = pd.read_csv(PIPELINE_RESULTS_CSV)
    fc_df.columns  = fc_df.columns.str.strip()
    pip_df.columns = pip_df.columns.str.strip()

    print(f"  {len(fc_df):,} forecast rows | {len(pip_df):,} pipeline rows")

    # Get last preliminary_severity per episode from pipeline_results
    prelim_col = next(
        (c for c in ["preliminary_severity", "severity", "prelim_severity"]
         if c in pip_df.columns), None
    )
    if prelim_col is None:
        print(f"  WARNING: No preliminary_severity column found. Available: {list(pip_df.columns)}")
        print("  Defaulting all episodes to P3 (WARNING) for demonstration.")
        prelim_map = {}
    else:
        # Last preliminary_severity per episode
        prelim_map = (
            pip_df.groupby("episode_id")[prelim_col].last().to_dict()
        )

    print(f"  Found {len(prelim_map):,} episode preliminary severities")

    # ── Process each forecasted episode ──────────────────────────────────────
    updater = SeverityUpdater(dwell_k=5, min_confidence=0.75)
    output_rows = []

    # ── Initialise DbWriter for node_severity_update table ──────────────────
    pipeline_db = DbWriter(DB_PATH)
    pipeline_db.setup()

    print(f"\n[run_severity_update] Processing {len(fc_df):,} episodes ...\n")

    for _idx, row in tqdm(fc_df.iterrows(), total=len(fc_df), desc="SeverityUpdate"):
        ep_id = row["episode_id"]
        failure_mode = row.get("failure_mode", "NONE")

        # Get preliminary severity
        raw_prelim = prelim_map.get(ep_id, "P3")
        prelim_sev = _map_preliminary_severity(raw_prelim)

        # Build forecast_result dict (matches convergence schema)
        forecast_result = {
            "time_to_failure":    row.get("time_to_failure"),
            "forecast_confidence": float(row.get("forecast_confidence", 0.0)),
            "threshold_crossed":  row.get("threshold_crossed", False),
            "forecast": {
                "earliest_ttf_feature": row.get("earliest_ttf_feature"),
                "algorithm_used":       row.get("algorithm_used", "linear"),
            }
        }

        # Run severity_update Steps 1-4
        result = updater.process_cycle(
            episode_id=ep_id,
            preliminary_severity=prelim_sev,
            forecast_result=forecast_result,
        )

        output_rows.append({
            "episode_id":          ep_id,
            "failure_mode":        failure_mode,
            "elapsed_s":           row.get("elapsed_s"),
            "preliminary_severity": prelim_sev,
            "forecast_confidence": row.get("forecast_confidence"),
            "time_to_failure":     row.get("time_to_failure"),
            "earliest_ttf_feature": row.get("earliest_ttf_feature"),
            "impact_band":         result["impact_band"],
            "urgency_band":        result["urgency_band"],
            "gate_passed":         result["gate_passed"],
            "candidate_severity":  result["candidate_severity"],
            "revised_severity":    result["revised_severity"],
            "is_escalated":        result["is_escalated"],
            "is_deescalated":      result["is_deescalated"],
            "dwell_count":         result["dwell_count"],
            "reason":              result["reason"],
        })

        # ── DB: Write to node_severity_update table ─────────────────────────
        try:
            pipeline_db.write_severity_update({
                "cycle":               _idx + 1,   # 1-based row index as cycle proxy
                "episode_id":          ep_id,
                "failure_mode":        failure_mode,
                "timestamp":           None,
                "elapsed_s":           row.get("elapsed_s"),
                "preliminary_severity": prelim_sev,
                "forecast_confidence": row.get("forecast_confidence"),
                "time_to_failure":     row.get("time_to_failure"),
                "earliest_ttf_feature": row.get("earliest_ttf_feature"),
                "impact_band":         result["impact_band"],
                "urgency_band":        result["urgency_band"],
                "gate_passed":         result["gate_passed"],
                "candidate_severity":  result["candidate_severity"],
                "revised_severity":    result["revised_severity"],
                "is_escalated":        result["is_escalated"],
                "is_deescalated":      result["is_deescalated"],
                "dwell_count":         result["dwell_count"],
                "reason":              result["reason"],
            })
        except Exception as _e:
            print(f"[run_severity_update] WARN: DB write failed: {_e}")

        updater.clear_episode(ep_id)

    # ── Write output ──────────────────────────────────────────────────────────
    pipeline_db.close()
    if not output_rows:
        print("\n[run_severity_update] No rows generated.")
        return

    out_df = pd.DataFrame(output_rows)
    cols   = [c for c in _OUTPUT_COLS if c in out_df.columns]
    out_df = out_df[cols]
    
    SEVERITY_UPDATE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    version = _get_next_version(SEVERITY_UPDATE_OUTPUT_DIR)
    versioned_csv = SEVERITY_UPDATE_OUTPUT_DIR / f"severity_update_output_{version}.csv"

    out_df.to_csv(versioned_csv, index=False)
    out_df.to_csv(SEVERITY_UPDATE_CSV, index=False)

    print(f"\n[run_severity_update] SUCCESS! Done: {len(out_df):,} rows")
    print(f"  [CSV Output Version {version}] : {versioned_csv}")
    print(f"  [CSV Output Latest]    : {SEVERITY_UPDATE_CSV}")
    print(f"  [SQLite Table]         : node_severity_update in {DB_PATH.name}")
    print("\n-- Summary -----------------------------------------------------------")
    print(f"  Episodes processed  : {len(out_df):,}")
    if "revised_severity" in out_df.columns:
        print("\n  Revised Severity distribution:")
        for sev, cnt in out_df["revised_severity"].value_counts().sort_index().items():
            print(f"    {sev}  :  {cnt:>5,} episodes")
    if "is_escalated" in out_df.columns:
        esc  = out_df["is_escalated"].sum()
        desc = out_df["is_deescalated"].sum()
        print(f"\n  Immediate Escalations  : {esc:,}")
        print(f"  Dwell De-escalations   : {desc:,}")
    if "gate_passed" in out_df.columns:
        gp = out_df["gate_passed"].sum()
        print(f"  Gate Passed (conf>=0.75): {gp:,} / {len(out_df):,}")
    print("---------------------------------------------------------------------")


if __name__ == "__main__":
    main()
