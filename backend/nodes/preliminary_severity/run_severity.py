"""
app_simulator/offline/run_severity.py
======================================
Offline Preliminary Severity batch processor.

Usage:
    python app_simulator/offline/run_severity.py

Steps:
    1. Deletes any existing pipeline/output/preliminary_severity.csv.
    2. Deletes old records from SQLite 'severity' table.
    3. Reads engineered_features.csv.
    4. Evaluates row-level severity using DEVOPS SeverityEngine (thresholds.yaml).
    5. Saves outputs to preliminary_severity.csv and the database.
"""
from __future__ import annotations

import csv
import os
import re
import shutil
import sqlite3
import sys
from pathlib import Path

# ── resolve package root ─────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent))

from Simulator.app_data_generator_for_offline.config import (
    DB_PATH,
    ENGINEERED_FEAT_CSV,
    PRELIM_SEVERITY_CSV,
    PRELIM_SEVERITY_OUTPUT_DIR,
)
from nodes.preliminary_severity.severity_node import SeverityNode
from Simulator.app_data_generator_for_offline.state import PipelineState
from Simulator.app_data_generator_for_offline.storage.db_writer import DbWriter


def _get_next_version(out_dir: Path) -> int:
    """Find next version N by scanning out_dir for preliminary_severity_N.csv files."""
    pattern = re.compile(r"^preliminary_severity_(\d+)\.csv$")
    max_v = 0
    if out_dir.exists():
        for f in out_dir.iterdir():
            m = pattern.match(f.name)
            if m:
                max_v = max(max_v, int(m.group(1)))
    return max_v + 1


def main() -> None:
    print("=" * 65)
    print("  Offline Preliminary Severity Batch Runner")
    print("=" * 65)

    PRELIM_SEVERITY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    version = _get_next_version(PRELIM_SEVERITY_OUTPUT_DIR)
    versioned_csv = PRELIM_SEVERITY_OUTPUT_DIR / f"preliminary_severity_{version}.csv"

    # Check engineered features exist
    if not ENGINEERED_FEAT_CSV.exists():
        print(f"[ERROR] Engineered features not found at {ENGINEERED_FEAT_CSV}.")
        print("        Run Node 2 (orchestrator.py) first!")
        sys.exit(1)

    # Clean up SQLite severity table
    db_writer = None
    if DB_PATH.exists():
        print(f"[Severity] Initialising DbWriter for database: {DB_PATH}...")
        try:
            db_writer = DbWriter(DB_PATH)
            db_writer.setup()
        except Exception as exc:
            print(f"[Severity] Warning initialising database: {exc}")

    # Initialise SeverityNode (which wraps the DEVOPS SeverityEngine)
    node = SeverityNode(db_writer=db_writer)

    try:
        # Load features from CSV
        print(f"[Severity] Reading features from: {ENGINEERED_FEAT_CSV}")
        with ENGINEERED_FEAT_CSV.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            # Filter out any duplicated header rows in CSV if present
            rows = [r for r in reader if r.get("episode_id") and r.get("episode_id") != "episode_id"]

        total_rows = len(rows)
        print(f"[Severity] Found {total_rows:,} feature rows across all episodes.")

        if total_rows == 0:
            print("[ERROR] Features CSV is empty.")
            sys.exit(1)

        print("[Severity] Processing all episodes through DEVOPS SeverityEngine...")
        processed_count = 0

        for r in rows:
            processed_count += 1
            # Build mock state
            state = PipelineState(
                episode_id   = r.get("episode_id", ""),
                failure_mode = r.get("failure_mode", "NONE"),
                timestamp    = float(r.get("timestamp", 0.0)),
                elapsed_s    = float(r.get("elapsed_s", 0.0)),
            )
            # Reconstruct classifier input (cast back to numeric)
            clean_feats = {}
            for k, v in r.items():
                if k not in ["episode_id", "failure_mode", "timestamp", "elapsed_s"]:
                    if v is None or v == "":
                        clean_feats[k] = 0.0
                    else:
                        try:
                            clean_feats[k] = float(v)
                        except (ValueError, TypeError):
                            clean_feats[k] = v
            state.classifier_input = clean_feats

            # Run evaluation
            node.evaluate(state, cycle=processed_count)

            if processed_count % 10000 == 0 or processed_count == total_rows:
                print(f"  Processed {processed_count:,} / {total_rows:,} rows...")

        # Close node handle to flush preliminary_severity.csv
        node.close()

        # Copy preliminary_severity.csv -> preliminary_severity_N.csv
        if PRELIM_SEVERITY_CSV.exists():
            shutil.copy(str(PRELIM_SEVERITY_CSV), str(versioned_csv))

        print(f"\n[Severity] SUCCESS! Processed all {total_rows:,} rows across all episodes.")
        print(f"  [CSV Output Version {version}] : {versioned_csv}")
        print(f"  [CSV Output Latest]    : {PRELIM_SEVERITY_CSV}")
        print(f"  [SQLite Table]         : node_preliminary_severity in {DB_PATH.name}")

    except Exception as exc:
        print(f"[ERROR] Severity execution failed: {exc}")
        sys.exit(1)
    finally:
        if db_writer:
            db_writer.close()


if __name__ == "__main__":
    main()

