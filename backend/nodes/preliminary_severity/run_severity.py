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
import sqlite3
import sys
from pathlib import Path

# ── resolve package root ─────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent))

from app_data_generator.config import (
    DB_PATH,
    ENGINEERED_FEAT_CSV,
    PRELIM_SEVERITY_CSV,
)
from nodes.preliminary_severity.severity_node import SeverityNode
from app_data_generator.state import PipelineState
from app_data_generator.storage.db_writer import DbWriter


def main() -> None:
    print("=" * 65)
    print("  Offline Preliminary Severity Batch Runner")
    print("=" * 65)

    # 1. Clean up old severity CSV
    if PRELIM_SEVERITY_CSV.exists():
        print(f"[Severity] Removing existing severity CSV: {PRELIM_SEVERITY_CSV}")
        PRELIM_SEVERITY_CSV.unlink()

    # Check engineered features exist
    if not ENGINEERED_FEAT_CSV.exists():
        print(f"[ERROR] Engineered features not found at {ENGINEERED_FEAT_CSV}.")
        print("        Run Node 2 (run_feature_engineering.py) first!")
        sys.exit(1)

    # Clean up SQLite severity table
    db_writer = None
    if DB_PATH.exists():
        print(f"[Severity] Clearing existing 'severity' table in {DB_PATH}...")
        try:
            conn = sqlite3.connect(str(DB_PATH))
            conn.execute("DELETE FROM severity")
            conn.commit()
            conn.close()
            db_writer = DbWriter(DB_PATH)
            db_writer.setup()
        except Exception as exc:
            print(f"[Severity] Warning clearing database: {exc}")

    # Initialise SeverityNode (which wraps the DEVOPS SeverityEngine)
    node = SeverityNode(db_writer=db_writer)

    try:
        # Load features from CSV
        print(f"[Severity] Reading features from: {ENGINEERED_FEAT_CSV}")
        with ENGINEERED_FEAT_CSV.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        total_rows = len(rows)
        print(f"[Severity] Found {total_rows:,} feature rows.")

        if total_rows == 0:
            print("[ERROR] Features CSV is empty.")
            sys.exit(1)

        print("[Severity] Processing rows through SeverityEngine...")
        processed_count = 0

        for r in rows:
            processed_count += 1
            # Build mock state
            state = PipelineState(
                episode_id   = r["episode_id"],
                failure_mode = r["failure_mode"],
                timestamp    = float(r["timestamp"]),
                elapsed_s    = float(r["elapsed_s"]),
            )
            # Reconstruct classifier input (cast back to numeric)
            clean_feats = {}
            for k, v in r.items():
                if k not in ["episode_id", "failure_mode", "timestamp", "elapsed_s"]:
                    try:
                        clean_feats[k] = float(v)
                    except ValueError:
                        clean_feats[k] = v
            state.classifier_input = clean_feats

            # Run evaluation
            node.evaluate(state, cycle=processed_count)

            if processed_count % 1000 == 0 or processed_count == total_rows:
                print(f"  Processed {processed_count:,} / {total_rows:,} rows...")

        print(f"\n[Severity] Success! Severity written to: {PRELIM_SEVERITY_CSV}")

    except Exception as exc:
        print(f"[ERROR] Severity execution failed: {exc}")
        sys.exit(1)
    finally:
        node.close()
        if db_writer:
            db_writer.close()


if __name__ == "__main__":
    main()
