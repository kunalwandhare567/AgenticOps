"""
app_simulator/offline/run_feature_engineering.py
===================================================
Offline Feature Engineering batch processor.

Usage:
    python app_simulator/offline/run_feature_engineering.py

Steps:
    1. Deletes any existing pipeline/output/engineered_features.csv.
    2. Opens simulator_db.sqlite.
    3. Reads all ticks from metrics and logs tables.
    4. Runs feature engineering per cycle.
    5. Saves features to pipeline/output/engineered_features.csv.
"""
from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path

# ── resolve package root ─────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent))

from Simulator.app_data_generator_for_offline.config import (
    DB_PATH,
    DRAIN_INI,
    DRAIN_STATE,
    KNOWN_TEMPLATES_JSON,
    ENGINEERED_FEAT_CSV,
)
from nodes.feature_engineering.orchestrator import run_feature_engineering_from_raw
from nodes.feature_engineering.log_features import (
    load_template_miner,
    load_known_template_ids,
)


def main() -> None:
    print("=" * 65)
    print("  Offline Feature Engineering Batch Runner")
    print("=" * 65)

    # 1. Clean up old feature CSV
    if ENGINEERED_FEAT_CSV.exists():
        print(f"[FE] Removing existing feature CSV: {ENGINEERED_FEAT_CSV}")
        ENGINEERED_FEAT_CSV.unlink()

    # Check database exists
    if not DB_PATH.exists():
        print(f"[ERROR] Database not found at {DB_PATH}. Run simulator first!")
        sys.exit(1)

    print("[FE] Loading Drain3 state miner and known templates...")
    template_miner = load_template_miner(str(DRAIN_STATE), str(DRAIN_INI))
    known_ids      = load_known_template_ids(str(KNOWN_TEMPLATES_JSON))

    # Connect to SQLite
    print(f"[FE] Connecting to database: {DB_PATH}")
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    try:
        # Fetch all metrics
        cur = conn.execute("SELECT * FROM metrics ORDER BY id ASC")
        rows = cur.fetchall()
        total_ticks = len(rows)
        print(f"[FE] Found {total_ticks:,} telemetry cycles in database.")

        if total_ticks == 0:
            print("[ERROR] Metrics table is empty.")
            sys.exit(1)

        print("[FE] Processing ticks...")
        processed_count = 0

        for r in rows:
            metric = dict(r)
            ep_id  = metric["episode_id"]
            ts     = metric["timestamp"]
            el_s   = metric["elapsed_s"]
            mode   = metric["failure_mode"]

            # Query logs for this specific cycle
            log_cur = conn.execute(
                "SELECT log_level, exception_type, log_message FROM logs "
                "WHERE episode_id = ? AND timestamp = ?",
                (ep_id, ts)
            )
            logs = [{"log_level": row[0], "exception_type": row[1], "log_message": row[2]} for row in log_cur.fetchall()]

            # Run feature engineering
            run_feature_engineering_from_raw(
                metric             = metric,
                log                = logs,
                episode_id         = ep_id,
                failure_mode       = mode,
                timestamp          = ts,
                elapsed_s          = el_s,
                template_miner     = template_miner,
                known_template_ids = known_ids,
            )

            processed_count += 1
            if processed_count % 1000 == 0 or processed_count == total_ticks:
                print(f"  Processed {processed_count:,} / {total_ticks:,} cycles...")

        print(f"\n[FE] Success! Features written to: {ENGINEERED_FEAT_CSV}")

    except Exception as exc:
        print(f"[ERROR] FE execution failed: {exc}")
        sys.exit(1)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
