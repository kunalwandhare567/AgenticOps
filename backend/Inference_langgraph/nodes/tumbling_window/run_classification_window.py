"""
app_simulator/offline/run_classification_window.py
===================================================
Offline Classification and Tumbling Window batch runner.

Usage:
    python app_simulator/offline/run_classification_window.py

Steps:
    1. Deletes existing pipeline_results.csv and tumbling_window_output.csv.
    2. Reads engineered_features.csv.
    3. Loads LightGBM model.
    4. Evaluates predictions and vote smoothing.
    5. Appends rows to tumbling_window_output.csv and pipeline_results.csv.
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
    LGBM_MODEL_PKL,
    PIPELINE_OUTPUT_DIR,
    PIPELINE_RESULTS_CSV,
    PRELIM_SEVERITY_CSV,
    TUMBLING_WINDOW_CSV,
    TUMBLING_WINDOW_OUTPUT_DIR,
)
from nodes.classification.classifier import load_classifier, classify
from Simulator.app_data_generator_for_offline.state import PipelineState
from nodes.tumbling_window.tumbling_window import TumblingWindow
from Simulator.app_data_generator_for_offline.storage.db_writer import DbWriter


def _get_next_version(out_dir: Path) -> int:
    """Find next version N by scanning out_dir for tumbling_window_output_N.csv files."""
    pattern = re.compile(r"^tumbling_window_output_(\d+)\.csv$")
    max_v = 0
    if out_dir.exists():
        for f in out_dir.iterdir():
            m = pattern.match(f.name)
            if m:
                max_v = max(max_v, int(m.group(1)))
    return max_v + 1


def main() -> None:
    print("=" * 65)
    print("  Offline Classification & Tumbling Window Batch Runner")
    print("=" * 65)

    TUMBLING_WINDOW_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    version = _get_next_version(TUMBLING_WINDOW_OUTPUT_DIR)
    versioned_csv = TUMBLING_WINDOW_OUTPUT_DIR / f"tumbling_window_output_{version}.csv"

    # Clean up old default output files for fresh run
    for p in [PIPELINE_RESULTS_CSV, TUMBLING_WINDOW_CSV]:
        if p.exists():
            try:
                p.unlink()
            except Exception:
                pass

    # Verify input exists
    if not ENGINEERED_FEAT_CSV.exists():
        print(f"[ERROR] Engineered features not found at {ENGINEERED_FEAT_CSV}.")
        print("        Run Node 2 (orchestrator.py) first!")
        sys.exit(1)

    # Load classifier
    print("[Pipeline] Loading LightGBM model...")
    model_loaded = load_classifier()
    if not model_loaded:
        print("[ERROR] Model could not be loaded. Please run offline training first:")
        print("        python nodes/classification/train_classifier.py --tune --gpu")
        sys.exit(1)

    # Clean up and setup SQLite DbWriter
    db_writer = None
    if DB_PATH.exists():
        print(f"[Pipeline] Initialising DbWriter for SQLite database: {DB_PATH.name}...")
        try:
            db_writer = DbWriter(DB_PATH)
            db_writer.setup()
        except Exception as exc:
            print(f"[Pipeline] Warning initialising database: {exc}")

    # Open pipeline_results.csv
    PIPELINE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fh = PIPELINE_RESULTS_CSV.open("w", newline="", encoding="utf-8")
    RESULT_FIELDS = [
        "cycle", "episode_id", "failure_mode", "timestamp", "elapsed_s",
        "preliminary_severity", "severity_raw", "severity_weighted_score",
        "severity_critical_count", "severity_warning_count", "severity_blast_size",
        "severity_reason",
        "predicted_failure", "prediction_probability",
        "dominant_state", "vote_distribution", "window_margin", "window_full", "window_size"
    ]
    results_writer = csv.DictWriter(fh, fieldnames=RESULT_FIELDS, extrasaction="ignore")
    results_writer.writeheader()

    window = TumblingWindow()

    try:
        # Read engineered features
        print(f"[Pipeline] Reading features from: {ENGINEERED_FEAT_CSV}")
        with ENGINEERED_FEAT_CSV.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = [r for r in reader if r.get("episode_id") and r.get("episode_id") != "episode_id"]

        total_rows = len(rows)
        print(f"[Pipeline] Processing {total_rows:,} rows across all episodes...")
        processed_count = 0

        for r in rows:
            processed_count += 1
            # 1. State setup
            state = PipelineState(
                episode_id   = r.get("episode_id", ""),
                failure_mode = r.get("failure_mode", "NONE"),
                timestamp    = float(r.get("timestamp", 0.0)),
                elapsed_s    = float(r.get("elapsed_s", 0.0)),
                step         = processed_count,
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

            # 2. Classification
            state = classify(state)

            # 3. Tumbling Window smoothing
            state = window.update(state, cycle=processed_count)

            # Write to SQLite DB table: node_tumbling_window
            if db_writer:
                db_writer.write_tumbling_window({
                    "cycle":             processed_count,
                    "episode_id":        state.episode_id,
                    "failure_mode":      state.failure_mode,
                    "timestamp":         state.timestamp,
                    "elapsed_s":         state.elapsed_s,
                    "dominant_state":    state.summarized_failure,
                    "vote_distribution": state.vote_distribution,
                    "window_margin":      state.window_margin,
                    "window_full":       state.window_full,
                    "window_size":       len(state.window_predictions),
                })

            # 4. Write results
            results_writer.writerow({
                "cycle":                    processed_count,
                "episode_id":               state.episode_id,
                "failure_mode":             state.failure_mode,
                "timestamp":                state.timestamp,
                "elapsed_s":                state.elapsed_s,
                "preliminary_severity":     getattr(state, "preliminary_severity", "P4"),
                "predicted_failure":        state.predicted_failure,
                "prediction_probability":   state.prediction_probability,
                "dominant_state":           state.summarized_failure,
                "vote_distribution":        str(state.vote_distribution),
                "window_margin":            state.window_margin,
                "window_full":              state.window_full,
                "window_size":              len(state.window_predictions),
            })

            if processed_count % 20000 == 0 or processed_count == total_rows:
                print(f"  Processed {processed_count:,} / {total_rows:,} rows...")

        # Close window handle to flush file
        window.close()

        # Copy tumbling_window_output.csv -> tumbling_window_output_N.csv
        if TUMBLING_WINDOW_CSV.exists():
            shutil.copy(str(TUMBLING_WINDOW_CSV), str(versioned_csv))

        print(f"\n[Pipeline] SUCCESS! Processed all {total_rows:,} rows.")
        print(f"  [CSV Output Version {version}] : {versioned_csv}")
        print(f"  [CSV Output Latest]    : {TUMBLING_WINDOW_CSV}")
        print(f"  [SQLite Table]         : node_tumbling_window in {DB_PATH.name}")

    except Exception as exc:
        print(f"[ERROR] Pipeline execution failed: {exc}")
        sys.exit(1)
    finally:
        fh.close()
        if db_writer:
            db_writer.close()


if __name__ == "__main__":
    main()

