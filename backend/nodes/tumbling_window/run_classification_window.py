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
import sys
from pathlib import Path

# ── resolve package root ─────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent))

from app_data_generator.config import (
    ENGINEERED_FEAT_CSV,
    LGBM_MODEL_PKL,
    PIPELINE_OUTPUT_DIR,
    PIPELINE_RESULTS_CSV,
    PRELIM_SEVERITY_CSV,
    TUMBLING_WINDOW_CSV,
)
from nodes.classification.classifier import load_classifier, classify
from app_data_generator.state import PipelineState
from nodes.tumbling_window.tumbling_window import TumblingWindow


def main() -> None:
    print("=" * 65)
    print("  Offline Classification & Tumbling Window Batch Runner")
    print("=" * 65)

    # 1. Clean up old output files
    for p in [PIPELINE_RESULTS_CSV, TUMBLING_WINDOW_CSV]:
        if p.exists():
            print(f"[Pipeline] Removing existing output: {p}")
            p.unlink()

    # Verify input exists
    if not ENGINEERED_FEAT_CSV.exists():
        print(f"[ERROR] Engineered features not found at {ENGINEERED_FEAT_CSV}.")
        print("        Run Node 2 (run_feature_engineering.py) first!")
        sys.exit(1)

    # Load classifier
    print("[Pipeline] Loading LightGBM model...")
    model_loaded = load_classifier()
    if not model_loaded:
        print("[ERROR] Model could not be loaded. Please run offline training first:")
        print("        python app_simulator/offline/train_classifier.py")
        sys.exit(1)

    # Load preliminary severity results if they exist (to combine into pipeline_results)
    severity_map = {}
    if PRELIM_SEVERITY_CSV.exists():
        print(f"[Pipeline] Loading severity levels from {PRELIM_SEVERITY_CSV}...")
        try:
            with PRELIM_SEVERITY_CSV.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    # Key by episode_id and timestamp
                    k = (r["episode_id"], r["timestamp"])
                    severity_map[k] = {
                        "Severity":             r.get("Severity", "P4"),
                        "RawSeverity":          r.get("RawSeverity", ""),
                        "WeightedScore":        r.get("WeightedScore", ""),
                        "CriticalCount":        r.get("CriticalCount", ""),
                        "WarningCount":         r.get("WarningCount", ""),
                        "BlastSize":            r.get("BlastSize", ""),
                        "Reason":               r.get("Reason", ""),
                    }
            print(f"  Loaded {len(severity_map):,} severity rows.")
        except Exception as exc:
            print(f"[Pipeline] Warning loading severity CSV: {exc}")
    else:
        print("[Pipeline] Info: preliminary_severity.csv not found. Severity fields will be empty.")

    # Open pipeline_results.csv
    PIPELINE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not PIPELINE_RESULTS_CSV.exists()
    fh = PIPELINE_RESULTS_CSV.open("a", newline="", encoding="utf-8")
    RESULT_FIELDS = [
        "cycle", "episode_id", "failure_mode", "timestamp", "elapsed_s",
        "preliminary_severity", "severity_raw", "severity_weighted_score",
        "severity_critical_count", "severity_warning_count", "severity_blast_size",
        "severity_reason",
        "predicted_failure", "prediction_probability",
        "dominant_state", "vote_distribution", "window_margin", "window_full", "window_size"
    ]
    results_writer = csv.DictWriter(fh, fieldnames=RESULT_FIELDS, extrasaction="ignore")
    if write_header:
        results_writer.writeheader()

    window = TumblingWindow()

    try:
        # Read engineered features
        print(f"[Pipeline] Reading features from: {ENGINEERED_FEAT_CSV}")
        with ENGINEERED_FEAT_CSV.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        total_rows = len(rows)
        print(f"[Pipeline] Processing {total_rows:,} rows...")
        processed_count = 0

        for r in rows:
            processed_count += 1
            # 1. State setup
            state = PipelineState(
                episode_id   = r["episode_id"],
                failure_mode = r["failure_mode"],
                timestamp    = float(r["timestamp"]),
                elapsed_s    = float(r["elapsed_s"]),
                step         = processed_count,
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

            # Match offline severity record if available
            sev_record = severity_map.get((state.episode_id, str(state.timestamp)))
            if sev_record:
                state.preliminary_severity = sev_record["Severity"]
            else:
                state.preliminary_severity = "P4"

            # 2. Classification
            state = classify(state)

            # 3. Tumbling Window smoothing
            state = window.update(state, cycle=processed_count)

            # 4. Write results
            results_writer.writerow({
                "cycle":                    processed_count,
                "episode_id":               state.episode_id,
                "failure_mode":             state.failure_mode,
                "timestamp":                state.timestamp,
                "elapsed_s":                state.elapsed_s,
                # Severity
                "preliminary_severity":     state.preliminary_severity,
                "severity_raw":             sev_record["RawSeverity"] if sev_record else "",
                "severity_weighted_score":  sev_record["WeightedScore"] if sev_record else "",
                "severity_critical_count":  sev_record["CriticalCount"] if sev_record else "",
                "severity_warning_count":   sev_record["WarningCount"] if sev_record else "",
                "severity_blast_size":      sev_record["BlastSize"] if sev_record else "",
                "severity_reason":          sev_record["Reason"] if sev_record else "",
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

            if processed_count % 1000 == 0 or processed_count == total_rows:
                print(f"  Processed {processed_count:,} / {total_rows:,} rows...")

        print(f"\n[Pipeline] Success!")
        print(f"  Results saved: {PIPELINE_RESULTS_CSV}")
        print(f"  Window saved : {TUMBLING_WINDOW_CSV}")

    except Exception as exc:
        print(f"[ERROR] Pipeline execution failed: {exc}")
        sys.exit(1)
    finally:
        fh.close()
        window.close()


if __name__ == "__main__":
    main()
