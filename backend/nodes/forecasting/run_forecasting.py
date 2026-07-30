"""
app_simulator/offline/run_forecasting.py
=========================================
Offline batch runner — Forecasting Node (Stage 6).

DESIGN: For each episode, ALL feature rows are fed into the buffer
to build up the full history, then ONE forecast is run using the
final accumulated history. This is the correct offline approach:
- Fast: one ARIMA fit per episode (not per cycle)
- Accurate: uses the maximum available history for best confidence
- Realistic: mirrors what the live pipeline would see at episode end

Reads:
    pipeline/output/engineered_features.csv
    pipeline/output/tumbling_window_output.csv

Writes:
    pipeline/output/forecasting_output.csv

Usage:
    # Full mode (pmdarima Auto-ARIMA — accurate, ~2 min/episode):
    python app_simulator/offline/run_forecasting.py

    # Fast mode (linear regression only — for quick visualization, seconds total):
    python app_simulator/offline/run_forecasting.py --fast
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

# Suppress statsmodels convergence warnings in batch mode
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from tqdm import tqdm

from app_data_generator.config import (
    ENGINEERED_FEAT_CSV,
    FORECASTING_OUTPUT_CSV,
    TUMBLING_WINDOW_CSV,
    DB_PATH,
)
from app_data_generator.storage.db_writer import DbWriter
from nodes.forecasting.buffer import reset_all, clear_episode, append_feature_row
from nodes.forecasting.router import route_forecast

_OUTPUT_COLS = [
    "episode_id", "failure_mode", "elapsed_s",
    "algorithm_used", "history_steps",
    "time_to_failure", "earliest_ttf_feature",
    "forecast_confidence", "confidence_reason",
    "threshold_crossed",
    "feature_ttfs_json", "feature_slopes_json",
    "current_values_json", "predictions_json",
    "critical_thresholds_json",
]


def _safe_json(obj) -> str:
    try:
        return json.dumps(obj, default=str)
    except Exception:
        return "{}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fast", action="store_true",
        help="Use linear regression only (no ARIMA). ~10 seconds for 650 episodes."
    )
    args = parser.parse_args()

    if args.fast:
        # Disable pmdarima and statsmodels so _linear_fallback() runs everywhere.
        # This makes the batch run in seconds — ideal for generating visualization data.
        import nodes.forecasting.algorithms as _alg
        _alg._pmdarima_auto_arima = None
        _alg._statsmodels_ARIMA   = None
        print("[run_forecasting] FAST mode: using linear regression only (no ARIMA).")
    else:
        print("[run_forecasting] FULL mode: using pmdarima Auto-ARIMA (slower, more accurate).")

    # ── Load inputs ───────────────────────────────────────────────────────────
    print("[run_forecasting] Loading CSVs ...")
    if not ENGINEERED_FEAT_CSV.exists():
        print(f"  ERROR: {ENGINEERED_FEAT_CSV} not found.")
        sys.exit(1)
    if not TUMBLING_WINDOW_CSV.exists():
        print(f"  ERROR: {TUMBLING_WINDOW_CSV} not found.")
        sys.exit(1)

    feat_df   = pd.read_csv(ENGINEERED_FEAT_CSV)
    window_df = pd.read_csv(TUMBLING_WINDOW_CSV)
    feat_df.columns   = feat_df.columns.str.strip()
    window_df.columns = window_df.columns.str.strip()

    print(f"  {len(feat_df):,} feature rows | {len(window_df):,} window rows")

    # Find dominant_state column
    dominant_col = next(
        (c for c in ["dominant_state", "predicted_label", "dominant_failure_mode",
                      "hmm_state", "final_state"]
         if c in window_df.columns),
        None
    )
    if dominant_col is None:
        print(f"  ERROR: no dominant_state column. Available: {list(window_df.columns)}")
        sys.exit(1)
    print(f"  Dominant state column: '{dominant_col}'")

    reset_all()
    FORECASTING_OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    # ── Initialise DbWriter for node_forecasting table ─────────────────────
    pipeline_db = DbWriter(DB_PATH)
    pipeline_db.setup()

    episodes   = feat_df["episode_id"].unique()
    output_rows: list[dict] = []

    print(f"\n[run_forecasting] Running forecast for {len(episodes)} episodes ...\n")

    for ep_id in tqdm(episodes, desc="Forecasting"):
        ep_feat = feat_df[feat_df["episode_id"] == ep_id].copy()
        if "elapsed_s" in ep_feat.columns:
            ep_feat = ep_feat.sort_values("elapsed_s")

        # Determine failure mode
        if "failure_mode" in ep_feat.columns:
            failure_mode = ep_feat["failure_mode"].mode().iloc[0]
        else:
            ep_windows = window_df[window_df["episode_id"] == ep_id]
            failure_mode = ep_windows[dominant_col].mode().iloc[0] if not ep_windows.empty else "NONE"

        if failure_mode == "NONE":
            clear_episode(ep_id)
            continue

        # ── Feed ALL rows into buffer to build full history ───────────────────
        for row in ep_feat.to_dict("records"):
            append_feature_row(ep_id, row)

        # ── Get final dominant state from last window ─────────────────────────
        ep_windows = window_df[window_df["episode_id"] == ep_id]
        if not ep_windows.empty:
            if "elapsed_s" in ep_windows.columns:
                ep_windows = ep_windows.sort_values("elapsed_s")
            dominant_state = ep_windows.iloc[-1][dominant_col]
        else:
            dominant_state = failure_mode

        if dominant_state == "NONE":
            clear_episode(ep_id)
            continue

        # ── Run ONE forecast using full accumulated history ────────────────────
        last_row = ep_feat.iloc[-1].to_dict()
        elapsed  = last_row.get("elapsed_s", len(ep_feat) * 2)

        result = route_forecast(
            failure_mode     = dominant_state,
            episode_id       = ep_id,
            current_features = last_row,
        )

        clear_episode(ep_id)

        if not result:
            continue

        fc = result.get("forecast", {})
        row_out = {
            "episode_id":               ep_id,
            "failure_mode":             dominant_state,
            "elapsed_s":                elapsed,
            "algorithm_used":           fc.get("algorithm_used", ""),
            "history_steps":            fc.get("history_steps", 0),
            "time_to_failure":          result.get("time_to_failure"),
            "earliest_ttf_feature":     fc.get("earliest_ttf_feature"),
            "forecast_confidence":      result.get("forecast_confidence", 0.0),
            "confidence_reason":        result.get("confidence_reason", ""),
            "threshold_crossed":        result.get("threshold_crossed", False),
            "feature_ttfs_json":        _safe_json(fc.get("feature_ttfs")),
            "feature_slopes_json":      _safe_json(fc.get("feature_slopes")),
            "current_values_json":      _safe_json(fc.get("current_values")),
            "predictions_json":         _safe_json(fc.get("predictions")),
            "critical_thresholds_json": _safe_json(fc.get("critical_thresholds")),
        }
        output_rows.append(row_out)

        # ── DB: Write to node_forecasting table ───────────────────────────────
        try:
            pipeline_db.write_forecasting({
                "cycle":               None,          # offline batch (no cycle counter)
                "episode_id":          ep_id,
                "failure_mode":        dominant_state,
                "timestamp":           None,
                "elapsed_s":           elapsed,
                "algorithm_used":      fc.get("algorithm_used", ""),
                "history_steps":       fc.get("history_steps", 0),
                "forecast_horizon_s":  fc.get("forecast_horizon_s"),
                "time_to_failure":     result.get("time_to_failure"),
                "earliest_ttf_feature": fc.get("earliest_ttf_feature"),
                "forecast_confidence": result.get("forecast_confidence", 0.0),
                "confidence_reason":   result.get("confidence_reason", ""),
                "threshold_crossed":   result.get("threshold_crossed", False),
                "feature_ttfs":        fc.get("feature_ttfs"),
                "feature_slopes":      fc.get("feature_slopes"),
                "predictions":         fc.get("predictions"),
                "current_values":      fc.get("current_values"),
            })
        except Exception as _e:
            print(f"[run_forecasting] WARN: DB write failed: {_e}")

    # ── Write output ──────────────────────────────────────────────────────────
    pipeline_db.close()
    if not output_rows:
        print("\n[run_forecasting] No forecast rows generated.")
        return

    out_df = pd.DataFrame(output_rows)
    defined = [c for c in _OUTPUT_COLS if c in out_df.columns]
    extra   = [c for c in out_df.columns if c not in defined]
    out_df  = out_df[defined + extra]
    out_df.to_csv(FORECASTING_OUTPUT_CSV, index=False)

    print(f"\n[run_forecasting] Done: {len(out_df):,} rows -> {FORECASTING_OUTPUT_CSV}")
    print("\n── Summary ──────────────────────────────────────────────────────────")
    print(f"  Episodes processed : {len(episodes):,}")
    print(f"  Forecast rows      : {len(out_df):,}")
    if "forecast_confidence" in out_df.columns:
        print(f"  Mean confidence    : {out_df['forecast_confidence'].mean():.4f}")
        print(f"  Min  confidence    : {out_df['forecast_confidence'].min():.4f}")
        print(f"  Max  confidence    : {out_df['forecast_confidence'].max():.4f}")
    if "algorithm_used" in out_df.columns:
        print("\n-- Summary ----------------------------------------------------------")
        for algo, cnt in out_df["algorithm_used"].value_counts().items():
            print(f"    {str(algo):45s} {cnt:>5,}")
    if "threshold_crossed" in out_df.columns:
        crossed = out_df["threshold_crossed"].sum()
        print(f"\n  Episodes with TTF prediction: {crossed:,} / {len(out_df):,}")
    if "time_to_failure" in out_df.columns:
        ttf_valid = out_df["time_to_failure"].dropna()
        if len(ttf_valid):
            print(f"  Median TTF (seconds)        : {ttf_valid.median():.1f}")
    print("─────────────────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
