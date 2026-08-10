"""
backend/nodes/reliability/run_weibull_fitter.py
================================================
CLI runner for the Reliability Node 4-Group Weibull Fitter & Plotter (Stage 8b).

Outputs:
  - Formatted console table matching mentor's exact requirements.
  - nodes/reliability/output/weibull_km_groups.png
  - nodes/reliability/weibull_params.json  ← NEW: sidecar for live pipeline
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

# Path bootstrap
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from nodes.reliability.weibull_fitter import fit_all_groups
from nodes.reliability.plot_weibull_km import plot_weibull_km_overlay

import math
from scipy.special import gamma

from Simulator.app_data_generator_for_offline.config import DB_PATH, PIPELINE_OUTPUT_DIR
from Simulator.app_data_generator_for_offline.storage.db_writer import DbWriter

_LIFE_DATA_CSV       = _HERE / "output" / "life_data_extracted.csv"
_WEIBULL_PARAMS_JSON  = _HERE / "weibull_params.json"
_RELIABILITY_OUT_DIR = _HERE / "output"
_RELIABILITY_CSV     = _RELIABILITY_OUT_DIR / "reliability_output.csv"


def _get_next_version(out_dir: Path) -> int:
    """Find next version N by scanning out_dir for reliability_output_N.csv files."""
    import re
    pattern = re.compile(r"^reliability_output_(\d+)\.csv$")
    max_v = 0
    if out_dir.exists():
        for f in out_dir.iterdir():
            m = pattern.match(f.name)
            if m:
                max_v = max(max_v, int(m.group(1)))
    return max_v + 1


def _write_params_sidecar(results_df: pd.DataFrame) -> Path:
    """
    Write fitted Weibull parameters + MTTF to weibull_params.json.
    """
    groups: dict = {}
    for _, row in results_df.iterrows():
        group_name = str(row["group"])
        beta = float(row["beta"])
        eta  = float(row["eta"])
        mttf = eta * gamma(1.0 + (1.0 / beta)) if beta > 0 and eta > 0 else eta
        groups[group_name] = {
            "beta":   round(beta, 4),
            "eta":    round(eta,  4),
            "mttf_s": round(mttf, 2),
            "n":      int(row.get("n",      0)),
            "events": int(row.get("events", 0)),
        }

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "groups":        groups,
    }

    _WEIBULL_PARAMS_JSON.write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(f"[Weibull Runner] Params sidecar -> {_WEIBULL_PARAMS_JSON}")
    return _WEIBULL_PARAMS_JSON


def run_weibull_analysis(csv_path: Path | None = None, plot: bool = True) -> tuple[pd.DataFrame, Path | None]:
    """Run full 4-group Weibull fitting, compute MTTF, populate DB + CSV, and generate plot."""
    path = Path(csv_path or _LIFE_DATA_CSV)
    if not path.exists():
        raise FileNotFoundError(
            f"[Weibull Runner] life_data_extracted.csv not found at:\n  {path}\n"
            "Run extractor first: python backend/nodes/reliability/run_extractor.py"
        )

    print(f"\n[Weibull Runner] Loading life data from {path} ...")
    df = pd.read_csv(path)
    print(f"[Weibull Runner] Loaded {len(df):,} total records.")

    # 1. Fit Weibull MLE for all 4 groups
    results_df = fit_all_groups(df)

    # Compute MTTF per group using Weibull formula: MTTF = eta * Gamma(1 + 1/beta)
    mttf_list = []
    for _, r in results_df.iterrows():
        beta = float(r["beta"])
        eta  = float(r["eta"])
        val  = eta * gamma(1.0 + (1.0 / beta)) if beta > 0 and eta > 0 else eta
        mttf_list.append(round(val, 2))
    results_df["mttf_seconds"] = mttf_list

    # Print exact table matching mentor's requested output format
    print("\n" + "=" * 85)
    print("  4-GROUP CENSORED 2-PARAMETER WEIBULL FIT RESULTS (MLE + MTTF)")
    print("=" * 85)
    cols = ["group", "n", "events", "censored", "beta", "eta", "mttf_seconds", "log_likelihood"]
    print(results_df[cols].to_string(index=False))
    print("=" * 85 + "\n")

    # 2. Write params sidecar JSON (read by live pipeline)
    _write_params_sidecar(results_df)

    # 3. Persist episode reliability records into SQLite DB + CSV
    param_map = {
        row["group"]: (float(row["beta"]), float(row["eta"]), float(row["mttf_seconds"]))
        for _, row in results_df.iterrows()
    }

    db = DbWriter(DB_PATH)
    db.setup()

    rel_rows = []
    for idx, row in df.iterrows():
        gname = str(row.get("group", "Immediate trigger"))
        beta, eta, mttf = param_map.get(gname, (1.0, 100.0, 100.0))
        ttf = float(row.get("ttf_seconds", 0.0))
        
        # Reliability function R(t) = exp(-(t/eta)^beta) * 100%
        r_t = math.exp(-((max(ttf, 0.0) / max(eta, 1e-6)) ** beta)) * 100.0
        # Hazard rate h(t) = (beta / eta) * (t / eta)^(beta - 1)
        h_t = (beta / eta) * ((max(ttf, 0.01) / eta) ** (beta - 1.0)) if eta > 0 else 0.0

        rec = {
            "cycle":                idx + 1,
            "episode_id":           str(row.get("episode_id", f"ep_{idx}")),
            "failure_mode":         str(row.get("failure_mode", "NONE")),
            "group_name":           gname,
            "ttf_seconds":          round(ttf, 2),
            "mttf_seconds":         mttf,
            "survival_probability": round(r_t, 4),
            "hazard_rate":          round(h_t, 6),
            "event":                int(row.get("event", 1)),
            "data_source":          str(row.get("data_source", "extracted")),
            "beta":                 beta,
            "eta":                  eta,
            "recorded_at":          datetime.now(timezone.utc).isoformat(),
        }
        rel_rows.append(rec)
        db.write_reliability(rec)

    db.close()

    # Save versioned and master CSV
    _RELIABILITY_OUT_DIR.mkdir(parents=True, exist_ok=True)
    v_num = _get_next_version(_RELIABILITY_OUT_DIR)
    v_csv = _RELIABILITY_OUT_DIR / f"reliability_output_{v_num}.csv"

    out_df = pd.DataFrame(rel_rows)
    out_df.to_csv(v_csv, index=False)
    out_df.to_csv(_RELIABILITY_CSV, index=False)

    print(f"[Weibull Runner] Saved {len(out_df):,} episode reliability records:")
    print(f"  [CSV Output Version {v_num}] : {v_csv}")
    print(f"  [CSV Output Latest]    : {_RELIABILITY_CSV}")
    print(f"  [SQLite Table]         : node_reliability in {DB_PATH.name}")

    # 4. Generate KM + Weibull plot
    plot_path = None
    if plot:
        plot_path = plot_weibull_km_overlay(df)

    return results_df, plot_path


def main() -> None:
    parser = argparse.ArgumentParser(description="AIOps Reliability Node — 4-Group Weibull Fitter")
    parser.add_argument("--csv", type=str, default=str(_LIFE_DATA_CSV), help="Path to life_data_extracted.csv")
    parser.add_argument("--no-plot", action="store_true", help="Disable PNG plot generation")
    args = parser.parse_args()

    t0 = time.perf_counter()
    run_weibull_analysis(csv_path=Path(args.csv), plot=not args.no_plot)
    print(f"[Weibull Runner] Finished in {time.perf_counter() - t0:.2f}s\n")


if __name__ == "__main__":
    main()

