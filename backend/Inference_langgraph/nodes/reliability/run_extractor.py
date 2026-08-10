"""
backend/nodes/reliability/run_extractor.py
==========================================
Standalone CLI runner for the Reliability Node Life-Data Extractor.

Pipeline Position
-----------------
  Stage 8 — Runs AFTER severity_update_output.csv has been produced.

  Previous step : python backend/nodes/severity_update/run_severity_update.py
  This step     : python backend/nodes/reliability/run_extractor.py
  Next step     : python backend/nodes/reliability/run_weibull_fitter.py  (Stage 8b)

What this runner does
---------------------
  1. Reads pipeline_results.csv        (78,000 cycle-level rows)
  2. Reads severity_update_output.csv  (600 episode-level rows)
  3. Reads human_gate_output.csv       (for MTTR data, if available)
  4. Extracts TTF + censoring flag per episode
  5. Writes life_data_extracted.csv    (650 rows — one per unique episode)

Usage
-----
  From project root (d:/AIOps_Incident_Management):
      python backend/nodes/reliability/run_extractor.py

Output
------
  backend/nodes/reliability/output/life_data_extracted.csv
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent   # → backend/
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from nodes.reliability.extractor import LifeDataExtractor


# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------

def _print_banner() -> None:
    print()
    print("=" * 65)
    print("  AIOps Reliability Node — Life-Data Extractor")
    print("  Stage 8a: TTF Extraction + Right-Censoring")
    print("=" * 65)
    print("  Inputs:")
    print("    nodes/classification/output/pipeline_results.csv")
    print("    nodes/severity_update/output/severity_update_output.csv")
    print("    nodes/human_gate/output/human_gate_output.csv  (MTTR, optional)")
    print("  Output:")
    print("    nodes/reliability/output/life_data_extracted.csv")
    print("=" * 65)
    print()


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def main() -> None:
    _print_banner()

    t0 = time.perf_counter()

    # Run extractor
    extractor = LifeDataExtractor()
    df = extractor.run()

    # Save CSV
    output_path = extractor.save()

    elapsed = time.perf_counter() - t0

    # Print summary report
    print(extractor.summary())

    # Show first few rows of the output CSV
    print("  Sample rows from life_data_extracted.csv:")
    print("-" * 65)
    print(df[["episode_id", "failure_mode", "ttf_seconds", "event", "data_source"]].head(15).to_string(index=False))
    print("-" * 65)

    # Show event=0 (right-censored) sample
    censored_sample = df[df["event"] == 0].head(5)
    print("\n  Sample RIGHT-CENSORED rows (event=0):")
    print(censored_sample[["episode_id", "failure_mode", "ttf_seconds", "event", "data_source"]].to_string(index=False))

    # Show event=1 (complete failure) sample
    failed_sample = df[df["event"] == 1].head(5)
    print("\n  Sample COMPLETE FAILURE rows (event=1):")
    print(failed_sample[["episode_id", "failure_mode", "ttf_seconds", "event", "data_source"]].to_string(index=False))

    print()
    print("=" * 65)
    print(f"  Completed in {elapsed:.2f}s")
    print(f"  Output written: {output_path}")
    print("=" * 65)
    print()
    print("  Next step: Run the Weibull fitter:")
    print("    python backend/nodes/reliability/run_weibull_fitter.py")
    print()


if __name__ == "__main__":
    main()
