"""
backend/nodes/reliability/extractor.py
=======================================
Life-Data Extractor for the Reliability Node (Stage 8).

What this module does
----------------------
For each of the 650 unique episodes in the AIOps pipeline, this extractor:

  1. Computes Time-To-Failure (TTF):
       TTF = elapsed_s at the FIRST cycle where preliminary_severity == 'P1'
       (i.e. the exact second when the Severity Node first raised a P1 alarm)

  2. Applies Right-Censoring (mentor's recommendation — Section 2 of reference doc):
       event = 1   ->  Complete failure (P1 was detected during the episode window)
       event = 0   ->  Right-censored  (episode ran to 238s with no P1 detection)

  3. Extracts MTTR data from human_gate_output.csv (when available):
       ttr_seconds = decided_at - created_at  (operator resolution time)
       Used by metrics.py to compute Availability = MTTF / (MTTF + MTTR)

Output
------
  nodes/reliability/output/life_data_extracted.csv

  Columns:
    episode_id        str    Unique episode identifier
    failure_mode      str    Failure mode label (or NONE for healthy runs)
    ttf_seconds       float  Time-To-Failure or censored duration (in seconds)
    event             int    1 = complete failure, 0 = right-censored
    preliminary_severity str   Raw severity at the time of first P1 detection (from severity_update)
    revised_severity  str    Final severity from Severity Update node
    total_window_s    float  Total episode observation window (238.0s)
    data_source       str    'pipeline_results_P1' or 'right_censored'
    ttr_seconds       float  Time-To-Repair from Human Gate (NaN if not available)

Usage
-----
    from nodes.reliability.extractor import LifeDataExtractor, extract_life_data

    # Option 1: Functional (quick)
    df = extract_life_data()

    # Option 2: Class-based (more control)
    extractor = LifeDataExtractor()
    df = extractor.run()
    print(extractor.summary())
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd

# ---------------------------------------------------------------------------
# Path bootstrap — allows running from any working directory
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent  # backend/
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from Simulator.app_data_generator_for_offline.config import (
    PIPELINE_OUTPUT_DIR,
    HUMAN_GATE_OUTPUT_CSV,
    SEVERITY_UPDATE_CSV,
)

# ---------------------------------------------------------------------------
# Paths to input CSV files
# ---------------------------------------------------------------------------
_PIPELINE_RESULTS_CSV = (
    _PROJECT_ROOT / "nodes" / "classification" / "output" / "pipeline_results.csv"
)
_PRELIM_SEVERITY_CSV = (
    _PROJECT_ROOT / "nodes" / "preliminary_severity" / "output" / "preliminary_severity.csv"
)
_OUTPUT_DIR = _HERE / "output"
_OUTPUT_CSV = _OUTPUT_DIR / "life_data_extracted.csv"

# Total observation window per episode (seconds) — all episodes run 238s
_EPISODE_WINDOW_S = 238.0

# ---------------------------------------------------------------------------
# 4-Group Reliability Classification (Mentor's Stratification Strategy)
# ---------------------------------------------------------------------------
FAILURE_GROUP_MAP: dict[str, str] = {
    "BAD_DEPLOY":          "Immediate trigger",
    "CACHE_STAMPEDE":      "Immediate trigger",
    "CASCADING_FAILURE":   "Immediate trigger",
    "CPU_SATURATION":      "Immediate trigger",
    "DEPENDENCY_TIMEOUT":  "Immediate trigger",
    "ERROR_STORM":         "Immediate trigger",
    "QUEUE_BACKUP":        "Fast accumulation",
    "RETRY_STORM":         "Fast accumulation",
    "DISK_IO_SATURATION":  "Progressive resource degradation",
    "DB_SLOWDOWN":         "Slow or latent degradation",
    "MEMORY_LEAK":         "Slow or latent degradation",
    "LATENCY_SPIKE":       "Slow or latent degradation",
}


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class LifeDataExtractor:
    """
    Transforms pipeline episode CSV outputs into Weibull life-data format.

    Parameters
    ----------
    pipeline_results_csv : Path, optional
        Path to pipeline_results.csv (cycle-level, 78,000 rows).
    severity_update_csv  : Path, optional
        Path to severity_update_output.csv (episode-level summary, 600–650 rows).
    human_gate_csv       : Path, optional
        Path to human_gate_output.csv (resolved incident MTTR data).
    episode_window_s     : float, optional
        Total episode observation window in seconds. Default: 238.0.
    """

    def __init__(
        self,
        pipeline_results_csv: Path | None = None,
        severity_update_csv: Path | None = None,
        human_gate_csv: Path | None = None,
        episode_window_s: float = _EPISODE_WINDOW_S,
    ) -> None:
        self._pipeline_csv   = Path(pipeline_results_csv or _PIPELINE_RESULTS_CSV)
        self._severity_csv   = Path(severity_update_csv  or SEVERITY_UPDATE_CSV)
        self._human_gate_csv = Path(human_gate_csv       or HUMAN_GATE_OUTPUT_CSV)
        self._window         = float(episode_window_s)

        # Internal state (populated by run())
        self._life_data: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> pd.DataFrame:
        """
        Execute the full extraction pipeline and return life_data DataFrame.

        Returns
        -------
        pd.DataFrame
            One row per unique episode with columns:
            episode_id, failure_mode, ttf_seconds, event,
            revised_severity, total_window_s, data_source, ttr_seconds.
        """
        print("[Extractor] Starting life-data extraction ...")

        # STEP 1 ── Load pipeline_results.csv (cycle-level data, 78k rows)
        df_pipeline = self._load_pipeline_results()

        # STEP 2 ── Load severity_update_output.csv (episode summary, ~600 rows)
        df_summary = self._load_severity_summary()

        # STEP 3 ── Compute first P1 detection time per episode
        first_p1 = self._compute_first_p1(df_pipeline)

        # STEP 4 ── Build life-data records (TTF + censoring flag)
        records = self._build_records(df_summary, first_p1)

        # STEP 5 ── Merge MTTR data from Human Gate (if available)
        records = self._merge_human_gate_ttr(records)

        # STEP 6 ── Convert to DataFrame and validate
        df = pd.DataFrame(records)
        self._validate(df)

        self._life_data = df
        print(f"[Extractor] Done. {len(df):,} episodes extracted.")
        return df

    def save(self, path: Path | None = None) -> Path:
        """
        Write life_data_extracted.csv to disk.

        Parameters
        ----------
        path : Path, optional
            Override output path. Default: nodes/reliability/output/life_data_extracted.csv

        Returns
        -------
        Path
            Absolute path to the written CSV file.
        """
        if self._life_data is None:
            raise RuntimeError("Call run() before save().")

        out_path = Path(path or _OUTPUT_CSV)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self._life_data.to_csv(out_path, index=False)
        print(f"[Extractor] CSV written -> {out_path}")
        return out_path

    def summary(self) -> str:
        """Return a formatted text summary of the extracted life-data."""
        if self._life_data is None:
            return "[Extractor] No data yet. Call run() first."

        df = self._life_data
        total       = len(df)
        n_failed    = (df["event"] == 1).sum()
        n_censored  = (df["event"] == 0).sum()
        n_with_ttr  = df["ttr_seconds"].notna().sum()
        avg_ttf     = df.loc[df["event"] == 1, "ttf_seconds"].mean()

        lines = [
            "",
            "=" * 60,
            "  Life-Data Extraction Summary",
            "=" * 60,
            f"  Total Episodes            : {total:>6,}",
            f"  Complete Failures  (C=1)  : {n_failed:>6,}  (event=1)",
            f"  Right-Censored     (C=0)  : {n_censored:>6,}  (event=0)",
            f"  Avg TTF (failures)        : {avg_ttf:>6.1f} s",
            f"  Human Gate TTR rows       : {n_with_ttr:>6,}",
            "-" * 60,
            "  Breakdown by Failure Mode (event=1 / total):",
        ]

        mode_counts = (
            df.groupby("failure_mode")["event"]
            .agg(failed=lambda x: (x == 1).sum(), total="count")
            .reset_index()
            .sort_values("total", ascending=False)
        )
        for _, row in mode_counts.iterrows():
            lines.append(
                f"    {row['failure_mode']:<25} "
                f"failed={row['failed']:>3}  total={row['total']:>3}"
            )

        lines += ["=" * 60, ""]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_pipeline_results(self) -> pd.DataFrame:
        """Load cycle-level pipeline_results.csv."""
        if not self._pipeline_csv.exists():
            raise FileNotFoundError(
                f"[Extractor] pipeline_results.csv not found at:\n  {self._pipeline_csv}\n"
                "  Run the full pipeline first:\n"
                "  python backend/run_pipeline.py"
            )
        print(f"[Extractor] Loading pipeline_results.csv ({self._pipeline_csv}) ...")
        df = pd.read_csv(self._pipeline_csv)
        df.columns = df.columns.str.strip()
        print(f"[Extractor]   Loaded {len(df):,} rows, {df['episode_id'].nunique():,} unique episodes.")
        return df

    def _load_severity_summary(self) -> pd.DataFrame:
        """Load episode-level severity_update_output.csv."""
        if not self._severity_csv.exists():
            raise FileNotFoundError(
                f"[Extractor] severity_update_output.csv not found at:\n  {self._severity_csv}\n"
                "  Run: python backend/nodes/severity_update/run_severity_update.py"
            )
        print(f"[Extractor] Loading severity_update_output.csv ...")
        df = pd.read_csv(self._severity_csv)
        df.columns = df.columns.str.strip()
        print(f"[Extractor]   Loaded {len(df):,} episode summary rows.")
        return df

    def _compute_first_p1(self, df_pipeline: pd.DataFrame) -> pd.Series:
        """
        Find the earliest elapsed_s (seconds) where preliminary_severity == 'P1'
        for each episode_id.

        Returns
        -------
        pd.Series
            Index: episode_id, Values: elapsed_s at first P1 detection.
            Episodes that never reach P1 are NOT in this series.
        """
        p1_rows = df_pipeline[df_pipeline["preliminary_severity"] == "P1"]
        first_p1 = p1_rows.groupby("episode_id")["elapsed_s"].min()
        print(
            f"[Extractor] First-P1 detection found for "
            f"{len(first_p1):,} episodes  "
            f"(avg={first_p1.mean():.1f}s, min={first_p1.min():.1f}s, max={first_p1.max():.1f}s)"
        )
        return first_p1

    def _build_records(
        self,
        df_summary: pd.DataFrame,
        first_p1: pd.Series,
    ) -> list[dict]:
        """
        Build the core life-data record for every episode.

        Logic
        -----
        For each episode in severity_update_output.csv:

          If failure_mode != 'NONE'  AND  episode_id is in first_p1:
              -> Complete Failure  (event=1)
              -> ttf_seconds = first_p1[episode_id]
              -> data_source = 'pipeline_results_P1'

          Otherwise (NONE episodes, or failure episodes where P1 was never raised):
              -> Right-Censored    (event=0)
              -> ttf_seconds = total_window_s (238.0 s)
              -> data_source = 'right_censored'
        """
        records: list[dict] = []

        for _, row in df_summary.iterrows():
            ep_id         = str(row.get("episode_id", ""))
            failure_mode  = str(row.get("failure_mode", "NONE"))
            prelim_sev    = str(row.get("preliminary_severity", "P4"))
            revised_sev   = str(row.get("revised_severity", "P4"))

            is_failure_mode = (failure_mode.upper() != "NONE")
            has_p1_detected = ep_id in first_p1.index if hasattr(first_p1, 'index') else False
            ttf_val = row.get("time_to_failure")
            
            ttf_s = None
            if is_failure_mode and pd.notna(ttf_val) and float(ttf_val) >= 0:
                raw_ttf = float(ttf_val)
                if raw_ttf <= self._window:
                    ttf_s = max(0.1, raw_ttf)
                    event = 1
                    data_source = "auto_arima_forecast"
                else:
                    ttf_s = self._window
                    event = 0
                    data_source = "right_censored_beyond_window"
            elif is_failure_mode and revised_sev in ("P1", "P2"):
                ttf_s = min(self._window, float(row.get("elapsed_s", 238.0))) if pd.notna(row.get("elapsed_s")) else 238.0
                event = 1
                data_source = "severity_update_P1"
            elif is_failure_mode and has_p1_detected:
                ttf_s = min(self._window, float(first_p1[ep_id]))
                event = 1
                data_source = "pipeline_results_P1"
            else:
                ttf_s = self._window
                event = 0
                data_source = "right_censored"

            records.append({
                "episode_id":            ep_id,
                "failure_mode":          failure_mode,
                "failure_group":         FAILURE_GROUP_MAP.get(failure_mode, "Healthy / Unassigned"),
                "preliminary_severity":  prelim_sev,
                "ttf_seconds":           ttf_s,
                "event":                 event,
                "revised_severity":      revised_sev,
                "total_window_s":        self._window,
                "data_source":           data_source,
                "ttr_seconds":           None,   # filled by _merge_human_gate_ttr()
            })

        n_failed   = sum(1 for r in records if r["event"] == 1)
        n_censored = sum(1 for r in records if r["event"] == 0)
        print(
            f"[Extractor] Built {len(records):,} records -> "
            f"Complete failures (C=1): {n_failed:,}  |  "
            f"Right-censored  (C=0): {n_censored:,}"
        )
        return records

    def _merge_human_gate_ttr(self, records: list[dict]) -> list[dict]:
        """
        Optionally merge Time-To-Repair (TTR) from human_gate_output.csv.

        TTR = decided_at - created_at  (operator resolution time in seconds)

        This is used by metrics.py to compute:
            MTTR = mean(TTR)
            Availability = MTTF / (MTTF + MTTR)

        If human_gate_output.csv is empty or missing, ttr_seconds stays None.
        That is expected if the Human Gate has not been run yet.
        """
        if not self._human_gate_csv.exists():
            print("[Extractor] human_gate_output.csv not found — TTR will be NaN.")
            return records

        df_gate = pd.read_csv(self._human_gate_csv)
        if df_gate.empty:
            print("[Extractor] human_gate_output.csv is empty — TTR will be NaN.")
            return records

        # Compute TTR per episode from Human Gate timestamps
        ttr_map: dict[str, float] = {}
        if "created_at" in df_gate.columns and "decided_at" in df_gate.columns:
            df_gate = df_gate.dropna(subset=["created_at", "decided_at"])
            if not df_gate.empty:
                df_gate["created_at"] = pd.to_datetime(df_gate["created_at"], utc=True, errors="coerce")
                df_gate["decided_at"] = pd.to_datetime(df_gate["decided_at"], utc=True, errors="coerce")
                df_gate = df_gate.dropna(subset=["created_at", "decided_at"])
                df_gate["ttr_s"] = (
                    df_gate["decided_at"] - df_gate["created_at"]
                ).dt.total_seconds()
                ttr_map = df_gate.set_index("episode_id")["ttr_s"].to_dict()

        n_merged = 0
        for rec in records:
            ep_id = rec["episode_id"]
            if ep_id in ttr_map:
                rec["ttr_seconds"] = ttr_map[ep_id]
                n_merged += 1

        print(f"[Extractor] Merged TTR (Time-To-Repair) for {n_merged:,} episodes from Human Gate.")
        return records

    @staticmethod
    def _validate(df: pd.DataFrame) -> None:
        """Run basic data-quality assertions on the extracted life-data."""
        assert not df.empty, "Life-data DataFrame must not be empty!"

        # All TTF values must be positive and ≤ total window
        assert (df["ttf_seconds"] > 0).all(), "All TTF values must be > 0!"
        assert (df["ttf_seconds"] <= _EPISODE_WINDOW_S + 0.01).all(), (
            f"Some TTF values exceed total window ({_EPISODE_WINDOW_S}s)!"
        )

        # event must be 0 or 1
        assert df["event"].isin([0, 1]).all(), "event column must contain only 0 or 1!"

        # Right-censored episodes must have ttf_seconds >= window
        censored = df[df["event"] == 0]
        assert (censored["ttf_seconds"] >= _EPISODE_WINDOW_S - 0.01).all(), (
            "All right-censored episodes must have ttf_seconds >= 238.0!"
        )

        print(f"[Extractor] Validation passed OK  ({len(df):,} rows, no data-quality issues)")


# ---------------------------------------------------------------------------
# Convenience functional interface
# ---------------------------------------------------------------------------

def extract_life_data(
    pipeline_results_csv: Path | None = None,
    severity_update_csv: Path | None = None,
    human_gate_csv: Path | None = None,
    episode_window_s: float = _EPISODE_WINDOW_S,
    save: bool = True,
) -> pd.DataFrame:
    """
    One-shot life-data extraction function.

    Parameters
    ----------
    pipeline_results_csv : Path, optional
    severity_update_csv  : Path, optional
    human_gate_csv       : Path, optional
    episode_window_s     : float, optional (default 238.0)
    save                 : bool, optional  (default True — writes CSV to disk)

    Returns
    -------
    pd.DataFrame
        life_data_extracted DataFrame.
    """
    extractor = LifeDataExtractor(
        pipeline_results_csv=pipeline_results_csv,
        severity_update_csv=severity_update_csv,
        human_gate_csv=human_gate_csv,
        episode_window_s=episode_window_s,
    )
    df = extractor.run()
    if save:
        extractor.save()
    return df
