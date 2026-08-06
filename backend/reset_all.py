"""
backend/reset_all.py
=====================
Idempotent data-reset script.

Wipes ALL output artifacts (CSVs, SQLite databases, trained models, Drain3
state, Optuna study, LangGraph checkpoints, Weibull params) so that the next
pipeline run starts from a perfectly clean slate.

All paths are read from app_data_generator/config.py — no hardcoded strings.

Usage:
    # Full wipe:
    python backend/reset_all.py

    # Preview only (no deletions):
    python backend/reset_all.py --dry-run

    # Keep trained model but wipe everything else:
    python backend/reset_all.py --keep-model
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ── Resolve package root ──────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from Simulator.app_data_generator_for_offline.config import (
    # Simulator database
    DB_PATH,
    # Node output CSVs
    ENGINEERED_FEAT_CSV,
    PRELIM_SEVERITY_CSV,
    PIPELINE_RESULTS_CSV,
    TUMBLING_WINDOW_CSV,
    FORECASTING_OUTPUT_CSV,
    SEVERITY_UPDATE_CSV,
    HUMAN_GATE_OUTPUT_CSV,
    HUMAN_GATE_AUDIT_DB,
    # Classification model artifacts
    LGBM_MODEL_PKL,
    LABEL_ENCODER_PKL,
    FEATURE_NAMES_JSON,
    MODELS_DIR,
    # Drain3 artifacts
    DRAIN_STATE,
    KNOWN_TEMPLATES_JSON,
    # Tuning artifacts
    TUNING_STUDY_DB,
    LANGGRAPH_CHECKPOINT_DB,
    # Reliability output
    NODES_DIR,
)

# ── Derived paths from NODES_DIR ──────────────────────────────────────────────
RELIABILITY_OUTPUT_DIR   = NODES_DIR / "reliability" / "output"
WEIBULL_PNG              = RELIABILITY_OUTPUT_DIR / "weibull_km_groups.png"
LIFE_DATA_CSV            = RELIABILITY_OUTPUT_DIR / "life_data_extracted.csv"
WEIBULL_PARAMS_JSON      = NODES_DIR / "reliability" / "weibull_params.json"
CONFUSION_MATRIX_PNG     = MODELS_DIR / "confusion_matrix_lgbm.png"
FEATURE_IMPORTANCE_PNG   = MODELS_DIR / "feature_importance_lgbm.png"
F1_COMPARISON_PNG        = MODELS_DIR / "f1_comparison_bar.png"
TUNING_RESULTS_JSON      = MODELS_DIR / "tuning_results.json"
DRAIN_INI_BACKUP         = MODELS_DIR / "drain3.ini"


# ── Build the full artifact list ──────────────────────────────────────────────
def _build_artifact_list(keep_model: bool) -> list[tuple[str, Path]]:
    """
    Return list of (label, path) tuples representing every artifact to delete.
    If keep_model=True, trained model files are excluded.
    """
    artifacts: list[tuple[str, Path]] = [
        # ── Simulator raw database ────────────────────────────────────────────
        ("Simulator DB",              DB_PATH),

        # ── Node output CSVs ──────────────────────────────────────────────────
        ("Engineered Features CSV",   ENGINEERED_FEAT_CSV),
        ("Preliminary Severity CSV",  PRELIM_SEVERITY_CSV),
        ("Pipeline Results CSV",      PIPELINE_RESULTS_CSV),
        ("Tumbling Window CSV",       TUMBLING_WINDOW_CSV),
        ("Forecasting Output CSV",    FORECASTING_OUTPUT_CSV),
        ("Severity Update CSV",       SEVERITY_UPDATE_CSV),
        ("Human Gate Output CSV",     HUMAN_GATE_OUTPUT_CSV),
        ("Human Gate Audit DB",       HUMAN_GATE_AUDIT_DB),

        # ── Reliability outputs ───────────────────────────────────────────────
        ("Life Data CSV",             LIFE_DATA_CSV),
        ("Weibull KM Plot",           WEIBULL_PNG),
        ("Weibull Params JSON",       WEIBULL_PARAMS_JSON),

        # ── Drain3 state ──────────────────────────────────────────────────────
        ("Drain3 State Binary",       DRAIN_STATE),
        ("Known Log Templates JSON",  KNOWN_TEMPLATES_JSON),

        # ── LangGraph checkpoints ─────────────────────────────────────────────
        ("LangGraph Checkpoint DB",   Path(LANGGRAPH_CHECKPOINT_DB)),
    ]

    # Model artifacts — only deleted when keep_model=False
    model_artifacts: list[tuple[str, Path]] = [
        ("LightGBM Model PKL",        LGBM_MODEL_PKL),
        ("Label Encoder PKL",         LABEL_ENCODER_PKL),
        ("Feature Names JSON",        FEATURE_NAMES_JSON),
        ("Optuna Study DB",           Path(TUNING_STUDY_DB)),
        ("Confusion Matrix PNG",      CONFUSION_MATRIX_PNG),
        ("Feature Importance PNG",    FEATURE_IMPORTANCE_PNG),
        ("F1 Comparison PNG",         F1_COMPARISON_PNG),
        ("Tuning Results JSON",       TUNING_RESULTS_JSON),
    ]

    if not keep_model:
        artifacts.extend(model_artifacts)

    return artifacts


def _format_size(path: Path) -> str:
    """Return human-readable file size string."""
    try:
        size = path.stat().st_size
        if size < 1024:
            return f"{size} B"
        elif size < 1024 ** 2:
            return f"{size / 1024:.1f} KB"
        elif size < 1024 ** 3:
            return f"{size / 1024**2:.1f} MB"
        else:
            return f"{size / 1024**3:.2f} GB"
    except OSError:
        return "unknown"


def run_reset(dry_run: bool, keep_model: bool) -> None:
    """
    Execute the reset.

    Args:
        dry_run:    If True, print what would be deleted without actually deleting.
        keep_model: If True, skip trained model + Optuna artifacts.
    """
    artifacts = _build_artifact_list(keep_model)

    mode_label = "[DRY-RUN] " if dry_run else ""
    keep_label = " (model artifacts preserved)" if keep_model else ""

    print("=" * 70)
    print(f"  {mode_label}AIOps Data Reset{keep_label}")
    print("=" * 70)

    deleted_count = 0
    skipped_count = 0
    total_size    = 0

    for label, path in artifacts:
        if path.exists():
            size_str = _format_size(path)
            size_bytes = path.stat().st_size
            total_size += size_bytes
            if dry_run:
                print(f"  [WOULD DELETE] {label:<28} {size_str:>8}   {path}")
            else:
                try:
                    path.unlink()
                    print(f"  [✓ DELETED]   {label:<28} {size_str:>8}   {path}")
                    deleted_count += 1
                except PermissionError as exc:
                    print(f"  [✗ LOCKED]    {label:<28}          {exc}")
                except Exception as exc:
                    print(f"  [✗ ERROR]     {label:<28}          {exc}")
        else:
            print(f"  [— MISSING]   {label:<28}            {path}")
            skipped_count += 1

    print("-" * 70)
    total_mb = total_size / 1024 ** 2
    if dry_run:
        print(f"  DRY-RUN complete. Would free {total_mb:.1f} MB across {len(artifacts) - skipped_count} files.")
    else:
        print(f"  Reset complete.   Freed {total_mb:.1f} MB. "
              f"Deleted={deleted_count}, Already-missing={skipped_count}.")
    print("=" * 70)

    if not dry_run:
        print("\n  Next steps:")
        print("  1. python backend/app_data_generator/run_simulator.py --speed 50")
        print("  2. python backend/run_langgraph.py --speed 50")
        print("  3. python backend/nodes/classification/train_classifier.py --tune --gpu")
        print("  4. python backend/nodes/reliability/run_weibull_fitter.py")
        print("  5. python backend/api/main.py")
        print("  6. cd frontend && npm run dev\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Wipe all AIOps output artifacts for a clean pipeline re-run.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python backend/reset_all.py             # full wipe
  python backend/reset_all.py --dry-run   # preview only
  python backend/reset_all.py --keep-model  # keep trained LightGBM model
        """,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Preview deletions without actually deleting anything.",
    )
    parser.add_argument(
        "--keep-model",
        action="store_true",
        default=False,
        help="Preserve trained model artifacts (lgbm_model.pkl, label_encoder.pkl, etc.).",
    )
    args = parser.parse_args()
    run_reset(dry_run=args.dry_run, keep_model=args.keep_model)


if __name__ == "__main__":
    main()
