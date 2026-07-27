"""
app_simulator/offline/train_classifier.py
==========================================
Offline LightGBM training script.

Reads the engineered_features.csv produced by the live pipeline (or the
latest versioned CSV from the output/ folder), trains a LightGBM classifier,
and saves:
  models/lgbm_model.pkl        -- trained LightGBM model
  models/label_encoder.pkl     -- LabelEncoder (str labels <-> int indices)
  models/feature_names.json    -- ordered list of 32 feature column names

Also saves visual outputs to models/:
  confusion_matrix_lgbm.png    -- confusion matrix with full X/Y class labels
  feature_importance_lgbm.png  -- top-20 feature importances

Usage:
    python app_simulator/offline/train_classifier.py
    python app_simulator/offline/train_classifier.py --csv pipeline/output/engineered_features.csv
    python app_simulator/offline/train_classifier.py --episodes 5  # quick test

Training input CSV format (from engineered_features.csv):
    episode_id, failure_mode, timestamp, elapsed_s,
    [27 raw metric cols], log_count, log_max_severity, log_critical_count,
    log_has_exception, log_has_novel_template

    failure_mode column = training label (Y)
    All other non-meta columns = features (X)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# ── resolve package root ─────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent))

from app_data_generator.config import (
    ALL_MODES,
    ENGINEERED_FEAT_CSV,
    FEATURE_NAMES_JSON,
    LABEL_ENCODER_PKL,
    LGBM_MODEL_PKL,
    LOG_FEATURE_COLS,
    MODELS_DIR,
    RAW_METRIC_FEATURE_COLS,
    PIPELINE_OUTPUT_DIR,
)

# ── Constants ────────────────────────────────────────────────────────────────
RANDOM_SEED = 42
TEST_SIZE   = 0.20

# All 32 feature columns (27 raw metrics + 5 log)
FEATURE_COLS = RAW_METRIC_FEATURE_COLS + LOG_FEATURE_COLS

# Columns to drop before building X matrix
META_COLS = ["episode_id", "failure_mode", "timestamp", "elapsed_s", "preliminary_severity"]

# Circuit breaker encoding (must match metrics_features.py)
CB_ENCODE = {"closed": 0, "half-open": 1, "open": 2}


# =============================================================================
# Data loading
# =============================================================================

def load_training_data(csv_path: str) -> pd.DataFrame:
    """Load engineered_features.csv. Ensures failure_mode column present."""
    print(f"\n[Load] Reading {csv_path} ...")
    df = pd.read_csv(csv_path, dtype=str).fillna("0")
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")

    if "failure_mode" not in df.columns:
        raise ValueError("CSV missing 'failure_mode' column. Cannot train.")

    # Encode circuit_breaker_state string → int
    if "circuit_breaker_state" in df.columns:
        df["circuit_breaker_state"] = (
            df["circuit_breaker_state"].str.lower().map(CB_ENCODE).fillna(0).astype(int)
        )

    # Convert all feature cols to float
    for col in FEATURE_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    print(f"  Failure mode distribution:")
    for mode, cnt in sorted(df["failure_mode"].value_counts().items()):
        print(f"    {mode:<25}: {cnt:>6,}")

    return df


# =============================================================================
# Train/test split (stratified by failure_mode)
# =============================================================================

def build_matrices(df: pd.DataFrame) -> tuple:
    """
    Episode-level stratified 80/20 split. Returns (X_train, X_test, y_train, y_test, feature_cols).
    """
    # Use only columns that exist in FEATURE_COLS
    available_features = [c for c in FEATURE_COLS if c in df.columns]
    missing = set(FEATURE_COLS) - set(available_features)
    if missing:
        print(f"  [WARN] Missing feature columns (will use 0): {sorted(missing)}")
        for col in missing:
            df[col] = 0.0

    X = df[available_features].values.astype(float)
    y = df["failure_mode"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_SEED,
    )

    print(f"\n[Split] Stratified 80/20 (seed={RANDOM_SEED}):")
    print(f"  X_train : {X_train.shape}   y_train : {y_train.shape}")
    print(f"  X_test  : {X_test.shape}    y_test  : {y_test.shape}")

    classes = sorted(set(y))
    print(f"  Classes : {len(classes)}  ->  {classes}")

    return X_train, X_test, y_train, y_test, available_features


# =============================================================================
# LightGBM training
# =============================================================================

def train_lgbm(X_train, y_train, le: LabelEncoder):
    """Train LightGBM with encoded labels."""
    try:
        import lightgbm as lgb
    except ImportError:
        print("\n[ERROR] LightGBM not installed. Run: pip install lightgbm")
        sys.exit(1)

    y_tr_enc = le.fit_transform(y_train)

    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.1,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=20,
        random_state=RANDOM_SEED,
        verbosity=-1,
        n_jobs=-1,
    )
    print("\n[Train] Fitting LightGBM ...")
    model.fit(X_train, y_tr_enc)
    print("  Training done.")
    return model


# =============================================================================
# Evaluation
# =============================================================================

def evaluate(model, le: LabelEncoder,
             X_train, X_test, y_train, y_test,
             feature_names: list[str],
             out_dir: Path) -> None:
    """Evaluate model, print metrics, save confusion matrix and feature importance."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y_tr_enc = le.transform(y_train)
    y_te_enc = le.transform(y_test)

    pred_train = model.predict(X_train)
    pred_test  = model.predict(X_test)

    train_acc = accuracy_score(y_tr_enc, pred_train)
    test_acc  = accuracy_score(y_te_enc, pred_test)

    # Decode predictions back to string labels
    pred_test_labels  = le.inverse_transform(pred_test)
    pred_train_labels = le.inverse_transform(pred_train)

    print(f"\n{'='*65}")
    print(f"  LightGBM EVALUATION")
    print(f"{'='*65}")
    print(f"  Train accuracy : {train_acc*100:.2f}%")
    print(f"  Test  accuracy : {test_acc*100:.2f}%")
    print(f"\n  Classification Report (Test):")
    print(classification_report(y_test, pred_test_labels, zero_division=0))

    # ── Confusion matrix with proper X/Y class labels ────────────────────────
    classes = sorted(set(y_test))
    cm = confusion_matrix(y_test, pred_test_labels, labels=classes)

    fig, ax = plt.subplots(figsize=(16, 12))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(
        ax=ax,
        xticks_rotation=45,
        colorbar=True,
        cmap="Blues",
        include_values=True,
    )

    # Axis labels
    ax.set_xlabel("Predicted Label", fontsize=13, fontweight="bold", labelpad=12)
    ax.set_ylabel("True Label",      fontsize=13, fontweight="bold", labelpad=12)
    ax.set_title(
        f"LightGBM Confusion Matrix (Test Set)\n"
        f"Accuracy: {test_acc*100:.2f}%  |  Classes: {len(classes)}",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )

    # Make tick labels readable
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(classes, fontsize=9)

    plt.tight_layout()
    cm_path = out_dir / "confusion_matrix_lgbm.png"
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  [Saved] Confusion matrix -> {cm_path}")

    # ── Feature importance ───────────────────────────────────────────────────
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        top_n = min(20, len(feature_names))
        indices = np.argsort(importances)[::-1][:top_n]
        top_feat = [feature_names[i] for i in indices]
        top_imp  = importances[indices]

        fig2, ax2 = plt.subplots(figsize=(10, 8))
        bars = ax2.barh(range(top_n), top_imp[::-1], color="steelblue", edgecolor="white")
        ax2.set_yticks(range(top_n))
        ax2.set_yticklabels(top_feat[::-1], fontsize=9)
        ax2.set_xlabel("Feature Importance (Gain)", fontsize=11)
        ax2.set_ylabel("Feature Name",              fontsize=11)
        ax2.set_title(
            f"Top {top_n} Feature Importances — LightGBM",
            fontsize=13,
            fontweight="bold",
        )
        ax2.bar_label(bars, fmt="%.3f", padding=2, fontsize=7)
        plt.tight_layout()
        fi_path = out_dir / "feature_importance_lgbm.png"
        plt.savefig(fi_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [Saved] Feature importance  -> {fi_path}")

        print(f"\n  Top 10 features:")
        for i in range(min(10, top_n)):
            print(f"    {top_feat[i]:<30} {top_imp[i]:.4f}")


# =============================================================================
# Save model artifacts
# =============================================================================

def save_artifacts(model, le: LabelEncoder, feature_names: list[str]) -> None:
    """Save model, label encoder, and feature names list."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, LGBM_MODEL_PKL)
    print(f"\n  [Saved] Model         -> {LGBM_MODEL_PKL}")

    joblib.dump(le, LABEL_ENCODER_PKL)
    print(f"  [Saved] LabelEncoder  -> {LABEL_ENCODER_PKL}")

    with open(FEATURE_NAMES_JSON, "w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=2)
    print(f"  [Saved] Feature names -> {FEATURE_NAMES_JSON}")

    print(f"\n  Classes ({len(le.classes_)}): {list(le.classes_)}")


# =============================================================================
# Entry point
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train LightGBM classifier on engineered_features.csv"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=str(ENGINEERED_FEAT_CSV),
        help=f"Path to engineered_features.csv (default: {ENGINEERED_FEAT_CSV})",
    )
    args = parser.parse_args()

    csv_path = args.csv
    if not os.path.exists(csv_path):
        # Fallback: check pipeline output dir
        alt = PIPELINE_OUTPUT_DIR / "engineered_features.csv"
        if alt.exists():
            csv_path = str(alt)
        else:
            print(f"[ERROR] CSV not found: {csv_path}")
            print(f"        Run the simulator + pipeline first to generate features.")
            sys.exit(1)

    print("=" * 65)
    print("  LightGBM Classifier Training")
    print("=" * 65)

    # Load
    df = load_training_data(csv_path)

    # Split
    X_train, X_test, y_train, y_test, feature_names = build_matrices(df)

    # Encode labels
    le = LabelEncoder()
    le.fit(y_train)

    # Train
    model = train_lgbm(X_train, y_train, le)

    # Evaluate + save plots
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    evaluate(model, le, X_train, X_test, y_train, y_test, feature_names, MODELS_DIR)

    # Save artifacts
    save_artifacts(model, le, feature_names)

    print(f"\n{'='*65}")
    print(f"  Training complete.")
    print(f"  Run the pipeline: python app_simulator/run_pipeline.py")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
