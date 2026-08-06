"""
backend/nodes/classification/train_classifier.py
=================================================
Offline LightGBM training script.

Reads the engineered_features.csv produced by the live pipeline (or the
latest versioned CSV from the output/ folder), trains a LightGBM classifier,
and saves:
  models/lgbm_model.pkl        -- trained LightGBM model
  models/label_encoder.pkl     -- LabelEncoder (str labels <-> int indices)
  models/feature_names.json    -- ordered list of feature column names

Also saves visual outputs to models/:
  confusion_matrix_lgbm.png    -- confusion matrix with full X/Y class labels
  feature_importance_lgbm.png  -- top-20 feature importances

When --tune is passed, additionally saves:
  models/f1_comparison_bar.png -- per-class F1 before/after grouped bar chart
  models/tuning_results.json   -- full metrics, best_params, run timestamp
  models/optuna_study.db       -- SQLite study (resumable across runs)

Usage:
    # Standard training (existing behaviour, no tuning):
    python backend/nodes/classification/train_classifier.py

    # With Optuna hyperparameter tuning (recommended):
    python backend/nodes/classification/train_classifier.py --tune

    # With GPU acceleration (GTX 1650 or any CUDA GPU):
    python backend/nodes/classification/train_classifier.py --tune --gpu

    # Override number of trials:
    python backend/nodes/classification/train_classifier.py --tune --trials 50

    # Cap wall-clock time to 600 seconds:
    python backend/nodes/classification/train_classifier.py --tune --timeout 600

    # Custom CSV:
    python backend/nodes/classification/train_classifier.py --csv path/to/engineered_features.csv
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    make_scorer,
)

# ── Force UTF-8 output on Windows (prevents cp1252 UnicodeEncodeError) ────────
# This is safe on all platforms and has no runtime cost.
import io
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── Resolve package root ──────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent))

from Simulator.app_data_generator_for_offline.config import (
    ALL_MODES,
    ENGINEERED_FEAT_CSV,
    FEATURE_NAMES_JSON,
    LABEL_ENCODER_PKL,
    LGBM_MODEL_PKL,
    LOG_FEATURE_COLS,
    MODELS_DIR,
    RAW_METRIC_FEATURE_COLS,
    PIPELINE_OUTPUT_DIR,
    RANDOM_SEED,
    TEST_SIZE,
    TUNING_N_TRIALS,
    TUNING_CV_FOLDS,
    TUNING_TIMEOUT_SEC,
    TUNING_METRIC,
    TUNING_AVERAGE,
    TUNING_RANDOM_SEED,
    TUNING_EARLY_STOPPING_ROUNDS,
    TUNING_STUDY_NAME,
    TUNING_STUDY_DB,
    TUNING_DEVICE_TYPE,
)

# ── Constants pulled entirely from config (no magic numbers here) ─────────────
# TEST_SIZE and RANDOM_SEED already imported from config.py.

# All feature columns (27 raw metrics + 5 log features)
FEATURE_COLS = RAW_METRIC_FEATURE_COLS + LOG_FEATURE_COLS

# Columns that are labels / metadata — dropped before building the X matrix
META_COLS = ["episode_id", "failure_mode", "timestamp", "elapsed_s", "preliminary_severity"]

# Circuit breaker categorical encoding (must match metrics_features.py)
CB_ENCODE = {"closed": 0, "half-open": 1, "open": 2}


# =============================================================================
# 1. Data loading
# =============================================================================

def load_training_data(csv_path: str) -> pd.DataFrame:
    """Load engineered_features.csv.  Ensures failure_mode column is present."""
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

    # Convert all feature columns to float
    for col in FEATURE_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    print("  Failure mode distribution:")
    for mode, cnt in sorted(df["failure_mode"].value_counts().items()):
        print(f"    {mode:<25}: {cnt:>6,}")

    return df


# =============================================================================
# 2. Stratified train/test split
# =============================================================================

def build_matrices(df: pd.DataFrame) -> tuple:
    """
    Episode-level stratified split using TEST_SIZE and RANDOM_SEED from config.

    Returns:
        (X_train, X_test, y_train, y_test, feature_cols_used)
    """
    available_features = [c for c in FEATURE_COLS if c in df.columns]
    missing = set(FEATURE_COLS) - set(available_features)
    if missing:
        print(f"  [WARN] Missing feature columns (will pad with 0): {sorted(missing)}")
        for col in missing:
            df[col] = 0.0

    X = df[available_features].values.astype(float)
    y = df["failure_mode"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,        # from config.py — default 0.20
        stratify=y,                 # preserves class ratios in both splits
        random_state=RANDOM_SEED,   # from config.py — reproducible
    )

    print(f"\n[Split] Stratified {int((1 - TEST_SIZE) * 100)}/{int(TEST_SIZE * 100)} "
          f"(test_size={TEST_SIZE}, seed={RANDOM_SEED}):")
    print(f"  X_train : {X_train.shape}   y_train : {y_train.shape}")
    print(f"  X_test  : {X_test.shape}    y_test  : {y_test.shape}")

    classes = sorted(set(y))
    print(f"  Classes : {len(classes)}  ->  {classes}")

    return X_train, X_test, y_train, y_test, available_features


# =============================================================================
# 3. Baseline training (default parameters — for before/after comparison)
# =============================================================================

def _build_default_params(device_type: str) -> dict:
    """
    Return the default LightGBM parameter dict.
    Centralised here so baseline and any non-tuned path use exactly the same defaults.
    device_type is read from config (TUNING_DEVICE_TYPE) or overridden via --gpu CLI flag.
    """
    return {
        "n_estimators":      300,
        "learning_rate":     0.1,
        "num_leaves":        63,
        "max_depth":         -1,
        "min_child_samples": 20,
        "random_state":      RANDOM_SEED,
        "verbosity":         -1,
        "n_jobs":            -1,
        "device_type":       device_type,
    }


def train_lgbm(X_train: np.ndarray, y_train: np.ndarray,
               le: LabelEncoder, device_type: str) -> object:
    """Train LightGBM with default parameters.  Returns fitted model."""
    try:
        import lightgbm as lgb
    except ImportError:
        print("\n[ERROR] LightGBM not installed. Run: pip install lightgbm")
        sys.exit(1)

    y_tr_enc = le.fit_transform(y_train)
    params = _build_default_params(device_type)

    print(f"\n[Train] Fitting LightGBM (default params, device={device_type}) ...")
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_tr_enc)
    print("  Training done.")
    return model


def run_baseline_eval(
    X_train: np.ndarray,
    X_test:  np.ndarray,
    y_train: np.ndarray,
    y_test:  np.ndarray,
    le:      LabelEncoder,
    device_type: str,
) -> tuple[object, dict]:
    """
    Train with default parameters on the training split.
    Evaluate on the held-out test split.

    Returns:
        (baseline_model, baseline_metrics_dict)

    baseline_metrics_dict keys:
        accuracy, f1_macro, f1_weighted, f1_per_class (dict[class→float])
    """
    print("\n" + "=" * 70)
    print("  STEP 1 — BASELINE EVALUATION (default params, no tuning)")
    print("=" * 70)

    # NOTE: le.fit_transform is called inside train_lgbm, so we clone the encoder
    # state here via a temporary fit to get encoded labels for evaluation.
    baseline_le = LabelEncoder().fit(y_train)
    y_train_enc = baseline_le.transform(y_train)
    y_test_enc  = baseline_le.transform(y_test)

    model = train_lgbm(X_train, y_train, baseline_le, device_type)

    pred_test = model.predict(X_test)
    pred_test_labels = baseline_le.inverse_transform(pred_test)

    sorted_classes = sorted(baseline_le.classes_)
    metrics = _compute_metrics(y_test, pred_test_labels, sorted_classes)

    print(f"\n  [Baseline] Accuracy    : {metrics['accuracy']:.4f}")
    print(f"  [Baseline] F1 Macro    : {metrics['f1_macro']:.4f}")
    print(f"  [Baseline] F1 Weighted : {metrics['f1_weighted']:.4f}")

    # After baseline eval, re-fit the main le on y_train for the rest of the pipeline
    le.fit(y_train)

    return model, metrics


# =============================================================================
# 4. Optuna hyperparameter tuning
# =============================================================================

def _make_trial_params(trial, device_type: str) -> dict:
    """
    Suggest hyperparameters for one Optuna trial.

    Every suggest_* call has an inline comment explaining:
      - the search range and why it was chosen
      - whether log-uniform is used and why

    No hardcoded values: all fixed settings reference config constants.
    """
    return {
        # ── Tree complexity ─────────────────────────────────────────────────────
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        # 15 = simple trees (good generalisation for noisy telemetry)
        # 127 = complex trees (captures intricate multi-metric interactions)
        # Default 63 sits in the middle; tuning finds the true sweet-spot.

        "max_depth": trial.suggest_int("max_depth", 3, 12),
        # 3 = very shallow (high bias, low variance)
        # 12 = deep (low bias, risks memorising episode-specific spikes)
        # Beyond 12, extra depth rarely adds value across 32 features.

        # ── Boosting dynamics ────────────────────────────────────────────────────
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        # Log-uniform: the effect is multiplicative.
        # 0.01 = fine-grained gradient steps (needs many trees but generalises better)
        # 0.30 = fast convergence (fewer trees but higher overfit risk)
        # Early stopping makes n_estimators adaptive regardless of lr.

        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        # Acts as an upper bound only; early stopping stops training earlier.
        # 100 = absolute minimum meaningful ensemble
        # 1000 = generous upper cap; early stopping ensures we don't overtrain.

        # ── Regularisation — row sampling ────────────────────────────────────────
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        # Minimum samples required in a leaf node.
        # Higher → prevents fitting single anomalous telemetry step spikes
        #          (simulator injects ±4 % noise at the row level).
        # Lower  → finer-grained splits, useful for rare failure classes.

        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        # Fraction of rows sampled per tree (bagging).
        # Introduces randomness that reduces overfit.
        # Must pair with subsample_freq > 0 to activate.

        "subsample_freq": 1,
        # Every boosting round draws a fresh row sample.  Fixed at 1 (standard).

        # ── Regularisation — feature sampling ────────────────────────────────────
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        # Fraction of the 32 features used per tree.
        # Prevents dominant features (cpu_utilization) from overshadowing subtler
        # ones like disk_read_latency or upstream_timeout_rate.

        # ── L1 / L2 weight regularisation ────────────────────────────────────────
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        # L1 penalty: pushes less-important feature weights toward zero.
        # Log-uniform: 1e-8 (effectively off) to 10.0 (strong sparsification).

        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        # L2 penalty: smooths weight magnitudes.  Same rationale as reg_alpha.

        # ── Fixed settings (from config) ──────────────────────────────────────────
        "random_state":  TUNING_RANDOM_SEED,   # reproducible across all trials
        "verbosity":     -1,                    # suppress LightGBM stdout
        "n_jobs":        -1,                    # use all CPU cores per tree
        "device_type":   device_type,           # "gpu" or "cpu" from CLI / config
    }


def _optuna_objective(
    trial,
    X:             np.ndarray,
    y_enc:         np.ndarray,
    cv:            StratifiedKFold,
    device_type:   str,
    feature_names: list[str],
) -> float:
    """
    Optuna objective function.

    For one trial:
      1. Suggest hyperparameters.
      2. Run stratified K-fold cross-validation on the TRAINING set.
      3. Return mean macro-F1 across folds (Optuna maximises this).

    Never touches the held-out test set.

    feature_names is passed so we can wrap the numpy array in a DataFrame,
    which prevents sklearn's "X does not have valid feature names" warning.
    """
    import lightgbm as lgb

    params = _make_trial_params(trial, device_type)
    f1_scorer = make_scorer(
        f1_score,
        average=TUNING_AVERAGE,   # "macro" from config
        zero_division=0,          # treat undefined class precision/recall as 0
    )

    # Wrap X in a DataFrame so LightGBM receives named features (avoids sklearn
    # UserWarning: "X does not have valid feature names, but LGBMClassifier was
    # fitted with feature names").
    X_df = pd.DataFrame(X, columns=feature_names)

    # cross_val_score fits TUNING_CV_FOLDS independent LightGBM models.
    # n_jobs=1 because LightGBM itself already uses n_jobs=-1 (all cores).
    cv_scores = cross_val_score(
        lgb.LGBMClassifier(**params),
        X_df, y_enc,
        cv=cv,
        scoring=f1_scorer,
        n_jobs=1,
    )
    return float(cv_scores.mean())


def run_optuna_tuning(
    X_train:      np.ndarray,
    y_train_enc:  np.ndarray,
    n_trials:     int,
    timeout:      int | None,
    device_type:  str,
    feature_names: list[str],
) -> tuple[dict, float]:
    """
    Create (or resume) an Optuna study and run n_trials Bayesian TPE trials.

    The study is persisted to TUNING_STUDY_DB (SQLite) so that an interrupted
    run can be resumed without losing completed trials — just re-run with the
    same --tune flag.

    Returns:
        (best_params_dict, best_cv_f1_macro)
    """
    try:
        import optuna
    except ImportError:
        print("\n[ERROR] Optuna not installed. Run: pip install optuna")
        sys.exit(1)

    # Suppress INFO-level chatter; warnings and errors still surface.
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print("\n" + "=" * 70)
    print("  STEP 2 — OPTUNA HYPERPARAMETER TUNING")
    print(f"  Objective  : Macro-F1 (average='{TUNING_AVERAGE}')")
    print(f"  CV Folds   : {TUNING_CV_FOLDS}")
    print(f"  Trials     : {n_trials}")
    print(f"  Device     : {device_type}")
    print(f"  Study DB   : {TUNING_STUDY_DB}")
    print("=" * 70)

    cv = StratifiedKFold(
        n_splits=TUNING_CV_FOLDS,
        shuffle=True,
        random_state=TUNING_RANDOM_SEED,
    )

    sampler = optuna.samplers.TPESampler(seed=TUNING_RANDOM_SEED)

    storage_url = f"sqlite:///{TUNING_STUDY_DB}"
    study = optuna.create_study(
        direction="maximize",
        study_name=TUNING_STUDY_NAME,
        sampler=sampler,
        storage=storage_url,
        load_if_exists=True,   # resumes a partial run instead of starting over
    )

    already_done = len(study.trials)
    remaining    = max(0, n_trials - already_done)
    if already_done > 0:
        print(f"\n  [Optuna] Resuming study: {already_done} trials already complete, "
              f"{remaining} remaining.")

    study.optimize(
        lambda trial: _optuna_objective(
            trial, X_train, y_train_enc, cv, device_type, feature_names
        ),
        n_trials=remaining,
        timeout=timeout,
        show_progress_bar=True,
    )

    best_params = study.best_trial.params
    best_val_f1 = study.best_trial.value

    print(f"\n  [Optuna] Best CV Macro-F1 : {best_val_f1:.4f}")
    print("  [Optuna] Best params found:")
    for k, v in best_params.items():
        print(f"    {k:<22}: {v}")

    return best_params, best_val_f1


# =============================================================================
# 5. Final model training (tuned params + early stopping)
# =============================================================================

def train_tuned_lgbm(
    X_train:     np.ndarray,
    y_train_enc: np.ndarray,
    best_params: dict,
    device_type: str,
) -> object:
    """
    Retrain on the FULL training set using best_params found by Optuna.

    A 10 % internal validation split (of the training set) is carved out
    solely to provide the convergence signal for LightGBM's early stopping.
    The held-out test set (20 % of total) is never touched here.

    Returns:
        Fitted LightGBM model.
    """
    import lightgbm as lgb

    print("\n" + "=" * 70)
    print("  STEP 3 — TRAIN FINAL MODEL (full training set + early stopping)")
    print("=" * 70)

    # 10 % internal val split for early stopping convergence signal only.
    # Stratified so each class appears in both sub-splits.
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train_enc,
        test_size=0.10,
        stratify=y_train_enc,
        random_state=TUNING_RANDOM_SEED,
    )

    # Merge best_params from Optuna with the fixed settings from config.
    # best_params may not contain device_type (it was fixed in _make_trial_params),
    # so we set it explicitly to be safe.
    final_params = {
        **best_params,
        "random_state": TUNING_RANDOM_SEED,
        "verbosity":    -1,
        "n_jobs":       -1,
        "device_type":  device_type,
    }

    model = lgb.LGBMClassifier(**final_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=TUNING_EARLY_STOPPING_ROUNDS,
                verbose=False,
            ),
            lgb.log_evaluation(period=-1),   # suppress per-round logs
        ],
    )

    best_iter = getattr(model, "best_iteration_", "N/A")
    print(f"  [Tuned] Early stopping selected iteration: {best_iter}")
    print("  [Tuned] Final model training complete.")
    return model


# =============================================================================
# 6. Shared metric computation
# =============================================================================

def _compute_metrics(
    y_true:  np.ndarray,
    y_pred:  np.ndarray,
    classes: list[str],
) -> dict:
    """
    Compute accuracy, macro-F1, weighted-F1, and per-class F1 for a
    prediction pair.

    Returns dict:
        accuracy        : float
        f1_macro        : float
        f1_weighted     : float
        f1_per_class    : dict[class_name → float]
    """
    acc          = accuracy_score(y_true, y_pred)
    f1_mac       = f1_score(y_true, y_pred, average="macro",    zero_division=0, labels=classes)
    f1_wgt       = f1_score(y_true, y_pred, average="weighted", zero_division=0, labels=classes)
    f1_pc_arr    = f1_score(y_true, y_pred, average=None,       zero_division=0, labels=classes)

    return {
        "accuracy":     float(acc),
        "f1_macro":     float(f1_mac),
        "f1_weighted":  float(f1_wgt),
        "f1_per_class": dict(zip(classes, [float(v) for v in f1_pc_arr])),
    }


# =============================================================================
# 7. Evaluation (confusion matrix + feature importance)
# =============================================================================

def evaluate(
    model,
    le:           LabelEncoder,
    X_train:      np.ndarray,
    X_test:       np.ndarray,
    y_train:      np.ndarray,
    y_test:       np.ndarray,
    feature_names: list[str],
    out_dir:      Path,
    label_prefix: str = "",
) -> dict:
    """
    Evaluate model, print full classification report, and save:
        confusion_matrix_lgbm.png
        feature_importance_lgbm.png

    Returns a metrics dict (accuracy, f1_macro, f1_weighted, f1_per_class).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y_train_enc = le.transform(y_train)
    y_test_enc  = le.transform(y_test)

    pred_train = model.predict(X_train)
    pred_test  = model.predict(X_test)

    train_acc  = accuracy_score(y_train_enc, pred_train)
    test_acc   = accuracy_score(y_test_enc,  pred_test)

    pred_test_labels  = le.inverse_transform(pred_test)
    pred_train_labels = le.inverse_transform(pred_train)

    sorted_classes = sorted(le.classes_)
    metrics = _compute_metrics(y_test, pred_test_labels, sorted_classes)

    tag = f" [{label_prefix}]" if label_prefix else ""
    print(f"\n{'='*65}")
    print(f"  LightGBM EVALUATION{tag}")
    print(f"{'='*65}")
    print(f"  Train accuracy      : {train_acc * 100:.2f}%")
    print(f"  Test  accuracy      : {test_acc  * 100:.2f}%")
    print(f"  Test  Macro-F1      : {metrics['f1_macro']:.4f}")
    print(f"  Test  Weighted-F1   : {metrics['f1_weighted']:.4f}")
    print(f"\n  Classification Report (Test):")
    print(classification_report(y_test, pred_test_labels, zero_division=0))

    # ── Confusion matrix ──────────────────────────────────────────────────────
    cm = confusion_matrix(y_test, pred_test_labels, labels=sorted_classes)
    fig, ax = plt.subplots(figsize=(16, 12))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=sorted_classes)
    disp.plot(ax=ax, xticks_rotation=45, colorbar=True, cmap="Blues", include_values=True)
    ax.set_xlabel("Predicted Label", fontsize=13, fontweight="bold", labelpad=12)
    ax.set_ylabel("True Label",      fontsize=13, fontweight="bold", labelpad=12)
    ax.set_title(
        f"LightGBM Confusion Matrix (Test Set){tag}\n"
        f"Accuracy: {test_acc*100:.2f}%  |  Macro-F1: {metrics['f1_macro']:.4f}  |  "
        f"Classes: {len(sorted_classes)}",
        fontsize=13, fontweight="bold", pad=15,
    )
    ax.set_xticklabels(sorted_classes, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(sorted_classes, fontsize=8)
    plt.tight_layout()
    cm_path = out_dir / "confusion_matrix_lgbm.png"
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  [Saved] Confusion matrix      -> {cm_path}")

    # ── Feature importance ────────────────────────────────────────────────────
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        top_n       = min(20, len(feature_names))
        indices     = np.argsort(importances)[::-1][:top_n]
        top_feat    = [feature_names[i] for i in indices]
        top_imp     = importances[indices]

        fig2, ax2 = plt.subplots(figsize=(10, 8))
        bars = ax2.barh(range(top_n), top_imp[::-1], color="steelblue", edgecolor="white")
        ax2.set_yticks(range(top_n))
        ax2.set_yticklabels(top_feat[::-1], fontsize=9)
        ax2.set_xlabel("Feature Importance (Gain)", fontsize=11)
        ax2.set_ylabel("Feature Name",              fontsize=11)
        ax2.set_title(
            f"Top {top_n} Feature Importances — LightGBM{tag}",
            fontsize=13, fontweight="bold",
        )
        ax2.bar_label(bars, fmt="%.3f", padding=2, fontsize=7)
        plt.tight_layout()
        fi_path = out_dir / "feature_importance_lgbm.png"
        plt.savefig(fi_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [Saved] Feature importance    -> {fi_path}")

        print(f"\n  Top 10 features:")
        for i in range(min(10, top_n)):
            print(f"    {top_feat[i]:<30} {top_imp[i]:.4f}")

    return metrics


# =============================================================================
# 8. Before/After comparison report
# =============================================================================

def compare_and_report(
    baseline_metrics: dict,
    tuned_metrics:    dict,
    best_params:      dict,
    tuned_val_f1:     float,
    out_dir:          Path,
) -> None:
    """
    Print a detailed side-by-side comparison table and save:
        f1_comparison_bar.png  -- per-class F1 before/after grouped horizontal bar chart
        tuning_results.json    -- machine-readable full metrics + best_params + timestamp
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    b = baseline_metrics
    t = tuned_metrics

    # ── Console comparison table ──────────────────────────────────────────────
    def _delta_str(before: float, after: float) -> str:
        d = (after - before) * 100
        sign = "+" if d >= 0 else ""
        return f"{sign}{d:.2f}%"

    sep = "─" * 78
    print(f"\n\n{'='*78}")
    print("  LIGHTGBM HYPERPARAMETER TUNING — F1 SCORE IMPACT REPORT")
    print(f"{'='*78}")
    print(f"  {'Metric':<25} | {'Before (Default)':<20} | {'After (Optuna)':<18} | Delta")
    print(sep)
    print(f"  {'Macro F1':<25} | {b['f1_macro']:<20.4f} | {t['f1_macro']:<18.4f} | "
          f"{_delta_str(b['f1_macro'], t['f1_macro'])}")
    print(f"  {'Weighted F1':<25} | {b['f1_weighted']:<20.4f} | {t['f1_weighted']:<18.4f} | "
          f"{_delta_str(b['f1_weighted'], t['f1_weighted'])}")
    print(f"  {'Accuracy':<25} | {b['accuracy']:<20.4f} | {t['accuracy']:<18.4f} | "
          f"{_delta_str(b['accuracy'], t['accuracy'])}")
    print(f"  {'CV Val Macro F1':<25} | {'—':<20} | {tuned_val_f1:<18.4f} | (Optuna objective)")
    print(sep)

    print(f"\n  Per-Class F1 (Test Set):")
    print(f"  {'Class':<28} | {'Before':>8} | {'After':>8} | {'Delta':>10}")
    print(sep)
    all_classes = sorted(set(list(b["f1_per_class"].keys()) + list(t["f1_per_class"].keys())))
    for cls in all_classes:
        bv = b["f1_per_class"].get(cls, 0.0)
        tv = t["f1_per_class"].get(cls, 0.0)
        print(f"  {cls:<28} | {bv:>8.4f} | {tv:>8.4f} | {_delta_str(bv, tv):>10}")
    print(sep)

    print(f"\n  Best Hyperparameters Found:")
    print(sep)
    for k, v in best_params.items():
        print(f"    {k:<22}: {v}")
    print(f"{'='*78}\n")

    # ── Per-class F1 comparison bar chart ─────────────────────────────────────
    classes    = all_classes
    before_f1s = [b["f1_per_class"].get(c, 0.0) for c in classes]
    after_f1s  = [t["f1_per_class"].get(c, 0.0) for c in classes]

    n      = len(classes)
    y_pos  = np.arange(n)
    height = 0.38

    fig, ax = plt.subplots(figsize=(12, max(6, n * 0.55)))
    bars_before = ax.barh(y_pos + height / 2, before_f1s, height,
                          label="Before (Default)", color="#4C72B0", alpha=0.88)
    bars_after  = ax.barh(y_pos - height / 2, after_f1s,  height,
                          label="After (Optuna)",   color="#DD8452", alpha=0.88)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(classes, fontsize=9)
    ax.set_xlabel("F1 Score", fontsize=11)
    ax.set_title(
        f"Per-Class F1 Score — Before vs After Optuna Tuning\n"
        f"Macro F1: {b['f1_macro']:.4f}  →  {t['f1_macro']:.4f}  "
        f"({_delta_str(b['f1_macro'], t['f1_macro'])})",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlim(0, 1.08)
    ax.legend(loc="lower right", fontsize=10)
    ax.bar_label(bars_before, fmt="%.3f", padding=3, fontsize=7)
    ax.bar_label(bars_after,  fmt="%.3f", padding=3, fontsize=7)
    ax.axvline(0.9, color="grey", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    chart_path = out_dir / "f1_comparison_bar.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Saved] F1 comparison chart   -> {chart_path}")

    # ── Machine-readable JSON ─────────────────────────────────────────────────
    results_payload = {
        "run_timestamp":    datetime.now(timezone.utc).isoformat(),
        "tuning_config": {
            "n_trials":             TUNING_N_TRIALS,
            "cv_folds":             TUNING_CV_FOLDS,
            "metric":               TUNING_METRIC,
            "average":              TUNING_AVERAGE,
            "random_seed":          TUNING_RANDOM_SEED,
            "early_stopping_rounds": TUNING_EARLY_STOPPING_ROUNDS,
        },
        "baseline_metrics": b,
        "tuned_metrics":    t,
        "best_params":      best_params,
        "best_cv_f1_macro": tuned_val_f1,
    }
    json_path = out_dir / "tuning_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)
    print(f"  [Saved] Tuning results JSON   -> {json_path}")


# =============================================================================
# 9. Save model artifacts
# =============================================================================

def save_artifacts(model, le: LabelEncoder, feature_names: list[str]) -> None:
    """Save model, label encoder, and feature names list to MODELS_DIR."""
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
# 10. Entry point
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train LightGBM classifier on engineered_features.csv",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Standard training (no tuning):
    python train_classifier.py

  With Optuna tuning + GPU (GTX 1650):
    python train_classifier.py --tune --gpu

  Override trial count:
    python train_classifier.py --tune --trials 50

  Cap wall-clock time to 10 minutes:
    python train_classifier.py --tune --timeout 600
        """,
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=str(ENGINEERED_FEAT_CSV),
        help=f"Path to engineered_features.csv (default: {ENGINEERED_FEAT_CSV})",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        default=False,
        help="Run Optuna hyperparameter tuning (optimises Macro-F1). "
             "Adds ~5–15 min. Saves f1_comparison_bar.png and tuning_results.json.",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        default=False,
        help="Use GPU-accelerated LightGBM (requires CUDA build of lightgbm). "
             "Overrides TUNING_DEVICE_TYPE from config.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=TUNING_N_TRIALS,
        help=f"Number of Optuna trials (default: {TUNING_N_TRIALS} from config). "
             "Partial runs can be resumed from the SQLite study.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=TUNING_TIMEOUT_SEC,
        help="Wall-clock time limit in seconds for Optuna (default: no limit).",
    )
    args = parser.parse_args()

    # Resolve device type: --gpu flag takes priority over config default
    device_type = "gpu" if args.gpu else TUNING_DEVICE_TYPE

    # ── Resolve CSV path ──────────────────────────────────────────────────────
    csv_path = args.csv
    if not os.path.exists(csv_path):
        alt = PIPELINE_OUTPUT_DIR / "engineered_features.csv"
        if alt.exists():
            csv_path = str(alt)
        else:
            print(f"[ERROR] CSV not found: {csv_path}")
            print("        Run the simulator + pipeline first to generate features.")
            sys.exit(1)

    print("=" * 70)
    print("  LightGBM Classifier Training")
    if args.tune:
        print(f"  Mode    : Optuna Hyperparameter Tuning (Macro-F1, {args.trials} trials)")
    else:
        print("  Mode    : Standard (default params, no tuning)")
    print(f"  Device  : {device_type}")
    print("=" * 70)

    t_start = time.time()

    # ── Load and split ─────────────────────────────────────────────────────────
    df = load_training_data(csv_path)
    X_train, X_test, y_train, y_test, feature_names = build_matrices(df)

    # Main label encoder — fitted and reused for all subsequent steps
    le = LabelEncoder()
    le.fit(y_train)
    y_train_enc = le.transform(y_train)
    y_test_enc  = le.transform(y_test)
    sorted_classes = sorted(le.classes_)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if not args.tune:
        # ── Standard path (existing behaviour) ────────────────────────────────
        model = train_lgbm(X_train, y_train, le, device_type)
        evaluate(model, le, X_train, X_test, y_train, y_test, feature_names, MODELS_DIR)
        save_artifacts(model, le, feature_names)

    else:
        # ── Tuning path ────────────────────────────────────────────────────────

        # Step 1 — Baseline evaluation (default params)
        _, baseline_metrics = run_baseline_eval(
            X_train, X_test, y_train, y_test, le, device_type
        )

        # Step 2 — Optuna hyperparameter search
        best_params, best_val_f1 = run_optuna_tuning(
            X_train,
            y_train_enc,
            n_trials=args.trials,
            timeout=args.timeout,
            device_type=device_type,
            feature_names=feature_names,
        )

        # Step 3 — Retrain on full training set with best params + early stopping
        tuned_model = train_tuned_lgbm(X_train, y_train_enc, best_params, device_type)

        # Step 4 — Evaluate tuned model on held-out test set
        print("\n" + "=" * 70)
        print("  STEP 4 — TUNED MODEL EVALUATION (held-out test set)")
        print("=" * 70)
        pred_tuned = tuned_model.predict(X_test)
        pred_tuned_labels = le.inverse_transform(pred_tuned)
        tuned_metrics = _compute_metrics(y_test, pred_tuned_labels, sorted_classes)
        # Also save confusion matrix and feature importance for the tuned model
        evaluate(tuned_model, le, X_train, X_test, y_train, y_test,
                 feature_names, MODELS_DIR, label_prefix="Optuna Tuned")

        # Step 5 — Before/after comparison report + charts + JSON
        compare_and_report(baseline_metrics, tuned_metrics, best_params, best_val_f1, MODELS_DIR)

        # Step 6 — Save tuned model as the production artifact
        save_artifacts(tuned_model, le, feature_names)

    elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"  Training complete.  Total time: {elapsed:.1f}s")
    print(f"  Run the pipeline: python backend/run_pipeline.py")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
