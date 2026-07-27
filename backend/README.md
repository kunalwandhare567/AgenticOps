# AIOps Backend — 7-Stage Failure Detection Pipeline

## Architecture Overview

```
Telemetry Data (simulator_db.sqlite)
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        PIPELINE (7 Stages)                          │
│                                                                     │
│  [1] Collect          Raw metrics + logs + traces from simulator    │
│         │                                                           │
│  [2] Feature Eng.     27 metric features + 5 Drain3 log features   │
│         │                                                           │
│  [3] Prelim Severity  Threshold-based P1/P2/P3/P4 scoring          │
│         │                                                           │
│  [4] Classification   LightGBM failure mode detector               │
│         │                                                           │
│  [5] Tumbling Window  10-cycle majority-vote smoother               │
│         │                                                           │
│  [6] Forecasting      ARIMA/pmdarima TTF + confidence (7 algos)    │
│         │                                                           │
│  [7] Severity Update  Urgency-adjusted revised severity             │
└─────────────────────────────────────────────────────────────────────┘
```

## Folder Structure

```
backend/
├── README.md
├── requirements.txt
├── run_pipeline.py                  # Live pipeline (Stages 1-5, real-time)
│
├── app_data_generator/              # Simulator — generates failure episode data
│   ├── config.py                   # All paths, constants, column definitions
│   ├── state.py                    # PipelineState dataclass
│   ├── run_simulator.py            # Entry point: generates simulator_db.sqlite
│   ├── generators/                 # Metric / log / trace signal generators
│   ├── physics/                    # Failure degradation physics models
│   ├── storage/                    # SQLite DB writer helpers
│   └── output/                     # simulator_db.sqlite (generated)
│
└── nodes/
    ├── collect/                     # Stage 1: Telemetry Collector
    │   └── queue_bridge.py
    │
    ├── feature_engineering/         # Stage 2: Feature Engineering
    │   ├── orchestrator.py
    │   ├── metrics_features.py
    │   ├── log_features.py
    │   ├── run_feature_engineering.py   # Offline batch runner
    │   └── output/
    │       └── engineered_features.csv
    │
    ├── preliminary_severity/        # Stage 3: Preliminary Severity
    │   ├── severity_node.py
    │   ├── severity.py
    │   ├── run_severity.py              # Offline batch runner
    │   └── output/
    │       └── preliminary_severity.csv
    │
    ├── classification/              # Stage 4: LightGBM Classifier
    │   ├── classifier.py
    │   ├── train_classifier.py
    │   ├── train_drain.py
    │   ├── models/                  # Trained model artifacts
    │   │   ├── lgbm_model.pkl
    │   │   ├── label_encoder.pkl
    │   │   └── ...
    │   └── output/
    │       └── pipeline_results.csv
    │
    ├── tumbling_window/             # Stage 5: Majority-Vote Smoother
    │   ├── tumbling_window.py
    │   ├── run_classification_window.py # Offline batch runner
    │   └── output/
    │       └── tumbling_window_output.csv
    │
    ├── forecasting/                 # Stage 6: ARIMA/pmdarima TTF Forecaster
    │   ├── router.py                # Mode-to-algorithm routing
    │   ├── algorithms.py            # ARIMA, exponential, linear algorithms
    │   ├── thresholds.py            # Critical thresholds per failure mode
    │   ├── feature_lookup.py        # Feature alias table
    │   ├── buffer.py                # Rolling history buffer
    │   ├── base_forecaster.py       # Base class
    │   ├── _mode_runner.py          # Shared engine for all 12 modes
    │   ├── modes/                   # One file per failure mode (12 total)
    │   ├── run_forecasting.py       # Offline batch runner
    │   ├── visualize_forecasting.py # Chart generator (6 chart types)
    │   └── output/
    │       ├── forecasting_output.csv
    │       └── forecasting_plots/
    │
    └── severity_update/             # Stage 7: Revised Severity
        ├── bands.py                 # Step 1: Impact band + Step 2: Gated urgency
        ├── matrix.py                # Step 3: Fixed 4x3 combination matrix
        ├── hysteresis.py            # Step 4: Escalation & dwell de-escalation
        ├── updater.py               # Orchestrates Steps 1-4
        ├── run_severity_update.py   # Offline batch runner
        ├── tests/
        │   └── test_severity_update.py
        └── output/
            └── severity_update_output.csv
```

## Quick Start — Full Pipeline (Offline)

### 1. Setup
```powershell
cd d:\AIOps_Incident_Management\backend
pip install -r requirements.txt
```

### 2. Generate Data (if not already done)
```powershell
python app_data_generator/run_simulator.py
```

### 3. Run Each Stage in Order
```powershell
# Stage 2: Feature Engineering
python nodes/feature_engineering/run_feature_engineering.py

# Stage 3: Preliminary Severity
python nodes/preliminary_severity/run_severity.py

# Stage 4: Train Classifier (first time only)
python nodes/classification/train_classifier.py

# Stage 5: Tumbling Window
python nodes/tumbling_window/run_classification_window.py

# Stage 6: Forecasting (fast mode ~10s)
python nodes/forecasting/run_forecasting.py --fast

# Stage 7: Severity Update
python nodes/severity_update/run_severity_update.py
```

### 4. Visualize Forecasting Results
```powershell
python nodes/forecasting/visualize_forecasting.py
```

### 5. Run Unit Tests
```powershell
python nodes/severity_update/tests/test_severity_update.py
```

## Live Pipeline (Real-time)

Run both terminals simultaneously:

```powershell
# Terminal 1: Data generator
python app_data_generator/run_simulator.py

# Terminal 2: Pipeline (Stages 1-5)
python run_pipeline.py
```

## Stage Details

| Stage | Node | Input | Output | Algorithm |
|-------|------|-------|--------|-----------|
| 1 | Collect | simulator_db.sqlite | TelemetryQueue | SQLite polling |
| 2 | Feature Eng. | Raw metrics + logs | engineered_features.csv | Drain3 + rolling stats |
| 3 | Prelim Severity | Engineered features | preliminary_severity.csv | Threshold scoring |
| 4 | Classification | Engineered features | pipeline_results.csv | LightGBM |
| 5 | Tumbling Window | Classifications | tumbling_window_output.csv | Majority vote (10-cycle) |
| 6 | Forecasting | pipeline_results.csv | forecasting_output.csv | pmdarima / statsmodels ARIMA |
| 7 | Severity Update | forecasting_output.csv | severity_update_output.csv | Impact x Urgency matrix |

## Severity Update Matrix (Stage 7)

```
                Urgency
Impact    │  Imminent  │   Near    │  Distant
──────────┼────────────┼───────────┼──────────
High      │     P1     │    P1     │    P2
Moderate  │     P1     │    P2     │    P3
Low       │     P2     │    P3     │    P4
None      │     P3     │    P4     │    P4
```

Gate: confidence >= 0.75 AND ttf_source NOT IN {not_applicable, rollback_decision}
If gate fails → Urgency forced to Distant.
