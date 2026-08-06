# Sentinel AIOps: Incident Detection & Mitigation Platform

An end-to-end AI-based incident prediction, classification, and mitigation platform. Sentinel ingests raw telemetry (metrics, logs, traces), processes it through a stateful **10-Node LangGraph AIOps Pipeline**, evaluates incident severity, forecasts Time-To-Failure (TTF) using mode-specific algorithms (ARIMA, Exponential Smoothing), assesses system survival probability using 4-group Weibull reliability analysis, and presents real-time health data on a modern React-based NOC Dashboard.

---

## 🏗️ System Architecture

```
                  ┌──────────────────────────────────────────────────────────┐
                  │                 NOC Front-end Dashboard                  │
                  │              (React + Vite + Tailwind CSS)               │
                  └────────────────────────────┬─────────────────────────────┘
                                               │ HTTP APIs (port 8080)
                                               ▼
                  ┌──────────────────────────────────────────────────────────┐
                  │                  FastAPI Web Server                      │
                  │                   (backend/api/main.py)                  │
                  └────────────────────────────┬─────────────────────────────┘
                                               │ Reads pipeline & telemetry
                                               ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│              10-NODE AIOps LangGraph PIPELINE (backend/run_langgraph.py)     │
│                                                                              │
│  [1] collect          Dequeue next telemetry row from simulator_db.sqlite    │
│         │                                                                    │
│  [2] feature_eng      Derive 32 features (Drain3 log + metrics)              │
│         │                                                                    │
│  [3] prelim_severity  Threshold-based P1–P4 severity (EMA + hysteresis)      │
│         │                                                                    │
│  [4] classify         LightGBM 13-class incident classifier                  │
│         │                                                                    │
│  [5] tumbling_window  10-cycle majority-vote label smoother                  │
│         │                                                                    │
│  [6] forecasting      Mode-specific TTF forecast (every N cycles)            │
│         │                                                                    │
│  [7] severity_update  Revised severity (Impact × Urgency matrix)             │
│         │                                                                    │
│  [8] reliability      Weibull S(t) survival probability                       │
│         │                                                                    │
│  [9] human_gate       Escalation/De-escalation with auto-approve timeout     │
│         │                                                                    │
│  [10] db_writer       Atomic write to all 8 pipeline DB tables + CSVs        │
└──────────────────────────────────────────────────────────────────────────────┘
                                               ▲
                                               │  simulator_db.sqlite (shared)
                                               │
                  ┌──────────────────────────────────────────────────────────┐
                  │   Telemetry Simulator  (run_simulator.py / live feed)    │
                  │   Generates 13 failure modes × 120 steps × 77 episodes  │
                  │   ≥ 120,000 rows per full run                            │
                  └──────────────────────────────────────────────────────────┘
```

---

## 📂 Repository Structure

```
AIOps_incident_management/
├── backend/
│   ├── api/                               # FastAPI REST + WebSocket server
│   ├── app_data_generator/
│   │   ├── config.py                      # ALL constants — no hardcoding anywhere
│   │   ├── run_simulator.py               # Offline 120 K-row simulator
│   │   └── storage/
│   │       ├── db_writer.py               # DbWriter: raw + pipeline DB writes
│   │       └── schema.sql                 # Full SQLite schema (8 tables)
│   ├── langgraph_pipeline/                # ← NEW LangGraph orchestration
│   │   ├── state.py                       # AIOpsLangState TypedDict + Pydantic validation
│   │   ├── graph.py                       # StateGraph assembly + routing
│   │   └── nodes/
│   │       ├── n01_collect.py             # DB polling collect
│   │       ├── n02_feature_engineering.py # Drain3 + metric feature derivation
│   │       ├── n03_prelim_severity.py     # Threshold-based severity
│   │       ├── n04_classify.py            # LightGBM classifier
│   │       ├── n05_tumbling_window.py     # Majority-vote smoother
│   │       ├── n06_forecasting.py         # TTF forecast (N-cycle throttle)
│   │       ├── n07_severity_update.py     # Impact × Urgency severity matrix
│   │       ├── n08_reliability.py         # Weibull S(t) survival probability
│   │       ├── n09_human_gate.py          # Escalation / de-escalation gate
│   │       └── n10_db_writer.py           # Centralised DB + CSV writes
│   ├── nodes/                             # Individual node implementations
│   │   ├── classification/
│   │   │   ├── train_classifier.py        # Optuna tuning + LightGBM training
│   │   │   ├── classifier.py              # Inference wrapper
│   │   │   └── models/                    # lgbm_model.pkl, label_encoder.pkl
│   │   ├── forecasting/
│   │   │   ├── router.py                  # Dispatches to mode-specific forecasters
│   │   │   ├── buffer.py                  # Per-episode feature buffer
│   │   │   └── modes/                     # 13 dedicated forecast functions
│   │   ├── reliability/
│   │   │   ├── weibull_fitter.py          # 4-group MLE Weibull fitting
│   │   │   ├── run_weibull_fitter.py      # Offline fitting CLI + sidecar JSON write
│   │   │   └── weibull_params.json        # ← Sidecar read by n08_reliability per-cycle
│   │   ├── human_gate/
│   │   │   ├── escalation_detector.py     # Detects P-level escalations
│   │   │   ├── review_builder.py          # Builds HumanReviewRequest objects
│   │   │   ├── interrupt_manager.py       # SQLite-backed review queue
│   │   │   └── approval_engine.py         # State machine + ApprovalResult
│   │   ├── feature_engineering/
│   │   ├── preliminary_severity/
│   │   ├── severity_update/
│   │   └── tumbling_window/
│   ├── reset_all.py                       # ← NEW clean-slate reset script
│   ├── run_langgraph.py                   # ← NEW LangGraph pipeline runner
│   ├── run_pipeline.py                    # Legacy pipeline runner (kept)
│   └── requirements.txt
│
├── frontend/                              # React + Vite + Tailwind dashboard
│   ├── src/
│   ├── package.json
│   └── vite.config.js
└── README.md
```

---

## 🛠️ Installation & Setup

### Prerequisites
- **Python**: `3.10+`
- **Node.js**: `v18+` & `npm`
- **GPU (optional)**: NVIDIA GTX 1650+ for accelerated LightGBM tuning

### Backend Setup

```bash
cd backend
pip install -r requirements.txt
```

Key dependencies: `lightgbm`, `optuna`, `langgraph≥1.0`, `pydantic≥2`, `fastapi`, `pmdarima`, `lifelines`, `reliability`, `drain3`

### Frontend Setup

```bash
cd frontend
npm install
```

---

## 🚀 Running the Platform

### Mode A — Initial Full Run (train + process 120K+ rows)

**Step 1 — Clean slate (optional but recommended)**
```bash
# Terminal (backend dir)
python reset_all.py
# Keep trained model if already done:
# python reset_all.py --keep-model
```

**Step 2 — Generate 120K+ telemetry rows**
```bash
# Terminal 1
python app_data_generator/run_simulator.py --speed 50
# ~120,120 rows (77 eps × 120 steps × 13 modes). Speed=50 takes ~5 min.
```

**Step 3 — Train the LightGBM classifier (run ONCE)**
```bash
# Terminal 2 (after simulator finishes ~30K rows minimum)
python nodes/classification/train_classifier.py --tune --gpu
# GPU accelerated Optuna tuning — 80 trials, 5-fold CV, Macro-F1 ≈ 0.989
# Saves: nodes/classification/models/lgbm_model.pkl
```

**Step 4 — Run the Weibull reliability fitter (offline batch)**
```bash
# Terminal 2
python nodes/reliability/run_weibull_fitter.py
# Requires life_data_extracted.csv (populated by n08_reliability during pipeline)
# Writes: nodes/reliability/weibull_params.json  ← read by n08 per-cycle
```

**Step 5 — Run the LangGraph pipeline**
```bash
# Terminal 2
python run_langgraph.py
# Options:
#   --speed 50    fast replay (50 cycles/sec)
#   --verbose     print per-node output
#   --diagram     print Mermaid graph and exit
```

**Step 6 — FastAPI server**
```bash
# Terminal 3
python api/main.py
```

**Step 7 — React dashboard**
```bash
# Terminal 4
cd ../frontend
npm run dev
# Open: http://localhost:5173
```

---

### Mode B — Live Feed (real-time incident injection)

After the initial run, you can continuously inject new incidents without resetting.

**Terminal 1 — Keep the LangGraph pipeline running**
```bash
# (already running from Mode A Step 5)
python run_langgraph.py
```

**Terminal 2 — Inject live incidents**
```bash
python app_data_generator/run_simulator.py --speed 1 --episodes 5
# Each simulator run appends new rows to the existing DB.
# The pipeline picks them up automatically via DB polling.
```

> Every new telemetry row from the simulator flows through the same 10-node
> pipeline. The Reliability node records each episode's TTF to
> `life_data_extracted.csv` — re-running `run_weibull_fitter.py` after
> accumulating more live events updates the fitted Weibull parameters and
> the sidecar JSON, which the pipeline reads on the next cycle.

---

## 📊 Pipeline Nodes — Detail

| Node | File | Responsibilities | Algorithm |
|------|------|-----------------|-----------|
| **1 collect** | `n01_collect.py` | Poll DB for next unprocessed row | SQLite cursor + `last_processed_id` |
| **2 feature_eng** | `n02_feature_engineering.py` | 27 metric + 5 Drain3 log features | Drain3 template mining, rolling stats |
| **3 prelim_severity** | `n03_prelim_severity.py` | Threshold-based P1–P4 | DEVOPS SeverityEngine (EMA + hysteresis) |
| **4 classify** | `n04_classify.py` | 13-class incident detection | LightGBM + Optuna tuning, Macro-F1 ≈ 0.989 |
| **5 tumbling_window** | `n05_tumbling_window.py` | Smooth label flicker | 10-cycle majority-vote |
| **6 forecasting** | `n06_forecasting.py` | TTF estimation | ARIMA / Exponential Smoothing (mode-specific, every N cycles) |
| **7 severity_update** | `n07_severity_update.py` | Revised severity | Impact × Urgency matrix + 5-cycle hysteresis |
| **8 reliability** | `n08_reliability.py` | Survival probability | Weibull S(t) = exp(-(t/η)^β) |
| **9 human_gate** | `n09_human_gate.py` | Escalation / de-escalation | SQLite review queue, auto-approve on timeout |
| **10 db_writer** | `n10_db_writer.py` | Persist full cycle state | DbWriter → 8 SQLite tables + 6 CSVs |

---

## 🔬 Classification Model — Hyperparameter Tuning

The LightGBM classifier is tuned via **Optuna TPE** with **5-fold stratified CV**:

| Parameter | Search Range | Reason |
|-----------|-------------|--------|
| `n_estimators` | 200–2000 | Controls model capacity |
| `learning_rate` | 0.005–0.3 | Step size for gradient boosting |
| `max_depth` | 4–12 | Tree complexity |
| `num_leaves` | 15–200 | Fine-grained leaf capacity |
| `min_child_samples` | 10–200 | Prevents overfitting on small classes |
| `subsample` | 0.5–1.0 | Row sampling per tree |
| `colsample_bytree` | 0.5–1.0 | Feature sampling per tree |
| `reg_alpha` | 1e-8–10 | L1 regularisation |
| `reg_lambda` | 1e-8–10 | L2 regularisation |

- **Objective**: Maximise **Macro F1** (equal weight across all 13 failure modes)
- **80 Optuna trials** with early stopping (50 rounds patience)
- **GPU acceleration** via `device_type="gpu"` when `--gpu` flag is used
- **Convergence**: Macro-F1 ≈ **0.989** at best trial

---

## ⚗️ Reliability Analysis — Weibull Groups

| Group | Failure Modes | β (shape) | η (scale/seconds) |
|-------|-------------|-----------|-------------------|
| Immediate trigger | BAD_DEPLOY, CASCADING_FAILURE, ERROR_STORM | ~23 | ~10s |
| Fast accumulation | CPU_SATURATION, DB_SLOWDOWN, DISK_IO_SATURATION | ~4.3 | ~15s |
| Progressive degradation | MEMORY_LEAK, LATENCY_SPIKE, QUEUE_BACKUP | ~2.0 | ~47s |
| Slow/latent degradation | RETRY_STORM, CACHE_STAMPEDE, DEPENDENCY_TIMEOUT | ~7.5 | ~373s |

Parameters are fitted offline using **4-group censored MLE Weibull** (via `lifelines`/`reliability`).
Survival probability S(t) is computed per-cycle: `S(t) = exp(-(t/η)^β) × 100`.

---

## 🚦 Human Gate — Escalation & De-escalation Logic

The Human Gate (`n09_human_gate.py`) implements a full review state machine:

```
           WAITING → REVIEWING → APPROVED   → COMPLETED
                               → REJECTED   → COMPLETED
                               → AUTO_APPROVED → COMPLETED
```

**Escalation** (P-level decreases, i.e., severity worsens):
1. `EscalationDetector.needs_review()` detects the upgrade.
2. A `HumanReviewRequest` is posted to the `pending_reviews` SQLite table.
3. The FastAPI UI exposes `/api/human-gate/pending` for operators to review.
4. `InterruptManager.poll_for_decision()` blocks until decision or timeout.
5. Auto-approve fires after `HUMAN_GATE_TIMEOUT_SECONDS` (default: 2s for demo; increase for production).

**De-escalation** (P-level increases, i.e., severity improves):
1. `SeverityUpdater.HysteresisTracker` requires 5 consecutive lower-severity cycles before de-escalating.
2. Once de-escalation is confirmed, the gate commits the new severity immediately — **no human review required**.
3. Logged with `decision = "DE_ESCALATED"`, `operator = "system"`.

**No-op** (severity unchanged):
- Gate passes through silently with `hg_needed = False`.

---

## 📈 Output Files

| File | Location | Description |
|------|----------|-------------|
| `simulator_db.sqlite` | `app_data_generator/output/` | Raw telemetry + all pipeline tables |
| `engineered_features.csv` | `nodes/feature_engineering/output/` | 32-feature FE output |
| `preliminary_severity.csv` | `nodes/preliminary_severity/output/` | Per-cycle P1–P4 scores |
| `pipeline_results.csv` | `nodes/classification/output/` | Combined snapshot |
| `tumbling_window_output.csv` | `nodes/tumbling_window/output/` | Vote distribution |
| `forecasting_output.csv` | `nodes/forecasting/output/` | TTF per cycle |
| `severity_update_output.csv` | `nodes/severity_update/output/` | Revised severity |
| `human_gate_output.csv` | `nodes/human_gate/output/` | Gate decisions |
| `human_gate_audit.db` | `nodes/human_gate/output/` | Full review history |
| `life_data_extracted.csv` | `nodes/reliability/output/` | TTF events for Weibull re-fitting |
| `weibull_params.json` | `nodes/reliability/` | Fitted β/η per group (sidecar) |
| `lgbm_model.pkl` | `nodes/classification/models/` | Trained LightGBM model |

---

## 🔧 Configuration (config.py)

All constants live in `backend/app_data_generator/config.py`. Key settings:

| Constant | Default | Description |
|----------|---------|-------------|
| `EPISODES_PER_MODE` | 77 | Episodes per failure mode (77 × 120 × 13 = 120,120 rows) |
| `POLL_INTERVAL_MS` | 100 | ms between DB polls when no data is ready |
| `FORECAST_EVERY_N_CYCLES` | 10 | Run forecasting once every N cycles |
| `HUMAN_GATE_TIMEOUT_SECONDS` | 2 | Auto-approve timeout (increase for production) |
| `TUNING_N_TRIALS` | 80 | Optuna tuning trials |
| `TUNING_CV_FOLDS` | 5 | Stratified CV folds |
| `LANGGRAPH_VERBOSE` | False | Print per-node state dicts |

---

## 💡 Troubleshooting

| Problem | Solution |
|---------|----------|
| `FileNotFoundError: lgbm_model.pkl` | Run `python nodes/classification/train_classifier.py --tune --gpu` first |
| `FileNotFoundError: life_data_extracted.csv` | Run pipeline for at least one episode before running Weibull fitter |
| `weibull_params.json not found` | Run `python nodes/reliability/run_weibull_fitter.py` |
| Pipeline picks up no data | Make sure `run_simulator.py` is running and has written at least 1 row |
| DB locked error | Close any SQLite browser tool; WAL mode handles concurrent reads fine |
| Reset everything | `python reset_all.py` (use `--keep-model` to preserve trained classifier) |

---

## 🧪 Validating the Graph

```bash
# Print Mermaid diagram of the StateGraph and exit
python run_langgraph.py --diagram

# Build graph without processing data
python run_langgraph.py --dry-run
```
