# AIOps Backend — Stateful 10-Node LangGraph Inference Engine

## Architecture Overview

The backend is built around a stateful **10-Node LangGraph Pipeline** (`Inference_langgraph`), serving real-time telemetry processing, failure classification, time-to-failure forecasting, and Weibull reliability analysis.

```
Raw Telemetry (Live Feed / Queue)
          │
          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      LANGGRAPH 10-NODE PIPELINE                             │
│                                                                             │
│  [1] collect          Dequeue tick from LiveTelemetryQueue or SQLite        │
│         │                                                                   │
│  [2] feature_eng      Extract 32 features (Drain3 log mining + metrics)     │
│         │                                                                   │
│  [3] prelim_severity  DEVOPS P1-P4 preliminary severity (EMA + hysteresis)  │
│         │                                                                   │
│  [4] classify         LightGBM 13-class incident failure mode classifier     │
│         │                                                                   │
│  [5] tumbling_window  10-cycle majority-vote label smoother                 │
│         │                                                                   │
│  [6] forecasting      Mode-specific TTF forecasting (ARIMA / Exponential)   │
│         │                                                                   │
│  [7] severity_update  Impact × Urgency matrix severity updater              │
│         │                                                                   │
│  [8] reliability      Weibull S(t) survival probability fitter              │
│         │                                                                   │
│  [9] human_gate       Escalation review queue & auto-approve manager        │
│         │                                                                   │
│  [10] db_writer       Atomic 8-table SQLite write + SSE broadcast           │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
backend/
├── run_server.py                  # 🚀 Unified Backend Launcher (Ports 8080 & 8001)
├── run_langgraph.py               # LangGraph 10-Node Pipeline Runner
├── reset_all.py                   # System state reset utility
├── requirements.txt               # Backend Python dependencies
│
├── Inference_langgraph/           # Core LangGraph Pipeline Package
│   ├── graph.py                   # StateGraph assembly & node routing
│   ├── state.py                   # AIOpsLangState TypedDict & state schema
│   │
│   ├── Graph_node/                # LangGraph Node Wrappers (n01 to n10)
│   │   ├── n01_collect.py
│   │   ├── n02_feature_engineering.py
│   │   ├── n03_prelim_severity.py
│   │   ├── n04_classify.py
│   │   ├── n05_tumbling_window.py
│   │   ├── n06_forecasting.py
│   │   ├── n07_severity_update.py
│   │   ├── n08_reliability.py
│   │   ├── n09_human_gate.py
│   │   └── n10_db_writer.py
│   │
│   └── nodes/                     # Domain Computing Modules
│       ├── classification/        # LightGBM classifier & Optuna tuning
│       ├── collect/               # Queue bridge & DB collector
│       ├── db_writer/             # Centralised DbWriter SQLite engine
│       ├── feature_engineering/   # Drain3 log parser & 32-feature extractor
│       ├── forecasting/           # Mode-specific ARIMA forecasters
│       ├── human_gate/            # Interrupt manager & approval engine
│       ├── preliminary_severity/  # Threshold-based severity engine
│       ├── reliability/           # Weibull survival fitter & plotter
│       ├── severity_update/       # Impact x Urgency matrix updater
│       └── tumbling_window/       # 10-cycle majority-vote label smoother
│
├── Simulator/                     # Failure Signal Generators
│   ├── live_feed_simulator/       # Real-time telemetry & incident injector
│   └── app_data_generator_for_offline/ # Offline 120K-row dataset generator
│
├── api/                           # Sentinel Core API Server (Port 8080)
│   ├── main.py                    # REST + SSE endpoints & live process manager
│   ├── services.py                # Data access services
│   └── sse_broadcaster.py         # SSE broadcaster engine (500ms)
│
└── QueryLanggraph/                # AI Chatbot Assistant Engine (Port 8001)
    └── QueryLangGraph02-main/     # Guardrailed LLM query engine
```

---

## ⚡ Execution Commands

### Running the Backend Server (Terminal 1)
```powershell
python backend/run_server.py
```
This single command automatically starts:
1. **Sentinel Dashboard API** on `http://localhost:8080`
2. **QueryLangGraph Chatbot API** on `http://localhost:8001`

### Running the LangGraph Pipeline Independently
```powershell
# Run in live mode (consumes LiveTelemetryQueue & live_feed_db.sqlite)
python backend/run_langgraph.py --live

# Run in offline batch mode
python backend/run_langgraph.py

# Print pipeline Mermaid diagram and exit
python backend/run_langgraph.py --diagram
```

### Running Model Training & Fitters
```powershell
# Train LightGBM classifier with Optuna tuning
python backend/Inference_langgraph/nodes/classification/train_classifier.py --tune --gpu

# Fit 4-group Weibull reliability parameters
python backend/Inference_langgraph/nodes/reliability/run_weibull_fitter.py
```
