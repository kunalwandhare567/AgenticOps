# Inference_langgraph Module

This package houses the stateful **10-Node LangGraph Inference Pipeline**.

---

## Architecture & Layout

```
Inference_langgraph/
├── graph.py               # Assembles StateGraph, nodes, and conditional edges
├── state.py               # Defines AIOpsLangState TypedDict state schema
├── README.md
│
├── Graph_node/            # LangGraph Node Wrappers (Function signatures matching state -> state)
│   ├── n01_collect.py
│   ├── n02_feature_engineering.py
│   ├── n03_prelim_severity.py
│   ├── n04_classify.py
│   ├── n05_tumbling_window.py
│   ├── n06_forecasting.py
│   ├── n07_severity_update.py
│   ├── n08_reliability.py
│   ├── n09_human_gate.py
│   └── n10_db_writer.py
│
└── nodes/                 # Domain Engines & Algorithms
    ├── classification/    # LightGBM 13-class model & Optuna tuning
    ├── collect/           # Telemetry queue bridge & SQLite cursor
    ├── db_writer/         # Centralised DbWriter SQLite engine
    ├── feature_engineering/ # Drain3 log parser & 32-feature extractor
    ├── forecasting/       # ARIMA & Exponential TTF forecasters
    ├── human_gate/        # Interrupt manager & approval state machine
    ├── preliminary_severity/# Threshold-based EMA severity engine
    ├── reliability/       # Weibull survival fitter & plotter
    ├── severity_update/   # Impact x Urgency matrix updater
    └── tumbling_window/   # 10-cycle majority-vote label smoother
```

---

## Node Pipeline Chain

1. **`n01_collect`**: Polls next raw tick from `LiveTelemetryQueue` or `live_feed_db.sqlite`.
2. **`n02_feature_engineering`**: Extracts 32 features (Drain3 log mining + metric rollups).
3. **`n03_prelim_severity`**: Computes baseline P1–P4 preliminary severity score.
4. **`n04_classify`**: Evaluates LightGBM model to predict active failure mode.
5. **`n05_tumbling_window`**: Applies 10-cycle majority-vote smoother to eliminate label jitter.
6. **`n06_forecasting`**: Forecasts Time-to-Failure (TTF) using mode-specific algorithms.
7. **`n07_severity_update`**: Adjusts severity using an Impact × Urgency matrix.
8. **`n08_reliability`**: Fits Weibull survival curve $S(t)$ per failure mode group.
9. **`n09_human_gate`**: Escalates P-level jumps to operator review queue.
10. **`n10_db_writer`**: Persists full cycle state to all 8 SQLite tables and triggers SSE broadcaster.
