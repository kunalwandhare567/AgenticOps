# Sentinel AIOps – Full Project Documentation

> **Version:** 1.0.0 · **Date:** August 2026 · **Author:** Kunal Wandhare  
> **Repository:** `kunalwandhare567/AgenticOps`

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [Existing System / Current Challenges](#3-existing-system--current-challenges)
4. [Proposed Solution](#4-proposed-solution)
5. [Project Objectives](#5-project-objectives)
6. [Project Scope](#6-project-scope)
7. [Expected Impact & Benefits](#7-expected-impact--benefits)
8. [High-Level System Architecture](#8-high-level-system-architecture)
9. [End-to-End System Flow](#9-end-to-end-system-flow)
10. [Detailed Module Architecture](#10-detailed-module-architecture)
11. [Folder Structure](#11-folder-structure)
12. [Technology Stack](#12-technology-stack)
13. [Backend Architecture](#13-backend-architecture)
14. [Frontend Architecture](#14-frontend-architecture)
15. [Router & API Architecture](#15-router--api-architecture)
16. [Complete API Endpoint Documentation](#16-complete-api-endpoint-documentation)
17. [API Request–Response Flow](#17-api-requestresponse-flow)
18. [Database Architecture & Data Models](#18-database-architecture--data-models)
19. [Telemetry Architecture](#19-telemetry-architecture)
20. [Metrics Collection & Processing](#20-metrics-collection--processing)
21. [Log Collection & Processing](#21-log-collection--processing)
22. [Distributed Tracing](#22-distributed-tracing)
23. [Telemetry Correlation](#23-telemetry-correlation)
24. [Data Preprocessing & Feature Engineering](#24-data-preprocessing--feature-engineering)
25. [ML Pipeline](#25-ml-pipeline)
26. [Failure Mode Classification](#26-failure-mode-classification)
27. [LightGBM Model](#27-lightgbm-model)
28. [Tumbling Window](#28-tumbling-window)
29. [Model Training & Evaluation](#29-model-training--evaluation)
30. [Root Cause Analysis (RCA)](#30-root-cause-analysis-rca)
31. [Correlation Engine](#31-correlation-engine)
32. [LangGraph / Agent Workflow](#32-langgraph--agent-workflow)
33. [LLM Explanation Layer – QueryLangGraph](#33-llm-explanation-layer--querylanggraph)
34. [Incident Detection & Management](#34-incident-detection--management)
35. [Incident Lifecycle](#35-incident-lifecycle)
36. [Frontend Components & UI](#36-frontend-components--ui)
37. [Reliability & Survival Analysis](#37-reliability--survival-analysis)
38. [Human-in-the-Loop Gate](#38-human-in-the-loop-gate)
39. [Live Feed Simulation](#39-live-feed-simulation)
40. [Forecasting Engine](#40-forecasting-engine)
41. [Severity Update & Hysteresis](#41-severity-update--hysteresis)
42. [Outputs & Results](#42-outputs--results)
43. [Running the Project](#43-running-the-project)
44. [Conclusion](#44-conclusion)

---

## 1. Project Overview

**Sentinel AIOps** is an AI-powered, agent-driven AIOps (Artificial Intelligence for IT Operations) platform designed to automatically detect, diagnose, classify, and manage production infrastructure incidents. It replaces slow, manual on-call processes with a real-time autonomous pipeline that processes telemetry signals (metrics, logs, distributed traces), identifies anomalies, predicts failure trajectories, and generates actionable incident summaries.

The platform has three primary modes of operation:

| Mode | Description |
|------|-------------|
| **Historical Pipeline** | Processes pre-generated simulator data for training, evaluation, and replay |
| **Live Feed Mode** | Streams real-time simulated telemetry through the full AI pipeline |
| **Chat Query Mode** | Enables natural-language conversation with the incident database via an LLM agent |

The system is built as a multi-service architecture, consisting of a Python FastAPI backend, a React/Vite frontend dashboard, and two separate LangGraph AI pipelines (one for inference, one for query answering).

---

## 2. Problem Statement

Modern cloud-native infrastructures operate as complex, highly distributed systems. Services communicate via APIs, depend on shared databases and caches, and expose thousands of metrics per second. When something goes wrong:

- **Alert storms** fire hundreds of Prometheus/Datadog alerts per minute, overwhelming on-call engineers.
- **Manual triage** requires experienced engineers to mentally correlate CPU spikes, error rate jumps, GC pauses, and slow DB queries from 5+ dashboards simultaneously.
- **Mean Time To Detect (MTTD)** is typically 15–45 minutes for complex cascading failures.
- **Mean Time To Resolve (MTTR)** averages 30–120 minutes due to manual root-cause investigation.
- **Severity misclassification** leads to either under-reaction (P4 incidents actually being P1 cascading failures) or false-alarm fatigue (P1 alerts for transient spikes).
- **Prediction blindspot**: No existing tool tells engineers _when_ a failure will breach a critical threshold, only _that_ it has already breached one.

The core problem is that **incident management requires a human expert to correlate multi-dimensional telemetry, classify failure modes, predict severity, and decide escalation paths** — all in real time under pressure.

---

## 3. Existing System / Current Challenges

| Challenge | Impact |
|-----------|--------|
| Rule-based alerting (Prometheus rules) | High false-positive rate; misses multi-dimensional failures |
| Manual log triage (Kibana/Splunk) | 10–20 min per incident before root-cause is narrowed |
| No time-to-failure prediction | Teams react after threshold breach, not before |
| Static severity mapping | P1/P2/P3 based on a single metric, ignoring compound effects |
| No cross-signal correlation | CPU, memory, DB latency, and log errors are analyzed in silos |
| No automated escalation | Severity decisions require manual approval even for obvious P1 events |
| No audit trail for decisions | Post-mortem analysis is manual and error-prone |
| No natural-language query interface | Engineers must write SQL or learn dashboard query languages |

---

## 4. Proposed Solution

Sentinel AIOps addresses all the above challenges with a layered AI pipeline:

1. **Telemetry Simulator**: Generates realistic multi-signal telemetry for 13 distinct failure modes across 4 microservices.
2. **Feature Engineering**: Normalizes raw signals into a 37-feature vector per cycle.
3. **Preliminary Severity**: Rule-based, fast P1–P4 triage using threshold checks on 5 critical metrics.
4. **LightGBM Classification**: Multi-class ML model identifies the exact failure mode from the feature vector.
5. **Tumbling Window**: 10-cycle majority voting smooths noisy per-cycle predictions into stable labels.
6. **Time-to-Failure Forecasting**: Mode-specific ARIMA/exponential/linear extrapolators predict _when_ a threshold will be breached.
7. **Severity Update with Hysteresis**: Cross-references forecast urgency with impact to produce a final P1–P4 that is stabilized against flapping via a dwell counter.
8. **Weibull Reliability Analysis**: Stratified survival analysis quantifies historical failure probability over time.
9. **Human Gate**: Auto-escalation with operator approval workflow for large severity jumps.
10. **DB Writer + SSE**: Atomic database persistence and real-time Server-Sent Events push to the frontend.
11. **QueryLangGraph Chatbot**: LLM-powered agent answers natural-language questions about incidents.

---

## 5. Project Objectives

1. **Automate failure detection** across 13 cloud-native failure modes with ≥95% classification accuracy.
2. **Reduce MTTD** from 15–45 minutes (manual) to under 10 seconds (automated pipeline cycle).
3. **Provide actionable time-to-failure (TTF) prediction** so engineers can intervene before a threshold is breached.
4. **Deliver a compound severity score** (P1–P4) that accounts for current impact + forecast urgency + historical reliability.
5. **Implement a human-in-the-loop gate** for large severity jumps, ensuring operator accountability.
6. **Maintain a full audit trail** of every decision, classification, forecast, and severity change in SQLite.
7. **Enable natural-language queries** over the incident database using a guarded LLM agent pipeline.
8. **Provide a real-time dashboard** that displays live pipeline state via Server-Sent Events.
9. **Support live simulation** mode that demonstrates the end-to-end pipeline on streaming telemetry.
10. **Build modular, extensible architecture** using LangGraph StateGraph so new nodes can be added without touching existing code.

---

## 6. Project Scope

**In Scope:**
- Telemetry simulation for 13 failure modes across 4 microservices
- 10-node LangGraph inference pipeline (collect → feature_eng → prelim_severity → classify → tumbling_window → forecasting → severity_update → reliability → human_gate → db_writer)
- Full SQLite database schema with 10 tables
- Kaplan-Meier + 2-Parameter Weibull survival analysis
- FastAPI REST + SSE backend at port 8080
- QueryLangGraph chatbot API at port 8001
- React/Vite frontend with real-time SSE rendering
- Live feed simulator with WAL-mode SQLite database
- Human Gate approval workflow with timeout auto-approval

**Out of Scope:**
- Production Kubernetes deployment
- Real Prometheus/OpenTelemetry integration
- Multi-tenant support
- Authentication/authorization layer
- GPU-accelerated model inference
- External alerting integration (PagerDuty, Slack, OpsGenie)

---

## 7. Expected Impact & Benefits

| Metric | Before Sentinel | With Sentinel |
|--------|----------------|---------------|
| Detection Time (MTTD) | 15–45 min | < 10 seconds |
| Classification Accuracy | N/A (manual) | ≥ 95% (LightGBM) |
| False Positive Rate | ~40% (rule-based) | < 5% (ML + tumbling window) |
| Severity Stability | Flaps per metric | Hysteresis-controlled |
| Time-to-Failure Visibility | None | Per-mode ARIMA forecast |
| Operator Accountability | Manual notes | Full audit trail in SQLite |
| Incident Query Access | SQL/PromQL required | Natural language via LLM |
| Live Monitoring | Static dashboards | Real-time SSE push (500ms) |

---

## 8. High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        SENTINEL AIOPS PLATFORM                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌─────────────────┐          ┌──────────────────────────────────────┐ │
│   │  SIMULATOR      │          │   INFERENCE LANGGRAPH PIPELINE        │ │
│   │                 │          │                                      │ │
│   │ live_feed /     │─SQLite──▶│  n01→n02→n03→n04→n05→n06→n07→n08   │ │
│   │ offline sim     │  WAL     │  →n09→n10 (10-node StateGraph)      │ │
│   └─────────────────┘          └────────────────┬─────────────────────┘ │
│                                                 │                       │
│                                   SSEBroadcaster│ SQLite writes          │
│                                                 │                       │
│   ┌─────────────────────────────────────────────▼─────────────────────┐ │
│   │             FASTAPI CORE API  (port 8080)                         │ │
│   │  /api/live-stream (SSE)  /api/episodes  /api/human-gate           │ │
│   │  /api/live/start         /api/live/stop  /api/reliability         │ │
│   └─────────────────────────────────┬─────────────────────────────────┘ │
│                                     │ REST / SSE                        │
│   ┌─────────────────────────────────▼─────────────────────────────────┐ │
│   │           REACT / VITE FRONTEND DASHBOARD (port 5173)             │ │
│   │  TopHeader │ Sidebar │ SummaryCard │ ReliabilityCard │ ChatPanel  │ │
│   │  HumanGatePanel │ MultiTrendCharts │ ForecastChart │ AIOpsChat   │ │
│   └────────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │         QUERYLANGGRAPH CHATBOT API  (port 8001)                 │   │
│   │  parse_guard→parse_query→validation→intent_router→retrieval     │   │
│   │  →sufficiency_router→synthesis_guard→synthesis→respond          │   │
│   └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 9. End-to-End System Flow

This section traces a single telemetry tick through the entire system, from data generation to dashboard display.

### Step 1 — Telemetry Generation

The **Live Feed Simulator** (`Simulator/live_feed_simulator/run_live_feed.py`) begins emitting synthetic microservice telemetry at 0.5-second intervals. Each tick represents a 2-second window of production metrics for one of four services:

- `auth-service`, `payment-service`, `order-service`, `inventory-service`

Each service can operate in one of 13 failure modes (e.g., `MEMORY_LEAK`, `CPU_SATURATION`, `CASCADING_FAILURE`, or `NONE` for healthy). The simulator writes one row to three SQLite tables: `metrics`, `logs`, `traces`, using **WAL (Write-Ahead Logging) mode** to allow concurrent reads by the pipeline.

### Step 2 — Trigger via Frontend

The operator clicks the **"Start Live Feed"** button in the React dashboard (`TopHeader.jsx`). This sends:

```
POST http://localhost:8080/api/live/start
```

The FastAPI main server starts two background subprocesses:
1. `Simulator/live_feed_simulator/run_live_feed.py` — writes telemetry to `live_feed_db.sqlite`
2. `run_langgraph.py --live` — starts the LangGraph pipeline reading from `live_feed_db.sqlite`

### Step 3 — LangGraph Polling Loop

`run_langgraph.py` runs a continuous loop. Each iteration calls `compiled_graph.invoke(state)` which triggers the full 10-node pipeline. The loop polls for new data every 100–500ms (configurable via `POLL_INTERVAL_MS`).

### Step 4 — 10-Node Pipeline Execution

For each telemetry tick:

```
n01_collect → n02_feature_eng → n03_prelim_severity → n04_classify
→ n05_tumbling_window → n06_forecasting → n07_severity_update
→ n08_reliability → n09_human_gate → n10_db_writer → END
```

Each node reads from the shared `AIOpsLangState` TypedDict, performs its computation, and returns only the keys it updated.

### Step 5 — SSE Push to Frontend

After `n10_db_writer` completes, it calls `SSEBroadcaster.update(state)`. The FastAPI `/api/live-stream` endpoint streams this as an SSE event every 500ms to any connected frontend client.

### Step 6 — React Dashboard Rendering

The frontend's `App.jsx` maintains a persistent `EventSource` connection to `/api/live-stream`. Each SSE event updates the React state, triggering re-renders of:
- `SummaryCard` — current incident status, severity, failure mode
- `MultiTrendCharts` — time-series metric charts
- `PredictionCard` — classification confidence and predicted failure
- `ReliabilityCard` — Weibull survival probability
- `HumanGatePanel` — pending approval reviews

---

## 10. Detailed Module Architecture

### Simulator
Generates synthetic telemetry modeled on real microservice failure patterns. Two modes:
- **Offline (batch)**: `app_data_generator_for_offline/` — generates 120,120 rows (77 episodes × 13 modes × 120 steps) for training data
- **Live (streaming)**: `live_feed_simulator/` — emits at 0.5s/tick for real-time demo

### Inference LangGraph
The core AI inference pipeline. A 10-node LangGraph `StateGraph` that processes one telemetry tick per invocation. Each node is a pure Python function `run(state) -> dict`.

### QueryLangGraph
A separate LangGraph-based chatbot agent. Processes natural-language questions about incidents through a guarded 11-node pipeline including parse, validation, intent routing, retrieval, synthesis, and respond stages.

### FastAPI Core API
The backend REST + SSE server (port 8080). Serves historical episode data, live feed state, human gate decisions, and real-time SSE push. Also manages subprocess lifecycle for live simulation.

### React Frontend
Single-page dashboard (port 5173) that subscribes to SSE events and renders real-time incident state, charts, reliability graphs, and the embedded chatbot.

---

## 11. Folder Structure

```
AIOps_Incident_Management/
│
├── README.md                          # Root documentation
├── requirements.txt                   # Top-level Python dependencies
│
├── backend/
│   ├── README.md
│   ├── requirements.txt               # Full backend dependency list
│   ├── run_server.py                  # Unified launcher (ports 8080 + 8001)
│   ├── run_langgraph.py               # LangGraph inference runner (historical + live)
│   ├── run_pipeline.py                # Legacy batch pipeline runner
│   ├── start_chatbot_server.py        # QueryLangGraph chatbot API launcher
│   ├── reset_all.py                   # Reset all DB and output files
│   ├── check_db.py / inspect_db.py    # DB utility scripts
│   │
│   ├── api/                           # FastAPI Core API (port 8080)
│   │   ├── main.py                    # All REST + SSE endpoints
│   │   ├── services.py                # Business logic (AIOpsDashboardService, etc.)
│   │   └── sse_broadcaster.py         # Thread-safe SSE state broadcaster
│   │
│   ├── Inference_langgraph/           # 10-node inference LangGraph pipeline
│   │   ├── graph.py                   # StateGraph assembly + routing logic
│   │   ├── state.py                   # AIOpsLangState TypedDict + Pydantic validation
│   │   ├── Graph_node/                # LangGraph node wrappers (n01–n10)
│   │   │   ├── n01_collect.py
│   │   │   ├── n02_feature_engineering.py
│   │   │   ├── n03_prelim_severity.py
│   │   │   ├── n04_classify.py
│   │   │   ├── n05_tumbling_window.py
│   │   │   ├── n06_forecasting.py
│   │   │   ├── n07_severity_update.py
│   │   │   ├── n08_reliability.py
│   │   │   ├── n09_human_gate.py
│   │   │   └── n10_db_writer.py
│   │   └── nodes/                     # Domain engines (business logic)
│   │       ├── classification/        # LightGBM model + Drain3 log parser
│   │       ├── collect/               # Telemetry collector
│   │       ├── db_writer/             # DbWriter (3-table + 7 node tables + pipeline_results)
│   │       ├── feature_engineering/   # 37-feature vector builder
│   │       ├── forecasting/           # Per-mode TTF forecasters + router
│   │       ├── human_gate/            # Approval engine, audit logger, interrupt manager
│   │       ├── preliminary_severity/  # Rule-based P1–P4 engine
│   │       ├── reliability/           # Weibull fitter + Kaplan-Meier
│   │       ├── severity_update/       # Impact/urgency matrix + hysteresis tracker
│   │       └── tumbling_window/       # 10-cycle label majority-vote smoother
│   │
│   ├── QueryLanggraph/                # QueryLangGraph chatbot agent
│   │   └── QueryLangGraph02-main/
│   │       ├── graph.py               # 11-node query workflow
│   │       ├── state.py               # QueryState TypedDict
│   │       ├── nodes/                 # parse_query, validation, retrieval, synthesis, respond
│   │       ├── guardrails/            # parse_guard, validation_guard, retrieval_guard, synthesis_guard
│   │       ├── routers/               # intent_router, sufficiency_router
│   │       ├── llm/                   # LLM client (Gemini/OpenAI)
│   │       ├── services/              # DB query services
│   │       └── schemas/               # Response schemas
│   │
│   └── Simulator/
│       ├── app_data_generator_for_offline/  # Offline batch data generator
│       │   ├── config.py              # Central configuration (ALL constants)
│       │   ├── state.py               # PipelineState dataclass
│       │   ├── storage/schema.sql     # SQLite schema DDL
│       │   └── output/simulator_db.sqlite
│       └── live_feed_simulator/       # Live streaming simulator
│           ├── run_live_feed.py       # Main live simulator loop
│           ├── live_queue.py          # In-process LiveTelemetryQueue
│           └── output/live_feed_db.sqlite
│
└── frontend/
    ├── README.md
    ├── package.json
    ├── vite.config.js
    └── src/
        ├── App.jsx                    # Root app, SSE listener, routing
        ├── index.css                  # Global styles
        ├── main.jsx                   # React entry point
        └── components/
            ├── TopHeader.jsx          # App header + Live Feed toggle
            ├── Sidebar.jsx            # Navigation sidebar
            ├── SummaryCard.jsx        # Incident summary widget
            ├── PredictionCard.jsx     # ML classification output
            ├── DiagnosisCard.jsx      # Root-cause diagnosis display
            ├── EvidenceCard.jsx       # Raw telemetry evidence
            ├── MultiTrendCharts.jsx   # Time-series metric charts (Recharts)
            ├── AIOpsChartRenderer.jsx # Dynamic chart selector
            ├── ReliabilityCard.jsx    # Weibull + KM survival chart
            ├── ReliabilityGraphSection.jsx  # Full reliability dashboard
            ├── HumanGatePanel.jsx     # Review queue + approval UI
            ├── HumanGateSidebarSection.jsx  # Compact sidebar gate view
            └── AIOpsChat.jsx          # LLM chatbot panel (port 8001)
```

---

## 12. Technology Stack

### Backend

| Layer | Technology | Version | Purpose |
|-------|-----------|---------|---------|
| Language | Python | 3.11+ | All backend logic |
| Agent Framework | LangGraph | 0.1.x | Stateful pipeline orchestration |
| ML Model | LightGBM | 4.x | 13-class failure mode classifier |
| Log Parser | Drain3 | 0.9.x | Template-based log anomaly detection |
| Web Framework | FastAPI | 0.111.x | REST + SSE API server |
| ASGI Server | Uvicorn | 0.29.x | FastAPI runtime |
| Database | SQLite | 3.x | WAL-mode telemetry + pipeline results |
| Data Processing | Pandas, NumPy | 2.x, 1.x | Feature engineering, data frames |
| Statistical Analysis | SciPy | 1.13.x | Weibull MLE fitting |
| Survival Analysis | lifelines / reliability | latest | Kaplan-Meier + Weibull (primary) |
| Time-Series Forecast | pmdarima (auto_arima) | 2.x | ARIMA forecasting |
| Validation | Pydantic | 2.x | State schema validation |
| LLM | Google Gemini / OpenAI | latest | Natural language synthesis |

### Frontend

| Layer | Technology | Version | Purpose |
|-------|-----------|---------|---------|
| Framework | React | 18.x | Component-based UI |
| Build Tool | Vite | 5.x | Dev server + bundler |
| Charts | Recharts | 2.x | Time-series + survival charts |
| Styling | Vanilla CSS | — | Full custom design system |
| HTTP | Browser fetch + EventSource | — | REST calls + SSE subscription |
| Font | Google Fonts (Inter) | — | Typography |

---

## 13. Backend Architecture

The backend is split into four distinct server/runner processes:

### Process 1 — Sentinel Dashboard API (port 8080)
- File: `backend/api/main.py`
- Framework: FastAPI + Uvicorn
- Responsibilities: Serve REST endpoints for historical episodes, live feed state, human gate decisions, SSE push, and subprocess management.
- CORS: Allows `http://localhost:5173` (Vite dev server) and wildcard.

### Process 2 — QueryLangGraph Chatbot API (port 8001)
- File: `backend/start_chatbot_server.py`
- Framework: FastAPI + Uvicorn
- Responsibilities: Accept natural-language queries, run them through the 11-node QueryLangGraph pipeline, return JSON responses.

### Process 3 — LangGraph Inference Pipeline (no port)
- File: `backend/run_langgraph.py`
- Type: Standalone Python process (no HTTP server)
- Responsibilities: Continuously poll the SQLite database for new telemetry rows and invoke the 10-node StateGraph.

### Process 4 — Live Feed Simulator (no port)
- File: `backend/Simulator/live_feed_simulator/run_live_feed.py`
- Type: Standalone Python process (no HTTP server)
- Responsibilities: Emit synthetic telemetry at 0.5s/tick to `live_feed_db.sqlite`.

### Unified Launch
`backend/run_server.py` starts all backend services from a single command:
1. Launches the Chatbot API (port 8001) as a background subprocess.
2. Runs the Dashboard API (port 8080) directly via uvicorn (blocking).
3. Cleanup handlers (SIGINT/SIGTERM) terminate background subprocesses gracefully.

Processes 3 and 4 are triggered on-demand via `POST /api/live/start` from the frontend.

---

## 14. Frontend Architecture

The frontend is a React + Vite single-page application.

### Core Data Flow

```
SSE EventSource (/api/live-stream)
        │
        ▼
App.jsx (useState: incident, episodes, reliabilityData)
        │
        ├── TopHeader.jsx       ← Live Feed toggle button
        ├── Sidebar.jsx         ← Episode list navigation
        └── Main Panel
            ├── SummaryCard     ← Incident overview
            ├── PredictionCard  ← LightGBM result
            ├── DiagnosisCard   ← RCA + evidence
            ├── MultiTrendCharts ← 7-metric time-series
            ├── ReliabilityGraphSection ← Weibull + KM
            ├── HumanGatePanel  ← Approval queue
            └── AIOpsChat       ← LLM chatbot (port 8001)
```

### State Management
- `App.jsx` owns all shared state via React `useState` hooks.
- Real-time data arrives via the `EventSource` SSE stream from port 8080.
- Episode history is fetched via REST `GET /api/episodes`.
- Chart data is derived directly from the SSE payload (no separate fetch needed).

### SSE Connection Lifecycle
```javascript
const es = new EventSource('http://localhost:8080/api/live-stream');
es.onmessage = (e) => {
  const data = JSON.parse(e.data);
  setIncident(data);
};
```
The connection is established on mount and automatically reconnects (native browser behavior).

### Live Feed Toggle
When the operator clicks the "Live Feed" button:
1. `TopHeader.jsx` calls `onToggleLiveFeed()` passed from `App.jsx`.
2. `App.jsx` sends `POST /api/live/start` or `POST /api/live/stop`.
3. The API server spawns or terminates the simulator + pipeline subprocesses.
4. SSE events start/stop flowing within 1–2 seconds.

---

## 15. Router & API Architecture

### FastAPI Routing Strategy

All routes are defined in `backend/api/main.py`. No separate router files are used — all endpoints are directly registered on the FastAPI `app` instance. CORS middleware is applied globally.

The API is organized into four functional groups:

```
/api/health              ← Health check
/api/live                ← Latest pipeline state (historical CSV)
/api/episodes            ← Episode list (historical)
/api/episodes/{id}       ← Episode detail (historical)
/api/incident/{id}       ← Status update (PATCH)
/api/reliability/summary ← Weibull + KM summary
/api/live-stream         ← SSE stream (real-time push)
/api/live-feed/state     ← Latest live feed DB state (REST fallback)
/api/live-feed/episodes  ← Live feed episode list
/api/live-feed/status    ← Live feed session metadata
/api/live/start          ← Start simulation subprocesses
/api/live/stop           ← Stop simulation subprocesses
/api/live/simulation-status ← Check if simulation is running
/api/human-gate/pending  ← Pending reviews
/api/human-gate/review/{id} ← Review detail
/api/human-gate/decision/{id} ← Submit APPROVED/REJECTED
/api/human-gate/metrics  ← Gate KPIs
/api/human-gate/history  ← Gate audit log
```

### QueryLangGraph Routing (port 8001)

The chatbot API exposes:
```
POST /query    ← Accept user_query string, return JSON response
GET  /health   ← Health check
```

The QueryLangGraph routes the query through 11 nodes internally (see Section 33).

### LangGraph Graph Routing (Internal)

`graph.py` defines two routing functions:

**`_route_after_collect(state)`**  
After node n01 (collect):
- `state.error == "no_data"` → `END` (outer loop retries after `POLL_INTERVAL_MS`)
- Otherwise → `feature_eng`

**`_route_on_error(state)`**  
After every middle node (n02–n09):
- `state.error is set` → `db_writer` (write partial state, then END)
- Otherwise → next node in linear chain

This ensures that even if a node fails midway, the pipeline writes whatever state it has accumulated before terminating — preventing silent data loss.

---

## 16. Complete API Endpoint Documentation

### Health & Core

#### `GET /api/health`
Returns API health status.
```json
{ "status": "healthy", "service": "AIOps Sentinel Core" }
```

---

#### `GET /api/live`
Returns the latest pipeline cycle state from historical CSV output files.
- **Response**: Full incident object (episode_id, failure_mode, predicted_failure, severity, forecast, etc.)
- **Error**: `404` if no pipeline run found

---

#### `GET /api/episodes`
Returns a list of all processed historical episodes.
- **Response**: `List[EpisodeSummary]` (episode_id, failure_mode, timestamp, severity, status)

---

#### `GET /api/episodes/{episode_id}`
Returns full detail for one historical episode.
- **Path param**: `episode_id` (string)
- **Response**: Detailed episode object including all node outputs
- **Error**: `404` if not found

---

#### `PATCH /api/incident/{episode_id}`
Updates the status of an incident (in-memory override).
- **Path param**: `episode_id`
- **Body**: `{ "status": "ACKNOWLEDGED" | "IN_PROGRESS" | "RESOLVED" | "OPEN" }`
- **Response**: `{ "episode_id": ..., "status": ..., "message": ... }`
- **Error**: `400` for invalid status value

---

#### `GET /api/reliability/summary`
Returns 4-group Weibull parameters, Kaplan-Meier step points, and Weibull survival curves.
- **Response**: Nested object with `groups` array, each containing `beta`, `eta`, `survival_curve`, `km_curve`

---

### Live Feed & SSE

#### `GET /api/live-stream`
**Server-Sent Events** endpoint. Streams the latest enriched pipeline state every 500ms.
- **Media type**: `text/event-stream`
- **Format**: `data: {json_payload}\n\n`
- **Headers**: `Cache-Control: no-cache`, `X-Accel-Buffering: no`
- The payload includes current incident state + chart data + reliability + forecast

---

#### `GET /api/live-feed/state`
REST fallback for the latest live pipeline state (reads from `live_feed_db.sqlite`).
- **Error**: `404` if no live data yet

---

#### `GET /api/live-feed/episodes`
Returns all episodes from the live feed database, newest first.

---

#### `GET /api/live-feed/status`
Returns live feed session metadata: queue depth, episode count, latest failure mode.

---

#### `POST /api/live/start?speed=1.0`
Launches live telemetry simulator + LangGraph pipeline as background subprocesses.
- **Query param**: `speed` (float, default 1.0) — simulation speed multiplier
- **Response**: `{ "status": "started", "running": true, "message": "..." }`

---

#### `POST /api/live/stop`
Terminates both background subprocesses.
- **Response**: `{ "status": "stopped", "running": false, "message": "..." }`

---

#### `GET /api/live/simulation-status` (also `/api/live/status-check`)
Checks subprocess poll status.
- **Response**: `{ "running": bool, "simulator_running": bool, "langgraph_running": bool }`

---

### Human Gate

#### `GET /api/human-gate/pending`
Returns all reviews currently awaiting operator decision.
- **Response**: `List[ReviewSummary]` (review_id, episode_id, old_severity, new_severity, created_at)

---

#### `GET /api/human-gate/review/{review_id}`
Returns full detail for one review and marks it as `REVIEWING`.
- **Error**: `404` if not found

---

#### `POST /api/human-gate/decision/{review_id}`
Submit operator's decision.
- **Body**: `{ "decision": "APPROVED" | "REJECTED", "operator": "john.doe", "reason": "..." }`
- **Error**: `400` for invalid decision string

---

#### `GET /api/human-gate/metrics`
Returns Human Gate KPIs: total reviews, auto-approval rate, avg response time, etc.

---

#### `GET /api/human-gate/history?limit=50`
Returns recent audit records, newest first.

---

## 17. API Request–Response Flow

### Flow: Frontend starts live feed

```
[Browser] TopHeader "Start Live Feed" click
    │
    ▼
POST /api/live/start
    │
    ▼
[FastAPI] api/main.py → _live_feed_proc = subprocess.Popen(run_live_feed.py)
                        _langgraph_proc = subprocess.Popen(run_langgraph.py --live)
    │
    ▼
[Response] { "status": "started", "running": true }
    │
    ▼
[run_live_feed.py] Writes to live_feed_db.sqlite (metrics/logs/traces) every 0.5s
    │
    ▼
[run_langgraph.py] Polls DB → invokes 10-node graph → n10 calls SSEBroadcaster.update()
    │
    ▼
GET /api/live-stream (SSE)
    │
    ▼
[Browser] App.jsx EventSource.onmessage → React setState → component re-render
```

### Flow: Human Gate approval

```
[n09_human_gate] Detects large severity jump (P4→P2)
    │ Creates review in human_gate_audit.db
    │ Sets state.hg_needed=True, hg_review_id=uuid
    │
    ▼
GET /api/human-gate/pending
    │
    ▼
[HumanGatePanel] Renders review with old/new severity + evidence
    │
    ▼
POST /api/human-gate/decision/{review_id}
    Body: { "decision": "APPROVED", "operator": "kunal", "reason": "confirmed P2" }
    │
    ▼
[HumanGateService] Writes decision to audit DB → unblocks interrupt_manager
    │
    ▼
Pipeline continues with approved severity
```

---

## 18. Database Architecture & Data Models

The system uses two SQLite databases:

| Database | Path | Purpose |
|----------|------|---------|
| `simulator_db.sqlite` | `Simulator/app_data_generator_for_offline/output/` | Historical training data |
| `live_feed_db.sqlite` | `Simulator/live_feed_simulator/output/` | Live streaming telemetry |

Both use **WAL (Write-Ahead Logging) mode** and `PRAGMA synchronous=NORMAL` for concurrent read/write access.

### Schema: Raw Telemetry Tables

#### `metrics` table
Stores one row per tick per service. Primary key: `id INTEGER PRIMARY KEY AUTOINCREMENT`.  
Key columns:
- `episode_id TEXT` — unique episode identifier (e.g., `ep_MEMORY_LEAK_20260801_001`)
- `failure_mode TEXT` — simulator ground-truth label
- `service TEXT` — one of `auth-service`, `payment-service`, etc.
- `timestamp REAL` — Unix epoch float
- `elapsed_s REAL` — seconds since episode start
- `cpu_utilization REAL`, `memory_utilization REAL`, `heap_mb REAL` — 27+ metric columns
- Unique constraint: `(episode_id, timestamp)`

#### `logs` table
One row per log entry per tick.
- `episode_id TEXT`, `timestamp REAL`, `log_level TEXT` (INFO/WARNING/ERROR/CRITICAL)
- `exception_type TEXT`, `log_message TEXT`

#### `traces` table
One distributed trace span per row.
- `episode_id TEXT`, `timestamp REAL`, `trace_id TEXT`, `span_id TEXT`, `parent_span_id TEXT`
- `span_name TEXT`, `span_duration_ms REAL`, `span_status TEXT`, `peer_service TEXT`

### Schema: Pipeline Output Tables

#### `node_feature_engineering`
37 feature columns (32 metrics + 5 log features) per cycle.

#### `node_preliminary_severity`
P1–P4 score with weighted_score, blast_size, high_risk_mode, reason.

#### `node_classification`
`predicted_failure TEXT`, `prediction_probability REAL` per cycle.

#### `node_tumbling_window`
`dominant_state TEXT`, `vote_distribution TEXT (JSON)`, `window_margin REAL`, `window_full INTEGER`.

#### `node_forecasting`
`algorithm_used TEXT`, `time_to_failure REAL`, `forecast_confidence REAL`, `threshold_crossed INTEGER`, `feature_ttfs TEXT (JSON)`, `predictions TEXT (JSON)`.

#### `node_severity_update`
`impact_band TEXT`, `urgency_band TEXT`, `candidate_severity TEXT`, `revised_severity TEXT`, `is_escalated INTEGER`, `is_deescalated INTEGER`, `dwell_count INTEGER`.

#### `node_reliability`
`group_name TEXT`, `ttf_seconds REAL`, `survival_probability REAL`, `beta REAL`, `eta REAL`.

#### `node_human_gate`
`review_id TEXT PRIMARY KEY`, `decision TEXT`, `operator TEXT`, `response_ms INTEGER`, `is_large_jump INTEGER`, `escalation_summary TEXT`.

#### `pipeline_results` (main aggregated table)
A denormalized view combining all node outputs for one cycle. This is the primary table read by `LiveFeedService.get_live_feed_state()` and served to the frontend.

---

## 19. Telemetry Architecture

The telemetry system models a realistic microservice observatory across four services (`auth-service`, `payment-service`, `order-service`, `inventory-service`) and four downstream dependencies (`user-db`, `payment-gateway`, `cache-cluster`, `inventory-api`).

### Three Telemetry Pillars

| Pillar | Table | Signal Type | Examples |
|--------|-------|-------------|---------|
| Metrics | `metrics` | Numeric time-series | CPU%, p99 latency, heap, error_rate |
| Logs | `logs` | Structured text events | WARN/ERROR log messages, exceptions |
| Traces | `traces` | Distributed spans | Span duration, parent-child relationships |

### Healthy Baseline Values
The simulator models realistic baseline values:
```
cpu_utilization:    22%   memory_utilization: 25%
p50_latency:        90ms  p99_latency:        130ms
error_rate:         1%    cache_hit_ratio:    95%
db_query_latency:   21ms  gc_pause_ms:        15ms
```

---

## 20. Metrics Collection & Processing

### Raw Metric Columns (32 features)

```
cpu_utilization        memory_utilization    heap_mb
db_p99                 disk_read_latency     disk_write_latency
error_rate             gc_pause_p99          cache_hit_rate
cache_miss_rate        active_connections    network_errors
p50_latency            p95_latency           p99_latency
queue_lag              retry_count_per_request  rps
upstream_timeout_rate  circuit_breaker_state   http_4xx_rate
http_5xx_rate          iops_utilization      thread_pool_queue
cpu_saturation         db_connection_pool    db_connection_wait
```

### Metric Generation per Failure Mode

Each of the 13 failure modes applies a **deterministic injection function** on top of the healthy baseline:

| Failure Mode | Primary Signal Injection |
|-------------|--------------------------|
| `MEMORY_LEAK` | `heap_mb` grows linearly; `gc_pause_p99` increases |
| `CPU_SATURATION` | `cpu_utilization` ramps to 95%+; `p99_latency` degrades |
| `LATENCY_SPIKE` | `p99_latency` spikes; `gc_pause_p99` elevated |
| `ERROR_STORM` | `error_rate` and `http_5xx_rate` spike; exceptions in logs |
| `DB_SLOWDOWN` | `db_p99` and `db_connection_wait` increase |
| `CACHE_STAMPEDE` | `cache_miss_rate` spikes; `db_p99` secondary increase |
| `QUEUE_BACKUP` | `queue_lag` and `thread_pool_queue` saturate |
| `DEPENDENCY_TIMEOUT` | `upstream_timeout_rate` high; `p99_latency` degrades |
| `BAD_DEPLOY` | `error_rate` + `http_5xx_rate` immediate spike; `NullPointerException` in logs |
| `RETRY_STORM` | `retry_count_per_request` and `rps` both high |
| `DISK_IO_SATURATION` | `iops_utilization` and `disk_read_latency` saturate |
| `CASCADING_FAILURE` | Multiple metrics breach P1 simultaneously; `SystemOverloadException` in logs |
| `NONE` | All metrics at baseline |

Gaussian noise of ±4% (`NOISE_LEVEL = 0.04`) is applied to all metric values to simulate real-world jitter.

---

## 21. Log Collection & Processing

### Log Generation

Each tick generates one structured log entry per service. The log level follows the failure mode severity:

| Failure Mode | Log Level | Exception Type |
|-------------|-----------|----------------|
| `NONE` | INFO | — |
| `MEMORY_LEAK` | WARNING | — |
| `ERROR_STORM` | ERROR | RuntimeException |
| `DEPENDENCY_TIMEOUT` | ERROR | SocketTimeoutException |
| `BAD_DEPLOY` | ERROR | NullPointerException |
| `CASCADING_FAILURE` | CRITICAL | SystemOverloadException |

### Drain3 Log Parsing (inside classifier)

The `nodes/classification/` module uses **Drain3** (a streaming log template miner) to:
1. Parse free-text log messages into templates (e.g., "WARN heap=<*>MB gc_p99=<*>ms")
2. Detect **novel templates** (unseen patterns) — signals anomalous behavior
3. Extract 5 log feature columns:
   - `log_count` — number of log entries in this tick
   - `log_max_severity` — numeric severity level (INFO=0, WARN=1, ERROR=2, CRITICAL=3)
   - `log_critical_count` — count of CRITICAL entries
   - `log_has_exception` — 1 if any exception_type is set
   - `log_has_novel_template` — 1 if Drain3 saw an unseen template

---

## 22. Distributed Tracing

### Trace Schema

Each tick generates 2–5 distributed trace spans. Each span records:

- `trace_id TEXT` — groups all spans for one request chain
- `span_id TEXT` — unique ID for this operation
- `parent_span_id TEXT` — links child to parent (NULL for root spans)
- `span_name TEXT` — operation name (e.g., "payment.process", "db.query")
- `span_duration_ms REAL` — how long the span took
- `span_status TEXT` — OK or ERROR
- `peer_service TEXT` — the downstream service called (from `DOWNSTREAM_SERVICES`)
- `service_version TEXT` — e.g., "v2.0.1"

### Trace-Derived Signals

In failure modes like `DB_SLOWDOWN`, span durations for `"db.query"` spans are inflated. The `n02_feature_engineering` node can use span data to compute:
- Max span duration
- Count of ERROR-status spans
- Count of cross-service calls

---

## 23. Telemetry Correlation

The `n01_collect.py` node performs a **temporal join** across the three telemetry pillars:

```python
# Step 1: Fetch the next metrics row by id > last_processed_id
metric_row = conn.execute("SELECT * FROM metrics WHERE id > ? LIMIT 1", (last_id,))

# Step 2: Fetch logs for the same (episode_id, timestamp)
logs = conn.execute(
    "SELECT log_level, exception_type, log_message FROM logs 
     WHERE episode_id = ? AND timestamp = ?", (ep_id, ts)
)

# Step 3: Fetch traces for the same (episode_id, timestamp)
spans = conn.execute(
    "SELECT * FROM traces WHERE episode_id = ? AND timestamp = ?", (ep_id, ts)
)
```

This produces a perfectly correlated snapshot where metrics, logs, and traces share the same `episode_id` and `timestamp`, guaranteeing that the feature engineering node receives a coherent multi-signal view.

---

## 24. Data Preprocessing & Feature Engineering

### Node: `n02_feature_engineering` (Graph Node n02)

Transforms raw correlated telemetry into a **37-feature vector** ready for ML inference.

**Input**: `state.raw_metric` (dict), `state.raw_log` (list), `state.raw_traces` (list)

**Output**: `state.classifier_input` (dict, 37 keys) + `state.evidence` (metadata dict)

### Feature Vector Composition

#### Metric Features (32 columns)
Direct extraction from `raw_metric` with type coercion:
```
cpu_utilization, memory_utilization, heap_mb, db_p99, disk_read_latency,
disk_write_latency, error_rate, gc_pause_p99, cache_hit_rate, cache_miss_rate,
active_connections, network_errors, p50_latency, p95_latency, p99_latency,
queue_lag, retry_count_per_request, rps, upstream_timeout_rate,
circuit_breaker_state (encoded: closed=0, half-open=1, open=2),
http_4xx_rate, http_5xx_rate, iops_utilization, thread_pool_queue,
cpu_saturation, db_connection_pool, db_connection_wait
```

#### Log Features (5 columns)
Computed from `raw_log` entries:
```
log_count              = len(raw_log)
log_max_severity       = max(level_numeric for each log entry)
log_critical_count     = sum(1 for log if level == CRITICAL)
log_has_exception      = 1 if any log has exception_type else 0
log_has_novel_template = 1 if Drain3 detects new template else 0
```

**Total: 37 features** passed as `classifier_input` to `n03_prelim_severity` and `n04_classify`.

The `evidence` dict captures diagnostic metadata: which features are anomalous, trace error counts, and the correlated log messages — this feeds the RCA display in the frontend.

---

## 25. ML Pipeline

The ML pipeline runs end-to-end within the LangGraph graph per telemetry cycle:

```
Raw Telemetry (37 features)
        │
        ▼
[n03] Preliminary Severity (rule-based P1–P4)
        │
        ▼
[n04] LightGBM Classification (13-class softmax)
        │
        ▼
[n05] Tumbling Window (10-cycle majority vote → dominant_state)
        │
        ▼
[n06] Mode-Specific TTF Forecasting (ARIMA/linear/exponential per mode)
        │
        ▼
[n07] Severity Update (Impact × Urgency matrix → revised P1–P4 with hysteresis)
        │
        ▼
[n08] Weibull Reliability (survival probability for this failure group)
```

The ML pipeline is designed to be **gracefully degradable**: if the LightGBM model `.pkl` is not found at startup, the classifier returns `"PENDING"` with 0.0 probability and triggers auto-training from `engineered_features.csv`.

---

## 26. Failure Mode Classification

### 13 Failure Modes

```
NONE               MEMORY_LEAK         CPU_SATURATION
LATENCY_SPIKE      ERROR_STORM         DB_SLOWDOWN
CACHE_STAMPEDE     QUEUE_BACKUP        DEPENDENCY_TIMEOUT
BAD_DEPLOY         RETRY_STORM         DISK_IO_SATURATION
CASCADING_FAILURE
```

### Training Data

- **Episodes**: 77 episodes per mode × 13 modes = 1001 episodes
- **Steps per episode**: 120 ticks × 2 seconds = 4 minutes each
- **Total rows**: ~120,120 metric rows

### Preprocessing for Training

1. `engineered_features.csv` is built by running the offline pipeline over `simulator_db.sqlite`
2. Target label: `failure_mode` (13-class)
3. Features: All 37 engineered feature columns
4. Train/test split: 80/20 (`TEST_SIZE = 0.20`)
5. Random seed: 42 (`RANDOM_SEED = 42`)

---

## 27. LightGBM Model

### Why LightGBM

LightGBM was selected over XGBoost, Random Forest, and LSTM because:
- **Gradient boosted trees** handle mixed numeric/categorical features natively
- **Leaf-wise tree growth** is faster than level-wise (critical for 120k rows)
- **Built-in class balancing** (`class_weight="balanced"`) handles minority failure modes
- **Native probability output** via `predict_proba` gives confidence scores for TTF gating
- No normalization required (robust to the 10× variance range across metrics)

### Hyperparameter Tuning

Tuned via **Optuna** TPE sampler with:
- `TUNING_N_TRIALS = 80` — number of trials
- `TUNING_CV_FOLDS = 5` — StratifiedKFold (20% per fold)
- `TUNING_METRIC = "f1"`, `TUNING_AVERAGE = "macro"` — equal weight to all 13 classes
- `TUNING_EARLY_STOPPING_ROUNDS = 50` — prevents overfitting
- Study is persisted to `optuna_study.db` for resumable tuning

### Search Space (9 parameters)
```
n_estimators, learning_rate, max_depth, num_leaves,
min_child_samples, reg_alpha, reg_lambda, feature_fraction, bagging_fraction
```

### Model Artifacts

| File | Path | Purpose |
|------|------|---------|
| `lgbm_model.pkl` | `nodes/classification/models/` | Trained LightGBM classifier |
| `label_encoder.pkl` | `nodes/classification/models/` | Sklearn LabelEncoder for 13 classes |
| `feature_names.json` | `nodes/classification/models/` | Ordered list of 37 feature names |
| `drain3.ini` | `nodes/classification/models/` | Drain3 configuration |
| `drain_state.bin` | `nodes/classification/models/` | Trained Drain3 template state |
| `known_log_templates.json` | `nodes/classification/models/` | Known template registry |

### Runtime Inference

```python
# n04_classify.py
df = pd.DataFrame([features_dict])[feature_names]
proba = model.predict_proba(df)[0]
idx = proba.argmax()
predicted = label_encoder.inverse_transform([idx])[0]  # e.g., "MEMORY_LEAK"
confidence = float(proba[idx])                          # e.g., 0.94
```

---

## 28. Tumbling Window

### Purpose

The raw LightGBM classifier can produce noisy per-cycle predictions — a `MEMORY_LEAK` episode might generate `MEMORY_LEAK`, `CPU_SATURATION`, `MEMORY_LEAK`, `NONE`, `MEMORY_LEAK` across 5 cycles due to metric jitter. The Tumbling Window stabilizes the label sequence.

### Mechanism

- **Buffer size**: 10 cycles (configurable via `WINDOW_SIZE = 10`)
- **Buffer type**: `collections.deque(maxlen=10)` — oldest prediction is dropped when full
- **Voting**: `Counter(buffer)` → majority label → `dominant_state`
- **Margin**: `votes_for_winner - votes_for_runner_up` (indicates prediction confidence)
- **Window full flag**: True once 10+ predictions have been seen

### Design Decisions

1. **Only labels are smoothed** — raw metric features are never averaged. Averaging 10 cycles of CPU% would dilute genuine spikes.
2. **No episode boundary reset** — the buffer carries forward across episode changes. This prevents cold-start instability when a new episode begins.
3. **Margin exported to state** — downstream nodes (forecasting) use `window_margin` to gauge prediction confidence.

### Example

```
Buffer: [MEMORY_LEAK, MEMORY_LEAK, CPU_SAT, MEMORY_LEAK, MEMORY_LEAK,
         MEMORY_LEAK, MEMORY_LEAK, CPU_SAT, MEMORY_LEAK, MEMORY_LEAK]

dominant_state    = MEMORY_LEAK
vote_distribution = { "MEMORY_LEAK": 8, "CPU_SATURATION": 2 }
window_margin     = 6
window_full       = True
```

---

## 29. Model Training & Evaluation

### Training Script

`nodes/classification/offline/train_classifier.py` performs:
1. Load `engineered_features.csv`
2. Drop metadata columns (`episode_id`, `timestamp`, etc.)
3. Encode labels with `LabelEncoder`
4. Stratified train/test split (80/20)
5. Optuna TPE hyperparameter search (80 trials, 5-fold StratifiedKFold)
6. Final training on full train set with best params
7. Evaluate on test set: per-class F1, confusion matrix
8. Serialize: `lgbm_model.pkl`, `label_encoder.pkl`, `feature_names.json`

### Auto-Training

If `lgbm_model.pkl` is missing at pipeline startup, `classifier.py` checks:
- `MIN_TRAIN_ROWS = None` → train on whatever data exists
- Otherwise waits until N rows are available

Then launches `train_classifier.py` as a subprocess, blocks until completion, and loads the freshly trained model.

### Evaluation Metrics

- **Primary**: Macro-averaged F1 (equal weight across all 13 classes)
- **Secondary**: Per-class precision, recall, F1
- **Visualization**: Confusion matrix saved as PNG in `classification/output/`

---

## 30. Root Cause Analysis (RCA)

RCA in Sentinel is not a standalone module — it is **emergent from the evidence collected by the feature engineering node** and enriched by the classifier's output.

### What constitutes RCA in Sentinel

The `evidence` dict built by `n02_feature_engineering` contains:
- The top-3 anomalous metrics by deviation from baseline
- Whether logs contained exceptions and what type
- Trace spans with ERROR status
- The Drain3 novel template flag (new log pattern detected)

The classifier's `predicted_failure` names the root cause category (e.g., "MEMORY_LEAK").

The `preliminary_severity` node's `reason` field lists the exact thresholds breached (e.g., `"cpu=91.2%>90%, p99=1043ms>1000ms"`).

The severity_update node's `su_reason` field combines impact band, urgency band, TTF, and escalation information.

Together, these fields form a **complete, auditable explanation** that the frontend `DiagnosisCard` and `EvidenceCard` components display.

### LLM-Augmented RCA (QueryLangGraph)

For deeper analysis, the QueryLangGraph chatbot (port 8001) can answer questions like:
- "What was the root cause of episode ep_MEMORY_LEAK_2026?"
- "Show me all P1 incidents in the last 24 hours"
- "What is the time-to-failure forecast for the current episode?"

The LLM retrieves structured data from the pipeline SQLite DB and synthesizes a human-readable explanation.

---

## 31. Correlation Engine

Correlation in Sentinel occurs at two levels:

### Temporal Correlation (n01_collect)
The collect node correlates metrics + logs + traces by `(episode_id, timestamp)` to ensure all three pillars refer to the same service at the same moment.

### Cross-Signal Correlation (n02_feature_engineering)
The feature engineering node fuses the 32 raw metric signals and 5 log-derived signals into a single 37-dimensional vector. This is the mathematical correlation step — the LightGBM model learns non-linear interaction effects between signals (e.g., `cache_miss_rate` high AND `db_p99` high → `CACHE_STAMPEDE`).

### Temporal Label Correlation (n05_tumbling_window)
The tumbling window correlates 10 consecutive cycle predictions to produce a temporally stable label, filtering transient correlation noise.

### Impact-Urgency Correlation (n07_severity_update)
The severity update node correlates:
- **Impact** (from preliminary severity — how bad is it now?)
- **Urgency** (from forecasting — how fast is it getting worse?)

This cross-dimensional correlation produces the final **revised severity** that factors in both current state and future trajectory.

---

## 32. LangGraph / Agent Workflow

### Architecture

The Inference LangGraph pipeline is a **LangGraph `StateGraph`** — a directed graph of Python functions that share a single `AIOpsLangState` TypedDict. LangGraph handles:
- State merging (each node returns only updated keys)
- Conditional routing via edge functions
- SQLite checkpointing (full state persisted after each node via `SqliteSaver`)

### The 10 Nodes

| Node | File | Responsibility |
|------|------|----------------|
| `n01_collect` | `Graph_node/n01_collect.py` | Read next telemetry row from SQLite |
| `n02_feature_eng` | `Graph_node/n02_feature_engineering.py` | Build 37-feature vector |
| `n03_prelim_severity` | `Graph_node/n03_prelim_severity.py` | Rule-based P1–P4 triage |
| `n04_classify` | `Graph_node/n04_classify.py` | LightGBM 13-class inference |
| `n05_tumbling_window` | `Graph_node/n05_tumbling_window.py` | 10-cycle label smoothing |
| `n06_forecasting` | `Graph_node/n06_forecasting.py` | Mode-specific TTF prediction |
| `n07_severity_update` | `Graph_node/n07_severity_update.py` | Impact × Urgency → revised severity |
| `n08_reliability` | `Graph_node/n08_reliability.py` | Weibull survival probability |
| `n09_human_gate` | `Graph_node/n09_human_gate.py` | Operator approval for large jumps |
| `n10_db_writer` | `Graph_node/n10_db_writer.py` | Atomic DB writes + SSE broadcast |

### State Flow

```python
class AIOpsLangState(TypedDict, total=False):
    # Cycle identity
    cycle: int
    last_processed_id: int

    # Raw telemetry
    raw_metric: dict          # 35+ columns from metrics table
    raw_log: list             # list of log dicts
    raw_traces: list          # list of span dicts
    episode_id: str           # ep_MEMORY_LEAK_20260801_001
    failure_mode: str         # simulator ground truth
    timestamp: float
    elapsed_s: float
    service: str

    # Feature Engineering
    classifier_input: dict    # 37-feature vector
    evidence: dict            # diagnostic metadata

    # Preliminary Severity
    preliminary_severity: str # P1 / P2 / P3 / P4
    severity_result: dict     # weighted score + reason

    # Classification
    predicted_failure: str    # MEMORY_LEAK, etc.
    prediction_probability: float

    # Tumbling Window
    dominant_state: str       # majority-vote label
    vote_distribution: dict   # {mode: count}
    window_margin: float
    window_full: bool

    # Forecasting
    forecast_result: dict
    time_to_failure: Optional[float]    # seconds until breach
    forecast_confidence: Optional[float]
    threshold_crossed: Optional[bool]
    earliest_ttf_feature: Optional[str]

    # Severity Update
    revised_severity: Optional[str]
    candidate_severity: Optional[str]
    impact_band: Optional[str]
    urgency_band: Optional[str]
    is_escalated: Optional[bool]
    is_deescalated: Optional[bool]
    dwell_count: Optional[int]

    # Reliability
    survival_probability: Optional[float]
    weibull_beta: Optional[float]
    weibull_eta: Optional[float]

    # Human Gate
    hg_needed: Optional[bool]
    hg_review_id: Optional[str]
    hg_decision: Optional[str]    # APPROVED | REJECTED | AUTO_APPROVED
    hg_final_severity: Optional[str]
    hg_response_ms: Optional[int]

    # Routing
    error: Optional[str]
```

### Checkpointing

`SqliteSaver` persists the full state after each node to `langgraph_checkpoints.db`. If the process crashes mid-episode, the next restart resumes from the last completed node — no data is lost.

### Error Handling

Every node is wrapped with a try/except. On exception:
1. `state.error = str(exception)` is set
2. `_route_on_error()` detects `error is set` → routes to `db_writer`
3. `db_writer` writes the partial state
4. Graph terminates at `END`
5. Outer loop retries the cycle

---

## 33. LLM Explanation Layer – QueryLangGraph

### Purpose

The QueryLangGraph chatbot (port 8001) allows operators to query the incident database using natural language instead of SQL or dashboard queries.

### Architecture: 11-Node Guarded Pipeline

```
User Query
    │
    ▼
[parse_guard]       ← Safety: block harmful input (prompt injection, PII)
    │ safe?
    ▼
[parse_query]       ← NLP: extract intent + entities from query
    │
    ▼
[validation_guard]  ← Safety: validate parsed output format
    │ safe?
    ▼
[validation]        ← Semantic: validate entities against DB schema
    │ valid?
    ▼
[intent_router]     ← Route to correct query strategy (episode_lookup, severity_trend, etc.)
    │
    ▼
[retrieval_guard]   ← Safety: prevent unauthorized DB access patterns
    │ safe?
    ▼
[retrieval]         ← Execute structured DB query → raw data
    │
    ▼
[sufficiency_router] ← Check if retrieved data is sufficient to answer
    │ sufficient?
    ▼
[synthesis_guard]   ← Safety: validate data before LLM synthesis
    │ safe?
    ▼
[synthesis]         ← LLM (Gemini) generates human-readable response
    │
    ▼
[respond]           ← Format final JSON response payload
    │
    ▼
API Response
```

### Guardrails

Four guardrail nodes act as safety gates:
- `parse_guard` — blocks SQL injection, prompt injection, PII queries
- `validation_guard` — ensures parsed output matches expected schema
- `retrieval_guard` — prevents unauthorized data access patterns
- `synthesis_guard` — validates raw data before passing to LLM

If any guardrail sets `state.is_safe = False`, the graph short-circuits directly to the `respond` node with an appropriate error message.

### QueryState

```python
class QueryState(TypedDict):
    user_query: str
    parsed_intent: dict        # extracted intent + entities
    is_safe: bool              # guardrail flag
    is_validated: bool         # validation flag
    is_sufficient: bool        # data sufficiency flag
    retrieved_data: dict       # raw DB query results
    synthesized_response: str  # LLM output
    final_response: dict       # formatted API response
```

### Fallback

If `langgraph` is not installed, a `PurePythonQueryGraphRunner` executes the same nodes sequentially without graph routing.

---

## 34. Incident Detection & Management

### What constitutes an "Incident"

In Sentinel, an **incident** is one **episode** — a bounded time window during which a service operates in a non-`NONE` failure mode. Each episode is identified by a unique `episode_id` (e.g., `ep_MEMORY_LEAK_20260810_001`).

### Detection Mechanism

Detection is NOT threshold-based in the traditional sense. Instead:

1. **ML Detection**: The LightGBM classifier detects the failure mode from the 37-feature vector with a confidence score.
2. **Stability Gate**: The tumbling window confirms the classification is stable (not a transient spike) before declaring an incident.
3. **Severity Triage**: The preliminary severity engine immediately assigns P1–P4 based on threshold rules, providing a fast initial assessment before ML.

### Incident Representation

After `n10_db_writer` runs, the `pipeline_results` table contains a full record:
- `episode_id`, `failure_mode` (ground truth), `predicted_failure` (ML output)
- `preliminary_severity`, `revised_severity`, `hg_final_severity`
- `time_to_failure`, `forecast_confidence`
- `survival_probability`, `weibull_beta`, `weibull_eta`
- Human gate fields

The `AIOpsDashboardService.get_live_state()` reads the latest row from `pipeline_results` and enriches it with chart data, Weibull parameters, and historical context.

---

## 35. Incident Lifecycle

```
EPISODE START (simulator generates non-NONE failure mode)
        │
        ▼
[DETECTED] — n01_collect sees new episode_id
        │
        ▼
[TRIAGE] — n03_prelim_severity assigns P1–P4 within first cycle
        │
        ▼
[CLASSIFIED] — n04_classify identifies failure mode with confidence
        │
        ▼
[STABILIZED] — n05_tumbling_window confirms dominant state after 10 cycles
        │
        ▼
[FORECASTED] — n06_forecasting computes TTF (seconds until threshold breach)
        │
        ▼
[SEVERITY UPDATED] — n07 produces revised_severity (P1–P4 with hysteresis)
        │
        ├── Large jump detected? (e.g., P4→P2)
        │   └── [PENDING REVIEW] — n09_human_gate creates review record
        │         │ Operator approves/rejects via dashboard
        │         └── [GATE RESOLVED] — decision recorded in audit DB
        │
        ▼
[PERSISTED] — n10_db_writer writes to all tables, broadcasts via SSE
        │
        ▼ (cycle repeats every 0.5–2 seconds)
        │
        ▼
[EPISODE END] — episode_id changes (simulator moves to next episode)
        │
        ▼
[CLOSED] — Forecasting episode registry cleared, survival probability finalized
        │
        ▼
[STATUS UPDATED] — Operator can mark ACKNOWLEDGED → IN_PROGRESS → RESOLVED
```

---

## 36. Frontend Components & UI

### Component Inventory

| Component | Purpose | Data Source |
|-----------|---------|-------------|
| `TopHeader.jsx` | App title + Live Feed on/off toggle | `POST /api/live/start` / `stop` |
| `Sidebar.jsx` | Episode list navigation | `GET /api/episodes` |
| `SummaryCard.jsx` | Current incident headline | SSE `incident` state |
| `PredictionCard.jsx` | ML classification + confidence | SSE `predicted_failure`, `prediction_probability` |
| `DiagnosisCard.jsx` | Root cause + evidence | SSE `evidence`, `dominant_state` |
| `EvidenceCard.jsx` | Raw telemetry evidence | SSE `raw_metric`, `raw_log` |
| `MultiTrendCharts.jsx` | 7 time-series metric charts | SSE `chart_data` |
| `AIOpsChartRenderer.jsx` | Dynamic chart selector | `MultiTrendCharts` sub-charts |
| `ReliabilityCard.jsx` | Weibull survival curve + KM | SSE `reliability` |
| `ReliabilityGraphSection.jsx` | Full 4-group reliability dashboard | `GET /api/reliability/summary` |
| `HumanGatePanel.jsx` | Full review queue + decision UI | `GET /api/human-gate/pending` |
| `HumanGateSidebarSection.jsx` | Compact sidebar gate alerts | `GET /api/human-gate/pending` |
| `AIOpsChat.jsx` | LLM chatbot panel | `POST http://localhost:8001/query` |

### Design System

- **Color scheme**: Dark mode with teal/cyan accent (#00d4ff), purple gradients, glassmorphism panels
- **Typography**: Google Fonts (Inter) for readability
- **Charts**: Recharts `AreaChart`, `LineChart`, `ComposedChart` with animated transitions
- **Animations**: CSS keyframe animations for live-status pulses and card hover effects

---

## 37. Reliability & Survival Analysis

### Purpose

The reliability node quantifies **historical failure probability over time** for each failure group. This tells operators not just "you have a MEMORY_LEAK" but "given a MEMORY_LEAK episode, there is a 72% probability it will still be active at 180 seconds".

### 4 Failure Groups (Weibull Stratification)

| Group | Failure Modes |
|-------|--------------|
| Immediate trigger | BAD_DEPLOY, CACHE_STAMPEDE, CASCADING_FAILURE, CPU_SATURATION, DEPENDENCY_TIMEOUT, ERROR_STORM |
| Fast accumulation | QUEUE_BACKUP, RETRY_STORM |
| Progressive resource degradation | DISK_IO_SATURATION |
| Slow or latent degradation | DB_SLOWDOWN, MEMORY_LEAK, LATENCY_SPIKE |

### Kaplan-Meier Estimator

Non-parametric survival curve computed from historical episode durations:

```
S(t) = ∏_{tᵢ ≤ t} (1 - dᵢ/nᵢ)

Where:
  dᵢ = number of failures at time tᵢ
  nᵢ = number at risk just before tᵢ
```

With **Greenwood confidence interval**: `SE = S(t) × √(Σ dᵢ / (nᵢ(nᵢ - dᵢ)))`

### 2-Parameter Weibull MLE

Parametric fit via Maximum Likelihood Estimation on right-censored data:

```
f(t; β, η) = (β/η) × (t/η)^(β-1) × exp(-(t/η)^β)

Parameters:
  β (beta/shape): <1 = early failures, =1 = random, >1 = wear-out
  η (eta/scale): characteristic life (63.2% failure probability)
```

**Fitting priority**:
1. `reliability` library (Reid, most accurate for small samples)
2. `lifelines` library (Davidson-Pilon)
3. SciPy L-BFGS-B optimization (always available fallback)

### Runtime Application

In `n08_reliability`, the `weibull_params.json` (pre-fitted parameters) is loaded. For each cycle, the current `elapsed_s` is used to compute:

```python
survival_probability = exp(-(elapsed_s / eta) ** beta)
```

This is written to `state.survival_probability` and displayed in the `ReliabilityCard` component.

---

## 38. Human-in-the-Loop Gate

### Purpose

Prevents fully-automated severity escalation for large jumps (e.g., P4→P1) where an operator should confirm before paging the entire on-call team.

### Trigger Condition

The `escalation_detector.py` detects a "large jump" when:
- `committed_severity` (last confirmed severity) differs from `candidate_severity` by 2+ levels (e.g., P4→P2 or P3→P1)
- OR the `candidate_severity` is P1 for the first time in this episode

### Gate Flow

```
n09_human_gate detects large jump
        │
        ├── Creates review record in human_gate_audit.db
        ├── Calls interrupt_manager to pause state
        └── Sets state.hg_needed=True, state.hg_review_id=uuid

Operator views HumanGatePanel in dashboard
        │
        ├── Clicks "APPROVE"
        │   └── POST /api/human-gate/decision/{id}  {decision: "APPROVED", operator: "kunal"}
        │       └── HumanGateService writes to audit DB → interrupt_manager unblocks
        │
        └── Clicks "REJECT"
            └── POST /api/human-gate/decision/{id}  {decision: "REJECTED"}
                └── Severity reverts to committed_severity

Auto-approve timeout: 2 seconds (HUMAN_GATE_TIMEOUT_SECONDS = 2 in config.py)
```

### Audit Trail

Every gate decision is recorded in `node_human_gate`:
- `review_id`, `decision`, `operator`, `reason`, `response_ms`
- `old_severity`, `new_severity`, `final_severity`
- `is_large_jump`, `escalation_summary`
- `created_at`, `decided_at`, `recorded_at`

---

## 39. Live Feed Simulation

### Architecture

```
run_live_feed.py
    │
    ├── Generates one tick every 0.5 seconds (LIVE_STEP_INTERVAL_S = 0.5)
    ├── Each episode = 120 ticks = 60 seconds wall time
    │
    ├── Writes to live_feed_db.sqlite (WAL mode, separate from simulator_db.sqlite)
    │
    └── Also pushes to LiveTelemetryQueue (in-process deque for single-process mode)

run_langgraph.py --live
    │
    ├── Calls n01_collect.set_live_mode(enabled=True, db_path=live_feed_db.sqlite)
    │
    ├── n01_collect first checks LiveTelemetryQueue (in-process, zero-latency)
    │   └── Falls back to SQLite read if queue is empty (multi-process mode)
    │
    └── Polls every LIVE_POLL_INTERVAL_MS = 500ms when queue is empty
```

### WAL Mode Rationale

SQLite's default journal mode locks the database during writes, which would block the LangGraph reader. **WAL mode** allows concurrent readers and one writer simultaneously:
- Writer appends to the WAL file without blocking readers
- Readers see the last committed checkpoint
- `PRAGMA synchronous=NORMAL` ensures crash safety without fsync overhead

### Subprocess Management

```python
# In api/main.py
_live_feed_proc = subprocess.Popen([python, "run_live_feed.py", "--speed", str(speed)], ...)
_langgraph_proc = subprocess.Popen([python, "run_langgraph.py", "--live"], ...)

# Liveness check
sim_running = _live_feed_proc is not None and _live_feed_proc.poll() is None
```

---

## 40. Forecasting Engine

### Architecture

The forecasting engine is a **mode-specific routing architecture**. Each failure mode has a dedicated forecasting function optimized for its characteristic degradation pattern.

### Routing Table

```python
_ROUTER = {
    "MEMORY_LEAK":        forecast_memory_leak,        # Linear heap regression
    "CPU_SATURATION":     forecast_cpu_saturation,     # Exponential ARIMA
    "LATENCY_SPIKE":      forecast_latency_spike,      # auto_arima on p99
    "DB_SLOWDOWN":        forecast_db_slowdown,        # Linear db_p99
    "CACHE_STAMPEDE":     forecast_cache_stampede,     # Linear cache_miss
    "QUEUE_BACKUP":       forecast_queue_backup,       # Linear queue_lag
    "DEPENDENCY_TIMEOUT": forecast_dependency_timeout, # Threshold convergence
    "BAD_DEPLOY":         forecast_bad_deployment,     # Step-function detection
    "ERROR_STORM":        forecast_error_storm,        # ARIMA on error_rate
    "RETRY_STORM":        forecast_retry_storm,        # Linear retry_count
    "DISK_IO_SATURATION": forecast_disk_io_saturation, # Linear IOPS
    "CASCADING_FAILURE":  forecast_cascading_failure,  # Multi-feature convergence
}
```

### Algorithms

| Algorithm | Used For | Description |
|-----------|---------|-------------|
| `auto_arima` | LATENCY_SPIKE, ERROR_STORM, CPU | Automatic ARIMA order selection via pmdarima |
| Linear extrapolation | MEMORY_LEAK, DB_SLOWDOWN, QUEUE_BACKUP | `ttf = (threshold - current) / slope` |
| Exponential growth | CPU, RETRY_STORM | `ttf = log(threshold/current) / growth_rate` |
| Multi-feature convergence | CASCADING_FAILURE | Minimum TTF across multiple features |

### Output Format (v2 schema)

```python
{
    "failure_mode":        "MEMORY_LEAK",
    "algorithm_used":      "linear_extrapolation",
    "time_to_failure":     87.3,             # seconds until threshold breach
    "forecast_confidence": 0.84,             # 0.0–1.0
    "confidence_reason":   "8/10 features converging",
    "threshold_crossed":   False,
    "feature_ttfs":        {"heap_mb": 87.3, "memory_utilization": 102.1},
    "feature_slopes":      {"heap_mb": 2.34, "memory_utilization": 0.18},
    "predictions":         [...],            # future value trajectory
}
```

### Episode Buffer

Each episode maintains a **rolling buffer** of feature rows. The forecasting functions fit their models to this buffer to compute the current slope/trajectory. The buffer is managed by `forecasting/buffer.py` and keyed by `episode_id`.

---

## 41. Severity Update & Hysteresis

### Purpose

Raw preliminary severity (P1–P4) reflects only the current cycle's threshold breaches. The severity update node produces a **refined severity** that accounts for:
1. **Current impact** (how bad is it right now?)
2. **Forecast urgency** (how fast is it getting worse?)
3. **Stability** (is this a genuine trend or a transient spike?)

### 4-Step Process

#### Step 1 — Impact Band
Maps preliminary_severity to an impact band:
```
P1 → "High"
P2 → "Moderate"
P3 / P4 → "None"
```

#### Step 2 — Urgency Band (gated by forecast confidence)
Maps TTF (time_to_failure in seconds) to an urgency band:
```
TTF < 60s  → "Imminent"
TTF < 300s → "Near"
TTF ≥ 300s → "Distant"
```
This step is **gated**: if `forecast_confidence < 0.75 (min_confidence)`, urgency defaults to `"Distant"` regardless of TTF. This prevents over-escalation when the forecast is unreliable.

#### Step 3 — Matrix Lookup
Produces `candidate_severity` from a 3×3 Impact × Urgency matrix:
```
          Imminent   Near    Distant
High   →  P1         P1      P2
Moderate→ P1         P2      P3
None   →  P2         P3      P4
```

#### Step 4 — Hysteresis Reconciliation
Prevents severity flapping:
- **Escalation** (going higher) is **immediate** — no dwell required
- **De-escalation** (going lower) requires `dwell_k = 5` consecutive cycles at the lower candidate before the severity is reduced

```python
# Escalation: immediate
if candidate > current:
    revised = candidate

# De-escalation: must dwell
elif candidate < current:
    dwell_count += 1
    if dwell_count >= dwell_k:
        revised = candidate
    else:
        revised = current  # hold at current
```

---

## 42. Outputs & Results

### Pipeline Output Files

| File | Location | Content |
|------|----------|---------|
| `engineered_features.csv` | `nodes/feature_engineering/output/` | 37-feature vectors per cycle |
| `preliminary_severity.csv` | `nodes/preliminary_severity/output/` | P1–P4 per cycle with reason |
| `pipeline_results.csv` | `nodes/classification/output/` | Full classification results |
| `tumbling_window_output.csv` | `nodes/tumbling_window/output/` | Window state per cycle |
| `forecasting_output.csv` | `nodes/forecasting/output/` | TTF predictions per cycle |
| `severity_update_output.csv` | `nodes/severity_update/output/` | Revised severity history |
| `human_gate_output.csv` | `nodes/human_gate/output/` | Gate decision records |
| `weibull_params.json` | `nodes/reliability/` | Pre-fitted Weibull β, η per group |

### Database Tables Written

All 10 pipeline tables + `pipeline_results` (denormalized) in `live_feed_db.sqlite`.

### Dashboard UI Output

- **Real-time incident card** updated every 500ms via SSE
- **7 time-series charts** for CPU, memory, latency, error rate, DB, queue, and cache metrics
- **Weibull survival curve** with KM step function overlay
- **Forecast trajectory** showing predicted metric values for the next N steps
- **Human Gate queue** with pending reviews and decision buttons
- **Chatbot panel** for natural-language incident queries

---

## 43. Running the Project

### Prerequisites
```bash
Python 3.11+
Node.js 18+
pip install -r backend/requirements.txt
cd frontend && npm install
```

### Terminal 1 — Backend (All Services)
```bash
cd d:/AIOps_Incident_Management
python backend/run_server.py
```
This launches:
- **Port 8001** — QueryLangGraph Chatbot API
- **Port 8080** — Sentinel Dashboard API (main process, blocking)

### Terminal 2 — Frontend
```bash
cd d:/AIOps_Incident_Management/frontend
npm run dev
```
Opens: `http://localhost:5173`

### Starting Live Feed (from UI)
1. Open `http://localhost:5173`
2. Click **"Start Live Feed"** button in the top header
3. The dashboard begins streaming real-time incident data

### Optional: Manual Pipeline
```bash
# Run historical pipeline (offline mode)
python backend/run_langgraph.py

# Run only live feed simulator
python backend/Simulator/live_feed_simulator/run_live_feed.py

# Run only LangGraph in live mode
python backend/run_langgraph.py --live
```

### Reset Everything
```bash
python backend/reset_all.py
```

---

## 44. Conclusion

Sentinel AIOps represents a complete implementation of an AI-powered IT Operations platform that addresses the core challenges of modern cloud-native incident management.

### Key Achievements

1. **End-to-End Automation**: From raw telemetry to classified incident to forecast to severity decision — all within 500ms per cycle, without human intervention unless escalation is required.

2. **Multi-Signal Fusion**: Three telemetry pillars (metrics + logs + traces) are correlated in real-time and fused into a coherent 37-dimensional feature vector, capturing cross-signal failure signatures that rule-based systems miss.

3. **ML + Rule Hybrid**: The system uses rule-based triage (fast, interpretable P1–P4 from thresholds) AND ML classification (accurate 13-class failure mode identification), combining the best of both approaches.

4. **Temporal Stability**: The tumbling window majority-vote smoother prevents alert storms from noisy per-cycle classifier outputs, delivering stable incident labels without sacrificing detection speed.

5. **Predictive Forecasting**: Mode-specific ARIMA and linear extrapolators provide time-to-failure estimates, enabling proactive intervention before threshold breach.

6. **Accountable Decisions**: Every severity decision, gate approval, and forecast is persisted to a full audit trail in SQLite, supporting post-mortem analysis and compliance.

7. **Human-in-the-Loop**: The Human Gate ensures that large severity jumps are human-confirmed before triggering full escalation workflows, balancing automation with operator oversight.

8. **Natural Language Access**: The QueryLangGraph chatbot brings LLM-powered question-answering to incident data, eliminating the barrier of SQL/PromQL for non-technical stakeholders.

9. **Production-Ready Patterns**: WAL-mode SQLite for concurrent access, LangGraph checkpointing for crash recovery, SSE for real-time push, and hysteresis for severity stability — all production patterns implemented from the ground up.

10. **Extensible Architecture**: The LangGraph `StateGraph` is fully modular. New analysis nodes (e.g., anomaly correlation, capacity planning) can be added by registering a new node and edge without touching existing code.

### Metric Summary

| Capability | Implementation |
|-----------|---------------|
| Failure modes detected | 13 (NONE + 12 failure types) |
| Pipeline nodes | 10 (LangGraph StateGraph) |
| Feature dimensions | 37 (32 metrics + 5 log-derived) |
| Pipeline cycle latency | < 500ms end-to-end |
| Classification model | LightGBM with Optuna HPO |
| Forecasting algorithms | ARIMA + linear + exponential |
| Survival analysis | 2-Parameter Weibull + Kaplan-Meier |
| Database tables | 11 (3 raw + 7 node + 1 combined) |
| API endpoints | 20 REST + 1 SSE |
| Server ports | 3 (8080, 8001, 5173) |
| Chatbot pipeline nodes | 11 (4 guardrails + 7 processing) |

---

*This documentation was generated from deep code analysis of the Sentinel AIOps source code. All technical details reflect the actual implementation in the `kunalwandhare567/AgenticOps` repository.*
