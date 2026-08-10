# Sentinel AIOps: Incident Detection, Forecasting & Mitigation Platform

Sentinel is an enterprise-grade AI platform for real-time incident prediction, failure mode classification, Time-to-Failure (TTF) forecasting, Weibull reliability analysis, and human-in-the-loop mitigation. It ingests raw telemetry (metrics, logs, traces), processes it through a **stateful 10-Node LangGraph Pipeline**, and streams live health telemetry to a modern React NOC Dashboard.

---

## 🏗️ End-to-End System Architecture

```
                               ┌──────────────────────────────────────────────────────────┐
                               │                 NOC Dashboard Frontend                   │
                               │                (React + Vite + Recharts)                 │
                               └───────────────┬──────────────────────────┬───────────────┘
                                               │                          │
                                HTTP REST (8080)│                          │ HTTP POST (8001)
                                & SSE Stream   │                          │ AI Natural Query
                                               ▼                          ▼
┌──────────────────────────────────────────────────────────┐   ┌──────────────────────────┐
│              Sentinel Core API Server (8080)             │   │ QueryLangGraph Chat (8001│
│             (python backend/run_server.py)               │   │    (start_chatbot_server)│
└──────────────────────────────┬───────────────────────────┘   └──────────────────────────┘
                               │ Manages Subprocesses
           ┌───────────────────┴───────────────────┐
           ▼                                       ▼
┌──────────────────────────────┐       ┌──────────────────────────────────────────────────┐
│  Live Telemetry Simulator    │       │ 10-NODE LANGGRAPH PIPELINE (run_langgraph.py)   │
│  (run_live_feed.py)          │       │                                                  │
└──────────────┬───────────────┘       │ [1] collect          Poll tick from queue / DB  │
               │ Writes telemetry      │ [2] feature_eng      32 Drain3 log + metrics    │
               ▼                       │ [3] prelim_severity  DEVOPS P1-P4 engine        │
┌──────────────────────────────┐       │ [4] classify         LightGBM failure mode      │
│  live_feed_db.sqlite (WAL)   │◄──────┤ [5] tumbling_window  10-cycle majority vote     │
└──────────────────────────────┘       │ [6] forecasting      ARIMA / Exponential TTF    │
                                       │ [7] severity_update  Impact x Urgency matrix    │
                                       │ [8] reliability      Weibull S(t) survival      │
                                       │ [9] human_gate       Escalation review queue    │
                                       │ [10] db_writer       Atomic 8-table DB write    │
                                       └──────────────────────────────────────────────────┘
```

---

## 📂 Repository Structure

```
AIOps_incident_management/
├── backend/                               # Python Backend Architecture
│   ├── run_server.py                      # 🚀 Unified Backend Launcher (Ports 8080 & 8001)
│   ├── run_langgraph.py                   # LangGraph 10-Node Pipeline Runner
│   ├── reset_all.py                       # Clean-slate DB reset tool
│   ├── requirements.txt                   # Complete Python dependencies
│   │
│   ├── Inference_langgraph/               # Stateful LangGraph Engine
│   │   ├── graph.py                       # StateGraph assembly & routing
│   │   ├── state.py                       # AIOpsLangState schema & validation
│   │   ├── Graph_node/                    # LangGraph Node Wrappers (n01_collect .. n10_db_writer)
│   │   └── nodes/                         # Domain Engines & Algorithms
│   │       ├── classification/            # LightGBM failure mode model & training
│   │       ├── collect/                   # Telemetry queue bridge & DB poll
│   │       ├── db_writer/                 # Centralised DbWriter SQLite storage
│   │       ├── feature_engineering/       # Drain3 log parser & 32-feature extractor
│   │       ├── forecasting/               # Mode-specific ARIMA / Exponential forecasters
│   │       ├── human_gate/                # Interrupt manager & approval engine
│   │       ├── preliminary_severity/      # Threshold-based EMA severity engine
│   │       ├── reliability/               # 4-group Weibull S(t) survival fitter
│   │       ├── severity_update/           # Impact × Urgency matrix updater
│   │       └── tumbling_window/           # 10-cycle majority-vote label smoother
│   │
│   ├── Simulator/                         # Telemetry Signal Generators
│   │   ├── live_feed_simulator/           # Real-time incident injector
│   │   └── app_data_generator_for_offline/# 120K-row historical batch simulator
│   │
│   ├── api/                               # Sentinel Core FastAPI Server (Port 8080)
│   │   ├── main.py                        # REST + SSE endpoints & live process manager
│   │   ├── services.py                    # SQLite data providers & service layer
│   │   └── sse_broadcaster.py             # Server-Sent Events push engine (500ms)
│   │
│   └── QueryLanggraph/                    # Natural-Language AI Assistant (Port 8001)
│       └── QueryLangGraph02-main/         # LLM pipeline with 4 security guardrails
│
├── frontend/                              # React + Vite NOC Dashboard (Port 5173)
│   ├── src/
│   │   ├── App.jsx                        # Main state container & Live feed controller
│   │   ├── components/
│   │   │   ├── TopHeader.jsx              # Header with interactive "Start Live Feed" button
│   │   │   ├── MultiTrendCharts.jsx       # 7 Core NOC metric graphs (Recharts)
│   │   │   ├── SummaryCard.jsx            # Severity badge & status control
│   │   │   ├── PredictionCard.jsx         # TTF forecast & LightGBM prediction
│   │   │   ├── ReliabilityCard.jsx        # Weibull S(t) survival curve gauge
│   │   │   └── AIOpsChat.jsx              # Right-drawer natural language AI assistant
│   ├── package.json
│   └── vite.config.js
│
├── requirements.txt                       # Root Python requirement pointer
└── README.md                              # Main platform documentation
```

---

## ⚡ Quick Start — Running the Platform (2 Terminals)

### 1. Prerequisites
- **Python**: `3.10+`
- **Node.js**: `v18+` & `npm`

### 2. Installation
```powershell
# Install backend Python packages
pip install -r requirements.txt

# Install frontend Node packages
cd frontend
npm install
```

### 3. Launching the Application (Only 2 Terminals Needed!)

#### **Terminal 1: Unified Backend Server**
Launches the **Sentinel API (Port 8080)** and **QueryLangGraph Chatbot API (Port 8001)** together:
```powershell
python backend/run_server.py
```

#### **Terminal 2: React NOC Dashboard**
Launches the **Vite React Frontend (Port 5173)**:
```powershell
cd frontend
npm run dev
```

---

## 🎮 Single-Click Live Telemetry & Pipeline Trigger

1. Open **http://localhost:5173** in your web browser.
2. In the top right header, click **"▶️ Start Live Feed"**.
3. The backend automatically launches `run_live_feed.py` (Simulator) and `run_langgraph.py --live` (LangGraph Inference Pipeline) in the background.
4. Real-time telemetry, failure predictions, TTF countdowns, and Weibull survival curves will begin streaming to your dashboard!
5. Click **"⏹️ Stop Live Feed"** anytime to pause the live simulation.

---

## 🔌 API Endpoints Reference

| Endpoint | Method | Port | Description |
| :--- | :--- | :--- | :--- |
| **`/api/live/start`** | `POST` | `8080` | Starts background simulator and 10-node LangGraph pipeline. |
| **`/api/live/stop`** | `POST` | `8080` | Terminates background simulation and pipeline processes. |
| **`/api/live/simulation-status`**| `GET` | `8080` | Returns `{ "running": true/false }` process status. |
| **`/api/live-stream`** | `GET` | `8080` | **SSE Endpoint (`text/event-stream`)** pushing pipeline state every 500ms. |
| **`/api/live-feed/state`** | `GET` | `8080` | REST fallback returning latest pipeline cycle snapshot. |
| **`/api/human-gate/pending`** | `GET` | `8080` | Fetches pending P-level escalation reviews for human decision. |
| **`/api/human-gate/decision/{id}`**| `POST` | `8080` | Submits operator `APPROVED` or `REJECTED` decision. |
| **`/query`** | `POST` | `8001` | Processes natural language queries via QueryLangGraph LLM pipeline. |

---

## 🛠️ Maintenance & Offline Utilities

- **Clean Slate Reset**:
  ```powershell
  python backend/reset_all.py --keep-model
  ```
- **Train LightGBM Incident Classifier Model**:
  ```powershell
  python backend/Inference_langgraph/nodes/classification/train_classifier.py --tune --gpu
  ```
- **Run Weibull Reliability Fitter**:
  ```powershell
  python backend/Inference_langgraph/nodes/reliability/run_weibull_fitter.py
  ```
- **Generate Pipeline Mermaid Graph Diagram**:
  ```powershell
  python backend/run_langgraph.py --diagram
  ```
