# Sentinel AIOps: Incident Detection & Mitigation Platform

An end-to-end AI-based incident prediction, classification, and mitigation dashboard. Sentinel parses raw telemetry (metrics, logs, traces), processes it through a stateful **7-Stage AIOps Pipeline**, evaluates incident severity via a custom rules-engine, forecasts Time-To-Failure (TTF) using ARIMA, and presents real-time health data on a modern React-based NOC Dashboard.

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
┌────────────────────────────────────────────────────────────────────────────┐
│                        7-STAGE AIOps PIPELINE (Backend)                    │
│                                                                            │
│  [1] Collect             Dequeue telemetry from DB/Queue                   │
│         │                                                                  │
│  [2] Feature Eng.        Generate 27 metric + 5 Drain3 log features        │
│         │                                                                  │
│  [3] Prelim Severity     Threshold-based P1/P2/P3/P4 scoring               │
│         │                                                                  │
│  [4] Classify            LightGBM failure mode detection                   │
│         │                                                                  │
│  [5] Tumbling Window     10-cycle majority-vote label smoother             │
│         │                                                                  │
│  [6] Forecasting         ARIMA/pmdarima TTF + confidence estimation        │
│         │                                                                  │
│  [7] Severity Update     Revised severity matrix (Impact x Urgency)        │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 📂 Repository Structure

```
AIOps_incident_management/
├── backend/                         # FastAPI, AI Pipeline & Data Generator
│   ├── api/                         # Dashboard API layer (FastAPI)
│   ├── app_data_generator/          # Telemetry simulator (produces SQLite DB)
│   │   ├── config.py                # Central constants & threshold configs
│   │   └── run_simulator.py         # Main simulator script
│   ├── nodes/                       # Pipeline nodes (Stages 1 to 7)
│   │   ├── collect/                 # Stage 1: Telemetry ingestion
│   │   ├── feature_engineering/     # Stage 2: Feature Extraction
│   │   ├── preliminary_severity/    # Stage 3: Severity Engine Node
│   │   │   ├── severity_config/     # Threshold yaml configs
│   │   │   ├── severity_engine/     # EMA & Hysteresis rules engine package
│   │   │   └── severity_node.py     # Stage 3 wrapper node
│   │   ├── classification/          # Stage 4: LightGBM ML Classifier
│   │   ├── tumbling_window/         # Stage 5: Prediction smoother
│   │   ├── forecasting/             # Stage 6: ARIMA TTF Forecaster
│   │   └── severity_update/         # Stage 7: Impact-Urgency matrix updater
│   ├── requirements.txt             # Python requirements
│   └── run_pipeline.py              # Main real-time pipeline runner
│
├── frontend/                        # React Dashboard
│   ├── src/                         # React components, charts & layout
│   ├── package.json                 # Node dependencies
│   └── vite.config.js               # Vite configurations
```

---

## 🛠️ Installation & Setup

### Prerequisites
* **Python**: `3.9` to `3.11`
* **Node.js**: `v18+` & `npm`

---

### 1. Backend Setup (FastAPI & Pipeline)

1. Open a terminal and navigate to the backend folder:
   ```bash
   cd backend
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
   ```

3. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

---

### 2. Frontend Setup (React Dashboard)

1. Open a new terminal and navigate to the frontend folder:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

---

## 🚀 Running the Platform

To run the entire system in real-time, launch the following processes in separate terminals:

### Terminal 1: Telemetry Data Generator
Simulates active incidents (such as memory leaks, CPU saturation, dependency timeouts) and writes telemetry to the SQLite DB.
```bash
cd backend
.venv\Scripts\activate
python app_data_generator/run_simulator.py
```

### Terminal 2: Pipeline Runner
Reads telemetry logs, runs features engineering, classifications, and tumbling windows.
```bash
cd backend
.venv\Scripts\activate
python run_pipeline.py
```

### Terminal 3: FastAPI Backend API Server
Serves the incident telemetry & predictions to the frontend.
```bash
cd backend
.venv\Scripts\activate
python api/main.py
```

### Terminal 4: React Frontend Client
Launches the web UI dashboard.
```bash
cd frontend
npm run dev
```
Open **[http://localhost:5173](http://localhost:5173)** in your browser to view the incident dashboard.

---

## 📊 Pipeline Stages & Details

| Stage | Node | Responsibilities | Algorithm / Technology |
|---|---|---|---|
| **1** | **Collect** | Fetches raw metrics, logs, and spans from the SQLite database. | SQL / Telemetry Queue |
| **2** | **Feature Engineering** | Computes 27 metrics & 5 log-based features. | Drain3 Log Parser & Rolling Stats |
| **3** | **Prelim Severity** | Scores current system severity (P1 to P4). | DEVOPS SeverityEngine (Hysteresis & EMA) |
| **4** | **Classification** | Detects active failure modes/incidents. | LightGBM ML Classifier |
| **5** | **Tumbling Window** | Smooths out flickering/intermittent ML classifications. | 10-Cycle Majority-Vote Smoother |
| **6** | **Forecasting** | Estimates Time-To-Failure (TTF) of the primary metrics. | ARIMA (`pmdarima` / `statsmodels`) |
| **7** | **Severity Update** | Merges Urgency (TTF) and Impact to revise the incident severity. | Combination Matrix |

---

## 💡 Troubleshooting
* **Missing Database Error**: If the pipeline or API reports a missing database, run the `run_simulator.py` script first to initialize the SQLite database at `backend/app_data_generator/output/simulator_db.sqlite`.
* **Database Reset**: You can clear or reset simulation state by deleting the SQLite database at `backend/app_data_generator/output/simulator_db.sqlite` and running `run_simulator.py` again.
