# Sentinel NOC Dashboard Frontend

A modern, high-performance React + Vite dashboard designed for real-time AIOps incident monitoring, failure forecasting, Weibull survival curve visualization, and human-in-the-loop incident mitigation.

---

## 🎨 Dashboard Features

1. **Interactive Live Feed Controller**:
   - Includes a **"Start Live Feed" / "Stop Live Feed"** button in the header.
   - Automatically triggers the backend live feed simulator and 10-node LangGraph pipeline with a single click.

2. **7 Core NOC Metric Trend Charts (Recharts)**:
   - Real-time interactive charts for **CPU Utilization**, **Heap Memory**, **P99 Latency**, **Database Latency**, **Error Rate**, **Cache Miss Rate**, and **Queue Lag**.
   - Highlights threshold breaches and projected breach trends.

3. **Real-Time Telemetry Streaming (SSE)**:
   - Primary data transport via Server-Sent Events (`GET http://localhost:8080/api/live-stream`).
   - Seamless REST fallback (`GET /api/live-feed/state`) when SSE is proxied.

4. **Incident Diagnosis & Prediction Cards**:
   - Displays preliminary severity (P1–P4), LightGBM predicted failure mode, Time-to-Failure (TTF) countdowns, log cluster evidence, and trace span diagnostics.

5. **Weibull Reliability Engine**:
   - Real-time $S(t)$ survival probability gauge and 4-group Weibull parameter distribution modal.

6. **Right-Drawer Natural Language AI Assistant**:
   - Natural language query drawer powered by the QueryLangGraph LLM pipeline (`http://localhost:8001`).

---

## 🛠️ Tech Stack

- **Framework**: React 18 + Vite
- **Styling**: Vanilla CSS (Tailwind utilities + Dark Glassmorphism aesthetic)
- **Charts**: Recharts
- **Icons**: Lucide React

---

## 🚀 Running the Frontend

### 1. Install Dependencies
```powershell
cd frontend
npm install
```

### 2. Start Dev Server
```powershell
npm run dev
```

The application will launch on **http://localhost:5173**.

---

## 🔌 API Endpoints Consumed

- `POST http://localhost:8080/api/live/start` — Triggers backend simulator and LangGraph pipeline.
- `POST http://localhost:8080/api/live/stop` — Stops backend simulator and LangGraph pipeline.
- `GET http://localhost:8080/api/live/simulation-status` — Syncs live simulation status on page reload.
- `GET http://localhost:8080/api/live-stream` — Real-time Server-Sent Events (SSE) telemetry stream.
- `POST http://localhost:8001/query` — AI Assistant query endpoint.
