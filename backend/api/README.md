# Sentinel API Server (Port 8080)

FastAPI Web Server providing REST endpoints, Server-Sent Events (SSE) telemetry streaming, Human Gate review endpoints, and live simulation process management.

---

## Files & Components

```
api/
├── main.py                    # FastAPI app declaration, CORS, REST/SSE routes & live process manager
├── services.py                # Service layer (AIOpsDashboardService, HumanGateService, LiveFeedService)
├── sse_broadcaster.py         # Thread-safe SSE broadcaster pushing telemetry every 500ms
└── README.md
```

---

## Key Endpoints

- `POST /api/live/start` — Launches live feed simulator and LangGraph pipeline.
- `POST /api/live/stop` — Terminates live feed simulator and LangGraph pipeline.
- `GET /api/live/simulation-status` — Returns live feed process status.
- `GET /api/live-stream` — Server-Sent Events (SSE) stream (`text/event-stream`).
- `GET /api/live-feed/state` — REST fallback for pipeline cycle snapshot.
- `GET /api/human-gate/pending` — Fetches pending P-level escalation reviews.
- `POST /api/human-gate/decision/{review_id}` — Submits operator decision (`APPROVED` / `REJECTED`).
