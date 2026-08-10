"""
backend/api/main.py
===================
FastAPI Web Server for AIOps Incident Detection Dashboard.
Provides real-time endpoints and past episode history.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

# Add parent dir (backend) to path for imports
_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from api.services import AIOpsDashboardService, HumanGateService, LiveFeedService
from api.sse_broadcaster import SSEBroadcaster

app = FastAPI(
    title="Sentinel AIOps Incident Detection API",
    description="Backend API serving incident summaries, forecasts and severity evaluations.",
    version="1.0.0"
)

# Enable CORS for frontend Vite dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache in-memory overrides for incident statuses
_status_overrides: Dict[str, str] = {}

# Pydantic models
class StatusUpdate(BaseModel):
    status: str

class GateDecision(BaseModel):
    decision: str    # "APPROVED" or "REJECTED"
    operator: str    # Reviewer username / name
    reason:   str = ""


@app.get("/api/health")
def health_check() -> Dict[str, str]:
    return {"status": "healthy", "service": "AIOps Sentinel Core"}


# =============================================================================
# Historical Pipeline Routes (unchanged)
# =============================================================================

@app.get("/api/live")
def get_live_incident() -> Dict[str, Any]:
    """Get the latest cycle's incident state from historical pipeline CSV outputs."""
    data = AIOpsDashboardService.get_live_state()
    if not data:
        raise HTTPException(status_code=404, detail="No active pipeline run telemetry found.")
    ep_id = data["episode_id"]
    if ep_id in _status_overrides:
        data["status"] = _status_overrides[ep_id]
    return data


@app.get("/api/episodes")
def list_episodes() -> List[Dict[str, Any]]:
    """Get list of all processed historical episodes."""
    return AIOpsDashboardService.get_all_episodes()


@app.get("/api/episodes/{episode_id}")
def get_episode(episode_id: str) -> Dict[str, Any]:
    """Get details for a specific historical episode."""
    data = AIOpsDashboardService.get_episode_details(episode_id)
    if not data:
        raise HTTPException(status_code=404, detail=f"Episode {episode_id} not found.")
    if episode_id in _status_overrides:
        data["status"] = _status_overrides[episode_id]
    return data


@app.patch("/api/incident/{episode_id}")
def update_incident_status(episode_id: str, body: StatusUpdate) -> Dict[str, str]:
    """Updates the status of an incident."""
    status = body.status.upper()
    if status not in ["OPEN", "ACKNOWLEDGED", "IN_PROGRESS", "RESOLVED"]:
        raise HTTPException(status_code=400, detail="Invalid status value.")
    _status_overrides[episode_id] = status
    return {"episode_id": episode_id, "status": status, "message": "Status updated successfully."}


@app.get("/api/reliability/summary")
def get_reliability_summary() -> Dict[str, Any]:
    """Return 4-group Weibull parameters, KM step points, and Weibull curves."""
    return AIOpsDashboardService.get_reliability_summary()


# =============================================================================
# Live Feed Routes — SSE + REST
# =============================================================================

@app.get("/api/live-stream")
async def live_stream():
    """
    Server-Sent Events endpoint — pushes latest pipeline state every 500ms.

    Frontend connects with:
        const es = new EventSource('http://localhost:8080/api/live-stream');
        es.onmessage = (e) => setIncident(JSON.parse(e.data));

    Returns text/event-stream (never closes until client disconnects).
    """
    return StreamingResponse(
        SSEBroadcaster.stream(interval_s=0.5),
        media_type="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",
            "Access-Control-Allow-Origin": "*",
        },
    )


@app.get("/api/live-feed/state")
def get_live_feed_state() -> Dict[str, Any]:
    """
    REST fallback — latest pipeline_results row from live_feed_db.sqlite.
    Used when SSE is unavailable (proxied environments).
    """
    data = LiveFeedService.get_live_feed_state()
    if not data:
        raise HTTPException(
            status_code=404,
            detail="No live feed data yet. Start run_live_feed.py and run_langgraph.py --live first."
        )
    return data


@app.get("/api/live-feed/episodes")
def get_live_feed_episodes() -> List[Dict[str, Any]]:
    """List all episodes from the live feed database (newest first)."""
    return LiveFeedService.get_live_feed_episodes()


# ── Live Process Manager (Start/Stop Live Feed & LangGraph) ──────────────────
_live_feed_proc = None
_langgraph_proc = None

@app.post("/api/live/start")
def start_live_feed(speed: float = 1.0) -> Dict[str, Any]:
    """Launch live telemetry simulator and LangGraph pipeline in background."""
    global _live_feed_proc, _langgraph_proc
    backend_dir = _HERE.parent

    if _live_feed_proc is None or _live_feed_proc.poll() is not None:
        sim_script = backend_dir / "Simulator" / "live_feed_simulator" / "run_live_feed.py"
        _live_feed_proc = subprocess.Popen(
            [sys.executable, str(sim_script), "--speed", str(speed)],
            cwd=str(backend_dir),
        )

    if _langgraph_proc is None or _langgraph_proc.poll() is not None:
        lg_script = backend_dir / "run_langgraph.py"
        _langgraph_proc = subprocess.Popen(
            [sys.executable, str(lg_script), "--live"],
            cwd=str(backend_dir),
        )

    return {
        "status": "started",
        "message": "Live feed simulator and LangGraph pipeline started.",
        "running": True,
    }

@app.post("/api/live/stop")
def stop_live_feed() -> Dict[str, Any]:
    """Terminate background live feed simulator and LangGraph pipeline."""
    global _live_feed_proc, _langgraph_proc

    if _live_feed_proc and _live_feed_proc.poll() is None:
        _live_feed_proc.terminate()
        _live_feed_proc = None

    if _langgraph_proc and _langgraph_proc.poll() is None:
        _langgraph_proc.terminate()
        _langgraph_proc = None

    return {
        "status": "stopped",
        "message": "Live feed simulation and LangGraph pipeline stopped.",
        "running": False,
    }

@app.get("/api/live/simulation-status")
@app.get("/api/live/status-check")
def live_simulation_status() -> Dict[str, Any]:
    """Check if background simulation and pipeline processes are running."""
    sim_running = _live_feed_proc is not None and _live_feed_proc.poll() is None
    lg_running = _langgraph_proc is not None and _langgraph_proc.poll() is None
    return {
        "running": sim_running or lg_running,
        "simulator_running": sim_running,
        "langgraph_running": lg_running,
    }


@app.get("/api/live-feed/status")
def get_live_feed_status() -> Dict[str, Any]:
    """Return live feed session metadata: queue depth, episode count, latest mode."""
    return LiveFeedService.get_live_feed_status()


# =============================================================================
# Human Gate Routes
# =============================================================================

@app.get("/api/human-gate/pending")
def get_pending_reviews() -> List[Dict[str, Any]]:
    """Return all Human Gate reviews currently awaiting operator decision."""
    return HumanGateService.get_pending_reviews()


@app.get("/api/human-gate/review/{review_id}")
def get_review(review_id: str) -> Dict[str, Any]:
    """Return full details of one pending review and mark it as REVIEWING."""
    data = HumanGateService.get_review(review_id)
    if not data:
        raise HTTPException(status_code=404, detail=f"Review {review_id} not found.")
    return data


@app.post("/api/human-gate/decision/{review_id}")
def submit_gate_decision(review_id: str, body: GateDecision) -> Dict[str, Any]:
    """Submit an operator's APPROVE or REJECT decision for a pending review."""
    decision = body.decision.upper().strip()
    if decision not in ("APPROVED", "REJECTED"):
        raise HTTPException(status_code=400, detail="Invalid decision. Use 'APPROVED' or 'REJECTED'.")
    result = HumanGateService.submit_decision(
        review_id = review_id,
        decision  = decision,
        operator  = body.operator or "operator",
        reason    = body.reason,
    )
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@app.get("/api/human-gate/metrics")
def get_gate_metrics() -> Dict[str, Any]:
    """Return Human Gate KPI metrics."""
    return HumanGateService.get_metrics()


@app.get("/api/human-gate/history")
def get_gate_history(limit: int = 50) -> List[Dict[str, Any]]:
    """Return recent Human Gate audit records (newest first)."""
    return HumanGateService.get_history(limit=limit)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8080, reload=True)
