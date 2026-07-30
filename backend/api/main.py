"""
backend/api/main.py
===================
FastAPI Web Server for AIOps Incident Detection Dashboard.
Provides real-time endpoints and past episode history.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

# Add parent dir (backend) to path for imports
_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from api.services import AIOpsDashboardService, HumanGateService

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
    reason:   str = ""  # Optional comment (required on REJECT is a UI concern)

@app.get("/api/health")
def health_check() -> Dict[str, str]:
    return {"status": "healthy", "service": "AIOps Sentinel Core"}

@app.get("/api/live")
def get_live_incident() -> Dict[str, Any]:
    """Get the latest cycle's incident state from active pipeline outputs."""
    data = AIOpsDashboardService.get_live_state()
    if not data:
        raise HTTPException(status_code=404, detail="No active pipeline run telemetry found. Please start run_pipeline.py first.")
    
    # Apply status override if user modified it
    ep_id = data["episode_id"]
    if ep_id in _status_overrides:
        data["status"] = _status_overrides[ep_id]
        
    return data

@app.get("/api/episodes")
def list_episodes() -> List[Dict[str, Any]]:
    """Get list of all processed episodes."""
    return AIOpsDashboardService.get_all_episodes()

@app.get("/api/episodes/{episode_id}")
def get_episode(episode_id: str) -> Dict[str, Any]:
    """Get details for a specific episode."""
    data = AIOpsDashboardService.get_episode_details(episode_id)
    if not data:
        raise HTTPException(status_code=404, detail=f"Episode {episode_id} not found.")
    
    # Apply status override if user modified it
    if episode_id in _status_overrides:
        data["status"] = _status_overrides[episode_id]
        
    return data

@app.patch("/api/incident/{episode_id}")
def update_incident_status(episode_id: str, body: StatusUpdate) -> Dict[str, str]:
    """Updates the status of an incident (e.g. from OPEN to ACKNOWLEDGED)."""
    status = body.status.upper()
    if status not in ["OPEN", "ACKNOWLEDGED", "IN_PROGRESS", "RESOLVED"]:
        raise HTTPException(status_code=400, detail="Invalid status value. Choose: OPEN, ACKNOWLEDGED, IN_PROGRESS, RESOLVED")
    
    _status_overrides[episode_id] = status
    return {"episode_id": episode_id, "status": status, "message": "Status updated successfully."}

# =============================================================================
# Human Gate Routes
# =============================================================================

@app.get("/api/human-gate/pending")
def get_pending_reviews() -> List[Dict[str, Any]]:
    """
    Return all Human Gate reviews currently awaiting operator decision.
    Frontend polls this every 2 s to show the HumanGatePanel badge.
    """
    return HumanGateService.get_pending_reviews()


@app.get("/api/human-gate/review/{review_id}")
def get_review(review_id: str) -> Dict[str, Any]:
    """
    Return full details of one pending review and mark it as REVIEWING.
    Called when the operator opens the review panel.
    """
    data = HumanGateService.get_review(review_id)
    if not data:
        raise HTTPException(status_code=404, detail=f"Review {review_id} not found.")
    return data


@app.post("/api/human-gate/decision/{review_id}")
def submit_gate_decision(review_id: str, body: GateDecision) -> Dict[str, Any]:
    """
    Submit an operator's APPROVE or REJECT decision for a pending review.
    The pipeline runner polling poll_for_decision() will see the status change
    within its next 0.1-second poll cycle and resume.

    Body:
        decision : "APPROVED" | "REJECTED"
        operator : reviewer name/username
        reason   : optional comment (required on REJECT by UI convention)
    """
    decision = body.decision.upper().strip()
    if decision not in ("APPROVED", "REJECTED"):
        raise HTTPException(
            status_code=400,
            detail="Invalid decision. Use 'APPROVED' or 'REJECTED'."
        )
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
    """
    Return Human Gate KPI metrics: approval %, rejection %, auto-approval %,
    avg response time, false escalation count.
    """
    return HumanGateService.get_metrics()


@app.get("/api/human-gate/history")
def get_gate_history(limit: int = 50) -> List[Dict[str, Any]]:
    """
    Return recent Human Gate audit records (newest first).
    Used for historical analysis and offline model improvement.
    """
    return HumanGateService.get_history(limit=limit)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8080, reload=True)
