import React from "react";
import { AlertCircle } from "lucide-react";

const SEVERITY_BADGES = {
  P1: "badge p1",
  P2: "badge p2",
  P3: "badge p3",
  P4: "badge p4",
};

export default function SummaryCard({ incident, onStatusChange }) {
  if (!incident) return null;

  return (
    <section className="card card-summary">
      <div className="card-head">
        <h2>
          <span className="ico"><AlertCircle size={14} /></span>
          Incident Summary
        </h2>
      </div>
      <div className="summary-grid">
        <div className="summary-field">
          <span className="f-label">Episode ID</span>
          <span className="f-value mono" style={{ fontSize: "11px" }}>
            {incident.episode_id}
          </span>
        </div>
        <div className="summary-field">
          <span className="f-label">Incident ID</span>
          <span className="f-value mono">{incident.incident_id}</span>
        </div>
        <div className="summary-field">
          <span className="f-label">Failure Mode</span>
          <span className="f-value" style={{ fontSize: "12px" }}>{incident.label}</span>
        </div>
        <div className="summary-field">
          <span className="f-label">Severity</span>
          <span className={SEVERITY_BADGES[incident.severity] || "badge p4"}>
            {incident.severity}
          </span>
        </div>
        <div className="summary-field">
          <span className="f-label">Status</span>
          <select
            className="status-dropdown"
            value={incident.status}
            onChange={(e) => onStatusChange(e.target.value)}
          >
            <option value="OPEN">Open</option>
            <option value="ACKNOWLEDGED">Acknowledged</option>
            <option value="IN_PROGRESS">In Progress</option>
            <option value="RESOLVED">Resolved</option>
          </select>
        </div>
        <div className="summary-field">
          <span className="f-label">Detected Cycle</span>
          <span className="f-value mono">Cycle {incident.cycle}</span>
        </div>
      </div>
    </section>
  );
}
