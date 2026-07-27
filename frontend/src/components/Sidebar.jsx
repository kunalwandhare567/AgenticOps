import React from "react";
import { Terminal, Database, Activity, RefreshCw } from "lucide-react";

const MODE_ORDER = [
  "MEMORY_LEAK",
  "CPU_SATURATION",
  "LATENCY_SPIKE",
  "ERROR_STORM",
  "DB_SLOWDOWN",
  "QUEUE_BACKUP",
  "DEPENDENCY_TIMEOUT",
  "BAD_DEPLOY",
  "RETRY_STORM",
  "DISK_IO_SATURATION",
  "CASCADING_FAILURE",
  "UNKNOWN",
];

const SEVERITY_CLASSES = {
  P1: "p1",
  P2: "p2",
  P3: "p3",
  P4: "p4",
};

export default function Sidebar({
  activeMode,
  episodes,
  selectedEpisodeId,
  onSelectEpisode,
  onViewLive,
  isLive,
}) {
  return (
    <aside className="sidebar">
      {/* Brand Header */}
      <div className="sidebar-brand">
        <span className="brand-mark">◆</span>
        <div className="brand-text">
          <span className="brand-title">SENTINEL</span>
          <span className="brand-sub">AIOps Core</span>
        </div>
      </div>

      {/* Trained Modes Health checklist */}
      <div className="sidebar-label">System Health Status</div>
      <div className="health-list">
        {MODE_ORDER.map((mode) => {
          const isIncident = mode === activeMode;
          const displayLabel = mode.replace(/_/g, " ").title || mode.replace(/_/g, " ");

          return (
            <div
              key={mode}
              className={`health-item ${
                isIncident ? "status-incident" : "status-healthy"
              }`}
            >
              <span className="dot"></span>
              <span style={{ textTransform: "capitalize" }}>
                {displayLabel.toLowerCase()}
              </span>
              <span className="health-tick">
                {isIncident ? "Incident Detected" : "Healthy"}
              </span>
            </div>
          );
        })}
      </div>

      {/* Incident History List */}
      <div
        className="sidebar-label"
        style={{
          marginTop: "15px",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
        }}
      >
        <span>Incident History</span>
        {!isLive && (
          <button
            onClick={onViewLive}
            style={{
              background: "none",
              border: "none",
              color: "#22c7de",
              fontSize: "9.5px",
              fontWeight: 700,
              cursor: "pointer",
              display: "flex",
              alignItems: "center",
              gap: "3px",
              padding: 0,
              textTransform: "uppercase",
              letterSpacing: "0.5px",
            }}
          >
            <RefreshCw size={10} /> Live
          </button>
        )}
      </div>
      <div className="history-list">
        {episodes.map((ep) => {
          const isActive = ep.episode_id === selectedEpisodeId;
          const label = ep.failure_mode.replace(/_/g, " ");
          const sevClass = SEVERITY_CLASSES[ep.severity] || "p4";

          return (
            <div
              key={ep.episode_id}
              className={`history-item ${isActive ? "active" : ""}`}
              onClick={() => onSelectEpisode(ep.episode_id)}
            >
              <div className="history-item-top">
                <span className="history-id">
                  {ep.episode_id.split("_")[0] + "_" + ep.episode_id.split("_")[1]}
                </span>
                <span className={`badge ${sevClass} history-sev`}>
                  {ep.severity}
                </span>
              </div>
              <span className="history-mode" style={{ textTransform: "capitalize" }}>
                {label.toLowerCase()}
              </span>
            </div>
          );
        })}
      </div>

      {/* Footer info */}
      <div className="sidebar-footer">
        <div className="sidebar-footer-row">
          <span className="pulse-dot"></span>
          <span>Model v2.4.2 · online</span>
        </div>
      </div>
    </aside>
  );
}
