import React from "react";
import { Radio } from "lucide-react";
import HumanGateSidebarSection from "./HumanGateSidebarSection";

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

export default function Sidebar({
  activeMode,
  isLiveFeed,
  onToggleLiveFeed,
  mobileSidebarOpen,
  setMobileSidebarOpen,
}) {
  return (
    <aside className={`sidebar ${mobileSidebarOpen ? "open" : ""}`}>
      {/* Brand Header */}
      <div className="sidebar-brand">
        <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
          <span className="brand-mark">◆</span>
          <div className="brand-text">
            <span className="brand-title">SENTINEL</span>
            <span className="brand-sub">AIOps Core</span>
          </div>
        </div>
        {/* Close Button visible on mobile screens */}
        <button
          className="mobile-sidebar-close"
          onClick={() => setMobileSidebarOpen(false)}
          aria-label="Close menu"
        >
          ✕
        </button>
      </div>

      {/* ─────────── LIVE FEED TOGGLE ─────────── */}
      <button
        id="live-feed-toggle-btn"
        onClick={onToggleLiveFeed}
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          gap: "8px",
          width: "calc(100% - 24px)",
          margin: "8px 12px 4px",
          padding: "9px 14px",
          borderRadius: "8px",
          border: isLiveFeed
            ? "1px solid rgba(34, 197, 94, 0.6)"
            : "1px solid rgba(34, 199, 222, 0.35)",
          background: isLiveFeed
            ? "rgba(34, 197, 94, 0.12)"
            : "rgba(34, 199, 222, 0.07)",
          color: isLiveFeed ? "#22c55e" : "#22c7de",
          fontSize: "11px",
          fontWeight: 700,
          letterSpacing: "0.6px",
          textTransform: "uppercase",
          cursor: "pointer",
          transition: "all 0.25s ease",
        }}
      >
        {isLiveFeed ? (
          <>
            <span
              style={{
                width: 7,
                height: 7,
                borderRadius: "50%",
                background: "#22c55e",
                boxShadow: "0 0 6px #22c55e",
                animation: "livePulse 1s ease-in-out infinite",
                flexShrink: 0,
              }}
            />
            Stop Live Feed
          </>
        ) : (
          <>
            <Radio size={12} />
            ▶ Start Live Feed
          </>
        )}
      </button>

      {isLiveFeed && (
        <div
          style={{
            margin: "0 12px 8px",
            padding: "6px 10px",
            borderRadius: "6px",
            background: "rgba(34, 197, 94, 0.07)",
            border: "1px solid rgba(34, 197, 94, 0.2)",
            fontSize: "9.5px",
            color: "rgba(34, 197, 94, 0.85)",
            lineHeight: "1.5",
          }}
        >
          🟢 Live feed active — showing real-time pipeline output from <strong>live_feed_db</strong>
        </div>
      )}

      {/* Trained Modes Health checklist */}
      <div className="sidebar-label">System Health Status</div>
      <div className="health-list">
        {MODE_ORDER.map((mode) => {
          const isIncident = mode === activeMode;
          const displayLabel = mode.replace(/_/g, " ");

          return (
            <div
              key={mode}
              className={`health-item ${isIncident ? "status-incident" : "status-healthy"}`}
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

      {/* ─────────── INLINE HUMAN GATE APPROVAL SECTION (Replaces Incident History) ─────────── */}
      <HumanGateSidebarSection />

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
