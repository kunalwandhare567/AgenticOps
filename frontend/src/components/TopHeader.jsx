import React, { useState, useEffect } from "react";
import { Activity, Radio, Play, Square } from "lucide-react";

export default function TopHeader({
  isLiveFeed = false,
  liveFeedMode = "",
  liveFeedTick = 0,
  onToggleLiveFeed = () => {},
}) {
  const [time, setTime] = useState("");
  const [date, setDate] = useState("");

  useEffect(() => {
    const updateTime = () => {
      const now = new Date();
      setTime(now.toLocaleTimeString("en-US", { hour12: false }));
      setDate(
        now.toLocaleDateString("en-US", {
          month: "short",
          day: "numeric",
          year: "numeric",
        })
      );
    };
    updateTime();
    const interval = setInterval(updateTime, 1000);
    return () => clearInterval(interval);
  }, []);

  return (
    <header className="topbar">
      <div className="topbar-left">
        <h1>AIOps Incident Detection Dashboard</h1>
        <span className="topbar-subtitle">Real-time failure classification &amp; forecasting</span>
      </div>
      <div className="topbar-right" style={{ display: "flex", alignItems: "center", gap: "12px" }}>

        {/* Interactive Start / Stop Live Feed Button */}
        <button
          id="toggle-live-feed-btn"
          onClick={onToggleLiveFeed}
          title={isLiveFeed ? "Stop live feed simulation & LangGraph runner" : "Start live feed simulation & LangGraph runner"}
          style={{
            display: "flex",
            alignItems: "center",
            gap: "6px",
            padding: "6px 14px",
            borderRadius: "8px",
            fontSize: "12px",
            fontWeight: 600,
            cursor: "pointer",
            border: isLiveFeed ? "1px solid rgba(239, 68, 68, 0.5)" : "1px solid rgba(34, 197, 94, 0.5)",
            background: isLiveFeed ? "rgba(239, 68, 68, 0.15)" : "rgba(34, 197, 94, 0.15)",
            color: isLiveFeed ? "#ef4444" : "#22c55e",
            transition: "all 0.2s ease",
            boxShadow: isLiveFeed ? "0 0 10px rgba(239, 68, 68, 0.2)" : "0 0 10px rgba(34, 197, 94, 0.2)",
          }}
        >
          {isLiveFeed ? (
            <>
              <Square size={13} fill="currentColor" />
              <span>Stop Live Feed</span>
            </>
          ) : (
            <>
              <Play size={13} fill="currentColor" />
              <span>Start Live Feed</span>
            </>
          )}
        </button>

        {/* Live Feed Badge — shown when live mode is active */}
        {isLiveFeed && (
          <div
            className="status-chip"
            style={{
              background: "rgba(34, 197, 94, 0.15)",
              border: "1px solid rgba(34, 197, 94, 0.45)",
              color: "#22c55e",
              animation: "livePulse 1.5s ease-in-out infinite",
              gap: "6px",
              fontWeight: 700,
            }}
          >
            <span
              style={{
                display: "inline-block",
                width: 8,
                height: 8,
                borderRadius: "50%",
                background: "#22c55e",
                boxShadow: "0 0 8px #22c55e",
                animation: "livePulse 1s ease-in-out infinite",
              }}
            />
            LIVE FEED ACTIVE
            {liveFeedMode && (
              <span style={{ opacity: 0.8, fontSize: "10px", fontWeight: 500 }}>
                &nbsp;·&nbsp;{liveFeedMode.replace(/_/g, " ")}
              </span>
            )}
            {liveFeedTick > 0 && (
              <span style={{ opacity: 0.6, fontSize: "10px" }}>
                &nbsp;tick&nbsp;{liveFeedTick}
              </span>
            )}
          </div>
        )}

        {/* AI Monitoring status chip — shown when NOT in live feed mode */}
        {!isLiveFeed && (
          <div className="status-chip">
            <Activity size={14} className="status-dot" style={{ animation: "pulse 2s infinite" }} />
            AI Monitoring Active
          </div>
        )}

        <div className="clock-chip">
          <span id="clockTime">{time || "--:--:--"}</span>
          <span className="clock-date" id="clockDate">{date || "--"}</span>
        </div>
      </div>
    </header>
  );
}
