import React, { useState, useEffect } from "react";
import { Activity } from "lucide-react";

export default function TopHeader() {
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
      <div className="topbar-right">
        <div className="status-chip">
          <Activity size={14} className="status-dot" style={{ animation: "pulse 2s infinite" }} />
          AI Monitoring Active
        </div>
        <div className="clock-chip">
          <span id="clockTime">{time || "--:--:--"}</span>
          <span className="clock-date" id="clockDate">{date || "--"}</span>
        </div>
      </div>
    </header>
  );
}
