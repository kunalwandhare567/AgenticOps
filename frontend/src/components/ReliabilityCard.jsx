import React from "react";
import { Activity } from "lucide-react";

export default function ReliabilityCard({ incident }) {
  if (!incident) return null;

  const mtbfHours = incident.mtbf_hours || 100;
  const mttrHours = incident.mttr_hours || 1.0;
  const weibull = incident.weibull_confidence || 0.70;
  const weibullPct = Math.round(weibull * 100);

  return (
    <section className="card card-reliability">
      <div className="card-head">
        <h2>
          <span className="ico"><Activity size={14} /></span>
          Reliability
        </h2>
        <span className="card-tag">Historical health metrics.</span>
      </div>
      
      <div className="rel-bar-item">
        <div className="rel-bar-label">
          <span>MTBF</span>
          <span className="mono">{mtbfHours} h</span>
        </div>
        <div className="track">
          <div
            className="track-fill fill-blue"
            style={{ width: `${Math.min((mtbfHours / 200) * 100, 100)}%` }}
          ></div>
        </div>
      </div>
      
      <div className="rel-bar-item">
        <div className="rel-bar-label">
          <span>MTTR</span>
          <span className="mono">{mttrHours} h</span>
        </div>
        <div className="track">
          <div
            className="track-fill fill-amber"
            style={{ width: `${Math.min((mttrHours / 3.0) * 100, 100)}%` }}
          ></div>
        </div>
      </div>
      
      <div className="rel-bar-item">
        <div className="rel-bar-label">
          <span>Weibull Confidence</span>
          <span className="mono">{weibullPct}%</span>
        </div>
        <div className="track">
          <div
            className="track-fill fill-violet"
            style={{ width: `${weibullPct}%` }}
          ></div>
        </div>
      </div>
      
      <div className="rel-trend-line" style={{ marginTop: "auto" }}>
        <span className="f-label">Reliability Trend</span>
        <span className={`f-value ${incident.reliability_trend?.toLowerCase() === "degrading" ? "crit" : "good"}`}>
          {incident.reliability_trend || "Stable"}
        </span>
      </div>
    </section>
  );
}
