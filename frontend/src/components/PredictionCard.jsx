import React from "react";
import { Zap } from "lucide-react";

export default function PredictionCard({ incident }) {
  if (!incident) return null;

  const ttfSeconds = incident.ttf_seconds;
  const confidence = incident.prediction_confidence;
  const confidencePct = Math.round(confidence * 100);

  const formatTTF = (s) => {
    if (s === null || s === undefined) return "Failure Probability Available";
    if (s === 0.0) return "BREACHED NOW";
    if (s < 60) return `${Math.round(s)}s`;
    const m = Math.floor(s / 60);
    const rem = Math.round(s % 60);
    return `${m}m ${rem}s`;
  };

  return (
    <section className="card card-prediction">
      <div className="card-head">
        <h2>
          <span className="ico"><Zap size={14} /></span>
          Prediction
        </h2>
        <span className="card-tag">Forecasts the likely future outcome.</span>
      </div>
      <div className="pred-row">
        <div>
          <span className="f-label">Predicted Failure</span>
          <span className="f-value" style={{ fontSize: "12.5px" }}>{incident.predicted_failure}</span>
        </div>
        <div>
          <span className="f-label">Time To Failure</span>
          <span className="f-value mono ttf">{formatTTF(ttfSeconds)}</span>
        </div>
      </div>
      <div className="confidence-track">
        <span className="f-label">Prediction Confidence</span>
        <div className="track">
          <div
            className="track-fill"
            style={{ width: `${confidencePct}%` }}
          ></div>
        </div>
        <span className="mono track-pct">{confidencePct}%</span>
      </div>
      <div className="rec-action-line">
        <span className="f-label">Recommended Action</span>
        <span className="f-value">{incident.recommended_action}</span>
      </div>
    </section>
  );
}
