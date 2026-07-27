import React from "react";
import { ShieldAlert } from "lucide-react";

export default function DiagnosisCard({ incident }) {
  if (!incident) return null;

  return (
    <section className="card card-diagnosis">
      <div className="card-head">
        <h2>
          <span className="ico"><ShieldAlert size={14} /></span>
          Diagnosis
        </h2>
        <span className="card-tag">Explains why AI detected this incident.</span>
      </div>
      <p className="diagnosis-desc" title={incident.root_cause}>
        {incident.root_cause}
      </p>
      <div className="stat-row">
        <div className="stat-box">
          <span className="stat-num">
            {Math.round(incident.classifier_confidence * 100)}%
          </span>
          <span className="stat-name">Classifier Conf</span>
        </div>
        <div className="stat-box">
          <span className="stat-num warn">{incident.warning_count}</span>
          <span className="stat-name">Warning Count</span>
        </div>
        <div className="stat-box">
          <span className="stat-num crit">{incident.critical_count}</span>
          <span className="stat-name">Critical Count</span>
        </div>
        <div className="stat-box">
          <span className="stat-num">{incident.blast_size}</span>
          <span className="stat-name">Blast Size</span>
        </div>
      </div>
      <p className="micro-note">
        Diagnosis explains why this incident was detected using metrics, logs, traces and AI reasoning.
      </p>
    </section>
  );
}
