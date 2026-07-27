import React from "react";
import { CheckSquare } from "lucide-react";

export default function EvidenceCard({ incident }) {
  if (!incident) return null;

  return (
    <section className="card card-evidence">
      <div className="card-head">
        <h2>
          <span className="ico"><CheckSquare size={14} /></span>
          Evidence
        </h2>
        <span className="card-tag">Supporting metrics and logs.</span>
      </div>
      <ul className="evidence-list">
        <li>
          <span className="check">✔</span>
          <span className="ev-label">Critical Metric</span>
          <span className="ev-val mono">{incident.critical_metric}</span>
        </li>
        <li>
          <span className="check">✔</span>
          <span className="ev-label">Current Value</span>
          <span className="ev-val mono">{incident.metric_value}</span>
        </li>
        <li>
          <span className="check">✔</span>
          <span className="ev-label">Warning Count</span>
          <span className="ev-val mono">{incident.warning_count}</span>
        </li>
        <li>
          <span className="check">✔</span>
          <span className="ev-label">Critical Count</span>
          <span className="ev-val mono">{incident.critical_count}</span>
        </li>
        <li>
          <span className="check">✔</span>
          <span className="ev-label">Blast Size</span>
          <span className="ev-val mono">{incident.blast_size}</span>
        </li>
        <li>
          <span className="check">✔</span>
          <span className="ev-label">Classifier Conf</span>
          <span className="ev-val mono">
            {Math.round(incident.classifier_confidence * 100)}%
          </span>
        </li>
        <li>
          <span className="check">✔</span>
          <span className="ev-label">Trace Evidence</span>
          <span className="ev-val">{incident.trace_evidence}</span>
        </li>
        <li>
          <span className="check">✔</span>
          <span className="ev-label">Log Evidence</span>
          <span className="ev-val">{incident.log_evidence}</span>
        </li>
      </ul>
    </section>
  );
}
