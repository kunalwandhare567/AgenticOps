import React from "react";
import { X, RefreshCw, Activity, Info, ShieldCheck, Zap, AlertTriangle, Layers } from "lucide-react";

export default function ReliabilityGraphSection({
  reliabilitySummary,
  incident,
  refreshCountdown = 20,
  onClose,
}) {
  if (!reliabilitySummary) return null;

  const activeGroupName = incident?.active_group_name || "Progressive resource degradation";
  const elapsedS = incident?.elapsed_s || 0;
  const activeBeta = incident?.active_beta || 2.0;
  const activeEta = incident?.active_eta || 46.5;

  const groupKeys = [
    "Immediate trigger",
    "Fast accumulation",
    "Progressive resource degradation",
    "Slow or latent degradation",
  ];

  const groupTitles = {
    "Immediate trigger": "Group 1: Immediate Trigger",
    "Fast accumulation": "Group 2: Fast Accumulation",
    "Progressive resource degradation": "Group 3: Progressive Resource Degradation",
    "Slow or latent degradation": "Group 4: Slow or Latent Degradation",
  };

  // Render SVG Chart for one group in the 2x2 grid
  const renderSubplot = (groupKey) => {
    const groupData = reliabilitySummary[groupKey];
    if (!groupData) {
      return (
        <div className="rel-grid-card empty">
          <span>No data for {groupKey}</span>
        </div>
      );
    }

    const {
      km_points = [],
      weibull_points = [],
      beta,
      eta,
      n,
      events,
      censored,
      note,
      suitability,
      badge_color,
    } = groupData;

    const width = 340;
    const height = 150;
    const padding = 25;
    const maxT = 240;

    const scaleX = (t) => padding + (Math.min(t, maxT) / maxT) * (width - 2 * padding);
    const scaleY = (s) => height - padding - s * (height - 2 * padding);

    // Build Weibull curve path
    let weibullPath = "";
    weibull_points.forEach((pt, i) => {
      const x = scaleX(pt.t);
      const y = scaleY(pt.s);
      weibullPath += i === 0 ? `M ${x} ${y}` : ` L ${x} ${y}`;
    });

    // Build KM step path & confidence band
    let kmPath = "";
    let bandPathUpper = "";
    let bandPathLower = "";

    km_points.forEach((pt, i) => {
      const x = scaleX(pt.t);
      const y = scaleY(pt.s);
      const yUpper = scaleY(pt.upper_ci);
      const yLower = scaleY(pt.lower_ci);

      if (i === 0) {
        kmPath += `M ${x} ${y}`;
        bandPathUpper += `M ${x} ${yUpper}`;
        bandPathLower += `L ${x} ${yLower}`;
      } else {
        kmPath += ` L ${x} ${scaleY(km_points[i - 1].s)} L ${x} ${y}`;
        bandPathUpper += ` L ${x} ${scaleY(km_points[i - 1].upper_ci)} L ${x} ${yUpper}`;
        bandPathLower = `L ${x} ${yLower} L ${x} ${scaleY(km_points[i - 1].lower_ci)} ` + bandPathLower;
      }
    });

    const confidenceBand = bandPathUpper + " " + bandPathLower + " Z";

    const isGroupActive = groupKey === activeGroupName;
    const markerX = scaleX(elapsedS);
    const liveRatio = Math.exp(-Math.pow(elapsedS / activeEta, activeBeta));
    const markerY = scaleY(Math.max(0, Math.min(1, liveRatio)));

    return (
      <div key={groupKey} className={`rel-grid-card ${isGroupActive ? "is-active-group" : ""}`}>
        {/* Subplot Header */}
        <div className="rel-grid-card-head">
          <div className="grid-head-title-wrap">
            <span className="grid-card-title">{groupTitles[groupKey]}</span>
            {isGroupActive && (
              <span className="active-incident-pill" title="Currently Monitored Incident Belongs to This Group">
                <span className="pulse-dot red"></span> Active Monitored Incident
              </span>
            )}
          </div>
          <span className={`suitability-badge ${badge_color}`}>{suitability}</span>
        </div>

        {/* Stats Row */}
        <div className="rel-grid-stats-row">
          <span className="stat-pill">N = <strong>{n}</strong></span>
          <span className="stat-pill">Events = <strong>{events}</strong></span>
          <span className="stat-pill">Censored = <strong>{censored}</strong></span>
          <span className="stat-pill highlight">β = <strong>{beta}</strong></span>
          <span className="stat-pill highlight">η = <strong>{eta}s</strong></span>
        </div>

        {/* SVG Curve Canvas */}
        <div className="rel-grid-svg-container">
          <svg viewBox={`0 0 ${width} ${height}`} className="rel-grid-svg">
            {/* Gridlines */}
            <line x1={padding} y1={scaleY(0.75)} x2={width - padding} y2={scaleY(0.75)} stroke="#1e293b" strokeDasharray="2,2" />
            <line x1={padding} y1={scaleY(0.50)} x2={width - padding} y2={scaleY(0.50)} stroke="#334155" strokeDasharray="3,3" />
            <line x1={padding} y1={scaleY(0.25)} x2={width - padding} y2={scaleY(0.25)} stroke="#1e293b" strokeDasharray="2,2" />
            <line x1={padding} y1={scaleY(1.0)} x2={width - padding} y2={scaleY(1.0)} stroke="#475569" />
            <line x1={padding} y1={scaleY(0.0)} x2={width - padding} y2={scaleY(0.0)} stroke="#475569" />

            {/* X-axis tick labels */}
            <text x={padding} y={height - 5} fill="#64748b" fontSize="8" textAnchor="middle">0s</text>
            <text x={scaleX(60)} y={height - 5} fill="#64748b" fontSize="8" textAnchor="middle">60s</text>
            <text x={scaleX(120)} y={height - 5} fill="#64748b" fontSize="8" textAnchor="middle">120s</text>
            <text x={scaleX(180)} y={height - 5} fill="#64748b" fontSize="8" textAnchor="middle">180s</text>
            <text x={width - padding} y={height - 5} fill="#64748b" fontSize="8" textAnchor="middle">238s</text>

            {/* Y-axis tick labels */}
            <text x={padding - 5} y={scaleY(1.0) + 3} fill="#64748b" fontSize="8" textAnchor="end">1.0</text>
            <text x={padding - 5} y={scaleY(0.5) + 3} fill="#64748b" fontSize="8" textAnchor="end">0.5</text>
            <text x={padding - 5} y={scaleY(0.0) + 3} fill="#64748b" fontSize="8" textAnchor="end">0.0</text>

            {/* Greenwood Confidence Band Area */}
            {bandPathUpper && (
              <path d={confidenceBand} fill="#38bdf8" fillOpacity="0.15" />
            )}

            {/* Kaplan-Meier Step Line (Empirical) */}
            <path d={kmPath} fill="none" stroke="#38bdf8" strokeWidth="2" strokeLinecap="round" />

            {/* Weibull Theoretical Curve (Dashed) */}
            <path d={weibullPath} fill="none" stroke="#f97316" strokeWidth="1.8" strokeDasharray="4,3" />

            {/* Real-time live marker dot when this group is active */}
            {isGroupActive && elapsedS > 0 && elapsedS <= 238 && (
              <g>
                <circle cx={markerX} cy={markerY} r="6" fill="#ef4444" opacity="0.4" />
                <circle cx={markerX} cy={markerY} r="4" fill="#ef4444" />
                <circle cx={markerX} cy={markerY} r="2" fill="#ffffff" />
              </g>
            )}
          </svg>
        </div>

        {/* Subplot Commentary Note */}
        <div className="rel-grid-note-box">
          <Info size={11} className="note-icon" />
          <span className="note-text">{note}</span>
        </div>
      </div>
    );
  };

  return (
    <div className="rel-modal-backdrop">
      <div className="rel-modal-container">
        {/* Modal Header */}
        <div className="rel-modal-header">
          <div className="modal-title-group">
            <div className="icon-badge">
              <Activity size={18} />
            </div>
            <div>
              <h2 className="modal-main-title">AIOps Reliability Engine — 4-Group Stratified Analysis</h2>
              <p className="modal-subtitle">
                Live Kaplan–Meier Empirical Steps vs 2-Parameter Weibull MLE Fits
              </p>
            </div>
          </div>

          <div className="modal-header-actions">
            {/* Live 20s Countdown Timer Badge */}
            <div className="auto-refresh-pill">
              <RefreshCw size={12} className={refreshCountdown === 20 ? "spin-once" : ""} />
              <span>Updating in <strong>{refreshCountdown}s</strong></span>
            </div>

            {/* Close Button */}
            {onClose && (
              <button className="modal-close-btn" onClick={onClose}>
                <X size={18} />
              </button>
            )}
          </div>
        </div>

        {/* Legend Bar */}
        <div className="rel-modal-legend-bar">
          <span className="lg-item"><span className="lg-line km"></span> Kaplan–Meier Empirical Step</span>
          <span className="lg-item"><span className="lg-band"></span> 95% Greenwood Confidence Band</span>
          <span className="lg-item"><span className="lg-line weibull"></span> Weibull Theoretical Fit</span>
          <span className="lg-item"><span className="lg-dot live"></span> Live Monitored Incident</span>
        </div>

        {/* 2x2 Subplot Grid */}
        <div className="rel-2x2-grid">
          {groupKeys.map((key) => renderSubplot(key))}
        </div>
      </div>
    </div>
  );
}
