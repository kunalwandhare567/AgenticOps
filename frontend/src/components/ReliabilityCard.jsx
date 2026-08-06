import React, { useState, useEffect } from "react";
import { Activity, ShieldAlert, CheckCircle2, AlertTriangle, Layers, Info } from "lucide-react";

export default function ReliabilityCard({
  incident,
  reliabilitySummary,
  refreshCountdown = 20,
  onOpenModal,
}) {
  const activeGroupName = incident?.active_group_name || "Progressive resource degradation";
  const [selectedGroup, setSelectedGroup] = useState(activeGroupName);

  // Auto-sync selected group tab when active incident changes (if live)
  useEffect(() => {
    if (activeGroupName && reliabilitySummary && reliabilitySummary[activeGroupName]) {
      setSelectedGroup(activeGroupName);
    }
  }, [activeGroupName, reliabilitySummary]);

  if (!incident) return null;


  // Group definitions order
  const groupKeys = [
    "Immediate trigger",
    "Fast accumulation",
    "Progressive resource degradation",
    "Slow or latent degradation",
  ];

  const shortNames = {
    "Immediate trigger": "Immediate",
    "Fast accumulation": "Fast",
    "Progressive resource degradation": "Progressive",
    "Slow or latent degradation": "Slow / Latent",
  };

  // Extract selected group data
  const currentGroupData = reliabilitySummary ? reliabilitySummary[selectedGroup] : null;

  // Active incident metrics
  const activeBeta = incident.active_beta ?? (currentGroupData?.beta || 2.0);
  const activeEta = incident.active_eta ?? (currentGroupData?.eta || 46.5);
  const currentSurvivalPct = incident.current_survival_pct ?? 95.0;
  const elapsedS = incident.elapsed_s || 0;

  // Render SVG Chart for Kaplan-Meier vs Weibull
  const renderChart = () => {
    if (!currentGroupData) {
      return (
        <div className="rel-chart-placeholder">
          <span>Loading 4-group life data...</span>
        </div>
      );
    }

    const { km_points = [], weibull_points = [] } = currentGroupData;

    const width = 280;
    const height = 120;
    const padding = 20;

    const maxT = 240; // observation window 238s

    const scaleX = (t) => padding + (Math.min(t, maxT) / maxT) * (width - 2 * padding);
    const scaleY = (s) => height - padding - s * (height - 2 * padding);

    // Build Weibull smooth path
    let weibullPath = "";
    weibull_points.forEach((pt, i) => {
      const x = scaleX(pt.t);
      const y = scaleY(pt.s);
      weibullPath += i === 0 ? `M ${x} ${y}` : ` L ${x} ${y}`;
    });

    // Build Kaplan-Meier step path & confidence band path
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
        const prevX = scaleX(km_points[i - 1].t);
        // Step function: horizontal then vertical
        kmPath += ` L ${x} ${scaleY(km_points[i - 1].s)} L ${x} ${y}`;
        bandPathUpper += ` L ${x} ${scaleY(km_points[i - 1].upper_ci)} L ${x} ${yUpper}`;
        bandPathLower = `L ${x} ${yLower} L ${x} ${scaleY(km_points[i - 1].lower_ci)} ` + bandPathLower;
      }
    });

    const confidenceBand = bandPathUpper + " " + bandPathLower + " Z";

    // Current live marker position on Weibull curve
    const markerX = scaleX(elapsedS);
    const liveRatio = Math.exp(-Math.pow(elapsedS / activeEta, activeBeta));
    const markerY = scaleY(Math.max(0, Math.min(1, liveRatio)));

    const isSelectedActive = selectedGroup === activeGroupName;

    return (
      <div className="rel-chart-wrapper">
        <svg viewBox={`0 0 ${width} ${height}`} className="rel-svg-chart">
          {/* Grid lines */}
          <line x1={padding} y1={scaleY(0.5)} x2={width - padding} y2={scaleY(0.5)} stroke="#2a2e3d" strokeDasharray="3,3" />
          <line x1={padding} y1={scaleY(1.0)} x2={width - padding} y2={scaleY(1.0)} stroke="#2a2e3d" />

          {/* Greenwood Confidence Band Area */}
          {bandPathUpper && (
            <path d={confidenceBand} fill="#3b82f6" fillOpacity="0.12" />
          )}

          {/* Kaplan-Meier Step Curve (Empirical) */}
          <path d={kmPath} fill="none" stroke="#38bdf8" strokeWidth="2" strokeLinecap="round" />

          {/* Weibull Theoretical Curve (Dashed) */}
          <path d={weibullPath} fill="none" stroke="#f97316" strokeWidth="1.8" strokeDasharray="4,3" />

          {/* Real-time live marker dot (when viewing active incident's group) */}
          {isSelectedActive && elapsedS > 0 && elapsedS <= 238 && (
            <g className="rel-live-dot-group">
              <circle cx={markerX} cy={markerY} r="5" fill="#ef4444" className="pulse-dot" />
              <circle cx={markerX} cy={markerY} r="2.5" fill="#ffffff" />
            </g>
          )}
        </svg>

        <div className="rel-chart-legend">
          <span className="legend-item"><span className="legend-line km"></span> KM Step</span>
          <span className="legend-item"><span className="legend-line weibull"></span> Weibull Fit</span>
          {isSelectedActive && <span className="legend-item"><span className="legend-dot live"></span> Live (t={Math.round(elapsedS)}s)</span>}
        </div>
      </div>
    );
  };

  return (
    <section className="card card-reliability">
      <div className="card-head">
        <h2>
          <span className="ico"><Activity size={14} /></span>
          Reliability Engine
        </h2>
        <span className="timer-badge-pill" title="Auto-refreshes 4-group Weibull data every 20 seconds">
          20s Live ({refreshCountdown}s)
        </span>
      </div>

      {/* Active Incident Group Highlight */}
      <div className="rel-active-banner">
        <div className="banner-left">
          <span className="banner-label">Active Failure Group</span>
          <span className="banner-group-title">{activeGroupName}</span>
        </div>
        <div className="banner-right">
          <span className={`status-pill ${incident.reliability_trend?.toLowerCase() === "degrading" ? "deg" : "good"}`}>
            {incident.reliability_trend || "Stable"}
          </span>
        </div>
      </div>

      {/* Live Survival Probability, Weibull Parameters & MTTF Metric Bar */}
      {(() => {
        const beta = currentGroupData?.beta ?? activeBeta;
        const eta = currentGroupData?.eta ?? activeEta;
        // Approximate MTTF = eta * Gamma(1 + 1/beta)
        const approxGamma = beta >= 5 ? 1.0 : beta >= 2 ? 0.886 : beta >= 1 ? 1.0 : 2.0;
        const mttfSec = Math.round(currentGroupData?.mttf_seconds || (eta * approxGamma));
        const mttfStr = mttfSec > 3600 ? `${(mttfSec / 3600).toFixed(1)}h` : `${mttfSec}s`;

        return (
          <div className="rel-metrics-grid">
            <div className="rel-metric-box highlight">
              <span className="m-label">Survival P(T &gt; t)</span>
              <span className="m-value bold">{currentSurvivalPct}%</span>
              <div className="mini-progress-track">
                <div
                  className={`mini-progress-fill ${currentSurvivalPct > 70 ? "good" : currentSurvivalPct > 40 ? "warn" : "crit"}`}
                  style={{ width: `${currentSurvivalPct}%` }}
                ></div>
              </div>
            </div>

            <div className="rel-metric-box">
              <span className="m-label">Shape (β)</span>
              <span className="m-value mono">{beta}</span>
              <span className="m-sub">
                {beta > 5 ? "Point Mass" : beta > 1.5 ? "Wear-out" : "Random"}
              </span>
            </div>

            <div className="rel-metric-box">
              <span className="m-label">Scale (η)</span>
              <span className="m-value mono">{eta}s</span>
              <span className="m-sub">63.2% Life</span>
            </div>

            <div className="rel-metric-box highlight-amber" style={{ background: "rgba(255, 159, 10, 0.06)", borderColor: "rgba(255, 159, 10, 0.3)" }}>
              <span className="m-label" style={{ color: "#ff9f0a" }}>MTTF</span>
              <span className="m-value bold" style={{ color: "#ff9f0a" }}>{mttfStr}</span>
              <span className="m-sub" style={{ color: "rgba(255, 159, 10, 0.8)" }}>Mean Time To Fail</span>
            </div>
          </div>
        );
      })()}


      {/* 4-Group Tabs Switcher */}
      <div className="rel-tabs-bar">
        {groupKeys.map((gKey) => {
          const isActive = gKey === selectedGroup;
          const isCurrentIncident = gKey === activeGroupName;
          return (
            <button
              key={gKey}
              className={`rel-tab-btn ${isActive ? "active" : ""} ${isCurrentIncident ? "is-incident-group" : ""}`}
              onClick={() => setSelectedGroup(gKey)}
            >
              {shortNames[gKey]}
              {isCurrentIncident && <span className="active-dot" title="Active Monitored Group">●</span>}
            </button>
          );
        })}
      </div>

      {/* Embedded Chart View */}
      {renderChart()}

      {/* Expand 4-Group Grid Modal Button */}
      {onOpenModal && (
        <button className="expand-grid-btn" onClick={onOpenModal}>
          <Layers size={13} />
          <span>Expand 4-Group Grid (20s Live)</span>
        </button>
      )}

      {/* Group Model Suitability Note */}
      {currentGroupData && (
        <div className="rel-suitability-box">
          <div className="suit-head">
            <Info size={12} />
            <span className="suit-title">{currentGroupData.suitability}</span>
            <span className="suit-counts">n={currentGroupData.n} (E={currentGroupData.events}, C={currentGroupData.censored})</span>
          </div>
          <p className="suit-note">{currentGroupData.note}</p>
        </div>
      )}
    </section>
  );
}
