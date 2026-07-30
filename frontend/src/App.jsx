import React, { useState, useEffect } from "react";
import { Menu } from "lucide-react";
import Sidebar from "./components/Sidebar";
import TopHeader from "./components/TopHeader";
import SummaryCard from "./components/SummaryCard";
import DiagnosisCard from "./components/DiagnosisCard";
import PredictionCard from "./components/PredictionCard";
import EvidenceCard from "./components/EvidenceCard";
import MultiTrendCharts from "./components/MultiTrendCharts";
import ReliabilityCard from "./components/ReliabilityCard";
import HumanGatePanel from "./components/HumanGatePanel";

const BACKEND_URL = "http://localhost:8080";

// Fallback incident state for demo before backend telemetry starts
const DEMO_FALLBACK = {
  episode_id: "ep_00000_STANDBY",
  incident_id: "INC-0000",
  failure_mode: "NONE",
  label: "Normal Operation",
  severity: "P4",
  status: "OPEN",
  detected_time: "--:--:--",
  elapsed_s: 0,
  cycle: 0,
  root_cause: "System monitoring active. Telemetry signals stable. No anomalous signature detected.",
  classifier_confidence: 1.0,
  warning_count: 0,
  critical_count: 0,
  blast_size: 0,
  predicted_failure: "None",
  ttf_seconds: null,
  prediction_confidence: 1.0,
  recommended_action: "Keep monitoring system signals.",
  critical_metric: "cpu_utilization",
  metric_value: "22%",
  trace_evidence: "Not Available",
  log_evidence: "Not Available",
  reliability_trend: "Stable",
  charts: {
    cpu_utilization: {
      label: "CPU Utilization",
      metric_key: "cpu_utilization",
      current_value: 22,
      threshold: 90,
      unit: "%",
      history: [20, 22, 21, 23, 22, 20, 22, 21, 23, 22],
      forecast: [],
      breached: false,
      projected_breach: false,
      trend_direction: "Stable →"
    },
    heap_mb: {
      label: "Heap Memory",
      metric_key: "heap_mb",
      current_value: 512,
      threshold: 3500,
      unit: "MB",
      history: [510, 512, 510, 515, 512, 510, 512, 510, 515, 512],
      forecast: [],
      breached: false,
      projected_breach: false,
      trend_direction: "Stable →"
    },
    p99_latency: {
      label: "P99 Latency",
      metric_key: "p99_latency",
      current_value: 120,
      threshold: 700,
      unit: "ms",
      history: [115, 120, 118, 122, 120, 115, 120, 118, 122, 120],
      forecast: [],
      breached: false,
      projected_breach: false,
      trend_direction: "Stable →"
    },
    db_p99: {
      label: "Database Latency",
      metric_key: "db_p99",
      current_value: 25,
      threshold: 1500,
      unit: "ms",
      history: [22, 25, 23, 26, 25, 22, 25, 23, 26, 25],
      forecast: [],
      breached: false,
      projected_breach: false,
      trend_direction: "Stable →"
    },
    error_rate: {
      label: "Error Rate",
      metric_key: "error_rate",
      current_value: 0.01,
      threshold: 0.50,
      unit: "ratio",
      history: [0.01, 0.01, 0.02, 0.01, 0.01, 0.01, 0.01, 0.02, 0.01, 0.01],
      forecast: [],
      breached: false,
      projected_breach: false,
      trend_direction: "Stable →"
    },
    cache_miss_rate: {
      label: "Cache Miss Rate",
      metric_key: "cache_miss_rate",
      current_value: 0.05,
      threshold: 0.90,
      unit: "ratio",
      history: [0.05, 0.06, 0.04, 0.07, 0.05, 0.05, 0.06, 0.04, 0.07, 0.05],
      forecast: [],
      breached: false,
      projected_breach: false,
      trend_direction: "Stable →"
    },
    queue_lag: {
      label: "Queue Lag",
      metric_key: "queue_lag",
      current_value: 2,
      threshold: 500,
      unit: "ms",
      history: [1, 2, 1, 3, 2, 1, 2, 1, 3, 2],
      forecast: [],
      breached: false,
      projected_breach: false,
      trend_direction: "Stable →"
    }
  },
  mtbf_hours: 168,
  mttr_hours: 0.1,
  weibull_confidence: 0.99,
  steps: ["System Healthy", "No Action Required"],
};

export default function App() {
  const [incident, setIncident] = useState(DEMO_FALLBACK);
  const [episodes, setEpisodes] = useState([]);
  const [selectedEpisodeId, setSelectedEpisodeId] = useState(null);
  const [isLive, setIsLive] = useState(true);
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false);
  const [activeMobileTab, setActiveMobileTab] = useState("metrics"); // "metrics" | "incident"

  // Helper to load list of episodes
  const fetchEpisodes = async () => {
    try {
      const res = await fetch(`${BACKEND_URL}/api/episodes`);
      if (res.ok) {
        const data = await res.json();
        setEpisodes(data);
      }
    } catch (err) {
      console.warn("Failed to fetch episodes list", err);
    }
  };

  // Helper to load active live telemetry
  const fetchLiveState = async () => {
    try {
      const res = await fetch(`${BACKEND_URL}/api/live`);
      if (res.ok) {
        const data = await res.json();
        setIncident(data);
      }
    } catch (err) {
      console.log("Awaiting pipeline API...");
    }
  };

  // Helper to load details for specific selected incident
  const fetchEpisodeDetails = async (id) => {
    try {
      const res = await fetch(`${BACKEND_URL}/api/episodes/${id}`);
      if (res.ok) {
        const data = await res.json();
        setIncident(data);
      }
    } catch (err) {
      console.error("Failed to fetch details for episode", id, err);
    }
  };

  // Fetch episodes list on load
  useEffect(() => {
    fetchEpisodes();
    const interval = setInterval(fetchEpisodes, 10000); // refresh list every 10s
    return () => clearInterval(interval);
  }, []);

  // Poll live incident details
  useEffect(() => {
    if (!isLive) return;

    fetchLiveState(); // fetch immediately
    const interval = setInterval(fetchLiveState, 2000); // poll every 2s
    return () => clearInterval(interval);
  }, [isLive]);

  // Handle manual status changes via dropdown
  const handleStatusChange = async (newStatus) => {
    if (!incident || !incident.episode_id) return;
    
    // Update local state instantly
    setIncident((prev) => ({ ...prev, status: newStatus }));

    try {
      await fetch(`${BACKEND_URL}/api/incident/${incident.episode_id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ status: newStatus }),
      });
      // Refresh episodes list to show any changes
      fetchEpisodes();
    } catch (err) {
      console.error("Failed to update incident status on server", err);
    }
  };

  // Switch to viewing a specific past incident log
  const handleSelectEpisode = (id) => {
    setIsLive(false);
    setSelectedEpisodeId(id);
    fetchEpisodeDetails(id);
    setMobileSidebarOpen(false);
  };

  // Switch back to viewing live updates
  const handleViewLive = () => {
    setIsLive(true);
    setSelectedEpisodeId(null);
    fetchLiveState();
    setMobileSidebarOpen(false);
  };

  // Voted active failure mode
  const activeFailureMode = incident ? incident.failure_mode : "NONE";

  return (
    <div className={`app-container ${mobileSidebarOpen ? "sidebar-open" : ""}`}>
      {/* Mobile Header Bar */}
      <div className="mobile-header">
        <button className="mobile-menu-toggle" onClick={() => setMobileSidebarOpen(true)}>
          <Menu size={22} />
        </button>
        <div className="mobile-logo">
          <span className="logo-mark">◆</span>
          <span className="logo-text">SENTINEL</span>
        </div>
        <div className="mobile-tabs">
          <button 
            className={`mobile-tab-btn ${activeMobileTab === "metrics" ? "active" : ""}`}
            onClick={() => setActiveMobileTab("metrics")}
          >
            Metrics
          </button>
          <button 
            className={`mobile-tab-btn ${activeMobileTab === "incident" ? "active" : ""}`}
            onClick={() => setActiveMobileTab("incident")}
          >
            Incident
          </button>
        </div>
      </div>

      {/* Backdrop overlay for mobile sidebar */}
      {mobileSidebarOpen && (
        <div className="sidebar-backdrop" onClick={() => setMobileSidebarOpen(false)} />
      )}

      {/* COLUMN 1: LEFT SIDEBAR (Width: ~240px) */}
      <Sidebar
        activeMode={activeFailureMode}
        episodes={episodes}
        selectedEpisodeId={selectedEpisodeId}
        onSelectEpisode={handleSelectEpisode}
        onViewLive={handleViewLive}
        isLive={isLive}
        mobileSidebarOpen={mobileSidebarOpen}
        setMobileSidebarOpen={setMobileSidebarOpen}
      />

      {/* COLUMN 2: MIDDLE AREA (NOC Charts Panel) */}
      <div className={`center-panel ${activeMobileTab === "metrics" ? "mobile-active" : "mobile-hidden"}`}>
        <TopHeader />
        
        {/* NOC grid displaying 7 core metrics graphs */}
        <MultiTrendCharts charts={incident.charts} failureMode={incident.failure_mode} />
      </div>

      {/* COLUMN 3: RIGHT PANEL (Incident metadata stack) */}
      <div className={`right-panel ${activeMobileTab === "incident" ? "mobile-active" : "mobile-hidden"}`}>
        <SummaryCard 
          incident={incident} 
          onStatusChange={handleStatusChange} 
        />
        <PredictionCard incident={incident} />
        <EvidenceCard incident={incident} />
        <DiagnosisCard incident={incident} />
        <ReliabilityCard incident={incident} />
      </div>

      {/* HUMAN GATE PANEL — floats above all content as an overlay */}
      <HumanGatePanel />
    </div>
  );
}
