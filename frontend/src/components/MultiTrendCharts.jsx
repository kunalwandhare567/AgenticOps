import React from "react";
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
} from "recharts";
import { AlertTriangle, ShieldCheck, Zap } from "lucide-react";

// Priority metrics lookup per failure mode
const FAILURE_MODE_PRIORITIES = {
  MEMORY_LEAK: ["heap_mb", "cpu_utilization"],
  CPU_SATURATION: ["cpu_utilization", "p99_latency"],
  LATENCY_SPIKE: ["p99_latency", "db_p99"],
  DB_SLOWDOWN: ["db_p99", "queue_lag"],
  CACHE_STAMPEDE: ["cache_miss_rate", "db_p99"],
  QUEUE_BACKUP: ["queue_lag", "p99_latency"],
  DEPENDENCY_TIMEOUT: ["p99_latency", "error_rate"],
  BAD_DEPLOY: ["error_rate", "cpu_utilization"],
  ERROR_STORM: ["error_rate", "p99_latency"],
  RETRY_STORM: ["error_rate", "cpu_utilization"],
  DISK_IO_SATURATION: ["db_p99", "cpu_utilization"],
  CASCADING_FAILURE: ["p99_latency", "error_rate"],
};

export default function MultiTrendCharts({ charts, failureMode }) {
  if (!charts) return null;

  // Get active priority list
  const priorities = FAILURE_MODE_PRIORITIES[failureMode] || [];

  // Sort charts list: float priority metrics to the top
  const sortedEntries = Object.entries(charts).sort(([keyA], [keyB]) => {
    const idxA = priorities.indexOf(keyA);
    const idxB = priorities.indexOf(keyB);

    if (idxA !== -1 && idxB !== -1) return idxA - idxB;
    if (idxA !== -1) return -1;
    if (idxB !== -1) return 1;
    return 0;
  });

  return (
    <div className="noc-grid">
      {sortedEntries.map(([key, chart]) => {
        // Build combined data array for Recharts
        const chartData = [];
        
        // 1. Add historical points (last 20 logs)
        chart.history.forEach((val, idx) => {
          chartData.push({
            name: `H${idx + 1}`,
            historyValue: val,
            forecastValue: null,
          });
        });

        // 2. Add forecast projection (connecting history to forecast)
        if (chart.forecast && chart.forecast.length > 0) {
          const lastHistVal = chart.history[chart.history.length - 1];
          
          chartData.push({
            name: "Now",
            historyValue: lastHistVal,
            forecastValue: lastHistVal,
          });

          chart.forecast.forEach((val, idx) => {
            chartData.push({
              name: `F${idx + 1}`,
              historyValue: null,
              forecastValue: val,
            });
          });
        }

        // Determine badge styling based on breach state
        let badgeClass = "badge-status healthy";
        let BadgeIcon = ShieldCheck;
        let badgeText = "SAFE";

        if (chart.breached) {
          badgeClass = "badge-status breached blinking";
          BadgeIcon = AlertTriangle;
          badgeText = "BREACHED";
        } else if (chart.projected_breach) {
          badgeClass = "badge-status projected";
          BadgeIcon = Zap;
          badgeText = "PROJECTED BREACH";
        }

        // Color theme coding per chart card
        const gradientId = `grad_${key}`;

        return (
          <div key={key} className="card chart-card">
            <div className="chart-card-header">
              <div className="chart-title-group">
                <span className="chart-label">{chart.label}</span>
                <span className="chart-unit">({chart.unit})</span>
              </div>
              <div className={badgeClass}>
                <BadgeIcon size={11} style={{ marginRight: "3px" }} />
                <span>{badgeText}</span>
              </div>
            </div>

            <div className="chart-meta-row">
              <div className="meta-item">
                <span className="meta-lbl">Current</span>
                <span className="meta-val mono">
                  {chart.current_value}
                  {chart.unit === "%" ? "%" : ""}
                </span>
              </div>
              <div className="meta-item">
                <span className="meta-lbl">Threshold</span>
                <span className="meta-val mono" style={{ color: "var(--p1)" }}>
                  {chart.threshold}
                </span>
              </div>
              <div className="meta-item">
                <span className="meta-lbl">Trend</span>
                <span className="meta-val">{chart.trend_direction}</span>
              </div>
            </div>

            <div className="chart-wrapper">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart
                  data={chartData}
                  margin={{ top: 5, right: 5, left: -25, bottom: 5 }}
                >
                  <defs>
                    <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#4f5fff" stopOpacity={0.25} />
                      <stop offset="95%" stopColor="#4f5fff" stopOpacity={0.0} />
                    </linearGradient>
                    <linearGradient id={`fc_${gradientId}`} x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#8a4dff" stopOpacity={0.15} />
                      <stop offset="95%" stopColor="#8a4dff" stopOpacity={0.0} />
                    </linearGradient>
                  </defs>
                  
                  <CartesianGrid
                    strokeDasharray="3 3"
                    stroke="rgba(20,30,60,0.04)"
                  />
                  
                  <XAxis
                    dataKey="name"
                    tick={{ fontSize: 8, fill: "#8791a4" }}
                    axisLine={false}
                    tickLine={false}
                  />
                  
                  <YAxis
                    tick={{ fontSize: 8, fill: "#8791a4" }}
                    axisLine={false}
                    tickLine={false}
                    domain={[0, "auto"]}
                  />
                  
                  <Tooltip
                    contentStyle={{
                      background: "#101b35",
                      border: "none",
                      borderRadius: "8px",
                      color: "#fff",
                      fontFamily: "JetBrains Mono",
                      fontSize: "10px",
                    }}
                  />
                  
                  {/* Critical Threshold Line */}
                  <ReferenceLine
                    y={chart.threshold}
                    stroke="var(--p1)"
                    strokeWidth={1}
                    strokeDasharray="3 3"
                  />

                  {/* History Solid Line */}
                  <Area
                    type="monotone"
                    dataKey="historyValue"
                    stroke="#4f5fff"
                    strokeWidth={1.8}
                    fillOpacity={1}
                    fill={`url(#${gradientId})`}
                    connectNulls={false}
                  />

                  {/* Forecast Dashed Line */}
                  <Area
                    type="monotone"
                    dataKey="forecastValue"
                    stroke="#8a4dff"
                    strokeWidth={1.8}
                    strokeDasharray="4 4"
                    fillOpacity={1}
                    fill={`url(#fc_${gradientId})`}
                    connectNulls={false}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>
        );
      })}
    </div>
  );
}
