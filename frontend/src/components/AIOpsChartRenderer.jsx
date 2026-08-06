/**
 * AIOpsChartRenderer.jsx
 * =======================
 * Compact Recharts-based chart component rendered inside QueryLangGraph chat messages.
 *
 * Accepts the `visualization` payload directly from the QueryLangGraph ResponseFormatter:
 *   {
 *     chart_type: "line" | "bar" | "multi_line",
 *     title: string,
 *     x_axis: string,
 *     y_axis: string,
 *     series: [{ name: string, data: [{x: any, y: number}] }],
 *     data_points: number,
 *     metadata: { theme, colors, ... }
 *   }
 *
 * Uses existing recharts dependency (already in package.json).
 * Dark-themed to match the Sentinel Dashboard design system.
 */

import {
  LineChart, Line, BarChart, Bar,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer
} from 'recharts';

// ── Color palette matching VisualizationService defaults ─────────────
const CHART_COLORS = [
  '#4E9AF1',  // blue
  '#F16B4E',  // coral
  '#4EF19A',  // green
  '#F1C44E',  // amber
  '#C44EF1',  // violet
  '#4EC4F1',  // cyan
  '#F14E8A',  // pink
  '#A8F14E',  // lime
];

// ── Custom dark-themed tooltip ─────────────────────────────────────────
const DarkTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background: '#0a1226',
      border: '1px solid #2a3650',
      borderRadius: 8,
      padding: '8px 12px',
      fontSize: 12,
      color: '#e7ecfb',
      boxShadow: '0 4px 16px rgba(0,0,0,0.5)',
    }}>
      <p style={{ margin: '0 0 4px', color: '#8fa0d1', fontSize: 11 }}>{label}</p>
      {payload.map((entry, i) => (
        <p key={i} style={{ margin: '2px 0', color: entry.color, fontWeight: 600 }}>
          {entry.name}: <span style={{ color: '#fff' }}>{
            typeof entry.value === 'number' ? entry.value.toFixed(3) : entry.value
          }</span>
        </p>
      ))}
    </div>
  );
};

// ── Transform VisualizationService series format to Recharts format ────
// Input:  series = [{ name: "cpu_utilization", data: [{x: "t1", y: 87.3}, ...] }]
// Output: [{ x: "t1", cpu_utilization: 87.3, ... }, ...]
function transformSeriesForRecharts(series) {
  if (!series || series.length === 0) return [];

  // Build a map from x-value → merged row
  const xMap = new Map();
  for (const serie of series) {
    const name = serie.name || 'value';
    for (const point of (serie.data || [])) {
      const key = String(point.x ?? '');
      if (!xMap.has(key)) xMap.set(key, { x: key });
      const val = point.y;
      xMap.get(key)[name] = typeof val === 'number' ? parseFloat(val.toFixed(4)) : val;
    }
  }
  return Array.from(xMap.values());
}

// ── Format X-axis tick labels (truncate timestamps) ────────────────────
function formatXTick(value) {
  if (!value) return '';
  const s = String(value);
  // If it looks like a Unix timestamp number, format it
  if (/^\d{10,}(\.\d+)?$/.test(s)) {
    const d = new Date(parseFloat(s) * 1000);
    return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  }
  // If already an ISO-like string, trim it
  if (s.includes('T') && s.length > 16) return s.slice(11, 16);
  if (s.length > 12) return s.slice(0, 10);
  return s;
}

// ── Main chart renderer component ──────────────────────────────────────
export default function AIOpsChartRenderer({ visualization }) {
  if (!visualization) return null;

  const { chart_type, title, x_axis, y_axis, series, data_points } = visualization;

  if (!series || series.length === 0) {
    return (
      <div className="aiops-chart-empty">
        <span>📊 Chart requested but no series data returned.</span>
      </div>
    );
  }

  // Check that at least one series has data points
  const hasData = series.some(s => s.data && s.data.length > 0);
  if (!hasData) {
    return (
      <div className="aiops-chart-empty">
        <span>📊 No data points in visualization series.</span>
      </div>
    );
  }

  const chartData   = transformSeriesForRecharts(series);
  const seriesNames = series.map(s => s.name || 'value');
  const type        = (chart_type || 'line').toLowerCase();

  // Common axis styling (dark theme)
  const axisStyle = {
    tick:   { fill: '#8fa0d1', fontSize: 10, fontFamily: 'JetBrains Mono, monospace' },
    label:  { fill: '#5a6a99', fontSize: 10 },
    stroke: '#1e2a42',
  };

  const gridStroke = 'rgba(42, 54, 80, 0.7)';

  return (
    <div className="aiops-chart-wrapper">
      {/* Chart header */}
      <div className="aiops-chart-header">
        <span className="aiops-chart-title">{title || 'AIOps Chart'}</span>
        <div className="aiops-chart-badges">
          <span className="aiops-chart-badge">{data_points || chartData.length} pts</span>
          <span className="aiops-chart-badge aiops-chart-type-badge">
            {type.toUpperCase()}
          </span>
        </div>
      </div>

      {/* Recharts chart */}
      <ResponsiveContainer width="100%" height={200}>
        {type === 'bar' ? (
          <BarChart data={chartData} margin={{ top: 4, right: 8, left: -10, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={gridStroke} vertical={false} />
            <XAxis
              dataKey="x"
              tick={axisStyle.tick}
              tickLine={false}
              axisLine={{ stroke: axisStyle.stroke }}
              tickFormatter={formatXTick}
              label={{ value: x_axis || '', position: 'insideBottom', offset: -2, style: axisStyle.label }}
              interval="preserveStartEnd"
            />
            <YAxis
              tick={axisStyle.tick}
              tickLine={false}
              axisLine={false}
              label={{ value: y_axis || '', angle: -90, position: 'insideLeft', style: axisStyle.label }}
              width={45}
            />
            <Tooltip content={<DarkTooltip />} cursor={{ fill: 'rgba(78,154,241,0.08)' }} />
            {seriesNames.length > 1 && (
              <Legend
                iconSize={8}
                wrapperStyle={{ fontSize: 10, color: '#8fa0d1', paddingTop: 4 }}
              />
            )}
            {seriesNames.map((name, idx) => (
              <Bar
                key={name}
                dataKey={name}
                fill={CHART_COLORS[idx % CHART_COLORS.length]}
                radius={[3, 3, 0, 0]}
                maxBarSize={40}
              />
            ))}
          </BarChart>
        ) : (
          /* Default: line / multi_line */
          <LineChart data={chartData} margin={{ top: 4, right: 8, left: -10, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={gridStroke} vertical={false} />
            <XAxis
              dataKey="x"
              tick={axisStyle.tick}
              tickLine={false}
              axisLine={{ stroke: axisStyle.stroke }}
              tickFormatter={formatXTick}
              label={{ value: x_axis || '', position: 'insideBottom', offset: -2, style: axisStyle.label }}
              interval="preserveStartEnd"
            />
            <YAxis
              tick={axisStyle.tick}
              tickLine={false}
              axisLine={false}
              label={{ value: y_axis || '', angle: -90, position: 'insideLeft', style: axisStyle.label }}
              width={45}
            />
            <Tooltip content={<DarkTooltip />} />
            {seriesNames.length > 1 && (
              <Legend
                iconSize={8}
                wrapperStyle={{ fontSize: 10, color: '#8fa0d1', paddingTop: 4 }}
              />
            )}
            {seriesNames.map((name, idx) => (
              <Line
                key={name}
                type="monotone"
                dataKey={name}
                stroke={CHART_COLORS[idx % CHART_COLORS.length]}
                strokeWidth={2}
                dot={false}
                activeDot={{ r: 4, strokeWidth: 0 }}
              />
            ))}
          </LineChart>
        )}
      </ResponsiveContainer>
    </div>
  );
}
