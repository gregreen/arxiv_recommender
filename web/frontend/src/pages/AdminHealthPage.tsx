import { useState, useEffect, useRef, useMemo } from "react";
import { useNavigate } from "react-router-dom";
import * as d3 from "d3";
import {
  getAdminHealth,
  type AdminHealth,
  type HealthDaemon,
  type HealthCompletionTime,
} from "../api/admin";

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

const CHART_ASPECT = 600 / 200;
const MARGIN = { top: 16, right: 42, bottom: 56, left: 42 };

function useContainerWidth() {
  const ref = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(400);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      for (const entry of entries) setWidth(entry.contentRect.width);
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return { ref, width };
}

// ---------------------------------------------------------------------------
// Mini completion-time chart (D3, no viewBox)
// ---------------------------------------------------------------------------

function CompletionTimeChart({
  data,
  label,
  color,
  periodHours,
}: {
  data: HealthCompletionTime[];
  label: string;
  color: string;
  periodHours: number;
}) {
  const { ref: containerRef, width: containerW } = useContainerWidth();
  const svgRef = useRef<SVGSVGElement>(null);
  const [hovered, setHovered] = useState<number | null>(null);

  const chartH = Math.max(120, containerW / CHART_ASPECT);
  const innerW = Math.max(40, containerW - MARGIN.left - MARGIN.right);
  const innerH = Math.max(40, chartH - MARGIN.top - MARGIN.bottom);

  const series = useMemo(() => {
    if (data.length === 0) return null;
    // Parse dates; use UTC to avoid timezone shifts
    const pts = data.map((d) => ({
      date: new Date(d.completed_at + "Z"),
      ms: d.duration_ms,
    }));
    const lastDate = d3.max(pts, (d) => d.date)!;
    const firstDate = new Date(lastDate.getTime() - periodHours * 3_600_000);

    const maxMs = (d3.max(pts, (d) => d.ms) ?? 0) * 1.2 || 1000;
    const x = d3.scaleUtc().domain([firstDate, lastDate]).range([0, innerW]);
    const y = d3.scaleLinear().domain([0, maxMs]).range([innerH, 0]).nice();
    const line = d3.line<{ date: Date; ms: number }>()
      .x((d) => x(d.date))
      .y((d) => y(d.ms))
      .defined((d, i) => i === 0 || (d.date.getTime() - pts[i - 1].date.getTime()) <= 3_600_000);
    const avgMs = d3.mean(pts, (d) => d.ms) ?? 0;

    // Major ticks: midnight each day
    const majorDates = d3.utcDay.range(firstDate, lastDate);
    // Minor tick interval depends on period
    const minorInterval = periodHours <= 4 ? 1 : periodHours <= 24 ? 4 : 6;
    const firstMinor = d3.utcHour.offset(d3.utcHour.floor(firstDate), 0);
    const minorDates = d3.utcHour.every(minorInterval)!.range(firstMinor, lastDate)!;

    return { x, y, line, avgMs, pts, majorDates, minorDates };
  }, [data, innerW, innerH, periodHours]);

  const clipId = useMemo(() => `clip-${Math.random().toString(36).slice(2)}`, []);

  if (data.length === 0) {
    return (
      <div ref={containerRef} className="text-center text-gray-400 text-xs py-4">
        No {label} data yet.
      </div>
    );
  }

  const { x, y, line, avgMs, pts, majorDates, minorDates } = series!;
  const yTicks = y.ticks(3);
  const tickLen = 4;

  /** Format "2026-06-17T10:30:00" → "17 Jun" */
  function fmtDate(iso: string) {
    return new Date(iso + "Z").toLocaleDateString("en-GB", {
      day: "2-digit",
      month: "short",
    });
  }

  return (
    <div ref={containerRef} className="relative">
      <svg ref={svgRef} width={containerW} height={chartH} className="block">
        <defs>
          <clipPath id={clipId}>
            <rect x={-2} y={-4} width={innerW + 4} height={innerH + 6} />
          </clipPath>
        </defs>

        {/* Grid lines + lines (clipped, translated to margins) */}
        <g clipPath={`url(#${clipId})`} transform={`translate(${MARGIN.left}, ${MARGIN.top})`}>
          {yTicks.map((t) => (
            <line key={t} x1={0} x2={innerW}
                  y1={y(t)} y2={y(t)}
                  stroke="#e5e7eb" strokeWidth={0.5} />
          ))}
          <line x1={0} x2={innerW}
                y1={y(avgMs)} y2={y(avgMs)}
                stroke={color} strokeWidth={0.5}
                strokeDasharray="4 3" opacity={0.5} />
          <path d={line(pts)!} fill="none"
                stroke={color} strokeWidth={1.5} />
        </g>

        {/* Hover dots */}
        {hovered != null && data[hovered] && (() => {
          const d = new Date(data[hovered].completed_at + "Z");
          return (
            <circle cx={MARGIN.left + x(d)}
                    cy={MARGIN.top + y(data[hovered].duration_ms)}
                    r={3} fill={color} stroke="#fff" strokeWidth={1.5} />
          );
        })()}

        {/* X-axis minor ticks (every 4h) */}
        {minorDates.map((d) => (
          <line key={`xt-${d.toISOString()}`}
                x1={MARGIN.left + x(d)} x2={MARGIN.left + x(d)}
                y1={innerH + MARGIN.top}
                y2={innerH + MARGIN.top + tickLen}
                stroke="#9ca3af" strokeWidth={1} />
        ))}
        {/* X-axis major ticks (midnight) */}
        {majorDates.map((d) => (
          <line key={`xm-${d.toISOString()}`}
                x1={MARGIN.left + x(d)} x2={MARGIN.left + x(d)}
                y1={innerH + MARGIN.top}
                y2={innerH + MARGIN.top + tickLen * 2}
                stroke="#6b7280" strokeWidth={1.5} />
        ))}
        {/* X-axis date labels (midnight, rotated 90°) */}
        {majorDates.map((d) => {
          const cx = MARGIN.left + x(d);
          const cy = innerH + MARGIN.top + tickLen * 2 + 22;
          return (
            <text key={`lbl-${d.toISOString()}`}
                  x={cx} y={cy}
                  textAnchor="middle"
                  className="fill-gray-400 text-[9px] font-mono"
                  transform={`translate(-2.5,2) rotate(90, ${cx}, ${cy})`}>
              {fmtDate(d.toISOString().slice(0, 19))}
            </text>
          );
        })}
        {/* Time labels (at minor ticks except midnight) */}
        {minorDates.map((d) => {
          if (d.getUTCHours() === 0) return null;
          const cx = MARGIN.left + x(d);
          const cy = innerH + MARGIN.top + tickLen * 2 + 22;
          const hh = String(d.getUTCHours()).padStart(2, "0");
          return (
            <text key={`t-${d.toISOString()}`}
                  x={cx} y={cy}
                  textAnchor="middle"
                  className="fill-gray-300 text-[8px] font-mono"
                  transform={`translate(-2.5,2) rotate(90, ${cx}, ${cy})`}>
              {hh}:00
            </text>
          );
        })}

        {/* Invisible hover bars */}
        {data.map((d, i) => {
          const date = new Date(d.completed_at + "Z");
          const bw = Math.max(3, innerW / data.length);
          return (
            <rect key={d.completed_at}
                  x={MARGIN.left + x(date) - bw / 2}
                  y={MARGIN.top}
                  width={bw}
                  height={innerH}
                  fill="transparent"
                  onMouseEnter={() => setHovered(i)}
                  onMouseLeave={() => setHovered(null)} />
          );
        })}
      </svg>
      {/* Y-axis labels */}
      {yTicks.map((t) => (
        <span key={t}
              className="absolute text-gray-400 text-[9px] font-mono pointer-events-none"
              style={{ left: 4, top: MARGIN.top + y(t), transform: "translateY(-50%)" }}>
          {t >= 1000 ? `${(t / 1000).toFixed(1)}s` : `${t}ms`}
        </span>
      ))}

      {/* Legend */}
      <div className="flex items-center gap-4 justify-center mt-0.5 text-[10px] text-gray-400">
        <span className="flex items-center gap-1">
          <span className="inline-block w-2.5 h-0.5" style={{ backgroundColor: color }} />
          {label} task time
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block w-2.5 h-0.5 border-t border-dashed" style={{ borderColor: color, opacity: 0.5 }} />
          avg
        </span>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Daemon card
// ---------------------------------------------------------------------------

function DaemonCard({
  title,
  data,
  color,
  periodHours,
  extraRows,
}: {
  title: string;
  data: HealthDaemon;
  color: string;
  periodHours: number;
  extraRows?: { label: string; value: number | string; warn?: boolean }[];
}) {
  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4 flex flex-col gap-3">
      <h3 className="text-sm font-semibold text-gray-700">{title}</h3>

      {/* Stat rows */}
      <div className="grid grid-cols-2 gap-x-4 gap-y-1.5">
        <Stat label="Queue size" value={data.queue_size} />
        <Stat
          label="Avg time"
          value={
            data.avg_completion_ms != null
              ? data.avg_completion_ms >= 1000
                ? `${(data.avg_completion_ms / 1000).toFixed(1)}s`
                : `${data.avg_completion_ms}ms`
              : "—"
          }
        />
        <Stat
          label="Failure rate (24h)"
          value={
            data.recent_failure_rate != null
              ? `${(data.recent_failure_rate * 100).toFixed(1)}%`
              : "—"
          }
          warn={data.recent_failure_rate != null && data.recent_failure_rate > 0.05}
        />
        <Stat
          label="Failed (total)"
          value={data.permanently_failed}
          warn={data.permanently_failed > 0}
        />
        {extraRows?.map((r) => (
          <Stat key={r.label} label={r.label} value={r.value} warn={r.warn} />
        ))}
      </div>

      {/* Mini chart */}
      <CompletionTimeChart data={data.completion_times} label={title} color={color} periodHours={periodHours} />
    </div>
  );
}

function Stat({
  label,
  value,
  warn,
}: {
  label: string;
  value: number | string;
  warn?: boolean;
}) {
  return (
    <div className="flex flex-col">
      <span className="text-[10px] text-gray-400 uppercase">{label}</span>
      <span
        className={`text-sm font-semibold tabular-nums ${
          warn ? "text-amber-600" : "text-gray-800"
        }`}
      >
        {value}
      </span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

const PERIOD_OPTIONS = [
  { label: "4 hours", value: 4 },
  { label: "1 day",   value: 24 },
  { label: "3 days",  value: 72 },
] as const;

export default function AdminHealthPage() {
  const navigate = useNavigate();
  const [periodHours, setPeriodHours] = useState<number>(4);
  const [data, setData] = useState<AdminHealth | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    function fetch() {
      getAdminHealth()
        .then((d) => {
          if (!cancelled) {
            setData(d);
            setLoading(false);
            setError(null);
          }
        })
        .catch(() => {
          if (!cancelled) setError("Failed to load health data.");
        });
    }
    fetch();
    const timer = setInterval(fetch, 30_000);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, []);

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Mobile: back to category nav */}
      <div className="md:hidden shrink-0 flex items-center px-4 py-2 bg-white border-b border-gray-200">
        <button
          onClick={() => navigate("/admin")}
          className="flex items-center gap-1.5 text-sm text-red-700 hover:text-red-900 transition-colors"
        >
          ← Return to list
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-6 gap-4 flex flex-col">
        <div className="flex items-center justify-between flex-wrap gap-3">
          <h1 className="text-xl font-bold text-gray-800">Daemon Health</h1>
          <div className="flex gap-1">
            {PERIOD_OPTIONS.map(({ label, value }) => (
              <button
                key={value}
                onClick={() => setPeriodHours(value)}
                className={`px-3 py-1.5 text-sm font-medium rounded transition-colors ${
                  periodHours === value
                    ? "bg-red-700 text-white"
                    : "bg-white border border-gray-300 text-gray-600 hover:bg-gray-50"
                }`}
              >
                {label}
              </button>
            ))}
          </div>
        </div>

      {error && <div className="text-red-600 text-sm">{error}</div>}

      {loading && !data && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {[1, 2].map((i) => (
            <div key={i} className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
              <div className="h-4 w-24 bg-gray-100 rounded animate-pulse mb-3" />
              <div className="grid grid-cols-2 gap-3">
                {[1, 2, 3, 4].map((j) => (
                  <div key={j}>
                    <div className="h-3 w-16 bg-gray-100 rounded animate-pulse mb-1" />
                    <div className="h-5 w-12 bg-gray-100 rounded animate-pulse" />
                  </div>
                ))}
              </div>
              <div className="h-[100px] bg-gray-50 rounded animate-pulse mt-3" />
            </div>
          ))}
        </div>
      )}

      {data && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <DaemonCard
            title="Embed Daemon"
            data={data.embed}
            color="#3b82f6"
            periodHours={periodHours}
          />
          <DaemonCard
            title="Meta Daemon"
            data={data.meta}
            color="#f97316"
            periodHours={periodHours}
            extraRows={[
              {
                label: "Stale pending",
                value: data.meta.stale_pending,
                warn: data.meta.stale_pending > 0,
              },
            ]}
          />
        </div>
      )}
      </div>
    </div>
  );
}
