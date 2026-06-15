import { useState, useEffect, useRef, useMemo } from "react";
import { useNavigate } from "react-router-dom";
import * as d3 from "d3";
import {
  getAdminAnalytics,
  type AdminAnalytics,
  type AnalyticsDailyRow,
  type AnalyticsPageRow,
} from "../api/admin";

// ---------------------------------------------------------------------------
// Column resize hook (shared pattern from AdminUsersPage)
// ---------------------------------------------------------------------------

type PageColKey = "page" | "visits" | "users";

const PAGE_DEFAULT_WIDTHS: Record<PageColKey, number> = {
  page:   300,
  visits: 110,
  users:  110,
};

function useResizableColumns<K extends string>(defaults: Record<K, number>) {
  const [widths, setWidths] = useState<Record<K, number>>(defaults);

  function onDragStart(col: K, startX: number) {
    const startW = widths[col];
    function onMove(e: PointerEvent) {
      setWidths((w) => ({ ...w, [col]: Math.max(40, startW + (e.clientX - startX)) }));
    }
    function onUp() {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup",   onUp);
    }
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup",   onUp);
  }

  return { widths, onDragStart };
}

// ---------------------------------------------------------------------------
// Daily activity chart (D3, dual-axis: visits left, users right)
// ---------------------------------------------------------------------------

const CHART_HEIGHT = 220;
const MARGIN = { top: 12, right: 54, bottom: 36, left: 54 };

function DailyActivityChart({
  data,
  loading,
}: {
  data: AnalyticsDailyRow[];
  loading: boolean;
}) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [hovered, setHovered] = useState<string | null>(null);
  const [tooltip, setTooltip] = useState<{
    date: string; visits: number; users: number; x: number;
  } | null>(null);

  const width = 700;

  const innerW = width - MARGIN.left - MARGIN.right;
  const innerH = CHART_HEIGHT - MARGIN.top - MARGIN.bottom;

  const xScale = useMemo(() => {
    if (data.length === 0) return null;
    return d3.scalePoint<string>()
      .domain(data.map((d) => d.date))
      .range([0, innerW]);
  }, [data, innerW]);

  const yVisits = useMemo(() => {
    if (data.length === 0) return null;
    return d3.scaleLinear()
      .domain([0, d3.max(data, (d) => d.visits)! * 1.15])
      .range([innerH, 0]).nice();
  }, [data, innerH]);

  const yUsers = useMemo(() => {
    if (data.length === 0) return null;
    return d3.scaleLinear()
      .domain([0, d3.max(data, (d) => d.users)! * 1.15])
      .range([innerH, 0]).nice();
  }, [data, innerH]);

  const lineVisits = useMemo(() =>
    d3.line<AnalyticsDailyRow>()
      .x((d) => xScale!(d.date)!)
      .y((d) => yVisits!(d.visits)),
    [xScale, yVisits],
  );

  const lineUsers = useMemo(() =>
    d3.line<AnalyticsDailyRow>()
      .x((d) => xScale!(d.date)!)
      .y((d) => yUsers!(d.users)),
    [xScale, yUsers],
  );

  const clipId = useMemo(() => `clip-${Math.random().toString(36).slice(2)}`, []);

  if (loading) {
    return (
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
        <div className="h-[220px] bg-gray-100 rounded animate-pulse" />
      </div>
    );
  }

  if (!xScale || !yVisits || !yUsers || data.length === 0) {
    return (
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4 text-center text-gray-400 py-12">
        No data for this period.
      </div>
    );
  }

  // Show a subset of x-axis ticks so they don't overlap
  const tickInterval = Math.max(1, Math.floor(data.length / 12));
  const xTicks = data.filter((_, i) => i % tickInterval === 0);

  /** Format "2026-06-15" → "15.06." */
  function fmtDate(iso: string) {
    const parts = iso.split("-");
    return `${parts[2]}.${parts[1]}.`;
  }

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4 relative">
      <svg ref={svgRef} viewBox={`0 0 ${width} ${CHART_HEIGHT}`} className="w-full" preserveAspectRatio="xMidYMid meet">
        <defs>
          <clipPath id={clipId}>
            <rect x={-2} y={-4} width={innerW + 4} height={innerH + 6} />
          </clipPath>
        </defs>

        {/* Grid lines + lines (clipped, translated to margins) */}
        <g clipPath={`url(#${clipId})`} transform={`translate(${MARGIN.left}, ${MARGIN.top})`}>
          {yVisits.ticks(5).map((t) => (
            <line key={t} x1={0} x2={innerW}
                  y1={yVisits(t)} y2={yVisits(t)}
                  stroke="#e5e7eb" strokeWidth={0.5} />
          ))}
          <path d={lineVisits(data)!} fill="none" stroke="#3b82f6" strokeWidth={1.8} />
          <path d={lineUsers(data)!}  fill="none" stroke="#f97316" strokeWidth={1.8} />
        </g>

        {/* Hover dots */}
        {hovered && xScale(hovered) != null && (() => {
          const d = data.find((r) => r.date === hovered)!;
          return (
            <>
              <circle cx={MARGIN.left + xScale(hovered)!} cy={MARGIN.top + yVisits(d.visits)}
                      r={3.5} fill="#3b82f6" stroke="#fff" strokeWidth={1.5} />
              <circle cx={MARGIN.left + xScale(hovered)!} cy={MARGIN.top + yUsers(d.users)}
                      r={3.5} fill="#f97316" stroke="#fff" strokeWidth={1.5} />
            </>
          );
        })()}

        {/* X axis */}
        {xTicks.map((d) => (
          <text key={d.date}
                x={MARGIN.left + xScale(d.date)!}
                y={innerH + MARGIN.top + 16}
                textAnchor="middle"
                className="fill-gray-400" fontSize={10} fontFamily="monospace">
            {fmtDate(d.date)}
          </text>
        ))}

        {/* Y axis left (visits) */}
        {yVisits.ticks(5).map((t) => (
          <text key={t} x={MARGIN.left - 6} y={yVisits(t) + MARGIN.top + 3}
                textAnchor="end" className="fill-gray-500" fontSize={10}>
            {t}
          </text>
        ))}
        <text x={MARGIN.left - 32} y={MARGIN.top + innerH / 2}
              textAnchor="middle" className="fill-blue-600" fontSize={10} fontWeight={600}
              transform={`rotate(-90, ${MARGIN.left - 32}, ${MARGIN.top + innerH / 2})`}>
          visits
        </text>

        {/* Y axis right (users) */}
        {yUsers.ticks(5).map((t) => (
          <text key={t} x={width - MARGIN.right + 6} y={yUsers(t) + MARGIN.top + 3}
                textAnchor="start" className="fill-gray-500" fontSize={10}>
            {t}
          </text>
        ))}
        <text x={width - MARGIN.right + 32} y={MARGIN.top + innerH / 2}
              textAnchor="middle" className="fill-orange-600" fontSize={10} fontWeight={600}
              transform={`rotate(90, ${width - MARGIN.right + 32}, ${MARGIN.top + innerH / 2})`}>
          users
        </text>

        {/* Invisible hover bars */}
        {data.map((d) => (
          <rect key={d.date}
                x={MARGIN.left + (xScale(d.date) ?? 0) - (innerW / data.length) / 2}
                y={MARGIN.top}
                width={innerW / data.length}
                height={innerH}
                fill="transparent"
                onMouseEnter={(e) => {
                  setHovered(d.date);
                  const rect = e.currentTarget as SVGRectElement;
                  const svgRect = svgRef.current!.getBoundingClientRect();
                  const cx = rect.getBoundingClientRect().left - svgRect.left + rect.getBoundingClientRect().width / 2;
                  setTooltip({ date: d.date, visits: d.visits, users: d.users, x: cx });
                }}
                onMouseLeave={() => { setHovered(null); setTooltip(null); }}
          />
        ))}
      </svg>

      {/* Tooltip */}
      {tooltip && (
        <div
          className="absolute pointer-events-none bg-gray-900 text-white text-xs rounded px-2.5 py-1.5 shadow z-10 whitespace-nowrap"
          style={{ left: tooltip.x, top: 8, transform: "translate(-50%, 0)" }}
        >
          <div className="font-medium">{tooltip.date}</div>
          <div>
            <span className="text-blue-300">{tooltip.visits} visits</span>
            {" · "}
            <span className="text-orange-300">{tooltip.users} users</span>
          </div>
        </div>
      )}

      {/* Legend */}
      <div className="flex justify-center gap-6 mt-1 text-xs text-gray-500">
        <span className="flex items-center gap-1">
          <span className="inline-block w-3 h-0.5 bg-blue-500" /> Page visits
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block w-3 h-0.5 bg-orange-500" /> Unique users
        </span>
      </div>
    </div>
  );
}

type SortDir = "asc" | "desc";

function sortBy<T>(items: T[], col: keyof T, dir: SortDir): T[] {
  return [...items].sort((a, b) => {
    const av = a[col] ?? "";
    const bv = b[col] ?? "";
    const cmp = typeof av === "number" && typeof bv === "number"
      ? av - bv
      : String(av).localeCompare(String(bv));
    return dir === "asc" ? cmp : -cmp;
  });
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function StatCard({ label, value }: { label: string; value: number }) {
  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 px-6 py-4 flex flex-col gap-1 min-w-[120px]">
      <span className="text-2xl font-bold tabular-nums text-gray-800">{value.toLocaleString()}</span>
      <span className="text-xs font-medium text-gray-500 uppercase tracking-wide">{label}</span>
    </div>
  );
}

function ResizeHandle<K extends string>({
  col,
  onDragStart,
}: {
  col: K;
  onDragStart: (col: K, x: number) => void;
}) {
  return (
    <div
      onPointerDown={(e) => { e.preventDefault(); onDragStart(col, e.clientX); }}
      className="absolute right-0 top-0 h-full w-1.5 cursor-col-resize hover:bg-red-300 select-none"
    />
  );
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

const PERIOD_OPTIONS = [
  { label: "7 days",  value: 7  },
  { label: "30 days", value: 30 },
  { label: "90 days", value: 90 },
] as const;

export default function AdminAnalyticsPage() {
  const navigate = useNavigate();
  const [days, setDays]             = useState<number>(30);
  const [data, setData]             = useState<AdminAnalytics | null>(null);
  const [loading, setLoading]       = useState(true);
  const [error, setError]           = useState<string | null>(null);

  // Pages table sort
  const [pageSortCol, setPageSortCol] = useState<PageColKey>("visits");
  const [pageSortDir, setPageSortDir] = useState<SortDir>("desc");

  const page  = useResizableColumns<PageColKey>(PAGE_DEFAULT_WIDTHS);

  useEffect(() => {
    setLoading(true);
    setError(null);
    getAdminAnalytics(days)
      .then(setData)
      .catch(() => setError("Failed to load analytics."))
      .finally(() => setLoading(false));
  }, [days]);

  function handlePageSortClick(col: PageColKey) {
    if (col === pageSortCol) setPageSortDir((d) => (d === "asc" ? "desc" : "asc"));
    else { setPageSortCol(col); setPageSortDir("asc"); }
  }

  function SortIcon({ active, dir }: { active: boolean; dir: SortDir }) {
    if (!active) return <span className="ml-1 text-gray-300">↕</span>;
    return <span className="ml-1">{dir === "asc" ? "↑" : "↓"}</span>;
  }

  const sortedPages: AnalyticsPageRow[] = data
    ? sortBy(data.pages, pageSortCol, pageSortDir)
    : [];

  const pageTableWidth = Object.values(page.widths).reduce((a: number, b) => a + (b as number), 0);

  const PAGE_COLS: { key: PageColKey; label: string }[] = [
    { key: "page",   label: "Page"         },
    { key: "visits", label: "Visits"       },
    { key: "users",  label: "Unique users" },
  ];

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

      <div className="flex-1 overflow-y-auto p-6 flex flex-col gap-6">
        {/* Header + period selector */}
        <div className="flex items-center justify-between flex-wrap gap-3">
          <h1 className="text-xl font-bold text-gray-800">Analytics</h1>
          <div className="flex gap-1">
            {PERIOD_OPTIONS.map(({ label, value }) => (
              <button
                key={value}
                onClick={() => setDays(value)}
                className={`px-3 py-1.5 text-sm font-medium rounded transition-colors ${
                  days === value
                    ? "bg-red-700 text-white"
                    : "bg-white border border-gray-300 text-gray-600 hover:bg-gray-50"
                }`}
              >
                {label}
              </button>
            ))}
          </div>
        </div>

        {error && <div className="text-red-600">{error}</div>}

        {/* Summary cards */}
        {data && (
          <div className="flex gap-4 flex-wrap">
            <StatCard label="DAU (today)"       value={data.summary.dau} />
            <StatCard label="WAU (7 days)"      value={data.summary.wau} />
            <StatCard label="MAU (30 days)"     value={data.summary.mau} />
          </div>
        )}
        {!data && !error && (
          <div className="flex gap-4 flex-wrap">
            {["DAU (today)", "WAU (7 days)", "MAU (30 days)"].map((label) => (
              <div key={label} className="bg-white rounded-lg shadow-sm border border-gray-200 px-6 py-4 min-w-[120px]">
                <div className="h-8 w-12 bg-gray-100 rounded animate-pulse mb-1" />
                <span className="text-xs font-medium text-gray-500 uppercase tracking-wide">{label}</span>
              </div>
            ))}
          </div>
        )}

        {/* Daily activity chart */}
        <div>
          <h2 className="text-base font-semibold text-gray-700 mb-2">
            Daily activity
            {!loading && data && (
              <span className="ml-2 text-sm font-normal text-gray-400">
                (last {days} days, {data.daily.length} rows)
              </span>
            )}
          </h2>
          <DailyActivityChart data={data?.daily ?? []} loading={loading} />
        </div>

        {/* Page breakdown table */}
        <div>
          <h2 className="text-base font-semibold text-gray-700 mb-2">
            Page breakdown
            {!loading && data && (
              <span className="ml-2 text-sm font-normal text-gray-400">
                ({sortedPages.length} pages)
              </span>
            )}
          </h2>
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 overflow-x-auto">
            <table className="text-sm" style={{ tableLayout: "fixed", width: pageTableWidth + "px", minWidth: "100%" }}>
              <colgroup>
                {PAGE_COLS.map(({ key }) => <col key={key} style={{ width: page.widths[key] + "px" }} />)}
              </colgroup>
              <thead className="bg-gray-50 border-b border-gray-200">
                <tr>
                  {PAGE_COLS.map(({ key, label }) => (
                    <th
                      key={key}
                      onClick={() => handlePageSortClick(key)}
                      className="relative px-3 py-3 text-left font-medium text-gray-600 cursor-pointer hover:bg-gray-100 select-none overflow-hidden"
                    >
                      <span className="truncate">
                        {label}
                        <SortIcon active={pageSortCol === key} dir={pageSortDir} />
                      </span>
                      <ResizeHandle col={key} onDragStart={page.onDragStart} />
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100">
                {loading ? (
                  <tr><td colSpan={3} className="px-4 py-6 text-center text-gray-400">Loading…</td></tr>
                ) : sortedPages.length === 0 ? (
                  <tr><td colSpan={3} className="px-4 py-6 text-center text-gray-400">No data for this period.</td></tr>
                ) : sortedPages.map((row) => (
                  <tr key={row.page} className="hover:bg-gray-50 transition-colors">
                    <td className="px-3 py-2.5 font-mono text-xs text-gray-700 truncate" title={row.page}>{row.page}</td>
                    <td className="px-3 py-2.5 tabular-nums text-gray-600">{row.visits}</td>
                    <td className="px-3 py-2.5 tabular-nums text-gray-600">{row.users}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

      </div>
    </div>
  );
}
