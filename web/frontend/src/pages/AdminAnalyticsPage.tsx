import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import {
  getAdminAnalytics,
  type AdminAnalytics,
  type AnalyticsDailyRow,
  type AnalyticsPageRow,
} from "../api/admin";

// ---------------------------------------------------------------------------
// Column resize hook (shared pattern from AdminUsersPage)
// ---------------------------------------------------------------------------

type DailyColKey = "date" | "users" | "visits";
type PageColKey  = "page" | "visits" | "users";

const DAILY_DEFAULT_WIDTHS: Record<DailyColKey, number> = {
  date:   160,
  users:  110,
  visits: 110,
};

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
// Sort helpers
// ---------------------------------------------------------------------------

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

  // Daily table sort
  const [dailySortCol, setDailySortCol] = useState<DailyColKey>("date");
  const [dailySortDir, setDailySortDir] = useState<SortDir>("desc");

  // Pages table sort
  const [pageSortCol, setPageSortCol] = useState<PageColKey>("visits");
  const [pageSortDir, setPageSortDir] = useState<SortDir>("desc");

  const daily = useResizableColumns<DailyColKey>(DAILY_DEFAULT_WIDTHS);
  const page  = useResizableColumns<PageColKey>(PAGE_DEFAULT_WIDTHS);

  useEffect(() => {
    setLoading(true);
    setError(null);
    getAdminAnalytics(days)
      .then(setData)
      .catch(() => setError("Failed to load analytics."))
      .finally(() => setLoading(false));
  }, [days]);

  function handleDailySortClick(col: DailyColKey) {
    if (col === dailySortCol) setDailySortDir((d) => (d === "asc" ? "desc" : "asc"));
    else { setDailySortCol(col); setDailySortDir("asc"); }
  }

  function handlePageSortClick(col: PageColKey) {
    if (col === pageSortCol) setPageSortDir((d) => (d === "asc" ? "desc" : "asc"));
    else { setPageSortCol(col); setPageSortDir("asc"); }
  }

  function SortIcon({ active, dir }: { active: boolean; dir: SortDir }) {
    if (!active) return <span className="ml-1 text-gray-300">↕</span>;
    return <span className="ml-1">{dir === "asc" ? "↑" : "↓"}</span>;
  }

  const sortedDaily: AnalyticsDailyRow[] = data
    ? sortBy(data.daily, dailySortCol, dailySortDir)
    : [];
  const sortedPages: AnalyticsPageRow[] = data
    ? sortBy(data.pages, pageSortCol, pageSortDir)
    : [];

  const dailyTableWidth = Object.values(daily.widths).reduce((a: number, b) => a + (b as number), 0);
  const pageTableWidth  = Object.values(page.widths).reduce((a: number, b) => a + (b as number), 0);

  const DAILY_COLS: { key: DailyColKey; label: string }[] = [
    { key: "date",   label: "Date"         },
    { key: "users",  label: "Active users" },
    { key: "visits", label: "Page visits"  },
  ];

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

        {/* Daily active users table */}
        <div>
          <h2 className="text-base font-semibold text-gray-700 mb-2">
            Daily activity
            {!loading && data && (
              <span className="ml-2 text-sm font-normal text-gray-400">
                (last {days} days, {sortedDaily.length} rows)
              </span>
            )}
          </h2>
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 overflow-x-auto">
            <table className="text-sm" style={{ tableLayout: "fixed", width: dailyTableWidth + "px", minWidth: "100%" }}>
              <colgroup>
                {DAILY_COLS.map(({ key }) => <col key={key} style={{ width: daily.widths[key] + "px" }} />)}
              </colgroup>
              <thead className="bg-gray-50 border-b border-gray-200">
                <tr>
                  {DAILY_COLS.map(({ key, label }) => (
                    <th
                      key={key}
                      onClick={() => handleDailySortClick(key)}
                      className="relative px-3 py-3 text-left font-medium text-gray-600 cursor-pointer hover:bg-gray-100 select-none overflow-hidden"
                    >
                      <span className="truncate">
                        {label}
                        <SortIcon active={dailySortCol === key} dir={dailySortDir} />
                      </span>
                      <ResizeHandle col={key} onDragStart={daily.onDragStart} />
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100">
                {loading ? (
                  <tr><td colSpan={3} className="px-4 py-6 text-center text-gray-400">Loading…</td></tr>
                ) : sortedDaily.length === 0 ? (
                  <tr><td colSpan={3} className="px-4 py-6 text-center text-gray-400">No data for this period.</td></tr>
                ) : sortedDaily.map((row) => (
                  <tr key={row.date} className="hover:bg-gray-50 transition-colors">
                    <td className="px-3 py-2.5 tabular-nums text-gray-700">{row.date}</td>
                    <td className="px-3 py-2.5 tabular-nums text-gray-600">{row.users}</td>
                    <td className="px-3 py-2.5 tabular-nums text-gray-600">{row.visits}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
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
