import { useState, useEffect, useCallback } from "react";
import { getAdminTasks, type AdminTask } from "../api/admin";
import { formatTimestamp } from "../utils";

// ---------------------------------------------------------------------------
// Column resize hook
// ---------------------------------------------------------------------------

type ColKey = "id" | "type" | "payload" | "status" | "attempts" | "created_at" | "started_at" | "completed_at";

const DEFAULT_WIDTHS: Record<ColKey, number> = {
  id:            60,
  type:         110,
  payload:      200,
  status:        90,
  attempts:      80,
  created_at:   170,
  started_at:   170,
  completed_at: 170,
};

function useResizableColumns() {
  const [widths, setWidths] = useState<Record<ColKey, number>>(DEFAULT_WIDTHS);

  function onDragStart(col: ColKey, startX: number) {
    const startW = widths[col];
    function onMove(e: PointerEvent) {
      const next = Math.max(40, startW + (e.clientX - startX));
      setWidths((w) => ({ ...w, [col]: next }));
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

function sortTasks(items: AdminTask[], col: ColKey, dir: SortDir): AdminTask[] {
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
// Constants
// ---------------------------------------------------------------------------

const TYPE_OPTIONS   = ["fetch_meta", "embed", "recommend"] as const;
const STATUS_OPTIONS = ["pending", "running", "done", "failed"] as const;

const STATUS_COLORS: Record<string, string> = {
  pending: "bg-gray-100 text-gray-700",
  running: "bg-blue-100 text-blue-700",
  done:    "bg-green-100 text-green-700",
  failed:  "bg-red-100  text-red-700",
};

const COLS: { key: ColKey; label: string }[] = [
  { key: "id",           label: "ID"        },
  { key: "type",         label: "Type"      },
  { key: "payload",      label: "Payload"   },
  { key: "status",       label: "Status"    },
  { key: "attempts",     label: "Attempts"  },
  { key: "created_at",   label: "Created"   },
  { key: "started_at",   label: "Started"   },
  { key: "completed_at", label: "Completed" },
];

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function AdminTasksPage() {
  const [allItems, setAllItems] = useState<AdminTask[]>([]);
  const [total, setTotal]       = useState(0);
  const [loading, setLoading]   = useState(true);
  const [error, setError]       = useState<string | null>(null);

  // Server-side filters
  const [typeFilter,   setTypeFilter]   = useState("");
  const [statusFilter, setStatusFilter] = useState("");
  const [payloadQuery, setPayloadQuery] = useState("");  // committed value sent to API
  const [payloadInput, setPayloadInput] = useState("");  // live input field

  // Client-side sort
  const [sortCol, setSortCol] = useState<ColKey>("id");
  const [sortDir, setSortDir] = useState<SortDir>("desc");

  const { widths, onDragStart } = useResizableColumns();

  const load = useCallback((type: string, status: string, q: string) => {
    setLoading(true);
    getAdminTasks({
      type:   type   || undefined,
      status: status || undefined,
      q:      q      || undefined,
      limit:  8192,
      offset: 0,
    })
      .then((data) => { setAllItems(data.items); setTotal(data.total); })
      .catch(() => setError("Failed to load tasks."))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    load(typeFilter, statusFilter, payloadQuery);
  }, [load, typeFilter, statusFilter, payloadQuery]);

  function handleSortClick(col: ColKey) {
    if (col === sortCol) setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    else { setSortCol(col); setSortDir("asc"); }
  }

  function handlePayloadSearch(e: React.FormEvent) {
    e.preventDefault();
    setPayloadQuery(payloadInput);
  }

  const sorted = sortTasks(allItems, sortCol, sortDir);

  function SortIcon({ col }: { col: ColKey }) {
    if (col !== sortCol) return <span className="ml-1 text-gray-300">↕</span>;
    return <span className="ml-1">{sortDir === "asc" ? "↑" : "↓"}</span>;
  }

  function ResizeHandle({ col }: { col: ColKey }) {
    return (
      <div
        onPointerDown={(e) => { e.preventDefault(); onDragStart(col, e.clientX); }}
        className="absolute right-0 top-0 h-full w-1.5 cursor-col-resize hover:bg-red-300 select-none"
      />
    );
  }

  const tableWidth = Object.values(widths).reduce((a, b) => a + b, 0);

  return (
    <div className="p-6 flex flex-col gap-4">
      <h1 className="text-xl font-bold text-gray-800">
        Task Queue{" "}
        {!loading && (
          <span className="text-sm font-normal text-gray-500">
            ({sorted.length.toLocaleString()}{total > allItems.length ? ` shown of ${total.toLocaleString()} matched` : ""})
          </span>
        )}
      </h1>

      {/* Filters row */}
      <div className="flex gap-3 flex-wrap items-center">
        <select
          value={typeFilter}
          onChange={(e) => setTypeFilter(e.target.value)}
          className="border border-gray-300 rounded px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-red-400"
        >
          <option value="">All types</option>
          {TYPE_OPTIONS.map((t) => <option key={t} value={t}>{t}</option>)}
        </select>

        <select
          value={statusFilter}
          onChange={(e) => setStatusFilter(e.target.value)}
          className="border border-gray-300 rounded px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-red-400"
        >
          <option value="">All statuses</option>
          {STATUS_OPTIONS.map((s) => <option key={s} value={s}>{s}</option>)}
        </select>

        <form onSubmit={handlePayloadSearch} className="flex gap-2 items-center">
          <input
            type="text"
            placeholder="Search payload…"
            value={payloadInput}
            onChange={(e) => setPayloadInput(e.target.value)}
            className="border border-gray-300 rounded px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-red-400 w-48"
          />
          <button
            type="submit"
            className="bg-red-700 hover:bg-red-800 text-white text-sm font-medium rounded px-3 py-1.5 transition-colors"
          >
            Search
          </button>
          {payloadQuery && (
            <button
              type="button"
              onClick={() => { setPayloadInput(""); setPayloadQuery(""); }}
              className="text-sm text-gray-400 hover:text-gray-700 px-1"
            >
              ✕
            </button>
          )}
        </form>
      </div>

      {error && <div className="text-red-600">{error}</div>}

      <div className="bg-white rounded-lg shadow-sm border border-gray-200 overflow-x-auto">
        <table className="text-sm" style={{ tableLayout: "fixed", width: tableWidth + "px", minWidth: "100%" }}>
          <colgroup>
            {COLS.map(({ key }) => <col key={key} style={{ width: widths[key] + "px" }} />)}
          </colgroup>
          <thead className="bg-gray-50 border-b border-gray-200">
            <tr>
              {COLS.map(({ key, label }) => (
                <th
                  key={key}
                  onClick={() => handleSortClick(key)}
                  className="relative px-3 py-3 text-left font-medium text-gray-600 cursor-pointer hover:bg-gray-100 select-none overflow-hidden"
                >
                  <span className="truncate">{label}<SortIcon col={key} /></span>
                  <ResizeHandle col={key} />
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100">
            {loading ? (
              <tr><td colSpan={8} className="px-4 py-6 text-center text-gray-400">Loading…</td></tr>
            ) : sorted.length === 0 ? (
              <tr><td colSpan={8} className="px-4 py-6 text-center text-gray-400">No tasks found.</td></tr>
            ) : sorted.map((t) => (
              <tr key={t.id} className="hover:bg-gray-50 transition-colors">
                <td className="px-3 py-2.5 text-gray-400 tabular-nums truncate">{t.id}</td>
                <td className="px-3 py-2.5 text-gray-700 font-mono text-xs truncate">{t.type}</td>
                <td className="px-3 py-2.5 text-gray-500 font-mono text-xs truncate" title={t.payload}>{t.payload}</td>
                <td className="px-3 py-2.5">
                  <span className={`inline-block text-xs font-semibold px-2 py-0.5 rounded-full ${STATUS_COLORS[t.status] ?? "bg-gray-100 text-gray-700"}`}>
                    {t.status}
                  </span>
                  {t.error && (
                    <div className="text-xs text-red-500 mt-0.5 truncate" title={t.error}>{t.error}</div>
                  )}
                </td>
                <td className="px-3 py-2.5 text-gray-600 tabular-nums">{t.attempts}</td>
                <td className="px-3 py-2.5 text-gray-400 text-xs tabular-nums truncate">{formatTimestamp(t.created_at)}</td>
                <td className="px-3 py-2.5 text-gray-400 text-xs tabular-nums truncate">{t.started_at ? formatTimestamp(t.started_at) : "—"}</td>
                <td className="px-3 py-2.5 text-gray-400 text-xs tabular-nums truncate">{t.completed_at ? formatTimestamp(t.completed_at) : "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
