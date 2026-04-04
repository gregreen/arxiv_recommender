import { useState, useEffect, useCallback } from "react";
import { getAdminTasks, type AdminTask, type Paginated } from "../api/admin";
import { formatTimestamp } from "../utils";

const PAGE_SIZE = 50;

const TYPE_OPTIONS   = ["", "fetch_meta", "embed", "recommend"] as const;
const STATUS_OPTIONS = ["", "pending", "running", "done", "failed"] as const;

const STATUS_COLORS: Record<string, string> = {
  pending: "bg-gray-100 text-gray-700",
  running: "bg-blue-100 text-blue-700",
  done:    "bg-green-100 text-green-700",
  failed:  "bg-red-100 text-red-700",
};

export default function AdminTasksPage() {
  const [data, setData]       = useState<Paginated<AdminTask> | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError]     = useState<string | null>(null);
  const [typeFilter, setTypeFilter]     = useState("");
  const [statusFilter, setStatusFilter] = useState("");
  const [offset, setOffset]   = useState(0);

  const load = useCallback((type: string, status: string, off: number) => {
    setLoading(true);
    getAdminTasks({
      type:   type   || undefined,
      status: status || undefined,
      limit:  PAGE_SIZE,
      offset: off,
    })
      .then(setData)
      .catch(() => setError("Failed to load tasks."))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    load(typeFilter, statusFilter, offset);
  }, [load, typeFilter, statusFilter, offset]);

  function handleFilter(type: string, status: string) {
    setTypeFilter(type);
    setStatusFilter(status);
    setOffset(0);
  }

  return (
    <div className="p-6">
      <h1 className="text-xl font-bold text-gray-800 mb-4">Task Queue</h1>

      {/* Filters */}
      <div className="flex gap-3 mb-4 flex-wrap">
        <select
          value={typeFilter}
          onChange={(e) => handleFilter(e.target.value, statusFilter)}
          className="border border-gray-300 rounded px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-red-400"
        >
          <option value="">All types</option>
          {TYPE_OPTIONS.slice(1).map((t) => (
            <option key={t} value={t}>{t}</option>
          ))}
        </select>
        <select
          value={statusFilter}
          onChange={(e) => handleFilter(typeFilter, e.target.value)}
          className="border border-gray-300 rounded px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-red-400"
        >
          <option value="">All statuses</option>
          {STATUS_OPTIONS.slice(1).map((s) => (
            <option key={s} value={s}>{s}</option>
          ))}
        </select>
        {data && (
          <span className="self-center text-sm text-gray-500">
            {data.total.toLocaleString()} total
          </span>
        )}
      </div>

      {error && <div className="text-red-600 mb-4">{error}</div>}

      <div className="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden">
        <table className="min-w-full text-sm">
          <thead className="bg-gray-50 border-b border-gray-200">
            <tr>
              <th className="text-left px-4 py-3 font-medium text-gray-600">ID</th>
              <th className="text-left px-4 py-3 font-medium text-gray-600">Type</th>
              <th className="text-left px-4 py-3 font-medium text-gray-600">Payload</th>
              <th className="text-left px-4 py-3 font-medium text-gray-600">Status</th>
              <th className="text-left px-4 py-3 font-medium text-gray-600">Attempts</th>
              <th className="text-left px-4 py-3 font-medium text-gray-600">Created</th>
              <th className="text-left px-4 py-3 font-medium text-gray-600">Completed</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100">
            {loading ? (
              <tr>
                <td colSpan={7} className="px-4 py-6 text-center text-gray-400">Loading…</td>
              </tr>
            ) : data?.items.length === 0 ? (
              <tr>
                <td colSpan={7} className="px-4 py-6 text-center text-gray-400">No tasks found.</td>
              </tr>
            ) : (
              data?.items.map((t) => (
                <tr key={t.id} className="hover:bg-gray-50 transition-colors">
                  <td className="px-4 py-3 text-gray-400 tabular-nums">{t.id}</td>
                  <td className="px-4 py-3 text-gray-700 font-mono text-xs">{t.type}</td>
                  <td className="px-4 py-3 text-gray-500 font-mono text-xs max-w-48 truncate" title={t.payload}>
                    {t.payload}
                  </td>
                  <td className="px-4 py-3">
                    <span className={`inline-block text-xs font-semibold px-2 py-0.5 rounded-full ${STATUS_COLORS[t.status] ?? "bg-gray-100 text-gray-700"}`}>
                      {t.status}
                    </span>
                    {t.error && (
                      <div className="text-xs text-red-500 mt-0.5 max-w-48 truncate" title={t.error}>
                        {t.error}
                      </div>
                    )}
                  </td>
                  <td className="px-4 py-3 text-gray-600 tabular-nums">{t.attempts}</td>
                  <td className="px-4 py-3 text-gray-400 text-xs tabular-nums">{formatTimestamp(t.created_at)}</td>
                  <td className="px-4 py-3 text-gray-400 text-xs tabular-nums">
                    {t.completed_at ? formatTimestamp(t.completed_at) : "—"}
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      {data && data.total > PAGE_SIZE && (
        <div className="flex items-center gap-3 mt-4 text-sm">
          <button
            onClick={() => setOffset((o) => Math.max(0, o - PAGE_SIZE))}
            disabled={offset === 0}
            className="px-3 py-1.5 rounded border border-gray-300 text-gray-600 hover:bg-gray-50 disabled:opacity-40 transition-colors"
          >
            ← Prev
          </button>
          <span className="text-gray-500">
            {offset + 1}–{Math.min(offset + PAGE_SIZE, data.total)} of {data.total.toLocaleString()}
          </span>
          <button
            onClick={() => setOffset((o) => o + PAGE_SIZE)}
            disabled={offset + PAGE_SIZE >= data.total}
            className="px-3 py-1.5 rounded border border-gray-300 text-gray-600 hover:bg-gray-50 disabled:opacity-40 transition-colors"
          >
            Next →
          </button>
        </div>
      )}
    </div>
  );
}
