import { useState, useEffect } from "react";
import { getAdminUsers, patchAdminUser, resetUserImportLog, type AdminUser } from "../api/admin";
import { formatTimestamp } from "../utils";

// ---------------------------------------------------------------------------
// Column resize hook
// ---------------------------------------------------------------------------

type ColKey = "id" | "email" | "status" | "email_verified" | "admin" | "paper_count" | "import_count" | "model_trained_at" | "created_at" | "actions";

const DEFAULT_WIDTHS: Record<ColKey, number> = {
  id:              55,
  email:          240,
  status:          90,
  email_verified:  90,
  admin:           75,
  paper_count:     75,
  import_count:    80,
  model_trained_at:170,
  created_at:     170,
  actions:        150,
};

function useResizableColumns() {
  const [widths, setWidths] = useState<Record<ColKey, number>>(DEFAULT_WIDTHS);

  function onDragStart(col: ColKey, startX: number) {
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
type SortableCol = Exclude<ColKey, "actions">;

function getValue(u: AdminUser, col: SortableCol): string | number {
  switch (col) {
    case "id":              return u.id;
    case "email":           return u.email.toLowerCase();
    case "status":          return u.is_active ? 1 : 0;
    case "email_verified":  return u.email_verified ? 1 : 0;
    case "admin":           return u.is_admin ? 1 : 0;
    case "paper_count":     return u.paper_count;
    case "import_count":    return u.import_count;
    case "model_trained_at": return u.model_trained_at ?? "";
    case "created_at":      return u.created_at;
  }
}

function sortUsers(items: AdminUser[], col: SortableCol, dir: SortDir): AdminUser[] {
  return [...items].sort((a, b) => {
    const av = getValue(a, col);
    const bv = getValue(b, col);
    const cmp = typeof av === "number" && typeof bv === "number"
      ? av - bv
      : String(av).localeCompare(String(bv));
    return dir === "asc" ? cmp : -cmp;
  });
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const COLS: { key: ColKey; label: string; sortable: boolean }[] = [
  { key: "id",              label: "ID",            sortable: true  },
  { key: "email",           label: "Email",         sortable: true  },
  { key: "status",          label: "Status",        sortable: true  },
  { key: "email_verified",  label: "Verified",      sortable: true  },
  { key: "admin",           label: "Admin",         sortable: true  },
  { key: "paper_count",     label: "Papers",        sortable: true  },
  { key: "import_count",    label: "Imports",       sortable: true  },
  { key: "model_trained_at",label: "Model trained", sortable: true  },
  { key: "created_at",      label: "Registered",    sortable: true  },
  { key: "actions",         label: "",              sortable: false },
];

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function AdminUsersPage() {
  const [users, setUsers]     = useState<AdminUser[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError]     = useState<string | null>(null);
  const [toggling, setToggling] = useState<number | null>(null);
  const [actionPending, setActionPending] = useState<Set<number>>(new Set());

  // Filters
  const [search, setSearch]           = useState("");
  const [statusFilter, setStatusFilter] = useState<"" | "active" | "pending">("");
  const [adminFilter, setAdminFilter]   = useState<"" | "admin" | "non-admin">("");

  // Sort
  const [sortCol, setSortCol] = useState<SortableCol>("id");
  const [sortDir, setSortDir] = useState<SortDir>("asc");

  const { widths, onDragStart } = useResizableColumns();

  useEffect(() => {
    getAdminUsers()
      .then(setUsers)
      .catch(() => setError("Failed to load users."))
      .finally(() => setLoading(false));
  }, []);

  async function handleToggleActive(u: AdminUser) {
    setToggling(u.id);
    try {
      await patchAdminUser(u.id, !u.is_active);
      setUsers((prev) => prev.map((x) => x.id === u.id ? { ...x, is_active: !u.is_active } : x));
    } catch {
      // ignore
    } finally {
      setToggling(null);
    }
  }

  async function handleResetImports(u: AdminUser) {
    if (!window.confirm(`Reset import log for ${u.email}? This resets their rate-limit tier back to Tier A.`)) return;
    setActionPending((prev) => new Set(prev).add(u.id));
    try {
      await resetUserImportLog(u.id);
      setUsers((prev) => prev.map((x) => x.id === u.id ? { ...x, import_count: 0 } : x));
    } catch {
      // ignore
    } finally {
      setActionPending((prev) => { const s = new Set(prev); s.delete(u.id); return s; });
    }
  }

  function handleSortClick(col: SortableCol) {
    if (col === sortCol) setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    else { setSortCol(col); setSortDir("asc"); }
  }

  // Derive visible rows
  const filtered = users.filter((u) => {
    const q = search.trim().toLowerCase();
    if (q && !u.email.toLowerCase().includes(q) && !String(u.id).includes(q)) return false;
    if (statusFilter === "active"  && !u.is_active) return false;
    if (statusFilter === "pending" &&  u.is_active) return false;
    if (adminFilter  === "admin"     && !u.is_admin) return false;
    if (adminFilter  === "non-admin" &&  u.is_admin) return false;
    return true;
  });
  const sorted = sortUsers(filtered, sortCol, sortDir);

  function SortIcon({ col }: { col: ColKey }) {
    if (col === "actions" || col !== sortCol) return <span className="ml-1 text-gray-300">↕</span>;
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

  if (loading) return <div className="p-6 text-gray-500">Loading users…</div>;
  if (error)   return <div className="p-6 text-red-600">{error}</div>;

  return (
    <div className="p-6 flex flex-col gap-4">
      <h1 className="text-xl font-bold text-gray-800">
        Users{" "}
        <span className="text-sm font-normal text-gray-500">
          ({sorted.length.toLocaleString()}{sorted.length !== users.length ? ` of ${users.length}` : ""})
        </span>
      </h1>

      {/* Filters row */}
      <div className="flex gap-3 flex-wrap items-center">
        <input
          type="text"
          placeholder="Search email or ID…"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="border border-gray-300 rounded px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-red-400 w-56"
        />
        <select
          value={statusFilter}
          onChange={(e) => setStatusFilter(e.target.value as typeof statusFilter)}
          className="border border-gray-300 rounded px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-red-400"
        >
          <option value="">All statuses</option>
          <option value="active">Active</option>
          <option value="pending">Pending</option>
        </select>
        <select
          value={adminFilter}
          onChange={(e) => setAdminFilter(e.target.value as typeof adminFilter)}
          className="border border-gray-300 rounded px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-red-400"
        >
          <option value="">All roles</option>
          <option value="admin">Admin only</option>
          <option value="non-admin">Non-admin</option>
        </select>
      </div>

      <div className="bg-white rounded-lg shadow-sm border border-gray-200 overflow-x-auto">
        <table className="text-sm" style={{ tableLayout: "fixed", width: tableWidth + "px", minWidth: "100%" }}>
          <colgroup>
            {COLS.map(({ key }) => <col key={key} style={{ width: widths[key] + "px" }} />)}
          </colgroup>
          <thead className="bg-gray-50 border-b border-gray-200">
            <tr>
              {COLS.map(({ key, label, sortable }) => (
                <th
                  key={key}
                  onClick={sortable ? () => handleSortClick(key as SortableCol) : undefined}
                  className={`relative px-3 py-3 text-left font-medium text-gray-600 select-none overflow-hidden ${sortable ? "cursor-pointer hover:bg-gray-100" : ""}`}
                >
                  <span className="truncate">{label}{sortable && <SortIcon col={key} />}</span>
                  <ResizeHandle col={key} />
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100">
            {sorted.length === 0 ? (
              <tr><td colSpan={10} className="px-4 py-6 text-center text-gray-400">No users match the current filters.</td></tr>
            ) : sorted.map((u) => (
              <tr key={u.id} className="hover:bg-gray-50 transition-colors">
                <td className="px-3 py-2.5 text-gray-400 tabular-nums truncate">{u.id}</td>
                <td className="px-3 py-2.5 font-medium text-gray-800 truncate" title={u.email}>{u.email}</td>
                <td className="px-3 py-2.5">
                  <span className={`inline-block text-xs font-semibold px-2 py-0.5 rounded-full ${u.is_active ? "bg-green-100 text-green-700" : "bg-yellow-100 text-yellow-700"}`}>
                    {u.is_active ? "Active" : "Pending"}
                  </span>
                </td>
                <td className="px-3 py-2.5">
                  <span className={`inline-block text-xs font-semibold px-2 py-0.5 rounded-full ${u.email_verified ? "bg-blue-100 text-blue-700" : "bg-gray-100 text-gray-500"}`}>
                    {u.email_verified ? "Verified" : "Unverified"}
                  </span>
                </td>
                <td className="px-3 py-2.5">
                  {u.is_admin && (
                    <span className="inline-block text-xs font-semibold px-2 py-0.5 rounded-full bg-red-100 text-red-700">Admin</span>
                  )}
                </td>
                <td className="px-3 py-2.5 text-gray-600 tabular-nums">{u.paper_count}</td>
                <td className="px-3 py-2.5 text-gray-600 tabular-nums">{u.import_count}</td>
                <td className="px-3 py-2.5 text-gray-400 text-xs tabular-nums truncate">
                  {u.model_trained_at ? formatTimestamp(u.model_trained_at) : "—"}
                </td>
                <td className="px-3 py-2.5 text-gray-400 text-xs tabular-nums truncate">{formatTimestamp(u.created_at)}</td>
                <td className="px-3 py-2.5 text-right flex gap-1 justify-end">
                  <button
                    onClick={() => handleToggleActive(u)}
                    disabled={toggling === u.id}
                    className={`text-xs px-3 py-1 rounded font-medium transition-colors disabled:opacity-40 ${u.is_active ? "bg-yellow-100 hover:bg-yellow-200 text-yellow-800" : "bg-green-100 hover:bg-green-200 text-green-800"}`}
                  >
                    {u.is_active ? "Deactivate" : "Activate"}
                  </button>
                  <button
                    onClick={() => handleResetImports(u)}
                    disabled={actionPending.has(u.id)}
                    className="text-xs px-2 py-1 rounded font-medium transition-colors disabled:opacity-40 bg-gray-100 hover:bg-gray-200 text-gray-700"
                    title="Reset import log (resets rate-limit tier)"
                  >
                    Reset imports
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
