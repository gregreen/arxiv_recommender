import { useState, useEffect } from "react";
import { getAdminUsers, patchAdminUser, type AdminUser } from "../api/admin";
import { formatTimestamp } from "../utils";

export default function AdminUsersPage() {
  const [users, setUsers] = useState<AdminUser[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [toggling, setToggling] = useState<number | null>(null);

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
      setUsers((prev) =>
        prev.map((x) => (x.id === u.id ? { ...x, is_active: !u.is_active } : x))
      );
    } catch {
      // ignore
    } finally {
      setToggling(null);
    }
  }

  if (loading) return <div className="p-6 text-gray-500">Loading users…</div>;
  if (error)   return <div className="p-6 text-red-600">{error}</div>;

  return (
    <div className="p-6">
      <h1 className="text-xl font-bold text-gray-800 mb-4">Users <span className="text-sm font-normal text-gray-500">({users.length})</span></h1>
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden">
        <table className="min-w-full text-sm">
          <thead className="bg-gray-50 border-b border-gray-200">
            <tr>
              <th className="text-left px-4 py-3 font-medium text-gray-600">ID</th>
              <th className="text-left px-4 py-3 font-medium text-gray-600">Email</th>
              <th className="text-left px-4 py-3 font-medium text-gray-600">Status</th>
              <th className="text-left px-4 py-3 font-medium text-gray-600">Admin</th>
              <th className="text-left px-4 py-3 font-medium text-gray-600">Papers</th>
              <th className="text-left px-4 py-3 font-medium text-gray-600">Model trained</th>
              <th className="text-left px-4 py-3 font-medium text-gray-600">Registered</th>
              <th className="px-4 py-3" />
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100">
            {users.map((u) => (
              <tr key={u.id} className="hover:bg-gray-50 transition-colors">
                <td className="px-4 py-3 text-gray-400 tabular-nums">{u.id}</td>
                <td className="px-4 py-3 font-medium text-gray-800 max-w-64 truncate">{u.email}</td>
                <td className="px-4 py-3">
                  <span className={`inline-block text-xs font-semibold px-2 py-0.5 rounded-full ${
                    u.is_active
                      ? "bg-green-100 text-green-700"
                      : "bg-yellow-100 text-yellow-700"
                  }`}>
                    {u.is_active ? "Active" : "Pending"}
                  </span>
                </td>
                <td className="px-4 py-3">
                  {u.is_admin && (
                    <span className="inline-block text-xs font-semibold px-2 py-0.5 rounded-full bg-red-100 text-red-700">
                      Admin
                    </span>
                  )}
                </td>
                <td className="px-4 py-3 text-gray-600 tabular-nums">{u.paper_count}</td>
                <td className="px-4 py-3 text-gray-400 text-xs tabular-nums">
                  {u.model_trained_at ? formatTimestamp(u.model_trained_at) : "—"}
                </td>
                <td className="px-4 py-3 text-gray-400 text-xs tabular-nums">
                  {formatTimestamp(u.created_at)}
                </td>
                <td className="px-4 py-3 text-right">
                  <button
                    onClick={() => handleToggleActive(u)}
                    disabled={toggling === u.id}
                    className={`text-xs px-3 py-1 rounded font-medium transition-colors disabled:opacity-40 ${
                      u.is_active
                        ? "bg-yellow-100 hover:bg-yellow-200 text-yellow-800"
                        : "bg-green-100 hover:bg-green-200 text-green-800"
                    }`}
                  >
                    {u.is_active ? "Deactivate" : "Activate"}
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
