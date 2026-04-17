import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import {
  getAdminGroups,
  getAdminGroup,
  deleteAdminGroup,
  type AdminGroup,
  type AdminGroupDetail,
} from "../api/admin";
import { formatTimestamp } from "../utils";

export default function AdminGroupsPage() {
  const navigate = useNavigate();
  const [groups, setGroups]           = useState<AdminGroup[] | null>(null);
  const [loading, setLoading]         = useState(true);
  const [error, setError]             = useState<string | null>(null);
  const [selectedId, setSelectedId]   = useState<number | null>(null);
  const [detail, setDetail]           = useState<AdminGroupDetail | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [detailError, setDetailError] = useState<string | null>(null);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [deleting, setDeleting]       = useState(false);

  useEffect(() => {
    setLoading(true);
    getAdminGroups()
      .then(setGroups)
      .catch(() => setError("Failed to load groups."))
      .finally(() => setLoading(false));
  }, []);

  function handleSelect(id: number) {
    setSelectedId(id);
    setDetail(null);
    setDetailError(null);
    setConfirmDelete(false);
    setDetailLoading(true);
    getAdminGroup(id)
      .then(setDetail)
      .catch(() => setDetailError("Failed to load group detail."))
      .finally(() => setDetailLoading(false));
  }

  function handleClearSelection() {
    setSelectedId(null);
    setDetail(null);
    setConfirmDelete(false);
  }

  async function handleDelete() {
    if (selectedId === null) return;
    setDeleting(true);
    try {
      await deleteAdminGroup(selectedId);
      setGroups((prev) => prev?.filter((g) => g.id !== selectedId) ?? null);
      setSelectedId(null);
      setDetail(null);
      setConfirmDelete(false);
    } catch {
      setDetailError("Failed to delete group.");
    } finally {
      setDeleting(false);
    }
  }

  return (
    <div className="relative flex h-full overflow-hidden">
      {/* ── Left pane: group list ────────────────────────────────────────── */}
      <div className={`absolute inset-0 z-10 flex flex-col bg-white transition-transform duration-300 ease-in-out
        md:relative md:w-80 md:shrink-0 md:border-r md:border-gray-200 md:translate-x-0
        ${selectedId !== null ? "-translate-x-full" : "translate-x-0"}`}>

        {/* Mobile: back to category nav */}
        <div className="md:hidden shrink-0 flex items-center px-4 py-2 bg-white border-b border-gray-200">
          <button
            onClick={() => navigate("/admin")}
            className="flex items-center gap-1.5 text-sm text-red-700 hover:text-red-900 transition-colors"
          >
            ← Return to list
          </button>
        </div>

        <div className="px-4 py-3 border-b border-gray-200 shrink-0 flex items-center justify-between">
          <h2 className="text-sm font-semibold text-gray-700">Groups</h2>
          {groups && (
            <span className="text-xs text-gray-400">{groups.length.toLocaleString()} total</span>
          )}
        </div>

        <div className="flex-1 overflow-y-auto divide-y divide-gray-100">
          {error && <div className="p-4 text-red-600 text-sm">{error}</div>}
          {loading && !groups && (
            <div className="p-4 text-gray-400 text-sm">Loading…</div>
          )}
          {groups?.length === 0 && (
            <div className="p-4 text-gray-400 text-sm">No groups yet.</div>
          )}
          {groups?.map((g) => (
            <button
              key={g.id}
              onClick={() => handleSelect(g.id)}
              className={`w-full text-left px-4 py-3 hover:bg-gray-50 transition-colors ${
                selectedId === g.id ? "bg-red-50 border-l-2 border-red-600" : ""
              }`}
            >
              <div className="text-sm font-medium text-gray-800 truncate">{g.name}</div>
              <div className="text-xs text-gray-400 mt-0.5">
                {g.member_count} member{g.member_count !== 1 ? "s" : ""}
                {g.pending_invite_count > 0 && (
                  <span className="ml-1.5 text-amber-600">
                    · {g.pending_invite_count} pending invite{g.pending_invite_count !== 1 ? "s" : ""}
                  </span>
                )}
              </div>
              <div className="text-xs text-gray-400 truncate">
                admin: {g.admin_emails.join(", ") || "—"}
              </div>
              <div className="text-xs text-gray-300 mt-0.5">
                created {formatTimestamp(g.created_at)}
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* ── Right pane: group detail ─────────────────────────────────────── */}
      <div className={`absolute inset-0 w-full flex flex-col bg-white transition-transform duration-300 ease-in-out
        md:relative md:flex-1 md:min-w-0 md:translate-x-0
        ${selectedId !== null ? "translate-x-0" : "translate-x-full"}`}>

        {/* Back button — mobile only */}
        <div className="md:hidden shrink-0 flex items-center px-4 py-2 bg-white border-b border-gray-200">
          <button
            onClick={handleClearSelection}
            className="flex items-center gap-1.5 text-sm text-red-700 hover:text-red-900 transition-colors"
          >
            ← Return to list
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-6">
          {selectedId === null && (
            <p className="text-gray-400 text-sm">Select a group to view details.</p>
          )}
          {detailLoading && (
            <p className="text-gray-400 text-sm">Loading…</p>
          )}
          {detailError && (
            <p className="text-red-600 text-sm">{detailError}</p>
          )}
          {detail && !detailLoading && (
            <div className="space-y-6 max-w-2xl">
              {/* Header */}
              <div className="flex items-start justify-between gap-4">
                <div>
                  <h2 className="text-lg font-semibold text-gray-900">{detail.name}</h2>
                  <p className="text-xs text-gray-400 mt-0.5">
                    ID {detail.id} · created {formatTimestamp(detail.created_at)}
                  </p>
                </div>
                {/* Delete button */}
                {!confirmDelete ? (
                  <button
                    onClick={() => setConfirmDelete(true)}
                    className="shrink-0 px-3 py-1.5 text-sm font-medium rounded border border-red-300 text-red-600 hover:bg-red-50 transition-colors"
                  >
                    Delete group
                  </button>
                ) : (
                  <div className="shrink-0 flex items-center gap-2">
                    <span className="text-sm text-red-700 font-medium">Delete permanently?</span>
                    <button
                      onClick={handleDelete}
                      disabled={deleting}
                      className="px-3 py-1.5 text-sm font-medium rounded bg-red-600 text-white hover:bg-red-700 disabled:opacity-50 transition-colors"
                    >
                      {deleting ? "Deleting…" : "Confirm"}
                    </button>
                    <button
                      onClick={() => setConfirmDelete(false)}
                      className="px-3 py-1.5 text-sm font-medium rounded border border-gray-300 text-gray-600 hover:bg-gray-50 transition-colors"
                    >
                      Cancel
                    </button>
                  </div>
                )}
              </div>

              {/* Members */}
              <section>
                <h3 className="text-sm font-semibold text-gray-700 mb-2">
                  Members ({detail.members.length})
                </h3>
                {detail.members.length === 0 ? (
                  <p className="text-xs text-gray-400">No members.</p>
                ) : (
                  <div className="rounded border border-gray-200 overflow-hidden text-sm">
                    <table className="w-full">
                      <thead className="bg-gray-50 text-xs text-gray-500 uppercase tracking-wide">
                        <tr>
                          <th className="px-3 py-2 text-left font-medium">Email</th>
                          <th className="px-3 py-2 text-left font-medium">Role</th>
                          <th className="px-3 py-2 text-left font-medium">Joined</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-gray-100">
                        {detail.members.map((m) => (
                          <tr key={m.user_id} className="hover:bg-gray-50">
                            <td className="px-3 py-2 text-gray-800 truncate max-w-xs">{m.email}</td>
                            <td className="px-3 py-2">
                              {m.is_admin ? (
                                <span className="inline-block px-1.5 py-0.5 rounded text-xs font-medium bg-red-100 text-red-700">
                                  Admin
                                </span>
                              ) : (
                                <span className="text-gray-400 text-xs">Member</span>
                              )}
                            </td>
                            <td className="px-3 py-2 text-gray-400 text-xs tabular-nums whitespace-nowrap">
                              {formatTimestamp(m.joined_at)}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </section>

              {/* Pending invites */}
              <section>
                <h3 className="text-sm font-semibold text-gray-700 mb-2">
                  Pending invites ({detail.pending_invites.length})
                </h3>
                {detail.pending_invites.length === 0 ? (
                  <p className="text-xs text-gray-400">No pending invites.</p>
                ) : (
                  <div className="rounded border border-gray-200 overflow-hidden text-sm">
                    <table className="w-full">
                      <thead className="bg-gray-50 text-xs text-gray-500 uppercase tracking-wide">
                        <tr>
                          <th className="px-3 py-2 text-left font-medium">Token</th>
                          <th className="px-3 py-2 text-left font-medium">Created by</th>
                          <th className="px-3 py-2 text-left font-medium">Uses left</th>
                          <th className="px-3 py-2 text-left font-medium">Created</th>
                          <th className="px-3 py-2 text-left font-medium">Expires</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-gray-100">
                        {detail.pending_invites.map((inv) => (
                          <tr key={inv.id} className="hover:bg-gray-50">
                            <td className="px-3 py-2 font-mono text-xs text-gray-500 truncate max-w-[8rem]"
                                title={inv.token}>
                              {inv.token.slice(0, 12)}…
                            </td>
                            <td className="px-3 py-2 text-gray-700 truncate max-w-xs">{inv.created_by_email}</td>
                            <td className="px-3 py-2 tabular-nums text-xs">
                              <span className={inv.remaining_uses <= 3 ? "text-amber-600 font-medium" : "text-gray-600"}>
                                {inv.remaining_uses}
                              </span>
                            </td>
                            <td className="px-3 py-2 text-gray-400 text-xs tabular-nums whitespace-nowrap">
                              {formatTimestamp(inv.created_at)}
                            </td>
                            <td className="px-3 py-2 text-gray-400 text-xs tabular-nums whitespace-nowrap">
                              {formatTimestamp(inv.expires_at)}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </section>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
