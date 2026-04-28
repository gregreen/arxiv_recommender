import { useState, useEffect, useCallback } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { QRCodeSVG } from "qrcode.react";
import { useAuth } from "../AuthContext";
import { useGroups } from "../contexts/GroupsContext";
import AppNav from "../components/AppNav";
import {
  getGroup,
  createInvite,
  listInvites,
  revokeInvite,
  removeMember,
  makeAdmin,
  deleteGroup,
  type GroupDetail,
  type GroupInvite,
} from "../api/groups";

function InviteRow({ invite, origin, onRevoke }: {
  invite: GroupInvite;
  origin: string;
  onRevoke: (id: number) => void;
}) {
  const [copied, setCopied] = useState(false);
  const [showQr, setShowQr] = useState(false);
  const url = `${origin}/join-group?token=${encodeURIComponent(invite.token)}`;

  function handleCopy() {
    if (navigator.clipboard) {
      navigator.clipboard.writeText(url).then(() => {
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
      });
    } else {
      // Fallback for non-secure contexts (HTTP)
      const ta = document.createElement("textarea");
      ta.value = url;
      ta.style.position = "fixed";
      ta.style.opacity = "0";
      document.body.appendChild(ta);
      ta.focus();
      ta.select();
      document.execCommand("copy");
      document.body.removeChild(ta);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  }

  const expires = new Date(invite.expires_at).toLocaleDateString();
  const usesLabel = invite.remaining_uses === 1
    ? "1 use left"
    : `${invite.remaining_uses} uses left`;
  const usesColor = invite.remaining_uses <= 3 ? "text-amber-600" : "text-gray-400";

  return (
    <li className="flex flex-col bg-gray-50 border border-gray-200 rounded px-3 py-2 gap-2">
      <div className="flex items-center gap-2">
        <span className="font-mono text-xs text-gray-500 truncate flex-1" title={url}>{url}</span>
        <span className={`text-xs shrink-0 ${usesColor}`}>{usesLabel}</span>
        <span className="text-xs text-gray-400 shrink-0">exp {expires}</span>
        <button
          onClick={handleCopy}
          className="shrink-0 text-xs px-2 py-1 rounded bg-blue-600 hover:bg-blue-700 text-white transition-colors"
        >
          {copied ? "Copied!" : "Copy"}
        </button>
        <button
          onClick={() => setShowQr((v) => !v)}
          className={`shrink-0 text-xs px-2 py-1 rounded transition-colors ${showQr ? "bg-blue-100 text-blue-700 hover:bg-blue-200" : "bg-gray-200 text-gray-600 hover:bg-gray-300"}`}
        >
          QR
        </button>
        <button
          onClick={() => onRevoke(invite.id)}
          className="shrink-0 text-xs px-2 py-1 rounded bg-gray-200 hover:bg-red-100 hover:text-red-600 text-gray-600 transition-colors"
        >
          Revoke
        </button>
      </div>
      {showQr && (
        <div className="flex justify-center py-2">
          <QRCodeSVG value={url} size={180} />
        </div>
      )}
    </li>
  );
}

export default function GroupManagePage() {
  const { groupId: groupIdParam } = useParams<{ groupId: string }>();
  const groupId = Number(groupIdParam);
  const navigate = useNavigate();
  const { user } = useAuth();
  const { refetch: refetchGroups } = useGroups();

  const [group, setGroup] = useState<GroupDetail | null>(null);
  const [invites, setInvites] = useState<GroupInvite[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [deletingMember, setDeletingMember] = useState<number | null>(null);
  const [promoteConfirm, setPromoteConfirm] = useState<number | null>(null);
  const [deleteConfirm, setDeleteConfirm] = useState(false);
  const [showMultiUse, setShowMultiUse] = useState(false);
  const [multiUseCount, setMultiUseCount] = useState(25);

  const isCurrentUserAdmin = group?.members.some(
    (m) => m.user_id === user?.userId && m.is_admin,
  ) ?? false;

  const loadGroup = useCallback(async () => {
    try {
      const [g, inv] = await Promise.all([
        getGroup(groupId),
        listInvites(groupId).catch(() => [] as GroupInvite[]),
      ]);
      setGroup(g);
      setInvites(inv);
      // Redirect non-admins to home
      const isAdmin = g.members.some((m) => m.user_id === user?.userId && m.is_admin);
      if (!isAdmin) navigate("/", { replace: true });
    } catch {
      setError("Group not found or you don't have access.");
    } finally {
      setLoading(false);
    }
  }, [groupId, user?.userId, navigate]);

  useEffect(() => { loadGroup(); }, [loadGroup]);

  async function handleCreateInvite(maxUses: number = 1) {
    setActionError(null);
    try {
      const inv = await createInvite(groupId, maxUses);
      setInvites((prev) => [inv, ...prev]);
      setShowMultiUse(false);
    } catch (err: unknown) {
      setActionError(err instanceof Error ? err.message : "Failed to create invite");
    }
  }

  async function handleRevokeInvite(inviteId: number) {
    setActionError(null);
    try {
      await revokeInvite(groupId, inviteId);
      setInvites((prev) => prev.filter((i) => i.id !== inviteId));
    } catch (err: unknown) {
      setActionError(err instanceof Error ? err.message : "Failed to revoke invite");
    }
  }

  async function handleRemoveMember(userId: number) {
    setActionError(null);
    setDeletingMember(userId);
    try {
      await removeMember(groupId, userId);
      setGroup((prev) => prev
        ? { ...prev, members: prev.members.filter((m) => m.user_id !== userId) }
        : prev
      );
    } catch (err: unknown) {
      setActionError(err instanceof Error ? err.message : "Failed to remove member");
    } finally {
      setDeletingMember(null);
    }
  }

  async function handleMakeAdmin(userId: number) {
    setActionError(null);
    setPromoteConfirm(null);
    try {
      const updated = await makeAdmin(groupId, userId);
      setGroup(updated);
    } catch (err: unknown) {
      setActionError(err instanceof Error ? err.message : "Failed to grant admin");
    }
  }

  async function handleDeleteGroup() {
    setActionError(null);
    setDeleteConfirm(false);
    try {
      await deleteGroup(groupId);
      await refetchGroups();
      navigate("/groups");
    } catch (err: unknown) {
      setActionError(err instanceof Error ? err.message : "Failed to delete group");
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen text-gray-400">Loading…</div>
    );
  }

  if (error || !group) {
    return (
      <div className="flex flex-col items-center justify-center h-screen gap-4">
        <p className="text-red-600">{error ?? "Failed to load group"}</p>
        <Link to="/groups" className="text-blue-600 hover:underline">← Back to groups</Link>
      </div>
    );
  }

  const origin = window.location.origin;

  return (
    <div className="min-h-screen bg-gray-50">
      <AppNav />

      <div className="max-w-lg mx-auto px-4 py-6 space-y-8">
        <div>
          <Link to="/groups" className="inline-block mb-4 text-sm text-blue-600 hover:text-blue-800 transition-colors">← Groups</Link>
          <h1 className="text-2xl font-bold text-gray-900">{group.name}</h1>
        </div>

        {actionError && (
          <div className="text-sm text-red-600 bg-red-50 border border-red-200 rounded p-3">
            {actionError}
          </div>
        )}

        {/* Members */}
        <section>
          <h2 className="text-sm font-semibold text-gray-500 uppercase tracking-wide mb-3">Members</h2>
          <ul className="space-y-2">
            {group.members.map((m) => (
              <li key={m.user_id} className="flex items-center gap-2 bg-white border border-gray-200 rounded-lg px-4 py-3">
                <span className="flex-1 text-sm text-gray-800 truncate">{m.email}</span>
                {m.is_admin && (
                  <span className="text-xs bg-blue-100 text-blue-700 px-2 py-0.5 rounded-full">admin</span>
                )}
                {isCurrentUserAdmin && !m.is_admin && (
                  <>
                    {promoteConfirm === m.user_id ? (
                      <span className="flex items-center gap-1">
                        <button
                          onClick={() => handleMakeAdmin(m.user_id)}
                          className="text-xs px-2 py-1 rounded bg-blue-600 text-white hover:bg-blue-700 transition-colors"
                        >
                          Confirm
                        </button>
                        <button
                          onClick={() => setPromoteConfirm(null)}
                          className="text-xs px-2 py-1 rounded bg-gray-100 text-gray-600 hover:bg-gray-200 transition-colors"
                        >
                          Cancel
                        </button>
                      </span>
                    ) : (
                      <button
                        onClick={() => setPromoteConfirm(m.user_id)}
                        className="text-xs px-2 py-1 rounded bg-gray-100 text-gray-600 hover:bg-gray-200 transition-colors"
                      >
                        Make admin
                      </button>
                    )}
                  </>
                )}
                {isCurrentUserAdmin && (
                  <button
                    disabled={deletingMember === m.user_id}
                    onClick={() => handleRemoveMember(m.user_id)}
                    className="text-xs px-2 py-1 rounded bg-gray-100 text-gray-600 hover:bg-red-100 hover:text-red-600 disabled:opacity-50 transition-colors"
                  >
                    {m.user_id === user?.userId ? "Leave" : "Remove"}
                  </button>
                )}
              </li>
            ))}
          </ul>
        </section>

        {/* Invites */}
        {isCurrentUserAdmin && (
          <section>
            <div className="flex items-center justify-between mb-3">
              <h2 className="text-sm font-semibold text-gray-500 uppercase tracking-wide">Invite links</h2>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => handleCreateInvite(1)}
                  className="text-sm px-3 py-1 bg-blue-600 hover:bg-blue-700 text-white rounded transition-colors"
                >
                  + Single-use
                </button>
                <button
                  onClick={() => setShowMultiUse((v) => !v)}
                  className={`text-sm px-3 py-1 rounded border transition-colors ${
                    showMultiUse
                      ? "bg-blue-100 text-blue-700 border-blue-300 hover:bg-blue-200"
                      : "bg-white text-blue-700 border-blue-300 hover:bg-blue-50"
                  }`}
                >
                  + Multi-use
                </button>
              </div>
            </div>
            {showMultiUse && (
              <div className="flex items-center gap-2 mb-3 p-3 bg-blue-50 border border-blue-200 rounded">
                <label className="text-sm text-gray-700">Uses:</label>
                <input
                  type="number"
                  min={2}
                  max={50}
                  value={multiUseCount}
                  onChange={(e) => setMultiUseCount(Math.min(50, Math.max(2, Number(e.target.value))))}
                  className="w-20 text-sm px-2 py-1 border border-gray-300 rounded focus:outline-none focus:ring-1 focus:ring-blue-400"
                />
                <button
                  onClick={() => handleCreateInvite(multiUseCount)}
                  className="text-sm px-3 py-1 bg-blue-600 hover:bg-blue-700 text-white rounded transition-colors"
                >
                  Create
                </button>
                <button
                  onClick={() => setShowMultiUse(false)}
                  className="text-sm px-2 py-1 text-gray-500 hover:text-gray-700 transition-colors"
                >
                  Cancel
                </button>
              </div>
            )}
            {invites.length === 0 ? (
              <p className="text-sm text-gray-400">No pending invites.</p>
            ) : (
              <ul className="space-y-2">
                {invites.map((inv) => (
                  <InviteRow
                    key={inv.id}
                    invite={inv}
                    origin={origin}
                    onRevoke={handleRevokeInvite}
                  />
                ))}
              </ul>
            )}
          </section>
        )}

        {/* Danger zone */}
        {isCurrentUserAdmin && (
          <section className="border border-red-200 rounded-lg px-6 py-5">
            <h2 className="text-sm font-semibold text-red-600 uppercase tracking-wide mb-3">Danger zone</h2>
            {deleteConfirm ? (
              <div className="flex items-center gap-3">
                <span className="text-sm text-gray-700">Delete group "{group.name}"? This cannot be undone.</span>
                <button
                  onClick={handleDeleteGroup}
                  className="text-sm px-3 py-1.5 bg-red-600 hover:bg-red-700 text-white rounded transition-colors"
                >
                  Delete
                </button>
                <button
                  onClick={() => setDeleteConfirm(false)}
                  className="text-sm px-3 py-1.5 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded transition-colors"
                >
                  Cancel
                </button>
              </div>
            ) : (
              <button
                onClick={() => setDeleteConfirm(true)}
                className="text-sm px-3 py-1.5 border border-red-300 text-red-600 hover:bg-red-50 rounded transition-colors"
              >
                Delete group
              </button>
            )}
          </section>
        )}
      </div>
    </div>
  );
}
