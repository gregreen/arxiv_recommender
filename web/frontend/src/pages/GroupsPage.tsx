import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useGroups } from "../contexts/GroupsContext";
import { createGroup } from "../api/groups";

export default function GroupsPage() {
  const navigate = useNavigate();
  const { groups, isLoading, refetch } = useGroups();
  const [name, setName] = useState("");
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleCreate(e: React.FormEvent) {
    e.preventDefault();
    const trimmed = name.trim();
    if (!trimmed) return;
    setCreating(true);
    setError(null);
    try {
      const group = await createGroup(trimmed);
      await refetch();
      navigate(`/groups/${group.id}/manage`);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to create group");
      setCreating(false);
    }
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <nav className="flex items-center gap-4 px-4 py-2 border-b border-blue-200 shrink-0" style={{ background: "linear-gradient(42deg, #ebf5ff, #91caff)" }}>
        <Link to="/" className="font-bold text-blue-700 text-lg">arXiv Recommender</Link>
        <Link to="/" className="text-sm text-gray-600 hover:text-gray-900">← Recommendations</Link>
      </nav>

      <div className="max-w-lg mx-auto px-4 py-10">
        <h1 className="text-2xl font-bold text-gray-900 mb-6">Groups</h1>

        {isLoading && <p className="text-sm text-gray-400">Loading…</p>}

        {!isLoading && groups.length > 0 && (
          <div className="mb-8">
            <h2 className="text-sm font-semibold text-gray-500 uppercase tracking-wide mb-3">Your groups</h2>
            <ul className="space-y-2">
              {groups.map((g) => (
                <li key={g.id} className="flex items-center justify-between bg-white border border-gray-200 rounded-lg px-4 py-3">
                  <span className="font-medium text-gray-800">{g.name}</span>
                  <div className="flex items-center gap-3">
                    {g.is_admin && (
                      <Link
                        to={`/groups/${g.id}/manage`}
                        className="text-sm text-blue-600 hover:text-blue-800 transition-colors"
                      >
                        Manage
                      </Link>
                    )}
                  </div>
                </li>
              ))}
            </ul>
          </div>
        )}

        <div className="bg-white border border-gray-200 rounded-lg px-6 py-5">
          <h2 className="text-base font-semibold text-gray-800 mb-4">Create a new group</h2>
          <form onSubmit={handleCreate} className="flex flex-col gap-3">
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="Group name"
              maxLength={80}
              className="border border-gray-300 rounded px-3 py-2 text-sm focus:outline-none focus:border-blue-400 focus:ring-1 focus:ring-blue-200"
            />
            {error && (
              <p className="text-sm text-red-600">{error}</p>
            )}
            <button
              type="submit"
              disabled={creating || !name.trim()}
              className="self-start px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-200 disabled:text-gray-400 text-white text-sm font-medium rounded transition-colors"
            >
              {creating ? "Creating…" : "Create group"}
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}
