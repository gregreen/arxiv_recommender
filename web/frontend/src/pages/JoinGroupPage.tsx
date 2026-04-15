import { useState, useEffect } from "react";
import { useSearchParams, useNavigate, Link } from "react-router-dom";
import { useAuth } from "../AuthContext";
import { useGroups } from "../contexts/GroupsContext";
import { getJoinInfo, joinGroup, type JoinInfo } from "../api/groups";

export default function JoinGroupPage() {
  const [searchParams] = useSearchParams();
  const token = searchParams.get("token") ?? "";
  const navigate = useNavigate();
  const { user, isLoading: authLoading } = useAuth();
  const { refetch } = useGroups();

  const [info, setInfo] = useState<JoinInfo | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [joining, setJoining] = useState(false);

  useEffect(() => {
    if (authLoading) return;
    if (!token) {
      setError("No invite token provided.");
      setLoading(false);
      return;
    }
    // Redirect to login if not authenticated, preserving the invite link
    if (!user) {
      const next = encodeURIComponent(`/join-group?token=${encodeURIComponent(token)}`);
      navigate(`/login?next=${next}`, { replace: true });
      return;
    }
    getJoinInfo(token)
      .then((data) => setInfo(data))
      .catch(() => setError("Invite not found, already used, or expired."))
      .finally(() => setLoading(false));
  }, [token, user, authLoading, navigate]);

  async function handleJoin() {
    setJoining(true);
    setError(null);
    try {
      await joinGroup(token);
      await refetch();
      navigate("/", { replace: true });
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to join group");
      setJoining(false);
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen text-gray-400">Loading…</div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col items-center justify-center px-4">
      <div className="w-full max-w-sm bg-white border border-gray-200 rounded-lg px-8 py-10 shadow-sm">
        <h1 className="text-xl font-bold text-gray-900 mb-6 text-center">Group invitation</h1>

        {error ? (
          <>
            <p className="text-sm text-red-600 mb-6 text-center">{error}</p>
            <Link to="/" className="block text-center text-sm text-blue-600 hover:underline">
              Go to recommendations →
            </Link>
          </>
        ) : info ? (
          <>
            <p className="text-sm text-gray-600 text-center mb-6">
              You've been invited to join the group <strong>{info.group_name}</strong>.
            </p>
            <button
              onClick={handleJoin}
              disabled={joining}
              className="w-full py-2.5 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-200 disabled:text-gray-400 text-white font-medium rounded transition-colors"
            >
              {joining ? "Joining…" : `Join "${info.group_name}"`}
            </button>
          </>
        ) : null}
      </div>
    </div>
  );
}
