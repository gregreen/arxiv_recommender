import { useState, useEffect, type FormEvent } from "react";
import { Link } from "react-router-dom";
import { logout } from "../api/auth";
import { useAuth } from "../AuthContext";
import { getMyPapers, addPaper, updatePaper, deletePaper, importAds } from "../api/user";
import type { UserPaper } from "../api/types";
import { formatTimestamp } from "../utils";
import PaperDetail from "../components/PaperDetail";

export default function LibraryPage() {
  const { user, clearUser } = useAuth();
  const [papers, setPapers] = useState<UserPaper[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Detail panel
  const [selectedArxivId, setSelectedArxivId] = useState<string | null>(null);
  const [selectedLiked, setSelectedLiked] = useState<number | null>(null);
  const [likedCache, setLikedCache] = useState<Record<string, number>>({});

  function handleSelect(arxivId: string, liked: number) {
    setSelectedArxivId(arxivId);
    setSelectedLiked(likedCache[arxivId] ?? liked);
  }

  function handleLikedChange(arxivId: string, liked: 1 | -1 | 0) {
    setSelectedLiked(liked);
    setLikedCache((prev) => ({ ...prev, [arxivId]: liked }));
    if (liked === 0) {
      // Neutral/removed — drop from the library list entirely.
      setPapers((prev) => prev.filter((p) => p.arxiv_id !== arxivId));
      setSelectedArxivId(null);
    } else {
      setPapers((prev) => prev.map((p) => p.arxiv_id === arxivId ? { ...p, liked } : p));
    }
  }

  // Add paper form
  const [addArxivId, setAddArxivId] = useState("");
  const [addLiked, setAddLiked] = useState<1 | -1>(1);
  const [addError, setAddError] = useState<string | null>(null);
  const [addLoading, setAddLoading] = useState(false);

  // ADS import form
  const [adsText, setAdsText] = useState("");
  const [adsResult, setAdsResult] = useState<string | null>(null);
  const [adsError, setAdsError] = useState<string | null>(null);
  const [adsLoading, setAdsLoading] = useState(false);

  async function handleLogout() {
    await logout().catch(() => {});
    clearUser();
  }

  useEffect(() => {
    getMyPapers()
      .then(setPapers)
      .catch(() => setError("Failed to load library"))
      .finally(() => setLoading(false));
  }, []);

  async function handleAdd(e: FormEvent) {
    e.preventDefault();
    setAddError(null);
    setAddLoading(true);
    try {
      await addPaper(addArxivId.trim(), addLiked);
      setAddArxivId("");
      // Refresh full list (backend may have fetched metadata)
      const updated = await getMyPapers();
      setPapers(updated);
    } catch (err: unknown) {
      setAddError(err instanceof Error ? err.message : "Failed to add paper");
    } finally {
      setAddLoading(false);
    }
  }

  async function handleToggleLiked(arxivId: string, current: number) {
    const next = current === 1 ? -1 : 1;
    try {
      await updatePaper(arxivId, next as 1 | -1);
      setPapers((prev) =>
        prev.map((p) => (p.arxiv_id === arxivId ? { ...p, liked: next } : p))
      );
      setLikedCache((prev) => ({ ...prev, [arxivId]: next }));
      if (selectedArxivId === arxivId) setSelectedLiked(next);
    } catch {
      // ignore
    }
  }

  async function handleDelete(arxivId: string) {
    try {
      await deletePaper(arxivId);
      setPapers((prev) => prev.filter((p) => p.arxiv_id !== arxivId));
    } catch {
      // ignore
    }
  }

  async function handleImport(e: FormEvent) {
    e.preventDefault();
    setAdsError(null);
    setAdsResult(null);
    setAdsLoading(true);
    try {
      const result = await importAds(adsText.trim());
      setAdsResult(`Imported ${result.imported} paper(s), skipped ${result.skipped}.`);
      setAdsText("");
      // Refresh list
      const updated = await getMyPapers();
      setPapers(updated);
    } catch (err: unknown) {
      setAdsError(err instanceof Error ? err.message : "Import failed");
    } finally {
      setAdsLoading(false);
    }
  }

  return (
    <div className="flex flex-col h-screen bg-gray-50">
      {/* Navbar */}
      <nav className="flex items-center gap-4 px-4 py-2 bg-white border-b border-gray-200 shrink-0">
        <Link to="/" className="font-bold text-blue-700 text-lg">arXiv Recommender</Link>
        <span className="text-sm text-gray-600 font-medium">Library</span>
        <div className="ml-auto flex items-center gap-3">
          <span className="text-sm text-gray-500">{user?.email}</span>
          <button
            onClick={handleLogout}
            className="text-sm text-gray-500 hover:text-red-600 transition-colors"
          >
            Sign out
          </button>
        </div>
      </nav>

      <div className="flex flex-1 overflow-hidden">
        {/* Left: library management */}
        <div className="w-96 shrink-0 border-r border-gray-200 overflow-y-auto p-6">
        {/* Add paper */}
        <section className="bg-white rounded-lg shadow-sm border border-gray-200 p-4 mb-6">
          <h2 className="font-semibold text-gray-800 mb-3">Add Paper</h2>
          {addError && (
            <div className="mb-3 text-sm text-red-600 bg-red-50 border border-red-200 rounded p-2">
              {addError}
            </div>
          )}
          <form onSubmit={handleAdd} className="flex gap-2 flex-wrap">
            <input
              type="text"
              placeholder="arXiv ID (e.g. 2401.12345)"
              value={addArxivId}
              onChange={(e) => setAddArxivId(e.target.value)}
              required
              className="border border-gray-300 rounded px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 flex-1 min-w-48"
            />
            <select
              value={addLiked}
              onChange={(e) => setAddLiked(Number(e.target.value) as 1 | -1)}
              className="border border-gray-300 rounded px-2 py-1.5 text-sm focus:outline-none"
            >
              <option value={1}>Liked</option>
              <option value={-1}>Disliked</option>
            </select>
            <button
              type="submit"
              disabled={addLoading}
              className="bg-blue-600 hover:bg-blue-700 disabled:opacity-50 text-white text-sm font-medium rounded px-4 py-1.5 transition-colors"
            >
              {addLoading ? "Adding…" : "Add"}
            </button>
          </form>
        </section>

        {/* NASA ADS import */}
        <section className="bg-white rounded-lg shadow-sm border border-gray-200 p-4 mb-6">
          <h2 className="font-semibold text-gray-800 mb-3">Import from NASA ADS</h2>
          <p className="text-xs text-gray-500 mb-2">
            Paste an ADS bibliography export (BibTeX or plain list with arXiv IDs).
          </p>
          {adsError && (
            <div className="mb-3 text-sm text-red-600 bg-red-50 border border-red-200 rounded p-2">
              {adsError}
            </div>
          )}
          {adsResult && (
            <div className="mb-3 text-sm text-green-700 bg-green-50 border border-green-200 rounded p-2">
              {adsResult}
            </div>
          )}
          <form onSubmit={handleImport} className="flex flex-col gap-2">
            <textarea
              value={adsText}
              onChange={(e) => setAdsText(e.target.value)}
              rows={5}
              placeholder="Paste ADS export text here…"
              required
              className="border border-gray-300 rounded px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 resize-y"
            />
            <div>
              <button
                type="submit"
                disabled={adsLoading}
                className="bg-blue-600 hover:bg-blue-700 disabled:opacity-50 text-white text-sm font-medium rounded px-4 py-1.5 transition-colors"
              >
                {adsLoading ? "Importing…" : "Import"}
              </button>
            </div>
          </form>
        </section>

        {/* Paper list */}
        <section>
          <h2 className="font-semibold text-gray-800 mb-3">
            My Papers ({papers.length})
          </h2>
          {loading && <div className="text-sm text-gray-400">Loading…</div>}
          {error && <div className="text-sm text-red-500">{error}</div>}
          {!loading && papers.length === 0 && (
            <div className="text-sm text-gray-400">No papers in your library yet.</div>
          )}
          <div className="space-y-1.5">
            {papers.map((p) => (
              <div
                key={p.arxiv_id}
                onClick={() => handleSelect(p.arxiv_id, p.liked)}
                className={`flex items-start gap-3 border rounded p-3 bg-white cursor-pointer hover:border-blue-300 transition-colors ${
                  selectedArxivId === p.arxiv_id
                    ? "border-blue-400 bg-blue-50"
                    : p.liked === 1 ? "border-green-200" : "border-red-200"
                }`}
              >
                <div className="flex-1 min-w-0">
                  <div className="text-sm font-medium text-gray-800 truncate">
                    {p.title ?? p.arxiv_id}
                  </div>
                  <div className="text-xs text-gray-400 mt-0.5">
                    {p.arxiv_id} · {formatTimestamp(p.published_date)}
                  </div>
                </div>
                <div className="flex gap-1.5 shrink-0">
                  <button
                    onClick={(e) => { e.stopPropagation(); handleToggleLiked(p.arxiv_id, p.liked); }}
                    title={p.liked === 1 ? "Mark as disliked" : "Mark as liked"}
                    className={`text-xs px-2 py-1 rounded transition-colors ${
                      p.liked === 1
                        ? "bg-green-100 hover:bg-green-200 text-green-700"
                        : "bg-red-100 hover:bg-red-200 text-red-700"
                    }`}
                  >
                    {p.liked === 1 ? "👍" : "👎"}
                  </button>
                  <button
                    onClick={() => handleDelete(p.arxiv_id)}
                    title="Remove from library"
                    className="text-xs px-2 py-1 rounded bg-gray-100 hover:bg-red-100 text-gray-500 hover:text-red-600 transition-colors"
                  >
                    ✕
                  </button>
                </div>
              </div>
            ))}
          </div>
        </section>
        </div>

        {/* Right: paper detail */}
        <div className="flex-1 overflow-hidden">
          <PaperDetail
            arxivId={selectedArxivId}
            initialLiked={selectedLiked}
            onLikedChange={handleLikedChange}
          />
        </div>
      </div>
    </div>
  );
}
