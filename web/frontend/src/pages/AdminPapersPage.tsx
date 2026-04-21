import { useState, useEffect, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { getAdminPapers, type AdminPaper, type Paginated } from "../api/admin";
import { formatTimestamp } from "../utils";
import MathText from "../components/MathText";
import PaperDetail from "../components/PaperDetail";

const PAGE_SIZE = 50;

export default function AdminPapersPage() {
  const navigate = useNavigate();
  const [data, setData]         = useState<Paginated<AdminPaper> | null>(null);
  const [loading, setLoading]   = useState(true);
  const [error, setError]       = useState<string | null>(null);
  const [search, setSearch]     = useState("");
  const [query, setQuery]       = useState("");   // committed on submit
  const [offset, setOffset]     = useState(0);
  const [selectedArxivId, setSelectedArxivId] = useState<string | null>(null);
  const [selectedLiked, setSelectedLiked]     = useState<number | null>(null);

  const load = useCallback((q: string, off: number) => {
    setLoading(true);
    getAdminPapers({ q: q || undefined, limit: PAGE_SIZE, offset: off })
      .then(setData)
      .catch(() => setError("Failed to load papers."))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    load(query, offset);
  }, [load, query, offset]);

  function handleSearch(e: React.FormEvent) {
    e.preventDefault();
    setQuery(search);
    setOffset(0);
    setSelectedArxivId(null);
  }

  // Push a history entry when opening the detail panel so the browser back
  // gesture closes it instead of leaving the page.
  useEffect(() => {
    function handlePopState() {
      setSelectedArxivId(null);
      setSelectedLiked(null);
    }
    window.addEventListener("popstate", handlePopState);
    return () => window.removeEventListener("popstate", handlePopState);
  }, []);

  return (
    <div className="relative flex h-full overflow-hidden">
      {/* Left: list */}
      <div className={`absolute inset-0 w-full flex flex-col bg-white transition-transform duration-300 ease-in-out
        md:relative md:w-96 md:min-w-0 md:shrink-0 md:border-r md:border-gray-200 md:translate-x-0
        ${selectedArxivId !== null ? "-translate-x-full" : "translate-x-0"}`}>

        {/* Mobile: back to category nav */}
        <div className="md:hidden shrink-0 flex items-center px-4 py-2 bg-white border-b border-gray-200">
          <button
            onClick={() => navigate("/admin")}
            className="flex items-center gap-1.5 text-sm text-red-700 hover:text-red-900 transition-colors"
          >
            ← Return to list
          </button>
        </div>

        {/* Search bar */}
        <div className="p-4 border-b border-gray-200 shrink-0">
          <form onSubmit={handleSearch} className="flex gap-2">
            <input
              type="text"
              placeholder="Search by title or arXiv ID…"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="flex-1 border border-gray-300 rounded px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-red-400"
            />
            <button
              type="submit"
              className="bg-red-700 hover:bg-red-800 text-white text-sm font-medium rounded px-3 py-1.5 transition-colors"
            >
              Search
            </button>
          </form>
          {data && (
            <p className="text-xs text-gray-400 mt-1.5">{data.total.toLocaleString()} papers</p>
          )}
        </div>

        {/* Paper list */}
        <div className="flex-1 overflow-y-auto divide-y divide-gray-100">
          {error && <div className="p-4 text-red-600 text-sm">{error}</div>}
          {loading && !data && (
            <div className="p-4 text-gray-400 text-sm">Loading…</div>
          )}
          {data?.items.map((p) => (
            <button
              key={p.arxiv_id}
              onClick={() => { setSelectedArxivId(p.arxiv_id); setSelectedLiked(null); window.history.pushState({ detail: true }, ""); }}
              className={`w-full text-left px-4 py-3 hover:bg-gray-50 transition-colors ${
                selectedArxivId === p.arxiv_id ? "bg-red-50 border-l-2 border-red-600" : ""
              }`}
            >
              <div className="text-sm font-medium text-gray-800 line-clamp-2 leading-snug">
                <MathText text={p.title ?? p.arxiv_id} />
              </div>
              <div className="text-xs text-gray-400 mt-0.5">
                {p.arxiv_id} · {p.published_date ?? "?"} · embedded {formatTimestamp(p.embedded_at)}
              </div>
            </button>
          ))}
          {data?.items.length === 0 && (
            <div className="p-4 text-gray-400 text-sm">No papers found.</div>
          )}
        </div>

        {/* Pagination */}
        {data && data.total > PAGE_SIZE && (
          <div className="flex items-center justify-between px-4 py-2 border-t border-gray-200 shrink-0 text-sm">
            <button
              onClick={() => setOffset((o) => Math.max(0, o - PAGE_SIZE))}
              disabled={offset === 0}
              className="px-3 py-1 rounded border border-gray-300 text-gray-600 hover:bg-gray-50 disabled:opacity-40 transition-colors"
            >
              ←
            </button>
            <span className="text-gray-400 text-xs">
              {offset + 1}–{Math.min(offset + PAGE_SIZE, data.total)} / {data.total.toLocaleString()}
            </span>
            <button
              onClick={() => setOffset((o) => o + PAGE_SIZE)}
              disabled={offset + PAGE_SIZE >= data.total}
              className="px-3 py-1 rounded border border-gray-300 text-gray-600 hover:bg-gray-50 disabled:opacity-40 transition-colors"
            >
              →
            </button>
          </div>
        )}
      </div>

      {/* Right: paper detail */}
      <div className={`absolute inset-0 w-full flex flex-col transition-transform duration-300 ease-in-out
        md:relative md:flex-1 md:min-w-0 md:translate-x-0
        ${selectedArxivId !== null ? "translate-x-0" : "translate-x-full"}`}>
        {/* Back button — mobile only */}
        <div className="md:hidden shrink-0 flex items-center px-4 py-2 bg-white border-b border-gray-200">
          <button
            onClick={() => window.history.back()}
            className="flex items-center gap-1.5 text-sm text-red-700 hover:text-red-900 transition-colors"
          >
            ← Return to list
          </button>
        </div>
        <div className="flex-1 overflow-hidden">
          <PaperDetail
            arxivId={selectedArxivId}
            initialLiked={selectedLiked}
            score={null}
            onLikedChange={(_, liked) => setSelectedLiked(liked)}
          />
        </div>
      </div>
    </div>
  );
}
