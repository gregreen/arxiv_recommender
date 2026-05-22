import { useState, useEffect, useMemo, type FormEvent } from "react";
import { getMyPapers, addPaper, updatePaper, deletePaper, importAds } from "../api/user";
import type { UserPaper } from "../api/types";
import type { Paper } from "../api/types";
import MathText from "../components/MathText";
import AppNav from "../components/AppNav";
import PaperDetail from "../components/PaperDetail";

export default function LibraryPage() {
  const [papers, setPapers] = useState<UserPaper[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Detail panel
  const [selectedArxivId, setSelectedArxivId] = useState<string | null>(null);
  const [selectedLiked, setSelectedLiked] = useState<number | null>(null);
  const [likedCache, setLikedCache] = useState<Record<string, number>>({});

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

  function handleSelect(arxivId: string, liked: number) {
    setSelectedArxivId(arxivId);
    setSelectedLiked(likedCache[arxivId] ?? liked);
    window.history.pushState({ detail: true }, "");
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

  function handlePaperLoaded(paper: Paper) {
    // Update sidebar entry when it was previously missing metadata (title === null).
    setPapers((prev) => prev.map((p) =>
      p.arxiv_id === paper.arxiv_id && p.title === null
        ? { ...p, title: paper.title, published_date: paper.published_date }
        : p
    ));
  }

  // Add paper form
  const [addArxivId, setAddArxivId] = useState("");
  const [addLiked, setAddLiked] = useState<1 | -1>(1);
  const [addError, setAddError] = useState<string | null>(null);
  const [addLoading, setAddLoading] = useState(false);

  // ADS import form
  const [adsText, setAdsText] = useState("");
  const [adsResult, setAdsResult] = useState<string | null>(null);
  const [adsHasWarning, setAdsHasWarning] = useState(false);
  const [adsError, setAdsError] = useState<string | null>(null);
  const [adsLoading, setAdsLoading] = useState(false);

  // Import accordion
  const [importOpen, setImportOpen] = useState(false);

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
      let msg = `Imported ${result.imported} paper(s), skipped ${result.skipped}.`;
      let hasWarning = result.rate_limited > 0;
      if (result.rate_limited > 0) {
        msg += ` ${result.rate_limited} paper(s) were not imported due to the daily import limit.`;
      }
      if (result.invalid > 0) {
        msg += ` ${result.invalid} ID(s) were invalid and ignored.`;
        hasWarning = true;
      }
      setAdsResult(msg);
      setAdsHasWarning(hasWarning);
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

  // My Papers: filter, sort, search, pagination
  const PAGE_SIZE = 50;
  const [filterLiked, setFilterLiked] = useState<"all" | "relevant" | "not_relevant">("all");
  const [sortBy, setSortBy] = useState<"last_added" | "first_added" | "newest" | "oldest">("last_added");
  const [searchInput, setSearchInput] = useState("");
  const [searchQuery, setSearchQuery] = useState("");
  const [page, setPage] = useState(0);

  const filteredPapers = useMemo(() => {
    let result = papers;
    if (filterLiked === "relevant") result = result.filter((p) => p.liked === 1);
    else if (filterLiked === "not_relevant") result = result.filter((p) => p.liked === -1);
    if (searchQuery.trim()) {
      const q = searchQuery.trim().toLowerCase();
      result = result.filter((p) => p.arxiv_id.toLowerCase().includes(q));
    }
    return [...result].sort((a, b) => {
      switch (sortBy) {
        case "last_added":  return (b.added_at ?? "").localeCompare(a.added_at ?? "");
        case "first_added": return (a.added_at ?? "").localeCompare(b.added_at ?? "");
        case "newest":      return (b.published_date ?? "").localeCompare(a.published_date ?? "");
        case "oldest":      return (a.published_date ?? "").localeCompare(b.published_date ?? "");
        default:            return 0;
      }
    });
  }, [papers, filterLiked, sortBy, searchQuery]);

  const pagePapers = filteredPapers.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);

  function handleSearchSubmit(e: FormEvent) {
    e.preventDefault();
    setSearchQuery(searchInput);
    setPage(0);
  }

  return (
    <div className="flex flex-col h-screen overflow-x-hidden bg-gray-50">
      <AppNav />

      <div className="relative flex flex-1 overflow-hidden">
        {/* Left: library management */}
        <div className={`absolute inset-0 w-full overflow-y-auto flex flex-col bg-white transition-transform duration-300 ease-in-out
          md:relative md:w-96 md:min-w-0 md:shrink-0 md:border-r md:border-gray-200 md:translate-x-0
          ${selectedArxivId !== null ? "-translate-x-full" : "translate-x-0"}`}>
        <div className="shrink-0 p-6 pb-0">
        {/* Import papers accordion */}
        <section className="bg-white rounded-lg shadow-sm border border-gray-200">
          <button
            type="button"
            onClick={() => setImportOpen((o) => !o)}
            className="w-full flex items-center justify-between px-4 py-3 text-left"
          >
            <span className="font-semibold text-gray-800">Import Papers</span>
            <svg
              className={`w-4 h-4 text-gray-500 transition-transform duration-200 ${importOpen ? "rotate-180" : ""}`}
              viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"
            >
              <polyline points="6 9 12 15 18 9" />
            </svg>
          </button>

          {importOpen && (
            <div className="px-4 pb-4 space-y-4 border-t border-gray-200">
              {/* Add by arXiv ID */}
              <div className="pt-3">
                <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">Add by arXiv ID</h3>
                {addError && (
                  <div className="mb-2 text-sm text-red-600 bg-red-50 border border-red-200 rounded p-2">
                    {addError}
                  </div>
                )}
                <form onSubmit={handleAdd} className="flex gap-2">
                  <input
                    type="text"
                    placeholder="e.g. 2401.12345"
                    value={addArxivId}
                    onChange={(e) => setAddArxivId(e.target.value)}
                    required
                    className="border border-gray-300 rounded px-3 py-1.5 text-xs focus:outline-none focus:ring-2 focus:ring-blue-500 flex-1 min-w-0"
                  />
                  <div className="flex gap-1 shrink-0">
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
                      className="bg-blue-600 hover:bg-blue-700 disabled:opacity-50 text-white rounded w-8 h-8 flex items-center justify-center shrink-0 transition-colors"
                    >
                      {addLoading ? (
                        <svg className="w-4 h-4 animate-spin" viewBox="0 0 24 24" fill="none">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
                        </svg>
                      ) : (
                        <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
                          <line x1="12" y1="5" x2="12" y2="19" />
                          <line x1="5" y1="12" x2="19" y2="12" />
                        </svg>
                      )}
                    </button>
                  </div>
                </form>
              </div>

              <hr className="border-gray-200" />

              {/* Import from NASA ADS */}
              <div>
                <div className="flex items-center justify-between mb-2">
                  <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wide">Import from NASA ADS</h3>
                  <button
                    type="submit"
                    form="ads-import-form"
                    disabled={adsLoading}
                    className="bg-blue-600 hover:bg-blue-700 disabled:opacity-50 text-white text-xs font-medium rounded px-3 py-1 transition-colors"
                  >
                    {adsLoading ? "Importing…" : "Import"}
                  </button>
                </div>
                <p className="text-xs text-gray-500 mb-2">
                  Export an ADS library using the Custom %X format, and paste the contents below.
                </p>
                {adsError && (
                  <div className="mb-2 text-sm text-red-600 bg-red-50 border border-red-200 rounded p-2">
                    {adsError}
                  </div>
                )}
                {adsResult && (
                  <div className={`mb-2 text-sm rounded p-2 border ${
                    adsHasWarning
                      ? "text-amber-700 bg-amber-50 border-amber-200"
                      : "text-green-700 bg-green-50 border-green-200"
                  }`}>
                    {adsResult}
                  </div>
                )}
                <form id="ads-import-form" onSubmit={handleImport}>
                  <textarea
                    value={adsText}
                    onChange={(e) => setAdsText(e.target.value)}
                    rows={5}
                    placeholder={"arXiv:1234.56789\narXiv:0123.45678\n..."}
                    required
                    className="w-full border border-gray-300 rounded px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 resize-y"
                  />
                </form>
              </div>
            </div>
          )}
        </section>
        </div>{/* end top sections */}

        {/* Paper list */}
        <section className="flex-1 flex flex-col min-h-0 px-6 pb-6 pt-3">
          {/* Header + controls */}
          <div className="shrink-0">
            <h2 className="font-semibold text-gray-800 mb-2">
              My Papers ({papers.length})
            </h2>
            <div className="flex gap-1 mb-2 items-center">
              <select
                value={filterLiked}
                onChange={(e) => { setFilterLiked(e.target.value as "all" | "relevant" | "not_relevant"); setPage(0); }}
                className="border border-gray-300 rounded px-1.5 py-1 text-xs focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="all">All</option>
                <option value="relevant">Relevant</option>
                <option value="not_relevant">Not Relevant</option>
              </select>
              <select
                value={sortBy}
                onChange={(e) => { setSortBy(e.target.value as "last_added" | "first_added" | "newest" | "oldest"); setPage(0); }}
                className="border border-gray-300 rounded px-1.5 py-1 text-xs focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="last_added">Last Added</option>
                <option value="first_added">First Added</option>
                <option value="newest">Newest</option>
                <option value="oldest">Oldest</option>
              </select>
              <form onSubmit={handleSearchSubmit} className="flex gap-1 flex-1 min-w-0">
                <input
                  type="text"
                  placeholder="arXiv ID…"
                  value={searchInput}
                  onChange={(e) => setSearchInput(e.target.value)}
                  className="flex-1 min-w-0 border border-gray-300 rounded px-2 py-1 text-xs focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
                <button
                  type="submit"
                  title="Search"
                  aria-label="Search"
                  className="shrink-0 flex items-center justify-center w-6 h-6 rounded bg-blue-600 hover:bg-blue-700 text-white transition-colors"
                >
                  <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                    <circle cx="11" cy="11" r="8" />
                    <line x1="21" y1="21" x2="16.65" y2="16.65" />
                  </svg>
                </button>
              </form>
            </div>
          </div>

          {/* Status messages */}
          {loading && <div className="text-sm text-gray-400 shrink-0">Loading…</div>}
          {error && <div className="text-sm text-red-500 shrink-0">{error}</div>}
          {!loading && papers.length === 0 && (
            <div className="text-sm text-gray-400 shrink-0">No papers in your library yet.</div>
          )}

          {/* Return to list when search is active */}
          {searchQuery.trim() && (
            <div className="flex items-center justify-between gap-2 -mx-6 px-6 py-2 bg-blue-50 border-b border-blue-200 shrink-0 mb-1">
              <span className="text-sm text-blue-800 truncate">
                Search: &ldquo;{searchQuery}&rdquo;
              </span>
              <button
                onClick={() => { setSearchQuery(""); setSearchInput(""); setPage(0); }}
                className="shrink-0 text-sm font-medium text-blue-600 hover:text-blue-800 whitespace-nowrap"
              >
                ← Return to list
              </button>
            </div>
          )}
          {!loading && papers.length > 0 && filteredPapers.length === 0 && (
            <div className="text-sm text-gray-400 shrink-0">No papers match.</div>
          )}

          {/* Scrollable paper list */}
          <div className="flex-1 overflow-y-auto min-h-32 space-y-1.5">
            {pagePapers.map((p) => (
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
                  <div className="text-sm font-medium text-gray-800 line-clamp-2">
                    <MathText text={p.title ?? p.arxiv_id} />
                  </div>
                  <div className="text-xs text-gray-400 mt-0.5">
                    {p.arxiv_id} · {p.published_date?.slice(0, 10) ?? "?"}
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
                    onClick={(e) => { e.stopPropagation(); handleDelete(p.arxiv_id); }}
                    title="Remove from library"
                    className="text-xs px-2 py-1 rounded bg-gray-100 hover:bg-red-100 text-gray-500 hover:text-red-600 transition-colors"
                  >
                    ✕
                  </button>
                </div>
              </div>
            ))}
          </div>

          {/* Pagination footer */}
          {filteredPapers.length > PAGE_SIZE && (
            <div className="shrink-0 flex items-center justify-between pt-2 mt-1 border-t border-gray-100">
              <span className="text-xs text-gray-400">
                {page * PAGE_SIZE + 1}–{Math.min((page + 1) * PAGE_SIZE, filteredPapers.length)} / {filteredPapers.length}
              </span>
              <div className="flex gap-1">
                <button
                  onClick={() => setPage((p) => p - 1)}
                  disabled={page === 0}
                  className="text-xs px-2 py-1 rounded border border-gray-300 disabled:opacity-40 hover:bg-gray-50 transition-colors"
                >
                  ←
                </button>
                <button
                  onClick={() => setPage((p) => p + 1)}
                  disabled={(page + 1) * PAGE_SIZE >= filteredPapers.length}
                  className="text-xs px-2 py-1 rounded border border-gray-300 disabled:opacity-40 hover:bg-gray-50 transition-colors"
                >
                  →
                </button>
              </div>
            </div>
          )}
        </section>
        </div>

        {/* Right: paper detail */}
        <div className={`absolute inset-0 w-full flex flex-col transition-transform duration-300 ease-in-out
          md:relative md:flex-1 md:min-w-0 md:translate-x-0
          ${selectedArxivId !== null ? "translate-x-0" : "translate-x-full"}`}>
          {/* Back button — mobile only */}
          <div className="md:hidden shrink-0 flex items-center px-4 py-2 bg-white border-b border-gray-200">
            <button
              onClick={() => window.history.back()}
              className="flex items-center gap-1.5 text-sm text-blue-600 hover:text-blue-800 transition-colors"
            >
              ← Return to list
            </button>
          </div>
          <div className="flex-1 overflow-hidden">
            <PaperDetail
              arxivId={selectedArxivId}
              initialLiked={selectedLiked}
              onLikedChange={handleLikedChange}
              onPaperLoaded={handlePaperLoaded}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
