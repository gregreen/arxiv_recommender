import { useState, useEffect, useCallback, useRef } from "react";
import { getRecommendations } from "../api/recommendations";
import { searchPapers } from "../api/search";
import type { TimeWindow, Recommendation, SearchResponse } from "../api/types";
import PaperRow from "./PaperRow";

interface RecommendationListProps {
  selectedArxivId: string | null;
  onSelect: (arxivId: string, liked: number | null, score: number | null) => void;
  likedCache?: Record<string, number>;
}

const WINDOWS: { label: string; value: TimeWindow }[] = [
  { label: "Day", value: "day" },
  { label: "Week", value: "week" },
  { label: "Month", value: "month" },
];

export default function RecommendationList({ selectedArxivId, onSelect, likedCache = {} }: RecommendationListProps) {
  const [window, setWindow] = useState<TimeWindow>("week");
  const [results, setResults] = useState<Recommendation[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [onboarding, setOnboarding] = useState(false);

  // Search state — committedQuery only updates on submit, never on keystroke,
  // so typing does not re-render the paper list.
  const [committedQuery, setCommittedQuery] = useState("");
  const [searchResultsByWindow, setSearchResultsByWindow] = useState<SearchResponse | null>(null);
  const [isSearchActive, setIsSearchActive] = useState(false);
  const [isSearchLoading, setIsSearchLoading] = useState(false);
  const [searchError, setSearchError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const fetchRecs = useCallback(
    async (win: TimeWindow) => {
      setLoading(true);
      setError(null);
      try {
        const data = await getRecommendations(win);
        setResults(data.results);
        setOnboarding(data.onboarding ?? false);
      } catch (err: unknown) {
        setError(err instanceof Error ? err.message : "Failed to load recommendations");
        setResults([]);
        setOnboarding(false);
      } finally {
        setLoading(false);
      }
    },
    [],
  );

  // Auto-refresh paused while search is active
  useEffect(() => {
    fetchRecs(window);
    if (isSearchActive) return;
    const timer = setInterval(() => fetchRecs(window), 30_000);
    return () => clearInterval(timer);
  }, [window, fetchRecs, isSearchActive]);

  const doSearch = useCallback(async () => {
    const trimmed = inputRef.current?.value.trim() ?? "";
    if (!trimmed) return;
    setCommittedQuery(trimmed);
    setIsSearchLoading(true);
    setSearchError(null);
    try {
      const data = await searchPapers(trimmed);
      setSearchResultsByWindow(data);
      setIsSearchActive(true);
    } catch (err: unknown) {
      setSearchError(err instanceof Error ? err.message : "Search failed");
      setSearchResultsByWindow(null);
    } finally {
      setIsSearchLoading(false);
    }
  }, []);

  function clearSearch() {
    setIsSearchActive(false);
    setCommittedQuery("");
    setSearchResultsByWindow(null);
    setSearchError(null);
    if (inputRef.current) inputRef.current.value = "";
  }

  function handleWindowChange(win: TimeWindow) {
    setWindow(win);
  }

  function handleSearchKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
    if (e.key === "Enter") doSearch();
    if (e.key === "Escape") clearSearch();
  }

  const displayResults = isSearchActive ? (searchResultsByWindow?.[window] ?? []) : results;

  return (
    <div className="flex flex-col h-full">
      {/* Tabs */}
      <div className="flex gap-1 p-3 border-b border-gray-200 shrink-0">
        {WINDOWS.map((w) => (
          <button
            key={w.value}
            onClick={() => handleWindowChange(w.value)}
            className={`px-3 py-1 rounded text-sm font-medium transition-colors ${
              window === w.value
                ? "bg-blue-600 text-white"
                : "bg-gray-100 text-gray-600 hover:bg-gray-200"
            }`}
          >
            {w.label}
          </button>
        ))}
        {!isSearchActive && (
          <button
            onClick={() => fetchRecs(window)}
            disabled={loading}
            className="ml-auto text-xs text-gray-400 hover:text-gray-600 px-2"
            title="Refresh"
          >
            ↻
          </button>
        )}
      </div>

      {/* Search bar */}
      <div className="flex items-center gap-1.5 px-3 py-2 border-b border-gray-200 shrink-0">
        <div className="flex-1">
          <input
            ref={inputRef}
            type="text"
            defaultValue=""
            onKeyDown={handleSearchKeyDown}
            placeholder="Search papers…"
            className="w-full text-sm border border-gray-300 rounded px-2.5 py-1.5 focus:outline-none focus:border-blue-400 focus:ring-1 focus:ring-blue-200"
          />
        </div>
        <button
          onClick={() => doSearch()}
          disabled={isSearchLoading}
          className="shrink-0 flex items-center justify-center w-8 h-8 rounded bg-blue-600 hover:bg-blue-700 disabled:bg-gray-200 disabled:text-gray-400 text-white transition-colors"
          title="Search"
          aria-label="Search"
        >
          {isSearchLoading ? (
            <svg className="animate-spin w-4 h-4" viewBox="0 0 24 24" fill="none">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
            </svg>
          ) : (
            <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="11" cy="11" r="8" />
              <line x1="21" y1="21" x2="16.65" y2="16.65" />
            </svg>
          )}
        </button>
      </div>

      {/* Search active banner */}
      {isSearchActive && (
        <div className="flex items-center justify-between gap-2 px-3 py-2 bg-blue-50 border-b border-blue-200 shrink-0">
          <span className="text-sm text-blue-800 truncate">
            Search: &ldquo;{committedQuery}&rdquo;
          </span>
          <button
            onClick={clearSearch}
            className="shrink-0 text-sm font-medium text-blue-600 hover:text-blue-800 whitespace-nowrap"
          >
            ← Back to recommendations
          </button>
        </div>
      )}

      {/* List */}
      <div className="flex-1 overflow-y-auto p-3">
        {/* Search mode result count */}
        {isSearchActive && (searchResultsByWindow?.[window] ?? []).length > 0 && (
          <div className="text-xs text-gray-500 mb-2">
            {(searchResultsByWindow![window]).length} papers sorted by relevance to &ldquo;{committedQuery}&rdquo;
          </div>
        )}

        {/* Recommendation loading */}
        {loading && !isSearchActive && results.length === 0 && (
          <div className="text-sm text-gray-400 text-center mt-8">Loading…</div>
        )}

        {/* Recommendation error */}
        {!isSearchActive && error && (
          <div className="text-sm text-yellow-700 bg-yellow-50 border border-yellow-200 rounded p-3 mt-4">
            {error}
          </div>
        )}

        {/* Search error */}
        {isSearchActive && searchError && (
          <div className="text-sm text-yellow-700 bg-yellow-50 border border-yellow-200 rounded p-3 mt-4">
            {searchError}
          </div>
        )}

        {/* Onboarding (suppressed during search) */}
        {!isSearchActive && onboarding && (
          <div className="text-sm text-yellow-700 bg-yellow-50 border border-yellow-200 rounded p-3 mb-3">
            Not enough data to generate recommendations yet. Mark at least 4 papers as relevant or add papers to your Library.
          </div>
        )}

        {/* Empty states */}
        {!loading && !error && !onboarding && !isSearchActive && results.length === 0 && (
          <div className="text-sm text-gray-400 text-center mt-8">No recommendations yet.</div>
        )}
        {isSearchActive && !isSearchLoading && !searchError && (searchResultsByWindow?.[window] ?? []).length === 0 && (
          <div className="text-sm text-gray-400 text-center mt-8">No results found.</div>
        )}

        {/* Paper list */}
        {displayResults.map((rec) => (
          <PaperRow
            key={rec.arxiv_id}
            rec={{ ...rec, liked: likedCache[rec.arxiv_id] ?? rec.liked }}
            selected={rec.arxiv_id === selectedArxivId}
            onClick={() => onSelect(rec.arxiv_id, rec.liked, rec.score)}
          />
        ))}
      </div>
    </div>
  );
}

