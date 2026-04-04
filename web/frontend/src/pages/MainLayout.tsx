import { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import { logout } from "../api/auth";
import { useAuth } from "../AuthContext";
import { getMyPapers } from "../api/user";
import NavMenu from "../components/NavMenu";
import RecommendationList from "../components/RecommendationList";
import PaperDetail from "../components/PaperDetail";

export default function MainLayout() {
  const { user, clearUser } = useAuth();
  const [selectedArxivId, setSelectedArxivId] = useState<string | null>(null);
  const [selectedLiked, setSelectedLiked] = useState<number | null>(null);
  const [selectedScore, setSelectedScore] = useState<number | null>(null);
  // Liked map: initialised from the server on mount so previously-liked papers
  // show the correct colour immediately, then updated on every user interaction.
  const [likedCache, setLikedCache] = useState<Record<string, number>>({});

  useEffect(() => {
    getMyPapers()
      .then((papers) => {
        const map: Record<string, number> = {};
        for (const p of papers) map[p.arxiv_id] = p.liked;
        setLikedCache(map);
      })
      .catch(() => {}); // non-fatal; cache stays empty
  }, []);

  async function handleLogout() {
    await logout().catch(() => {});
    clearUser();
  }

  function handleSelect(arxivId: string, liked: number | null, score: number | null) {
    setSelectedArxivId(arxivId);
    setSelectedScore(score);
    // Prefer the cached (user-updated) value over the stale list value.
    setSelectedLiked(likedCache[arxivId] ?? liked);
  }

  function handleLikedChange(arxivId: string, liked: 1 | -1 | 0) {
    setSelectedLiked(liked);
    setLikedCache((prev) => ({ ...prev, [arxivId]: liked }));
  }

  function handleClearSelection() {
    setSelectedArxivId(null);
    setSelectedLiked(null);
    setSelectedScore(null);
  }

  return (
    <div className="flex flex-col h-screen overflow-x-hidden bg-gray-50">
      {/* Navbar */}
      <nav className="flex items-center gap-4 px-4 py-2 bg-white border-b border-gray-200 shrink-0">
        <span className="font-bold text-blue-700 text-lg">arXiv Recommender</span>
        <Link
          to="/library"
          className="text-sm text-gray-600 hover:text-gray-900"
        >
          Library
        </Link>
        <NavMenu email={user?.email} onLogout={handleLogout} />
      </nav>

      {/* Two-pane body */}
      <div className="relative flex flex-1 overflow-hidden">
        {/* Left: recommendation list */}
        <div className={`absolute inset-0 w-full flex flex-col bg-white transition-transform duration-300 ease-in-out
          md:relative md:w-96 md:min-w-0 md:shrink-0 md:border-r md:border-gray-200 md:translate-x-0
          ${selectedArxivId !== null ? "-translate-x-full" : "translate-x-0"}`}>
          <RecommendationList
            selectedArxivId={selectedArxivId}
            onSelect={handleSelect}
            likedCache={likedCache}
          />
        </div>

        {/* Right: paper detail */}
        <div className={`absolute inset-0 w-full flex flex-col transition-transform duration-300 ease-in-out
          md:relative md:flex-1 md:min-w-0 md:translate-x-0
          ${selectedArxivId !== null ? "translate-x-0" : "translate-x-full"}`}>
          {/* Back button — mobile only */}
          <div className="md:hidden shrink-0 flex items-center px-4 py-2 bg-white border-b border-gray-200">
            <button
              onClick={handleClearSelection}
              className="flex items-center gap-1.5 text-sm text-blue-600 hover:text-blue-800 transition-colors"
            >
              ← Return to list
            </button>
          </div>
          <div className="flex-1 overflow-hidden">
            <PaperDetail
              arxivId={selectedArxivId}
              initialLiked={selectedLiked}
              score={selectedScore}
              onLikedChange={handleLikedChange}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
