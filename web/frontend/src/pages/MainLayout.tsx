import { useState } from "react";
import { Link } from "react-router-dom";
import { logout } from "../api/auth";
import { useAuth } from "../AuthContext";
import RecommendationList from "../components/RecommendationList";
import PaperDetail from "../components/PaperDetail";

export default function MainLayout() {
  const { user, clearUser } = useAuth();
  const [selectedArxivId, setSelectedArxivId] = useState<string | null>(null);
  const [selectedLiked, setSelectedLiked] = useState<number | null>(null);
  const [selectedScore, setSelectedScore] = useState<number | null>(null);
  // Track liked values that the user has changed this session so re-selecting
  // a paper doesn't reset the button back to its original state.
  const [likedCache, setLikedCache] = useState<Record<string, number>>({});

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

  return (
    <div className="flex flex-col h-screen bg-gray-50">
      {/* Navbar */}
      <nav className="flex items-center gap-4 px-4 py-2 bg-white border-b border-gray-200 shrink-0">
        <span className="font-bold text-blue-700 text-lg">arXiv Recommender</span>
        <Link
          to="/library"
          className="text-sm text-gray-600 hover:text-gray-900"
        >
          Library
        </Link>
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

      {/* Two-pane body */}
      <div className="flex flex-1 overflow-hidden">
        {/* Left: recommendation list */}
        <div className="w-96 shrink-0 border-r border-gray-200 bg-white overflow-hidden flex flex-col">
          <RecommendationList
            selectedArxivId={selectedArxivId}
            onSelect={handleSelect}
            likedCache={likedCache}
          />
        </div>

        {/* Right: paper detail */}
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
  );
}
