import { useState, useEffect } from "react";
import { getMyPapers } from "../api/user";
import { useGroups } from "../contexts/GroupsContext";
import AppNav from "../components/AppNav";
import RecommendationList from "../components/RecommendationList";
import PaperDetail from "../components/PaperDetail";

export default function MainLayout() {
  const { groups } = useGroups();
  const [selectedArxivId, setSelectedArxivId] = useState<string | null>(null);
  const [selectedLiked, setSelectedLiked] = useState<number | null>(null);
  const [selectedScore, setSelectedScore] = useState<number | null>(null);
  const [activeGroupId, setActiveGroupId] = useState<number | null>(null);
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

  // Reset activeGroupId if that group no longer exists
  useEffect(() => {
    if (activeGroupId !== null && !groups.some((g) => g.id === activeGroupId)) {
      setActiveGroupId(null);
    }
  }, [groups, activeGroupId]);

  // Push a history entry when opening the detail panel so the browser back
  // gesture closes it instead of leaving the page.
  useEffect(() => {
    function handlePopState() {
      setSelectedArxivId(null);
      setSelectedLiked(null);
      setSelectedScore(null);
    }
    window.addEventListener("popstate", handlePopState);
    return () => window.removeEventListener("popstate", handlePopState);
  }, []);

  function handleSelect(arxivId: string, liked: number | null, score: number | null) {
    setSelectedArxivId(arxivId);
    setSelectedScore(score);
    // Prefer the cached (user-updated) value over the stale list value.
    setSelectedLiked(likedCache[arxivId] ?? liked);
    window.history.pushState({ detail: true }, "");
  }

  function handleLikedChange(arxivId: string, liked: 1 | -1 | 0) {
    setSelectedLiked(liked);
    setLikedCache((prev) => ({ ...prev, [arxivId]: liked }));
  }

  return (
    <div className="flex flex-col h-screen overflow-x-hidden bg-gray-50">
      <AppNav />

      {/* Two-pane body */}
      <div className="relative flex flex-1 overflow-hidden">
        {/* Left: recommendation list */}
        <div className={`absolute inset-0 w-full flex flex-col bg-white transition-transform duration-300 ease-in-out
          md:relative md:w-96 md:min-w-0 md:shrink-0 md:border-r md:border-gray-200 md:translate-x-0
          ${selectedArxivId !== null ? "-translate-x-full" : "translate-x-0"}`}>
          {/* Group/personal switcher — only shown when user is in at least one group */}
          {groups.length > 0 && (
            <div className="flex items-center gap-1 px-3 py-1.5 border-b border-gray-200 bg-white shrink-0 overflow-x-auto">
              <button
                onClick={() => setActiveGroupId(null)}
                className={`px-3 py-1 rounded text-sm font-medium whitespace-nowrap transition-colors ${
                  activeGroupId === null
                    ? "bg-blue-600 text-white"
                    : "bg-gray-100 text-gray-600 hover:bg-gray-200"
                }`}
              >
                Personal
              </button>
              {groups.map((g) => (
                <button
                  key={g.id}
                  onClick={() => setActiveGroupId(g.id)}
                  className={`px-3 py-1 rounded text-sm font-medium whitespace-nowrap transition-colors ${
                    activeGroupId === g.id
                      ? "bg-blue-600 text-white"
                      : "bg-gray-100 text-gray-600 hover:bg-gray-200"
                  }`}
                >
                  {g.name}
                </button>
              ))}
            </div>
          )}
          <RecommendationList
            selectedArxivId={selectedArxivId}
            onSelect={handleSelect}
            likedCache={likedCache}
            groupId={activeGroupId}
          />
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
              score={selectedScore}
              onLikedChange={handleLikedChange}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
