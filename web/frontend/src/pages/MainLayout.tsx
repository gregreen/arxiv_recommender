import { useState, useEffect, useRef } from "react";
import { getMyPapers } from "../api/user";
import { useGroups } from "../contexts/GroupsContext";
import { useTour } from "../contexts/TourContext";
import AppNav from "../components/AppNav";
import RecommendationList from "../components/RecommendationList";
import PaperDetail from "../components/PaperDetail";

export default function MainLayout() {
  const { groups } = useGroups();
  const { pendingPaperOpen, notifyTourPaperLoaded, closePaperPanelCount } = useTour();
  const [selectedArxivId, setSelectedArxivId] = useState<string | null>(null);
  const [selectedLiked, setSelectedLiked] = useState<number | null>(null);
  const [selectedScore, setSelectedScore] = useState<number | null>(null);
  const [activeGroupId, setActiveGroupId] = useState<number | null>(null);
  const [showGroupDropdown, setShowGroupDropdown] = useState(false);
  const groupDropdownRef = useRef<HTMLDivElement>(null);
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

  // Close group dropdown when clicking outside
  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (groupDropdownRef.current && !groupDropdownRef.current.contains(e.target as Node)) {
        setShowGroupDropdown(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

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

  // Open paper requested by the tour.
  useEffect(() => {
    if (!pendingPaperOpen) return;
    handleSelect(pendingPaperOpen.arxivId, null, null);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pendingPaperOpen]);

  // Close paper panel when the tour signals it (count increases).
  // Use a ref so this never fires on initial mount — only on increments that
  // happen while this component instance is alive.
  const seenClosePaperPanelCountRef = useRef(closePaperPanelCount);
  useEffect(() => {
    if (closePaperPanelCount === seenClosePaperPanelCountRef.current) return;
    seenClosePaperPanelCountRef.current = closePaperPanelCount;
    setSelectedArxivId(null);
    setSelectedLiked(null);
    setSelectedScore(null);
  }, [closePaperPanelCount]);

  function handleLikedChange(arxivId: string, liked: 1 | -1 | 0) {
    setSelectedLiked(liked);
    setLikedCache((prev) => ({ ...prev, [arxivId]: liked }));
  }

  return (
    <div className="flex flex-col h-screen bg-gray-50">
      <AppNav />

      {/* Two-pane body */}
      <div className="relative flex flex-1 overflow-hidden">
        {/* Left: recommendation list */}
        <div className={`absolute inset-0 w-full flex flex-col bg-white transition-transform duration-300 ease-in-out
          md:relative md:w-96 md:min-w-0 md:shrink-0 md:border-r md:border-gray-200 md:translate-x-0
          ${selectedArxivId !== null ? "-translate-x-full" : "translate-x-0"}`}>
          {/* Group/personal switcher — only shown when user is in at least one group */}
          {groups.length > 0 && (
            <div id="tour-group-switcher" className="flex items-center gap-1 px-3 py-1.5 border-b border-gray-200 bg-white shrink-0">
              <button
                onClick={() => { setActiveGroupId(null); setShowGroupDropdown(false); }}
                className={`px-3 py-1 rounded text-sm font-medium whitespace-nowrap transition-colors ${
                  activeGroupId === null
                    ? "bg-blue-600 text-white"
                    : "bg-gray-100 text-gray-600 hover:bg-gray-200"
                }`}
              >
                Personal
              </button>

              {/* Single group: direct button (original behaviour) */}
              {groups.length === 1 && (
                <button
                  onClick={() => setActiveGroupId(groups[0].id)}
                  className={`px-3 py-1 rounded text-sm font-medium whitespace-nowrap transition-colors max-w-[180px] truncate ${
                    activeGroupId === groups[0].id
                      ? "bg-blue-600 text-white"
                      : "bg-gray-100 text-gray-600 hover:bg-gray-200"
                  }`}
                >
                  {groups[0].name}
                </button>
              )}

              {/* Multiple groups: dropdown trigger */}
              {groups.length > 1 && (
                <div ref={groupDropdownRef} className="relative">
                  <button
                    onClick={() => setShowGroupDropdown((v) => !v)}
                    className={`flex items-center gap-1 px-3 py-1 rounded text-sm font-medium whitespace-nowrap transition-colors max-w-[180px] ${
                      activeGroupId !== null
                        ? "bg-blue-600 text-white"
                        : "bg-gray-100 text-gray-600 hover:bg-gray-200"
                    }`}
                  >
                    <span className="truncate">
                      {activeGroupId !== null
                        ? (groups.find((g) => g.id === activeGroupId)?.name ?? "Groups")
                        : "Groups"}
                    </span>
                    <svg className="w-3 h-3 shrink-0" viewBox="0 0 10 6" fill="currentColor">
                      <path d="M0 0l5 6 5-6z" />
                    </svg>
                  </button>
                  {showGroupDropdown && (
                    <div className="absolute left-0 top-full mt-1 z-50 bg-white border border-gray-200 rounded shadow-lg min-w-[160px] py-1">
                      {groups.map((g) => (
                        <button
                          key={g.id}
                          onClick={() => { setActiveGroupId(g.id); setShowGroupDropdown(false); }}
                          className={`w-full text-left px-3 py-1.5 text-sm truncate transition-colors ${
                            activeGroupId === g.id
                              ? "bg-blue-50 text-blue-700 font-medium"
                              : "text-gray-700 hover:bg-gray-100"
                          }`}
                        >
                          {g.name}
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              )}
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
              onPaperLoaded={notifyTourPaperLoaded}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
