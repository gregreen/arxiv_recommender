import { useCallback, useEffect, useRef, useState } from "react";
import * as d3 from "d3";
import AppNav from "../components/AppNav";
import PaperDetail from "../components/PaperDetail";
import { getExplore } from "../api/explore";
import { getMyPapers } from "../api/user";
import { scoreBar } from "../components/scoreColor";
import type { ExploreResponse, TimeWindow } from "../api/types";

const WINDOWS: { value: TimeWindow; label: string }[] = [
  { value: "day",   label: "Day"   },
  { value: "week",  label: "Week"  },
  { value: "month", label: "Month" },
];

// Base dot radius at zoom level 1.  Scales as r = BASE_R / sqrt(k).
const BASE_R = 5;

export default function ExplorePage() {
  // ------------------------------------------------------------------
  // Time-window + data state
  // ------------------------------------------------------------------
  const [timeWindow, setTimeWindow]   = useState<TimeWindow>("week");
  const [data, setData]               = useState<ExploreResponse | null>(null);
  const [loading, setLoading]         = useState(true);
  const [fetchError, setFetchError]   = useState<string | null>(null);
  const [showLiked, setShowLiked]     = useState(true);
  const [colorMode, setColorMode]     = useState<"score" | "flat">("score");
  const colorModeRef = useRef<"score" | "flat">("score");
  colorModeRef.current = colorMode;

  // Liked cache (mirrors MainLayout pattern for instant feedback)
  const [likedCache, setLikedCache] = useState<Record<string, number>>({});
  const likedCacheRef = useRef<Record<string, number>>({});
  likedCacheRef.current = likedCache;

  // Seed the liked cache from the user's paper list on mount.
  useEffect(() => {
    getMyPapers()
      .then((papers) => {
        const map: Record<string, number> = {};
        for (const p of papers) map[p.arxiv_id] = p.liked;
        setLikedCache(map);
      })
      .catch(() => {});
  }, []);

  // Fetch explore data whenever the window changes.
  useEffect(() => {
    setLoading(true);
    setFetchError(null);
    getExplore(timeWindow)
      .then((d) => { setData(d); setLoading(false); })
      .catch((e: unknown) => {
        setFetchError(e instanceof Error ? e.message : "Failed to load explorer data.");
        setLoading(false);
      });
  }, [timeWindow]);

  // ------------------------------------------------------------------
  // Paper detail panel state (same pattern as MainLayout)
  // ------------------------------------------------------------------
  const [selectedArxivId, setSelectedArxivId] = useState<string | null>(null);
  const [selectedLiked,   setSelectedLiked]   = useState<number | null>(null);
  const [selectedScore,   setSelectedScore]   = useState<number | null>(null);

  // Back gesture via browser history.
  useEffect(() => {
    function handlePopState() {
      setSelectedArxivId(null);
      setSelectedLiked(null);
      setSelectedScore(null);
    }
    globalThis.addEventListener("popstate", handlePopState);
    return () => globalThis.removeEventListener("popstate", handlePopState);
  }, []);

  const handleSelect = useCallback((arxivId: string, liked: number | null, score: number | null) => {
    setSelectedArxivId(arxivId);
    setSelectedLiked(likedCacheRef.current[arxivId] ?? liked);
    setSelectedScore(score);
    globalThis.history.pushState({ detail: true }, "");
  }, []);

  // Keep a stable ref so D3 event handlers always call the latest version.
  const handleSelectRef = useRef(handleSelect);
  handleSelectRef.current = handleSelect;

  function handleLikedChange(arxivId: string, liked: 1 | -1 | 0) {
    setSelectedLiked(liked);
    setLikedCache((prev) => ({ ...prev, [arxivId]: liked }));
  }

  // ------------------------------------------------------------------
  // D3 scatter plot
  // ------------------------------------------------------------------
  const containerRef = useRef<HTMLDivElement>(null);
  const svgRef       = useRef<SVGSVGElement>(null);
  const tooltipRef   = useRef<HTMLDivElement>(null);
  const showLikedRef = useRef(showLiked);

  // Keep refs in sync without triggering D3 rebuild.
  useEffect(() => { showLikedRef.current = showLiked; }, [showLiked]);

  // Container size — triggers chart rebuild on resize.
  const [size, setSize] = useState<{ w: number; h: number } | null>(null);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      const { width, height } = entries[0].contentRect;
      if (width > 0 && height > 0) setSize({ w: width, h: height });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // Tooltip helpers (manipulate DOM directly to avoid React re-renders).
  function showTooltip(event: MouseEvent, title: string) {
    const el = tooltipRef.current;
    if (!el) return;
    el.textContent = title;
    el.style.display = "block";
    el.style.left = `${event.pageX + 12}px`;
    el.style.top  = `${event.pageY - 8}px`;
  }
  function hideTooltip() {
    if (tooltipRef.current) tooltipRef.current.style.display = "none";
  }

  // Main D3 effect — rebuilds when data or container size changes.
  useEffect(() => {
    if (!svgRef.current || !size || !data?.lowres_proj_available) return;
    if (data.papers.length === 0 && data.liked_overlay.length === 0) return;

    const { w, h } = size;
    const pad = 24;
    const xScale = d3.scaleLinear().domain([0, 1]).range([pad, w - pad]);
    const yScale = d3.scaleLinear().domain([0, 1]).range([pad, h - pad]);

    const svg = d3.select(svgRef.current);
    svg.attr("width", w).attr("height", h);
    svg.selectAll("*").remove();

    // Root group — zoom transform applied here.
    const root = svg.append("g").attr("class", "zoom-root");

    // --- Normal papers layer ---
    const likedSet = new Set(data.liked_overlay.map((p) => p.arxiv_id));
    const normalPapers = data.papers.filter((p) => !likedSet.has(p.arxiv_id));

    const cm = colorModeRef.current;

    root.append("g")
      .attr("class", "papers-layer")
      .selectAll<SVGCircleElement, typeof normalPapers[0]>("circle")
      .data(normalPapers)
      .join("circle")
      .attr("cx", (d) => xScale(d.x))
      .attr("cy", (d) => yScale(d.y))
      .attr("r", BASE_R)
      .attr("fill", (d) => {
        if (cm === "score") {
          return d.score != null ? scoreBar(d.score).color : "#cbd5e1";
        }
        return "#94a3b8";
      })
      .attr("fill-opacity", 0.6)
      .attr("cursor", "pointer")
      .on("mouseover", (event: MouseEvent, d) => showTooltip(event, d.title))
      .on("mouseout", hideTooltip)
      .on("click", (_event: MouseEvent, d) => handleSelectRef.current(d.arxiv_id, d.liked, d.score));

    // --- Liked overlay layer ---
    const likedLayer = root.append("g")
      .attr("class", "liked-layer")
      .style("display", showLikedRef.current ? null : "none");

    likedLayer
      .selectAll<SVGCircleElement, typeof data.liked_overlay[0]>("circle")
      .data(data.liked_overlay)
      .join("circle")
      .attr("cx", (d) => xScale(d.x))
      .attr("cy", (d) => yScale(d.y))
      .attr("r", BASE_R + 1)
      .attr("fill", "#22c55e")
      .attr("fill-opacity", 0.85)
      .attr("cursor", "pointer")
      .on("mouseover", (event: MouseEvent, d) => showTooltip(event, d.title))
      .on("mouseout", hideTooltip)
      .on("click", (_event: MouseEvent, d) => {
        const liked = likedCacheRef.current[d.arxiv_id] ?? 1;
        handleSelectRef.current(d.arxiv_id, liked, d.score);
      });

    // --- Zoom ---
    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.3, 30])
      .on("zoom", (event) => {
        root.attr("transform", event.transform.toString());
        const k = event.transform.k;
        const r = Math.min(9, Math.max(2, BASE_R / Math.sqrt(k)));
        root.selectAll<SVGCircleElement, unknown>("circle").attr("r", r);
      });

    svg.call(zoom);

  }, [data, size, colorMode]); // eslint-disable-line react-hooks/exhaustive-deps

  // Toggle liked layer visibility without rebuilding the chart.
  useEffect(() => {
    if (!svgRef.current) return;
    d3.select(svgRef.current)
      .select(".liked-layer")
      .style("display", showLiked ? null : "none");
  }, [showLiked]);

  // ------------------------------------------------------------------
  // Render
  // ------------------------------------------------------------------
  return (
    <div className="flex flex-col h-screen overflow-x-hidden bg-gray-50">
      <AppNav />

      <div className="relative flex flex-1 overflow-hidden">

        {/* ---- Left / main pane: scatter plot ---- */}
        <div
          className={`absolute inset-0 w-full flex flex-col bg-white transition-transform duration-300 ease-in-out
            md:relative md:flex-1 md:min-w-0 md:translate-x-0
            ${selectedArxivId !== null ? "-translate-x-full" : "translate-x-0"}`}
        >
          {/* Controls: window tabs + liked toggle */}
          <div className="flex items-center gap-1 p-3 border-b border-gray-200 shrink-0">
            {WINDOWS.map((w) => (
              <button
                key={w.value}
                onClick={() => setTimeWindow(w.value)}
                className={`px-3 py-1 rounded text-sm font-medium transition-colors ${
                  timeWindow === w.value
                    ? "bg-blue-600 text-white"
                    : "bg-gray-100 text-gray-600 hover:bg-gray-200"
                }`}
              >
                {w.label}
              </button>
            ))}

            <button
              onClick={() => setShowLiked((v) => !v)}
              className={`ml-2 flex items-center gap-1.5 px-3 py-1 rounded text-sm font-medium transition-colors ${
                showLiked
                  ? "bg-green-100 text-green-700 hover:bg-green-200"
                  : "bg-gray-100 text-gray-500 hover:bg-gray-200"
              }`}
              title={showLiked ? "Hide liked papers" : "Show liked papers"}
            >
              <span
                className="inline-block w-2.5 h-2.5 rounded-full"
                style={{ background: showLiked ? "#22c55e" : "#d1d5db" }}
              />
              Liked
            </button>

            <button
              onClick={() => setColorMode((v) => v === "score" ? "flat" : "score")}
              className={`ml-1 flex items-center gap-1.5 px-3 py-1 rounded text-sm font-medium transition-colors ${
                colorMode === "score"
                  ? "bg-blue-100 text-blue-700 hover:bg-blue-200"
                  : "bg-gray-100 text-gray-500 hover:bg-gray-200"
              }`}
              title={colorMode === "score" ? "Switch to flat colour" : "Switch to score colour"}
            >
              Score
            </button>
          </div>

          {/* Chart area */}
          <div ref={containerRef} className="flex-1 relative overflow-hidden">
            {/* Loading overlay */}
            {loading && (
              <div className="absolute inset-0 flex items-center justify-center text-sm text-gray-400">
                Loading…
              </div>
            )}

            {/* Fetch error */}
            {!loading && fetchError && (
              <div className="absolute inset-0 flex items-center justify-center p-8">
                <div className="text-sm text-yellow-700 bg-yellow-50 border border-yellow-200 rounded p-3">
                  {fetchError}
                </div>
              </div>
            )}

            {/* No projection yet */}
            {!loading && !fetchError && data && !data.lowres_proj_available && (
              <div className="absolute inset-0 flex items-center justify-center p-8">
                <div className="text-sm text-gray-500 text-center max-w-sm">
                  <p className="font-medium mb-1">Paper map is being computed.</p>
                  <p>This usually takes a few minutes. Check back shortly.</p>
                </div>
              </div>
            )}

            {/* SVG scatter plot */}
            <svg ref={svgRef} className="w-full h-full" />

            {/* Tooltip */}
            <div
              ref={tooltipRef}
              className="fixed z-50 hidden max-w-xs px-2 py-1 text-xs text-white bg-gray-800 rounded shadow-lg pointer-events-none whitespace-normal"
            />
          </div>
        </div>

        {/* ---- Right pane: paper detail ---- */}
        <div
          className={`absolute inset-0 w-full flex flex-col transition-transform duration-300 ease-in-out
            md:relative md:w-1/2 md:min-w-0 md:shrink-0 md:border-l md:border-gray-200 md:translate-x-0
            ${selectedArxivId !== null ? "translate-x-0" : "translate-x-full"}`}
        >
          {/* Mobile back button */}
          <div className="md:hidden shrink-0 flex items-center px-4 py-2 bg-white border-b border-gray-200">
            <button
              onClick={() => globalThis.history.back()}
              className="flex items-center gap-1.5 text-sm text-blue-600 hover:text-blue-800 transition-colors"
            >
              ← Back to explorer
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
