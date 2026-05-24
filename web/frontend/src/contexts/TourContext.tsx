import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { useJoyride, EVENTS, ACTIONS, type EventData, type Controls } from "react-joyride";
import { useNavigate, useLocation } from "react-router-dom";
import { useAuth } from "../AuthContext";
import { getRecommendations } from "../api/recommendations";
import { getMyGroups } from "../api/groups";

interface TourStep {
  target: string;
  content: string;
  placement?: "top" | "bottom" | "left" | "right" | "auto" | "center";
  title?: string;
  route: string;
  floatingOptions?: Record<string, unknown>;
  spotlightTarget?: string;
  requiresPaper?: boolean;
  scrollOffset?: number;
}

interface TourState {
  startTour: () => Promise<void>;
  openImportAccordion: boolean;
  clearImportAccordion: () => void;
  pendingPaperOpen: { arxivId: string } | null;
  notifyTourPaperLoaded: () => void;
  closePaperPanelCount: number;
}

const TourContext = createContext<TourState>({
  startTour: async () => {},
  openImportAccordion: false,
  clearImportAccordion: () => {},
  pendingPaperOpen: null,
  notifyTourPaperLoaded: () => {},
  closePaperPanelCount: 0,
});

export function useTour(): TourState {
  return useContext(TourContext);
}

export function TourProvider({ children }: { children: ReactNode }) {
  const { user, setTutorialShown } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();

  const [steps, setSteps] = useState<TourStep[]>([]);
  const [openImportAccordion, setOpenImportAccordion] = useState(false);
  const [pendingPaperOpen, setPendingPaperOpen] = useState<{ arxivId: string } | null>(null);
  const [closePaperPanelCount, setClosePaperPanelCount] = useState(0);

  // arxiv_id of the first paper in the weekly recommendations list (set during buildSteps).
  const tourFirstPaperIdRef = useRef<string | null>(null);

  // Step index to resume at once the paper detail panel has loaded.
  const pendingResumeAfterPaperRef = useRef<number | null>(null);

  // When we pause mid-tour for navigation, record the destination here.
  const pendingResumeRef = useRef<{ index: number; route: string } | null>(null);

  // Set to true after steps are loaded so the start effect fires once.
  const [pendingStart, setPendingStart] = useState(false);

  // Prevent auto-show from firing more than once per session.
  const autoShownRef = useRef(false);

  // --------------------------------------------------------------------------
  // Build adaptive step list
  // --------------------------------------------------------------------------

  async function buildSteps(): Promise<TourStep[]> {
    let hasRealRecs = false;
    let isInGroup = false;

    try {
      const [recs, groups] = await Promise.all([
        getRecommendations("week"),
        getMyGroups(),
      ]);
      hasRealRecs = !recs.onboarding;
      isInGroup = groups.length > 0;
      tourFirstPaperIdRef.current = recs.results[0]?.arxiv_id ?? null;
    } catch {
      // non-fatal; use defaults
    }

    const list: TourStep[] = [
      {
        target: "body",
        title: "Welcome",
        content:
          "This site gives personalised arXiv paper recommendations, updated daily as new papers are processed and categorised. The website learns your preferences over time as you like or dislike papers.",
        placement: "center",
        route: "/",
      },
      {
        // Anchor the tooltip to the tab bar (small element, placement: bottom
        // puts tooltip below it into the visible list area on all screen sizes)
        // while spotlighting the full recommendations list.
        target: "#tour-time-tabs",
        spotlightTarget: "#tour-recs-list",
        title: "Your recommendations",
        content: hasRealRecs
          ? "Papers recommended for you based on your library. The more papers you like, the better these get."
          : "Once you mark papers here as relevant or add papers to your library, personalised recommendations will appear here.",
        placement: "bottom",
        route: "/",
      },
      {
        target: "#tour-time-tabs",
        title: "Time windows",
        content: "Switch between Day, Week, and Month to explore recently published papers across different time ranges.",
        placement: "bottom",
        route: "/",
      },
      {
        target: "#tour-search-btn",
        title: "Semantic search",
        content: "Search for papers by meaning — type a concept or phrase and find the most relevant results in any time window.",
        placement: "left",
        route: "/",
      },
    ];

    if (isInGroup) {
      list.push({
        target: "#tour-group-switcher",
        title: "Group recommendations",
        content: "Switch between your personal recommendations and aggregated group recommendations here.",
        placement: "bottom",
        route: "/",
      });
    }

    list.push(
      {
        target: "#tour-paper-list-first",
        title: "Opening a paper",
        content: "Click on any paper to see more details about it.",
        placement: "right",
        route: "/",
      },
      {
        target: "#tour-paper-like-buttons",
        title: "Like or dislike papers",
        content: "Rate a paper as 👍 Relevant or 👎 Not Relevant. Your ratings train the recommendation engine — the more papers you rate, the better your recommendations become.",
        placement: "bottom",
        route: "/",
        requiresPaper: true,
        scrollOffset: 100,
      },
      {
        target: "#tour-paper-summary",
        title: "Auto-generated summary",
        content: "Each paper includes an automatically generated structured summary, highlighting the scientific questions, data, methods and conclusions.",
        placement: "bottom",
        route: "/",
        requiresPaper: true,
        scrollOffset: 80,
      },
      {
        target: "#tour-import-accordion",
        title: "Import your library",
        content: "Expand this section to import papers you have already read and liked or disliked. Likes and dislikes train your recommendations.",
        placement: "right",
        route: "/library",
      },
      {
        target: "#tour-arxiv-input",
        title: "Add a paper",
        content: "Paste an arXiv ID (e.g. 2401.12345) and mark it as liked or disliked to start training your recommendations. Adding your own papers is an effective way to train the recommendation model.",
        placement: "bottom",
        route: "/library",
      },
      {
        target: "#tour-create-group",
        title: "Create a group",
        content: "Enter a name and click \"Create group\" to get started. You can then invite colleagues via a shareable link. You will then be able to see aggregated group recommendations.",
        placement: "top",
        route: "/groups",
      }
    );

    return list;
  }

  // --------------------------------------------------------------------------
  // Joyride setup (v3 hook API)
  // --------------------------------------------------------------------------

  const joyrideSteps = useMemo(
    () =>
      steps.map(({ target, content, title, placement, floatingOptions, spotlightTarget, scrollOffset }) => ({
        target,
        content,
        title,
        placement: placement ?? "auto",
        ...(floatingOptions ? { floatingOptions } : {}),
        ...(spotlightTarget ? { spotlightTarget } : {}),
        ...(scrollOffset !== undefined ? { scrollOffset } : {}),
      })),
    [steps]
  );

  // Keep mutable refs so handleOnEvent never needs to re-create (avoids
  // passing a new onEvent to useJoyride on every render, which can reset state).
  const stepsRef = useRef(steps);
  useEffect(() => { stepsRef.current = steps; }, [steps]);

  const pathnameRef = useRef(location.pathname);
  useEffect(() => { pathnameRef.current = location.pathname; }, [location.pathname]);

  const setTutorialShownRef = useRef(setTutorialShown);
  useEffect(() => { setTutorialShownRef.current = setTutorialShown; }, [setTutorialShown]);

  const handleOnEvent = useCallback(
    (data: EventData, controls: Controls) => {
      const { type, index, action } = data;

      // Tour fully ended or skipped — mark tutorial shown and return to recommendations.
      if (type === EVENTS.TOUR_END) {
        setTutorialShownRef.current().catch(() => {});
        setPendingPaperOpen(null);
        setClosePaperPanelCount((n) => n + 1);
        navigate("/");
        return;
      }

      // Target not found — skip that step rather than letting joyride terminate.
      if (type === EVENTS.TARGET_NOT_FOUND) {
        const stps = stepsRef.current;
        // If the missing target belongs to a requiresPaper step, the paper panel
        // may not have finished rendering yet.  Re-trigger the paper-load
        // mechanism so we retry once the paper is actually in the DOM.
        if (stps[index]?.requiresPaper && tourFirstPaperIdRef.current) {
          controls.stop(false);
          pendingResumeAfterPaperRef.current = index;
          setPendingPaperOpen({ arxivId: tourFirstPaperIdRef.current });
          return;
        }
        const nextIndex = index + 1;
        if (nextIndex >= stps.length) return;
        const nextStep = stps[nextIndex];
        if (nextStep.route !== pathnameRef.current) {
          controls.stop(false);
          if (nextStep.route === "/library") setOpenImportAccordion(true);
          pendingResumeRef.current = { index: nextIndex, route: nextStep.route };
          navigate(nextStep.route);
        } else {
          controls.go(nextIndex);
        }
        return;
      }

      // A step was completed — check if next step is on a different page.
      if (type === EVENTS.STEP_AFTER) {
        const stps = stepsRef.current;
        const nextIndex = action === ACTIONS.PREV ? index - 1 : index + 1;
        if (nextIndex < 0 || nextIndex >= stps.length) return;

        const nextStep = stps[nextIndex];
        // When going back from a requiresPaper step to a step that doesn't need
        // the paper panel, close it.
        if (action === ACTIONS.PREV && stps[index]?.requiresPaper && !nextStep.requiresPaper) {
          setPendingPaperOpen(null);
          setClosePaperPanelCount((n) => n + 1);
        }
        // Next step requires the paper detail panel — open it and wait for load.
        if (nextStep.requiresPaper && tourFirstPaperIdRef.current) {
          if (action !== ACTIONS.PREV && !stps[index]?.requiresPaper) {
            // Going forward from a non-paper step: stop and wait for paper to open.
            controls.stop(false);
            pendingResumeAfterPaperRef.current = nextIndex;
            setPendingPaperOpen({ arxivId: tourFirstPaperIdRef.current });
            return;
          } else if (nextStep.route !== pathnameRef.current) {
            // Going back to a requiresPaper step on a different route: navigate
            // there and re-open the paper panel (MainLayout will have remounted).
            controls.stop(false);
            pendingResumeAfterPaperRef.current = nextIndex;
            setPendingPaperOpen({ arxivId: tourFirstPaperIdRef.current });
            navigate(nextStep.route);
            return;
          }
          // Going back to a requiresPaper step on the same route: paper is
          // already open, let joyride advance normally.
        }
        if (nextStep.route !== pathnameRef.current) {
          controls.stop(false);
          if (nextStep.route === "/library") {
            setOpenImportAccordion(true);
          }
          pendingResumeRef.current = { index: nextIndex, route: nextStep.route };
          navigate(nextStep.route);
        }
        // Same route: joyride advances automatically in continuous mode.
      }
    },
    [navigate] // stable: navigate from useNavigate() never changes
  );

  const { controls, Tour } = useJoyride({
    steps: joyrideSteps,
    continuous: true,
    scrollToFirstStep: true,
    options: {
      skipBeacon: true,
      overlayClickAction: false,
      primaryColor: "#3b82f6",
      zIndex: 10000,
    },
    onEvent: handleOnEvent,
  });

  // Keep a stable ref to controls for use in effects after navigation.
  const controlsRef = useRef(controls);
  useEffect(() => {
    controlsRef.current = controls;
  }, [controls]);

  // --------------------------------------------------------------------------
  // Start tour once steps are loaded
  // --------------------------------------------------------------------------

  useEffect(() => {
    if (!pendingStart || joyrideSteps.length === 0) return;
    setPendingStart(false);

    if (location.pathname !== "/") {
      // Navigate to root first; resume effect will start the tour.
      pendingResumeRef.current = { index: 0, route: "/" };
      navigate("/");
    } else {
      setTimeout(() => controlsRef.current.start(0), 300);
    }
  }, [pendingStart, joyrideSteps, location.pathname, navigate]);

  // --------------------------------------------------------------------------
  // Resume tour after navigation
  // --------------------------------------------------------------------------

  useEffect(() => {
    const pending = pendingResumeRef.current;
    if (!pending || location.pathname !== pending.route) return;

    const timer = setTimeout(() => {
      pendingResumeRef.current = null;
      controlsRef.current.start(pending.index);
    }, 300);

    return () => clearTimeout(timer);
  }, [location.pathname]);

  // --------------------------------------------------------------------------
  // Auto-show for first-time users
  // --------------------------------------------------------------------------

  useEffect(() => {
    if (!user || user.tutorialShown || autoShownRef.current) return;
    autoShownRef.current = true;
    const timer = setTimeout(() => {
      startTour();
    }, 800);
    return () => clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [user]);

  // --------------------------------------------------------------------------
  // Public API
  // --------------------------------------------------------------------------

  const startTour = useCallback(async () => {
    controlsRef.current.stop();
    const built = await buildSteps();
    setSteps(built);
    setPendingStart(true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const clearImportAccordion = useCallback(() => {
    setOpenImportAccordion(false);
  }, []);

  const notifyTourPaperLoaded = useCallback(() => {
    const idx = pendingResumeAfterPaperRef.current;
    if (idx === null) return;
    pendingResumeAfterPaperRef.current = null;
    // Bind to a number-typed const so TypeScript doesn't lose narrowing inside
    // the tryStart closure (closures can outlive the null-guard scope).
    const stepIdx: number = idx;
    // Poll until the step's target element is present in the DOM (handles the
    // race between React re-rendering paper content and joyride searching for
    // the target), then start the tour.
    const target = stepsRef.current[stepIdx]?.target;
    function tryStart(retriesLeft: number) {
      if (!target || typeof target !== "string" || document.querySelector(target)) {
        setTimeout(() => controlsRef.current.start(stepIdx), 50);
        return;
      }
      if (retriesLeft <= 0) {
        controlsRef.current.start(stepIdx); // give up waiting; let TARGET_NOT_FOUND handle it
        return;
      }
      setTimeout(() => tryStart(retriesLeft - 1), 100);
    }
    tryStart(20); // poll up to ~2 s
  }, []);

  // --------------------------------------------------------------------------
  // Render
  // --------------------------------------------------------------------------

  return (
    <TourContext.Provider value={{ startTour, openImportAccordion, clearImportAccordion, pendingPaperOpen, notifyTourPaperLoaded, closePaperPanelCount }}>
      {Tour}
      {children}
    </TourContext.Provider>
  );
}
