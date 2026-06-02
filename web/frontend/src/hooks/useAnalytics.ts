import { useEffect } from "react";
import { useLocation } from "react-router-dom";

/**
 * Fire-and-forget page-visit beacon.
 *
 * Sends POST /api/analytics/event whenever the route changes.
 * Errors are silently swallowed — telemetry must never break the UI.
 *
 * Must be called from a component that is a descendant of <BrowserRouter>
 * (so useLocation() works).
 */
export function useAnalytics(): void {
  const location = useLocation();

  useEffect(() => {
    fetch("/api/analytics/event", {
      method: "POST",
      credentials: "include",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ page: location.pathname }),
    }).catch(() => {/* intentionally silenced */});
  }, [location.pathname]);
}
