/**
 * Format an ISO 8601 timestamp string (e.g. "2026-04-02T06:18:38Z" or
 * "2026-04-02") as "2026-04-02 @ 06:18:38 UTC".
 *
 * Plain date strings (no "T") are returned with " @ 00:00:00 UTC" appended
 * so the format is consistent.  Returns the original string unchanged if it
 * cannot be parsed.
 */
export function formatTimestamp(raw: string | null | undefined): string {
  if (!raw) return "";
  // Normalize: ensure a trailing Z so Date parses it as UTC
  const normalized = raw.includes("T")
    ? raw.endsWith("Z") ? raw : raw + "Z"
    : raw + "T00:00:00Z";
  const d = new Date(normalized);
  if (isNaN(d.getTime())) return raw;
  const date = d.toISOString().slice(0, 10);           // "2026-04-02"
  const time = d.toISOString().slice(11, 19);           // "06:18:38"
  return `${date} @ ${time} UTC`;
}
