// Bar width is exp(0.5 * score), clamped to [0, 1].
// Color: exp(0.5*score) = 1 → green, = 1/e (score=-2) → yellow, = 0 → red.
export function scoreBar(score: number): { pct: number; color: string; hue: number } {
  const v = Math.min(1, Math.exp(0.5 * score));
  const invE = 1 / Math.E;
  let hue: number;
  if (v <= invE) {
    hue = 60 * (v / invE);                      // 0° (red) → 60° (yellow)
  } else {
    hue = 60 + 60 * ((v - invE) / (1 - invE)); // 60° (yellow) → 120° (green)
  }
  return { pct: Math.round(v * 100), color: `hsl(${hue}, 75%, 42%)`, hue };
}
