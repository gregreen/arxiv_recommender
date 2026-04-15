import { memo } from "react";
import DOMPurify from "dompurify";
import katex from "katex";

// Split text into alternating plain/math segments.
// Handles $$...$$ (display) and $...$ (inline), in that order.
function tokenize(text: string): { type: "text" | "display" | "inline"; value: string }[] {
  const tokens: { type: "text" | "display" | "inline"; value: string }[] = [];
  // Match $$...$$ first, then $...$
  const re = /\$\$([\s\S]+?)\$\$|\$([^$\n]+?)\$/g;
  let last = 0;
  let m: RegExpExecArray | null;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) tokens.push({ type: "text", value: text.slice(last, m.index) });
    if (m[1] !== undefined) tokens.push({ type: "display", value: m[1] });
    else tokens.push({ type: "inline", value: m[2] });
    last = m.index + m[0].length;
  }
  if (last < text.length) tokens.push({ type: "text", value: text.slice(last) });
  return tokens;
}

function renderMath(src: string, display: boolean): string {
  try {
    const html = katex.renderToString(src, { displayMode: display, throwOnError: false });
    return DOMPurify.sanitize(html, { USE_PROFILES: { html: true, mathMl: true, svg: true } });
  } catch {
    return src;
  }
}

interface MathTextProps {
  text: string;
  className?: string;
}

function MathText({ text, className }: MathTextProps) {
  const tokens = tokenize(text);
  return (
    <span className={className}>
      {tokens.map((tok, i) => {
        if (tok.type === "text") return <span key={i}>{tok.value}</span>;
        return (
          <span
            key={i}
            dangerouslySetInnerHTML={{ __html: renderMath(tok.value, tok.type === "display") }}
          />
        );
      })}
    </span>
  );
}

export default memo(MathText);
