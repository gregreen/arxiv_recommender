import { Link, useNavigate } from "react-router-dom";
import { logout } from "../api/auth";
import { useAuth } from "../AuthContext";
import NavMenu from "../components/NavMenu";
import MathText from "../components/MathText";

// ---------------------------------------------------------------------------
// Demo content: LaTeX excerpt and its structured summary
// ---------------------------------------------------------------------------

const LATEX_EXCERPT = `...

\\title{Observational Evidence from Supernovae for an Accelerating Universe and a Cosmological Constant \\
\\vspace*{1.0cm}
{\\it To Appear in the Astronomical Journal}}
\\vspace*{0.3cm}

Adam G. Riess\\footnote{Department of Astronomy, University of California, 
Berkeley, CA 94720-3411}

...

\\section{Introduction}
This paper reports observations of 10 new high-redshift type Ia supernovae (SNe Ia) and the values of the cosmological parameters derived from them.   Together with the four high-redshift supernovae previously reported by our High-Z Supernova Search Team (Schmidt et al. 1998; Garnavich et al. 1998) and two others (Riess et al. 1998a), the sample of 16 is now large enough to yield interesting cosmological results of high statistical significance. Confidence in these results depends not on increasing the sample size but on improving our understanding of systematic uncertainties.

...

\\section {Conclusions}

1. We find the luminosity distances to well-observed SNe with 0.16 $\\leq$ $z$ $\\leq$ 0.97 measured by two methods to be in excess of the prediction of a low mass-density ($\\Omega_M$ $\\approx 0.2$) Universe by 0.25 to 0.28 mag.  A cosmological explanation is provided by a positive cosmological constant with 99.7\\% (3.0$\\sigma$) to $>$99.9\\% (4.0$\\sigma$) confidence using the complete spectroscopic SN Ia sample and the prior belief that $\\Omega_M \\geq 0$.

...`;

const SUMMARY_TEXT = `Keywords: Type Ia supernovae, cosmological constant, accelerating universe, Hubble constant, supernova cosmology

Scientific Questions: This study addresses the geometry and expansion history of the universe using standard candles. It seeks to constrain cosmological parameters such as the Hubble constant, matter density, and the cosmological constant. The primary goal is to determine if the universe is accelerating and to measure the value of the vacuum energy density.

Data: The study utilises spectral and photometric observations of 10 Type Ia supernovae in the redshift range $0.16 \\le z \\le 0.62$. Combined with previous data from the High-z Supernova Search Team, the sample includes 16 high-redshift supernovae and 34 nearby supernovae. Luminosity distances are derived using relations between supernova luminosity and light curve shape.

Methods: Luminosity distances are determined by methods employing relations between supernova Ia luminosity and light curve shape. Different light curve fitting methods and subsamples are analysed to test robustness. Constraints are placed on cosmological parameters using statistical confidence levels compared against prior constraints and theoretical models.

Results: High-redshift supernova Ia distances are on average 10%–15% farther than expected in a low mass density universe without a cosmological constant. Different methods and subsamples favour eternally expanding models with positive $\\Omega_\\Lambda$ and current acceleration $q_0 < 0$. Spectroscopically confirmed supernovae Ia are consistent with $q_0 < 0$ at $2.8\\,\\sigma$ and $3.9\\,\\sigma$ confidence levels. For a flat universe prior, $\\Omega_\\Lambda > 0$ is required at $7\\,\\sigma$ and $9\\,\\sigma$ significance. A universe closed by ordinary matter is ruled out at $7\\,\\sigma$ to $8\\,\\sigma$. The dynamical age is estimated at $14.2 \\pm 1.7\\,\\text{Gyr}$.

Conclusions: The data support a universe dominated by a cosmological constant rather than ordinary matter. Systematic errors do not reconcile the observations with a decelerating universe or zero vacuum energy. This implies a cosmological model requiring positive vacuum energy density and acceleration.

Key takeaway: Type Ia supernovae observations provide strong evidence for an accelerating universe driven by a positive cosmological constant. The results rule out a matter-closed universe and confirm $\\Omega_\\Lambda > 0$ at high statistical significance.`;

// Mirrors the heading-parsing logic in PaperDetail.tsx
const SUMMARY_HEADINGS = [
  "Keywords",
  "Scientific Questions",
  "Data",
  "Methods",
  "Results",
  "Conclusions",
  "Key takeaway",
];
const headingRe = new RegExp(`^(${SUMMARY_HEADINGS.join("|")}):`, "m");
const splitRe   = new RegExp(`(?=^(?:${SUMMARY_HEADINGS.join("|")}):)`, "m");

function SummaryPanel() {
  const parts = SUMMARY_TEXT.split(splitRe);
  return (
    <div className="text-sm text-gray-700 leading-relaxed space-y-[1.12em]">
      {parts.map((part, i) => {
        const m = headingRe.exec(part);
        if (!m) return <p key={i}><MathText text={part.trim()} /></p>;
        const heading = m[1];
        const body    = part.slice(m[0].length).trim();
        return (
          <p key={i}>
            <span className="font-semibold text-gray-800">{heading}: </span>
            <MathText text={body} />
          </p>
        );
      })}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Page component
// ---------------------------------------------------------------------------

export default function AboutPage() {
  const { user, clearUser } = useAuth();
  const navigate = useNavigate();

  async function handleLogout() {
    await logout().catch(() => {});
    clearUser();
    navigate("/login");
  }

  return (
    <div className="flex flex-col min-h-screen bg-gray-50">
      {/* Navbar */}
      <nav
        className="flex items-center gap-4 px-4 py-2 border-b border-blue-200 shrink-0"
        style={{ background: "linear-gradient(42deg, #ebf5ff, #91caff)" }}
      >
        <Link to="/" className="font-bold text-blue-700 text-lg">arXiv Recommender</Link>

        {user ? (
          <>
            <Link to="/library" className="text-sm text-gray-600 hover:text-gray-900">Library</Link>
            <span className="text-sm text-gray-600 font-medium">About</span>
            <NavMenu email={user.email} onLogout={handleLogout} />
          </>
        ) : (
          <>
            <span className="text-sm text-gray-600 font-medium">About</span>
            <Link to="/login" className="text-sm text-gray-600 hover:text-gray-900 ml-auto">Sign in / register</Link>
          </>
        )}
      </nav>

      {/* Body */}
      <main className="max-w-7xl mx-auto w-full px-6 py-8 space-y-6 text-gray-700">
        {/* Intro: "How it works:" label beside the text block on wide viewports, above on narrow */}
        <div className="flex flex-col items-start gap-4 lg:flex-row lg:items-start lg:justify-center lg:gap-8">
          <span className="font-bold text-blue-700 text-2xl whitespace-nowrap px-3 py-1 bg-blue-50 border border-blue-100 rounded-lg">How it works:</span>
          <div className="max-w-2xl space-y-6">
            <p className="text-base leading-relaxed">
              The{" "}
              <span className="font-bold text-blue-700">arXiv Recommender</span>{" "}
              shows you recently published papers that are related to your research interests,
              using two sources of information:
            </p>
            <ol className="list-decimal list-inside space-y-1 text-base leading-relaxed pl-2">
              <li>Papers that you mark as "relevant."</li>
              <li>Papers that you import into your library.</li>
            </ol>

            <p className="text-base leading-relaxed">
              The{" "}
              <span className="font-bold text-blue-700">arXiv Recommender</span>{" "}
              passes the full LaTeX source of every arXiv paper to an LLM (<a href="https://huggingface.co/Qwen/Qwen3.5-35B-A3B-Base" target="_blank" rel="noreferrer" className="text-blue-600 hover:underline">Qwen3.5-35B-A3B</a>), which distills it
              into a short, structured summary:
            </p>
          </div>
        </div>

        {/* Two-panel demo */}
        <div className="flex flex-col md:flex-row md:gap-12 items-start">
          {/* Left: LaTeX source — desktop arrow is absolutely anchored here */}
          <div className="w-full md:flex-1 relative">
            <pre className="bg-white border border-gray-200 rounded-lg p-4 text-xs font-mono leading-relaxed overflow-x-auto whitespace-pre-wrap break-words text-gray-700">
              {LATEX_EXCERPT}
            </pre>
            {/* Desktop: right arrow, centered vertically on the LaTeX block */}
            <div className="hidden md:flex absolute top-1/2 -translate-y-1/2 -right-10 items-center justify-center text-gray-400 select-none">
              <svg width="32" height="32" viewBox="0 0 36 36" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <line x1="4" y1="18" x2="30" y2="18" />
                <polyline points="21,10 30,18 21,26" />
              </svg>
            </div>
          </div>

          {/* Mobile: down arrow between the two panels */}
          <div className="flex md:hidden justify-center w-full py-1 text-gray-400 select-none">
            <svg width="32" height="32" viewBox="0 0 36 36" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <line x1="18" y1="4" x2="18" y2="30" />
              <polyline points="10,21 18,30 26,21" />
            </svg>
          </div>

          {/* Right: structured summary */}
          <div className="w-full md:flex-1 border border-gray-200 rounded-lg p-4 bg-white">
            <SummaryPanel />
          </div>
        </div>

        <div className="max-w-2xl mx-auto">
          <p className="text-base leading-relaxed">
            This summary, combined with metadata, is then passed to an Embedding LLM (<a href="https://huggingface.co/Qwen/Qwen3-Embedding-8B" target="_blank" rel="noreferrer" className="text-blue-600 hover:underline">Qwen3-Embedding-8B</a>), which converts the paper into a high-dimensional vector representing its subject and content. Similar papers will be converted into similar vectors. The recommendation algorithm uses these vector embeddings to predict which new papers you will find interesting.
          </p>
        </div>

        <div className="max-w-2xl lg:max-w-4xl mx-auto w-full flex flex-col lg:flex-row lg:items-center lg:gap-6">
          <div className="w-full lg:flex-1">
            <img src="/embedding_space.svg" alt="Embedding space visualisation" className="w-full h-auto" />
          </div>
          <p className="w-full text-sm text-gray-600 lg:w-52 lg:flex-shrink-0">
            <span className="font-bold">Paper embeddings</span>: Each paper is embedded into a high-dimensional vector space, based on its meaning and content. Here, we plot a low-dimensional visualization of the embedding. <span className="font-semibold text-gray-500">Gray</span> dots show random <span className="font-mono whitespace-nowrap">astro-ph</span> papers, tracing the astrophysics research landscape. Selected papers, highlighted in <span className="font-semibold text-blue-600">blue</span>, show that similar papers are positioned close together. We can also embed search terms (<span className="font-semibold text-green-600">green</span>) into the same space in order to retrieve relevant papers.
          </p>
        </div>
      </main>
    </div>
  );
}
