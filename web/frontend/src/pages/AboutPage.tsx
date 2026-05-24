import { Link, useNavigate, useParams } from "react-router-dom";
import { QRCodeSVG } from "qrcode.react";
import AppNav from "../components/AppNav";
import MathText from "../components/MathText";
import PaperRow from "../components/PaperRow";
import { useTour } from "../contexts/TourContext";
import { useAuth } from "../AuthContext";
import type { Recommendation } from "../api/types";

// ---------------------------------------------------------------------------
// Demo content: LaTeX excerpt and its structured summary
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Static demo data for the Tutorial page
// ---------------------------------------------------------------------------

const DEMO_PAPERS: Recommendation[] = [
  {
    arxiv_id: "1706.03762",
    title: "Attention Is All You Need",
    authors: ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar", "Jakob Uszkoreit", "Llion Jones", "Aidan N. Gomez", "Łukasz Kaiser", "Illia Polosukhin"],
    published_date: "2017-06-12T00:00:00",
    score: -0.01,
    rank: 1,
    liked: null,
    generated_at: null,
  },
  {
    arxiv_id: "astro-ph/9805200",
    title: "Observational Evidence from Supernovae for an Accelerating Universe and a Cosmological Constant",
    authors: ["Adam G. Riess", "Alexei V. Filippenko", "Peter Challis", "Alejandro Clocchiatti", "Alan Diercks", "Peter M. Garnavich", "Ron L. Gilliland", "Craig J. Hogan", "Saurabh Jha", "Robert P. Kirshner", "B. Leibundgut", "M. M. Phillips", "David Reiss", "Brian P. Schmidt", "Robert A. Schommer", "R. Chris Smith", "Jason Spyromilio", "Christopher Stubbs", "Nicholas B. Suntzeff", "John Tonry"],
    published_date: "1998-05-15T00:00:00",
    score: -0.3,
    rank: 2,
    liked: null,
    generated_at: null,
  },
  {
    arxiv_id: "astro-ph/9812133",
    title: "Measurements of \\Omega and \\Lambda from 42 High-Redshift Supernovae",
    authors: ["S. Perlmutter", "G. Aldering", "G. Goldhaber", "R. A. Knop", "P. Nugent", "P. G. Castro", "S. Deustua", "S. Fabbro", "A. Goobar", "D. E. Groom", "I. M. Hook", "A. G. Kim", "M. Y. Kim", "J. C. Lee", "N. J. Nunes", "R. Pain", "C. R. Pennypacker", "R. Quimby", "C. Lidman", "R. S. Ellis", "M. Irwin", "R. G. McMahon", "P. Ruiz-Lapuente", "N. Walton", "B. Schaefer", "B. J. Boyle", "A. V. Filippenko", "T. Matheson", "A. S. Fruchter", "N. Panagia", "H. J. M. Newberg", "W. J. Couch"],
    published_date: "1998-12-08T00:00:00",
    score: -0.7,
    rank: 3,
    liked: null,
    generated_at: null,
  },
  {
    arxiv_id: "astro-ph/9710327",
    title: "Maps of Dust Infrared Emission for Use in Estimation of Cosmic Microwave Background Radiation Foregrounds and Cosmic Fabric Constants",
    authors: ["David J. Schlegel", "Douglas P. Finkbeiner", "Marc Davis"],
    published_date: "1997-10-12T00:00:00",
    score: -1.0,
    rank: 4,
    liked: null,
    generated_at: null,
  },
  {
    arxiv_id: "1810.04805",
    title: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
    authors: ["Jacob Devlin", "Ming-Wei Chang", "Kenton Lee", "Kristina Toutanova"],
    published_date: "2018-10-11T00:00:00",
    score: -1.5,
    rank: 5,
    liked: null,
    generated_at: null,
  },
];
const DUST_PAPERS: Recommendation[] = [
  {
    arxiv_id: "astro-ph/9710327",
    title: "Maps of Dust Infrared Emission for Use in Estimation of Cosmic Microwave Background Radiation Foregrounds and Cosmic Fabric Constants",
    authors: ["David J. Schlegel", "Douglas P. Finkbeiner", "Marc Davis"],
    published_date: "1997-10-12T00:00:00",
    score: -0.2,
    rank: 1,
    liked: null,
    generated_at: null,
  },
  {
    arxiv_id: "astro-ph/0608003",
    title: "Infrared Emission from Interstellar Dust. IV. The Silicate-Graphite-PAH Model",
    authors: ["B. T. Draine", "Aigen Li"],
    published_date: "2006-08-01T00:00:00",
    score: -0.8,
    rank: 2,
    liked: null,
    generated_at: null,
  },
  {
    arxiv_id: "astro-ph/0304489",
    title: "Interstellar Dust Grains",
    authors: ["B. T. Draine"],
    published_date: "2003-04-28T00:00:00",
    score: -1.5,
    rank: 3,
    liked: null,
    generated_at: null,
  },
  {
    arxiv_id: "astro-ph/9809387",
    title: "Correcting for the Influence of the Milky Way on Extragalactic Distance Determinations",
    authors: ["Edward L. Fitzpatrick"],
    published_date: "1998-09-27T00:00:00",
    score: -3.0,
    rank: 4,
    liked: null,
    generated_at: null,
  },
];
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
// Static recommendations demo widget (used in Tutorial page)
// ---------------------------------------------------------------------------

const WINDOWS = ["Day", "Week", "Month"] as const;

interface StaticRecsDemoProps {
  papers?: Recommendation[];
  activeTab?: "Day" | "Week" | "Month";
  groupName?: string;
  showGroupMethod?: boolean;
}

function StaticRecsDemo({ papers = DEMO_PAPERS, activeTab = "Week", groupName, showGroupMethod }: StaticRecsDemoProps) {
  return (
    <div className="pointer-events-none select-none border border-gray-200 rounded-lg shadow-sm overflow-hidden bg-white">
      {/* Group switcher (only when groupName is provided) */}
      {groupName && (
        <div className="flex items-center gap-1 px-3 py-1.5 border-b border-gray-200 bg-white">
          <span className="px-3 py-1 rounded text-sm font-medium whitespace-nowrap bg-gray-100 text-gray-600">Personal</span>
          <span className="px-3 py-1 rounded text-sm font-medium whitespace-nowrap bg-blue-600 text-white">{groupName}</span>
        </div>
      )}
      {/* Tab bar */}
      <div className="flex items-center justify-between px-3 py-1.5 border-b border-gray-100 bg-gray-50">
        <div className="flex gap-1">
          {WINDOWS.map((w) => (
            <span
              key={w}
              className={`px-3 py-1 text-sm font-medium rounded transition-colors ${
                w === activeTab
                  ? "bg-blue-600 text-white"
                  : "text-gray-500 bg-white border border-gray-200"
              }`}
            >
              {w}
            </span>
          ))}
        </div>
        {/* Icon buttons (cosmetic) */}
        <div className="flex items-center gap-1.5 text-gray-300">
          <span className="p-1 rounded">
            <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <polyline points="23 4 23 10 17 10" />
              <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10" />
            </svg>
          </span>
          <span className="p-1 rounded">
            <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="11" cy="11" r="8" />
              <line x1="21" y1="21" x2="16.65" y2="16.65" />
            </svg>
          </span>
        </div>
      </div>
      {/* Paper list */}
      <div className="p-3">
        {showGroupMethod && (
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs text-gray-400">4 of 4 members active</span>
            <span className="border border-gray-200 rounded px-1.5 py-0.5 text-xs text-gray-500">Voting</span>
          </div>
        )}
        {papers.map((paper) => (
          <PaperRow key={paper.arxiv_id} rec={paper} selected={false} onClick={() => {}} />
        ))}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Static group creation + invite demo widget
// ---------------------------------------------------------------------------

const FAKE_INVITE_URL = "https://arxiv-recommender.example/join-group?token=abc123XYZ";

function StaticGroupDemo() {
  return (
    <div className="pointer-events-none select-none space-y-3">
      {/* Create-group card */}
      <div className="bg-white border border-gray-200 rounded-lg px-6 py-5 space-y-3">
        <h3 className="text-sm font-semibold text-gray-700">Create a group</h3>
        <div className="flex gap-2">
          <span className="flex-1 border border-blue-400 ring-1 ring-blue-200 rounded px-3 py-2 text-sm text-gray-700">
            Dust enthusiasts
          </span>
          <span className="px-4 py-2 bg-blue-600 text-white text-sm font-medium rounded">
            Create group
          </span>
        </div>
      </div>
      {/* Invite row */}
      <div className="bg-white border border-gray-200 rounded-lg px-4 py-4 space-y-2">
        <h3 className="text-sm font-semibold text-gray-700 mb-2">Share an invite link</h3>
        <div className="flex flex-col bg-gray-50 border border-gray-200 rounded px-3 py-2 gap-2">
          <div className="flex items-center gap-2">
            <span className="font-mono text-xs text-gray-500 truncate flex-1">{FAKE_INVITE_URL}</span>
            <span className="text-xs text-gray-400 shrink-0">10 uses left</span>
            <span className="text-xs text-gray-400 shrink-0">exp 01/06/2026</span>
            <span className="shrink-0 text-xs px-2 py-1 rounded bg-blue-600 text-white">Copy</span>
            <span className="shrink-0 text-xs px-2 py-1 rounded bg-blue-100 text-blue-700">QR</span>
            <span className="shrink-0 text-xs px-2 py-1 rounded bg-gray-200 text-gray-600">Revoke</span>
          </div>
          <div className="flex justify-center py-2">
            <QRCodeSVG value={FAKE_INVITE_URL} size={140} />
          </div>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Static library import demo widget
// ---------------------------------------------------------------------------

function StaticLibraryImportDemo() {
  return (
    <div className="pointer-events-none select-none border border-gray-200 rounded-lg shadow-sm overflow-hidden bg-white">
      {/* Accordion header (open state) */}
      <div className="flex items-center justify-between px-4 py-3 bg-white border-b border-gray-100">
        <span className="text-sm font-medium text-gray-700">Import papers</span>
        <svg className="w-4 h-4 text-gray-500 rotate-180" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <polyline points="6 9 12 15 18 9" />
        </svg>
      </div>
      {/* Accordion body */}
      <div className="px-4 py-4 space-y-5">
        {/* Add by arXiv ID */}
        <div>
          <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">Add by arXiv ID</h3>
          <div className="flex items-center gap-2">
            <span className="flex-1 border border-gray-300 rounded px-3 py-1.5 text-xs text-gray-700">astro-ph/9710327</span>
            <span className="border border-gray-300 rounded px-2 py-1.5 text-sm text-gray-600">Liked</span>
            <span className="w-8 h-8 flex items-center justify-center bg-blue-600 text-white rounded text-lg font-bold leading-none">+</span>
          </div>
        </div>
        {/* Bulk import */}
        <div>
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wide">Import from NASA ADS</h3>
            <span className="bg-blue-600 text-white text-xs font-medium rounded px-3 py-1">Import</span>
          </div>
          <p className="text-xs text-gray-500 mb-2">
            Export an ADS library using the Custom %X format, and paste the contents below.
          </p>
          <div className="w-full border border-gray-300 rounded px-3 py-2 text-sm text-gray-500 font-mono leading-relaxed whitespace-pre-wrap bg-white">
            {"arXiv:astro-ph/9710327\narXiv:astro-ph/0608003\narXiv:astro-ph/0304489\narXiv:astro-ph/9809387"}
          </div>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Static paper detail header demo widget
// ---------------------------------------------------------------------------

function StaticPaperDetailDemo() {
  // Hardcoded values for "Attention Is All You Need", score = -0.010, liked = 1
  // scoreBar(-0.01) → hue ≈ 120 (green)
  const hue = 120;
  return (
    <div className="pointer-events-none select-none border border-gray-200 rounded-lg shadow-sm bg-white p-6">
      <div className="flex items-start justify-between gap-4 mb-2">
        <h2 className="text-[23px] font-semibold text-gray-900 leading-snug">
          Attention Is All You Need
        </h2>
        <span
          className="text-xs font-mono whitespace-nowrap mt-1 shrink-0 px-2 py-0.5 rounded-md"
          style={{
            color: `hsl(${hue}, 70%, 35%)`,
            backgroundColor: `hsla(${hue}, 75%, 50%, 0.1)`,
            border: `1px solid hsla(${hue}, 70%, 45%, 0.5)`,
          }}
        >
          -0.010
        </span>
      </div>
      <div className="text-base text-gray-500 mb-1">
        Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones et al.
      </div>
      <div className="text-base text-gray-400 mb-4">2017-06-12 @ 00:00:00 UTC</div>
      <div className="flex gap-2 mb-4">
        <span className="px-3 py-1.5 rounded text-sm font-medium bg-green-600 text-white">
          👍 Relevant
        </span>
        <span className="px-3 py-1.5 rounded text-sm font-medium bg-gray-100 text-gray-700">
          👎 Not Relevant
        </span>
        <span className="px-3 py-1.5 rounded text-sm bg-blue-50 text-blue-700">
          arXiv:1706.03762 ↗
        </span>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// "How it works" tab content (existing About page body, extracted)
// ---------------------------------------------------------------------------

function HowItWorksContent() {
  return (
    <div className="space-y-6 text-gray-700">
      {/* Intro */}
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
            <li>Papers that you mark as <span className="inline-flex items-center px-1.5 py-0.5 rounded text-sm font-medium bg-green-600 text-white">👍 Relevant</span> or <span className="inline-flex items-center px-1.5 py-0.5 rounded text-sm font-medium bg-red-600 text-white">👎 Not Relevant</span>.</li>
            <li>Papers that you import into your library.</li>
          </ol>
          <p className="text-base leading-relaxed">
            The{" "}
            <span className="font-bold text-blue-700">arXiv Recommender</span>{" "}
            passes the full LaTeX source of every arXiv paper to a highly efficient reasoning LLM (<a href="https://huggingface.co/Qwen/Qwen3.5-35B-A3B-Base" target="_blank" rel="noreferrer" className="text-blue-600 hover:underline">Qwen3.5-35B-A3B</a>), which distills it
            into a short, structured summary:
          </p>
        </div>
      </div>

      {/* Two-panel demo */}
      <div className="flex flex-col md:flex-row md:gap-12 items-start">
        <div className="w-full md:flex-1 relative">
          <pre className="bg-white border border-gray-200 rounded-lg p-4 text-xs font-mono leading-relaxed overflow-x-auto whitespace-pre-wrap break-words text-gray-700">
            {LATEX_EXCERPT}
          </pre>
          <div className="hidden md:flex absolute top-1/2 -translate-y-1/2 -right-10 items-center justify-center text-gray-400 select-none">
            <svg width="32" height="32" viewBox="0 0 36 36" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <line x1="4" y1="18" x2="30" y2="18" />
              <polyline points="21,10 30,18 21,26" />
            </svg>
          </div>
        </div>
        <div className="flex md:hidden justify-center w-full py-1 text-gray-400 select-none">
          <svg width="32" height="32" viewBox="0 0 36 36" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <line x1="18" y1="4" x2="18" y2="30" />
            <polyline points="10,21 18,30 26,21" />
          </svg>
        </div>
        <div className="w-full md:flex-1 border border-gray-200 rounded-lg p-4 bg-white">
          <SummaryPanel />
        </div>
      </div>

      <div className="max-w-2xl mx-auto">
        <p className="text-base leading-relaxed">
          This summary, combined with metadata, is then passed to an embedding model (<a href="https://huggingface.co/Qwen/Qwen3-Embedding-8B" target="_blank" rel="noreferrer" className="text-blue-600 hover:underline">Qwen3-Embedding-8B</a>), which converts the paper into a high-dimensional vector representing its subject and content. Similar papers will be converted into similar vectors. The recommendation algorithm uses these vector embeddings to predict which new papers you will find interesting.
        </p>
      </div>

      <div className="max-w-2xl lg:max-w-4xl mx-auto w-full flex flex-col lg:flex-row lg:items-center lg:gap-6">
        <div className="w-full lg:flex-1">
          <img src="/embedding_space.svg" alt="Embedding space visualisation" className="w-full h-auto" />
        </div>
        <p className="w-full text-sm text-gray-600 lg:w-52 lg:flex-shrink-0">
          <span className="font-bold">Paper embeddings</span>: Each paper is embedded into a high-dimensional vector space, based on its meaning and content. Here, we plot a low-dimensional visualization of the embedding space. <span className="font-semibold text-gray-500">Gray</span> dots show random <span className="font-mono whitespace-nowrap">astro-ph</span> papers, tracing the astrophysics research landscape. Selected papers, highlighted in <span className="font-semibold text-blue-600">blue</span>, show that similar papers are positioned close together. We can also embed search terms (<span className="font-semibold text-green-600">green</span>) into the same space in order to retrieve relevant papers.
        </p>
      </div>

      <div className="max-w-2xl mx-auto">
        <p className="text-base leading-relaxed">
          By marking papers as{" "}
          <span className="inline-flex items-center px-1.5 py-0.5 rounded text-sm font-medium bg-green-600 text-white">👍 Relevant</span>{" "}
          or{" "}
          <span className="inline-flex items-center px-1.5 py-0.5 rounded text-sm font-medium bg-red-600 text-white">👎 Not Relevant</span>,
          you help train the{" "}
          <span className="font-bold text-blue-700">arXiv Recommender</span>{" "}
          to recognize papers that you are likely to be interested in. Papers from the past day, week or month will be automatically sorted based on your predicted interest in them.
          You can additionally enter specific terms (e.g., "<span className="font-mono">microlensing</span>" or "<span className="font-mono">observations of high-redshift quasars with JWST</span>") into the search bar{" "}
          (<span className="inline-flex items-center justify-center w-7 h-7 rounded bg-gray-100 text-gray-600 align-middle">
            <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="11" cy="11" r="8" />
              <line x1="21" y1="21" x2="16.65" y2="16.65" />
            </svg>
          </span>){" "}
          to find recent papers on any given topic.
        </p>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Tutorial tab content
// ---------------------------------------------------------------------------

function TutorialContent() {
  const { startTour } = useTour();
  const { user } = useAuth();
  return (
    <div className="space-y-8 text-gray-700 max-w-2xl">
      <h1 className="text-2xl font-bold text-gray-900">Tutorial</h1>
      <section className="space-y-2">
        <p className="text-base leading-relaxed">
          This website gives personalised arXiv paper recommendations, based on papers you have
          previously liked or disliked. Every day, each new arXiv paper is processed and
          categorised, and recommendations are updated for each user.
        </p>
        <p className="text-base leading-relaxed">
          Below, you can see some of the key features of the website.
        </p>
      </section>
      {user && (
        <section className="space-y-3">
          <h2 className="text-lg font-semibold text-gray-800">Interactive tour</h2>
          <p className="text-base leading-relaxed">
            Want a quick guided walkthrough? The tour highlights the key features across several pages.
          </p>
          <button
            type="button"
            onClick={startTour}
            className="inline-flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white text-sm font-medium rounded transition-colors"
          >
            Start tour
          </button>
        </section>
      )}

      <section className="space-y-3">
        <h2 className="text-lg font-semibold text-gray-800">Recommendations</h2>
        <p className="text-base leading-relaxed">
          The <Link to="/" className="text-blue-600 hover:underline">Recommendations</Link> page shows
          papers from the last day, week and month, ordered by their predicted relevance to your interests.
        </p>
        <StaticRecsDemo />
      </section>

      <section className="space-y-3">
        <h2 className="text-lg font-semibold text-gray-800">Training the recommendation model</h2>
        <p className="text-base leading-relaxed">
          These recommendations are based on the papers you have previously marked as{" "}
          <span className="inline-flex items-center px-1.5 py-0.5 rounded text-sm font-medium bg-green-600 text-white">👍 relevant</span>{" "}
          or{" "}
          <span className="inline-flex items-center px-1.5 py-0.5 rounded text-sm font-medium bg-red-600 text-white">👎 not relevant</span>.{" "}
          When you first sign up for the site, it has no way of knowing which papers you'll be interested
          in. There are two ways to provide data to the site:
        </p>
        <ol className="list-decimal list-outside space-y-2 text-base leading-relaxed pl-5">
          <li>
            Scroll through the papers on the{" "}
            <Link to="/" className="text-blue-600 hover:underline">Recommendations</Link>{" "}
            page and mark papers as{" "}
            <span className="inline-flex items-center px-1.5 py-0.5 rounded text-sm font-medium bg-green-600 text-white">👍 relevant</span>.
          </li>
          <li>
            On the <Link to="/library" className="text-blue-600 hover:underline">Library</Link> page,
            directly import relevant papers by entering their arXiv IDs.
          </li>
        </ol>
        <StaticPaperDetailDemo />
        <p className="text-base leading-relaxed">
          Once you have liked a few papers, the{" "}
          <Link to="/" className="text-blue-600 hover:underline">Recommendations</Link>{" "}
          page will start showing recommendations. The more papers you mark as relevant or irrelevant, the
          more accurate the recommendations should become.
        </p>
        <p className="text-base leading-relaxed">
          The search feature{" "}
          (<span className="inline-flex items-center justify-center w-6 h-6 rounded bg-gray-100 text-gray-600 align-middle">
            <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="11" cy="11" r="8" />
              <line x1="21" y1="21" x2="16.65" y2="16.65" />
            </svg>
          </span>){" "}
          on the <Link to="/" className="text-blue-600 hover:underline">Recommendations</Link> page allows
          you to search for papers using arbitrary scientific terms, such as "magnetic fields" or "dark
          matter." This can be a useful way of initially finding papers to mark as relevant.
        </p>
      </section>

      <section className="space-y-3">
        <h2 className="text-lg font-semibold text-gray-800">Library</h2>
        <p className="text-base leading-relaxed">
          The <Link to="/library" className="text-blue-600 hover:underline">Library</Link> page allows
          you to see all papers that you have previously marked as relevant or irrelevant, and to import
          papers (by arXiv ID) that you find relevant. You can either add papers one-by-one by arXiv ID,
          or import them in bulk from a list of arXiv IDs.
        </p>
        <StaticLibraryImportDemo />
        <p className="text-base leading-relaxed">
          If you want to import a list of papers from a{" "}
          <a href="https://ui.adsabs.harvard.edu" target="_blank" rel="noreferrer" className="text-blue-600 hover:underline">NASA ADS</a>{" "}
          library, export the library using the custom{" "}
          <code className="font-mono bg-gray-100 px-1 rounded text-sm">%X</code> format string and paste
          the result into the <Link to="/library" className="text-blue-600 hover:underline">Library</Link> page
          import box.
        </p>
        <p className="text-base leading-relaxed">
          Papers are not imported instantly. It takes time for the LLM that powers the site to read and
          categorize each paper. It can take as little as one minute for the site to ingest a new paper.
          However, if the site is currently handling a large volume of papers, this process can take longer.
        </p>
        <p className="text-base leading-relaxed">
          Once your imported papers have been processed, they will be used to help predict which new
          papers you will be interested in. <em><span className="font-medium">Importing your own papers</span> is a particularly effective way of training the recommendation engine.</em>
        </p>
      </section>

      <section className="space-y-3">
        <h2 className="text-lg font-semibold text-gray-800">Group recommendations</h2>
        <p className="text-base leading-relaxed">
          Sometimes, you may want to generate recommendations for a group of researchers. This can be
          useful for organizing arXiv discussions or journal clubs.
        </p>
        <p className="text-base leading-relaxed">
          The <Link to="/groups" className="text-blue-600 hover:underline">Groups</Link> page allows you
          to create user groups that will aggregate recommendations. After creating a group, you can
          generate invite links and QR codes that you can send to other users.
        </p>
        <StaticGroupDemo />
        <p className="text-base leading-relaxed">
          After you join or create a group, you will see that group's name on the{" "}
          <Link to="/" className="text-blue-600 hover:underline">Recommendations</Link> page.
          Clicking on the group name will show group recommendations, again with the option to look at
          papers from the last day, week and month. Clicking on "Personal" will bring you back to your
          individual recommendations.
        </p>
        <StaticRecsDemo groupName="Dust enthusiasts" showGroupMethod papers={DUST_PAPERS} />
        <p className="text-base leading-relaxed">
          There are two ways of aggregating group recommendations, which you can switch between on the{" "}
          <Link to="/" className="text-blue-600 hover:underline">Recommendations</Link> page.
          The "voting" method (which does not involve manual voting) tends to surface papers with the
          largest number of interested group members. The "consensus" method tends to surface papers that
          all group members are interested in.
        </p>
      </section>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Tab definitions
// ---------------------------------------------------------------------------

const TABS = [
  { id: "how-it-works", label: "How it works" },
  { id: "tutorial",     label: "Tutorial" },
] as const;

type TabId = typeof TABS[number]["id"];

function isValidTab(s: string | undefined): s is TabId {
  return s === "how-it-works" || s === "tutorial";
}

// ---------------------------------------------------------------------------
// Page component
// ---------------------------------------------------------------------------

export default function AboutPage() {
  const { tab } = useParams<{ tab: string }>();
  const navigate = useNavigate();

  const activeId: TabId = isValidTab(tab) ? tab : "tutorial";
  const content = activeId === "how-it-works" ? <HowItWorksContent /> : <TutorialContent />;

  return (
    <div className="flex flex-col h-screen bg-gray-50">
      <AppNav />
      <div className="flex-1 relative overflow-hidden min-h-0">

        {/* ── Mobile: tab list (slides out left when a tab is selected) ── */}
        <div
          className={`absolute inset-0 flex flex-col md:hidden bg-gray-50 transition-transform duration-300 ease-in-out ${
            tab ? "-translate-x-full" : "translate-x-0"
          }`}
        >
          <ul className="divide-y divide-gray-200">
            {TABS.map((t) => (
              <li key={t.id}>
                <button
                  onClick={() => navigate(`/about/${t.id}`)}
                  className="w-full flex items-center justify-between px-5 py-4 text-base text-gray-800 hover:bg-gray-100 active:bg-gray-200"
                >
                  {t.label}
                  <svg
                    className="w-4 h-4 text-gray-400 shrink-0"
                    viewBox="0 0 24 24" fill="none" stroke="currentColor"
                    strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"
                  >
                    <polyline points="9,18 15,12 9,6" />
                  </svg>
                </button>
              </li>
            ))}
          </ul>
        </div>

        {/* ── Mobile: content (slides in from right when a tab is selected) ── */}
        <div
          className={`absolute inset-0 flex flex-col md:hidden bg-white transition-transform duration-300 ease-in-out ${
            tab ? "translate-x-0" : "translate-x-full"
          }`}
        >
          <div className="shrink-0 flex items-center px-4 py-3 border-b border-gray-200 bg-white">
            <button
              onClick={() => navigate("/about")}
              className="flex items-center gap-1 text-blue-600 text-sm hover:text-blue-800"
            >
              <svg
                className="w-4 h-4"
                viewBox="0 0 24 24" fill="none" stroke="currentColor"
                strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"
              >
                <polyline points="15,18 9,12 15,6" />
              </svg>
              Back
            </button>
          </div>
          <div className="flex-1 overflow-y-auto px-4 py-6">
            {content}
          </div>
        </div>

        {/* ── Desktop: left tab list + right scrollable content ── */}
        <div className="hidden md:flex h-full">
          <nav className="w-48 border-r border-gray-200 bg-white shrink-0 pt-4">
            {TABS.map((t) => (
              <button
                key={t.id}
                onClick={() => navigate(`/about/${t.id}`)}
                className={`w-full text-left px-4 py-2.5 text-sm transition-colors ${
                  activeId === t.id
                    ? "bg-blue-50 text-blue-700 font-medium border-r-2 border-blue-600"
                    : "text-gray-600 hover:bg-gray-50 hover:text-gray-900"
                }`}
              >
                {t.label}
              </button>
            ))}
          </nav>
          <div className="flex-1 overflow-y-auto">
            <main className="max-w-7xl mx-auto w-full px-6 py-8">
              {content}
            </main>
          </div>
        </div>

      </div>
    </div>
  );
}
