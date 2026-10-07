import {
  Activity,
  ArrowRight,
  BookOpen,
  CheckCircle2,
  CircleHelp,
  GraduationCap,
  HeartPulse,
  RotateCcw,
  Ruler,
  Search,
  Send,
  Stethoscope,
  X,
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import {
  GLOSSARY_CATEGORIES,
  GLOSSARY_TERMS,
} from "../data/glossary.js";

const anatomyPoints = [
  { id: "aorta", title: "Aorta", left: "8%", top: "5%", side: "left", tags: ["Arteries", "Pulse"], marker: [200, 40] },
  { id: "vena-cava", title: "Vena Cava", left: "91%", top: "15%", side: "right", tags: ["Pulse", "Right"], marker: [270, 83] },
  { id: "pulmonary-artery", title: "Pulmonary Artery", left: "8%", top: "22%", side: "left", tags: ["Arteries", "Pulse"], marker: [139, 89] },
  { id: "cad", title: "Coronary Arteries", left: "91%", top: "31%", side: "right", tags: ["Arteries", "LCX", "RCA"], marker: [242, 149] },
  { id: "left-atrium", title: "Left Atrium", left: "8%", top: "39%", side: "left", tags: ["Left", "Valves", "Mitral"], marker: [162, 153] },
  { id: "right-atrium", title: "Right Atrium", left: "91%", top: "47%", side: "right", tags: ["Right", "Valves"], marker: [239, 157] },
  { id: "left-ventricle", title: "Left Ventricle", left: "91%", top: "63%", side: "right", tags: ["Left", "Measurements"], marker: [237, 210] },
  { id: "right-ventricle", title: "Right Ventricle", left: "8%", top: "68%", side: "left", tags: ["Right", "Measurements"], marker: [160, 210] },
  { id: "aortic-valve", title: "Valves", left: "8%", top: "82%", side: "left", tags: ["Valves", "Mitral"], marker: [199, 151] },
  { id: "lad", title: "LAD", left: "91%", top: "82%", side: "right", tags: ["Arteries"], marker: [181, 159] },
  { id: "lcx", title: "LCX", left: "91%", top: "91%", side: "right", tags: ["Arteries", "LCX"], marker: [246, 169] },
  { id: "rca", title: "RCA", left: "8%", top: "91%", side: "left", tags: ["Arteries", "RCA"], marker: [222, 188] },
  { id: "ecg", title: "ECG", left: "50%", top: "96%", side: "center", tags: ["ECG", "Measurements"], marker: [204, 232] },
];

const highlightChips = [
  "Grid",
  "Arteries",
  "Valves",
  "Pulse",
  "Left",
  "Right",
  "Mitral",
  "LCX",
  "RCA",
  "ECG",
  "Measurements",
];

const categoryStyles = {
  diseases: {
    tab: "bg-rose-100 text-rose-800 ring-rose-300",
    icon: "bg-rose-100 text-rose-700",
  },
  arteries: {
    tab: "bg-orange-100 text-orange-800 ring-orange-300",
    icon: "bg-orange-100 text-orange-700",
  },
  valves: {
    tab: "bg-violet-100 text-violet-800 ring-violet-300",
    icon: "bg-violet-100 text-violet-700",
  },
  measurements: {
    tab: "bg-emerald-100 text-emerald-800 ring-emerald-300",
    icon: "bg-emerald-100 text-emerald-700",
  },
};

const categoryIcons = {
  diseases: HeartPulse,
  arteries: Activity,
  valves: Activity,
  measurements: Ruler,
};

const actionDetails = {
  doctor: {
    title: "Connect to a Doctor",
    intro: "This demo form does not send medical information or book an appointment.",
    success: "Thanks. Your demo request has been recorded on this page.",
  },
  plan: {
    title: "Request a Personalized Educational Plan",
    intro: "Choose a topic and learning level to preview an educational plan request.",
    success: "Your demo educational plan request is ready.",
  },
  question: {
    title: "Ask a Question",
    intro: "Send a learning question for this demo. Do not include private medical details.",
    success: "Your demo question has been submitted.",
  },
};

function termForQuery(value) {
  if (!value) return null;
  const normalized = value.trim().toLowerCase();
  return GLOSSARY_TERMS.find(
    (term) =>
      term.id.toLowerCase() === normalized ||
      term.title.toLowerCase() === normalized ||
      term.title.toLowerCase().startsWith(`${normalized} `) ||
      term.title.toLowerCase().startsWith(`${normalized} (`),
  );
}

function HeartAnatomy({ selectedTerm, onSelect, activeChips, showGrid, slice, opacity }) {
  const selectedHighlights = [...activeChips].filter((chip) => chip !== "Grid");
  const hasFilter = selectedHighlights.length > 0;

  return (
    <div className="guide-anatomy-map relative mx-auto overflow-hidden rounded-3xl bg-gradient-to-b from-sky-50 to-blue-100">
      {showGrid && (
        <div
          aria-hidden="true"
          className="pointer-events-none absolute inset-0 z-0 opacity-40"
          style={{
            backgroundImage:
              "linear-gradient(rgba(30,64,175,.12) 1px, transparent 1px), linear-gradient(90deg, rgba(30,64,175,.12) 1px, transparent 1px)",
            backgroundSize: "24px 24px",
          }}
        />
      )}
      <svg
        aria-hidden="true"
        className="absolute inset-0 z-[1] h-full w-full"
        preserveAspectRatio="none"
        viewBox="0 0 400 300"
      >
        <ellipse cx="200" cy="270" fill="#9bbada" opacity=".34" rx="118" ry="18" />
        <ellipse cx="200" cy="264" fill="#d7e8f7" stroke="#9ebcda" strokeWidth="2" rx="108" ry="15" />
        <g
          opacity={opacity}
          style={{
            clipPath: `inset(${slice}% 0 0 0)`,
            transition: "clip-path 180ms ease, opacity 180ms ease",
          }}
        >
          <path d="M194 111 C169 73 127 77 112 111 C91 157 120 203 191 251 C197 255 203 255 210 250 C278 205 313 155 287 111 C267 76 228 77 204 111Z" fill="#e84f6c" stroke="#c52d51" strokeWidth="5" />
          <path d="M198 133 C165 113 137 120 133 151 C130 178 154 196 193 218Z" fill="#f87989" stroke="#ffe4e8" strokeWidth="3" />
          <path d="M206 133 C238 111 266 121 270 151 C273 177 248 199 207 220Z" fill="#d83b5b" stroke="#ffe4e8" strokeWidth="3" />
          <path d="M193 214 C163 190 143 187 139 209 C137 224 160 239 194 257Z" fill="#e5a0aa" stroke="#ffe4e8" strokeWidth="3" />
          <path d="M208 214 C237 190 257 188 263 210 C267 226 242 243 207 257Z" fill="#c52d51" stroke="#ffe4e8" strokeWidth="3" />
          <path d="M197 129 C192 94 194 64 191 35 M207 128 C216 96 224 77 232 50 M190 133 C165 118 149 103 133 89" fill="none" stroke="#e95052" strokeLinecap="round" strokeWidth="15" />
          <path d="M197 128 C191 100 190 74 190 47 M207 128 C215 102 223 81 230 59" fill="none" stroke="#fff1f2" strokeLinecap="round" strokeOpacity=".56" strokeWidth="3" />
          <path d="M190 151 C176 154 167 167 158 181 M209 153 C224 159 232 171 241 183 M195 157 C199 178 199 194 199 211" fill="none" stroke="#ffdd67" strokeLinecap="round" strokeWidth="5" />
          <path d="M202 145 C225 134 243 137 259 149" fill="none" stroke="#7c2d12" strokeLinecap="round" strokeWidth="4" />
        </g>
        {anatomyPoints.map((point, index) => {
          const [x, y] = point.marker;
          const targetX = point.side === "left" ? 70 : point.side === "right" ? 330 : 200;
          const targetY = Number.parseFloat(point.top) * 3;
          const visible = !hasFilter || point.tags.some((tag) => selectedHighlights.includes(tag));
          return (
            <g key={`${point.title}-${index}`} opacity={visible ? 1 : 0.15}>
              <path
                d={`M ${x} ${y} C ${(x + targetX) / 2} ${y}, ${(x + targetX) / 2} ${targetY}, ${targetX} ${targetY}`}
                fill="none"
                stroke="#3b82f6"
                strokeOpacity=".65"
                strokeWidth="1.4"
              />
              <circle cx={x} cy={y} fill="#fff" r="5" />
              <circle cx={x} cy={y} fill="#0ea5e9" r="3" />
            </g>
          );
        })}
      </svg>

      {anatomyPoints.map((point, index) => {
        const term = GLOSSARY_TERMS.find((item) => item.id === point.id);
        const active = selectedTerm.id === point.id;
        const visible = !hasFilter || point.tags.some((tag) => selectedHighlights.includes(tag));
        return (
          <button
            key={`${point.title}-${index}`}
            aria-label={`Select ${point.title}`}
            aria-pressed={active}
            className={`absolute z-[2] max-w-[46%] -translate-y-1/2 rounded-full border px-2 py-1 text-[10px] font-bold leading-tight shadow-sm transition sm:px-2.5 sm:text-xs ${
              point.side === "left" ? "-translate-x-0" : point.side === "right" ? "-translate-x-full" : "-translate-x-1/2"
            } ${
              active
                ? "border-blue-600 bg-blue-700 text-white shadow-blue-300"
                : "border-white bg-white/95 text-navy hover:border-blue-400 hover:bg-blue-50"
            }`}
            onClick={() => onSelect(term ?? GLOSSARY_TERMS[0])}
            style={{
              left: point.left,
              opacity: visible ? 1 : 0.24,
              top: point.top,
            }}
            type="button"
          >
            {point.title}
          </button>
        );
      })}
    </div>
  );
}

function ArteryComparison() {
  return (
    <div className="grid grid-cols-2 gap-3" role="img" aria-label="Comparison of a normal artery and an artery narrowed by plaque">
      <div className="rounded-xl bg-sky-50 p-3 text-center">
        <svg aria-hidden="true" className="mx-auto h-20 w-full" viewBox="0 0 180 80">
          <path d="M8 24 Q90 7 172 24 L172 56 Q90 73 8 56Z" fill="#e55368" />
          <path d="M8 34 Q90 21 172 34 L172 47 Q90 60 8 47Z" fill="#fda4af" />
          <path d="M13 40h154" stroke="#fef2f2" strokeWidth="5" />
        </svg>
        <p className="mt-1 text-xs font-bold text-navy">Normal artery</p>
      </div>
      <div className="rounded-xl bg-rose-50 p-3 text-center">
        <svg aria-hidden="true" className="mx-auto h-20 w-full" viewBox="0 0 180 80">
          <path d="M8 24 Q90 7 172 24 L172 56 Q90 73 8 56Z" fill="#e55368" />
          <path d="M8 34 Q90 21 172 34 L172 47 Q90 60 8 47Z" fill="#fda4af" />
          <path d="M13 40h47 Q73 40 84 36 Q94 29 102 36 Q112 41 123 40h44" fill="none" stroke="#fef2f2" strokeWidth="5" />
          <path d="M71 25 Q90 32 107 25 L111 49 Q91 57 68 49Z" fill="#facc15" stroke="#eab308" strokeWidth="2" />
        </svg>
        <p className="mt-1 text-xs font-bold text-navy">Plaque narrowing</p>
      </div>
    </div>
  );
}

function ContactModal({ action, onClose }) {
  const [submitted, setSubmitted] = useState(false);
  const dialogRef = useRef(null);
  const detail = actionDetails[action];

  useEffect(() => {
    const firstControl = dialogRef.current?.querySelector(
      "input, select, textarea, button",
    );
    firstControl?.focus();

    function handleEscape(event) {
      if (event.key === "Escape") {
        onClose();
        return;
      }
      if (event.key !== "Tab") return;

      const controls = dialogRef.current?.querySelectorAll(
        "input:not([disabled]), select:not([disabled]), textarea:not([disabled]), button:not([disabled])",
      );
      if (!controls?.length) return;
      const first = controls[0];
      const last = controls[controls.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    }
    window.addEventListener("keydown", handleEscape);
    return () => window.removeEventListener("keydown", handleEscape);
  }, [onClose]);

  function handleSubmit(event) {
    event.preventDefault();
    setSubmitted(true);
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/50 p-4"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <section
        aria-labelledby="guide-modal-title"
        aria-describedby="guide-modal-description"
        aria-modal="true"
        className="w-full max-w-lg rounded-3xl bg-white p-6 text-navy shadow-2xl sm:p-8"
        ref={dialogRef}
        role="dialog"
      >
        <div className="mb-4 flex items-start justify-between gap-4">
          <div>
            <h2 className="text-xl font-extrabold" id="guide-modal-title">{detail.title}</h2>
            <p className="mt-1 text-sm text-slate-600" id="guide-modal-description">{detail.intro}</p>
          </div>
          <button
            aria-label="Close dialog"
            className="rounded-full p-2 text-slate-500 hover:bg-slate-100"
            onClick={onClose}
            type="button"
          >
            <X aria-hidden="true" className="h-5 w-5" />
          </button>
        </div>

        {submitted ? (
          <div aria-live="polite" className="flex items-start gap-3 rounded-2xl bg-emerald-50 p-4 text-emerald-900">
            <CheckCircle2 aria-hidden="true" className="mt-0.5 h-5 w-5 shrink-0" />
            <p className="font-semibold">{detail.success}</p>
          </div>
        ) : (
          <form className="space-y-4" onSubmit={handleSubmit}>
            {action === "doctor" && (
              <>
                <label className="block text-sm font-bold">
                  Name
                  <input className="mt-1 w-full rounded-xl border border-slate-300 px-3 py-2 font-normal focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-200" name="name" required />
                </label>
                <label className="block text-sm font-bold">
                  Email
                  <input className="mt-1 w-full rounded-xl border border-slate-300 px-3 py-2 font-normal focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-200" name="email" required type="email" />
                </label>
                <label className="block text-sm font-bold">
                  General topic
                  <input className="mt-1 w-full rounded-xl border border-slate-300 px-3 py-2 font-normal focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-200" name="topic" placeholder="For example: understanding ECG results" required />
                </label>
              </>
            )}
            {action === "plan" && (
              <>
                <label className="block text-sm font-bold">
                  Topic
                  <input className="mt-1 w-full rounded-xl border border-slate-300 px-3 py-2 font-normal focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-200" defaultValue="Heart health basics" name="topic" required />
                </label>
                <label className="block text-sm font-bold">
                  Learning level
                  <select className="mt-1 w-full rounded-xl border border-slate-300 bg-white px-3 py-2 font-normal focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-200" defaultValue="beginner" name="level">
                    <option value="beginner">Beginner</option>
                    <option value="intermediate">Intermediate</option>
                    <option value="advanced">Advanced</option>
                  </select>
                </label>
              </>
            )}
            {action === "question" && (
              <label className="block text-sm font-bold">
                Your question
                <textarea className="mt-1 min-h-28 w-full resize-y rounded-xl border border-slate-300 px-3 py-2 font-normal focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-200" name="question" required />
              </label>
            )}
            <button className="inline-flex items-center gap-2 rounded-full bg-blue-700 px-5 py-2.5 font-bold text-white transition hover:bg-blue-800 focus-visible:outline-offset-2" type="submit">
              <Send aria-hidden="true" className="h-4 w-4" />
              Submit demo form
            </button>
          </form>
        )}
      </section>
    </div>
  );
}

function VisualReference({ term }) {
  if (term.id === "stenosis" || term.id === "cad") {
    return (
      <section className="guide-visual-reference mt-5">
        <h3 className="mb-2 text-xs font-extrabold uppercase tracking-wider text-slate-500">Visual Reference</h3>
        <ArteryComparison />
      </section>
    );
  }
  if (["lad", "lcx", "rca", "aorta", "pulmonary-artery"].includes(term.id)) {
    return (
      <section className="guide-visual-reference mt-5">
        <h3 className="mb-2 text-xs font-extrabold uppercase tracking-wider text-slate-500">Visual Reference</h3>
        <div className="flex items-center gap-3 rounded-2xl bg-orange-50 p-4 text-orange-950">
          <Activity aria-hidden="true" className="h-8 w-8 shrink-0 text-orange-600" />
          <p className="text-sm">Arteries carry blood away from the heart. Coronary arteries supply the heart muscle itself.</p>
        </div>
      </section>
    );
  }
  if (term.category === "valves") {
    return (
      <section className="guide-visual-reference mt-5">
        <h3 className="mb-2 text-xs font-extrabold uppercase tracking-wider text-slate-500">Visual Reference</h3>
        <div className="flex items-center gap-3 rounded-2xl bg-violet-50 p-4 text-violet-950">
          <Activity aria-hidden="true" className="h-8 w-8 shrink-0 text-violet-600" />
          <p className="text-sm">Heart valves open and close to keep blood moving forward through the chambers.</p>
        </div>
      </section>
    );
  }
  return null;
}

export default function Guide() {
  const [searchParams, setSearchParams] = useSearchParams();
  const requestedTerm = searchParams.get("term");
  const initialTerm = termForQuery(requestedTerm) ?? GLOSSARY_TERMS.find((term) => term.id === "stenosis");
  const [selectedTermId, setSelectedTermId] = useState(initialTerm.id);
  const [activeCategory, setActiveCategory] = useState("diseases");
  const [search, setSearch] = useState("");
  const [activeChips, setActiveChips] = useState(() => new Set());
  const [slice, setSlice] = useState(30);
  const [opacity, setOpacity] = useState(1);
  const [modal, setModal] = useState(null);

  useEffect(() => {
    const linkedTerm = termForQuery(requestedTerm);
    if (linkedTerm) setSelectedTermId(linkedTerm.id);
  }, [requestedTerm]);

  const selectedTerm =
    GLOSSARY_TERMS.find((term) => term.id === selectedTermId) ?? GLOSSARY_TERMS[0];
  const filteredTerms = useMemo(() => {
    const query = search.trim().toLowerCase();
    return GLOSSARY_TERMS.filter((term) => {
      if (term.category !== activeCategory) return false;
      if (!query) return true;
      return [term.title, term.short, term.definition, term.context, ...term.relatedConditions]
        .join(" ")
        .toLowerCase()
        .includes(query);
    });
  }, [activeCategory, search]);

  function selectTerm(term) {
    setSelectedTermId(term.id);
    const nextParams = new URLSearchParams(searchParams);
    nextParams.set("term", term.id);
    setSearchParams(nextParams, { replace: true });
  }

  function toggleChip(chip) {
    setActiveChips((current) => {
      const next = new Set(current);
      if (next.has(chip)) next.delete(chip);
      else next.add(chip);
      return next;
    });
  }

  function resetCrossSection() {
    setSlice(30);
    setOpacity(1);
  }

  return (
    <section className="guide-page page-light flex-1 px-4 py-5 sm:px-6 lg:px-8">
      <div className="mx-auto w-full max-w-screen-2xl">
        <header className="guide-page-header mb-5 flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <h1 className="text-2xl font-extrabold leading-tight text-navy sm:text-3xl lg:text-4xl">
              Cardiac Educational Guide &amp; Medical Glossary
            </h1>
            <p className="mt-2 max-w-3xl text-sm text-slate-600 sm:text-base">
              Learn about heart anatomy, key medical terms, and understand your AI prediction results.
            </p>
          </div>
          <aside className="flex max-w-sm items-center gap-3 rounded-2xl border border-blue-100 bg-white px-4 py-3 shadow-sm">
            <span className="rounded-xl bg-blue-50 p-2.5 text-blue-700">
              <GraduationCap aria-hidden="true" className="h-6 w-6" />
            </span>
            <div>
              <p className="font-extrabold text-navy">Educational Content</p>
              <p className="text-xs leading-snug text-slate-600">
                Interactive learning about heart health and cardiovascular terms
              </p>
            </div>
          </aside>
        </header>

        <div className="guide-layout grid items-start gap-4">
          <div className="guide-column guide-left-column space-y-4">
            <section className="guide-card rounded-3xl border border-blue-100 bg-white p-4 shadow-sm sm:p-5">
              <h2 className="guide-section-title mb-3 text-lg font-extrabold text-navy">Anatomical Explorer &amp; Term Highlights</h2>
              <HeartAnatomy
                activeChips={activeChips}
                onSelect={selectTerm}
                selectedTerm={selectedTerm}
                showGrid={activeChips.has("Grid")}
                slice={slice}
                opacity={opacity}
              />
              <p className="mt-2 text-xs text-slate-500">Select a label to explore the related term.</p>
            </section>

            <details className="guide-more-controls">
              <summary className="guide-more-summary">More anatomy controls</summary>
              <div className="guide-more-grid">
                <section className="guide-card rounded-3xl border border-blue-100 bg-white p-4 shadow-sm">
                  <h2 className="font-extrabold text-navy">Vessel Highlight</h2>
                  <p className="mt-1 text-xs text-slate-600">Choose one or more highlights on the anatomy map.</p>
                  <div className="mt-3 flex flex-wrap gap-2">
                    {highlightChips.map((chip) => (
                      <button
                        key={chip}
                        aria-pressed={activeChips.has(chip)}
                        className={`guide-chip rounded-full border px-3 py-1.5 text-xs font-bold transition focus-visible:outline focus-visible:outline-2 focus-visible:outline-blue-500 ${
                          activeChips.has(chip)
                            ? "border-blue-700 bg-blue-700 text-white"
                            : "border-slate-200 bg-slate-50 text-slate-700 hover:border-blue-300 hover:bg-blue-50"
                        }`}
                        onClick={() => toggleChip(chip)}
                        type="button"
                      >
                        {chip}
                      </button>
                    ))}
                  </div>
                </section>

                <section className="guide-card rounded-3xl border border-blue-100 bg-white p-4 shadow-sm">
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <h2 className="font-extrabold text-navy">Cross-Section View</h2>
                      <p className="mt-1 text-xs leading-relaxed text-slate-600">Adjust the heart layer to explore a simplified cross-section.</p>
                    </div>
                    <button
                      aria-label="Reset cross-section controls"
                      className="rounded-lg p-2 text-blue-700 hover:bg-blue-50"
                      onClick={resetCrossSection}
                      type="button"
                    >
                      <RotateCcw aria-hidden="true" className="h-4 w-4" />
                    </button>
                  </div>
                  <label className="mt-3 block text-xs font-bold text-slate-700">
                    <span className="flex justify-between"><span>Slice</span><span>{slice}</span></span>
                    <input aria-label="Slice" className="mt-1 w-full accent-blue-700" max="100" min="0" onChange={(event) => setSlice(Number(event.target.value))} type="range" value={slice} />
                  </label>
                  <label className="mt-3 block text-xs font-bold text-slate-700">
                    <span className="flex justify-between"><span>Opacity</span><span>{opacity.toFixed(2)}</span></span>
                    <input aria-label="Opacity" className="mt-1 w-full accent-blue-700" max="1" min="0" onChange={(event) => setOpacity(Number(event.target.value))} step="0.01" type="range" value={opacity} />
                  </label>
                </section>
              </div>
            </details>
          </div>

          <section className="guide-card guide-glossary rounded-3xl border border-blue-100 bg-white p-4 shadow-sm sm:p-5">
            <div className="flex items-center gap-2">
              <Activity aria-hidden="true" className="h-5 w-5 text-blue-700" />
              <h2 className="text-lg font-extrabold text-navy">Dynamic Term Explorer</h2>
            </div>
            <div aria-label="Glossary categories" className="mt-4 flex flex-wrap gap-2" role="tablist">
              {GLOSSARY_CATEGORIES.map((category) => {
                const Icon = categoryIcons[category.id];
                const active = activeCategory === category.id;
                return (
                  <button
                    key={category.id}
                    aria-selected={active}
                    className={`inline-flex items-center gap-1.5 rounded-full px-3 py-2 text-xs font-extrabold transition focus-visible:outline focus-visible:outline-2 focus-visible:outline-blue-600 sm:text-sm ${
                      active
                        ? `ring-2 ${categoryStyles[category.id].tab}`
                        : "bg-slate-100 text-slate-600 hover:bg-slate-200"
                    }`}
                    onClick={() => setActiveCategory(category.id)}
                    role="tab"
                    type="button"
                  >
                    <Icon aria-hidden="true" className="h-4 w-4" />
                    {category.label}
                  </button>
                );
              })}
            </div>
            <label className="mt-4 flex items-center gap-2 rounded-xl border border-slate-200 bg-slate-50 px-3 py-2.5 focus-within:border-blue-400 focus-within:ring-2 focus-within:ring-blue-100">
              <Search aria-hidden="true" className="h-4 w-4 shrink-0 text-slate-400" />
              <span className="sr-only">Search glossary terms</span>
              <input
                aria-label="Search glossary terms"
                className="min-w-0 flex-1 bg-transparent text-sm text-slate-800 outline-none placeholder:text-slate-400"
                onChange={(event) => setSearch(event.target.value)}
                placeholder="Search terms or descriptions..."
                type="search"
                value={search}
              />
            </label>

            {filteredTerms.length > 0 ? (
              <div className="mt-4 grid gap-3 sm:grid-cols-2">
                {filteredTerms.map((term) => {
                  const Icon = categoryIcons[term.category];
                  const active = selectedTerm.id === term.id;
                  return (
                    <button
                      key={term.id}
                      aria-label={`Select ${term.title}`}
                      aria-pressed={active}
                      className={`guide-term-card flex min-h-32 flex-col items-start rounded-2xl border p-3 text-left transition hover:-translate-y-0.5 hover:shadow-md focus-visible:outline focus-visible:outline-2 focus-visible:outline-blue-600 ${
                        active ? "border-blue-500 bg-blue-50 ring-2 ring-blue-100" : "border-slate-200 bg-white hover:border-blue-200"
                      }`}
                      onClick={() => selectTerm(term)}
                      type="button"
                    >
                      <span className={`mb-2 rounded-lg p-2 ${categoryStyles[term.category].icon}`}>
                        <Icon aria-hidden="true" className="h-4 w-4" />
                      </span>
                      <span className="font-extrabold leading-snug text-navy">{term.title}</span>
                      <span className="mt-1 text-xs leading-relaxed text-slate-600">{term.short}</span>
                    </button>
                  );
                })}
              </div>
            ) : (
              <p aria-live="polite" className="mt-6 rounded-xl bg-slate-50 p-4 text-sm text-slate-600">
                No terms in this category match “{search}”.
              </p>
            )}
          </section>

          <aside className="guide-card guide-detail-panel rounded-3xl border border-blue-100 bg-white p-4 shadow-sm sm:p-5">
            <div className="flex items-center gap-2">
              <BookOpen aria-hidden="true" className="h-5 w-5 text-blue-700" />
              <h2 className="text-lg font-extrabold text-navy">Detailed Term &amp; Context Panel</h2>
            </div>
            <p className="guide-term-title mt-4 text-2xl font-extrabold leading-tight text-blue-800">{selectedTerm.title}</p>
            <p className="mt-2 text-sm font-semibold leading-relaxed text-slate-600">{selectedTerm.short}</p>
            <section className="mt-5">
              <h3 className="text-xs font-extrabold uppercase tracking-wider text-slate-500">Definition</h3>
              <p className="mt-1 text-sm leading-relaxed text-slate-700">{selectedTerm.definition}</p>
            </section>
            <section className="mt-4">
              <h3 className="text-xs font-extrabold uppercase tracking-wider text-slate-500">Context</h3>
              <p className="mt-1 text-sm leading-relaxed text-slate-700">{selectedTerm.context}</p>
            </section>
            <VisualReference term={selectedTerm} />
            <section className="mt-5">
              <h3 className="text-xs font-extrabold uppercase tracking-wider text-slate-500">Related Conditions</h3>
              <div className="mt-2 flex flex-wrap gap-2">
                {selectedTerm.relatedConditions.map((condition) => (
                  <span key={condition} className="rounded-full bg-blue-50 px-3 py-1 text-xs font-semibold text-blue-800">
                    {condition}
                  </span>
                ))}
              </div>
            </section>
            <div className="guide-actions mt-6 grid gap-2">
              <Link
                className="inline-flex items-center justify-center gap-2 rounded-full bg-blue px-4 py-2.5 text-center text-sm font-bold text-white transition hover:bg-navy focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-blue"
                to="/analyze"
              >
                Analyze Specific Case <ArrowRight aria-hidden="true" className="h-4 w-4" />
              </Link>
              <button className="inline-flex items-center justify-center gap-2 rounded-full border border-blue-200 bg-white px-4 py-2.5 text-sm font-bold text-blue-800 transition hover:bg-blue-50" onClick={() => setModal("doctor")} type="button">
                <Stethoscope aria-hidden="true" className="h-4 w-4" /> Connect to a Doctor
              </button>
              <button className="inline-flex items-center justify-center gap-2 rounded-full border border-blue-200 bg-white px-4 py-2.5 text-sm font-bold text-blue-800 transition hover:bg-blue-50" onClick={() => setModal("plan")} type="button">
                <BookOpen aria-hidden="true" className="h-4 w-4" /> Request Personalized Educational Plan
              </button>
              <button className="inline-flex items-center justify-center gap-2 rounded-full border border-blue-200 bg-white px-4 py-2.5 text-sm font-bold text-blue-800 transition hover:bg-blue-50" onClick={() => setModal("question")} type="button">
                <CircleHelp aria-hidden="true" className="h-4 w-4" /> Ask a Question
              </button>
            </div>
          </aside>
        </div>
        <p className="mt-5 text-center text-xs text-slate-600">
          *Consult a medical professional for official diagnosis and advice.*
        </p>
      </div>
      {modal && <ContactModal action={modal} onClose={() => setModal(null)} />}
    </section>
  );
}
