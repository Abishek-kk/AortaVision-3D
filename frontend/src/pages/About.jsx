import {
  Activity,
  BrainCircuit,
  CheckCircle2,
  Code2,
  Database,
  HeartPulse,
  Layers3,
  Network,
  Stethoscope,
  Target,
  Users,
  TriangleAlert,
  Workflow,
} from "lucide-react";
import { useEffect, useRef, useState } from "react";
import HeartGlow from "../components/HeartGlow.jsx";

const objectives = [
  "Predict overall CAD status using clinical data",
  "Predict stenosis probability for LAD, LCX, and RCA",
  "Visualize predictions on an interactive 3D heart model",
  "Provide interpretable explanations (SHAP)",
  "Educate users about heart anatomy and key medical terms",
  "Support clinicians and learners (educational purpose)",
];

const features = [
  "Age, Sex, BMI",
  "Blood Pressure, Pulse Rate",
  "Diabetes, Hypertension, Smoking History",
  "Cholesterol, HDL, LDL, Triglycerides",
  "ECG: ST Elevation, T Inversion, LVH, BBB",
  "Echocardiography: EF-TTE, Region with RWMA, VHD",
];

const processSteps = [
  {
    number: "01",
    title: "Patient Input",
    description: "Clinical, laboratory, ECG and echocardiography features are provided.",
    Icon: Stethoscope,
    color: "about-step-blue",
  },
  {
    number: "02",
    title: "ML Prediction",
    description: "An XGBoost model estimates CAD and vessel-specific stenosis risk.",
    Icon: BrainCircuit,
    color: "about-step-purple",
  },
  {
    number: "03",
    title: "3D Visualization",
    description: "Results are mapped to an interactive 3D heart for exploration.",
    Icon: HeartPulse,
    color: "about-step-red",
  },
  {
    number: "04",
    title: "Explanation",
    description: "SHAP highlights features that influenced each prediction.",
    Icon: Activity,
    color: "about-step-green",
  },
];

const technologies = [
  { name: "React.js", role: "Frontend", Icon: Code2, color: "text-sky-600" },
  { name: "Node.js", role: "Backend API", Icon: Network, color: "text-emerald-600" },
  { name: "Python", role: "ML Model · XGBoost", Icon: BrainCircuit, color: "text-blue-700" },
  { name: "Three.js", role: "3D Visualization", Icon: Layers3, color: "text-violet-600" },
  { name: "SHAP", role: "Explainability", Icon: Activity, color: "text-rose-600" },
  { name: "MongoDB", role: "Database", Icon: Database, color: "text-green-700" },
];

const teamMembers = [
  { name: "Member 1", role: "Frontend & UI", initials: "M1", color: "about-avatar-blue" },
  { name: "Member 2", role: "Backend & ML", initials: "M2", color: "about-avatar-purple" },
  { name: "Member 3", role: "3D Visualization & Documentation", initials: "M3", color: "about-avatar-red" },
];

function Reveal({ children, className = "" }) {
  const ref = useRef(null);
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const element = ref.current;
    if (!element) return undefined;
    if (!("IntersectionObserver" in window)) {
      setVisible(true);
      return undefined;
    }

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setVisible(true);
          observer.unobserve(element);
        }
      },
      { threshold: 0.08 },
    );
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  return (
    <div
      className={`about-reveal ${visible ? "about-reveal-visible" : ""} ${className}`}
      ref={ref}
    >
      {children}
    </div>
  );
}

function SectionHeading({ Icon, children }) {
  return (
    <h2 className="mb-4 flex items-center gap-2.5 text-xl font-extrabold text-navy sm:text-2xl">
      <span className="rounded-xl bg-blue-50 p-2 text-blue-700">
        <Icon aria-hidden="true" className="h-5 w-5" />
      </span>
      {children}
    </h2>
  );
}

export default function About() {
  return (
    <div className="about-page flex-1">
      <section className="about-hero relative isolate overflow-hidden bg-[#04122e] text-white">
        <svg
          aria-hidden="true"
          className="about-ecg pointer-events-none absolute inset-0 h-full w-full opacity-30"
          preserveAspectRatio="none"
          viewBox="0 0 1440 420"
        >
          <defs>
            <linearGradient id="about-ecg-line" x1="0" x2="1">
              <stop stopColor="#38bdf8" stopOpacity="0" />
              <stop offset=".46" stopColor="#38bdf8" />
              <stop offset=".72" stopColor="#fb7185" />
              <stop offset="1" stopColor="#38bdf8" stopOpacity="0" />
            </linearGradient>
          </defs>
          <path
            d="M0 238h280l24-3 19-20 20 42 30-77 30 132 32-88 20 15h255l20-3 17-22 21 36 29-94 29 140 30-82 21 14h400"
            fill="none"
            stroke="url(#about-ecg-line)"
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth="3"
          />
        </svg>
        <div className="relative mx-auto grid w-full max-w-screen-2xl items-center gap-8 px-5 py-10 sm:px-8 sm:py-14 lg:min-h-[340px] lg:grid-cols-[1.35fr_0.9fr] lg:gap-12 lg:px-10">
          <div className="relative z-10">
            <p className="mb-2 text-sm font-bold uppercase tracking-[0.2em] text-cyan">
              Heart health, made clearer
            </p>
            <h1 className="max-w-3xl text-4xl font-extrabold leading-[1.08] sm:text-5xl lg:text-6xl">
              <span className="text-white">About </span>
              <span className="text-cyan">HeartCare AI</span>
            </h1>
            <p className="mt-5 max-w-3xl text-base leading-relaxed text-blue-100 sm:text-lg">
              A machine learning (XGBoost) powered interactive platform to predict
              coronary artery disease and vessel-specific stenosis using patient
              clinical data and 3D heart models.
            </p>
            <p className="mt-5 text-sm font-semibold text-sky-200">
              Developed for Multimodal AI Hackathon 2026 - Track A
            </p>
          </div>

          <div className="relative mx-auto flex w-full max-w-xl items-center justify-center lg:justify-end">
            <div className="about-hero-heart absolute right-[2%] top-1/2 w-36 -translate-y-1/2 opacity-70 sm:right-[5%] sm:w-44 lg:right-[3%] lg:w-52">
              <HeartGlow />
            </div>
            <article className="about-mission-card relative z-10 w-full max-w-md rounded-3xl border border-white/20 bg-white/10 p-5 shadow-2xl backdrop-blur-md sm:p-7">
              <div className="flex items-center gap-3">
                <span className="rounded-2xl bg-cyan/15 p-3 text-cyan">
                  <Target aria-hidden="true" className="h-7 w-7" />
                </span>
                <h2 className="text-2xl font-extrabold">Our Mission</h2>
              </div>
              <p className="mt-4 text-sm leading-relaxed text-blue-50 sm:text-base">
                To make cardiovascular risk prediction more intuitive,
                interpretable, and accessible through machine learning,
                interactive 3D visualization, helping clinicians and patients
                better understand heart health.
              </p>
            </article>
          </div>
        </div>
      </section>

      <section className="about-content px-4 py-7 sm:px-6 sm:py-10 lg:px-8">
        <div className="mx-auto max-w-screen-2xl">
          <div className="about-card-grid grid gap-5 lg:grid-cols-3">
            <Reveal className="about-card">
              <article className="h-full rounded-3xl border border-blue-100 bg-white p-5 shadow-sm sm:p-6">
                <SectionHeading Icon={HeartPulse}>Project Overview</SectionHeading>
                <p className="leading-relaxed text-slate-700">
                  HeartCare AI uses an <strong className="font-extrabold text-blue-700">XGBoost</strong> model
                  to predict overall CAD and vessel-specific stenosis (LAD, LCX,
                  RCA) from demographic, clinical, ECG, laboratory and
                  echocardiographic data. Predictions are mapped to an
                  interactive 3D heart with explainable insights.
                </p>
              </article>
            </Reveal>

            <Reveal className="about-card">
              <article className="h-full rounded-3xl border border-blue-100 bg-white p-5 shadow-sm sm:p-6">
                <SectionHeading Icon={CheckCircle2}>Key Objectives</SectionHeading>
                <ul className="space-y-2.5">
                  {objectives.map((objective) => (
                    <li className="flex items-start gap-2.5 text-sm leading-snug text-slate-700" key={objective}>
                      <CheckCircle2 aria-hidden="true" className="mt-0.5 h-4 w-4 shrink-0 text-blue-600" />
                      <span>{objective}</span>
                    </li>
                  ))}
                </ul>
              </article>
            </Reveal>

            <Reveal className="about-card">
              <article className="h-full rounded-3xl border border-blue-100 bg-white p-5 shadow-sm sm:p-6">
                <SectionHeading Icon={Activity}>Dataset &amp; Features</SectionHeading>
                <p className="mb-4 text-sm leading-relaxed text-slate-700">
                  The model works with a range of de-identified patient features
                  to produce its educational predictions.
                </p>
                <div className="rounded-2xl border border-blue-100 bg-blue-50/80 p-4">
                  <h3 className="mb-3 font-extrabold text-navy">Example Input Features</h3>
                  <ul className="space-y-2 text-sm leading-snug text-slate-700">
                    {features.map((feature) => (
                      <li className="flex gap-2" key={feature}>
                        <span aria-hidden="true" className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-blue-600" />
                        <span>{feature}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              </article>
            </Reveal>

            <Reveal className="about-card">
              <article className="h-full rounded-3xl border border-blue-100 bg-white p-5 shadow-sm sm:p-6">
                <SectionHeading Icon={Workflow}>How It Works</SectionHeading>
                <div className="about-process-grid grid gap-3 sm:grid-cols-2">
                  {processSteps.map(({ number, title, description, Icon, color }) => (
                    <div className={`about-process-step ${color}`} key={number}>
                      <div className="flex items-center justify-between">
                        <span className="text-xs font-extrabold tracking-widest opacity-70">{number}</span>
                        <Icon aria-hidden="true" className="h-5 w-5" />
                      </div>
                      <h3 className="mt-2 font-extrabold">{title}</h3>
                      <p className="mt-1 text-xs leading-relaxed">{description}</p>
                    </div>
                  ))}
                </div>
              </article>
            </Reveal>

            <Reveal className="about-card">
              <article className="h-full rounded-3xl border border-blue-100 bg-white p-5 shadow-sm sm:p-6">
                <SectionHeading Icon={Code2}>Technology Stack</SectionHeading>
                <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
                  {technologies.map(({ name, role, Icon, color }) => (
                    <div className="about-tech-tile" key={name}>
                      <Icon aria-hidden="true" className={`h-6 w-6 ${color}`} />
                      <h3 className="mt-2 text-sm font-extrabold text-navy">{name}</h3>
                      <p className="mt-0.5 text-xs leading-snug text-slate-500">{role}</p>
                    </div>
                  ))}
                </div>
              </article>
            </Reveal>

            <Reveal className="about-card">
              <article className="h-full rounded-3xl border border-blue-100 bg-white p-5 shadow-sm sm:p-6">
                <SectionHeading Icon={Users}>Team</SectionHeading>
                <h3 className="mb-4 text-lg font-extrabold text-blue-800">Team: Accelerate</h3>
                <ul className="space-y-4">
                  {teamMembers.map((member) => (
                    <li className="flex items-center gap-3" key={member.name}>
                      <span aria-hidden="true" className={`about-avatar ${member.color}`}>
                        {member.initials}
                      </span>
                      <span>
                        <span className="block font-bold text-navy">{member.name}</span>
                        <span className="block text-sm text-slate-600">{member.role}</span>
                      </span>
                    </li>
                  ))}
                </ul>
              </article>
            </Reveal>
          </div>

          <Reveal className="about-disclaimer mt-5">
            <aside className="flex items-start gap-4 rounded-3xl border border-amber-200 bg-amber-100 px-5 py-5 text-amber-950 shadow-sm sm:px-7 sm:py-6">
              <TriangleAlert aria-hidden="true" className="mt-0.5 h-8 w-8 shrink-0 text-amber-700 sm:h-10 sm:w-10" />
              <div>
                <h2 className="text-xl font-extrabold sm:text-2xl">Clinical Safety Disclaimer</h2>
                <p className="mt-2 leading-relaxed">
                  This system provides predictions for decision support and
                  educational purposes only and is not a substitute for formal
                  diagnostic imaging or professional medical advice. Always
                  consult a qualified healthcare professional for official
                  diagnosis and treatment.
                </p>
              </div>
            </aside>
          </Reveal>
        </div>
      </section>
    </div>
  );
}
