import { Suspense, Component, useEffect, useMemo, useRef, useState } from "react";
import {
  ArrowDown,
  ArrowLeft,
  ArrowRight,
  ArrowUp,
  Download,
  Info,
  LoaderCircle,
  Maximize2,
  Minimize2,
  Pause,
  Play,
  RotateCcw,
  ZoomIn,
  ZoomOut,
} from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { Html, OrbitControls, useGLTF } from "@react-three/drei";
import { Link } from "react-router-dom";
import { CatmullRomCurve3, MathUtils, MeshStandardMaterial, TubeGeometry, Vector3 } from "three";
import GlowButton from "../components/GlowButton.jsx";
import GlowCard from "../components/GlowCard.jsx";
import HeartGlow from "../components/HeartGlow.jsx";
import { usePatientData } from "../context/PatientDataContext.jsx";
import { SAMPLE_PATIENT } from "../data/samplePatient.js";
import { predictRisk } from "../services/predict.js";
import { getRiskColor, getRiskLevel } from "../utils/risk.js";

const VESSEL_DETAILS = {
  LAD: {
    fullName: "Left Anterior Descending",
    position: [0.08, -0.15, 0.72],
  },
  LCX: {
    fullName: "Left Circumflex",
    position: [-0.62, 0.22, 0.32],
  },
  RCA: {
    fullName: "Right Coronary Artery",
    position: [0.68, -0.15, 0.38],
  },
};

function riskName(value) {
  if (value < 50) return "Green";
  if (value <= 70) return "Yellow";
  return "Red";
}

function Gauge({ probability }) {
  const target = Math.min(1, Math.max(0, probability));
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    setProgress(0);
    let frame;
    let start;
    const animate = (time) => {
      if (start === undefined) start = time;
      const amount = Math.min(1, (time - start) / 1000);
      const eased = 1 - (1 - amount) ** 3;
      setProgress(target * eased);
      if (amount < 1) frame = window.requestAnimationFrame(animate);
    };
    frame = window.requestAnimationFrame(animate);
    return () => window.cancelAnimationFrame(frame);
  }, [target]);

  const angle = Math.PI * (1 - progress);
  const endpoint = {
    x: 120 + 78 * Math.cos(angle),
    y: 120 - 78 * Math.sin(angle),
  };
  const score = Math.round(target * 100);

  return (
    <div className="mx-auto w-full max-w-[185px]">
      <svg
        aria-label={`CAD probability gauge: ${score}%`}
        className="mx-auto block w-full"
        role="img"
        viewBox="0 0 240 145"
      >
        <defs>
          <linearGradient id="cad-gauge-gradient" x1="30" x2="210" y1="120" y2="120" gradientUnits="userSpaceOnUse">
            <stop stopColor="#FF3B5C" />
            <stop offset=".52" stopColor="#FACC15" />
            <stop offset="1" stopColor="#64748B" />
          </linearGradient>
        </defs>
        <path
          d="M30 120a90 90 0 0 1 180 0"
          fill="none"
          stroke="url(#cad-gauge-gradient)"
          strokeLinecap="round"
          strokeWidth="17"
        />
        <path d="M44 120a76 76 0 0 1 152 0" fill="none" stroke="#fff" strokeOpacity=".08" strokeWidth="1" />
        <line
          stroke="#F8FAFC"
          strokeLinecap="round"
          strokeWidth="4"
          x1="120"
          x2={endpoint.x}
          y1="120"
          y2={endpoint.y}
        />
        <circle cx="120" cy="120" fill="#F8FAFC" r="8" />
        <circle cx="120" cy="120" fill="#FF3B5C" r="3.5" />
        <text fill="#CBD5E1" fontSize="11" textAnchor="middle" x="32" y="141">
          0%
        </text>
        <text fill="#CBD5E1" fontSize="11" textAnchor="middle" x="208" y="141">
          100%
        </text>
        <text fill="#fff" fontSize="25" fontWeight="700" textAnchor="middle" x="120" y="101">
          {score}%
        </text>
      </svg>
    </div>
  );
}

function VesselRows({ vessels, selectedVessel, onHover, onPin }) {
  return (
    <div className="mt-3 space-y-1.5">
      {Object.entries(vessels).map(([name, value]) => {
        const color = getRiskColor(value);
        const selected = selectedVessel === name;
        return (
          <button
            key={name}
            aria-pressed={selected}
            className={`flex w-full items-center gap-3 rounded-xl border px-3 py-1.5 text-left transition duration-200 hover:brightness-125 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 ${
              selected ? "ring-2 ring-cyan" : ""
            }`}
            onBlur={() => onHover(null)}
            onClick={() => onPin(name)}
            onFocus={() => onHover(name)}
            onMouseEnter={() => onHover(name)}
            onMouseLeave={() => onHover(null)}
            style={{
              borderColor: `${color}85`,
              backgroundColor: `${color}18`,
              boxShadow: `0 0 14px ${color}25`,
            }}
            type="button"
          >
            <span
              aria-hidden="true"
              className="h-3 w-3 shrink-0 rounded-full"
              style={{ backgroundColor: color, boxShadow: `0 0 10px ${color}` }}
            />
            <span className="font-bold text-white">{name}</span>
            <span className="ml-auto text-sm font-semibold text-slate-200">
              {value}% - {riskName(value)}
            </span>
          </button>
        );
      })}
    </div>
  );
}

function Artery({ points, color = "#fb6474", radius = 0.024 }) {
  const geometry = useMemo(() => {
    const curve = new CatmullRomCurve3(points.map((point) => new Vector3(...point)));
    return new TubeGeometry(curve, 48, radius, 8, false);
  }, [points, radius]);
  return (
    <mesh geometry={geometry}>
      <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.7} roughness={0.32} />
    </mesh>
  );
}

function HeartHotspot({ name, value, active, onHover, onLeave, onPin }) {
  const color = getRiskColor(value);
  const [x, y, z] = VESSEL_DETAILS[name].position;

  return (
    <group position={[x, y, z]}>
      <mesh
        onClick={(event) => {
          event.stopPropagation();
          onPin(name);
        }}
        onPointerOut={(event) => {
          event.stopPropagation();
          onLeave();
        }}
        onPointerOver={(event) => {
          event.stopPropagation();
          onHover(name);
        }}
      >
        <sphereGeometry args={[0.11, 20, 20]} />
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={active ? 2.5 : 1.4}
          roughness={0.2}
        />
      </mesh>
      <mesh raycast={() => null} scale={active ? 1.65 : 1.35}>
        <sphereGeometry args={[0.11, 16, 16]} />
        <meshBasicMaterial color={color} transparent opacity={0.18} />
      </mesh>
      <Html center style={{ pointerEvents: "none" }}>
        <span className="whitespace-nowrap rounded-full border border-white/20 bg-deepNavy/80 px-2 py-0.5 text-[10px] font-bold text-white shadow-lg">
          {name}
        </span>
      </Html>
    </group>
  );
}

function HeartHotspots({ vessels, activeVessel, onHover, onLeave, onPin }) {
  return Object.entries(vessels).map(([name, value]) => (
    <HeartHotspot
      key={name}
      active={activeVessel === name}
      name={name}
      onHover={onHover}
      onLeave={onLeave}
      onPin={onPin}
      value={value}
    />
  ));
}

function ProceduralHeart({ vessels, activeVessel, onHover, onLeave, onPin }) {
  return (
    <group>
      <mesh position={[0, -0.04, 0]} scale={[0.83, 1.08, 0.55]}>
        <sphereGeometry args={[1, 48, 48]} />
        <meshStandardMaterial
          color="#bc234f"
          emissive="#631129"
          emissiveIntensity={0.6}
          metalness={0.18}
          roughness={0.32}
        />
      </mesh>
      <mesh position={[-0.38, 0.48, 0.12]} scale={[0.48, 0.59, 0.49]}>
        <sphereGeometry args={[1, 36, 36]} />
        <meshStandardMaterial color="#e94b65" emissive="#59152b" emissiveIntensity={0.45} roughness={0.28} />
      </mesh>
      <mesh position={[0.36, 0.51, 0.08]} scale={[0.5, 0.57, 0.48]}>
        <sphereGeometry args={[1, 36, 36]} />
        <meshStandardMaterial color="#dd405f" emissive="#641329" emissiveIntensity={0.45} roughness={0.3} />
      </mesh>
      <mesh position={[-0.4, -0.46, 0.1]} scale={[0.37, 0.63, 0.45]} rotation={[0, 0, -0.16]}>
        <sphereGeometry args={[1, 36, 36]} />
        <meshStandardMaterial color="#d94758" emissive="#681326" emissiveIntensity={0.52} roughness={0.29} />
      </mesh>
      <mesh position={[0.34, -0.43, 0.17]} scale={[0.42, 0.7, 0.48]} rotation={[0, 0, 0.16]}>
        <sphereGeometry args={[1, 36, 36]} />
        <meshStandardMaterial color="#b72250" emissive="#4b1230" emissiveIntensity={0.58} roughness={0.28} />
      </mesh>

      <Artery
        color="#ff735e"
        points={[
          [0.12, 0.75, 0.49],
          [0.12, 0.35, 0.65],
          [0.26, 0.15, 0.68],
          [0.27, -0.2, 0.61],
          [0.38, -0.56, 0.46],
        ]}
        radius={0.032}
      />
      <Artery
        color="#fd9b50"
        points={[
          [0.12, 0.35, 0.65],
          [-0.16, 0.33, 0.65],
          [-0.43, 0.19, 0.65],
          [-0.58, -0.01, 0.55],
        ]}
        radius={0.026}
      />
      <Artery
        color="#ff5f71"
        points={[
          [0.26, 0.14, 0.68],
          [0.51, 0.04, 0.57],
          [0.59, -0.17, 0.54],
          [0.63, -0.4, 0.52],
        ]}
        radius={0.025}
      />
      <mesh position={[0.02, 1, -0.03]} rotation={[0.13, 0.05, -0.08]}>
        <cylinderGeometry args={[0.21, 0.18, 0.9, 24]} />
        <meshStandardMaterial color="#3787b3" emissive="#133f6a" emissiveIntensity={0.5} />
      </mesh>
      <mesh position={[-0.38, 0.95, 0]}>
        <cylinderGeometry args={[0.16, 0.13, 0.7, 24]} />
        <meshStandardMaterial color="#db5266" emissive="#51182b" emissiveIntensity={0.4} />
      </mesh>

      <HeartHotspots
        activeVessel={activeVessel}
        onHover={onHover}
        onLeave={onLeave}
        onPin={onPin}
        vessels={vessels}
      />
    </group>
  );
}

function HeartModel({ src }) {
  const { scene } = useGLTF(src);
  return <primitive object={scene} scale={1.35} />;
}

class ModelErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError() {
    return { hasError: true };
  }

  render() {
    if (this.state.hasError) {
      return (
        <ProceduralHeart
          activeVessel={this.props.activeVessel}
          onHover={this.props.onHover}
          onLeave={this.props.onLeave}
          onPin={this.props.onPin}
          vessels={this.props.vessels}
        />
      );
    }
    return this.props.children;
  }
}

function HeartScene({
  autoRotate,
  activeVessel,
  modelExists,
  modelScale,
  onHover,
  onLeave,
  onPin,
  rotation,
  vessels,
}) {
  const groupRef = useRef(null);
  const { camera } = useThree();

  useEffect(() => {
    camera.position.set(0, 0.53, 6.8);
    camera.lookAt(0, 0.53, 0);
  }, [camera]);

  useFrame((_, delta) => {
    if (!groupRef.current) return;
    groupRef.current.rotation.x = MathUtils.damp(
      groupRef.current.rotation.x,
      rotation[0],
      5,
      delta,
    );
    groupRef.current.rotation.y = MathUtils.damp(
      groupRef.current.rotation.y,
      rotation[1],
      5,
      delta,
    );
    if (autoRotate) groupRef.current.rotation.y += delta * 0.2;
  });

  return (
    <>
      <ambientLight intensity={0.7} />
      <hemisphereLight args={["#fff1e8", "#33231f", 0.8]} />
      <directionalLight color="#fff3e7" intensity={2.3} position={[3, 5, 5]} />
      <pointLight color="#ffd7bd" intensity={2.2} position={[-3, 1, 2]} />
      <pointLight color="#f2a69a" intensity={1.4} position={[2, -2, -1]} />
      <group ref={groupRef} scale={modelScale}>
        <ModelErrorBoundary
          activeVessel={activeVessel}
          onHover={onHover}
          onLeave={onLeave}
          onPin={onPin}
          vessels={vessels}
        >
          {modelExists ? (
            <>
              <HeartModel src="/models/heart.glb?v=3" />
              <HeartHotspots
                activeVessel={activeVessel}
                onHover={onHover}
                onLeave={onLeave}
                onPin={onPin}
                vessels={vessels}
              />
            </>
          ) : (
            <ProceduralHeart
              activeVessel={activeVessel}
              onHover={onHover}
              onLeave={onLeave}
              onPin={onPin}
              vessels={vessels}
            />
          )}
        </ModelErrorBoundary>
      </group>
      <OrbitControls
        enableDamping
        enablePan={false}
        enableRotate
        enableZoom
        maxDistance={8.5}
        maxPolarAngle={Math.PI * 0.86}
        minDistance={2.8}
        rotateSpeed={0.8}
        target={[0, 0.53, 0]}
      />
    </>
  );
}

function ModelLoadingFallback() {
  return (
    <Html center>
      <div className="flex items-center gap-2 whitespace-nowrap rounded-xl bg-deepNavy/90 px-4 py-3 text-sm text-white">
        <LoaderCircle aria-hidden="true" className="h-4 w-4 animate-spin text-cyan" />
        Loading heart model…
      </div>
    </Html>
  );
}

function VesselTooltip({ name, value, pinned, onUnpin }) {
  if (!name) return null;
  const color = getRiskColor(value);
  const fullName = VESSEL_DETAILS[name].fullName;
  return (
    <div
      className="pointer-events-auto absolute bottom-4 left-1/2 z-10 w-[min(92%,360px)] -translate-x-1/2 rounded-xl border bg-deepNavy/95 p-3 text-left shadow-2xl backdrop-blur sm:left-4 sm:translate-x-0"
      style={{ borderColor: `${color}99`, boxShadow: `0 0 18px ${color}35` }}
      role="status"
    >
      <p className="text-sm leading-snug text-white">
        <strong>{name} ({fullName}):</strong> Predicted Stenosis:{" "}
        <strong style={{ color }}>
          {value}% ({getRiskLevel(value).replace(/^./, (letter) => letter.toUpperCase())} Risk - {riskName(value)})
        </strong>
      </p>
      {pinned && (
        <div className="mt-2 flex items-center justify-between gap-3">
          <Link
            className="text-sm font-bold text-cyan underline underline-offset-2 hover:text-white"
            to={`/guide?term=${name}`}
          >
            Learn more
          </Link>
          <button
            aria-label="Close vessel details"
            className="rounded px-2 py-1 text-xs text-slate-300 hover:bg-white/10"
            onClick={onUnpin}
            type="button"
          >
            Close
          </button>
        </div>
      )}
    </div>
  );
}

function HeartViewer({ vessels, activeVessel, onHover, onLeave, onPin }) {
  const viewerRef = useRef(null);
  const [showLegend, setShowLegend] = useState(false);
  const [autoRotate, setAutoRotate] = useState(true);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [fullscreenError, setFullscreenError] = useState("");
  const [modelScale, setModelScale] = useState(1);
  const [rotation, setRotation] = useState([0, 0]);
  const [modelExists, setModelExists] = useState(false);

  useEffect(() => {
    let mounted = true;
    fetch("/models/heart.glb", { headers: { Range: "bytes=0-3" } })
      .then((response) => {
        if (!response.ok || !response.body) return false;
        const reader = response.body.getReader();
        return reader.read().then(({ value }) => {
          void reader.cancel();
          return (
            value?.[0] === 0x67 &&
            value?.[1] === 0x6c &&
            value?.[2] === 0x54 &&
            value?.[3] === 0x46
          );
        });
      })
      .then((isModel) => {
        if (mounted) setModelExists(Boolean(isModel));
      })
      .catch(() => {
        if (mounted) setModelExists(false);
      });
    return () => {
      mounted = false;
    };
  }, []);

  useEffect(() => {
    function syncFullscreenState() {
      setIsFullscreen(document.fullscreenElement === viewerRef.current);
    }

    document.addEventListener("fullscreenchange", syncFullscreenState);
    return () => document.removeEventListener("fullscreenchange", syncFullscreenState);
  }, []);

  async function toggleFullscreen() {
    const viewer = viewerRef.current;
    if (!viewer) return;

    setFullscreenError("");
    try {
      if (document.fullscreenElement === viewer) {
        await document.exitFullscreen();
      } else {
        await viewer.requestFullscreen();
      }
    } catch (error) {
      console.error("Unable to toggle the heart viewer fullscreen mode.", error);
      setFullscreenError("Full screen is unavailable in this browser.");
    }
  }

  function rotateModel(x, y) {
    setRotation((current) => [current[0] + x, current[1] + y]);
  }

  function resetView() {
    setRotation([0, 0]);
    setModelScale(1);
  }

  return (
    <div
      ref={viewerRef}
      className={`relative overflow-hidden border border-cyan/30 bg-[#020a1c] ${
        isFullscreen ? "h-screen w-screen rounded-none" : "h-[280px] rounded-2xl sm:h-[320px] lg:h-[360px]"
      }`}
    >
      <div className="pointer-events-none absolute inset-0 z-[1] bg-[radial-gradient(ellipse_at_center,rgba(14,52,105,0.38),transparent_68%)]" />
      <div className="absolute left-3 top-3 z-10">
        <button
          aria-expanded={showLegend}
          aria-label="Toggle stenosis risk legend"
          className="rounded-lg border border-cyan/40 bg-deepNavy/90 p-2 text-cyan shadow-lg transition hover:bg-midNavy"
          onClick={() => setShowLegend((shown) => !shown)}
          type="button"
        >
          <Info aria-hidden="true" className="h-5 w-5" />
        </button>
        {showLegend && (
          <div className="mt-2 space-y-2 rounded-xl border border-cyan/30 bg-deepNavy/95 p-3 text-xs text-white shadow-xl">
            <p className="font-bold uppercase tracking-wider">Stenosis risk</p>
            <p className="flex items-center gap-2"><i className="h-2.5 w-2.5 rounded-full bg-risk-green" /> Green &lt;50%</p>
            <p className="flex items-center gap-2"><i className="h-2.5 w-2.5 rounded-full bg-risk-yellow" /> Yellow 50–70%</p>
            <p className="flex items-center gap-2"><i className="h-2.5 w-2.5 rounded-full bg-risk-red" /> Red &gt;70%</p>
          </div>
        )}
      </div>
      <div className="absolute right-3 top-3 z-10 flex flex-col gap-2">
        <button
          aria-label={isFullscreen ? "Exit full screen" : "View heart in full screen"}
          className="rounded-lg border border-cyan/35 bg-deepNavy/90 p-2 text-cyan transition hover:bg-midNavy"
          onClick={toggleFullscreen}
          title={isFullscreen ? "Exit full screen" : "View heart in full screen"}
          type="button"
        >
          {isFullscreen ? (
            <Minimize2 aria-hidden="true" className="h-4 w-4" />
          ) : (
            <Maximize2 aria-hidden="true" className="h-4 w-4" />
          )}
        </button>
        <button
          aria-label="Zoom in"
          className="rounded-lg border border-cyan/35 bg-deepNavy/90 p-2 text-cyan transition hover:bg-midNavy"
          onClick={() => setModelScale((scale) => Math.min(1.7, scale + 0.15))}
          title="Zoom in"
          type="button"
        >
          <ZoomIn aria-hidden="true" className="h-4 w-4" />
        </button>
        <button
          aria-label={autoRotate ? "Pause auto-rotate" : "Start auto-rotate"}
          className={`rounded-lg border border-cyan/35 bg-deepNavy/90 p-2 transition hover:bg-midNavy ${
            autoRotate ? "text-cyan" : "text-slate-300"
          }`}
          onClick={() => setAutoRotate((enabled) => !enabled)}
          title={autoRotate ? "Pause auto-rotate" : "Start auto-rotate"}
          type="button"
        >
          {autoRotate ? (
            <Pause aria-hidden="true" className="h-4 w-4" />
          ) : (
            <Play aria-hidden="true" className="h-4 w-4" />
          )}
        </button>
        <button
          aria-label="Reset heart view"
          className="rounded-lg border border-cyan/35 bg-deepNavy/90 p-2 text-cyan transition hover:bg-midNavy"
          onClick={resetView}
          title="Reset view"
          type="button"
        >
          <RotateCcw aria-hidden="true" className="h-4 w-4" />
        </button>
        <button
          aria-label="Zoom out"
          className="rounded-lg border border-cyan/35 bg-deepNavy/90 p-2 text-cyan transition hover:bg-midNavy"
          onClick={() => setModelScale((scale) => Math.max(0.65, scale - 0.15))}
          title="Zoom out"
          type="button"
        >
          <ZoomOut aria-hidden="true" className="h-4 w-4" />
        </button>
      </div>
      {fullscreenError && (
        <p className="absolute bottom-3 left-1/2 z-10 -translate-x-1/2 rounded-lg bg-deepNavy/95 px-3 py-2 text-center text-sm text-white">
          {fullscreenError}
        </p>
      )}
      <Canvas
        camera={{ fov: 43, position: [0, 0.53, 6.8] }}
        className="cursor-grab active:cursor-grabbing"
        dpr={[1, 1.75]}
        gl={{ alpha: true, antialias: true }}
        onPointerMissed={() => onLeave()}
      >
        <Suspense fallback={<ModelLoadingFallback />}>
          <HeartScene
            activeVessel={activeVessel}
            autoRotate={autoRotate}
            modelExists={modelExists}
            modelScale={modelScale}
            onHover={onHover}
            onLeave={onLeave}
            onPin={onPin}
            rotation={rotation}
            vessels={vessels}
          />
        </Suspense>
      </Canvas>
      <div className="absolute bottom-3 right-3 z-10 grid grid-cols-3 gap-1.5 rounded-xl border border-cyan/25 bg-deepNavy/85 p-2">
        <span />
        <button
          aria-label="Rotate heart up"
          className="rounded-md p-2 text-cyan hover:bg-white/10"
          onClick={() => rotateModel(-0.18, 0)}
          type="button"
        >
          <ArrowUp aria-hidden="true" className="h-5 w-5" />
        </button>
        <span />
        <button
          aria-label="Rotate heart left"
          className="rounded-md p-2 text-cyan hover:bg-white/10"
          onClick={() => rotateModel(0, -0.18)}
          type="button"
        >
          <ArrowLeft aria-hidden="true" className="h-5 w-5" />
        </button>
        <button
          aria-label="Reset heart rotation"
          className="rounded-md p-2 text-cyan hover:bg-white/10"
          onClick={resetView}
          type="button"
        >
          <RotateCcw aria-hidden="true" className="h-5 w-5" />
        </button>
        <button
          aria-label="Rotate heart right"
          className="rounded-md p-2 text-cyan hover:bg-white/10"
          onClick={() => rotateModel(0, 0.18)}
          type="button"
        >
          <ArrowRight aria-hidden="true" className="h-5 w-5" />
        </button>
        <span />
        <button
          aria-label="Rotate heart down"
          className="rounded-md p-2 text-cyan hover:bg-white/10"
          onClick={() => rotateModel(0.18, 0)}
          type="button"
        >
          <ArrowDown aria-hidden="true" className="h-5 w-5" />
        </button>
        <span />
      </div>
    </div>
  );
}

function ShapTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  const value = payload[0].value;
  return (
    <div className="rounded-lg border border-slate-600 bg-deepNavy px-3 py-2 text-sm text-white shadow-xl">
      <p className="font-semibold">{label}</p>
      <p style={{ color: value < 0 ? "#38BDF8" : "#FF2E7E" }}>
        Contribution: {value > 0 ? "+" : ""}{value}
      </p>
    </div>
  );
}

function ShapChart({ shap }) {
  const data = shap
    .slice(0, 8);
  const axisLimit = Math.max(
    0.01,
    ...data.map((item) => Math.abs(Number(item.value) || 0)),
  ) * 1.1;

  if (data.length === 0) {
    return (
      <p className="flex h-[220px] items-center justify-center text-center text-sm text-slate-300 sm:h-[250px] min-[960px]:h-[300px]">
        No feature-contribution explanations are available for this prediction.
      </p>
    );
  }

  return (
    <div className="h-[220px] w-full sm:h-[250px] min-[960px]:h-[300px]">
      <ResponsiveContainer height="100%" width="100%">
        <BarChart
          data={data}
          layout="vertical"
          margin={{ top: 8, right: 12, bottom: 8, left: 4 }}
        >
          <CartesianGrid stroke="#ffffff1c" strokeDasharray="3 3" horizontal={false} />
          <XAxis
            axisLine={{ stroke: "#64748b" }}
            domain={[-axisLimit, axisLimit]}
            tick={{ fill: "#cbd5e1", fontSize: 11 }}
            tickLine={{ stroke: "#64748b" }}
            type="number"
          />
          <YAxis
            axisLine={false}
            dataKey="feature"
            tick={{ fill: "#e2e8f0", fontSize: 11 }}
            tickLine={false}
            type="category"
            width={112}
          />
          <Tooltip content={<ShapTooltip />} cursor={{ fill: "#ffffff0c" }} />
          <ReferenceLine x={0} stroke="#cbd5e1" strokeOpacity={0.7} />
          <Bar dataKey="value" maxBarSize={30} radius={[0, 5, 5, 0]}>
            {data.map((item) => (
              <Cell
                key={item.feature}
                fill={item.value < 0 ? "#38BDF8" : "#FF2E7E"}
                radius={item.value < 0 ? [5, 0, 0, 5] : [0, 5, 5, 0]}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function EmptyDashboard({ onLoadSample, isLoading }) {
  return (
    <section className="flex flex-1 flex-col items-center justify-center px-5 py-16 text-center">
      <div className="glow-card w-full max-w-xl bg-deepNavy/70 p-8 sm:p-12">
        <h1 className="text-3xl font-extrabold text-white sm:text-4xl">
          No patient data yet
        </h1>
        <p className="mx-auto mt-3 max-w-md text-slate-300">
          Upload a patient report or load sample data to view the cardiac risk
          analysis dashboard.
        </p>
        <div className="mt-7 flex flex-col items-center justify-center gap-3 sm:flex-row">
          <Link
            className="rounded-full border border-cyan/70 px-6 py-3 font-bold text-white transition hover:bg-cyan/10"
            to="/"
          >
            Back to upload
          </Link>
          <GlowButton disabled={isLoading} onClick={onLoadSample}>
            {isLoading ? (
              <>
                <LoaderCircle aria-hidden="true" className="mr-2 h-4 w-4 animate-spin" />
                Loading sample…
              </>
            ) : (
              "Load sample patient"
            )}
          </GlowButton>
        </div>
      </div>
    </section>
  );
}

function downloadReport(result) {
  const blob = new Blob([JSON.stringify(result, null, 2)], {
    type: "application/json",
  });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = "heartcare-ai-report.json";
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 0);
}

export default function Analyze() {
  const { predictionResult, setUploadedFiles, setPredictionResult } =
    usePatientData();
  const [hoveredVessel, setHoveredVessel] = useState(null);
  const [pinnedVessel, setPinnedVessel] = useState(null);
  const [isLoadingSample, setIsLoadingSample] = useState(false);
  const [sampleError, setSampleError] = useState("");

  const result = predictionResult;
  const vessels = result?.vessels ?? {};
  const activeVessel = pinnedVessel ?? hoveredVessel;

  async function loadSample() {
    setIsLoadingSample(true);
    setSampleError("");
    try {
      const sampleJson = JSON.stringify(SAMPLE_PATIENT);
      setUploadedFiles([
        {
          name: "sample-patient.json",
          size: new Blob([sampleJson]).size,
          type: "application/json",
          parsedData: SAMPLE_PATIENT,
          isSample: true,
        },
      ]);
      const sampleResult = await predictRisk({ features: SAMPLE_PATIENT });
      setPredictionResult(sampleResult);
    } catch (error) {
      setSampleError(`Unable to load sample data: ${error.message}`);
    } finally {
      setIsLoadingSample(false);
    }
  }

  if (!result) {
    return (
      <>
        <EmptyDashboard isLoading={isLoadingSample} onLoadSample={loadSample} />
        {sampleError && (
          <p aria-live="assertive" className="pb-5 text-center text-sm font-semibold text-red-300">
            {sampleError}
          </p>
        )}
      </>
    );
  }

  const cadProbability = Number(result.cad?.probability ?? 0);
  const probabilityPercent = Math.round(cadProbability * 100);
  const shap = Array.isArray(result.shap) ? result.shap : [];

  return (
    <section className="mx-auto flex w-full max-w-[1700px] flex-1 flex-col px-4 py-2 sm:px-6 sm:py-3 lg:px-8">
      <div className="relative mb-2 flex min-h-14 items-center justify-center px-16 text-center sm:mb-3">
        <h1 className="text-2xl font-extrabold leading-tight tracking-wide text-white sm:text-3xl md:text-4xl">
          ANALYSIS <span className="text-cyan">RESULTS &amp; CLINICAL DASHBOARD</span>
        </h1>
        <HeartGlow className="absolute right-0 top-1 w-16 sm:right-3 sm:top-0 sm:w-20" />
      </div>

      <div className="grid items-stretch gap-3 min-[960px]:grid-cols-[minmax(225px,1fr)_minmax(360px,1.6fr)_minmax(260px,1fr)]">
        <GlowCard
          className="min-w-0"
          headerClassName="!py-2"
          title="PREDICTION SCORES"
        >
          <div className="p-3 sm:p-4">
            <h3 className="text-center text-xs font-extrabold tracking-widest text-slate-300">
              OVERALL CAD STATUS
            </h3>
            <Gauge probability={cadProbability} />
            <p
              className={`mt-1 text-center text-lg font-extrabold tracking-wider ${
                result.cad?.present ? "text-risk-red" : "text-risk-green"
              }`}
            >
              CAD {result.cad?.present ? "PRESENT" : "ABSENT"}
            </p>
            <p className="mt-1 text-center text-sm text-slate-300">
              Probability score: <strong className="text-white">{probabilityPercent}%</strong>
            </p>
            <div className="my-3 border-t border-white/15" />
            <h3 className="text-center text-xs font-extrabold tracking-widest text-slate-300">
              VESSEL-SPECIFIC STENOSIS
            </h3>
            <VesselRows
              onHover={setHoveredVessel}
              onPin={setPinnedVessel}
              selectedVessel={activeVessel}
              vessels={vessels}
            />
          </div>
        </GlowCard>

        <GlowCard
          className="min-w-0"
          headerClassName="!py-2"
          title="INTERACTIVE 3D HEART MODEL"
        >
          <div className="p-2.5 sm:p-3">
            <HeartViewer
              activeVessel={activeVessel}
              onHover={setHoveredVessel}
              onLeave={() => setHoveredVessel(null)}
              onPin={setPinnedVessel}
              vessels={vessels}
            />
            <VesselTooltip
              name={activeVessel}
              onUnpin={() => setPinnedVessel(null)}
              pinned={Boolean(pinnedVessel)}
              value={activeVessel ? vessels[activeVessel] : null}
            />
          </div>
        </GlowCard>

        <GlowCard
          className="min-w-0"
          headerClassName="!py-2"
          title="PREDICTION EXPLANATION (KEY FACTORS)"
        >
          <div className="p-3 sm:p-4">
            <p className="mb-2 min-h-9 text-sm leading-snug text-slate-200">
              {result.summary}
            </p>
            <div className="mb-3 flex items-center justify-end gap-3 text-[11px] text-slate-300">
              <span className="flex items-center gap-1.5">
                <i className="h-2.5 w-2.5 rounded-sm bg-cyan" /> Negative
              </span>
              <span className="flex items-center gap-1.5">
                <i className="h-2.5 w-2.5 rounded-sm bg-[#FF2E7E]" /> Positive
              </span>
            </div>
            <ShapChart shap={shap} />
          </div>
        </GlowCard>
      </div>

      <div className="mt-3 flex flex-col items-center justify-between gap-2 border-t border-white/10 pt-2 sm:flex-row">
        <Link
          className="text-sm font-semibold text-cyan underline-offset-4 transition hover:text-white hover:underline"
          to="/"
        >
          Re-upload data
        </Link>
        <GlowButton
          className="gap-2"
          onClick={() => downloadReport(result)}
        >
          <Download aria-hidden="true" className="h-4 w-4" />
          Export report (JSON)
        </GlowButton>
      </div>
    </section>
  );
}
